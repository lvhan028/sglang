"""Scheduler-side TPE controller: admission control, non-blocking drain, and state machine.

Implements R1, F3, and F4:
- Separates stop-admission (`CLOSED`) from scheduler execution so in-flight
  requests drain naturally to quiescence without blocking the event loop.
- Enforces drain before BOTH `hot_reload` and `teardown`.
- Enforces strict teardown ordering:
  1. Quiescence (`is_fully_idle` + empty `result_queue`)
  2. Synchronize all GPU streams (`forward_stream`, `schedule_stream`, `copy_stream`)
  3. Run `kv_canary` `finalize_job_check()` (before cache flush & before releasing refs)
  4. Release overlap keep-alive refs (`batch_record_buf`, guarded if uninitialized)
  5. Destroy old graphs -> release `candidate_state` -> restore baseline -> recapture
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from sglang.srt.tpe.state import TpeAdmissionMode, TpeInstanceState

logger = logging.getLogger(__name__)


@dataclass
class PendingDrain:
    op_id: str
    job_id: str
    instance_generation: int
    purpose: str  # "hot_reload" | "teardown"
    deadline_monotonic: float
    started_monotonic: float


class TpeSchedulerController:
    """Manages TPE admission, non-blocking drain, and transaction transitions."""

    def __init__(
        self,
        *,
        instance_generation: int = 1,
        responder_id: str = "rank0",
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.instance_generation = instance_generation
        self.responder_id = responder_id
        self._clock = clock_fn

        self.state: TpeInstanceState = TpeInstanceState.BASELINE_READY
        self.admission_mode: TpeAdmissionMode = TpeAdmissionMode.BASELINE_PROBE_ONLY
        self.active_job_id: Optional[str] = None
        self.active_artifact_hash: Optional[str] = None
        self.pending_drain: Optional[PendingDrain] = None
        self.taint_reason: Optional[str] = None

    def set_admission(
        self,
        mode: TpeAdmissionMode,
        *,
        job_id: Optional[str] = None,
    ) -> None:
        self.admission_mode = mode
        if mode == TpeAdmissionMode.JOB_ONLY:
            assert job_id is not None, "JOB_ONLY admission requires job_id"
            self.active_job_id = job_id

    def can_admit_request(
        self,
        *,
        req_job_id: Optional[str] = None,
        req_generation: Optional[int] = None,
        is_baseline_probe: bool = False,
    ) -> bool:
        """Check whether an incoming generation request is permitted under current lease."""
        if self.state in (TpeInstanceState.TAINTED, TpeInstanceState.RESTARTING):
            return False
        if self.admission_mode == TpeAdmissionMode.CLOSED:
            return False
        if self.admission_mode == TpeAdmissionMode.BASELINE_PROBE_ONLY:
            if not is_baseline_probe:
                return False
            return (
                req_generation is None or req_generation == self.instance_generation
            )
        if self.admission_mode == TpeAdmissionMode.JOB_ONLY:
            return (
                not is_baseline_probe
                and req_job_id is not None
                and req_job_id == self.active_job_id
                and req_generation == self.instance_generation
            )
        return False

    def begin_drain(
        self,
        *,
        op_id: str,
        job_id: str,
        instance_generation: int,
        purpose: str,
        timeout_s: float = 30.0,
    ) -> None:
        """Stop admission and arm non-blocking drain in the scheduler event loop."""
        if instance_generation != self.instance_generation:
            raise ValueError(
                f"Stale generation: got {instance_generation}, current {self.instance_generation}"
            )
        if purpose not in ("hot_reload", "teardown"):
            raise ValueError(f"Unsupported drain purpose: {purpose!r}")
        if self.state == TpeInstanceState.TAINTED:
            raise RuntimeError(f"Instance is TAINTED: {self.taint_reason}")
        if self.pending_drain is not None:
            raise RuntimeError(
                f"Drain already in progress (op_id={self.pending_drain.op_id})"
            )

        self.admission_mode = TpeAdmissionMode.CLOSED
        self.state = TpeInstanceState.DRAINING
        now = self._clock()
        self.pending_drain = PendingDrain(
            op_id=op_id,
            job_id=job_id,
            instance_generation=instance_generation,
            purpose=purpose,
            deadline_monotonic=now + timeout_s,
            started_monotonic=now,
        )

    def poll_drain(
        self,
        *,
        is_fully_idle_fn: Callable[[], bool],
        result_queue_empty_fn: Callable[[], bool],
        sync_streams_fn: Callable[[], None],
        finalize_canary_fn: Optional[Callable[[], Dict[str, object]]] = None,
        clear_keep_alive_fn: Optional[Callable[[], None]] = None,
    ) -> Optional[Dict[str, object]]:
        """Non-blocking tick called from the scheduler event loop.

        Returns a completion dict once quiescence + ordered cleanup finishes
        (or on timeout), else None while still draining.
        """
        drain = self.pending_drain
        if drain is None:
            return None

        now = self._clock()
        if now > drain.deadline_monotonic:
            self.pending_drain = None
            self.mark_tainted(f"Drain timed out for {drain.purpose} (job={drain.job_id})")
            return {
                "success": False,
                "op": "drain",
                "op_id": drain.op_id,
                "job_id": drain.job_id,
                "instance_generation": drain.instance_generation,
                "responder_id": self.responder_id,
                "state": self.state.value,
                "elapsed_s": now - drain.started_monotonic,
                "error_message": self.taint_reason,
            }

        if not (is_fully_idle_fn() and result_queue_empty_fn()):
            return None

        # Strict F3 ordering:
        # 1. Synchronize all GPU streams BEFORE touching canary or releasing keep-alive refs.
        sync_streams_fn()

        # 2. If draining for teardown, finalize kv_canary BEFORE cache flush / ref release.
        canary_summary: Optional[Dict[str, object]] = None
        if drain.purpose == "teardown" and finalize_canary_fn is not None:
            canary_summary = finalize_canary_fn()

        # 3. Release overlap keep-alive refs (batch_record_buf) only AFTER GPU sync.
        if clear_keep_alive_fn is not None:
            clear_keep_alive_fn()

        self.pending_drain = None
        self.state = (
            TpeInstanceState.PREPARING
            if drain.purpose == "hot_reload"
            else TpeInstanceState.RESTORING
        )
        return {
            "success": True,
            "op": "drain",
            "op_id": drain.op_id,
            "job_id": drain.job_id,
            "instance_generation": drain.instance_generation,
            "responder_id": self.responder_id,
            "state": self.state.value,
            "elapsed_s": self._clock() - drain.started_monotonic,
            "canary_summary": canary_summary,
            "error_message": None,
        }

    def mark_candidate_ready(self, job_id: str, artifact_hash: str) -> None:
        self.active_job_id = job_id
        self.active_artifact_hash = artifact_hash
        self.state = TpeInstanceState.CANDIDATE_READY
        self.admission_mode = TpeAdmissionMode.JOB_ONLY

    def begin_baseline_verifying(self) -> None:
        self.active_job_id = None
        self.active_artifact_hash = None
        self.state = TpeInstanceState.BASELINE_VERIFYING
        self.admission_mode = TpeAdmissionMode.BASELINE_PROBE_ONLY

    def complete_baseline_verifying(self, *, sentinel_passed: bool) -> None:
        if sentinel_passed:
            self.state = TpeInstanceState.BASELINE_READY
            self.admission_mode = TpeAdmissionMode.BASELINE_PROBE_ONLY
        else:
            self.mark_tainted("Independent baseline sentinel verification failed")

    def mark_tainted(self, reason: str) -> None:
        self.state = TpeInstanceState.TAINTED
        self.admission_mode = TpeAdmissionMode.CLOSED
        self.taint_reason = reason
        self.pending_drain = None
        logger.error(
            "TPE scheduler controller generation=%d TAINTED: %s",
            self.instance_generation,
            reason,
        )
