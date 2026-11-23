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
from typing import Callable, Dict, Optional, Tuple

from sglang.srt.tpe.state import TpeAdmissionMode, TpeInstanceState

logger = logging.getLogger(__name__)


def parse_tpe_admission_key(
    extra_key: Optional[str],
) -> Tuple[Optional[str], Optional[int], bool]:
    """Parse TPE lease metadata carried in `GenerateReqInput.extra_key`.

    Supported formats (Phase 1a with `--disable-radix-cache`):
      - ``"tpe:job=<job_id>:gen=<generation>"`` -> ``(job_id, generation, False)``
      - ``"tpe:sentinel:gen=<generation>"``     -> ``(None, generation, True)``
      - anything else / None                    -> ``(None, None, False)``
    """
    if not extra_key or not isinstance(extra_key, str):
        return None, None, False
    if not extra_key.startswith("tpe:"):
        return None, None, False
    parts = extra_key.split(":")
    if len(parts) < 3:
        return None, None, False
    if parts[1] == "sentinel" and parts[2].startswith("gen="):
        try:
            return None, int(parts[2][4:]), True
        except ValueError:
            return None, None, False
    if parts[1].startswith("job=") and parts[2].startswith("gen="):
        job_id = parts[1][4:]
        try:
            gen = int(parts[2][4:])
        except ValueError:
            return None, None, False
        return (job_id or None), gen, False
    return None, None, False


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
        self.drained_job_id: Optional[str] = None
        self.drained_purpose: Optional[str] = None
        self.pending_drain: Optional[PendingDrain] = None
        self.taint_reason: Optional[str] = None

    def assert_generation_and_healthy(self, instance_generation: int) -> None:
        """Validate transaction generation and ensure instance is not TAINTED."""
        if instance_generation != self.instance_generation:
            raise ValueError(
                f"Stale generation: got {instance_generation}, current {self.instance_generation}"
            )
        if self.state in (TpeInstanceState.TAINTED, TpeInstanceState.RESTARTING):
            raise RuntimeError(
                f"Instance is {self.state.value}: {self.taint_reason or 'restart required'}"
            )

    def assert_ready_for_hot_reload(
        self, *, job_id: str, instance_generation: int
    ) -> None:
        """Require completed pre-install drain and CLOSED admission for this job (N2, R4)."""
        self.assert_generation_and_healthy(instance_generation)
        if self.state != TpeInstanceState.PREPARING:
            raise RuntimeError(
                f"hot_reload requires state=PREPARING (completed drain), got {self.state.value}"
            )
        if self.admission_mode != TpeAdmissionMode.CLOSED:
            raise RuntimeError(
                f"hot_reload requires admission_mode=CLOSED, got {self.admission_mode.value}"
            )
        if self.drained_purpose != "hot_reload" or self.drained_job_id != job_id:
            raise RuntimeError(
                f"hot_reload drain mismatch: drained=({self.drained_purpose!r}, {self.drained_job_id!r}), "
                f"requested job={job_id!r}"
            )

    def assert_ready_for_teardown(
        self, *, job_id: str, instance_generation: int
    ) -> None:
        """Require completed pre-teardown drain and CLOSED admission for the active job (N2, R4)."""
        self.assert_generation_and_healthy(instance_generation)
        if self.state != TpeInstanceState.RESTORING:
            raise RuntimeError(
                f"teardown requires state=RESTORING (completed drain), got {self.state.value}"
            )
        if self.admission_mode != TpeAdmissionMode.CLOSED:
            raise RuntimeError(
                f"teardown requires admission_mode=CLOSED, got {self.admission_mode.value}"
            )
        if self.drained_purpose != "teardown" or self.drained_job_id != job_id:
            raise RuntimeError(
                f"teardown drain mismatch: drained=({self.drained_purpose!r}, {self.drained_job_id!r}), "
                f"requested job={job_id!r}"
            )
        if self.active_job_id is not None and self.active_job_id != job_id:
            raise RuntimeError(
                f"teardown job mismatch: active_job_id={self.active_job_id!r}, requested={job_id!r}"
            )

    def set_admission(
        self,
        mode: TpeAdmissionMode,
        *,
        job_id: Optional[str] = None,
        instance_generation: Optional[int] = None,
    ) -> None:
        """Update admission mode while preserving drain & active-candidate invariants (R4)."""
        if instance_generation is not None:
            self.assert_generation_and_healthy(instance_generation)
        elif self.state in (TpeInstanceState.TAINTED, TpeInstanceState.RESTARTING):
            raise RuntimeError(
                f"Instance is {self.state.value}: {self.taint_reason or 'restart required'}"
            )

        # Never allow reopening admission while draining or holding a completed drain barrier.
        if mode != TpeAdmissionMode.CLOSED and self.state in (
            TpeInstanceState.DRAINING,
            TpeInstanceState.PREPARING,
            TpeInstanceState.RESTORING,
        ):
            raise RuntimeError(
                f"Cannot open admission ({mode.value}) in state {self.state.value}; "
                "complete the pending hot_reload/teardown first."
            )

        if mode == TpeAdmissionMode.JOB_ONLY:
            if self.state not in (
                TpeInstanceState.CANDIDATE_READY,
                TpeInstanceState.EVALUATING,
            ):
                raise RuntimeError(
                    f"JOB_ONLY admission requires CANDIDATE_READY/EVALUATING, got {self.state.value}"
                )
            if job_id is None or self.active_job_id is None or job_id != self.active_job_id:
                raise RuntimeError(
                    f"JOB_ONLY admission job mismatch: active={self.active_job_id!r}, requested={job_id!r}"
                )
        elif mode == TpeAdmissionMode.BASELINE_PROBE_ONLY:
            if self.state not in (
                TpeInstanceState.BASELINE_READY,
                TpeInstanceState.BASELINE_VERIFYING,
            ):
                raise RuntimeError(
                    f"BASELINE_PROBE_ONLY admission requires BASELINE_READY/BASELINE_VERIFYING, got {self.state.value}"
                )

        self.admission_mode = mode

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
        self.assert_generation_and_healthy(instance_generation)
        if purpose not in ("hot_reload", "teardown"):
            raise ValueError(f"Unsupported drain purpose: {purpose!r}")
        if self.pending_drain is not None:
            raise RuntimeError(
                f"Drain already in progress (op_id={self.pending_drain.op_id})"
            )

        self.admission_mode = TpeAdmissionMode.CLOSED
        self.state = TpeInstanceState.DRAINING
        self.drained_job_id = None
        self.drained_purpose = None
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
        (or on timeout/canary violation/sync exception), else None while still draining.
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
                "canary_summary": None,
                "error_message": self.taint_reason,
            }

        if not (is_fully_idle_fn() and result_queue_empty_fn()):
            return None

        canary_summary: Optional[Dict[str, object]] = None
        try:
            # Strict F3 ordering:
            # 1. Synchronize all GPU streams BEFORE touching canary or releasing keep-alive refs.
            sync_streams_fn()

            # 2. If draining for teardown, finalize kv_canary BEFORE cache flush / ref release.
            if drain.purpose == "teardown" and finalize_canary_fn is not None:
                canary_summary = finalize_canary_fn()
                if bool(canary_summary.get("violations_detected", False)):
                    self.pending_drain = None
                    self.mark_tainted(
                        f"kv_canary violation detected during teardown drain (job={drain.job_id})"
                    )
                    return {
                        "success": False,
                        "op": "drain",
                        "op_id": drain.op_id,
                        "job_id": drain.job_id,
                        "instance_generation": drain.instance_generation,
                        "responder_id": self.responder_id,
                        "state": self.state.value,
                        "elapsed_s": self._clock() - drain.started_monotonic,
                        "canary_summary": canary_summary,
                        "error_message": self.taint_reason,
                    }

            # 3. Release overlap keep-alive refs (batch_record_buf) only AFTER GPU sync.
            if clear_keep_alive_fn is not None:
                clear_keep_alive_fn()
        except Exception as exc:
            # Catch sync / sweep / violation check exceptions even during deferred event-loop polling (R3).
            self.pending_drain = None
            self.mark_tainted(
                f"Drain finalization failed for {drain.purpose} (job={drain.job_id}): "
                f"{type(exc).__name__}: {exc}"
            )
            return {
                "success": False,
                "op": "drain",
                "op_id": drain.op_id,
                "job_id": drain.job_id,
                "instance_generation": drain.instance_generation,
                "responder_id": self.responder_id,
                "state": self.state.value,
                "elapsed_s": self._clock() - drain.started_monotonic,
                "canary_summary": canary_summary,
                "error_message": self.taint_reason,
            }

        self.pending_drain = None
        self.drained_job_id = drain.job_id
        self.drained_purpose = drain.purpose
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
        self.drained_job_id = None
        self.drained_purpose = None
        self.state = TpeInstanceState.CANDIDATE_READY
        self.admission_mode = TpeAdmissionMode.JOB_ONLY

    def begin_baseline_verifying(self) -> None:
        self.active_job_id = None
        self.active_artifact_hash = None
        self.drained_job_id = None
        self.drained_purpose = None
        self.state = TpeInstanceState.BASELINE_VERIFYING
        self.admission_mode = TpeAdmissionMode.BASELINE_PROBE_ONLY

    def complete_baseline_verifying(
        self,
        *,
        sentinel_passed: bool,
        instance_generation: Optional[int] = None,
    ) -> None:
        if instance_generation is not None:
            self.assert_generation_and_healthy(instance_generation)
        if self.state != TpeInstanceState.BASELINE_VERIFYING:
            raise RuntimeError(
                f"complete_baseline_verifying requires state=BASELINE_VERIFYING, got {self.state.value}"
            )
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
        self.drained_job_id = None
        self.drained_purpose = None
        logger.error(
            "TPE scheduler controller generation=%d TAINTED: %s",
            self.instance_generation,
            reason,
        )
