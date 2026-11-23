"""State definitions, admission modes, and Phase 1a config validation for TPE."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, List, Optional


class TpeInstanceState(str, enum.Enum):
    BASELINE_READY = "BASELINE_READY"
    DRAINING = "DRAINING"
    PREPARING = "PREPARING"
    CAPTURING = "CAPTURING"
    CANDIDATE_READY = "CANDIDATE_READY"
    EVALUATING = "EVALUATING"
    VERIFYING = "VERIFYING"
    RESTORING = "RESTORING"
    BASELINE_VERIFYING = "BASELINE_VERIFYING"
    TAINTED = "TAINTED"
    RESTARTING = "RESTARTING"


class TpeAdmissionMode(str, enum.Enum):
    CLOSED = "CLOSED"
    JOB_ONLY = "JOB_ONLY"
    BASELINE_PROBE_ONLY = "BASELINE_PROBE_ONLY"


@dataclass
class TpeJobOutcome:
    job_id: str
    artifact_hash: str
    instance_generation: int
    measurement_completed: bool = False
    verification_passed: bool = False
    restore_passed: bool = False
    eligible_for_promotion: bool = False
    failure_attribution: Optional[str] = None
    canary_finalized: bool = False
    canary_violations_detected: bool = False

    def compute_promotion_eligibility(self) -> bool:
        """Determine whether this job's measurement may enter Champion ranking (F7)."""
        taint_reasons = {
            "sentinel_mismatch",
            "kv_violation",
            "unknown_taint",
            "contract_violation",
        }
        if self.canary_violations_detected:
            self.eligible_for_promotion = False
            return False
        if self.failure_attribution in taint_reasons:
            self.eligible_for_promotion = False
            return False
        self.eligible_for_promotion = bool(
            self.measurement_completed
            and self.verification_passed
            and (
                self.restore_passed
                or self.failure_attribution in ("cleanup_timeout", "memory_budget")
            )
        )
        return self.eligible_for_promotion


def validate_phase1a_config(
    *,
    prefill_cuda_graph_disabled: bool,
    decode_cuda_graph_full: bool,
    enable_torch_compile: bool,
    enable_megakernel_or_monokernel: bool,
    enable_post_capture_kv_sizing: bool,
    enable_speculative: bool,
    disaggregation_mode: str,
    pp_size: int = 1,
    ep_size: int = 1,
    enable_dp_attention: bool = False,
    enable_hierarchical_cache: bool = False,
    enable_overlap: bool = False,
    allow_overlap: bool = False,
) -> List[str]:
    """Return a list of Phase 1a config violations (empty list means valid)."""
    reasons: List[str] = []
    if not prefill_cuda_graph_disabled:
        reasons.append("Phase 1a requires prefill CUDA graph to be disabled (prefill eager).")
    if not decode_cuda_graph_full:
        reasons.append("Phase 1a requires full decode CUDA graph backend.")
    if enable_torch_compile:
        reasons.append("Phase 1a does not support --enable-torch-compile.")
    if enable_megakernel_or_monokernel:
        reasons.append("Phase 1a disables unadapted megakernel/monokernel plugins (F5).")
    if enable_post_capture_kv_sizing:
        reasons.append(
            "Phase 1a requires SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=0 before pool construction (F6)."
        )
    if enable_speculative:
        reasons.append("Phase 1a does not support speculative decoding.")
    if disaggregation_mode not in ("null", "", None):
        reasons.append(
            f"Phase 1a does not support PD disaggregation (got {disaggregation_mode!r})."
        )
    if pp_size != 1:
        reasons.append(f"Phase 1a requires pp_size=1 (got {pp_size}).")
    if ep_size != 1:
        reasons.append(f"Phase 1a requires ep_size=1 (got {ep_size}).")
    if enable_dp_attention:
        reasons.append("Phase 1a does not support DP attention.")
    if enable_hierarchical_cache:
        reasons.append("Phase 1a does not support HiCache.")
    if enable_overlap and not allow_overlap:
        reasons.append("Phase 1a runs with overlap disabled (--disable-overlap-schedule).")
    return reasons


def assert_phase1a_config(**kwargs: Any) -> None:
    reasons = validate_phase1a_config(**kwargs)
    if reasons:
        raise ValueError(
            "Invalid TPE Phase 1a configuration:\n  - " + "\n  - ".join(reasons)
        )


class ReplicaLifecycleState(str, enum.Enum):
    IDLE = "IDLE"
    STARTING = "STARTING"
    READY = "READY"
    TAINTED = "TAINTED"
    RECYCLING = "RECYCLING"
    UNHEALTHY_GPU = "UNHEALTHY_GPU"


@dataclass(frozen=True)
class GpuHealthStatus:
    healthy: bool
    reason: str = ""
    free_memory_mb: Optional[int] = None


@dataclass
class ReplicaGenerationTracker:
    """Monotonic generation and phase-timeout tracker for an external TPE replica."""

    replica_id: str
    instance_generation: int = 0
    lifecycle_state: ReplicaLifecycleState = ReplicaLifecycleState.IDLE
    last_taint_reason: Optional[str] = None
    active_phase: Optional[str] = None
    active_phase_started_at: Optional[float] = None

    def begin_generation(self) -> int:
        """Advance monotonically to a new instance_generation on every cold start."""
        self.instance_generation += 1
        self.lifecycle_state = ReplicaLifecycleState.STARTING
        self.last_taint_reason = None
        self.active_phase = None
        self.active_phase_started_at = None
        return self.instance_generation

    def mark_ready(self) -> None:
        self.lifecycle_state = ReplicaLifecycleState.READY
        self.active_phase = None
        self.active_phase_started_at = None

    def begin_phase(self, phase: str, *, now: float) -> None:
        self.active_phase = phase
        self.active_phase_started_at = now

    def end_phase(self) -> None:
        self.active_phase = None
        self.active_phase_started_at = None

    def check_phase_timeout(self, *, now: float, timeout_sec: float) -> bool:
        if self.active_phase_started_at is None:
            return False
        return (now - self.active_phase_started_at) > timeout_sec

    def mark_tainted(self, reason: str) -> None:
        self.lifecycle_state = ReplicaLifecycleState.TAINTED
        self.last_taint_reason = reason

    def record_gpu_health(self, status: GpuHealthStatus) -> None:
        if not status.healthy:
            self.lifecycle_state = ReplicaLifecycleState.UNHEALTHY_GPU
            self.last_taint_reason = status.reason or "gpu_unhealthy"
        elif self.lifecycle_state == ReplicaLifecycleState.TAINTED:
            self.lifecycle_state = ReplicaLifecycleState.RECYCLING

