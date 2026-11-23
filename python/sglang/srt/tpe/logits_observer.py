"""SamplingObserver-based full-vocab logits capture and verification for TPE Phase 1a.

Reuses SGLang's native `SamplingObserver` -> `DeviceAuxiliaryOutput` ->
`HostAuxiliaryOutput` lifecycle:
- Strictly inactive (`is_active() == False`) during throughput measurements.
- Clones logits in `before_grammar()` so CUDA-graph static output buffers are
  never aliased across replays.
- Enforces neutral sampling at capture time (rejects active pre-grammar
  penalizers / custom logit processors) so `before_grammar_neutral_sampling`
  metadata is guaranteed rather than merely documented (N6).
- Records full per-step token trajectories and initial input token digests even
  when logits are captured sparsely, so any intermediate token divergence or
  prompt mismatch is caught before numerical comparison (N3).
- Binds capture ownership to `(owner_job_id, owner_generation)` so another job
  cannot wipe or re-attribute active capture records (N5).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.tpe.logits_verifier import verify_logits_pair

CAPTURE_POINT_TAG = "before_grammar_neutral_sampling"


def _clone_tensor_or_seq(x: Any) -> Any:
    if hasattr(x, "detach") and hasattr(x, "clone"):
        return x.detach().clone()
    if isinstance(x, list):
        return [row[:] if isinstance(row, list) else row for row in x]
    return x


def _row_to_float_list(row: Any) -> List[float]:
    if hasattr(row, "detach") and hasattr(row, "cpu"):
        return row.detach().to("cpu").float().tolist()
    return [float(v) for v in row]


def _digest_token_ids(token_ids: Optional[Sequence[int]]) -> Optional[str]:
    if token_ids is None:
        return None
    h = hashlib.sha256()
    h.update(",".join(str(int(t)) for t in token_ids).encode("utf-8"))
    return h.hexdigest()


def check_neutral_sampling_params(
    sampling_params: Any,
    *,
    custom_logit_processor: Any = None,
) -> Optional[str]:
    """Validate request-level sampling params before enqueue (N6). Returns error string or None."""
    if custom_logit_processor is not None:
        return (
            "TPE full-vocab logits capture requires neutral sampling: "
            "custom_logit_processor is forbidden."
        )
    if sampling_params is None:
        return None
    freq = float(getattr(sampling_params, "frequency_penalty", 0.0) or 0.0)
    pres = float(getattr(sampling_params, "presence_penalty", 0.0) or 0.0)
    rep_raw = getattr(sampling_params, "repetition_penalty", 1.0)
    rep = 1.0 if rep_raw is None else float(rep_raw)
    min_new = int(getattr(sampling_params, "min_new_tokens", 0) or 0)
    logit_bias = getattr(sampling_params, "logit_bias", None)
    if freq != 0.0 or pres != 0.0 or rep != 1.0 or min_new > 0 or bool(logit_bias):
        return (
            "TPE full-vocab logits capture requires neutral sampling: "
            f"got frequency_penalty={freq}, presence_penalty={pres}, "
            f"repetition_penalty={rep}, min_new_tokens={min_new}, "
            f"logit_bias_set={bool(logit_bias)}."
        )
    return None


def _detect_non_neutral_sampling_info(sampling_info: Any) -> Optional[str]:
    """Defensive sampling-time check (never raises inside scheduler forward loop)."""
    if sampling_info is None:
        return None
    penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
    if penalizer is not None and bool(getattr(penalizer, "is_required", False)):
        return "non_neutral_sampling_active_penalizer"
    if bool(getattr(sampling_info, "has_custom_logit_processor", False)):
        return "non_neutral_sampling_custom_logit_processor"
    return None


def _compare_rows(
    base_row: Any,
    cand_row: Any,
    *,
    max_abs_diff_threshold: float,
    min_cosine_similarity: float,
) -> Tuple[bool, float, float, bool, str]:
    """Compare two full-vocab logits rows (fast torch path if available, else stdlib)."""
    try:
        import torch

        if isinstance(base_row, torch.Tensor) and isinstance(cand_row, torch.Tensor):
            if base_row.shape != cand_row.shape:
                return (
                    False,
                    math.inf,
                    0.0,
                    False,
                    f"shape_mismatch: {tuple(base_row.shape)} vs {tuple(cand_row.shape)}",
                )
            b = base_row.detach().to("cpu").float().reshape(-1)
            c = cand_row.detach().to("cpu").float().reshape(-1)
            if b.numel() == 0:
                return False, math.inf, 0.0, False, "empty_logits"
            if not (torch.isfinite(b).all().item() and torch.isfinite(c).all().item()):
                return False, math.inf, 0.0, False, "non_finite_logits"
            max_abs = float((b - c).abs().max().item())
            denom = float(b.norm().item() * c.norm().item())
            cos_sim = 1.0 if denom == 0.0 else float(torch.dot(b, c).item() / denom)
            if max_abs > max_abs_diff_threshold:
                return (
                    False,
                    max_abs,
                    cos_sim,
                    True,
                    f"max_abs_diff_exceeded: {max_abs:.6g} > {max_abs_diff_threshold:.6g}",
                )
            if cos_sim < min_cosine_similarity:
                return (
                    False,
                    max_abs,
                    cos_sim,
                    True,
                    f"cosine_similarity_below_threshold: {cos_sim:.6f} < {min_cosine_similarity:.6f}",
                )
            return True, max_abs, cos_sim, True, "ok"
    except ModuleNotFoundError:
        pass

    res = verify_logits_pair(
        _row_to_float_list(base_row),
        _row_to_float_list(cand_row),
        max_abs_diff_threshold=max_abs_diff_threshold,
        min_cosine_similarity=min_cosine_similarity,
    )
    return (
        res.passed,
        res.max_abs_diff,
        res.cosine_similarity,
        res.all_finite,
        res.reason,
    )


class TpeLogitsCollector:
    """Host-side collector for per-request, per-step full-vocab logits & trajectories."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.capture_steps: Optional[frozenset[int]] = None
        self.owner_job_id: Optional[str] = None
        self.owner_generation: Optional[int] = None
        # rid -> output_step -> {"token_id": int, "logits": Tensor | List[float]}
        self._records: Dict[str, Dict[int, Dict[str, Any]]] = {}
        # rid -> output_step -> token_id (recorded for EVERY step even under sparse capture_steps)
        self._trajectories: Dict[str, Dict[int, int]] = {}
        # rid -> sha256 digest of origin_input_ids (if available)
        self._input_digests: Dict[str, str] = {}
        self._non_neutral_violation: Optional[str] = None

    def arm(
        self,
        *,
        capture_steps: Optional[Sequence[int]] = None,
        owner_job_id: Optional[str] = None,
        owner_generation: Optional[int] = None,
    ) -> None:
        # Ownership persists across disarm() until finalize/clear (N5).
        has_unfinalized_owner = self.owner_job_id is not None and (
            self.enabled or bool(self._records) or bool(self._trajectories)
        )
        if has_unfinalized_owner and (
            owner_job_id != self.owner_job_id
            or (
                self.owner_generation is not None
                and owner_generation is not None
                and owner_generation != self.owner_generation
            )
        ):
            raise RuntimeError(
                f"Logits capture owned by job {self.owner_job_id!r} (gen={self.owner_generation}); "
                f"cannot re-arm for {owner_job_id!r} without finalizing/clearing first."
            )
        self._records.clear()
        self._trajectories.clear()
        self._input_digests.clear()
        self._non_neutral_violation = None
        self.capture_steps = (
            frozenset(int(s) for s in capture_steps)
            if capture_steps is not None
            else None
        )
        self.owner_job_id = owner_job_id
        self.owner_generation = owner_generation
        self.enabled = True

    def assert_owner(
        self, *, job_id: Optional[str], instance_generation: Optional[int]
    ) -> None:
        if self.owner_job_id is not None and job_id != self.owner_job_id:
            raise RuntimeError(
                f"Logits capture ownership mismatch: owner_job_id={self.owner_job_id!r}, "
                f"requested={job_id!r}"
            )
        if (
            self.owner_generation is not None
            and instance_generation is not None
            and instance_generation != self.owner_generation
        ):
            raise RuntimeError(
                f"Logits capture generation mismatch: owner={self.owner_generation}, "
                f"requested={instance_generation}"
            )

    def disarm(self) -> None:
        self.enabled = False

    def clear(self) -> None:
        self._records.clear()
        self._trajectories.clear()
        self._input_digests.clear()
        self._non_neutral_violation = None
        self.owner_job_id = None
        self.owner_generation = None

    def mark_non_neutral_violation(self, reason: str) -> None:
        if self._non_neutral_violation is None:
            self._non_neutral_violation = reason

    def record_step(
        self,
        *,
        rid: str,
        output_step: int,
        token_id: int,
        logits_row: Any,
        input_digest: Optional[str] = None,
    ) -> None:
        rid_str = str(rid)
        step_int = int(output_step)
        tok_int = int(token_id)

        # Always record lightweight token trajectory & input digest (N3),
        # even when logits_row is not retained at this step.
        if input_digest is not None and rid_str not in self._input_digests:
            self._input_digests[rid_str] = input_digest
        self._trajectories.setdefault(rid_str, {})[step_int] = tok_int

        if self.capture_steps is not None and step_int not in self.capture_steps:
            return
        req_map = self._records.setdefault(rid_str, {})
        req_map[step_int] = {
            "token_id": tok_int,
            "logits": logits_row,
        }

    def summary(self) -> Dict[str, Any]:
        num_positions = sum(len(steps) for steps in self._records.values())
        vocab_size: Optional[int] = None
        for steps in self._records.values():
            for entry in steps.values():
                row = entry["logits"]
                vocab_size = int(row.shape[-1]) if hasattr(row, "shape") else len(row)
                break
            if vocab_size is not None:
                break
        return {
            "enabled": self.enabled,
            "owner_job_id": self.owner_job_id,
            "owner_generation": self.owner_generation,
            "capture_point": CAPTURE_POINT_TAG,
            "capture_steps": (
                sorted(self.capture_steps) if self.capture_steps is not None else None
            ),
            "num_requests": len(self._records),
            "num_positions": num_positions,
            "vocab_size": vocab_size,
        }

    def save_reference(
        self, path: str, *, extra_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if self._non_neutral_violation is not None:
            raise ValueError(
                f"Cannot save neutral logits reference: {self._non_neutral_violation}"
            )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        meta = self.summary()
        if extra_meta:
            meta.update(extra_meta)
        payload = {
            "meta": meta,
            "records": self._records,
            "trajectories": self._trajectories,
            "input_digests": self._input_digests,
        }
        if path.endswith(".json"):
            json_records = {
                rid: {
                    str(step): {
                        "token_id": entry["token_id"],
                        "logits": _row_to_float_list(entry["logits"]),
                    }
                    for step, entry in steps.items()
                }
                for rid, steps in self._records.items()
            }
            json_trajectories = {
                rid: {str(step): tok for step, tok in steps.items()}
                for rid, steps in self._trajectories.items()
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": meta,
                        "records": json_records,
                        "trajectories": json_trajectories,
                        "input_digests": self._input_digests,
                    },
                    f,
                )
        else:
            import torch

            torch.save(payload, path)
        meta["save_path"] = path
        return meta

    @staticmethod
    def load_reference(path: str) -> Dict[str, Any]:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = {
                rid: {int(step): entry for step, entry in steps.items()}
                for rid, steps in data["records"].items()
            }
            trajectories = {
                rid: {int(step): int(tok) for step, tok in steps.items()}
                for rid, steps in data.get("trajectories", {}).items()
            }
            return {
                "meta": data.get("meta", {}),
                "records": records,
                "trajectories": trajectories,
                "input_digests": data.get("input_digests", {}),
            }
        import torch

        return torch.load(path, map_location="cpu")

    def compare_against_reference(
        self,
        ref_path: str,
        *,
        max_abs_diff_threshold: float = 5e-2,
        min_cosine_similarity: float = 0.999,
    ) -> Dict[str, Any]:
        if self._non_neutral_violation is not None:
            return {
                "passed": False,
                "capture_point": CAPTURE_POINT_TAG,
                "ref_path": ref_path,
                "num_checked": 0,
                "num_failed": 1,
                "trajectory_divergences": 0,
                "worst_max_abs_diff": math.inf,
                "worst_cosine_similarity": 0.0,
                "failures": [{"reason": self._non_neutral_violation}],
            }
        ref = self.load_reference(ref_path)
        ref_meta = ref.get("meta", {})
        if (
            ref_meta.get("capture_point")
            and ref_meta.get("capture_point") != CAPTURE_POINT_TAG
        ):
            raise ValueError(
                f"Incompatible reference capture_point={ref_meta.get('capture_point')!r}, "
                f"expected {CAPTURE_POINT_TAG!r}"
            )

        ref_records: Dict[str, Dict[int, Dict[str, Any]]] = ref["records"]
        ref_trajectories: Dict[str, Dict[int, int]] = ref.get("trajectories", {})
        ref_input_digests: Dict[str, str] = ref.get("input_digests", {})

        num_checked = 0
        num_failed = 0
        trajectory_divergences = 0
        worst_max_abs = 0.0
        worst_cos = 1.0
        failures: List[Dict[str, Any]] = []

        for rid, ref_steps in ref_records.items():
            cand_steps = self._records.get(rid)
            ref_traj = ref_trajectories.get(rid, {})
            cand_traj = self._trajectories.get(rid, {})

            # Verify initial input token digest matches when recorded (N3).
            expected_digest = ref_input_digests.get(rid)
            actual_digest = self._input_digests.get(rid)
            if expected_digest is not None and actual_digest != expected_digest:
                for step in sorted(ref_steps.keys()):
                    num_checked += 1
                    num_failed += 1
                    trajectory_divergences += 1
                    if len(failures) < 10:
                        failures.append(
                            {
                                "rid": rid,
                                "step": step,
                                "reason": "input_prompt_digest_mismatch",
                            }
                        )
                continue

            for step in sorted(ref_steps.keys()):
                num_checked += 1
                ref_entry = ref_steps[step]
                if cand_steps is None or step not in cand_steps:
                    num_failed += 1
                    if len(failures) < 10:
                        failures.append(
                            {"rid": rid, "step": step, "reason": "missing_capture"}
                        )
                    continue

                # Verify FULL preceding token trajectory [0 .. step-1] matches (N3),
                # even when intermediate steps were not in capture_steps.
                prefix_diverged = False
                for prev_step in range(step):
                    if prev_step not in ref_traj or prev_step not in cand_traj:
                        prefix_diverged = True
                        break
                    if int(cand_traj[prev_step]) != int(ref_traj[prev_step]):
                        prefix_diverged = True
                        break

                if prefix_diverged:
                    num_failed += 1
                    trajectory_divergences += 1
                    if len(failures) < 10:
                        failures.append(
                            {
                                "rid": rid,
                                "step": step,
                                "reason": "token_trajectory_diverged_at_earlier_step",
                            }
                        )
                    continue

                cand_entry = cand_steps[step]
                passed, max_abs, cos_sim, all_finite, reason = _compare_rows(
                    ref_entry["logits"],
                    cand_entry["logits"],
                    max_abs_diff_threshold=max_abs_diff_threshold,
                    min_cosine_similarity=min_cosine_similarity,
                )
                if math.isfinite(max_abs):
                    worst_max_abs = max(worst_max_abs, max_abs)
                else:
                    worst_max_abs = math.inf
                worst_cos = min(worst_cos, cos_sim)

                if not passed:
                    num_failed += 1
                    if len(failures) < 10:
                        failures.append(
                            {
                                "rid": rid,
                                "step": step,
                                "max_abs_diff": max_abs,
                                "cosine_similarity": cos_sim,
                                "all_finite": all_finite,
                                "reason": reason,
                            }
                        )

        return {
            "passed": (num_checked > 0 and num_failed == 0),
            "capture_point": CAPTURE_POINT_TAG,
            "ref_path": ref_path,
            "num_checked": num_checked,
            "num_failed": num_failed,
            "trajectory_divergences": trajectory_divergences,
            "worst_max_abs_diff": worst_max_abs,
            "worst_cosine_similarity": worst_cos,
            "failures": failures,
        }


@dataclass(frozen=True)
class TpeHostLogitsOutput:
    """HostAuxiliaryOutput implementation recording committed steps into TpeLogitsCollector."""

    logits_cpu: Any
    collector: TpeLogitsCollector

    def consume(self, batch: Any, commits: Sequence[Any]) -> None:
        if not self.collector.enabled:
            return
        reqs = getattr(batch, "reqs", ())
        for idx, (req, commit) in enumerate(zip(reqs, commits)):
            if commit is None:
                continue
            token_ids = getattr(commit, "token_ids", ())
            if not token_ids:
                continue
            output_step = int(getattr(commit, "output_index", 0))
            rid = getattr(req, "rid", str(idx))
            origin_ids = getattr(req, "origin_input_ids", None)
            row = self.logits_cpu[idx]
            self.collector.record_step(
                rid=rid,
                output_step=output_step,
                token_id=int(token_ids[0]),
                logits_row=row,
                input_digest=_digest_token_ids(origin_ids),
            )


@dataclass(frozen=True)
class TpeDeviceLogitsOutput:
    """DeviceAuxiliaryOutput implementation with graph-safe cloned logits."""

    logits_device: Any
    collector: TpeLogitsCollector

    def copy_to_host(self, copy_tensor: Callable[[Any], Any]) -> TpeHostLogitsOutput:
        return TpeHostLogitsOutput(
            logits_cpu=copy_tensor(self.logits_device),
            collector=self.collector,
        )


class TpeLogitsObserver:
    """SamplingObserver that clones full-vocab logits only when explicitly armed."""

    def __init__(self, collector: TpeLogitsCollector) -> None:
        self.collector = collector

    def is_active(self, sampling_info: Any) -> bool:
        return self.collector.enabled

    def before_grammar(self, logits: Any, sampling_info: Any) -> Any:
        if not self.collector.enabled:
            return None
        # Defensive non-crashing guard: admission already rejects non-neutral requests with HTTP 400 (N6).
        violation = _detect_non_neutral_sampling_info(sampling_info)
        if violation is not None:
            self.collector.mark_non_neutral_violation(violation)
            return None
        # Must clone so CUDA-graph static buffers are never aliased across replays.
        return _clone_tensor_or_seq(logits)

    def after_sample(
        self, state: Any, token_ids: Any
    ) -> Optional[TpeDeviceLogitsOutput]:
        if state is None or not self.collector.enabled:
            return None
        return TpeDeviceLogitsOutput(
            logits_device=state,
            collector=self.collector,
        )
