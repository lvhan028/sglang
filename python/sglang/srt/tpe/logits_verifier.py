"""Minimal numerical logits verifier for TPE Phase 1a correctness gating."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LogitsVerificationResult:
    passed: bool
    max_abs_diff: float
    cosine_similarity: float
    all_finite: bool
    reason: str = ""


def _flatten_floats(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for item in values:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten_floats(item))
        else:
            out.append(float(item))
    return out


def verify_logits_pair(
    baseline_logits: Sequence[float],
    candidate_logits: Sequence[float],
    *,
    max_abs_diff_threshold: float = 5e-2,
    min_cosine_similarity: float = 0.999,
) -> LogitsVerificationResult:
    """Compare candidate logits against baseline logits without requiring torch/numpy."""
    base = _flatten_floats(baseline_logits)
    cand = _flatten_floats(candidate_logits)

    if len(base) != len(cand):
        return LogitsVerificationResult(
            passed=False,
            max_abs_diff=math.inf,
            cosine_similarity=0.0,
            all_finite=False,
            reason=f"shape_mismatch: baseline={len(base)} vs candidate={len(cand)}",
        )
    if not base:
        return LogitsVerificationResult(
            passed=False,
            max_abs_diff=math.inf,
            cosine_similarity=0.0,
            all_finite=False,
            reason="empty_logits",
        )

    max_abs = 0.0
    dot = 0.0
    norm_b = 0.0
    norm_c = 0.0
    for b, c in zip(base, cand):
        if not (math.isfinite(b) and math.isfinite(c)):
            return LogitsVerificationResult(
                passed=False,
                max_abs_diff=math.inf,
                cosine_similarity=0.0,
                all_finite=False,
                reason="non_finite_logits",
            )
        diff = abs(b - c)
        if diff > max_abs:
            max_abs = diff
        dot += b * c
        norm_b += b * b
        norm_c += c * c

    denom = math.sqrt(norm_b) * math.sqrt(norm_c)
    cos_sim = 1.0 if denom == 0.0 else (dot / denom)

    if max_abs > max_abs_diff_threshold:
        return LogitsVerificationResult(
            passed=False,
            max_abs_diff=max_abs,
            cosine_similarity=cos_sim,
            all_finite=True,
            reason=f"max_abs_diff_exceeded: {max_abs:.6g} > {max_abs_diff_threshold:.6g}",
        )
    if cos_sim < min_cosine_similarity:
        return LogitsVerificationResult(
            passed=False,
            max_abs_diff=max_abs,
            cosine_similarity=cos_sim,
            all_finite=True,
            reason=f"cosine_similarity_below_threshold: {cos_sim:.6f} < {min_cosine_similarity:.6f}",
        )

    return LogitsVerificationResult(
        passed=True,
        max_abs_diff=max_abs,
        cosine_similarity=cos_sim,
        all_finite=True,
        reason="ok",
    )
