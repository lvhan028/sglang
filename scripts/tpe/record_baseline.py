#!/usr/bin/env python3
"""Phase 0 utility: build eval/sentinel fixtures and record text + top-logprobs baselines.

Generates deterministic batch cases (with ignore_eos=True and fixed
max_new_tokens), records baseline generated text and top-5 logprobs via HTTP
`/generate`, and runs a native baseline-vs-baseline text self-comparison.

Note: HTTP `/generate` does not return full-vocab logits; full-vocab numerical
logits gating (using `sglang.srt.tpe.logits_verifier.verify_logits_pair`)
requires a runner-side logits export hook in Phase 1a GPU evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    description: str
    prompts: List[str]
    max_new_tokens: int = 32
    ignore_eos: bool = True
    temperature: float = 0.0


def _make_prompt(seed_tag: str, approx_words: int) -> str:
    unit = f"[{seed_tag}] The quick brown fox jumps over the lazy dog 12345. "
    repeats = max(1, approx_words // 10)
    return (unit * repeats).strip()


def build_default_eval_cases() -> List[EvalCase]:
    """Build representative Phase 0/1a deterministic batch cases."""
    return [
        EvalCase(
            case_id="bs1_short_unaligned",
            description="Single request, short non-aligned length",
            prompts=[_make_prompt("bs1_short", 65)],
            max_new_tokens=32,
        ),
        EvalCase(
            case_id="bs1_medium_unaligned",
            description="Single request, medium non-aligned length",
            prompts=[_make_prompt("bs1_med", 137)],
            max_new_tokens=32,
        ),
        EvalCase(
            case_id="bs4_graph_bucket",
            description="Batch size 4 inside decode CUDA graph bucket",
            prompts=[_make_prompt(f"bs4_{i}", 80 + i * 7) for i in range(4)],
            max_new_tokens=32,
        ),
        EvalCase(
            case_id="bs16_graph_bucket",
            description="Batch size 16 inside decode CUDA graph bucket",
            prompts=[_make_prompt(f"bs16_{i}", 60 + (i % 5) * 13) for i in range(16)],
            max_new_tokens=16,
        ),
        EvalCase(
            case_id="bs4_mixed_lengths",
            description="Mixed short and long prompts within a single batch",
            prompts=[
                _make_prompt("mix_short", 40),
                _make_prompt("mix_mid", 220),
                _make_prompt("mix_long", 900),
                _make_prompt("mix_unaligned", 137),
            ],
            max_new_tokens=16,
        ),
        EvalCase(
            case_id="bs24_eager_fallback",
            description="Batch size 24 exceeding max capture bucket (eager fallback)",
            prompts=[_make_prompt(f"bs24_{i}", 50 + (i % 3) * 11) for i in range(24)],
            max_new_tokens=8,
        ),
    ]


def build_default_sentinel_cases() -> List[EvalCase]:
    """Build lightweight independent sentinel cases for BASELINE_VERIFYING."""
    return [
        EvalCase(
            case_id="sentinel_bs1",
            description="Fast single-request baseline sentinel",
            prompts=[_make_prompt("sentinel_1", 55)],
            max_new_tokens=4,
        ),
        EvalCase(
            case_id="sentinel_bs4",
            description="Fast bs=4 graph-bucket baseline sentinel",
            prompts=[_make_prompt(f"sentinel_4_{i}", 70 + i * 5) for i in range(4)],
            max_new_tokens=4,
        ),
    ]


def write_cases_jsonl(cases: List[EvalCase], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            line = json.dumps(asdict(c), ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def build_generate_payload(
    case: EvalCase, *, instance_generation: int = 0
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "text": case.prompts,
        "sampling_params": {
            "temperature": case.temperature,
            "max_new_tokens": case.max_new_tokens,
            "ignore_eos": case.ignore_eos,
        },
        "return_logprob": True,
        "top_logprobs_num": 5,
    }
    if instance_generation > 0:
        payload["extra_key"] = f"tpe:sentinel:gen={instance_generation}"
    return payload


def validate_generate_results(
    case: EvalCase, results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if len(results) != len(case.prompts):
        raise RuntimeError(
            f"Case {case.case_id}: expected {len(case.prompts)} results, got {len(results)}"
        )
    for idx, item in enumerate(results):
        if "error" in item:
            raise RuntimeError(f"Case {case.case_id}[{idx}] error: {item['error']}")
        meta_info = item.get("meta_info") or {}
        finish_reason = meta_info.get("finish_reason") or {}
        if finish_reason.get("type") == "abort":
            raise RuntimeError(
                f"Case {case.case_id}[{idx}] aborted: {finish_reason}"
            )
        completion_tokens = meta_info.get("completion_tokens")
        if (
            case.ignore_eos
            and completion_tokens is not None
            and int(completion_tokens) != case.max_new_tokens
        ):
            raise RuntimeError(
                f"Case {case.case_id}[{idx}] completion_tokens={completion_tokens} "
                f"!= expected {case.max_new_tokens}"
            )
    return results


def _post_generate(
    base_url: str, case: EvalCase, *, instance_generation: int = 0
) -> List[Dict[str, Any]]:
    payload = build_generate_payload(case, instance_generation=instance_generation)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for case {case.case_id}")
        data = json.loads(resp.read().decode("utf-8"))
    results = data if isinstance(data, list) else [data]
    return validate_generate_results(case, results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Phase 0 TPE baseline fixtures")
    parser.add_argument("--out-dir", required=True, help="Output directory for fixtures/refs")
    parser.add_argument(
        "--base-url",
        default="",
        help="Optional running SGLang server URL (if omitted, only generates JSONL datasets)",
    )
    parser.add_argument(
        "--instance-generation",
        type=int,
        default=0,
        help="TPE instance generation (>0 attaches extra_key=tpe:sentinel:gen=<N>; 0 omits it for non-TPE servers)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eval_cases = build_default_eval_cases()
    sentinel_cases = build_default_sentinel_cases()

    eval_sha = write_cases_jsonl(eval_cases, os.path.join(args.out_dir, "eval_cases_v1.jsonl"))
    sentinel_sha = write_cases_jsonl(
        sentinel_cases, os.path.join(args.out_dir, "sentinel_cases_v1.jsonl")
    )

    meta: Dict[str, Any] = {
        "eval_cases_sha256": eval_sha,
        "sentinel_cases_sha256": sentinel_sha,
        "num_eval_cases": len(eval_cases),
        "num_sentinel_cases": len(sentinel_cases),
        "instance_generation": args.instance_generation,
    }

    if args.base_url:
        eval_ref = {
            c.case_id: _post_generate(
                args.base_url, c, instance_generation=args.instance_generation
            )
            for c in eval_cases
        }
        sentinel_ref = {
            c.case_id: _post_generate(
                args.base_url, c, instance_generation=args.instance_generation
            )
            for c in sentinel_cases
        }
        # Self-comparison pass to measure native baseline text reproducibility
        eval_pass2 = {
            c.case_id: _post_generate(
                args.base_url, c, instance_generation=args.instance_generation
            )
            for c in eval_cases
        }
        exact_matches = sum(
            1
            for cid in eval_ref
            if [r.get("text") for r in eval_ref[cid]]
            == [r.get("text") for r in eval_pass2[cid]]
        )
        meta["self_compare_exact_case_matches"] = exact_matches
        with open(os.path.join(args.out_dir, "baseline_ref_v1.json"), "w", encoding="utf-8") as f:
            json.dump(eval_ref, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out_dir, "sentinel_ref_v1.json"), "w", encoding="utf-8") as f:
            json.dump(sentinel_ref, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "baseline_meta_v1.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
