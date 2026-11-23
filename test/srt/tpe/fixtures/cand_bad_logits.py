"""Phase 1 acceptance fixture: intentionally corrupted logits output."""

CANDIDATE_META = {
    "name": "cand_bad_logits",
    "supported_modes": ["decode", "extend"],
}


def forward(model, input_ids, positions, forward_batch, state, *args, **kwargs):
    out = state["_engine_baseline_forward"](
        input_ids, positions, forward_batch, *args, **kwargs
    )
    if hasattr(out, "next_token_logits") and out.next_token_logits is not None:
        out.next_token_logits = out.next_token_logits * 0.0 - 999.0
    return out
