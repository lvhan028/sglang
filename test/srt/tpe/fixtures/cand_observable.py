"""Phase 1 acceptance fixture: observable candidate (proves active dispatch in graph & eager)."""

CANDIDATE_META = {
    "name": "cand_observable",
    "supported_modes": ["decode", "extend"],
    "observable_delta": 0.125,
}


def setup(model, state):
    state["delta"] = 0.125


def forward(model, input_ids, positions, forward_batch, state, *args, **kwargs):
    out = state["_engine_baseline_forward"](
        input_ids, positions, forward_batch, *args, **kwargs
    )
    # Apply a deterministic, non-in-place observable shift on logits output
    if hasattr(out, "next_token_logits") and out.next_token_logits is not None:
        out.next_token_logits = out.next_token_logits + state["delta"]
    return out


def teardown(model, state):
    state.clear()
