"""Phase 1 acceptance fixture: no-op passthrough candidate."""

CANDIDATE_META = {
    "name": "cand_noop",
    "supported_modes": ["decode", "extend"],
}


def setup(model, state):
    state["invocations"] = 0


def forward(model, input_ids, positions, forward_batch, state, *args, **kwargs):
    state["invocations"] = state.get("invocations", 0) + 1
    return state["_engine_baseline_forward"](
        input_ids, positions, forward_batch, *args, **kwargs
    )


def teardown(model, state):
    state.clear()
