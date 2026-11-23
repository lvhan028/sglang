"""Phase 1 acceptance fixture: intentionally hangs or raises fatal runtime error."""

import time

CANDIDATE_META = {
    "name": "cand_hang_or_ima",
    "supported_modes": ["decode", "extend"],
}


def forward(model, input_ids, positions, forward_batch, state, *args, **kwargs):
    mode = state.get("fault_mode", "hang")
    if mode == "hang":
        time.sleep(3600)
    raise RuntimeError("Simulated fatal CUDA illegal memory access (IMA)")
