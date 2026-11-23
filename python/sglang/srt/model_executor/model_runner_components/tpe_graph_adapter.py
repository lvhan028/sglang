"""Graph invalidation, cleanup, and recapture adapter for TPE Phase 1a (R3, F1, F5).

Preserves the full backend capture lifecycle (including borrow-pool cleanup and
`canary_manager.recapture_scope()`) while keeping weights, KV pool, and
attention backends resident.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TpeGraphRecaptureStats:
    cleanup_time_s: float
    capture_time_s: float
    capture_bs: tuple[int, ...]


class TpeGraphAdapter:
    """Manages decode CUDA graph invalidation and recapture for TPE."""

    def __init__(self, model_runner: ModelRunner) -> None:
        self.model_runner = model_runner

    def destroy_decode_graphs(self) -> float:
        """Destroy existing decode CUDA graphs and reset global graph/borrow pools."""
        from sglang.srt.model_executor.runner import set_global_graph_memory_pool
        from sglang.srt.model_executor.runner_utils import pool as graph_pool_utils
        from sglang.srt.platforms import current_platform

        tic = time.perf_counter()
        current_platform.synchronize()

        runner = self.model_runner.decode_cuda_graph_runner
        if runner is not None and runner is not self.model_runner.eager_runner:
            backend = getattr(runner, "backend", None)
            if backend is not None and hasattr(backend, "cleanup"):
                backend.cleanup()
            current_platform.synchronize()
        self.model_runner.decode_cuda_graph_runner = None
        self.model_runner.decode_cuda_graph_captured = False

        # Preserve full borrow-pool lifecycle rather than only clearing the global handle.
        reset_borrow = getattr(graph_pool_utils, "reset_graph_pool_borrow_state", None)
        if callable(reset_borrow):
            reset_borrow()
        set_global_graph_memory_pool(None)
        gc.collect()
        current_platform.empty_cache()
        return time.perf_counter() - tic

    def recapture_decode_graphs(self) -> TpeGraphRecaptureStats:
        """Invalidate old decode graphs and recapture inside `canary_manager.recapture_scope()`."""
        from sglang.srt.model_executor.graph_memory_usage import (
            replace_graph_memory_usage,
            replace_graph_time_usage,
        )
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            capture_decode_graph,
        )
        from sglang.srt.platforms import current_platform

        cleanup_s = self.destroy_decode_graphs()

        canary_mgr = getattr(self.model_runner, "canary_manager", None)
        recapture_ctx = (
            canary_mgr.recapture_scope()
            if canary_mgr is not None and not self.model_runner.is_draft_worker
            else contextlib.nullcontext()
        )

        tic = time.perf_counter()
        with recapture_ctx:
            self.model_runner._kernel_warmed_up = False
            capture = capture_decode_graph(model_runner=self.model_runner)
            if capture.runner is None:
                raise RuntimeError(
                    "TPE decode CUDA graph recapture produced no graph runner."
                )
            self.model_runner.decode_cuda_graph_runner = capture.runner
            self.model_runner.decode_cuda_graph_captured = True
            self.model_runner.decode_cuda_graph_capture_bs = list(
                getattr(capture.runner, "capture_bs", []) or []
            )
            self.model_runner.graph_memory_usage = replace_graph_memory_usage(
                self.model_runner.graph_memory_usage,
                capture.memory_usage,
                phases=("decode",),
            )
            self.model_runner.graph_time_usage = replace_graph_time_usage(
                self.model_runner.graph_time_usage,
                capture.time_usage,
                phases=("decode",),
            )
            current_platform.synchronize()
        capture_s = time.perf_counter() - tic

        return TpeGraphRecaptureStats(
            cleanup_time_s=cleanup_s,
            capture_time_s=capture_s,
            capture_bs=tuple(self.model_runner.decode_cuda_graph_capture_bs),
        )
