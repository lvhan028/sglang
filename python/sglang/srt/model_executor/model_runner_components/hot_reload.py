"""Stable dispatch shell and cooperative HotReloadManager for TPE (R2, F3, F5).

Design invariants:
1. Installed on `model.forward` BEFORE `install_canary()` so `kv_canary` always
   wraps the stable dispatch shell and can never be bypassed by hot reload.
2. Preserves `inspect.signature(baseline_forward)` via `functools.wraps` for
   graph runners and `support_pp`.
3. Candidate mutable state lives exclusively in `candidate_state`; teardown
   MUST run AFTER old CUDA graphs are destroyed so graphs never reference freed
   candidate tensors.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedCandidate:
    job_id: str
    artifact_hash: str
    module_name: str
    work_dir: str
    forward_fn: Callable[..., Any]
    setup_fn: Optional[Callable[[Any, Dict[str, Any]], None]]
    teardown_fn: Optional[Callable[[Any, Dict[str, Any]], None]]
    meta: Dict[str, Any]


class HotReloadManager:
    """Owns the stable dispatch shell on `model.forward` and candidate lifecycle."""

    def __init__(
        self,
        model: Any,
        *,
        rank: int = 0,
        staging_root: Optional[str] = None,
    ) -> None:
        self.model = model
        self.rank = rank
        self.staging_root = staging_root or os.path.join(
            tempfile.gettempdir(), "sglang_tpe"
        )
        self._baseline_forward: Callable[..., Any] = model.forward
        self._active_candidate: Optional[LoadedCandidate] = None
        self._candidate_state: Dict[str, Any] = {}
        self._installed: bool = False

    @property
    def active_candidate(self) -> Optional[LoadedCandidate]:
        return self._active_candidate

    @property
    def is_candidate_active(self) -> bool:
        return self._active_candidate is not None

    @property
    def candidate_state(self) -> Dict[str, Any]:
        return self._candidate_state

    def install_stable_shell(self) -> Callable[..., Any]:
        """Install the stable dispatch wrapper on `model.forward`.

        MUST be called BEFORE `install_canary()` so `kv_canary` wraps this shell.
        """
        if self._installed:
            return self.model.forward

        baseline = self._baseline_forward
        manager = self

        @functools.wraps(baseline)
        def _tpe_stable_forward(input_ids, positions, forward_batch, *args, **kwargs):
            cand = manager._active_candidate
            if cand is None:
                return baseline(input_ids, positions, forward_batch, *args, **kwargs)
            return cand.forward_fn(
                manager.model,
                input_ids,
                positions,
                forward_batch,
                manager._candidate_state,
                *args,
                **kwargs,
            )

        self.model.forward = _tpe_stable_forward
        self._installed = True
        return _tpe_stable_forward

    @staticmethod
    def compute_artifact_hash(code_utf8: str) -> str:
        return hashlib.sha256(code_utf8.encode("utf-8")).hexdigest()

    def load_and_activate(
        self,
        *,
        job_id: str,
        code_utf8: str,
        artifact_hash: Optional[str] = None,
        entry_fn: str = "forward",
    ) -> LoadedCandidate:
        """Stage candidate code by content hash, load module, run setup(), and activate."""
        if not self._installed:
            raise RuntimeError(
                "HotReloadManager.install_stable_shell() must be called before load_and_activate()"
            )
        if self._active_candidate is not None:
            raise RuntimeError(
                f"Candidate {self._active_candidate.job_id!r} is still active; teardown first."
            )

        actual_hash = self.compute_artifact_hash(code_utf8)
        if artifact_hash is not None and artifact_hash != actual_hash:
            raise ValueError(
                f"Artifact hash mismatch: expected {artifact_hash}, computed {actual_hash}"
            )

        safe_tag = f"{actual_hash[:16]}_{job_id}".replace("/", "_")
        work_dir = os.path.join(self.staging_root, f"rank_{self.rank}", safe_tag)
        os.makedirs(work_dir, exist_ok=True)
        src_path = os.path.join(work_dir, "candidate.py")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code_utf8)

        module_name = f"sglang_tpe_cand_{actual_hash[:12]}_{abs(hash(job_id)) & 0xFFFF:04x}"
        spec = importlib.util.spec_from_file_location(module_name, src_path)
        if spec is None or spec.loader is None:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to construct module spec for {src_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        forward_fn = getattr(module, entry_fn, None)
        if not callable(forward_fn):
            sys.modules.pop(module_name, None)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ValueError(
                f"Candidate module missing callable entry point {entry_fn!r}"
            )

        setup_fn = getattr(module, "setup", None)
        if setup_fn is not None and not callable(setup_fn):
            sys.modules.pop(module_name, None)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ValueError("Candidate attribute 'setup' must be callable if defined")

        teardown_fn = getattr(module, "teardown", None)
        if teardown_fn is not None and not callable(teardown_fn):
            sys.modules.pop(module_name, None)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ValueError("Candidate attribute 'teardown' must be callable if defined")

        meta = getattr(module, "CANDIDATE_META", {})
        if not isinstance(meta, dict):
            sys.modules.pop(module_name, None)
            shutil.rmtree(work_dir, ignore_errors=True)
            raise ValueError("CANDIDATE_META must be a dict if defined")

        state: Dict[str, Any] = {"_engine_baseline_forward": self._baseline_forward}
        if setup_fn is not None:
            try:
                setup_fn(self.model, state)
            except Exception:
                state.clear()
                sys.modules.pop(module_name, None)
                shutil.rmtree(work_dir, ignore_errors=True)
                raise

        candidate = LoadedCandidate(
            job_id=job_id,
            artifact_hash=actual_hash,
            module_name=module_name,
            work_dir=work_dir,
            forward_fn=forward_fn,
            setup_fn=setup_fn,
            teardown_fn=teardown_fn,
            meta=dict(meta),
        )
        self._candidate_state = state
        self._active_candidate = candidate
        return candidate

    def restore_baseline(
        self,
        *,
        destroy_graphs_fn: Optional[Callable[[], None]] = None,
    ) -> Optional[str]:
        """Restore baseline dispatch and clean up candidate resources in strict F3 order.

        1. Switch dispatch pointer back to baseline (None).
        2. Destroy old CUDA graphs BEFORE freeing `candidate_state` tensors.
        3. Call candidate `teardown(model, state)` and clear `candidate_state`.
        4. Unload candidate module from `sys.modules`, remove staging dir, run GC.

        Returns an error string if teardown raised (caller should mark TAINTED), else None.
        """
        cand = self._active_candidate
        self._active_candidate = None

        # Step 1: Destroy old graphs while candidate buffers are still alive.
        if destroy_graphs_fn is not None:
            destroy_graphs_fn()

        error_msg: Optional[str] = None
        if cand is not None:
            if cand.teardown_fn is not None:
                try:
                    cand.teardown_fn(self.model, self._candidate_state)
                except Exception as exc:
                    error_msg = f"Candidate teardown failed: {type(exc).__name__}: {exc}"
                    logger.exception(error_msg)
            self._candidate_state.clear()
            sys.modules.pop(cand.module_name, None)
            shutil.rmtree(cand.work_dir, ignore_errors=True)
        else:
            self._candidate_state.clear()

        gc.collect()
        return error_msg
