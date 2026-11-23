"""Local CPU unit tests for TPE Phase 0 & Phase 1a core invariants."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import math
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_module_direct(module_name: str, rel_path: str):
    """Load a module directly from file path without triggering heavy sglang.__init__ imports."""
    full_path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Ensure parent package namespace exists for internal imports inside controller.py
for pkg in ("sglang", "sglang.srt", "sglang.srt.tpe"):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m

tpe_state = _load_module_direct("sglang.srt.tpe.state", "python/sglang/srt/tpe/state.py")
tpe_comm = _load_module_direct(
    "sglang.srt.tpe.communicator", "python/sglang/srt/tpe/communicator.py"
)
tpe_ctrl = _load_module_direct(
    "sglang.srt.tpe.controller", "python/sglang/srt/tpe/controller.py"
)
tpe_logits = _load_module_direct(
    "sglang.srt.tpe.logits_verifier", "python/sglang/srt/tpe/logits_verifier.py"
)
tpe_hot_reload = _load_module_direct(
    "sglang.srt.model_executor.model_runner_components.hot_reload",
    "python/sglang/srt/model_executor/model_runner_components/hot_reload.py",
)


class _DummyModel:
    def __init__(self) -> None:
        self.baseline_calls = 0

    def forward(
        self,
        input_ids: Any,
        positions: Any,
        forward_batch: Any,
        pp_proxy_tensors: Optional[Any] = None,
    ) -> List[float]:
        self.baseline_calls += 1
        return [float(x) * 1.5 for x in input_ids]


class TestTpeStateAndConfig(unittest.TestCase):
    def test_promotion_eligibility_f7(self) -> None:
        ok = tpe_state.TpeJobOutcome(
            job_id="j1",
            artifact_hash="abc",
            instance_generation=1,
            measurement_completed=True,
            verification_passed=True,
            restore_passed=True,
        )
        self.assertTrue(ok.compute_promotion_eligibility())

        for attr in ("cleanup_timeout", "memory_budget"):
            outcome = tpe_state.TpeJobOutcome(
                job_id="j2",
                artifact_hash="abc",
                instance_generation=1,
                measurement_completed=True,
                verification_passed=True,
                restore_passed=False,
                failure_attribution=attr,
            )
            self.assertTrue(outcome.compute_promotion_eligibility())

        for attr in (
            "sentinel_mismatch",
            "kv_violation",
            "unknown_taint",
            "contract_violation",
        ):
            outcome = tpe_state.TpeJobOutcome(
                job_id="j3",
                artifact_hash="abc",
                instance_generation=1,
                measurement_completed=True,
                verification_passed=True,
                restore_passed=True,
                failure_attribution=attr,
            )
            self.assertFalse(outcome.compute_promotion_eligibility())

        canary_bad = tpe_state.TpeJobOutcome(
            job_id="j4",
            artifact_hash="abc",
            instance_generation=1,
            measurement_completed=True,
            verification_passed=True,
            restore_passed=True,
            canary_violations_detected=True,
        )
        self.assertFalse(canary_bad.compute_promotion_eligibility())

    def test_phase1a_config_matrix_f5_f6(self) -> None:
        valid_kwargs = dict(
            prefill_cuda_graph_disabled=True,
            decode_cuda_graph_full=True,
            enable_torch_compile=False,
            enable_megakernel_or_monokernel=False,
            enable_post_capture_kv_sizing=False,
            enable_speculative=False,
            disaggregation_mode="null",
            pp_size=1,
            ep_size=1,
            enable_dp_attention=False,
            enable_hierarchical_cache=False,
            enable_overlap=False,
        )
        self.assertEqual(tpe_state.validate_phase1a_config(**valid_kwargs), [])
        tpe_state.assert_phase1a_config(**valid_kwargs)

        invalid_overrides = [
            {"prefill_cuda_graph_disabled": False},
            {"decode_cuda_graph_full": False},
            {"enable_torch_compile": True},
            {"enable_megakernel_or_monokernel": True},
            {"enable_post_capture_kv_sizing": True},
            {"enable_speculative": True},
            {"disaggregation_mode": "decode"},
            {"pp_size": 2},
            {"ep_size": 4},
            {"enable_dp_attention": True},
            {"enable_hierarchical_cache": True},
            {"enable_overlap": True},
        ]
        for override in invalid_overrides:
            cfg = dict(valid_kwargs)
            cfg.update(override)
            reasons = tpe_state.validate_phase1a_config(**cfg)
            self.assertTrue(len(reasons) >= 1, f"Expected failure for {override}")

    def test_replica_generation_tracker(self) -> None:
        tracker = tpe_state.ReplicaGenerationTracker(replica_id="slow-0")
        self.assertEqual(tracker.begin_generation(), 1)
        self.assertEqual(
            tracker.lifecycle_state, tpe_state.ReplicaLifecycleState.STARTING
        )
        tracker.mark_ready()
        self.assertEqual(tracker.lifecycle_state, tpe_state.ReplicaLifecycleState.READY)

        tracker.begin_phase("capture", now=100.0)
        self.assertFalse(tracker.check_phase_timeout(now=150.0, timeout_sec=60.0))
        self.assertTrue(tracker.check_phase_timeout(now=161.0, timeout_sec=60.0))

        tracker.mark_tainted("capture_timeout")
        self.assertEqual(
            tracker.lifecycle_state, tpe_state.ReplicaLifecycleState.TAINTED
        )
        tracker.record_gpu_health(
            tpe_state.GpuHealthStatus(healthy=True, free_memory_mb=120000)
        )
        self.assertEqual(
            tracker.lifecycle_state, tpe_state.ReplicaLifecycleState.RECYCLING
        )
        self.assertEqual(tracker.begin_generation(), 2)


@dataclass
class _MockReq:
    op: str
    job_id: str
    op_id: str
    instance_generation: int
    expected_artifact_hash: Optional[str] = None


@dataclass
class _MockResp:
    op: str
    job_id: str
    op_id: str
    instance_generation: int
    responder_id: str
    success: bool = True
    error_message: Optional[str] = None
    state: str = "CANDIDATE_READY"


class TestTpeCommunicator(unittest.TestCase):
    def test_filters_stale_generation_wrong_op_and_duplicates_f4(self) -> None:
        sent: List[_MockReq] = []
        comm = tpe_comm.TpeTransactionCommunicator(
            send=sent.append,
            fan_out=2,
            expected_responders={"r0", "r1"},
        )

        async def _run():
            req = _MockReq(
                op="hot_reload", job_id="job-9", op_id="op-1", instance_generation=3
            )
            task = asyncio.create_task(comm.call(req, timeout_s=2.0))
            await asyncio.sleep(0)

            # Stale generation -> dropped
            self.assertFalse(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 2, responder_id="r0")
                )
            )
            # Wrong op_id -> dropped
            self.assertFalse(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-OLD", 3, responder_id="r0")
                )
            )
            # Unexpected responder -> dropped
            self.assertFalse(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, responder_id="r99")
                )
            )
            # Valid r0
            self.assertTrue(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, responder_id="r0")
                )
            )
            self.assertFalse(task.done())
            # Duplicate r0 -> dropped
            self.assertFalse(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, responder_id="r0")
                )
            )
            self.assertFalse(task.done())
            # Valid r1 -> completes waiter
            self.assertTrue(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, responder_id="r1")
                )
            )
            resps = await task
            self.assertEqual(len(resps), 2)
            self.assertEqual({r.responder_id for r in resps}, {"r0", "r1"})
            self.assertEqual(comm.dropped_response_count, 4)

        asyncio.run(_run())


class TestTpeController(unittest.TestCase):
    def test_admission_modes(self) -> None:
        ctrl = tpe_ctrl.TpeSchedulerController(instance_generation=1)
        # Starts in BASELINE_PROBE_ONLY
        self.assertTrue(
            ctrl.can_admit_request(req_generation=1, is_baseline_probe=True)
        )
        self.assertFalse(
            ctrl.can_admit_request(
                req_job_id="j1", req_generation=1, is_baseline_probe=False
            )
        )

        ctrl.set_admission(tpe_state.TpeAdmissionMode.CLOSED)
        self.assertFalse(
            ctrl.can_admit_request(req_generation=1, is_baseline_probe=True)
        )
        self.assertFalse(
            ctrl.can_admit_request(
                req_job_id="j1", req_generation=1, is_baseline_probe=False
            )
        )

        ctrl.set_admission(tpe_state.TpeAdmissionMode.JOB_ONLY, job_id="j1")
        self.assertTrue(
            ctrl.can_admit_request(
                req_job_id="j1", req_generation=1, is_baseline_probe=False
            )
        )
        self.assertFalse(
            ctrl.can_admit_request(
                req_job_id="j2", req_generation=1, is_baseline_probe=False
            )
        )
        self.assertFalse(
            ctrl.can_admit_request(
                req_job_id="j1", req_generation=2, is_baseline_probe=False
            )
        )
        self.assertFalse(
            ctrl.can_admit_request(req_generation=1, is_baseline_probe=True)
        )

    def test_non_blocking_drain_and_f3_cleanup_order(self) -> None:
        now = [100.0]
        ctrl = tpe_ctrl.TpeSchedulerController(
            instance_generation=1, clock_fn=lambda: now[0]
        )
        ctrl.begin_drain(
            op_id="op-td-1",
            job_id="j1",
            instance_generation=1,
            purpose="teardown",
            timeout_s=10.0,
        )
        self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.DRAINING)
        self.assertEqual(ctrl.admission_mode, tpe_state.TpeAdmissionMode.CLOSED)

        # Still has inflight work -> non-blocking poll returns None
        now[0] = 102.0
        res1 = ctrl.poll_drain(
            is_fully_idle_fn=lambda: False,
            result_queue_empty_fn=lambda: True,
            sync_streams_fn=lambda: None,
        )
        self.assertIsNone(res1)

        # Once idle -> strict F3 order: sync_streams -> finalize_canary -> clear_keep_alive
        now[0] = 103.0
        events: List[str] = []
        res2 = ctrl.poll_drain(
            is_fully_idle_fn=lambda: True,
            result_queue_empty_fn=lambda: True,
            sync_streams_fn=lambda: events.append("sync_streams"),
            finalize_canary_fn=lambda: (
                events.append("finalize_canary"),
                {"canary_violations_detected": False},
            )[1],
            clear_keep_alive_fn=lambda: events.append("clear_keep_alive"),
        )
        self.assertIsNotNone(res2)
        assert res2 is not None
        self.assertTrue(res2["success"])
        self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.RESTORING)
        self.assertEqual(
            events, ["sync_streams", "finalize_canary", "clear_keep_alive"]
        )

    def test_drain_timeout_marks_tainted(self) -> None:
        now = [10.0]
        ctrl = tpe_ctrl.TpeSchedulerController(
            instance_generation=1, clock_fn=lambda: now[0]
        )
        ctrl.begin_drain(
            op_id="op-3",
            job_id="j3",
            instance_generation=1,
            purpose="hot_reload",
            timeout_s=5.0,
        )
        now[0] = 16.0
        res = ctrl.poll_drain(
            is_fully_idle_fn=lambda: False,
            result_queue_empty_fn=lambda: False,
            sync_streams_fn=lambda: None,
        )
        self.assertIsNotNone(res)
        assert res is not None
        self.assertFalse(res["success"])
        self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.TAINTED)

    def test_baseline_sentinel_failure_marks_tainted(self) -> None:
        ctrl = tpe_ctrl.TpeSchedulerController(instance_generation=1)
        ctrl.begin_baseline_verifying()
        self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.BASELINE_VERIFYING)
        ctrl.complete_baseline_verifying(sentinel_passed=False)
        self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.TAINTED)
        self.assertEqual(ctrl.admission_mode, tpe_state.TpeAdmissionMode.CLOSED)


class TestHotReloadAndFixtures(unittest.TestCase):
    def _read_fixture(self, name: str) -> str:
        with open(os.path.join(FIXTURES_DIR, name), "r", encoding="utf-8") as f:
            return f.read()

    def test_stable_shell_signature_and_canary_wrapping_never_bypassed(self) -> None:
        model = _DummyModel()
        orig_sig = inspect.signature(model.forward)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = tpe_hot_reload.HotReloadManager(model, rank=0, staging_root=tmpdir)
            mgr.install_stable_shell()
            self.assertEqual(inspect.signature(model.forward), orig_sig)

            # Simulate kv_canary wrapping model.forward AFTER install_stable_shell()
            canary_hits = 0
            inner_shell = model.forward

            def fake_canary_wrapper(*args, **kwargs):
                nonlocal canary_hits
                canary_hits += 1
                return inner_shell(*args, **kwargs)

            model.forward = fake_canary_wrapper

            # 1. Baseline
            out_base = model.forward([1, 2, 3], [0, 1, 2], None)
            self.assertEqual(out_base, [1.5, 3.0, 4.5])
            self.assertEqual(canary_hits, 1)

            # 2. cand_noop
            mgr.load_and_activate(
                job_id="noop", code_utf8=self._read_fixture("cand_noop.py")
            )
            out_noop = model.forward([1, 2, 3], [0, 1, 2], None)
            self.assertEqual(out_noop, [1.5, 3.0, 4.5])
            self.assertEqual(canary_hits, 2)

            # Teardown order check: destroy_graphs MUST run BEFORE candidate_state is cleared
            order: List[str] = []

            def tracked_destroy():
                order.append("destroy_graphs")
                self.assertIn("_engine_baseline_forward", mgr.candidate_state)

            err = mgr.restore_baseline(destroy_graphs_fn=tracked_destroy)
            self.assertIsNone(err)
            self.assertEqual(order, ["destroy_graphs"])
            self.assertEqual(mgr.candidate_state, {})

            # 3. cand_observable
            mgr.load_and_activate(
                job_id="obs", code_utf8=self._read_fixture("cand_observable.py")
            )
            out_obs = model.forward([1, 2, 3], [0, 1, 2], None)
            self.assertEqual(out_obs, [1.5, 3.0, 4.5])
            self.assertEqual(canary_hits, 3)
            self.assertIsNone(mgr.restore_baseline())

            # 4. cand_syntax_err -> fails cleanly, baseline stays intact
            with self.assertRaises(SyntaxError):
                mgr.load_and_activate(
                    job_id="syn", code_utf8=self._read_fixture("cand_syntax_err.py")
                )
            self.assertFalse(mgr.is_candidate_active)
            out_after = model.forward([1, 2, 3], [0, 1, 2], None)
            self.assertEqual(out_after, [1.5, 3.0, 4.5])
            self.assertEqual(canary_hits, 4)


class TestLogitsVerifierAndStaticContracts(unittest.TestCase):
    def test_logits_verifier(self) -> None:
        ok = tpe_logits.verify_logits_pair([1.0, 2.0, 3.0], [1.001, 2.0, 2.999])
        self.assertTrue(ok.passed)

        bad_diff = tpe_logits.verify_logits_pair([1.0, 2.0, 3.0], [1.0, 5.0, 3.0])
        self.assertFalse(bad_diff.passed)
        self.assertIn("max_abs_diff_exceeded", bad_diff.reason)

        bad_nan = tpe_logits.verify_logits_pair([1.0, 2.0], [1.0, math.nan])
        self.assertFalse(bad_nan.passed)
        self.assertFalse(bad_nan.all_finite)

    def test_all_tpe_files_parse_cleanly(self) -> None:
        rel_files = [
            "python/sglang/srt/tpe/__init__.py",
            "python/sglang/srt/tpe/state.py",
            "python/sglang/srt/tpe/communicator.py",
            "python/sglang/srt/tpe/controller.py",
            "python/sglang/srt/tpe/logits_verifier.py",
            "python/sglang/srt/model_executor/model_runner_components/hot_reload.py",
            "python/sglang/srt/model_executor/model_runner_components/tpe_graph_adapter.py",
            "python/sglang/srt/utils/phase_checker.py",
            "python/sglang/srt/kv_canary/runner/future_tensor.py",
            "python/sglang/srt/kv_canary/runner/violation_manager.py",
            "python/sglang/srt/kv_canary/runner/canary_manager.py",
            "python/sglang/srt/managers/io_struct.py",
            "python/sglang/srt/model_executor/model_runner.py",
            "scripts/tpe/record_baseline.py",
            "test/srt/tpe/fixtures/cand_noop.py",
            "test/srt/tpe/fixtures/cand_observable.py",
            "test/srt/tpe/fixtures/cand_bad_logits.py",
            "test/srt/tpe/fixtures/cand_hang_or_ima.py",
        ]
        for rel in rel_files:
            full = os.path.join(REPO_ROOT, rel)
            with open(full, "r", encoding="utf-8") as f:
                ast.parse(f.read(), filename=full)

    def test_io_struct_tpe_reqs_inherit_basereq(self) -> None:
        full = os.path.join(REPO_ROOT, "python/sglang/srt/managers/io_struct.py")
        with open(full, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=full)
        found: Dict[str, List[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in (
                "TpeTransactionReqInput",
                "TpeTransactionReqOutput",
            ):
                found[node.name] = [
                    b.id for b in node.bases if isinstance(b, ast.Name)
                ]
        self.assertEqual(found.get("TpeTransactionReqInput"), ["BaseReq"])
        self.assertEqual(found.get("TpeTransactionReqOutput"), ["BaseReq"])

    def test_model_runner_installs_hot_reload_before_canary(self) -> None:
        full = os.path.join(
            REPO_ROOT, "python/sglang/srt/model_executor/model_runner.py"
        )
        with open(full, "r", encoding="utf-8") as f:
            src = f.read()
        idx_shell = src.index("install_stable_shell()")
        idx_canary = src.index("self.canary_manager = install_canary(")
        self.assertLess(idx_shell, idx_canary)

    def test_kv_canary_and_phase_checker_methods_present(self) -> None:
        expected_methods = {
            "python/sglang/srt/utils/phase_checker.py": ["disable_assert"],
            "python/sglang/srt/kv_canary/runner/future_tensor.py": ["drain"],
            "python/sglang/srt/kv_canary/runner/violation_manager.py": [
                "finalize_and_check"
            ],
            "python/sglang/srt/kv_canary/runner/canary_manager.py": [
                "recapture_scope",
                "snapshot_kernel_run_counts",
                "finalize_job_check",
            ],
        }
        for rel, methods in expected_methods.items():
            full = os.path.join(REPO_ROOT, rel)
            with open(full, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=full)
            defined = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            }
            for method in methods:
                self.assertIn(method, defined, f"Missing {method} in {rel}")


if __name__ == "__main__":
    unittest.main()
