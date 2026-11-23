"""Local CPU unit & extracted-method behavioral tests for TPE Phase 0 & Phase 1a."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import logging
import math
import os
import sys
import tempfile
import time
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


def _extract_method(rel_path: str, class_name: str, method_name: str, extra_globals: Dict[str, Any]):
    """Extract and compile a real production class method from AST for isolated behavioral testing."""
    full_path = os.path.join(REPO_ROOT, rel_path)
    with open(full_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=full_path)
    target_fn = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    item_copy = ast.FunctionDef(
                        name=item.name,
                        args=item.args,
                        body=item.body,
                        decorator_list=[],
                        returns=item.returns,
                    )
                    ast.fix_missing_locations(item_copy)
                    target_fn = item_copy
                    break
    assert target_fn is not None, f"Could not find {class_name}.{method_name} in {rel_path}"
    mod_ast = ast.Module(body=[target_fn], type_ignores=[])
    ast.fix_missing_locations(mod_ast)
    ns: Dict[str, Any] = {
        "os": os,
        "time": time,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "logger": logging.getLogger(f"test.{class_name}"),
    }
    ns.update(extra_globals)
    exec(compile(mod_ast, full_path, "exec"), ns)
    return ns[method_name]


# Ensure parent package namespace exists for internal imports inside TPE modules
for pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.tpe",
    "sglang.srt.model_executor",
    "sglang.srt.model_executor.model_runner_components",
):
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
tpe_graph_adapter = _load_module_direct(
    "sglang.srt.model_executor.model_runner_components.tpe_graph_adapter",
    "python/sglang/srt/model_executor/model_runner_components/tpe_graph_adapter.py",
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

        for override in (
            {"prefill_cuda_graph_disabled": False},
            {"enable_post_capture_kv_sizing": True},
            {"enable_overlap": True},
            {"enable_speculative": True},
        ):
            cfg = dict(valid_kwargs, **override)
            self.assertTrue(len(tpe_state.validate_phase1a_config(**cfg)) >= 1)

    def test_replica_generation_tracker(self) -> None:
        tracker = tpe_state.ReplicaGenerationTracker(replica_id="slow-0")
        self.assertEqual(tracker.begin_generation(), 1)
        tracker.mark_ready()
        tracker.begin_phase("capture", now=100.0)
        self.assertTrue(tracker.check_phase_timeout(now=161.0, timeout_sec=60.0))
        tracker.mark_tainted("capture_timeout")
        tracker.record_gpu_health(tpe_state.GpuHealthStatus(healthy=True))
        self.assertEqual(tracker.begin_generation(), 2)


@dataclass
class _MockReq:
    op: str
    job_id: str
    op_id: str
    instance_generation: int
    expected_artifact_hash: Optional[str] = None
    artifact_hash: Optional[str] = None
    code_utf8: Optional[str] = None
    entry_fn: str = "forward"
    drain_purpose: Optional[str] = None
    admission_mode: Optional[str] = None
    sentinel_passed: Optional[bool] = None
    timeout_s: float = 10.0


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
    artifact_hash: Optional[str] = None
    phase_timings_s: Optional[Dict[str, float]] = None
    canary_summary: Optional[Dict[str, Any]] = None


class TestTpeCommunicator(unittest.TestCase):
    def test_filters_stale_generation_hash_and_duplicates_f4(self) -> None:
        sent: List[_MockReq] = []
        comm = tpe_comm.TpeTransactionCommunicator(
            send=sent.append,
            fan_out=2,
            expected_responders={"r0", "r1"},
        )

        async def _run():
            req = _MockReq(
                op="hot_reload",
                job_id="job-9",
                op_id="op-1",
                instance_generation=3,
                expected_artifact_hash="hash_ok",
            )
            task = asyncio.create_task(comm.call(req, timeout_s=2.0))
            await asyncio.sleep(0)

            self.assertFalse(
                comm.handle_recv(_MockResp("hot_reload", "job-9", "op-1", 2, "r0"))
            )
            self.assertFalse(
                comm.handle_recv(_MockResp("hot_reload", "job-9", "op-OLD", 3, "r0"))
            )
            self.assertFalse(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, "r0", artifact_hash="wrong")
                )
            )
            self.assertTrue(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, "r0", artifact_hash="hash_ok")
                )
            )
            self.assertFalse(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, "r0", artifact_hash="hash_ok")
                )
            )
            self.assertTrue(
                comm.handle_recv(
                    _MockResp("hot_reload", "job-9", "op-1", 3, "r1", artifact_hash="hash_ok")
                )
            )
            resps = await task
            self.assertEqual({r.responder_id for r in resps}, {"r0", "r1"})
            self.assertEqual(comm.dropped_response_count, 4)

        asyncio.run(_run())


class TestExtractedProductionBehavior(unittest.TestCase):
    """Behavioral tests executing REAL extracted production methods (R1 #1, #3, R2 N1-N3)."""

    def test_model_runner_post_memory_pool_init_completes_both_modes(self) -> None:
        calls: List[str] = []

        def fake_install_canary(**kwargs):
            calls.append("install_canary")
            return "canary_mgr"

        fn = _extract_method(
            "python/sglang/srt/model_executor/model_runner.py",
            "ModelRunner",
            "_init_post_memory_pool_components",
            {
                "install_canary": fake_install_canary,
                "get_parallel": lambda: types.SimpleNamespace(tp_rank=0),
            },
        )

        for enable_tpe in ("0", "1"):
            calls.clear()
            os.environ["SGLANG_ENABLE_TPE"] = enable_tpe
            try:
                mr = types.SimpleNamespace(
                    is_draft_worker=False,
                    model=_DummyModel(),
                    server_args=None,
                    _token_oracle_manager=None,
                    init_kv_index_translator=lambda: calls.append("init_kv_index_translator"),
                    init_ngram_embedding_manager=lambda: calls.append("init_ngram_embedding_manager"),
                    maybe_init_watermark_state=lambda: calls.append("maybe_init_watermark_state"),
                    maybe_init_hisparse_coordinator=lambda: calls.append("maybe_init_hisparse_coordinator"),
                    init_routed_experts_capturer=lambda: calls.append("init_routed_experts_capturer"),
                    init_indexer_capturer=lambda: calls.append("init_indexer_capturer"),
                )
                fn(mr)
                self.assertEqual(
                    calls,
                    [
                        "init_kv_index_translator",
                        "install_canary",
                        "init_ngram_embedding_manager",
                        "maybe_init_watermark_state",
                        "maybe_init_hisparse_coordinator",
                        "init_routed_experts_capturer",
                        "init_indexer_capturer",
                    ],
                )
                self.assertIsNone(mr.graph_shared_output)
                self.assertFalse(mr.decode_cuda_graph_captured)
                self.assertEqual(mr.decode_cuda_graph_capture_bs, [])
                if enable_tpe == "1":
                    self.assertIsNotNone(mr.tpe_hot_reload_manager)
                    self.assertIsNotNone(mr.tpe_graph_adapter)
                else:
                    self.assertIsNone(mr.tpe_hot_reload_manager)
            finally:
                os.environ.pop("SGLANG_ENABLE_TPE", None)

    def test_sweep_orchestrator_run_sweep_ignores_interval_r1_3(self) -> None:
        launched: List[str] = []
        run_sweep = _extract_method(
            "python/sglang/srt/kv_canary/runner/sweep.py",
            "SweepOrchestrator",
            "run_sweep",
            {
                "PoolKind": types.SimpleNamespace(SWA="swa"),
                "build_verify_plan_radix_sweep": lambda **kw: "plan",
                "launch_endpoints_sweep": lambda **kw: launched.append("sweep"),
            },
        )
        maybe_run_sweep = _extract_method(
            "python/sglang/srt/kv_canary/runner/sweep.py",
            "SweepOrchestrator",
            "maybe_run_sweep",
            {},
        )

        orch = types.SimpleNamespace(
            _config=types.SimpleNamespace(sweep_interval=10, real_kv_hash_mode=False),
            _device_state=types.SimpleNamespace(violation_log="log"),
            _buffer_groups=(types.SimpleNamespace(kind="full", swa_index_lut=None),),
            _endpoints=(),
            _swa_window_size=0,
            _outer_step_counter_getter=lambda: 101,
            _radix_cache="radix",
            _last_sweep_step=100,
            _sweep_passes=0,
        )
        orch.run_sweep = lambda: run_sweep(orch)

        # Interval not reached (101 - 100 < 10): maybe_run_sweep skips
        maybe_run_sweep(orch)
        self.assertEqual(orch._sweep_passes, 0)
        self.assertEqual(len(launched), 0)

        # Explicit job-end run_sweep() MUST execute even when interval not reached
        self.assertTrue(orch.run_sweep())
        self.assertEqual(orch._sweep_passes, 1)
        self.assertEqual(len(launched), 1)
        self.assertEqual(orch._last_sweep_step, 101)

    def test_scheduler_process_input_requests_enforces_admission_n1(self) -> None:
        @dataclass
        class _FakeGenReq:
            rid: str
            extra_key: Optional[str] = None
            positional_embed_overrides: Any = None

        @dataclass
        class _FakeBatchGenReq:
            batch: List[_FakeGenReq]

        @dataclass
        class _FakeAbortReq:
            rid: str
            abort_message: str = ""

        extra_globals = {
            "TokenizedGenerateReqInput": _FakeGenReq,
            "BatchTokenizedGenerateReqInput": _FakeBatchGenReq,
            "AbortReq": _FakeAbortReq,
            "RpcReqOutput": type("RpcReqOutput", (), {}),
            "get_mm": lambda: types.SimpleNamespace(mm_feature_transport="none"),
            "get_disagg": lambda: types.SimpleNamespace(enable_pd_role_switch=False),
            "is_health_check_generate_req": lambda req: False,
        }
        process_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "process_input_requests",
            extra_globals,
        )
        filter_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_filter_tpe_admitted_request",
            extra_globals,
        )
        reject_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_reject_tpe_generation_req",
            extra_globals,
        )

        ctrl = tpe_ctrl.TpeSchedulerController(instance_generation=1)
        dispatched: List[Any] = []
        aborted: List[str] = []

        sched = types.SimpleNamespace(
            session_controller=types.SimpleNamespace(maybe_reap=lambda now: None),
            _tokenized_requests=lambda r: [r] if isinstance(r, _FakeGenReq) else [],
            _request_dispatcher=lambda r: (dispatched.append(r), None)[1],
            flush_wrapper=types.SimpleNamespace(check_pending=lambda: None),
            external_corpus_manager=None,
            rust_server=None,
            _get_tpe_controller=lambda: ctrl,
            send_abort_output=lambda abort_req, req: aborted.append(req.rid),
        )
        sched._reject_tpe_generation_req = lambda req, reason: reject_fn(sched, req, reason)
        sched._filter_tpe_admitted_request = lambda req: filter_fn(sched, req)

        os.environ["SGLANG_ENABLE_TPE"] = "1"
        try:
            # 1. CLOSED: ordinary generate rejected, control req passes through
            ctrl.set_admission(tpe_state.TpeAdmissionMode.CLOSED)
            ctrl_req = _MockReq("drain", "j1", "op1", 1)
            process_fn(sched, [_FakeGenReq("r_plain"), ctrl_req])
            self.assertEqual(aborted, ["r_plain"])
            self.assertEqual(dispatched, [ctrl_req])

            # 2. JOB_ONLY(j1): only matching job+gen admitted
            dispatched.clear()
            aborted.clear()
            ctrl.set_admission(tpe_state.TpeAdmissionMode.JOB_ONLY, job_id="j1")
            process_fn(
                sched,
                [
                    _FakeGenReq("r_ok", extra_key="tpe:job=j1:gen=1"),
                    _FakeGenReq("r_wrong_job", extra_key="tpe:job=j2:gen=1"),
                    _FakeGenReq("r_wrong_gen", extra_key="tpe:job=j1:gen=2"),
                    _FakeGenReq("r_unlabeled", extra_key=None),
                ],
            )
            self.assertEqual([r.rid for r in dispatched], ["r_ok"])
            self.assertEqual(aborted, ["r_wrong_job", "r_wrong_gen", "r_unlabeled"])

            # 3. BASELINE_PROBE_ONLY: only sentinel probe admitted
            dispatched.clear()
            aborted.clear()
            ctrl.set_admission(tpe_state.TpeAdmissionMode.BASELINE_PROBE_ONLY)
            process_fn(
                sched,
                [
                    _FakeGenReq("r_sentinel", extra_key="tpe:sentinel:gen=1"),
                    _FakeGenReq("r_job", extra_key="tpe:job=j1:gen=1"),
                ],
            )
            self.assertEqual([r.rid for r in dispatched], ["r_sentinel"])
            self.assertEqual(aborted, ["r_job"])
        finally:
            os.environ.pop("SGLANG_ENABLE_TPE", None)

    def test_scheduler_handle_tpe_transaction_preconditions_and_taint_n2_n3(self) -> None:
        extra_globals = {
            "TpeTransactionReqInput": _MockReq,
            "TpeTransactionReqOutput": _MockResp,
            "current_platform": types.SimpleNamespace(synchronize=lambda: None),
        }
        handle_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "handle_tpe_transaction",
            extra_globals,
        )
        poll_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_poll_tpe_drain",
            extra_globals,
        )
        validate_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_validate_tpe_phase1a_runtime",
            extra_globals,
        )

        def make_sched(*, gen=2, flush_ok=True, capture_exc=None):
            ctrl = tpe_ctrl.TpeSchedulerController(instance_generation=gen)
            sent: List[_MockResp] = []
            worker_calls: List[str] = []
            hr_mgr = types.SimpleNamespace(is_candidate_active=False)

            def fake_hot_reload(**kw):
                worker_calls.append("hot_reload")
                if capture_exc is not None:
                    hr_mgr.is_candidate_active = True
                    raise capture_exc
                hr_mgr.is_candidate_active = True
                return (
                    types.SimpleNamespace(artifact_hash="h123"),
                    types.SimpleNamespace(cleanup_time_s=0.1, capture_time_s=0.2),
                )

            def fake_teardown(**kw):
                worker_calls.append("teardown")
                hr_mgr.is_candidate_active = False
                return None, types.SimpleNamespace(cleanup_time_s=0.1, capture_time_s=0.2)

            sched = types.SimpleNamespace(
                server_args=types.SimpleNamespace(
                    disable_overlap_schedule=True,
                    cuda_graph_backend_prefill="disabled",
                    cuda_graph_backend_decode="full",
                    enable_torch_compile=False,
                    enable_megakernel=False,
                    enable_monokernel=False,
                    speculative_algorithm=None,
                    disaggregation_mode="null",
                    pp_size=1,
                    ep_size=1,
                    enable_dp_attention=False,
                    enable_hierarchical_cache=False,
                ),
                tp_worker=types.SimpleNamespace(
                    model_runner=types.SimpleNamespace(
                        canary_manager=None,
                        tpe_hot_reload_manager=hr_mgr,
                    ),
                    tpe_hot_reload=fake_hot_reload,
                    tpe_teardown=fake_teardown,
                ),
                _tpe_pending_drain_req=None,
                is_fully_idle=lambda: True,
                flush_cache=lambda: flush_ok,
                _get_tpe_controller=lambda: ctrl,
                _tpe_sync_streams=lambda: None,
                _tpe_finalize_canary=lambda: {"canary_enabled": False},
                _tpe_clear_keep_alive=lambda: None,
                send_control_output=lambda out, req: sent.append(out),
            )
            sched._poll_tpe_drain = lambda: poll_fn(sched)
            sched._validate_tpe_phase1a_runtime = lambda: validate_fn(sched)
            return sched, ctrl, sent, worker_calls

        os.environ["SGLANG_ENABLE_TPE"] = "1"
        try:
            # 1. N2: hot_reload without drain is rejected, worker NOT called
            sched, ctrl, sent, calls = make_sched(gen=2)
            handle_fn(sched, _MockReq("hot_reload", "j1", "op1", 2, code_utf8="def forward(*a): pass"))
            self.assertFalse(sent[-1].success)
            self.assertEqual(calls, [])

            # 2. N2: stale generation teardown is rejected, worker NOT called
            handle_fn(sched, _MockReq("teardown", "other", "op2", 1))
            self.assertFalse(sent[-1].success)
            self.assertEqual(calls, [])

            # 3. N2: teardown aborts if flush_cache() returns False
            sched, ctrl, sent, calls = make_sched(gen=2, flush_ok=False)
            handle_fn(sched, _MockReq("drain", "j1", "d1", 2, drain_purpose="hot_reload"))
            handle_fn(sched, _MockReq("hot_reload", "j1", "hr1", 2, code_utf8="def forward(*a): pass"))
            handle_fn(sched, _MockReq("drain", "j1", "d2", 2, drain_purpose="teardown"))
            handle_fn(sched, _MockReq("teardown", "j1", "td1", 2))
            self.assertFalse(sent[-1].success)
            self.assertNotIn("teardown", calls)

            # 4. N3: capture failure AFTER candidate activation marks TAINTED + CLOSED
            sched, ctrl, sent, calls = make_sched(
                gen=2,
                capture_exc=tpe_state.TpePostActivationError("cuda graph capture OOM"),
            )
            handle_fn(sched, _MockReq("drain", "j1", "d1", 2, drain_purpose="hot_reload"))
            handle_fn(sched, _MockReq("hot_reload", "j1", "hr1", 2, code_utf8="def forward(*a): pass"))
            self.assertFalse(sent[-1].success)
            self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.TAINTED)
            self.assertEqual(ctrl.admission_mode, tpe_state.TpeAdmissionMode.CLOSED)
            self.assertIsNotNone(ctrl.taint_reason)

            # 5. Full happy-path Phase 1a closed loop:
            # drain -> hot_reload -> drain(teardown) -> teardown -> complete_baseline_verifying
            sched, ctrl, sent, calls = make_sched(gen=2, flush_ok=True)
            handle_fn(sched, _MockReq("drain", "j1", "d1", 2, drain_purpose="hot_reload"))
            self.assertTrue(sent[-1].success)
            self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.PREPARING)
            handle_fn(sched, _MockReq("hot_reload", "j1", "hr1", 2, code_utf8="def forward(*a): pass"))
            self.assertTrue(sent[-1].success)
            self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.CANDIDATE_READY)
            handle_fn(sched, _MockReq("drain", "j1", "d2", 2, drain_purpose="teardown"))
            self.assertTrue(sent[-1].success)
            self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.RESTORING)
            handle_fn(sched, _MockReq("teardown", "j1", "td1", 2))
            self.assertTrue(sent[-1].success)
            self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.BASELINE_VERIFYING)
            handle_fn(
                sched,
                _MockReq("complete_baseline_verifying", "j1", "bv1", 2, sentinel_passed=True),
            )
            self.assertTrue(sent[-1].success)
            self.assertEqual(ctrl.state, tpe_state.TpeInstanceState.BASELINE_READY)
            self.assertEqual(calls, ["hot_reload", "teardown"])
        finally:
            os.environ.pop("SGLANG_ENABLE_TPE", None)


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

            canary_hits = 0
            inner_shell = model.forward

            def fake_canary_wrapper(*args, **kwargs):
                nonlocal canary_hits
                canary_hits += 1
                return inner_shell(*args, **kwargs)

            model.forward = fake_canary_wrapper

            self.assertEqual(model.forward([1, 2, 3], [0, 1, 2], None), [1.5, 3.0, 4.5])
            self.assertEqual(canary_hits, 1)

            mgr.load_and_activate(
                job_id="noop", code_utf8=self._read_fixture("cand_noop.py")
            )
            self.assertEqual(model.forward([1, 2, 3], [0, 1, 2], None), [1.5, 3.0, 4.5])
            self.assertEqual(canary_hits, 2)

            order: List[str] = []

            def tracked_destroy():
                order.append("destroy_graphs")
                self.assertIn("_engine_baseline_forward", mgr.candidate_state)

            self.assertIsNone(mgr.restore_baseline(destroy_graphs_fn=tracked_destroy))
            self.assertEqual(order, ["destroy_graphs"])
            self.assertEqual(mgr.candidate_state, {})

            # Syntax error fixture stored as .py.txt so repo check-ast passes
            with self.assertRaises(SyntaxError):
                mgr.load_and_activate(
                    job_id="syn", code_utf8=self._read_fixture("cand_syntax_err.py.txt")
                )
            self.assertFalse(mgr.is_candidate_active)
            self.assertEqual(model.forward([1, 2, 3], [0, 1, 2], None), [1.5, 3.0, 4.5])
            self.assertEqual(canary_hits, 3)


class TestLogitsVerifierAndRepoLintContracts(unittest.TestCase):
    def test_logits_verifier(self) -> None:
        self.assertTrue(
            tpe_logits.verify_logits_pair([1.0, 2.0, 3.0], [1.001, 2.0, 2.999]).passed
        )
        self.assertFalse(
            tpe_logits.verify_logits_pair([1.0, 2.0, 3.0], [1.0, 5.0, 3.0]).passed
        )
        self.assertFalse(
            tpe_logits.verify_logits_pair([1.0, 2.0], [1.0, math.nan]).passed
        )

    def test_all_tpe_python_files_parse_and_not_in_forbidden_test_srt(self) -> None:
        self.assertFalse(os.path.exists(os.path.join(REPO_ROOT, "test/srt/tpe")))
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
            "python/sglang/srt/kv_canary/runner/sweep.py",
            "python/sglang/srt/managers/io_struct.py",
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/srt/managers/tokenizer_control_mixin.py",
            "python/sglang/srt/managers/tp_worker.py",
            "python/sglang/srt/entrypoints/http_server.py",
            "python/sglang/srt/model_executor/model_runner.py",
            "scripts/tpe/record_baseline.py",
            "test/manual/tpe/test_tpe_phase1a.py",
            "test/manual/tpe/fixtures/cand_noop.py",
            "test/manual/tpe/fixtures/cand_observable.py",
            "test/manual/tpe/fixtures/cand_bad_logits.py",
            "test/manual/tpe/fixtures/cand_hang_or_ima.py",
        ]
        for rel in rel_files:
            full = os.path.join(REPO_ROOT, rel)
            with open(full, "r", encoding="utf-8") as f:
                ast.parse(f.read(), filename=full)


if __name__ == "__main__":
    unittest.main()
