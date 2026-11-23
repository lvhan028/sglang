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
class _FakeBackend:
    DISABLED = "disabled"
    FULL = "full"
    PIECEWISE = "piecewise"


if "sglang.srt.model_executor.cuda_graph_config" not in sys.modules:
    _cg_mod = types.ModuleType("sglang.srt.model_executor.cuda_graph_config")
    _cg_mod.Backend = _FakeBackend
    sys.modules["sglang.srt.model_executor.cuda_graph_config"] = _cg_mod

tpe_graph_adapter = _load_module_direct(
    "sglang.srt.model_executor.model_runner_components.tpe_graph_adapter",
    "python/sglang/srt/model_executor/model_runner_components/tpe_graph_adapter.py",
)
tpe_logits_obs = _load_module_direct(
    "sglang.srt.tpe.logits_observer",
    "python/sglang/srt/tpe/logits_observer.py",
)
record_baseline_mod = _load_module_direct(
    "tpe_record_baseline", "scripts/tpe/record_baseline.py"
)


def _make_resolved_runtime_globals(
    *,
    prefill_backend=_FakeBackend.DISABLED,
    decode_backend=_FakeBackend.FULL,
    graph_cfg_missing: bool = False,
    enable_torch_compile: bool = False,
    enable_nvidia_megakernel: bool = False,
    enable_amd_monokernel: bool = False,
    post_capture_kv_sizing: bool = False,
    speculative_algorithm=None,
    disaggregation_mode: str = "null",
    pp_size: int = 1,
    ep_size: int = 1,
    enable_dp_attention: bool = False,
    enable_hierarchical_cache: bool = False,
) -> Dict[str, Any]:
    graph_cfg = (
        None
        if graph_cfg_missing
        else types.SimpleNamespace(
            prefill=types.SimpleNamespace(backend=prefill_backend),
            decode=types.SimpleNamespace(backend=decode_backend),
        )
    )
    exec_cfg = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            cuda_graph_config=graph_cfg,
            enable_torch_compile=enable_torch_compile,
        ),
        kernel=types.SimpleNamespace(
            enable_nvidia_megakernel=enable_nvidia_megakernel,
            enable_amd_monokernel=enable_amd_monokernel,
        ),
    )
    envs_stub = types.SimpleNamespace(
        SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=types.SimpleNamespace(
            get=lambda: post_capture_kv_sizing
        )
    )
    return {
        "get_exec": lambda: exec_cfg,
        "get_spec": lambda: types.SimpleNamespace(speculative_algorithm=speculative_algorithm),
        "get_disagg": lambda: types.SimpleNamespace(disaggregation_mode=disaggregation_mode),
        "get_parallel": lambda: types.SimpleNamespace(
            pp_size=pp_size, ep_size=ep_size, enable_dp_attention=enable_dp_attention
        ),
        "get_memory": lambda: types.SimpleNamespace(
            enable_hierarchical_cache=enable_hierarchical_cache
        ),
        "envs": envs_stub,
    }


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
    capture_logits_enabled: Optional[bool] = None
    capture_steps: Optional[List[int]] = None
    logits_save_path: Optional[str] = None
    logits_ref_path: Optional[str] = None
    max_abs_diff_threshold: float = 5e-2
    min_cosine_similarity: float = 0.999
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
    logits_summary: Optional[Dict[str, Any]] = None


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
                    sampling_observer=None,
                    supports_sampling_observer=lambda: True,
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
            finished_reason: Optional[Dict[str, Any]] = None

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
        aborted: List[_FakeAbortReq] = []

        sched = types.SimpleNamespace(
            session_controller=types.SimpleNamespace(maybe_reap=lambda now: None),
            _tokenized_requests=lambda r: [r] if isinstance(r, _FakeGenReq) else [],
            _request_dispatcher=lambda r: (dispatched.append(r), None)[1],
            flush_wrapper=types.SimpleNamespace(check_pending=lambda: None),
            external_corpus_manager=None,
            rust_server=None,
            _get_tpe_controller=lambda: ctrl,
            send_abort_output=lambda abort_req, req: aborted.append(abort_req),
        )
        sched._reject_tpe_generation_req = lambda req, reason, **kw: reject_fn(
            sched, req, reason, **kw
        )
        sched._filter_tpe_admitted_request = lambda req: filter_fn(sched, req)

        os.environ["SGLANG_ENABLE_TPE"] = "1"
        try:
            # 1. CLOSED: rejected with HTTP 503 finished_reason, control req passes through
            ctrl.set_admission(tpe_state.TpeAdmissionMode.CLOSED)
            ctrl_req = _MockReq("drain", "j1", "op1", 1)
            process_fn(sched, [_FakeGenReq("r_plain"), ctrl_req])
            self.assertEqual([a.rid for a in aborted], ["r_plain"])
            self.assertEqual(aborted[0].finished_reason["type"], "abort")
            self.assertEqual(aborted[0].finished_reason["status_code"], 503)
            self.assertEqual(dispatched, [ctrl_req])

            # 2. BASELINE_PROBE_ONLY: sentinel admitted, others rejected with HTTP 409
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
            self.assertEqual([a.rid for a in aborted], ["r_job"])
            self.assertEqual(aborted[0].finished_reason["status_code"], 409)

            # 3. JOB_ONLY(j1) after candidate ready: only matching job+gen admitted
            dispatched.clear()
            aborted.clear()
            ctrl.mark_candidate_ready("j1", "h1")
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
            self.assertEqual([a.rid for a in aborted], ["r_wrong_job", "r_wrong_gen", "r_unlabeled"])
            self.assertTrue(all(a.finished_reason["status_code"] == 409 for a in aborted))

            # 4. N6: when logits capture is armed, non-neutral sampling rejected BEFORE enqueue with HTTP 400
            dispatched.clear()
            aborted.clear()
            col_live = tpe_logits_obs.TpeLogitsCollector()
            col_live.arm(capture_steps=[0], owner_job_id="j1", owner_generation=1)
            sched.tp_worker = types.SimpleNamespace(
                model_runner=types.SimpleNamespace(tpe_logits_collector=col_live)
            )
            r_non_neutral = _FakeGenReq("r_pen", extra_key="tpe:job=j1:gen=1")
            r_non_neutral.sampling_params = types.SimpleNamespace(
                repetition_penalty=1.5, frequency_penalty=0.0, presence_penalty=0.0, min_new_tokens=0
            )
            r_neutral = _FakeGenReq("r_neut", extra_key="tpe:job=j1:gen=1")
            r_neutral.sampling_params = types.SimpleNamespace(
                repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, min_new_tokens=0
            )
            process_fn(sched, [r_non_neutral, r_neutral])
            self.assertEqual([r.rid for r in dispatched], ["r_neut"])
            self.assertEqual([a.rid for a in aborted], ["r_pen"])
            self.assertEqual(aborted[0].finished_reason["status_code"], 400)
        finally:
            os.environ.pop("SGLANG_ENABLE_TPE", None)

    def test_r1_r2_r3_r4_regressions(self) -> None:
        # R1: record_baseline payload tagging & abort/token validation
        case = record_baseline_mod.EvalCase(
            case_id="c1", description="d", prompts=["hi"], max_new_tokens=4
        )
        payload = record_baseline_mod.build_generate_payload(case, instance_generation=3)
        self.assertEqual(payload["extra_key"], "tpe:sentinel:gen=3")
        with self.assertRaises(RuntimeError):
            record_baseline_mod.validate_generate_results(
                case,
                [
                    {
                        "text": "",
                        "meta_info": {
                            "finish_reason": {"type": "abort", "message": "rejected"},
                            "completion_tokens": 0,
                        },
                    }
                ],
            )

        # R2: resolved runtime context validation
        fn_ok = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_validate_tpe_phase1a_runtime",
            _make_resolved_runtime_globals(),
        )
        fn_ok(types.SimpleNamespace(enable_overlap=False))
        with self.assertRaises(ValueError):
            fn_ok(types.SimpleNamespace(enable_overlap=True))

        fn_mega = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_validate_tpe_phase1a_runtime",
            _make_resolved_runtime_globals(enable_nvidia_megakernel=True),
        )
        with self.assertRaises(ValueError):
            fn_mega(types.SimpleNamespace(enable_overlap=False))

        fn_missing = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_validate_tpe_phase1a_runtime",
            _make_resolved_runtime_globals(graph_cfg_missing=True),
        )
        with self.assertRaises(ValueError):
            fn_missing(types.SimpleNamespace(enable_overlap=False))

        # R3: canary violation & exception during poll_drain -> TAINTED + success=False
        ctrl_v = tpe_ctrl.TpeSchedulerController(instance_generation=1)
        ctrl_v.begin_drain(op_id="d1", job_id="j1", instance_generation=1, purpose="teardown")
        res_v = ctrl_v.poll_drain(
            is_fully_idle_fn=lambda: True,
            result_queue_empty_fn=lambda: True,
            sync_streams_fn=lambda: None,
            finalize_canary_fn=lambda: {"violations_detected": True, "canary_enabled": True},
        )
        assert res_v is not None
        self.assertFalse(res_v["success"])
        self.assertEqual(ctrl_v.state, tpe_state.TpeInstanceState.TAINTED)
        self.assertTrue(res_v["canary_summary"]["violations_detected"])
        self.assertIsNone(ctrl_v.drained_job_id)

        ctrl_e = tpe_ctrl.TpeSchedulerController(instance_generation=1)
        ctrl_e.begin_drain(op_id="d2", job_id="j2", instance_generation=1, purpose="teardown")
        res_e = ctrl_e.poll_drain(
            is_fully_idle_fn=lambda: True,
            result_queue_empty_fn=lambda: True,
            sync_streams_fn=lambda: (_ for _ in ()).throw(RuntimeError("cuda IMA")),
        )
        assert res_e is not None
        self.assertFalse(res_e["success"])
        self.assertEqual(ctrl_e.state, tpe_state.TpeInstanceState.TAINTED)

        # R4: set_admission cannot reopen in PREPARING or overwrite active_job_id
        ctrl_r4 = tpe_ctrl.TpeSchedulerController(instance_generation=1)
        ctrl_r4.begin_drain(op_id="d3", job_id="j3", instance_generation=1, purpose="hot_reload")
        ctrl_r4.poll_drain(
            is_fully_idle_fn=lambda: True,
            result_queue_empty_fn=lambda: True,
            sync_streams_fn=lambda: None,
        )
        with self.assertRaises(RuntimeError):
            ctrl_r4.set_admission(tpe_state.TpeAdmissionMode.JOB_ONLY, job_id="j3")
        ctrl_r4.mark_candidate_ready("j3", "h3")
        with self.assertRaises(RuntimeError):
            ctrl_r4.set_admission(tpe_state.TpeAdmissionMode.JOB_ONLY, job_id="other")
        self.assertEqual(ctrl_r4.active_job_id, "j3")

    def test_scheduler_handle_tpe_transaction_preconditions_and_taint_n2_n3(self) -> None:
        extra_globals = {
            "TpeTransactionReqInput": _MockReq,
            "TpeTransactionReqOutput": _MockResp,
            "current_platform": types.SimpleNamespace(synchronize=lambda: None),
            **_make_resolved_runtime_globals(),
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
        snap_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_tpe_snapshot_canary_counts",
            extra_globals,
        )
        fin_fn = _extract_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_tpe_finalize_canary",
            extra_globals,
        )

        def make_sched(*, gen=2, flush_ok=True, capture_exc=None):
            ctrl = tpe_ctrl.TpeSchedulerController(instance_generation=gen)
            sent: List[_MockResp] = []
            worker_calls: List[str] = []
            hr_mgr = types.SimpleNamespace(is_candidate_active=False)
            fake_canary = types.SimpleNamespace(
                snapshot_kernel_run_counts=lambda: {"decode": 10},
                finalize_job_check=lambda **kw: {
                    "violations_detected": False,
                    "passed_baseline": kw.get("baseline_kernel_counts"),
                    "raise_on_violation": kw.get("raise_on_violation"),
                },
            )

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
                enable_overlap=False,
                tp_worker=types.SimpleNamespace(
                    model_runner=types.SimpleNamespace(
                        canary_manager=fake_canary,
                        tpe_hot_reload_manager=hr_mgr,
                    ),
                    tpe_hot_reload=fake_hot_reload,
                    tpe_teardown=fake_teardown,
                ),
                _tpe_pending_drain_req=None,
                _tpe_baseline_canary_counts=None,
                is_fully_idle=lambda: True,
                flush_cache=lambda: flush_ok,
                _get_tpe_controller=lambda: ctrl,
                _tpe_sync_streams=lambda: None,
                _tpe_clear_keep_alive=lambda: None,
                send_control_output=lambda out, req: sent.append(out),
            )
            sched._poll_tpe_drain = lambda: poll_fn(sched)
            sched._validate_tpe_phase1a_runtime = lambda: validate_fn(sched)
            sched._tpe_snapshot_canary_counts = lambda: snap_fn(sched)
            sched._tpe_finalize_canary = lambda: fin_fn(sched)
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


class TestLogitsObserverAndRepoLintContracts(unittest.TestCase):
    def test_sampling_observer_clones_and_verifies_full_vocab_logits(self) -> None:
        collector = tpe_logits_obs.TpeLogitsCollector()
        observer = tpe_logits_obs.TpeLogitsObserver(collector)

        # Inactive by default -> zero work during throughput benchmarking
        self.assertFalse(observer.is_active(None))
        self.assertIsNone(observer.before_grammar([[1.0, 2.0]], None))

        batch = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="req_a")])

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "baseline_logits.json")

            # 1. Record baseline steps 0 and 1 (verify clone protects against buffer overwrite)
            collector.arm(capture_steps=[0, 1])
            self.assertTrue(observer.is_active(None))

            buf0 = [[1.0, 2.5, -0.5, 4.0]]
            state0 = observer.before_grammar(buf0, None)
            buf0[0][0] = 999.0  # simulate graph static buffer overwrite
            dev0 = observer.after_sample(state0, [3])
            assert dev0 is not None
            host0 = dev0.copy_to_host(lambda x: x)
            host0.consume(batch, [types.SimpleNamespace(output_index=0, token_ids=(3,))])

            state1 = observer.before_grammar([[0.5, 3.0, 1.0, -1.0]], None)
            dev1 = observer.after_sample(state1, [1])
            assert dev1 is not None
            dev1.copy_to_host(lambda x: x).consume(
                batch, [types.SimpleNamespace(output_index=1, token_ids=(1,))]
            )

            # Step 2 not in capture_steps -> ignored
            state2 = observer.before_grammar([[9.0, 9.0, 9.0, 9.0]], None)
            dev2 = observer.after_sample(state2, [0])
            assert dev2 is not None
            dev2.copy_to_host(lambda x: x).consume(
                batch, [types.SimpleNamespace(output_index=2, token_ids=(0,))]
            )

            collector.disarm()
            meta = collector.save_reference(ref_path)
            self.assertEqual(meta["num_requests"], 1)
            self.assertEqual(meta["num_positions"], 2)
            self.assertEqual(meta["vocab_size"], 4)

            # 2. Matching candidate -> PASS
            collector.arm(capture_steps=[0, 1])
            collector.record_step(
                rid="req_a", output_step=0, token_id=3, logits_row=[1.001, 2.5, -0.5, 3.999]
            )
            collector.record_step(
                rid="req_a", output_step=1, token_id=1, logits_row=[0.5, 3.001, 1.0, -1.0]
            )
            report_ok = collector.compare_against_reference(ref_path)
            self.assertTrue(report_ok["passed"])
            self.assertEqual(report_ok["num_checked"], 2)
            self.assertEqual(report_ok["trajectory_divergences"], 0)

            # 3. Perturbed logits -> FAIL (max_abs_diff_exceeded)
            collector.arm(capture_steps=[0, 1])
            collector.record_step(
                rid="req_a", output_step=0, token_id=3, logits_row=[1.0, 2.5, -0.5, 5.5]
            )
            collector.record_step(
                rid="req_a", output_step=1, token_id=1, logits_row=[0.5, 3.0, 1.0, -1.0]
            )
            report_bad = collector.compare_against_reference(ref_path)
            self.assertFalse(report_bad["passed"])
            self.assertEqual(report_bad["num_failed"], 1)

            # 4. Token trajectory divergence at step 0 -> step 1 flagged as trajectory_diverged
            collector.arm(capture_steps=[0, 1])
            collector.record_step(
                rid="req_a", output_step=0, token_id=2, logits_row=[1.0, 2.5, -0.5, 4.0]
            )
            collector.record_step(
                rid="req_a", output_step=1, token_id=1, logits_row=[0.5, 3.0, 1.0, -1.0]
            )
            report_div = collector.compare_against_reference(ref_path)
            self.assertFalse(report_div["passed"])
            self.assertEqual(report_div["trajectory_divergences"], 1)

    def test_round4_n1_to_n6_regressions(self) -> None:
        # N2: tpe_graph_adapter no longer imports unused graph_pool_utils
        with open(
            os.path.join(
                REPO_ROOT,
                "python/sglang/srt/model_executor/model_runner_components/tpe_graph_adapter.py",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            self.assertNotIn("graph_pool_utils", f.read())

        # N6a: request-level neutral sampling check before enqueue
        self.assertIsNotNone(
            tpe_logits_obs.check_neutral_sampling_params(
                types.SimpleNamespace(repetition_penalty=1.2, frequency_penalty=0.0, presence_penalty=0.0, min_new_tokens=0)
            )
        )
        self.assertIsNone(
            tpe_logits_obs.check_neutral_sampling_params(
                types.SimpleNamespace(repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, min_new_tokens=0)
            )
        )

        # N6b: defensive before_grammar guard NEVER crashes scheduler; marks violation & fails save/compare
        collector = tpe_logits_obs.TpeLogitsCollector()
        observer = tpe_logits_obs.TpeLogitsObserver(collector)
        collector.arm(capture_steps=[0])
        bad_sampling = types.SimpleNamespace(
            penalizer_orchestrator=types.SimpleNamespace(is_required=True),
            has_custom_logit_processor=False,
        )
        self.assertIsNone(observer.before_grammar([[1.0, 2.0]], bad_sampling))
        with self.assertRaises(ValueError):
            collector.save_reference("/tmp/should_not_write.json")

        # N5: ownership persists across disarm() until finalize/clear
        col_n5 = tpe_logits_obs.TpeLogitsCollector()
        col_n5.arm(capture_steps=[0], owner_job_id="baseline_1", owner_generation=1)
        col_n5.record_step(rid="r1", output_step=0, token_id=7, logits_row=[1.0, 2.0])
        col_n5.disarm()
        with self.assertRaises(RuntimeError):
            col_n5.arm(capture_steps=[0], owner_job_id="baseline_2", owner_generation=1)
        self.assertEqual(col_n5.summary()["num_positions"], 1)
        col_n5.clear()
        col_n5.arm(capture_steps=[0], owner_job_id="baseline_2", owner_generation=1)

        # N3: sparse capture [0, 2] catches divergence at UNCAPTURED step 1 and prompt digest mismatch
        with tempfile.TemporaryDirectory() as tmpdir:
            sparse_ref = os.path.join(tmpdir, "sparse_ref.json")
            collector.arm(capture_steps=[0, 2])
            for step, tok, row in (
                (0, 1, [1.0, 2.0]),
                (1, 5, [3.0, 4.0]),
                (2, 3, [5.0, 6.0]),
            ):
                collector.record_step(
                    rid="r1",
                    output_step=step,
                    token_id=tok,
                    logits_row=row,
                    input_digest="digest_prompt_A",
                )
            collector.save_reference(sparse_ref)

            # Candidate diverges at uncaptured step 1 (tok=6 vs 5), even with identical logits at [0, 2]
            collector.arm(capture_steps=[0, 2])
            for step, tok, row in (
                (0, 1, [1.0, 2.0]),
                (1, 6, [3.0, 4.0]),
                (2, 3, [5.0, 6.0]),
            ):
                collector.record_step(
                    rid="r1",
                    output_step=step,
                    token_id=tok,
                    logits_row=row,
                    input_digest="digest_prompt_A",
                )
            rep_sparse = collector.compare_against_reference(sparse_ref)
            self.assertFalse(rep_sparse["passed"])
            self.assertEqual(rep_sparse["trajectory_divergences"], 1)

            # Candidate with mismatched input prompt digest -> FAIL
            collector.arm(capture_steps=[0, 2])
            for step, tok, row in (
                (0, 1, [1.0, 2.0]),
                (1, 5, [3.0, 4.0]),
                (2, 3, [5.0, 6.0]),
            ):
                collector.record_step(
                    rid="r1",
                    output_step=step,
                    token_id=tok,
                    logits_row=row,
                    input_digest="digest_prompt_B",
                )
            rep_prompt = collector.compare_against_reference(sparse_ref)
            self.assertFalse(rep_prompt["passed"])
            self.assertEqual(
                rep_prompt["failures"][0]["reason"], "input_prompt_digest_mismatch"
            )

            # N1, N4, N5 via real extracted Scheduler.handle_tpe_transaction()
            extra_globals = {
                "TpeTransactionReqInput": _MockReq,
                "TpeTransactionReqOutput": _MockResp,
                "current_platform": types.SimpleNamespace(synchronize=lambda: None),
                **_make_resolved_runtime_globals(),
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

            # N1: canary snapshot failure after candidate activation marks TAINTED + CLOSED
            ctrl_n1 = tpe_ctrl.TpeSchedulerController(instance_generation=1)
            sent_n1: List[_MockResp] = []
            hr_mgr = types.SimpleNamespace(is_candidate_active=False)

            def ok_hr(**kw):
                hr_mgr.is_candidate_active = True
                return (
                    types.SimpleNamespace(artifact_hash="h1"),
                    types.SimpleNamespace(cleanup_time_s=0.1, capture_time_s=0.1),
                )

            sched_n1 = types.SimpleNamespace(
                enable_overlap=False,
                tp_worker=types.SimpleNamespace(
                    model_runner=types.SimpleNamespace(
                        canary_manager=None,
                        tpe_hot_reload_manager=hr_mgr,
                        tpe_logits_collector=collector,
                    ),
                    tpe_hot_reload=ok_hr,
                ),
                _tpe_pending_drain_req=None,
                is_fully_idle=lambda: True,
                _get_tpe_controller=lambda: ctrl_n1,
                _tpe_sync_streams=lambda: None,
                _tpe_finalize_canary=lambda: {"violations_detected": False},
                _tpe_clear_keep_alive=lambda: None,
                _tpe_snapshot_canary_counts=lambda: (_ for _ in ()).throw(
                    RuntimeError("gpu counter copy failed")
                ),
                send_control_output=lambda out, req: sent_n1.append(out),
            )
            sched_n1._poll_tpe_drain = lambda: poll_fn(sched_n1)
            sched_n1._validate_tpe_phase1a_runtime = lambda: validate_fn(sched_n1)

            os.environ["SGLANG_ENABLE_TPE"] = "1"
            try:
                handle_fn(
                    sched_n1,
                    _MockReq("drain", "j1", "d1", 1, drain_purpose="hot_reload"),
                )
                handle_fn(
                    sched_n1,
                    _MockReq("hot_reload", "j1", "hr1", 1, code_utf8="def forward(*a): pass"),
                )
                self.assertFalse(sent_n1[-1].success)
                self.assertEqual(ctrl_n1.state, tpe_state.TpeInstanceState.TAINTED)
                self.assertEqual(ctrl_n1.admission_mode, tpe_state.TpeAdmissionMode.CLOSED)

                # N4 & N5: same save/ref path rejected without overwriting ref; wrong job rejected
                ctrl_n45 = tpe_ctrl.TpeSchedulerController(instance_generation=1)
                ctrl_n45.mark_candidate_ready("jA", "hA")
                col2 = tpe_logits_obs.TpeLogitsCollector()
                sent_n45: List[_MockResp] = []
                sched_n45 = types.SimpleNamespace(
                    enable_overlap=False,
                    tp_worker=types.SimpleNamespace(
                        model_runner=types.SimpleNamespace(tpe_logits_collector=col2),
                    ),
                    _tpe_pending_drain_req=None,
                    is_fully_idle=lambda: True,
                    _get_tpe_controller=lambda: ctrl_n45,
                    send_control_output=lambda out, req: sent_n45.append(out),
                )
                sched_n45._validate_tpe_phase1a_runtime = lambda: validate_fn(sched_n45)

                handle_fn(
                    sched_n45,
                    _MockReq(
                        "configure_logits_capture",
                        "jA",
                        "c1",
                        1,
                        capture_logits_enabled=True,
                        capture_steps=[0],
                    ),
                )
                self.assertTrue(sent_n45[-1].success)
                col2.record_step(rid="r1", output_step=0, token_id=1, logits_row=[100.0, 200.0])

                # N5: wrong job jB cannot re-arm or finalize jA's capture
                handle_fn(
                    sched_n45,
                    _MockReq(
                        "configure_logits_capture",
                        "jB",
                        "c2",
                        1,
                        capture_logits_enabled=True,
                    ),
                )
                self.assertFalse(sent_n45[-1].success)
                self.assertEqual(col2.summary()["num_positions"], 1)

                handle_fn(
                    sched_n45,
                    _MockReq(
                        "finalize_logits_capture",
                        "jB",
                        "f1",
                        1,
                        logits_ref_path=sparse_ref,
                    ),
                )
                self.assertFalse(sent_n45[-1].success)
                self.assertEqual(col2.summary()["num_positions"], 1)

                # N4: specifying both save_path and ref_path fails and preserves original ref file
                with open(sparse_ref, "r", encoding="utf-8") as f:
                    orig_ref_text = f.read()
                handle_fn(
                    sched_n45,
                    _MockReq(
                        "finalize_logits_capture",
                        "jA",
                        "f2",
                        1,
                        logits_save_path=sparse_ref,
                        logits_ref_path=sparse_ref,
                    ),
                )
                self.assertFalse(sent_n45[-1].success)
                with open(sparse_ref, "r", encoding="utf-8") as f:
                    self.assertEqual(f.read(), orig_ref_text)
            finally:
                os.environ.pop("SGLANG_ENABLE_TPE", None)

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
            "python/sglang/srt/tpe/logits_observer.py",
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
