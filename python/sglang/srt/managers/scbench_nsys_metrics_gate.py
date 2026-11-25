"""Collect one real full Decode GPU dispatches in one Nsight window."""
import json
import os
import time

import torch

path = os.environ.get("SCBENCH_METRICS_LOG")
streak = 0
phase = "unprimed"
taken = 0


def log(label, batch, **fields):
    with open(path, "a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps({
            "event": label, "at_ns": time.monotonic_ns(), "forward": batch.forward_iter,
            "launch": batch.launch_ts, "requests": len(batch.reqs), **fields,
        })+"\n")


def begin():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def end_capture():
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def before(batch):
    global phase,streak,taken
    if not path or phase == "done":
        return False
    decode = batch.forward_mode.is_decode()
    full = decode and len(batch.reqs) == 160 and all(not request.finished() and not request.is_retracted for request in batch.reqs)
    if phase == "active" and (taken == 1 or not full):
        torch.cuda.nvtx.mark("AgentLoop nsys 200k capture end")
        end_capture()
        log("capture_stop", batch, profiled_steps=taken, reason="one_step" if taken==1 else "batch_shrank")
        phase = "done"
        return False
    if not decode:
        streak = 0
        return False
    if phase == "unprimed":
        phase = "priming"
        begin()
        torch.cuda.nvtx.mark("AgentLoop nsys 200k priming")
        log("priming_start", batch)
        return True
    streak = streak+1 if full else 0
    if phase == "armed" and streak >= 8:
        begin()
        phase = "active"
        log("capture_start", batch)
    if phase != "active":
        return False
    assert full
    taken += 1
    torch.cuda.nvtx.mark(f"AgentLoop nsys 200k B160 step={batch.forward_iter}")
    log("profiled_step", batch, number=taken)
    return True


def after(batch):
    global phase
    if phase == "priming":
        end_capture()
        phase = "armed"
        log("priming_stop", batch)
    elif phase == "active":
        log("step_enqueued", batch, number=taken)
