"""Compare DSpark dense and sparse rejection with identical random draws."""

import os
import statistics
from types import SimpleNamespace
import unittest

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
from sglang.kernels.ops.speculative.dspark.sparse_topk_rejection import sparse_topk_rejection
from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton
from sglang.srt.speculative.dflash_utils import (
    _dflash_topk_support,
    build_speculative_verify_target_probs,
)


def dense_rejection(candidates, target_probs, draft_probs, coins, final_coins, cutoff):
    bs, slots = candidates.shape
    device = candidates.device
    predicts = torch.empty(bs * slots, dtype=torch.int32, device=device)
    indices = torch.arange(bs * slots, dtype=torch.int64, device=device).view(bs, slots)
    accept_indices = torch.empty_like(indices, dtype=torch.int32)
    accepted = torch.empty(bs, dtype=torch.int32, device=device)
    chain_speculative_sampling_triton(
        predicts, accept_indices, accepted, candidates, indices, indices, indices,
        coins, final_coins, target_probs, draft_probs, 1.0, 1.0, True,
    )
    kept = accepted if cutoff is None else torch.minimum(accepted, cutoff - 1)
    rows = torch.arange(bs, device=device)
    bonus = predicts[accept_indices[rows, kept.long()]].long()
    return kept, bonus, accepted - kept


def prepare(bs, vocab, gamma, k, greedy, varying):
    device = "cuda"
    torch.manual_seed(bs + vocab + gamma + k + greedy + varying)
    target = torch.randn(bs, gamma + 1, vocab, device=device)
    if varying:
        temps = torch.linspace(0.6, 1.4, bs, device=device)
        top_ps = torch.linspace(0.8, 1.0, bs, device=device)
        top_ks = torch.arange(bs, device=device, dtype=torch.int32) % k + 1
    else:
        temps = torch.ones(bs, device=device)
        top_ps = torch.full((bs,), 0.95, device=device)
        top_ks = torch.full((bs,), k, dtype=torch.int32, device=device)
    # High-mass candidates exercise accept and cap paths; some proposals are outside top-k.
    slots = gamma + 1
    target[:, :, :k] += 1.5
    candidates = target.argmax(dim=-1).roll(1, dims=1)
    candidates[::3, 1:] = torch.randint(vocab, (len(candidates[::3]), gamma), device=device)
    corrected = target[:, :gamma].to(torch.bfloat16).contiguous()
    sampling = SimpleNamespace(
        temperatures=temps[:, None], top_ps=top_ps, top_ks=top_ks,
        need_top_p_sampling=True, need_top_k_sampling=True,
    )
    return target.reshape(bs * slots, vocab), candidates, corrected, sampling


def build_paths(bs, vocab, gamma, k, greedy, varying, cap):
    target, candidates, corrected, sampling = prepare(bs, vocab, gamma, k, greedy, varying)
    slots = gamma + 1
    uniform_k = None if varying else k
    adjusted_temp = sampling.temperatures.repeat_interleave(slots, dim=0)
    coins = torch.rand(bs, gamma, device="cuda")
    final_coins = torch.rand(bs, device="cuda")
    cutoff = torch.randint(1, slots + 1, (bs,), device="cuda", dtype=torch.int32) if cap else None

    def dense():
        p = build_speculative_verify_target_probs(
            next_token_logits=target, sampling_info=sampling, draft_token_num=slots,
            bs=bs, max_top_k=k, uniform_top_k_value=uniform_k,
        )
        if greedy:
            q = torch.zeros((bs, gamma, vocab), device="cuda", dtype=torch.float32)
            q.scatter_(-1, candidates[:, 1:].unsqueeze(-1), 1.0)
        else:
            q = SoftmaxTemp.execute(
                logits=corrected.view(-1, vocab), temperatures=sampling.temperatures,
                rows_per_request=gamma,
            ).view(bs, gamma, vocab)
        return dense_rejection(candidates, p, q, coins, final_coins, cutoff)

    def sparse():
        indices, probs = _dflash_topk_support(
            target / adjusted_temp if varying else target, sampling, slots, k, uniform_k
        )
        return sparse_topk_rejection(
            candidates=candidates, top_indices=indices, top_probs=probs,
            draft_logits=None if greedy else corrected,
            draft_temperatures=sampling.temperatures, vocab=vocab,
            cutoff_verify_lens=cutoff, coins=coins, final_coins=final_coins,
        )

    return dense, sparse


class SparseRejectionTest(unittest.TestCase):
    def test_identical_draws(self):
        cases = [
            (160, 131, 3, 1, False, False, False),
            (160, 257, 3, 20, True, False, True),
            (256, 4097, 5, 64, False, True, True),
            (64, 8192 + 57, 3, 20, False, True, False),
            (160, 248320, 3, 20, False, False, False),
        ]
        for case in cases:
            with self.subTest(case=case):
                dense, sparse = build_paths(*case)
                expected = dense()
                actual = sparse()
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_cuda_graph(self):
        dense, sparse = build_paths(160, 9217, 3, 20, False, True, True)
        expected = dense()
        sparse()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse()
        for _ in range(3):
            graph.replay()
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

    @unittest.skipUnless(os.getenv("DSPARK_BENCH_SPARSE_TOPK") == "1", "timing is opt-in")
    def test_b160_latency(self):
        dense, sparse = build_paths(160, 248320, 3, 20, False, False, False)
        graphs = []
        for path in (dense, sparse):
            path()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = path()
            graphs.append((graph, out))
        samples = [[], []]
        for round_index in range(48):
            for arm in (round_index % 2, 1 - round_index % 2):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[arm][0].replay()
                stop.record()
                stop.synchronize()
                samples[arm].append(start.elapsed_time(stop))
        medians = [statistics.median(samples_arm) for samples_arm in samples]
        print(f"B160/G3/K20 den/sparse median milliseconds: {medians}", flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
