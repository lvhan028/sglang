"""Check FP32 long-row Top-K support against Torch on B300."""

from types import SimpleNamespace
import unittest

import torch

from sglang.kernels.ops.speculative.dspark.sparse_topk_rejection import (
    select_b300_topk,
)
from sglang.srt.speculative.dflash_utils import _dflash_topk_support


class TestB300TopK(unittest.TestCase):
    def test_support_and_top_p_parity(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("requires B300")
        torch.manual_seed(91382)
        logits = torch.randn((640, 248320), device="cuda")
        sampling = SimpleNamespace(
            need_top_p_sampling=True, top_ps=torch.full((160,), 0.95, device="cuda")
        )

        def sample_support(select):
            return _dflash_topk_support(
                logits, sampling, 4, 20, 20, select_topk=select
            )

        def canonical(result):
            ids, probs = result
            sorted_ids, order = ids.sort(dim=-1)
            return sorted_ids, probs.gather(-1, order)

        sample_support(select_b300_topk)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sample_support(select_b300_topk)
        for case in ("random", "rounded", "plateau", "masked_tail"):
            with self.subTest(case=case):
                if case == "rounded":
                    logits.copy_(logits.to(torch.bfloat16))
                elif case == "plateau":
                    logits.fill_(-80.0)
                    logits[:, :10] = 4.0
                    logits[:, 513:650] = 3.0
                    logits[:, 9000:9300] = 3.0
                elif case == "masked_tail":
                    logits.fill_(-float("inf"))
                    logits[:, 31:44] = 1.0
                    logits[:, 5:12] = 2.0
                    logits[:, 513:700] = 0.0
                graph.replay()
                want_ids, want_probs = canonical(sample_support(torch.topk))
                got_ids, got_probs = canonical(actual)
                torch.testing.assert_close(got_ids, want_ids, rtol=0, atol=0)
                torch.testing.assert_close(got_probs, want_probs, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
