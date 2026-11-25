"""Check layered commits against the original API, including nulls, tracks and 256 folds.

The full 30-layer shape exercises 64-bit offsets beyond 2 Gi elements. Metadata
and untouched slots are exact; changed FP32 state uses the existing fold test's
2-epsilon allowance for these deterministic inputs, not a general error bound.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import torch
from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as candidate
from sglang.test.test_utils import CustomTestCase

REFERENCE=Path(os.environ['SCBENCH_GDN_REFERENCE_SOURCE'])
spec=importlib.util.spec_from_file_location('original_scbench_gdn',REFERENCE)
reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
ATOL=2*torch.finfo(torch.float32).eps


class TestLayeredCommit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        assert os.geteuid()==0 and torch.cuda.device_count()==1
        torch.manual_seed(9628)
        def normal(shape,dtype,scale):return torch.empty(shape,device='cuda',dtype=dtype).normal_(0,scale)
        cls.initial=normal((30,257,32,128,128),torch.float32,.005)
        cls.state=cls.initial.clone()
        d=normal((30,161,32,16,128),torch.bfloat16,.01)
        k=normal((30,161,16,16,128),torch.bfloat16,.01)
        cls.args=dict(checkpoint_state=cls.state,d_cache=d,k_cache=k,
            g_cache=torch.full((30,161,32,16),-.002,device='cuda',dtype=torch.float32),
            d_residual_cache=normal(d.shape,d.dtype,.00003),k_residual_cache=normal(k.shape,k.dtype,.00003),
            state_batch_indices=torch.arange(1,161,device='cuda',dtype=torch.int64),
            replay_indices=torch.arange(1,161,device='cuda',dtype=torch.int32),
            write_pos=torch.empty(161,device='cuda',dtype=torch.int32),cache_base=torch.empty(161,device='cuda',dtype=torch.int32),
            is_flush=torch.empty(161,device='cuda',dtype=torch.int8),
            accept_lens=3+torch.arange(160,device='cuda',dtype=torch.int32)%2,null_block_id=-1)
        cls.results=[]

    @classmethod
    def tearDownClass(cls):
        cls.args=None;cls.state=None;cls.initial=None
        torch.cuda.empty_cache()

    def reset_metadata(self,folds,padded=None,track=False,step=0):
        args=self.args
        slots=torch.arange(1,161,device='cuda',dtype=torch.int64)
        if padded is not None:slots[:10]=padded
        args['state_batch_indices'].copy_(slots)
        args['null_block_id']=0 if padded==0 else -1
        selected=(torch.arange(folds,device='cuda')*73+step*13)%160+1
        flags=torch.zeros_like(args['is_flush']);flags[selected]=1;args['is_flush'].copy_(flags)
        pos=torch.arange(161,device='cuda',dtype=torch.int32)%8+3;pos[selected]=13+(selected%4).to(torch.int32)
        args['write_pos'].copy_(pos)
        args['cache_base'].copy_((torch.arange(161,device='cuda',dtype=torch.int32)*5+step)%16)
        if track:
            indices=torch.full((160,),-1,device='cuda',dtype=torch.int64);indices[10:13]=torch.arange(161,164,device='cuda')
            args['mamba_track_indices']=indices;args['mamba_steps_to_track']=args['accept_lens']-1
        else:
            args.pop('mamba_track_indices',None);args.pop('mamba_steps_to_track',None)
        return selected,slots

    def test_state_metadata_and_untouched_slots(self):
        for folds,padded,track in [(0,None,False),(44,None,False),(160,None,False),(44,-1,False),(44,0,False),(44,None,True)]:
            with self.subTest(folds=folds,padded=padded,track=track):
                self.state.copy_(self.initial);selected,slots=self.reset_metadata(folds,padded,track)
                reference.commit_gdn_replayssm_circular(**self.args)
                expected=self.state.clone();metadata={k:self.args[k].clone() for k in ['write_pos','cache_base','is_flush']}
                self.state.copy_(self.initial);self.reset_metadata(folds,padded,track)
                candidate.commit_gdn_replayssm_circular(**self.args)
                torch.testing.assert_close(self.state,expected,atol=ATOL,rtol=0)
                for k,values in metadata.items():self.assertTrue(torch.equal(self.args[k],values),k)
                touched=set(int(slots[int(i)-1]) for i in selected.tolist() if int(slots[int(i)-1])>self.args['null_block_id'])
                if track:touched.update([161,162,163])
                untouched=torch.tensor([i for i in range(257) if i not in touched],device='cuda')
                self.assertTrue(torch.equal(self.state.index_select(1,untouched),self.initial.index_select(1,untouched)))
                if track:self.assertTrue(torch.equal(self.state,expected))
                error=float((self.state-expected).abs().max())
                self.results.append(dict(folds=folds,padded=padded,track=track,max_abs_error=error,metadata_exact=True,untouched_exact=True))
                del expected

    def test_256_chained_commits(self):
        finals=[]
        for module in [reference,candidate]:
            self.state.copy_(self.initial)
            for step in range(256):
                self.reset_metadata(44,step=step)
                module.commit_gdn_replayssm_circular(**self.args)
            finals.append(self.state.clone())
        torch.testing.assert_close(finals[0],finals[1],atol=ATOL,rtol=0)
        error=float((finals[0]-finals[1]).abs().max())
        self.results.append(dict(chained_commits=256,max_abs_error=error,atol=ATOL))

    @classmethod
    def doClassCleanups(cls):
        path=os.environ.get('SCBENCH_GDN_VALIDATION_OUTPUT')
        if path and hasattr(cls,'results'):Path(path).write_text(json.dumps(cls.results,indent=2)+'\n')
        return super().doClassCleanups()


if __name__=='__main__':
    unittest.main(verbosity=2)
