"""Classic DSpark rejection sampling over the support of the target top-k."""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


try:
    from flashinfer import __version__ as _flashinfer_version
    from flashinfer.topk import TopKTieBreak, top_k as _flashinfer_top_k
except ImportError:
    _flashinfer_version = None
    _flashinfer_top_k = None


def can_select_b300_bf16_topk(logits: torch.Tensor, k: int) -> bool:
    return (
        envs.SGLANG_OPT_DSPARK_B300_BF16_TARGET_TOPK.get()
        and envs.SGLANG_OPT_DSPARK_B300_FLASHINFER_TOPK.get()
        and envs.SGLANG_OPT_DSPARK_SPARSE_TOPK_REJECTION.get()
        and _flashinfer_version == "0.7.3"
        and k == 20
        and logits.dtype == torch.bfloat16
        and logits.shape == (640, 248320)
        and logits.is_contiguous()
        and logits.is_cuda
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(logits.device) == (10, 3)
    )


def select_b300_topk(
    logits: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the same support as the pinned CUDA multiblock Top-K."""
    if (
        _flashinfer_top_k is None
        or (
            logits.dtype != torch.float32
            and not can_select_b300_bf16_topk(logits, k)
        )
        or logits.shape != (640, 248320)
        or not logits.is_contiguous()
        or not logits.is_cuda
        or torch.version.cuda is None
        or not 1 < k <= 32
        or torch.cuda.get_device_capability(logits.device) != (10, 3)
    ):
        return torch.topk(logits, k)

    # Torch 2.14's CUDA multiblock gather selects lower token IDs when a
    # tie crosses the K boundary. Order within the selected equal logits
    # can differ. Top-p thresholds by value and the final CDF sorts by ID.
    values, indices = _flashinfer_top_k(
        logits, k, sorted=True, deterministic=True, tie_break=TopKTieBreak.SMALL
    )
    # Widen only the selected values: temperature and top-p always use FP32.
    return values.float(), indices


def can_fuse_b300_top_p(logits: torch.Tensor, k: int) -> bool:
    return (
        _flashinfer_version == "0.7.3"
        and k == 20
        and (logits.dtype == torch.float32 or can_select_b300_bf16_topk(logits, k))
        and logits.shape == (640, 248320)
        and logits.is_cuda
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(logits.device) == (10, 3)
    )


_DRAFT_TILE = 8192


@triton.jit
def _sparse_draft_stats_kernel(
    logits, temperatures, maxima, sums, vocab, row_stride, step_stride,
    GAMMA: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    request, step = row // GAMMA, row % GAMMA
    token = tile * TILE + tl.arange(0, TILE)
    temp = tl.load(temperatures + request).to(tl.float32)
    ptr = logits + request.to(tl.int64) * row_stride + step * step_stride
    x = tl.load(ptr + token, token < vocab, other=-float("inf")).to(tl.float32)
    x = x / temp
    maximum = tl.max(x, axis=0)
    scaled_sum = tl.sum(tl.exp(x - tl.where(maximum == -float("inf"), 0.0, maximum)))
    tl.store(maxima + row * TILES + tile, maximum)
    tl.store(sums + row * TILES + tile, scaled_sum)


@triton.jit
def _draft_prob_at(
    tokens, valid, logits, maxima, sums, temp, row,
    TILES: tl.constexpr, PADDED_TILES: tl.constexpr,
):
    tile = tl.arange(0, PADDED_TILES)
    local_max = tl.load(maxima + row * TILES + tile, tile < TILES, other=-float("inf"))
    local_sum = tl.load(sums + row * TILES + tile, tile < TILES, other=0.0)
    maximum = tl.max(local_max, axis=0)
    denominator = tl.sum(local_sum * tl.exp(local_max - maximum))
    x = tl.load(logits + tokens, valid, other=-float("inf")).to(tl.float32) / temp
    return tl.exp(x - maximum) / denominator


@triton.jit
def _sum_width20_vec4(p):
    # FlashInfer's FP32 width-20 path uses five vector-4 loads in the
    # first warp. CUB combines those sums in shuffle offsets 1, 2, 4.
    evens, odds = tl.split(tl.reshape(p, (16, 2)))
    v0, v2 = tl.split(tl.reshape(evens, (8, 2)))
    v1, v3 = tl.split(tl.reshape(odds, (8, 2)))
    groups = ((v0 + v1) + v2) + v3
    g0 = tl.sum(tl.gather(groups, tl.full((1,), 0, tl.int32), 0), 0)
    g1 = tl.sum(tl.gather(groups, tl.full((1,), 1, tl.int32), 0), 0)
    g2 = tl.sum(tl.gather(groups, tl.full((1,), 2, tl.int32), 0), 0)
    g3 = tl.sum(tl.gather(groups, tl.full((1,), 3, tl.int32), 0), 0)
    g4 = tl.sum(tl.gather(groups, tl.full((1,), 4, tl.int32), 0), 0)
    return ((g0 + g1) + (g2 + g3)) + g4


@triton.jit
def _flashinfer_073_width20_top_p(p, cutoff):
    # Preserve the native pivot search, including equal probabilities
    # at the boundary and exact FP32 comparisons against top-p.
    retained = p
    if cutoff < 1.0:
        low = 0.0
        high = tl.max(tl.where(p > 0.0, p, 0.0), 0)
        s = 1.0
        searching = True
        while searching:
            t0 = tl.div_rn(high + 2.0 * low, 3.0)
            t1 = tl.div_rn(2.0 * high + low, 3.0)
            after_low = tl.min(tl.where(p > low, p, high), 0)
            before_high = tl.max(tl.where(p <= high, p, low), 0)
            m0 = _sum_width20_vec4(tl.where(p > t0, p, 0.0))
            m1 = _sum_width20_vec4(tl.where(p > t1, p, 0.0))
            if m1 >= cutoff:
                low = t1
                s = m1
            elif m0 >= cutoff:
                low = t0
                high = tl.minimum(t1, before_high)
                s = m0
            else:
                high = tl.minimum(t0, before_high)
            next_after = (after_low.to(tl.int32, bitcast=True) + 1).to(tl.float32, bitcast=True)
            searching = (after_low < before_high) & (next_after < before_high)
        retained = tl.where(p > low, p, 0.0)
        s = tl.maximum(s, 1e-8)
    else:
        s = _sum_width20_vec4(p)
        s = tl.where(s <= 1e-8, 1.0, s)

    rcp = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f", args=[s], dtype=tl.float32, is_pure=True, pack=1,
    )
    return retained * rcp


@triton.jit
def _sparse_topk_rejection_kernel(
    candidates, top_indices, top_probs, target_top_ps, draft_logits, draft_temp, maxima, sums,
    coins, final_coins, cutoff, correct_len, bonus, trim, vocab,
    candidate_stride, draft_row_stride, draft_step_stride, cutoff_stride,
    GAMMA: tl.constexpr, TOP_K: tl.constexpr, TOP_K_PAD: tl.constexpr,
    TILES: tl.constexpr, PADDED_TILES: tl.constexpr,
    HAS_DRAFT: tl.constexpr, HAS_CUTOFF: tl.constexpr, FUSE_TOP_P: tl.constexpr,
):
    req = tl.program_id(0)
    candidate_ptr = candidates + req * candidate_stride
    target_base = req * (GAMMA + 1) * TOP_K
    lane = tl.arange(0, TOP_K_PAD)
    active = lane < TOP_K
    cutoff_prob = 1.0
    if FUSE_TOP_P:
        cutoff_prob = tl.load(target_top_ps + req)

    temp = 1.0
    if HAS_DRAFT:
        temp = tl.load(draft_temp + req).to(tl.float32)

    accepted = 0
    for step in range(GAMMA):
        if accepted == step:
            token = tl.load(candidate_ptr + step + 1)
            offset = target_base + step * TOP_K + lane
            ids = tl.load(top_indices + offset, active, other=-1)
            weights = tl.load(top_probs + offset, active, other=0.0)
            if FUSE_TOP_P:
                weights = _flashinfer_073_width20_top_p(weights, cutoff_prob)
                weights = tl.where(active, weights, 0.0)
            p = tl.sum(tl.where(ids == token, weights, 0.0))
            q = 1.0
            if HAS_DRAFT:
                ptr = draft_logits + req.to(tl.int64) * draft_row_stride + step * draft_step_stride
                q = _draft_prob_at(
                    token, (token >= 0) & (token < vocab), ptr, maxima, sums,
                    temp, req * GAMMA + step, TILES, PADDED_TILES,
                )
            coin = tl.load(coins + req * GAMMA + step)
            accepted += ((q > 0.0) & (q <= 1.0) & (coin * q < p)).to(tl.int32)

    capped = accepted
    if HAS_CUTOFF:
        capped = tl.minimum(accepted, tl.load(cutoff + req * cutoff_stride).to(tl.int32) - 1)
    if capped < accepted:
        # The dense sampler's accept-index lookup exposes the next accepted
        # proposal as the bonus when the verified prefix is cut short.
        selected = tl.load(candidate_ptr + capped + 1).to(tl.int32)
    else:
        offset = target_base + accepted * TOP_K + lane
        ids = tl.load(top_indices + offset, active, other=vocab).to(tl.int32)
        # Match the dense CDF's token-ID order when using the same draw.
        sorted_keys = tl.sort(ids * TOP_K_PAD + lane, descending=False)
        ids = sorted_keys // TOP_K_PAD
        original_lane = sorted_keys % TOP_K_PAD
        if FUSE_TOP_P:
            # Top-p reduces in top-k rank order. Only reorder afterward to
            # preserve the token-ID CDF's behavior with identical draws.
            p = tl.load(top_probs + offset, active, other=0.0)
            p = _flashinfer_073_width20_top_p(p, cutoff_prob)
            p = tl.gather(p, original_lane, axis=0)
            p = tl.where(original_lane < TOP_K, p, 0.0)
        else:
            p = tl.load(top_probs + target_base + accepted * TOP_K + original_lane,
                        original_lane < TOP_K, other=0.0)
        weights = p
        if accepted < GAMMA:
            if HAS_DRAFT:
                ptr = draft_logits + req.to(tl.int64) * draft_row_stride + accepted * draft_step_stride
                q = _draft_prob_at(
                    ids, ids < vocab, ptr, maxima, sums, temp,
                    req * GAMMA + accepted, TILES, PADDED_TILES,
                )
            else:
                q = (ids == tl.load(candidate_ptr + accepted + 1)).to(tl.float32)
            q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
            diff = p - q
            weights = tl.where(diff > 0.0, diff, 0.0)
        threshold = tl.load(final_coins + req) * tl.sum(weights, axis=0)
        hit = tl.cumsum(weights, axis=0) > threshold
        first = tl.min(tl.where(hit, lane, TOP_K_PAD), axis=0)
        selected = tl.sum(tl.where(lane == first, ids, 0), axis=0)
        # The classic kernel leaves this sentinel for zero residual mass.
        selected = tl.where(first < TOP_K_PAD, selected, vocab - 1)

    tl.store(correct_len + req, capped)
    tl.store(bonus + req, selected)
    tl.store(trim + req, accepted - capped)


def sparse_topk_rejection(
    *,
    candidates: torch.Tensor,
    top_indices: torch.Tensor,
    top_probs: torch.Tensor,
    draft_logits: Optional[torch.Tensor],
    draft_temperatures: torch.Tensor,
    vocab: int,
    cutoff_verify_lens: Optional[torch.Tensor] = None,
    coins: Optional[torch.Tensor] = None,
    final_coins: Optional[torch.Tensor] = None,
    target_top_ps: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify a linear chain without materializing dense target or draft probabilities."""
    bs, slots = candidates.shape
    gamma = slots - 1
    top_k = top_indices.shape[-1]
    device = candidates.device
    if target_top_ps is not None and (top_k != 20 or target_top_ps.numel() != bs):
        raise ValueError("Fused top-p requires width 20 and one probability per request")
    if coins is None:
        coins = torch.rand((bs, gamma), dtype=torch.float32, device=device)
    if final_coins is None:
        final_coins = torch.rand((bs,), dtype=torch.float32, device=device)

    lens = torch.empty((2, bs), dtype=torch.int32, device=device)
    bonus = torch.empty(bs, dtype=torch.int64, device=device)
    temperatures = draft_temperatures.reshape(-1).contiguous()
    has_draft = draft_logits is not None
    n_tiles = triton.cdiv(vocab, _DRAFT_TILE) if has_draft else 1
    if has_draft:
        stats = torch.empty((2, bs * gamma, n_tiles), dtype=torch.float32, device=device)
        maxima, sums = stats[0], stats[1]
        row_stride, step_stride, _ = draft_logits.stride()
        _sparse_draft_stats_kernel[(bs * gamma, n_tiles)](
            draft_logits, temperatures, maxima, sums, vocab, row_stride, step_stride,
            GAMMA=gamma, TILES=n_tiles, TILE=_DRAFT_TILE,
        )
    else:
        maxima = sums = top_probs
        row_stride = step_stride = 0

    _sparse_topk_rejection_kernel[(bs,)](
        candidates, top_indices.contiguous(), top_probs.contiguous(),
        target_top_ps.reshape(-1).contiguous() if target_top_ps is not None else top_probs,
        draft_logits if has_draft else top_probs, temperatures, maxima, sums,
        coins, final_coins,
        cutoff_verify_lens if cutoff_verify_lens is not None else lens,
        lens[0], bonus, lens[1], vocab, candidates.stride(0), row_stride, step_stride,
        cutoff_verify_lens.stride(0) if cutoff_verify_lens is not None else 0,
        GAMMA=gamma, TOP_K=top_k, TOP_K_PAD=triton.next_power_of_2(top_k),
        TILES=n_tiles, PADDED_TILES=triton.next_power_of_2(n_tiles),
        HAS_DRAFT=has_draft, HAS_CUTOFF=cutoff_verify_lens is not None,
        FUSE_TOP_P=target_top_ps is not None,
        num_warps=1, enable_fp_fusion=False,
    )
    return lens[0], bonus, lens[1]
