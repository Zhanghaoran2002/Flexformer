"""Fused causal recurrence for rectangular linear-attention states.

Q/K feature maps are computed outside these kernels, so PyTorch autograd
continues to differentiate the learned Fourier frequencies and temperature.
Inspired by the implementation approach of Flash Linear Attention (FLA):
https://github.com/fla-org/flash-linear-attention

This is a recurrent scan, not FLA's chunk-parallel algorithm. Each program
handles one (batch, head); time is sequential inside the GPU program. Forward
retains only the final state. Backward recomputes the state history and uses
O(B*H*T*M*Dv) temporary storage, released after that layer's backward pass.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def triton_supported(q, v):
    if triton is None or not q.is_cuda:
        return False
    m, dv = q.shape[-1], v.shape[-1]
    return (
        q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and m <= 256
        and dv <= 128
        and triton.next_power_of_2(m) * triton.next_power_of_2(dv) <= 32768
    )


if triton is not None:

    @triton.jit
    def _scan_forward(
        Q, K, V, MASK, S0, Z0, OUT, SF, ZF, HISTORY_S, HISTORY_Z,
        LENGTH, HEADS: tl.constexpr, M: tl.constexpr, DV: tl.constexpr,
        EPS: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr,
        HAS_MASK: tl.constexpr, WRITE_OUTPUT: tl.constexpr,
        WRITE_HISTORY: tl.constexpr, WRITE_FINAL: tl.constexpr,
    ):
        bh = tl.program_id(0)
        rm, rv = tl.arange(0, BM), tl.arange(0, BV)
        state_offsets = (bh * M + rm[:, None]) * DV + rv[None, :]
        state_mask = (rm[:, None] < M) & (rv[None, :] < DV)
        s = tl.load(S0 + state_offsets, mask=state_mask, other=0).to(tl.float32)
        z = tl.load(Z0 + bh * M + rm, mask=rm < M, other=0).to(tl.float32)

        for t in range(0, LENGTH):
            valid = True
            if HAS_MASK:
                valid = tl.load(MASK + (bh // HEADS) * LENGTH + t)
            kt = tl.load(K + (bh * LENGTH + t) * M + rm,
                         mask=rm < M, other=0).to(tl.float32)
            vt = tl.load(V + (bh * LENGTH + t) * DV + rv,
                         mask=rv < DV, other=0).to(tl.float32)
            kt = tl.where(valid, kt, 0.0)
            vt = tl.where(valid, vt, 0.0)
            s += kt[:, None] * vt[None, :]
            z += kt
            if WRITE_OUTPUT:
                qt = tl.load(Q + (bh * LENGTH + t) * M + rm,
                             mask=rm < M, other=0).to(tl.float32)
                numerator = tl.sum(s * qt[:, None], axis=0)
                denominator = tl.sum(qt * z, axis=0) + EPS
                tl.store(OUT + (bh * LENGTH + t) * DV + rv,
                         numerator / denominator, mask=rv < DV)
            if WRITE_HISTORY:
                hist_s = ((bh * LENGTH + t) * M + rm[:, None]) * DV + rv[None, :]
                tl.store(HISTORY_S + hist_s, s, mask=state_mask)
                tl.store(HISTORY_Z + (bh * LENGTH + t) * M + rm, z, mask=rm < M)

        if WRITE_FINAL:
            tl.store(SF + state_offsets, s, mask=state_mask)
            tl.store(ZF + bh * M + rm, z, mask=rm < M)

    @triton.jit
    def _scan_backward(
        Q, K, V, MASK, HISTORY_S, HISTORY_Z, DOUT, DSF, DZF,
        DQ, DK, DV_OUT, DS0, DZ0,
        LENGTH, HEADS: tl.constexpr, M: tl.constexpr, DV: tl.constexpr,
        EPS: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr,
        HAS_MASK: tl.constexpr, HAS_DOUT: tl.constexpr,
        HAS_DSF: tl.constexpr, HAS_DZF: tl.constexpr,
    ):
        bh = tl.program_id(0)
        rm, rv = tl.arange(0, BM), tl.arange(0, BV)
        state_offsets = (bh * M + rm[:, None]) * DV + rv[None, :]
        state_mask = (rm[:, None] < M) & (rv[None, :] < DV)
        ds = tl.full((BM, BV), 0, tl.float32)
        dz = tl.full((BM,), 0, tl.float32)
        if HAS_DSF:
            ds = tl.load(DSF + state_offsets, mask=state_mask, other=0).to(tl.float32)
        if HAS_DZF:
            dz = tl.load(DZF + bh * M + rm, mask=rm < M, other=0).to(tl.float32)

        for t in range(LENGTH - 1, -1, -1):
            valid = True
            if HAS_MASK:
                valid = tl.load(MASK + (bh // HEADS) * LENGTH + t)
            qt = tl.load(Q + (bh * LENGTH + t) * M + rm,
                         mask=rm < M, other=0).to(tl.float32)
            kt = tl.load(K + (bh * LENGTH + t) * M + rm,
                         mask=rm < M, other=0).to(tl.float32)
            vt = tl.load(V + (bh * LENGTH + t) * DV + rv,
                         mask=rv < DV, other=0).to(tl.float32)
            hist_s = ((bh * LENGTH + t) * M + rm[:, None]) * DV + rv[None, :]
            s = tl.load(HISTORY_S + hist_s, mask=state_mask, other=0).to(tl.float32)
            z = tl.load(HISTORY_Z + (bh * LENGTH + t) * M + rm,
                        mask=rm < M, other=0).to(tl.float32)
            dout = tl.full((BV,), 0, tl.float32)
            if HAS_DOUT:
                dout = tl.load(DOUT + (bh * LENGTH + t) * DV + rv,
                               mask=rv < DV, other=0).to(tl.float32)

            numerator = tl.sum(s * qt[:, None], axis=0)
            denominator = tl.sum(qt * z, axis=0) + EPS
            dnum = dout / denominator
            dden = -tl.sum(dout * numerator, axis=0) / (denominator * denominator)
            dq = tl.sum(s * dnum[None, :], axis=1) + z * dden
            ds += qt[:, None] * dnum[None, :]
            dz += qt * dden
            dk = tl.sum(ds * vt[None, :], axis=1) + dz
            dv = tl.sum(ds * kt[:, None], axis=0)
            dk = tl.where(valid, dk, 0.0)
            dv = tl.where(valid, dv, 0.0)

            tl.store(DQ + (bh * LENGTH + t) * M + rm, dq, mask=rm < M)
            tl.store(DK + (bh * LENGTH + t) * M + rm, dk, mask=rm < M)
            tl.store(DV_OUT + (bh * LENGTH + t) * DV + rv, dv, mask=rv < DV)

        tl.store(DS0 + state_offsets, ds, mask=state_mask)
        tl.store(DZ0 + bh * M + rm, dz, mask=rm < M)


def _launch_options(q, v, eps, has_mask):
    bm = triton.next_power_of_2(q.shape[-1])
    bv = triton.next_power_of_2(v.shape[-1])
    return dict(
        HEADS=q.shape[1], M=q.shape[-1], DV=v.shape[-1], EPS=eps,
        BM=bm, BV=bv, HAS_MASK=has_mask,
        num_warps=4 if bm * bv <= 8192 else 8,
        enable_fp_fusion=False,
    )


class _CausalLinearScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, s0, z0, eps):
        q, k, v = (x.contiguous() for x in (q, k, v))
        s0, z0 = s0.contiguous(), z0.contiguous()
        mask = mask.contiguous()
        b, h, length, m = q.shape
        dv = v.shape[-1]
        out = torch.empty_like(v)
        sf, zf = torch.empty_like(s0), torch.empty_like(z0)
        dummy = torch.empty(0, device=q.device, dtype=torch.float32)
        options = _launch_options(q, v, eps, mask.numel() != 0)
        with torch.cuda.device(q.device):
            _scan_forward[(b * h,)](
                q, k, v, mask, s0, z0, out, sf, zf, dummy, dummy, length,
                WRITE_OUTPUT=True, WRITE_HISTORY=False, WRITE_FINAL=True, **options,
            )
        ctx.save_for_backward(q, k, v, mask, s0, z0)
        ctx.eps = eps
        ctx.set_materialize_grads(False)
        return out, sf, zf

    @staticmethod
    def backward(ctx, dout, dsf, dzf):
        q, k, v, mask, s0, z0 = ctx.saved_tensors
        b, h, length, m = q.shape
        dv = v.shape[-1]
        history_s = torch.empty(b, h, length, m, dv, device=q.device, dtype=torch.float32)
        history_z = torch.empty(b, h, length, m, device=q.device, dtype=torch.float32)
        dq, dk, dv_out = (torch.empty_like(x) for x in (q, k, v))
        ds0, dz0 = torch.empty_like(s0), torch.empty_like(z0)
        options = _launch_options(q, v, ctx.eps, mask.numel() != 0)
        with torch.cuda.device(q.device):
            # Recompute just the histories; do not recompute attention outputs.
            _scan_forward[(b * h,)](
                q, k, v, mask, s0, z0, v, s0, z0, history_s, history_z, length,
                WRITE_OUTPUT=False, WRITE_HISTORY=True, WRITE_FINAL=False, **options,
            )
            _scan_backward[(b * h,)](
                q, k, v, mask, history_s, history_z,
                dout.contiguous() if dout is not None else v,
                dsf.contiguous() if dsf is not None else s0,
                dzf.contiguous() if dzf is not None else z0,
                dq, dk, dv_out, ds0, dz0, length,
                HAS_DOUT=dout is not None, HAS_DSF=dsf is not None,
                HAS_DZF=dzf is not None, **options,
            )
        return dq, dk, dv_out, None, ds0, dz0, None


def triton_causal_linear_attention(
    q, k, v, valid_mask=None, initial_state=None, return_state=False, eps=1e-6
):
    """Low-level entry point; validation/dispatch live in causal_linear_attention."""
    if not triton_supported(q, v):
        raise RuntimeError("This configuration is not supported by the Triton scan.")
    b, h, _, m = q.shape
    if initial_state is None:
        s0 = torch.zeros(b, h, m, v.shape[-1], device=q.device, dtype=torch.float32)
        z0 = torch.zeros(b, h, m, device=q.device, dtype=torch.float32)
    else:
        s0, z0 = (x.to(torch.float32) for x in initial_state)
    mask = valid_mask if valid_mask is not None else torch.empty(0, device=q.device, dtype=torch.bool)
    out, sf, zf = _CausalLinearScan.apply(q, k, v, mask, s0, z0, float(eps))
    return (out, (sf, zf)) if return_state else out
