"""Causal linear attention on already mapped Q/K features.

Layouts are Q/K [batch, heads, length, features] and V/output
[batch, heads, length, value_dim]. Padding masks mark *valid* keys.
The torch implementation is also the differentiable CPU reference.
"""

import math

import torch

from models.linear_triton import triton_causal_linear_attention, triton_supported


def _validate(q, k, v, valid_mask, initial_state, eps):
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("Q/K must have equal [B,H,T,M] shapes and V must be [B,H,T,Dv].")
    if v.shape[:3] != q.shape[:3] or min(*q.shape, v.shape[-1]) <= 0:
        raise ValueError("Q/K/V batch, head and nonempty sequence dimensions must match.")
    if not q.is_floating_point() or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V must have the same floating-point dtype.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same device.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    b, h, t, m = q.shape
    if valid_mask is not None:
        if valid_mask.shape != (b, t) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [B,T], with True for valid keys.")
        if valid_mask.device != q.device:
            raise ValueError("valid_mask must be on the Q/K/V device.")
    if initial_state is not None:
        if len(initial_state) != 2:
            raise ValueError("initial_state must be the pair (S, z).")
        s, z = initial_state
        if s.shape != (b, h, m, v.shape[-1]) or z.shape != (b, h, m):
            raise ValueError("Initial S/z shapes must be [B,H,M,Dv] and [B,H,M].")
        if s.device != q.device or z.device != q.device:
            raise ValueError("Initial states must be on the Q/K/V device.")
        if not s.is_floating_point() or not z.is_floating_point():
            raise ValueError("Initial states must be floating-point tensors.")


def torch_causal_linear_attention(
    q, k, v, valid_mask=None, initial_state=None, return_state=False, eps=1e-6
):
    """Differentiable recurrent reference, including gradients through states.

    Only the current state is explicitly maintained. During training PyTorch
    autograd can still retain per-step intermediates; this is not a claim of
    constant training memory. Float16/bfloat16 states accumulate in float32.
    """
    _validate(q, k, v, valid_mask, initial_state, eps)
    b, h, length, m = q.shape
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_acc, k_acc, v_acc = (x.to(acc_dtype) for x in (q, k, v))
    if initial_state is None:
        s = torch.zeros(b, h, m, v.shape[-1], device=q.device, dtype=acc_dtype)
        z = torch.zeros(b, h, m, device=q.device, dtype=acc_dtype)
    else:
        s, z = (x.to(acc_dtype) for x in initial_state)
    outputs = []
    for t in range(length):
        kt, vt = k_acc[:, :, t], v_acc[:, :, t]
        if valid_mask is not None:
            valid = valid_mask[:, t, None, None]
            kt = torch.where(valid, kt, 0.0)
            vt = torch.where(valid, vt, 0.0)
        s = s + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        z = z + kt
        qt = q_acc[:, :, t]
        numerator = torch.einsum("bhm,bhmv->bhv", qt, s)
        denominator = (qt * z).sum(dim=-1, keepdim=True) + eps
        outputs.append(numerator / denominator)
    out = torch.stack(outputs, dim=2).to(v.dtype)
    return (out, (s, z)) if return_state else out


def causal_linear_attention(
    q,
    k,
    v,
    valid_mask=None,
    initial_state=None,
    return_state=False,
    eps=1e-6,
    backend="auto",
):
    """Evaluate causal attention using torch or fused Triton recurrence.

    ``auto`` selects Triton on supported CUDA tensors and otherwise torch.
    ``triton`` fails explicitly if unavailable, instead of silently falling
    back. Arbitrary feature/value dimensions (within the kernel limits) need
    not be equal or powers of two. Initial/final states are differentiable.
    """
    _validate(q, k, v, valid_mask, initial_state, eps)
    if backend not in ("auto", "torch", "triton"):
        raise ValueError("backend must be 'auto', 'torch', or 'triton'.")
    supported = triton_supported(q, v)
    if backend == "triton" and not supported:
        raise RuntimeError(
            "Triton requires CUDA, triton, float16/bfloat16/float32, "
            "M <= 256, Dv <= 128, and padded M*Dv <= 32768. "
            "Use backend='torch' or backend='auto' on other configurations."
        )
    if backend == "triton" or (backend == "auto" and supported):
        return triton_causal_linear_attention(
            q, k, v, valid_mask, initial_state, return_state, eps
        )
    return torch_causal_linear_attention(
        q, k, v, valid_mask, initial_state, return_state, eps
    )
