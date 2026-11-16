"""fused_kernels.py: Prototypes of 3 high-impact fused Triton kernels for FlintTorch.

1. Fused RMSNorm (fp32):
   Computes y = (x / sqrt(mean(x^2) + eps)) * w in a single pass.
   Replaces 6 PyTorch operator launches (pow, mean, add, rsqrt, mul, mul) and temporary buffers.
   Autograd-safe: dispatches to fused Triton kernel in forward/no-grad, and exact graph in autograd backward.

2. Fused SwiGLU (fp32):
   Computes y = silu(g) * u = (g / (1 + exp(-g))) * u in a single streaming elementwise pass.
   Eliminates SiLU temporary buffer and fuses 2 launches into 1.
   Autograd-safe: dispatches to fused Triton kernel in forward/no-grad, and exact graph in autograd backward.

3. Fused Multiplier Application (fp32):
   Computes out[g, t, n] = z[g, t, n] * multp[eids[g], n, 0] directly.
   Eliminates the separate index_select launch, the mult temporary tensor, and transpose.
   Works directly inside custom autograd Function ExpLinK.
"""

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# 1. Fused RMSNorm
# -----------------------------------------------------------------------------
@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_xm, stride_outm,
    D, eps,
    BLOCK_D: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_D)
    mask = col_offsets < D

    # Load input row
    x = tl.load(x_ptr + row_idx * stride_xm + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute mean of squares
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)

    # Load scale weight
    w = tl.load(w_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute output
    out = (x * inv_rms) * w
    tl.store(out_ptr + row_idx * stride_outm + col_offsets, out, mask=mask)


def fused_rmsnorm_triton(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton implementation of RMSNorm matching FlintTorch:
    x: [..., D], w: [D] -> y: [..., D] in float32.
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    M = x.numel() // D

    x_2d = x.reshape(M, D).contiguous()
    w_1d = w.reshape(D).contiguous()
    out_2d = torch.empty_like(x_2d)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_D = max(BLOCK_D, 128)

    num_warps = 8 if D >= 2048 else 4
    _rmsnorm_fwd_kernel[(M,)](
        x_2d, w_1d, out_2d,
        x_2d.stride(0), out_2d.stride(0),
        D, eps,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out_2d.reshape(orig_shape)


def fused_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Autograd-safe wrapper for RMSNorm."""
    if torch.is_grad_enabled() and x.requires_grad:
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w
    return fused_rmsnorm_triton(x, w, eps)


# -----------------------------------------------------------------------------
# 2. Fused SwiGLU
# -----------------------------------------------------------------------------
@triton.jit
def _swiglu_fwd_kernel(
    g_ptr, u_ptr, out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    g = tl.load(g_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU(g) = g / (1.0 + exp(-g))
    sig_g = 1.0 / (1.0 + tl.exp(-g))
    silu_g = g * sig_g
    out = silu_g * u

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Triton implementation of SwiGLU:
    gate: [..., N], up: [..., N] -> out: [..., N] in float32.
    Computes silu(gate) * up in a single pass.
    """
    assert gate.shape == up.shape, f"Shape mismatch: {gate.shape} vs {up.shape}"
    g_cont = gate.contiguous()
    u_cont = up.contiguous()
    out = torch.empty_like(g_cont)

    total_elements = g_cont.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

    _swiglu_fwd_kernel[grid](
        g_cont, u_cont, out,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Autograd-safe wrapper for SwiGLU."""
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        return torch.nn.functional.silu(gate.float()) * up.float()
    return fused_swiglu_triton(gate, up)


# -----------------------------------------------------------------------------
# 3. Fused Multiplier Application
# -----------------------------------------------------------------------------
@triton.jit
def _mult_apply_kernel(
    z_ptr, multp_ptr, eids_ptr, out_ptr,
    G, MT, N,
    stride_zg, stride_zt,
    stride_outg, stride_outt,
    stride_me, stride_mn,
    BLOCK_N: tl.constexpr
):
    pid_g = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_n = tl.program_id(2)

    eid = tl.load(eids_ptr + pid_g).to(tl.int64)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offsets < N

    z_offset = pid_g * stride_zg + pid_t * stride_zt + n_offsets
    m_offset = eid * stride_me + n_offsets * stride_mn

    z_val = tl.load(z_ptr + z_offset, mask=mask, other=0.0).to(tl.float32)
    m_val = tl.load(multp_ptr + m_offset, mask=mask, other=0.0).to(tl.float32)

    out_val = z_val * m_val
    tl.store(out_ptr + pid_g * stride_outg + pid_t * stride_outt + n_offsets, out_val, mask=mask)


def fused_mult_apply_triton(z: torch.Tensor, multp: torch.Tensor, eids: torch.Tensor) -> torch.Tensor:
    """Computes out[g, t, n] = z[g, t, n] * multp[eids[g], n, 0] directly.
    z: [G, MT, N]
    multp: [E, N, 1]
    eids: [G]
    """
    G, MT, N = z.shape
    z_cont = z.contiguous()
    eids_cont = eids.contiguous().to(torch.int64)
    out = torch.empty_like(z_cont)

    BLOCK_N = 128
    grid = (G, MT, triton.cdiv(N, BLOCK_N))

    _mult_apply_kernel[grid](
        z_cont, multp, eids_cont, out,
        G, MT, N,
        z_cont.stride(0), z_cont.stride(1),
        out.stride(0), out.stride(1),
        multp.stride(0), multp.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out
