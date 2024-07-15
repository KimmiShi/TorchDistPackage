import torch

import torch.distributed as dist

import math

import time
from flash_attn import (
    flash_attn_func,)

BLOCK_SIZE=128

def loop_q_tile_attn(q, k, v, scale, sram_m):
    """tiled attn

    Args:
        q,k,v (tensor): shape (b, nh, N, dim)

    Notes:
        1. split k,v into bc blocks, each N//bc tokens;
    """

    k = k.transpose(-1, -2)      # k: (b, nh, dim, N)
    b, nh, n, d = q.shape

    q=q*scale

    q_block_size = min(BLOCK_SIZE, n)
    kv_block_size = min(BLOCK_SIZE, d)
    q_block_num = n//q_block_size           # num blocks of Q, each of size (q_block_size,d)
    kv_block_num = n//kv_block_size          # num blocks of K,V, each of size (kv_block_size,d)

    # compute block of O, each (q_block_size,d)
    # blocks of l and m are of size (q_block_size)
    # l = torch.zeros(n).cuda().to(q.dtype)
    # m = torch.zeros(n).cuda().to(q.dtype)
    O = torch.zeros_like(q)

    for ind_q in range(q_block_num):
        qi = q[:,:, ind_q*q_block_size:(ind_q+1)*q_block_size]
        # qi: (nq, d)
        # attn=q@kt: (nq, d) @ (d, n): (nq, n)
        # attn@v: (nq, n) @ (n, d): (nq, d)
        attn_i = qi @ k
        score_i = attn_i.softmax(dim=-1)
        O[:,:, ind_q*q_block_size:(ind_q+1)*q_block_size] = score_i @ v

    return O

def loop_kv_tile_attn(q, k, v, scale, sram_m):
    """tiled attn

    Args:
        q,k,v (tensor): shape (b, nh, N, dim)

    Notes:
        1. split q into bc blocks, each N//bc tokens;
    """
    b, nh, n, d = q.shape

    q=q*scale
    BLOCK_SIZE=2
    q_block_size = min(BLOCK_SIZE, n)
    kv_block_size = min(BLOCK_SIZE, d)
    q_block_num = n//q_block_size           # num blocks of Q, each of size (q_block_size,d)
    kv_block_num = n//kv_block_size          # num blocks of K,V, each of size (kv_block_size,d)

    # compute block of O, each (q_block_size,d)
    # blocks of l and m are of size (q_block_size)
    exp_sum = torch.zeros([b,nh,n]).cuda().to(q.dtype)
    maxes = torch.zeros([b,nh,n]).cuda().to(q.dtype)
    O = torch.zeros_like(q)

    for j in range(kv_block_num):
        # ki,vi: (b, nh, blk, d)
        kj = k[:,:, j*kv_block_size:(j+1)*kv_block_size]
        vj = v[:,:, j*kv_block_size:(j+1)*kv_block_size]
        # ki: (b, nh, d, blk)
        kj = kj.transpose(-1, -2)

        attn_j = q @ kj     # (b, nh, N, blk)

        # prepare for scaled-softmax
        m_j, _ = torch.max(attn_j, dim=-1, keepdims=False)      # (b, nh, n)
        exp_j = torch.exp(attn_j - m_j)
        exp_sum_j = torch.sum(exp_j, dim=-1)                    # (b, nh, n)
        breakpoint()
        # score_j = exp_j / exp_sum_j

        # update with history max
        new_m = torch.max(m_j, maxes)
        # update exp sum
        new_exp_sum = exp_sum_j*torch.exp(new_m - m_j) + exp_sum[j]
        breakpoint()


    return O


def loop_qkv_tile_attn_fwd(q, k, v, scale):
    """tiled attn

    Args:
        q,k,v (tensor): shape (b, nh, N, dim)

    """
    b, nh, n, d = q.shape

    q=q*scale
    BLOCK_SIZE=2 if n < 10 else n//8
    q_block_size = min(BLOCK_SIZE, n)
    kv_block_size = min(BLOCK_SIZE, d)
    q_block_num = n//q_block_size           # num blocks of Q, each of size (q_block_size,d)
    kv_block_num = n//kv_block_size          # num blocks of K,V, each of size (kv_block_size,d)

    # compute block of O, each (q_block_size,d)
    # blocks of l and m are of size (q_block_size)
    expsum = torch.zeros([b,nh,n,1]).cuda().to(q.dtype)
    maxes = torch.zeros([b,nh,n,1]).cuda().to(q.dtype) - 1e-10
    O = torch.zeros_like(q)

    for j in range(kv_block_num):
        # ki,vi: (b, nh, nk, d)
        kj = k[:,:, j*kv_block_size:(j+1)*kv_block_size]
        vj = v[:,:, j*kv_block_size:(j+1)*kv_block_size]
        # ki: (b, nh, d, nk)
        kj = kj.transpose(-1, -2)

        for i in range(q_block_num):
            qi = q[:,:, i*q_block_size:(i+1)*q_block_size]
            oi = O[:,:, i*q_block_size:(i+1)*q_block_size]
            qikj = qi @ kj     # (b, nh, nq, nk)

            # prepare for scaled-softmax
            max_ij,_ = torch.max(qikj, dim=-1, keepdim=True)  # (b, nh, nq, 1)
            exp_ij = torch.exp(qikj - max_ij)               # (b, nh, nq, nk)
            expsum_ij = torch.sum(exp_ij, dim=-1, keepdim=True) + 1e-10 # (b, nh, nq, 1)

            # update max, and update expsum
            max_i = maxes[:,:, i*q_block_size:(i+1)*q_block_size]
            expsum_i = expsum[:,:, i*q_block_size:(i+1)*q_block_size]
            new_max_i = torch.maximum(max_ij, max_i)
            new_expsum_i = expsum_ij*torch.exp(max_ij - new_max_i) + torch.exp(max_i-new_max_i)*expsum_i

            # current softmax
            softmax_ij = exp_ij *torch.exp(max_ij-new_max_i)  / new_expsum_i    # (b, nh, nq, nk)

            O[:,:, i*q_block_size:(i+1)*q_block_size] = softmax_ij @ vj + \
                oi*torch.exp(max_i-new_max_i)*expsum_i/new_expsum_i

            maxes[:,:, i*q_block_size:(i+1)*q_block_size] = new_max_i
            expsum[:,:, i*q_block_size:(i+1)*q_block_size] = new_expsum_i

    return O, maxes, expsum

def loop_qkv_tile_attn_bwd(q, k, v, scale, o, maxes, expsum, do):
    """tiled attn

    Args:
        q,k,v (tensor): shape (b, nh, N, dim)
        do: grad of fwd output, shape (b, nh, N, dim)
        maxes, expsum: (b, nh, N, 1)
    """
    b, nh, n, d = q.shape

    BLOCK_SIZE=2 if n < 10 else n//8
    q_block_size = min(BLOCK_SIZE, n)
    kv_block_size = min(BLOCK_SIZE, d)
    q_block_num = n//q_block_size           # num blocks of Q, each of size (q_block_size,d)
    kv_block_num = n//kv_block_size          # num blocks of K,V, each of size (kv_block_size,d)

    # prepeare grad output
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for j in range(kv_block_num):
        # ki,vi: (b, nh, nk, d)
        kj = k[:,:, j*kv_block_size:(j+1)*kv_block_size]
        vj = v[:,:, j*kv_block_size:(j+1)*kv_block_size]
        # ki: (b, nh, d, nk)
        kj = kj.transpose(-1, -2)

        for i in range(q_block_num):
            qi = q[:,:, i*q_block_size:(i+1)*q_block_size]
            oi = o[:,:, i*q_block_size:(i+1)*q_block_size]    # (b, nh, nq, d)
            doi = do[:,:, i*q_block_size:(i+1)*q_block_size]    # (b, nh, nq, d)
            # 1. compute softmax(q@k^T)
            qi_scaled = qi*scale
            qikj = qi_scaled @ kj                           # (b, nh, nq, nk)
            max_i = maxes[:,:, i*q_block_size:(i+1)*q_block_size]
            expsum_i = expsum[:,:, i*q_block_size:(i+1)*q_block_size]
            exp_ij = torch.exp(qikj - max_i)                # (b, nh, nq, nk)
            # current softmax
            softmax_ij = exp_ij / expsum_i                  # (b, nh, nq, nk)

            d_vj_cur = softmax_ij.transpose(-1,-2) @ doi    # (b, nh, nk, d)
            # sum the grad for different i
            dv[:,:, j*kv_block_size:(j+1)*kv_block_size] += d_vj_cur

            d_softmax_ij = doi @ vj.transpose(-1,-2)        # (b, nh, nq, nk)
            # how to calculate d_qikj?
            di = torch.sum(doi*oi, dim=-1, keepdim=True)
            d_qikj = softmax_ij * (d_softmax_ij - di)       # (b, nh, nq, nk)

            # dq, dk
            dqi_cur = d_qikj @ kj.transpose(-1, -2)
            dkj_cur = d_qikj.transpose(-1, -2) @ qi
            dq[:,:, i*q_block_size:(i+1)*q_block_size] += dqi_cur * scale
            dk[:,:, j*kv_block_size:(j+1)*kv_block_size] += dkj_cur * scale

    return dq, dk, dv


def baseline_attn(q,k,v, drop_out, scale, mode='math'):
    if mode=='math':
        attn = q*scale @ k.transpose(-1,-2)
        score = attn.softmax(dim=-1)
        x = score @ v
        return x
    else:
        x = flash_attn_func(q,k,v, drop_out.p, softmax_scale=scale, causal=False)
        return x


def test_core_attn(b=1, s=8192, d=4096, nh=32, mode='math', p=0.):
    attn_drop = torch.nn.Dropout(p)
    if mode == 'math':
        shape = [b, nh, s, d//nh]
    elif mode=='flash':
        shape = [b, s, nh, d//nh]
    dt = torch.float
    q = torch.rand(shape, dtype=dt, device='cuda').requires_grad_()
    k = torch.rand(shape, dtype=dt, device='cuda').requires_grad_()
    v = torch.rand(shape, dtype=dt, device='cuda').requires_grad_()

    scale = (d//nh)**-0.5

    out0 = baseline_attn(q,k,v, attn_drop, scale, mode=mode)

    out, maxes, expsum = loop_qkv_tile_attn_fwd(q, k, v, scale)
    print(torch.allclose(out, out0))

    # bwd:
    do = torch.ones_like(out).requires_grad_(False)
    dq, dk, dv = loop_qkv_tile_attn_bwd(q, k, v, scale, out, maxes, expsum, do)

    out0.backward(do)
    print(torch.allclose(q.grad, dq, atol=1e-5))
    print(torch.allclose(k.grad, dk, atol=1e-5))
    print(torch.allclose(v.grad, dv))
    breakpoint()

# test_core_attn(s=8, d=2, nh=1)
test_core_attn(s=1024, d=1024, nh=1)