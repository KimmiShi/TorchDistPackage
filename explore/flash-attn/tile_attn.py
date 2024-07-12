import torch

import torch.distributed as dist

import math

import time
from flash_attn import (
    flash_attn_func,)



def tile_attn_fwd(q, k, v, scale, sram_m):
    """tiled attn

    Args:
        q (tensor): shape (N, dim)
        k (tensor): shape (N, dim)
        q (tensor): shape (N, dim)

    """
    k = k.transpose(-1, -2)      # k: (dim, N)
    n,d = q.shape
    bc = sram_m//(4*d)
    br = min(bc, d)
    tr = n//br          # num blocks of Q, each of size (br,d)
    tc = n//bc          # num blocks of K,V, each of size (bc,d)
    # compute block of O, each (br,d)
    # blocks of l and m are of size (br)
    l = torch.zeros(n).cuda().to(q.dtype)
    m = torch.zeros(n).cuda().to(q.dtype)
    o = torch.zeros_like(q)
    for j in range(tc):
        kj = k[:,j*bc : (j+1)*bc]      # d, bc
        vj = v[j*bc : (j+1)*bc]         # bc, d
        for i in range(tr):
            qi = q[i*br: (i+1)*br]
            oi = o[i*br: (i+1)*br]
            li = l[i*br: (i+1)*br]
            mi = m[i*br: (i+1)*br]  # br

            # import pdb;pdb.set_trace()

            sij = scale*qi@kj       # br,bc
            # compute m,p,l for softmax
            mij = sij.max(1).values            # (br), 对内层求max，得到br个max值
            pij = torch.exp(sij-mij)    # br,bc
            lij = torch.sum(pij, 1)     # br, 对内层求和

            # update mij,lij,
            mi_new = torch.maximum(mij, mi)         # br
            li_new =lij*torch.exp(mij-mi_new) + li*torch.exp(mi-mi_new)   # br

            pij_new = pij*torch.exp(mij-mi_new)      # br, bc
            # compute o
            oij = (pij_new / li_new )@vj
            # import pdb;pdb.set_trace()
            tmp = li*torch.exp(mi-mi_new)/li_new
            oi = oi*tmp.unsqueeze(-1) + oij

    return o



saved_tensors = []

def pack_hook(x):
    # global saved_tensors
    print("Packing", x.numel(), x.grad_fn, x.data_ptr())
    # saved_tensors.append(x)
    return x

def unpack_hook(x):
    print("Unpacking", x.numel())
    return x


def core_attn(q,k,v, drop_out, scale, mode='math'):
    mem_alloc_beg = torch.cuda.memory_allocated()
    if mode=='math':
        attn = q*scale @ k.transpose(-1,-2)
        print("q@kT", (torch.cuda.memory_allocated() - mem_alloc_beg)/1024**2)
        mem_alloc_beg = torch.cuda.memory_allocated()

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):

            score = attn.softmax(dim=-1)
        print("softmax", (torch.cuda.memory_allocated() - mem_alloc_beg)/1024**2)
        mem_alloc_beg = torch.cuda.memory_allocated()
        # score = attn
        # score = drop_out(score)
        # print("dropout", (torch.cuda.memory_allocated() - mem_alloc_beg)/1024**2)
        # mem_alloc_beg = torch.cuda.memory_allocated()
        # return score
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):

            x = score @ v
        print("attn@v", (torch.cuda.memory_allocated() - mem_alloc_beg)/1024**2)
        mem_alloc_beg = torch.cuda.memory_allocated()
        # print("inside attn",mem_alloc_beg)
        return x
    else:
        x = flash_attn_func(q,k,v, drop_out.p, softmax_scale=scale, causal=False)
        return x


def test_core_attn(b=1, s=8192, d=4096, nh=32, mode='math', p=0.):

    attn_drop = torch.nn.Dropout(p)
    # shape = [b, nh, s, d//nh]
    shape = (d,d)
    # if mode == 'math':
    #     shape = [b, nh, s, d//nh]
    # elif mode=='flash':
    #     shape = [b, s, nh, d//nh]

    q = torch.rand(shape, dtype=torch.bfloat16, device='cuda').requires_grad_()
    k = torch.rand(shape, dtype=torch.bfloat16, device='cuda').requires_grad_()
    v = torch.rand(shape, dtype=torch.bfloat16, device='cuda').requires_grad_()

    print("q size:", q.numel()*2/1024**2)
    scale = (d//nh)**-0.5

    mem_alloc_beg = torch.cuda.memory_allocated()
    out0 = core_attn(q,k,v, attn_drop, scale, mode=mode)
    out = tile_attn_fwd(q,k,v, scale, 64*d)
    import pdb;pdb.set_trace()

    # print("after attn",torch.cuda.memory_allocated())
    print("memory used", (torch.cuda.memory_allocated() - mem_alloc_beg)/1024**2)
    if mode=='math':
        theo_mem =(5 if p>0 else 2 )*b*nh*(s**2)  # 6*b*s*d+
    else:
        theo_mem = 2*b*s*d
    print("theo_mem:", theo_mem/1024**2)

test_core_attn()