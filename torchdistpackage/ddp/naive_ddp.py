import os
import time
from threading import Lock

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


__all__ = ["NaiveDDP"]


class NaiveDDP(torch.nn.Module):
    r""" NaiveDDP wraps torch.nn.Module with distribued data parallel support

    Args:
        module (torch.nn.Module, required):
            model used for computing.
        sync (boolean, default: False):
            True -> the gradient allreduce will happen after backward;
            False -> the gradient allreduce will overlap with backward.
        gradient_as_bucket_view: bucket grads
        process_group: DP process group
        dp_rank0: the first rank (rank0) of dp process group, when used with Model Parallel, rank0 is not always equal to '0'
        reduce_op: 'avg' or 'sum
        kwargs:
            num_grad_acc_iter: only do reduce grad after backward for num_grad_acc_iter times

    Note: for bucket, a warmup iter is needed to build up bucket, and correctly reduce grads
    """

    def __init__(
        self,
        module,
        sync=False,
        bucket_cap_mb=25,
        gradient_as_bucket_view=False,
        process_group=None,
        dp_rank0=0,
        reduce_op="avg",
        **kwargs,
    ):
        super(NaiveDDP, self).__init__()
        self.module = module

        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
        else:
            self.parameters_to_ignore = []

        self.group = process_group or None
        self.dp_rank0 = dp_rank0
        self.reduce_op = ReduceOp.SUM if reduce_op.lower == "sum" else ReduceOp.AVG

        # Holds all_reduce handles, used when async_reduction is True
        # self.async_handles = set()

        self.broadcast_params()

        self.sync = sync
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.bucket_cap_bytes = bucket_cap_mb * 1024 * 1024
        self.buckets = {}
        self.buckets_idx = 0

        self.num_iter = 0
        # self.lock = Lock()

        self.reduce_time = 0.0
        self.hook_time = 0.0
        self.verbose = kwargs.get("verbose", False)

        self.num_grad_acc_iter = kwargs.get("num_grad_acc_iter", 1)
        self.grad_reduce_cnts = {}

        self.reduce_stream = torch.cuda.Stream()
        if not sync and dist.get_world_size(self.group) > 1:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.module.named_parameters()):
            if p.requires_grad and name not in self.parameters_to_ignore:
                if dist.get_world_size(self._get_group(name, p)) <= 1:
                    continue

                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)  # ! very important

    def _get_group(self, name, param):
        return self.group

    def sync_comm(self):
        beg = time.perf_counter()
        self.reduce_stream.synchronize()

        self.reduce_time += time.perf_counter() - beg

    def _reduce_grads(self, grad, group, name):
        if self.sync:
            dist.all_reduce(grad, group=group, op=self.reduce_op)
        else:
            if self.grad_reduce_cnts.get(name, 0) < self.num_grad_acc_iter - 1:
                self.grad_reduce_cnts[name] = self.grad_reduce_cnts.get(name, 0) + 1
                return
            # beg = time.perf_counter()
            stream = self.reduce_stream
            stream.wait_stream(torch.cuda.current_stream())
            # self.reduce_time += time.perf_counter()-beg

            # handle = dist.all_reduce(grad, group=self.group, async_op=True, op=self.reduce_op)
            # self.async_handles.add(handle)
            with torch.cuda.stream(self.reduce_stream):
                try:
                    dist.all_reduce(
                        grad, group=self.group, async_op=False, op=self.reduce_op
                    )
                except Exception as e:
                    import pdb

                    pdb.set_trace()
                    print("Exception at _reduce_grads")

    def reduce_dispatch(self, name, p, idx=None):
        should_bucket = (
            lambda grad: grad.element_size() * grad.numel()
            < self.bucket_cap_bytes * 4 // 5
        )
        if self.gradient_as_bucket_view and should_bucket(p.grad):
            # if param has no "bucket", assign a 'bucket' to this param
            if not hasattr(p, "grad_bucket"):
                bucket_info = (p.grad.dtype, p.grad.device, self._get_group(name, p))

                bucket = None
                # if bucket_info exists, get a existing bucket
                if bucket_info in self.buckets:
                    bucket = self.buckets[bucket_info]

                # if no existing bucket or bucket cannot hold current param, create a new bucket
                if not bucket or not bucket.can_fit(p.grad):
                    bucket = self.buckets[bucket_info] = GradBucket(
                        f"grad_bucket_{self.buckets_idx}",
                        self.bucket_cap_bytes,
                        p.grad.element_size(),
                        bucket_info,
                    )
                    self.buckets_idx += 1

                p.grad = bucket.push(name, p.grad)
                p.grad_bucket = bucket

                # launch a reduce every time a new tensor comes
                self._reduce_grads(
                    p.grad.data, self._get_group(name, p), "bucket_warmup"
                )

                # we should remove the full buckets from self to make sure that bucket is not resued?
                #       not needed, since the bucket will be full, and will be replaced by next new bucket

            else:  # if param already has a 'bucket', mark current param ready, and if bucket is ready, reduce the bucket
                bucket = p.grad_bucket
                if bucket.grad_ready():
                    self._reduce_grads(bucket.data, bucket.group, bucket.name)
                    bucket.grad_reset()
        else:
            self._reduce_grads(p.grad.data, self._get_group(name, p), name)

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            # make grad thread safe
            # with self.lock:

            self.reduce_dispatch(name, p, i)

        return hook

    def _reset_iter(self):
        if self.verbose and dist.get_rank(self.group) == 0:
            print(
                "rank:",
                dist.get_rank(),
                " Total Reduce of last iter: ",
                self.reduce_time,
            )
        self.num_iter += 1
        self.reduce_time = 0.0

        # clear grad reduce cnt every iter
        for key in self.grad_reduce_cnts:
            self.grad_reduce_cnts[key] = 0

    def reduce_gradients(self):
        """ call this after a iter, to reudce grads and sync """

        # no need sync when not distributed
        if dist.get_world_size(self.group) <= 1:
            return

        beg = time.perf_counter()

        if self.sync:
            for i, (name, param) in enumerate(self.module.named_parameters()):
                if (
                    name not in self.parameters_to_ignore
                    and param.requires_grad
                    and param.grad is not None
                ):
                    if dist.get_world_size(self._get_group(name, param)) <= 1:
                        continue
                    self.reduce_dispatch(name, param, i)
        # else:
        #     for handle in self.async_handles:
        #         handle.wait()
        #     self.async_handles.clear()

        torch.cuda.synchronize()
        self.reduce_time += time.perf_counter() - beg

        self._reset_iter()

    def broadcast_params(self):
        """ broadcast model parameters """
        for name, param in self.module.state_dict().items():
            if name not in self.parameters_to_ignore:
                dist.broadcast(param, self.dp_rank0, group=self._get_group(name, param))


class MoEDP(torch.nn.Module):
    r""" NaiveDDP wraps torch.nn.Module with distribued data parallel support

    Args:
        module (torch.nn.Module, required):
            model used for computing.
        sync (boolean, default: False):
            True -> the gradient allreduce will happen after backward;
            False -> the gradient allreduce will overlap with backward.
        gradient_as_bucket_view: bucket grads
        process_group: DP process group
        dp_rank0: the first rank (rank0) of dp process group, when used with Model Parallel, rank0 is not always equal to '0'
        reduce_op: 'avg' or 'sum
        kwargs:
            num_grad_acc_iter: only do reduce grad after backward for num_grad_acc_iter times

    Note: for bucket, a warmup iter is needed to build up bucket, and correctly reduce grads
    """

    def __init__(
        self,
        expert_params,
        sync=True,
        bucket_cap_mb=25,
        gradient_as_bucket_view=False,
        process_group=None,
        dp_rank0=0,
        reduce_op="avg",
        **kwargs,
    ):
        super(MoEDP, self).__init__()
        self.expert_params = expert_params

        self.group = process_group
        self.dp_rank0 = dp_rank0
        self.reduce_op = ReduceOp.SUM if reduce_op.lower == "sum" else ReduceOp.AVG

        self.broadcast_params()

        self.sync = sync
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.bucket_cap_bytes = bucket_cap_mb * 1024 * 1024
        self.buckets = {}
        self.buckets_idx = 0

        self.num_iter = 0
        # self.lock = Lock()

        # Holds all_reduce handles, used when async_reduction is True
        self.async_handles = set()
        self.use_sync_handle = True

        self.reduce_time = 0.0
        self.hook_time = 0.0
        self.verbose = kwargs.get("verbose", False)

        self.num_grad_acc_iter = kwargs.get("num_grad_acc_iter", 1)
        self.grad_reduce_cnts = {}

        self.reduce_stream = torch.cuda.Stream()
        if dist.get_world_size(self.group) > 1:
            self._grad_accs = []
            self._register_hooks()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def broadcast_params(self):
        """ broadcast model parameters """
        for param in self.expert_params.values():
            dist.broadcast(param, self.dp_rank0, group=self.group)

    def _register_hooks(self):
        for name, p in self.expert_params.items():
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p))
                self._grad_accs.append(grad_acc)  # ! very important

    def _reduce_grads(self, grad, group, name):
        if self.sync:
            dist.all_reduce(grad, group=group, op=self.reduce_op)
        elif self.use_sync_handle:
            if self.grad_reduce_cnts.get(name, 0) < self.num_grad_acc_iter - 1:
                self.grad_reduce_cnts[name] = self.grad_reduce_cnts.get(name, 0) + 1
                return

            # handle = dist.all_reduce(grad, group=group, async_op=True, op=self.reduce_op)
            # self.async_handles.add(handle)
        else:
            stream = self.reduce_stream
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(self.reduce_stream):
                try:
                    dist.all_reduce(
                        grad, group=self.group, async_op=False, op=self.reduce_op
                    )
                except Exception as e:
                    import pdb

                    pdb.set_trace()
                    print("Exception at _reduce_grads")

    def reduce_dispatch(self, name, p):
        should_bucket = (
            lambda grad: grad.element_size() * grad.numel()
            < self.bucket_cap_bytes * 4 // 5
        )
        if self.gradient_as_bucket_view and should_bucket(p.grad):
            # if param has no "bucket", assign a 'bucket' to this param
            if not hasattr(p, "grad_bucket"):
                bucket_info = (p.grad.dtype, p.grad.device, self.group)

                bucket = None
                # if bucket_info exists, get a existing bucket
                if bucket_info in self.buckets:
                    bucket = self.buckets[bucket_info]

                # if no existing bucket or bucket cannot hold current param, create a new bucket
                if not bucket or not bucket.can_fit(p.grad):
                    bucket = self.buckets[bucket_info] = GradBucket(
                        f"grad_bucket_{self.buckets_idx}",
                        self.bucket_cap_bytes,
                        p.grad.element_size(),
                        bucket_info,
                    )
                    self.buckets_idx += 1

                p.grad = bucket.push(name, p.grad)
                p.grad_bucket = bucket

                # launch a reduce every time a new tensor comes
                self._reduce_grads(p.grad.data, self.group, "bucket_warmup")

                # we should remove the full buckets from self to make sure that bucket is not resued?
                #       not needed, since the bucket will be full, and will be replaced by next new bucket

            else:  # if param already has a 'bucket', mark current param ready, and if bucket is ready, reduce the bucket
                bucket = p.grad_bucket
                if bucket.grad_ready():
                    self._reduce_grads(bucket.data, bucket.group, bucket.name)
                    bucket.grad_reset()
        else:
            self._reduce_grads(p.grad.data, self.group, name)

    def _make_hook(self, name, p):
        def hook(*ignore):
            # make grad thread safe
            # with self.lock:

            self.reduce_dispatch(name, p)

        return hook

    def reduce_gradients(self):
        """ call this after a iter, to reudce grads and sync """

        # no need sync when not distributed
        if dist.get_world_size(self.group) <= 1:
            return

        if self.sync:
            for name, param in self.expert_params.items():
                if param.requires_grad and param.grad is not None:
                    if dist.get_world_size(self.group) <= 1:
                        continue
                    self.reduce_dispatch(name, param, i)
        else:
            for handle in self.async_handles:
                handle.wait()
            self.async_handles.clear()

        # clear grad reduce cnt every iter
        for key in self.grad_reduce_cnts:
            self.grad_reduce_cnts[key] = 0

        torch.cuda.synchronize()


moe_dp_mod = None


def moe_dp_iter_step():
    global moe_dp_mod
    moe_dp_mod.reduce_gradients()


def create_moe_dp_hooks(
    params: dict,
    moe_dp_group,
    moe_dp_rank0,
    overlap_comm=True,
    reduce_op="avg",
    sync=False,
    num_grad_acc_iter=1,
):
    global moe_dp_mod
    moe_dp_mod = MoEDP(
        params,
        sync=sync,
        process_group=moe_dp_group,
        dp_rank0=moe_dp_rank0,
        reduce_op=reduce_op,
        gradient_as_bucket_view=True,
        num_grad_acc_iter=num_grad_acc_iter,
    )
    return moe_dp_mod


class GradBucket(object):
    def __init__(self, name, size, element_size, bucket_info):
        self.element_size = element_size
        self.numel = size // self.element_size
        self.name = name
        self.dtype, self.device, self.group = bucket_info
        self.data = torch.zeros(self.numel, dtype=self.dtype, device=self.device)

        self.offset = 0
        self.grads = []
        self.ready = 0

    def get_aligned_size(self, tensor):
        aligned_size = (tensor.element_size() * tensor.numel() + 2 ** 9 - 1) & ~(
            2 ** 9 - 1
        )
        assert aligned_size % tensor.element_size() == 0
        return aligned_size // tensor.element_size()

    def grad_ready(self):
        self.ready += 1
        return self.ready >= len(self.grads)

    def grad_reset(self):
        self.ready = 0

    def can_fit(self, grad):
        return self.offset + self.get_aligned_size(grad) <= self.numel

    def push(self, name, grad):
        new_grad = self.data.narrow(0, self.offset, grad.numel()).view_as(grad)
        new_grad.copy_(grad)
        self.offset += self.get_aligned_size(grad)
        self.grads.append(name)
        return new_grad
