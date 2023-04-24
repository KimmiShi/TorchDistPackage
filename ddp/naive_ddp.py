import os
import pickle

import torch
import torch.distributed as dist
from threading import Lock

# from ..core import allreduce, get_world_size, get_group_size, broadcast, synchronize
# from ..comm import allreduce_async, allreduce_async_nocoord

__all__ = ['SdxDdp']


class SdxDdp(torch.nn.Module):
    r""" SdxDdp wraps torch.nn.Module with distribued data parallel support

    Args:
        module (torch.nn.Module, required):
            model used for computing.
        sync (boolean, default: False):
            True -> the gradient allreduce will happen after backward;
            False -> the gradient allreduce will overlap with backward.
    """
    def __init__(self, module, sync=False,
                 bucket_cap_mb=25,
                 gradient_as_bucket_view=False,
                 process_group=None,
                 params_to_group=None,
                 dp_rank0=0):
        super(SdxDdp, self).__init__()
        self.module = module

        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = module._ddp_params_and_buffers_to_ignore
        else:
            self.parameters_to_ignore = []

        self.group = process_group or None
        self.dp_rank0 = dp_rank0
        self.params_to_group = params_to_group or {}

        self.broadcast_params()

        # self.use_nocoord = os.getenv('LINKLINK_ALLREDUCE_USE_NCCL_STREAM', '0') == '1'
        self.use_nocoord = False
        self.sync = sync
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.bucket_cap_bytes = bucket_cap_mb * 1024 * 1024
        self.buckets = {}
        self.buckets_idx = 0

        self.num_iter = 0
        self.param_infos = {}
        self.lock = Lock()
        self.current_reduce_idx = 0
        self.grad_ready_queue = []
        self.first_iter_queue = []
        self.reduce_stream = torch.cuda.Stream()
        if not sync and dist.get_world_size(self.group) > 1:
            self._grad_accs = []
            self._register_hooks()
            self.first_iter_queue.reverse()

    def forward(self, *inputs, **kwargs):
        # sync reduce stream before next forward
        self.reduce_stream.synchronize()
        return self.module(*inputs, **kwargs)

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.module.named_parameters()):
            if p.requires_grad and name not in self.parameters_to_ignore:
                if dist.get_world_size(self._get_group(name, p)) <= 1:
                    continue

                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)
                self.param_infos[name] = ParamInfo(name, p, i, self._get_group(name, p))
                self.first_iter_queue.append(name)

    def _get_group(self, name, param):
        if hasattr(param, '_reduce_group') and param._reduce_group:
            return param._reduce_group
        return self.params_to_group.get(name, self.group)

    def _reduce_grads(self, name, grad, group):
        if self.sync:
            dist.all_reduce(grad, group=group)
        elif self.use_nocoord:
            allreduce_async_nocoord(grad, group)
        else:
            with torch.cuda.stream(self.reduce_stream):
                dist.all_reduce(name, grad, group, async_op=True)

    def _do_grad_reduce(self, name, p, idx):
        should_bucket = lambda grad: self._get_group(name, p) == 0 and \
            grad.element_size() * grad.numel() < self.bucket_cap_bytes * 4 // 5
        if self.gradient_as_bucket_view and should_bucket(p.grad):
            if not hasattr(p, 'grad_bucket'):
                bucket_info = (p.grad.dtype, p.grad.device, self._get_group(name, p))

                bucket = None
                if bucket_info in self.buckets:
                    bucket = self.buckets[bucket_info]

                if not bucket or not bucket.can_fit(p.grad):
                    bucket = self.buckets[bucket_info] = GradBucket(
                        f'grad_bucket_{self.buckets_idx}', self.bucket_cap_bytes, p.grad.element_size(), bucket_info)
                    self.buckets_idx += 1

                p.grad = bucket.push(name, p.grad)
                p.grad_bucket = bucket
                self._reduce_grads(name, p.grad.data, self._get_group(name, p))
            else:
                bucket = p.grad_bucket
                if bucket.grad_ready():
                    self._reduce_grads(bucket.name, bucket.data, bucket.group)
                    bucket.grad_reset()
        else:
            self._reduce_grads(name, p.grad.data, self._get_group(name, p))

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            # make grad thread safe
            with self.lock:
                assert self.param_infos[name].grad_ready_iter < self.num_iter
                self.param_infos[name].grad_ready_iter = self.num_iter

                grad_queue = self.first_iter_queue if self.num_iter == 0 else self.grad_ready_queue
                if self.num_iter == 0:
                    self.grad_ready_queue.append(name)
                while self._grad_is_ready(grad_queue, self.current_reduce_idx):
                    pname = grad_queue[self.current_reduce_idx]
                    pn, pp, pi = self.param_infos[pname].get_info()
                    self._do_grad_reduce(pn, pp, pi)
                    self.current_reduce_idx += 1
        return hook

    def _grad_is_ready(self, grad_queue, index):
        if index >= len(grad_queue):
            return False
        name = grad_queue[index]
        return self.param_infos[name].grad_ready_iter >= self.num_iter

    def _broadcast_reduce_order(self):
        # serialize to tensor
        buffer = pickle.dumps(self.grad_ready_queue)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to(device=torch.device("cpu"))
        # broadcast rand_0's reduce order
        dist.broadcast(tensor, 0)

        # deserialize
        grad_ready_queue_zero = pickle.loads(tensor.numpy().tobytes())

        # check
        assert len(grad_ready_queue_zero) == len(self.grad_ready_queue)
        assert set(grad_ready_queue_zero) == set(self.grad_ready_queue)

        self.grad_ready_queue = grad_ready_queue_zero

    def _reset_iter(self):
        self.num_iter += 1
        self.current_reduce_idx = 0

    def reduce_gradients(self):
        """ average gradients """

        # no need sync when not distributed
        if dist.get_world_size(self.group) <= 1:
            return

        if self.sync:
            for i, (name, param) in enumerate(self.module.named_parameters()):
                if name not in self.parameters_to_ignore and param.requires_grad and param.grad is not None:
                    if get_group_size(self._get_group(name, param)) <= 1:
                        continue
                    self._do_grad_reduce(name, param, i)
        else:
            if self.use_nocoord:
                dummy = torch.rand(1).cuda()
                allreduce_async_nocoord(dummy, wait=True, group=self.group)
            torch.cuda.synchronize()

            # when first iteration, we need to fix the order of gradient reduction
            # this is necessary when multi thread backward used and the gradient
            # reduction hook is called in arbitrary order
            if self.num_iter == 0:
                self._broadcast_reduce_order()

        self._reset_iter()

    def broadcast_params(self):
        """ broadcast model parameters """
        for name, param in self.module.state_dict().items():
            if name not in self.parameters_to_ignore:
                # pass
                # import pdb;pdb.set_trace()
                # broadcast(param, 0, group=self._get_group(name, param))
                dist.broadcast(param, self.dp_rank0, group=self._get_group(name, param))


class ParamInfo(object):
    def __init__(self, name, param, index, group):
        self.name = name
        self.param = param
        self.index = index
        self.group = group
        self.grad_ready_iter = -1

    def get_info(self):
        return (self.name, self.param, self.index)


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
        aligned_size = ((tensor.element_size() * tensor.numel() + 2 ** 9 - 1)
                        & ~(2 ** 9 - 1))
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
