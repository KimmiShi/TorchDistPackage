# A simple zero impl that:
#  1. shards the opt states
#  2. shards the grads
#  Supports bf16 only


# work flow of bf16 optim:
#   model param in bf16
#   -> grads in bf16
#   reduce and remove grads not needed in current partition
#   copy grads to fp32
#   optim updates fp32 copy of param using fp32 grad
#   update fp16 param using fp32 param

import torch
import torch.distributed as dist
import math

def partition_params(params, num_partitions, numel_per_partition):
    """partitions params

    Args:
        params (list): the complete list of params to partition
        num_partitions (int): zero dp world size
        numel_per_partition (int): max number of param cnt

    Returns:
        list: list of partitions
    """
    partitions = []
    elcnt = 0
    partition_id = 0
    for ind in range(num_partitions):
        partitions.append([])
    for param in params:
        partitions[partition_id].append(param)
        elcnt+=param.numel()
        if elcnt > numel_per_partition:
            partition_id+=1
            elcnt=0
    return partitions


class Bf16ZeroOptimizer():
    def __init__(self, optim, dp_group=None, bf16_master_weights=False, overlap_comm=True) -> None:
        self.optim = optim
        self.dp_group = dp_group
        self.bf16_master_weights = bf16_master_weights
        # self.partition_grad = stage==2

        self.grad_accs = []
        if torch.distributed.is_initialized():
            self.partition_id = dist.get_rank(self.dp_group)
            num_partitions = dist.get_world_size(self.dp_group)
        else:
            self.partition_id = 0
            num_partitions = 1
        self.num_partitions = num_partitions

        overlap_comm=overlap_comm
        self.reduce_stream = torch.cuda.Stream() if overlap_comm else torch.cuda.current_stream()
        self.reduce_op = dist.ReduceOp.AVG

        self.past_grad = []

        self.all_param_groups_partitions = []
        self.bit16_params_shard_groups = []
        self.master_weight_shard_groups = []
        self.bf16_param_id_in_partition = set()
        self.bf16_param_to_master_weight_map = dict()

        for param_group in self.optim.param_groups:
            trainable_parameters = [param for param in param_group['params'] if param.requires_grad]
            total_num_elements = sum([p.numel() for p in trainable_parameters])
            target_partition_numel = math.ceil(total_num_elements//num_partitions)
            all_partitions = partition_params(trainable_parameters, num_partitions, target_partition_numel)
            self.all_param_groups_partitions.append(all_partitions)
            params_in_cur_partition = all_partitions[self.partition_id]
            self.bit16_params_shard_groups.append(params_in_cur_partition)

            for param in params_in_cur_partition:
                self.bf16_param_id_in_partition.add(id(param))

            if bf16_master_weights:
                self.master_weight_shard_groups.append(params_in_cur_partition)

            # update optim's param group
            param_group['params'] = self.master_weight_shard_groups[-1]

            for ind,param in enumerate(trainable_parameters):
                    def wrapper(param, ind):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]
                        def reduce_partition_and_remove_grads(*notneeded):
                            reduce_and_remove_grad(param)

                        grad_acc.register_hook(reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)
                    wrapper(param, ind)

        # create hook that does reduce & remove grad
        def reduce_and_remove_grad(param):
            if self.num_partitions > 1:
                self.reduce_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.reduce_stream):
                    dist.all_reduce(
                        param.grad.data, group=self.dp_group, async_op=False, op=self.reduce_op
                    )
                    param.reduced=True
                    if id(param) in self.bf16_param_id_in_partition:
                        if not self.bf16_master_weights:
                            master_weight = self.bf16_param_to_master_weight_map[id(param)]
                            master_weight.grad.data.copy_(param.grad.data)
                    else:
                        param.grad = None

    def step(self):
        # check grads
        # for param_group in self.optim.param_groups:
        #     for param in param_group['params']:
        #         if id(param) in self.bf16_param_id_in_partition:
        #             if param.grad is None:
        #                 import pdb;pdb.set_trace()
        #                 print('err')
        #         else:
        #             import pdb;pdb.set_trace()
        #             print('err')

        # 1. param update of single partition
        self.optim.step()


        # 2. update bf16 param with fp32 param in current partition
        # if not self.bf16_master_weights:
        #     for ind in range(len(self.bit16_params_shard_groups)):
        #         # self.bit16_params_shard_groups[ind].data.copy_(self.master_weight_shard_groups[ind])
        #         for param_ind in range(len(self.bit16_params_shard_groups[ind])):
        #             self.bit16_params_shard_groups[ind][param_ind].data.copy_(self.master_weight_shard_groups[ind][param_ind])
        # 3. all-gather bit16 params
        #    do this by broadcast
        if self.num_partitions ==1:
            return
        for param_partitions in self.all_param_groups_partitions:
            for partition_id in range(self.num_partitions):
                partition = param_partitions[partition_id]
                # broadcast partition from rank partition_id to the rest
                for param in partition:
                    dist.broadcast(param.data, partition_id, self.dp_group)

    def zero_grad(self):
        # self.optim.zero_grad()
        for pg_partitions in self.all_param_groups_partitions:
            for parition in pg_partitions:
                for p in parition:
                    if p.grad is not None:
                        p.grad.zero_()
    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optim.state

    def _set_state(self, value):
        self.optim.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optim.param_groups

    def _set_param_groups(self, value):
        self.optim.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)



