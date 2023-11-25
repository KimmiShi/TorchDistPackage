import torch
import torch.distributed as dist
from collections import OrderedDict

from torchdistpackage.utils import partition_params

class ShardedEMA():
    def __init__(self, model, group=None) -> None:
        self.rank = dist.get_rank(group)
        # self.cpu_group = dist.new_group(backend='gloo')
        self.group = group
        # divide param in buckets
        self.all_param_shards = partition_params(model, dist.get_world_size(group), return_dict=True)
        self.param_shard = {}
        #import pdb;pdb.set_trace()
        for name, p in self.all_param_shards[self.rank].items():
            self.param_shard[name] = p.clone().detach().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, decay=0.9999, only_trainable=True):
        model_params = OrderedDict(model.named_parameters())
        for name in self.param_shard.keys():
            if only_trainable and (not model_params[name].requires_grad):
                continue
            self.param_shard[name].mul_(decay).add_(model_params[name].data, alpha=1 - decay)

    def state_dict_shard(self):
        return self.param_shard

    def summon_full_cpu(self):
        
        state_dict = {}
        for name, val in self.param_shard.items():
            if self.rank==0:
                state_dict[name] = val.cpu()

        for rank in range(1, len(self.all_param_shards)):
            # send from rank to rank0
            params_in_cur_rank = self.all_param_shards[rank]
            for param_name in params_in_cur_rank.keys():
                if self.rank == rank:
                    src_p = self.param_shard[param_name]
                    dist.send(src_p.cuda(), 0, self.group)
                    #print(f"send {param_name} from rank {rank} to 0")
                if self.rank == 0:
                    recv_buffer = torch.empty_like(self.all_param_shards[rank][param_name])
                    dist.recv(recv_buffer, rank, self.group)
                    #print(f"rank0 recv {param_name} from rank {rank}")
                    state_dict[param_name] = recv_buffer.cpu()
                dist.barrier()

        for name, param in state_dict.items():
            if param.device != torch.device('cpu'):
                import pdb;pdb.set_trace()    
        return state_dict
