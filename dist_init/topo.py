from functools import partial

import numpy
import torch.distributed as dist

class SingletonMeta(type):
    """
    Adapted from colossalai.p2p.

    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            assert len(args) == 0 and len(
                kwargs) == 0, f'{cls.__name__} is a singleton class and a instance has been created.'
        return cls._instances[cls]

def gen_inner_ranks(world_size, group_size):
    num_groups = int(world_size//group_size)
    return [list(range(g*group_size, (g+1)*group_size)) for g in range(num_groups)]

def gen_groups(world_size, group_size, strides, hook):
    group_size = int(group_size)
    num_groups = int(world_size//group_size)
    if strides is None or len(strides)==0:
        # most inner
        list_of_ranks = gen_inner_ranks(world_size, group_size)
        # print(list_of_ranks)
        for ranks in list_of_ranks:
            hook(ranks)
    else:
        inner_group_size = numpy.prod(strides)
        list_of_inner_ranks = gen_inner_ranks(world_size, inner_group_size)
        # print(list_of_inner_ranks)

        for ind in range(inner_group_size):
            chunks = [list_of_inner_ranks[x:x+group_size] for x in range(0, len(list_of_inner_ranks), group_size)]
            # print(chunks)
            for chunk in chunks:
                cur_ranks = [ranks[ind] for ranks in chunk]
                hook(cur_ranks)

class ProcessTopology(metaclass=SingletonMeta):
    def __init__(self):
        self._groups = dict()
        self._ranks_in_group = dict()
        self._dp_ranks_all = []

    def setup_process_groups(self, config:list):
        """
            Example: setup_process_groups([('data',4), ('pipe',2), ('tensor',2)])   # world_size=16

            Result:
                tensor parallel groups:
                    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
                pipeline parallel groups:
                    [0, 2]
                    [4, 6]
                    [8, 10]
                    [12, 14]
                    [1, 3]
                    [5, 7]
                    [9, 11]
                    [13, 15]
                data parallel groups:
                    [0, 4, 8, 12]
                    [1, 5, 9, 13]
                    [2, 6, 10, 14]
                    [3, 7, 11, 15]
            Usage:
                # setup
                dist_init_slurm()
                dist_config = [('data',world_size/(2*pp_size)), ('pipe',pp_size), ('tensor',2)]
                global_context.setup_process_groups(dist_config)
                # api example
                test_comm()

        """
        def build_group(type, ranks):
            if type == 'data':
                self._dp_ranks_all.append(ranks)
            from datetime import timedelta
            grp = dist.new_group(ranks, timeout=timedelta(seconds=100))
            if dist.get_rank() in ranks:
                self._groups[type] = grp
                self._ranks_in_group[type] = ranks

                if dist.get_rank() == ranks[0]:
                    print(f"group {type}, ranks: {ranks}")

        dims = [item[0] for item in config]
        sizes = [int(item[1]) for item in config]


        for (dim, size) in config:
            cur_dim_ind = dims.index(dim)
            strides=sizes[cur_dim_ind+1:] if cur_dim_ind+1 < len(sizes) else []

            gen_groups(dist.get_world_size(), size, strides, partial(build_group, dim))

        # build Model Parallel Group Automatically
        if "tensor" in dims or "pipe" in dims:

            for g in range(len(self._dp_ranks_all[0])):
                model_ranks = [dp_ranks[g] for dp_ranks in self._dp_ranks_all]
                build_group("model", model_ranks)

    def get_group(self, type_name):
        return self._groups[type_name]


    def get_group_rank(self, type_name):
        return dist.get_rank(group=self._groups[type_name])


    def get_ranks_in_group(self, type_name):
        return self._ranks_in_group[type_name]

    def get_tp_rank(self):
        return self.get_group_rank('tensor')

    def get_pp_rank(self):
        return self.get_group_rank('pipe')

    def get_dp_rank(self):
        return self.get_group_rank('data')

    def get_mp_rank(self):
        return self.get_group_rank('model')

    def get_group_size(self, type_name):
        return len(self._ranks_in_group[type_name])

    def get_tp_size(self):
        return self.get_group_size('tensor')

    def get_pp_size(self):
        return self.get_group_size('pipe')

    def get_dp_size(self):
        return self.get_group_size('data')

    def get_mp_size(self):
        return self.get_group_size('model')

    def is_first_in_group(self, type_name):
        return self.get_group_rank(type_name) == 0

    def is_last_in_group(self, type_name):
        return dist.get_rank() == self._ranks_in_group[type_name][-1]

    def is_first_in_tensor_group(self):
        return self.is_first_in_group('tensor')

    def is_last_in_tensor_group(self):
        return self.is_last_in_group('tensor')

    def is_first_in_pipeline_group(self):
        return self.is_first_in_group('pipe')

    def is_last_in_pipeline_group(self):
        return self.is_last_in_group('pipe')

    def is_first_in_data_group(self):
        return self.is_first_in_group('data')

    def is_last_in_data_group(self):
        return self.is_last_in_group('data')

    def is_first_in_model_group(self):
        return self.is_first_in_group('model')

    def is_last_in_model_group(self):
        return self.is_last_in_group('model')

    def get_prev_global_rank(self, type_name = 'pipe'):
        local_rank = self.get_group_rank(type_name)
        world_size = self.get_group_size(type_name)
        ranks_in_group = self.get_ranks_in_group(type_name)

        return ranks_in_group[(local_rank - 1) % world_size]

    def get_next_global_rank(self, type_name = 'pipe'):
        local_rank = self.get_group_rank(type_name)
        world_size = self.get_group_size(type_name)
        ranks_in_group = self.get_ranks_in_group(type_name)

        return ranks_in_group[(local_rank + 1) % world_size]

    def is_mode_inited(self, type_name):
        return type_name in self._groups and self.get_group_size(type_name)>1

    def all_dp_ranks(self):
        return self._dp_ranks_all

global_context = ProcessTopology()

def is_using_pp():
    return global_context.is_mode_inited('pipe')

def test_comm():
    import torch
    tmp = torch.rand([100,1024]).cuda()
    torch.cuda.synchronize()
    dist.all_reduce(tmp, group=None)
    for mode in ['data', 'tensor', 'pipe', 'model']:
        if global_context.is_mode_inited(mode):
            dist.all_reduce(tmp, group=global_context.get_group(mode))
            torch.cuda.synchronize()

    len_dl_tensor = torch.tensor([0], dtype=torch.long).cuda()
    if global_context.is_first_in_group('model'):
        len_dl = 10
        len_dl_tensor = torch.tensor([len_dl], dtype=torch.long).cuda()

    dist.broadcast(len_dl_tensor,0)

    dist.broadcast(len_dl_tensor, global_context.get_ranks_in_group('model')[0], global_context.get_group('model'))
    torch.cuda.synchronize()

    outs = [torch.rand_like(tmp) for _ in range(global_context.get_group_size('tensor'))]
    dist.all_gather(outs, tmp, group=global_context.get_group('tensor'))
    torch.cuda.synchronize()


    if global_context.is_first_in_pipeline_group():
        dist.send(tmp, global_context.get_next_global_rank('pipe'))
    if global_context.is_last_in_pipeline_group():
        dist.recv(tmp, global_context.get_prev_global_rank('pipe'))
    torch.cuda.synchronize()
    dist.barrier()
    print("Finished test_comm --- ")
