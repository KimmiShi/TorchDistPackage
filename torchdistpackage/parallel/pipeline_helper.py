import torch
import torch.nn as nn
from torchdistpackage import tpc


def partition_uniform(flat_sequence, extra_len=0) -> list:
    """
        accept a list and return a list of Modules
    """
    rank = tpc.get_group_rank("pipe")
    world_size = tpc.get_group_size("pipe")
    leng = len(flat_sequence) + extra_len
    length = leng // world_size
    beg = rank * length
    end = (rank + 1) * length if rank != world_size - 1 else len(flat_sequence)
    kept_flat_sequence = flat_sequence[beg:end]
    return kept_flat_sequence


def _heap_addition(weights: list, intervals: list, add_cnt: int):
    import heapq

    def _heap_push(heap, st, ed):
        value = weights[ed - 1]
        if st > 0:
            value -= weights[st - 1]
        heapq.heappush(heap, (-value, st, ed))

    ret_intervals = []
    heap = []

    for st, ed in intervals:
        _heap_push(heap, st, ed)

    while add_cnt > 0:
        _, st, ed = heapq.heappop(heap)
        if ed - st == 1:
            ret_intervals.append((st, ed))
        else:
            l, m, r = _binary_partition(weights, st, ed)
            _heap_push(heap, l, m)
            _heap_push(heap, m, r)
            add_cnt -= 1

    while heap:
        _, st, ed = heapq.heappop(heap)
        ret_intervals.append((st, ed))

    ret_intervals.sort()
    return ret_intervals


def _calc_partitions(weights, value):
    prev = 0
    prefix = 0
    num_block = 0
    intervals = []

    for idx, w in enumerate(weights):
        if weights[idx] - prefix > value:
            intervals.append((prev, idx))
            prev = idx
            prefix = weights[idx - 1]
            num_block += 1

    intervals.append((prev, len(weights)))
    return num_block + 1, intervals


def _binary_search(weights, num):
    length = len(weights)
    prefix = [1 if w == 0 else w for w in weights]
    for i in range(1, length):
        prefix[i] += prefix[i - 1]

    lower_bound = max(weights)
    upper_bound = prefix[length - 1]

    while upper_bound > lower_bound:
        mid = (upper_bound + lower_bound) // 2
        number, _ = _calc_partitions(prefix, mid)
        if number <= num:
            upper_bound = mid
        else:
            lower_bound = mid + 1

    num_block, intervals = _calc_partitions(prefix, upper_bound)
    if num_block < num:
        intervals = _heap_addition(prefix, intervals, num - num_block)

    return intervals


def partition_balanced(flat_sequence, sequence, **kwargs):
    rank = tpc.get_group_rank("pipe")
    world_size = tpc.get_group_size("pipe")
    assert len(flat_sequence) >= world_size

    def count_params(model):
        param_count = 0
        for param in model.parameters():
            param_count += param.numel()
        if param_count == 0:
            param_count = 1
        return param_count

    weight_sequence = [count_params(item) for item in flat_sequence]
    del count_params
    intervals = _binary_search(weight_sequence, world_size)
    sequence = flat_sequence[intervals[rank][0] : intervals[rank][1]]
    return sequence


def flatten_sequence(sequence, level=1):
    """
        Flatten a give model of type nn.Sequential, since the child module maybe nn.Sequential
    """
    if level == 0:
        if isinstance(sequence, list):
            return sequence
        elif isinstance(sequence, torch.nn.Sequential):
            return [i for i in sequence]
        else:
            return [sequence]
    res = []
    for element in sequence:
        res += flatten_sequence(element, level - 1)
    return res


def flatten_model(model, layer_list, return_list=False):
    """
        flatten a model that is not a nn.Sequential, but according to a list of layer name
        Example:
            exec_seq = [
                'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool',
                lambda x: torch.flatten(x, 1), 'fc'
            ]
            seq_model = flatten_model(model, exec_seq)
    """
    module_list = []
    for layer_name in layer_list:
        if isinstance(layer_name, str):
            sub_mod = getattr(model, layer_name)
            if isinstance(sub_mod, torch.nn.modules.container.Sequential) or isinstance(
                sub_mod, torch.nn.ModuleList
            ):
                for op in sub_mod:
                    module_list.append(op)
            else:
                module_list.append(sub_mod)
        elif isinstance(layer_name, torch.nn.Module):
            module_list.append(layer_name)
        elif callable(layer_name):
            # maybe lambda, or functions, this maynot work for nn.Sequential
            module_list.append(layer_name)
        else:
            # unknown
            print("flatten_model do not support layer: ", layer_name, type(layer_name))
            raise NotImplementedError()
    if return_list:
        return module_list
    return nn.Sequential(*module_list)


def flat_and_partition(sequence, flat_level=1, partition_policy="uniform", **kwargs):
    flattened = flatten_sequence(sequence, flat_level)
    partition_fn = eval(f"partition_{partition_policy}")
    cur_partition = partition_fn(flattened, **kwargs)
    return cur_partition


class ListModule(torch.nn.Module):
    """
        wraps a list of [Module, callable] into a nn.Module
    """

    def __init__(self, modules):
        super(ListModule, self).__init__()
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
