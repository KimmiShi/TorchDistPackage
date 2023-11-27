import torch
import os

def fix_rand(rank=0):
    import numpy as np
    import random

    seed = 2222 + rank
    print(f"Setting random seed to {seed}")

    # PyTorch random number generator (for cpu and cuda)
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    # python random
    random.seed(seed)

    # numpy RNG
    np.random.seed(seed)

    # cuda benchmarking
    torch.backends.cudnn.benchmark = False

    # deterministic algos
    # torch.use_deterministic_algorithms(True)

    # cudnn conv deterministic
    torch.backends.cudnn.deterministic = True

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.random.manual_seed(seed)

def partition_params(model, num_partitions, return_dict=False):
    """partitions params

    Args:
        model (list): the model
        num_partitions (int): zero dp world size
        numel_per_partition (int): max number of param cnt

    Returns:
        list: list of partitions
    """
    partitions = []
    elcnt = 0
    partition_id = 0
    numel_per_partition = sum([p.numel() for p in model.parameters()]) // num_partitions
    for ind in range(num_partitions):
        if return_dict:
            partitions.append({})
        else:
            partitions.append([])

    for name, param in model.named_parameters():
        if return_dict:
            partitions[partition_id][name] = param
        else:
            partitions[partition_id].append(param)
        elcnt+=param.numel()
        if elcnt > numel_per_partition:
            partition_id+=1
            elcnt=0
    return partitions