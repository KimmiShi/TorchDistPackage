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
