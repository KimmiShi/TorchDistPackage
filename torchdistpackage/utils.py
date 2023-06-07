def fix_random_seed():
    import torch

    torch.manual_seed(0)
    import random

    random.seed(0)
    import numpy as np

    np.random.seed(0)

    torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms(True)
