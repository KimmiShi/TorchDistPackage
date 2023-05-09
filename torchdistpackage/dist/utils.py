import ctypes
import torch
from torch import nn
import math
import warnings
from torch.nn import init


_cudart = ctypes.CDLL('libcudart.so')

def cu_prof_start():
    """
    This function and cu_prof_stop  are two functions used to do multi process Nsight Profiling.
    Example::
        if self.iter == 20 and torch.distributed.get_rank() == 0:
            cu_prof_start()
        if self.iter >= 40:
            cu_prof_stop()
            exit()
    """
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception('cudaProfilerStart() returned %d' % ret)
    else:
        print('cudaProfilerStart() returned %d' % ret)

def cu_prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception('cudaProfilerStop() returned %d' % ret)
    else:
        print('cudaProfilerStop() returned %d' % ret)
        exit()

def nvtx_decorator(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""

    def wrapped_fn(*args, **kwargs):
        torch.cuda.nvtx.range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return ret_val

    return wrapped_fn

class NVTX_Context(object):
    """ A simple context used to record performance info.
    """

    def __init__(self, context_name, record_time=False):
        self.context_name = context_name
        self.start_time = None
        self.exit_time = None
        self.record_time = record_time
        self.synchronize = False
        
    def __enter__(self):
        torch.cuda.nvtx.range_push(self.context_name)
        if self.record_time
            torch.cuda.synchronize()
            self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        if self.record_time
            torch.cuda.synchronize()
            self.exit_time = time.time()
            print(f"{context_name} duration is {self.exit_time - self.start_time}")
        torch.cuda.nvtx.range_pop()
    
def _has_inf_or_nan(x, j=None):
    try:
        # if x is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as x
        # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # cpu_sum = float(x.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        return False