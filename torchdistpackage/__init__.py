from .ddp.naive_ddp import NaiveDDP, moe_dp_iter_step, create_moe_dp_hooks
from .ddp.zero_optim import Bf16ZeroOptimizer

# from .ddp.torch_py_ddp import PythonDDP

from .dist.launch_from_slurm import setup_distributed_slurm

from .dist.process_topo import torch_parallel_context as tpc
from .dist.process_topo import test_comm, is_using_pp
from .dist.node_group import setup_node_groups
from .dist.sharded_ema import ShardedEMA


from .utils import fix_rand, sliced_run

from .tools.module_profiler import report_prof, register_profile_hooks, get_model_profile

from .tools.module_replace import replace_all_module
from .tools.bnb_fc import replace_linear_by_bnb
from .tools.bminf_int8 import replace_linear_by_bminf
from .tools.debug_nan import register_debug_nan_hooks,check_model_params