# this file include functions for saving/loading checkpoints for Model Parallelism
from torchdistpackage import tpc

def get_mp_ckpt_suffix():
    """
        get the suffix of state_dict filename, in format: f"_tp_{tp_rank}_pp_{pp_rank}.pth"
        example:
            "_tp_1_pp_2.pth", mean this partition is Rank 1 in TensorParallel and Rank 2 in PipelineParallel
    """
    fname = ""

    if is_mode_inited('tensor'):
        tp_rank = tpc.get_group_rank("tensor")
        fname += f"_tp_{tp_rank}"
    if is_mode_inited('pipe'):
        pp_rank = tpc.get_group_rank("pipe")
        fname += f"_pp_{pp_rank}"

    fname += f".pth"

    return fname
