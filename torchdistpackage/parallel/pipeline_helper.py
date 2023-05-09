from torchdistpackage import tpc

def partition_uniform(flat_sequence, extra_len=0):
    rank = tpc.get_group_rank('pipe')
    world_size = tpc.get_group_size('pipe')
    leng = len(flat_sequence) + extra_len
    length = leng // world_size
    beg = rank*length
    end = (rank+1)*length if rank!=world_size-1 else len(flat_sequence)
    kept_flat_sequence = flat_sequence[beg:end]
    return kept_flat_sequence