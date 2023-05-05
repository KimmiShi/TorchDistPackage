import os
import torch
import collections
from easydict import EasyDict as edict
import spring.linklink as link
from torch.nn import Module
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import numpy as np
import multiprocessing as mp

from collections import namedtuple
from itertools import product as cartesian_product

from core.utils import named_buffers, sync_print
from enum import Enum
from typing import Union, List, Tuple

from slurm_dist_init.launch_from_slurm import setup_distributed_slurm

_GLOBAL_TOPO = None
_DP_GROUP = None
_TP_GROUP = None
_PP_GROUP = None
_MP_GROUP = None

class ProcessTopology:
    """ Manages the mapping of n-dimensional Cartesian coordinates to linear
    indices. This mapping is used to map the rank of processes to the grid
    for various forms of parallelism.

    Each axis of the tensor is accessed by its name. The provided ordering
    of the axes defines the layout of the topology. ProcessTopology uses a "row-major"
    layout of the tensor axes, and so axes=['x', 'y'] would map coordinates (x,y) and
    (x,y+1) to adjacent linear indices. If instead axes=['y', 'x'] was used, coordinates
    (x,y) and (x+1,y) would be adjacent.

    Some methods return ProcessCoord namedtuples.
    """
    def __init__(self, axes, dims):
        """Create a mapping of n-dimensional tensor coordinates to linear indices.

        Arguments:
            axes (list): the names of the tensor axes
            dims (list): the dimension (length) of each axis of the topology tensor
        """

        self.axes = axes  # names of each topology axis
        self.dims = dims  # length of each topology axis


        # This is actually a class that lets us hash {'row':3, 'col':2} mappings
        self.ProcessCoord = namedtuple('ProcessCoord', axes)

        self.mapping = {}

        ranges = [range(d) for d in dims]
        # example: 1, (0,0,1)
        for global_rank, coord in enumerate(cartesian_product(*ranges)):
            key = {axis: coord[self.axes.index(axis)] for axis in self.axes}
            key = self.ProcessCoord(**key)
            # for example, {ProcessCoord(row=0, col=1) : 1}
            self.mapping[key] = global_rank

    def get_rank(self, **coord_kwargs):
        """Return the global rank of a process via its coordinates.

        Coordinates are specified as kwargs. For example:

            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_rank(x=0, y=1)
            1
        """
        if len(coord_kwargs) != len(self.axes):
            raise ValueError('get_rank() does not support slices. Use filter_match())')

        key = self.ProcessCoord(**coord_kwargs)
        assert key in self.mapping, f'key {coord_kwargs} invalid'
        return self.mapping[key]

    def get_axis_names(self):
        """Return a list of the axis names in the ordering of the topology. """
        return self.axes

    def get_rank_repr(self,
                      rank,
                      omit_axes=['data',
                                 'pipe'],
                      inner_sep='_',
                      outer_sep='-'):
        """Return a string representation of a rank.

        This method is primarily used for checkpointing model data.

        For example:
            >>> topo = Topo(axes=['a', 'b'], dims=[2, 2])
            >>> topo.get_rank_repr(rank=3)
            'a_01-b_01'
            >>> topo.get_rank_repr(rank=3, omit_axes=['a'])
            'b_01'

        Args:
            rank (int): A rank in the topology.
            omit_axes (list, optional): Axes that should not be in the representation. Defaults to ['data', 'pipe'].
            inner_sep (str, optional): [description]. Defaults to '_'.
            outer_sep (str, optional): [description]. Defaults to '-'.

        Returns:
            str: A string representation of the coordinate owned by ``rank``.
        """
        omit_axes = frozenset(omit_axes)
        axes = [a for a in self.get_axis_names() if a not in omit_axes]
        names = []
        for ax in axes:
            ax_rank = getattr(self.get_coord(rank=rank), ax)
            names.append(f'{ax}{inner_sep}{ax_rank:02d}')
        return outer_sep.join(names)

    def get_dim(self, axis):
        """Return the number of processes along the given axis.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_dim('y')
            3
        """
        if axis not in self.axes:
            return 0
        return self.dims[self.axes.index(axis)]

    def get_coord(self, rank):
        """Return the coordinate owned by a process rank.

        The axes of the returned namedtuple can be directly accessed as members. For
        example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> coord = X.get_coord(rank=1)
            >>> coord.x
            0
            >>> coord.y
            1
        """
        for coord, idx in self.mapping.items():
            if idx == rank:
                return coord
        raise ValueError(f'rank {rank} not found in topology.')

    def get_axis_comm_lists(self, axis):
        """ Construct lists suitable for a communicator group along axis ``axis``.

        Example:
            >>> topo = Topo(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> topo.get_axis_comm_lists('pipe')
            [
                [0, 4], # data=0, model=0
                [1, 5], # data=0, model=1
                [2, 6], # data=1, model=0
                [3, 7], # data=1, model=1
            ]

        Returns:
            A list of lists whose coordinates match in all axes *except* ``axis``.
        """

        # We don't want to RuntimeError because it allows us to write more generalized
        # code for hybrid parallelisms.
        if axis not in self.axes:
            return []

        # Grab all axes but `axis`
        other_axes = [a for a in self.axes if a != axis]

        lists = []

        # Construct all combinations of coords with other_axes
        ranges = [range(self.get_dim(a)) for a in other_axes]
        for coord in cartesian_product(*ranges):
            other_keys = {a: coord[other_axes.index(a)] for a in other_axes}
            # now go over all ranks in `axis`.
            sub_list = []
            for axis_key in range(self.get_dim(axis)):
                key = self.ProcessCoord(**other_keys, **{axis: axis_key})
                sub_list.append(self.mapping[key])
            lists.append(sub_list)

        return lists

    def filter_match(self, **filter_kwargs):
        """Return the list of ranks whose coordinates match the provided criteria.

        Example:
            >>> X = ProcessTopology(axes=['pipe', 'data', 'model'], dims=[2, 2, 2])
            >>> X.filter_match(pipe=0, data=1)
            [2, 3]
            >>> [X.get_coord(rank) for rank in X.filter_match(pipe=0, data=1)]
            [ProcessCoord(pipe=0, data=1, model=0), ProcessCoord(pipe=0, data=1, model=1)]

        Arguments:
            **filter_kwargs (dict): criteria used to select coordinates.

        Returns:
            The list of ranks whose coordinates match filter_kwargs.
        """
        def _filter_helper(x):
            for key, val in filter_kwargs.items():
                if getattr(x, key) != val:
                    return False
            return True

        coords = filter(_filter_helper, self.mapping.keys())
        return [self.mapping[coord] for coord in coords]

    def get_axis_list(self, axis, idx):
        """Returns the list of global ranks whose coordinate in an axis is idx.

        For example:
            >>> X = ProcessTopology(axes=['x', 'y'], dims=[2,3])
            >>> X.get_axis_list(axis='x', idx=0)
            [0, 1, 2]
            >>> X.get_axis_list(axis='y', idx=0)
            [0, 3]
        """

        # This could be faster by generating the desired keys directly instead of
        # filtering.
        axis_num = self.axes.index(axis)
        ranks = [self.mapping[k] for k in self.mapping.keys() if k[axis_num] == idx]
        return ranks

    def world_size(self):
        return len(self.mapping)

    def __str__(self):
        return str(self.mapping)

    @classmethod
    def get_topo(cls, axes=None, dims=None):
        global _GLOBAL_TOPO
        if _GLOBAL_TOPO is None:
            _GLOBAL_TOPO = cls(axes, dims)
        return _GLOBAL_TOPO


    def get_global_rank(self):
        return dist.get_rank()


    def create_group(self, type_name):
        assert type_name in self.axes

        if type_name == 'data':
            global _DP_GROUP
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
            rank = self.get_global_rank()
            for l in comm_lists:
                if rank in l:
                    _DP_GROUP = torch.distributed.new_group(l)

        if type_name == 'tensor':
            global _TP_GROUP
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
            rank = self.get_global_rank()
            for l in comm_lists:
                if rank in l:
                    _TP_GROUP = torch.distributed.new_group(l)

        if type_name == 'pipe':
            global _PP_GROUP
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
            rank = self.get_global_rank()
            for l in comm_lists:
                if rank in l:
                    _PP_GROUP = torch.distributed.new_group(l)

        if type_name == 'model':
            global _MP_GROUP
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
            rank = self.get_global_rank()
            for l in comm_lists:
                if rank in l:
                    _MP_GROUP = torch.distributed.new_group(l)

    def get_group_rank(self, type_name):
        assert type_name in self.axes
        if type_name in ['tensor', 'pipe', 'model']:
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
        else:
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
        rank = self.get_global_rank()
        for l in comm_lists:
            if rank in l:
                return l.index(rank)

    def get_group_ranks(self, type_name):
        assert type_name in self.axes
        if type_name in ['tensor', 'pipe', 'model']:
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
        else:
            comm_lists = self.get_topo().get_axis_comm_lists(type_name)
        rank = self.get_global_rank()
        for l in comm_lists:
            if rank in l:
                return l

    def get_tp_rank(self):
        return self.get_group_rank('tensor')

    def get_pp_rank(self):
        return self.get_group_rank('pipe')

    def get_dp_rank(self):
        return self.get_group_rank('data')

    def get_mp_rank(self):
        return self.get_group_rank('model')

    def get_group_size(self, type_name):
        assert type_name in self.axes
        comm_lists = self.get_topo().get_axis_comm_lists(type_name)
        rank = self.get_global_rank()
        for l in comm_lists:
            if rank in l:
                return len(l)

    def get_tp_size(self):
        return self.get_group_size('tensor')

    def get_pp_size(self):
        return self.get_group_size('pipe')

    def get_dp_size(self):
        return self.get_group_size('data')

    def get_mp_size(self):
        return self.get_group_size('model')

    def is_first_in_group(self, type_name):
        assert type_name in self.axes
        comm_lists = self.get_topo().get_axis_comm_lists(type_name)
        rank = self.get_global_rank()
        for l in comm_lists:
            if rank in l:
                if rank == l[0]:
                    return True
                break
        return False

    def is_last_in_group(self, type_name):
        assert type_name in self.axes
        comm_lists = self.get_topo().get_axis_comm_lists(type_name)
        rank = self.get_global_rank()
        for l in comm_lists:
            if rank in l:
                if rank == l[-1]:
                    return True
                break
        return False

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
        assert type_name in self.axes
        ranks = self.get_group_ranks(type_name)
        rank = self.get_group_rank(type_name)
        idx = ranks.index(rank) - 1

    def get_next_global_rank(self, type_name = 'pipe'):
        assert type_name in self.axes
        ranks = self.get_group_ranks(type_name)
        rank = self.get_group_rank(type_name)
        idx = ranks.index(rank) + 1

def launch_from_slurm(config):
    """
    Usage:
        CONFIG = { 'TOPO' : [
            'data': {'SIZE': 4},
            'pipe': {'SIZE': 2},
            ]
        }
        launch_from_slurm(CONFIG)
    """
    setup_distributed_slurm()
    axes = []
    dims = []
    for idx, item in enumerate(config['TOPO']):
        axes.append(item["TYPE"])
        dims.append(int(item["SIZE"]))
    topo = ProcessTopology.get_topo(axes, dims)
    assert dist.is_initialized()
    for axe in axes:
        topo.create_group(axe)


def global_topo():
    global _GLOBAL_TOPO
    return _GLOBAL_TOPO
