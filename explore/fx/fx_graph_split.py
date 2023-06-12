import copy

import torch
import torch.nn as nn
from torch import fx
from torch.fx import symbolic_trace
import statistics, tabulate, time
from typing import Any, Dict, List
from torch.fx import Interpreter


import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

import torchvision.models as models

class ProfilingInterpreter(Interpreter):
    def __init__(self, gm):
        super().__init__(gm)

        self.total_runtime_sec : List[float] = []
        self.runtimes_sec : Dict[torch.fx.Node, List[float]] = {}


    def run(self, *args) -> Any:
        # Record the time we started running the model
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        # Run the model by delegating back into Interpreter.run()
        return_val = super().run(*args)
        # Record the time we finished running the model
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        # Store the total elapsed time this model execution took in the
        # ProfilingInterpreter
        self.total_runtime_sec.append(t_end - t_start)
        self.module.__setattr__('time_cost', t_end - t_start)

        return return_val

    def run_node(self, n : torch.fx.Node) -> Any:
        # Record the time we started running the op
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        # Run the op by delegating back into Interpreter.run_node()
        return_val = super().run_node(n)
        # Record the time we finished running the op
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        # If we don't have an entry for this node in our runtimes_sec
        # data structure, add one with an empty list value.
        self.runtimes_sec.setdefault(n, [])
        # Record the total elapsed time for this single invocation
        # in the runtimes_sec data structure
        self.runtimes_sec[n].append(t_end - t_start)

        n.__setattr__('time_cost', t_end - t_start)
        return return_val

    ######################################################################
    # Finally, we are going to define a method (one which doesn't override
    # any ``Interpreter`` method) that provides us a nice, organized view of
    # the data we have collected.

    def summary(self, should_sort : bool = False) -> str:
        # Build up a list of summary information for each node
        node_summaries : List[List[Any]] = []
        # Calculate the mean runtime for the whole network. Because the
        # network may have been called multiple times during profiling,
        # we need to summarize the runtimes. We choose to use the
        # arithmetic mean for this.
        mean_total_runtime = statistics.mean(self.total_runtime_sec)

        # For each node, record summary statistics
        for node, runtimes in self.runtimes_sec.items():
            # Similarly, compute the mean runtime for ``node``
            mean_runtime = statistics.mean(runtimes)
            # For easier understanding, we also compute the percentage
            # time each node took with respect to the whole network.
            pct_total = mean_runtime / mean_total_runtime * 100
            # Record the node's type, name of the node, mean runtime, and
            # percent runtim
            node_summaries.append(
                [node.op, str(node), mean_runtime, pct_total])

        # One of the most important questions to answer when doing performance
        # profiling is "Which op(s) took the longest?". We can make this easy
        # to see by providing sorting functionality in our summary view
        if should_sort:
            node_summaries.sort(key=lambda s: s[2], reverse=True)

        # Use the ``tabulate`` library to create a well-formatted table
        # presenting our summary information
        headers : List[str] = [
            'Op type', 'Op', 'Average runtime (s)', 'Pct total runtime'
        ]
        return tabulate.tabulate(node_summaries, headers=headers)


rn18 = models.resnet18().cuda()
rn18.train()

input = torch.randn(32, 3, 224, 224).cuda()
output = rn18(input)


traced_rn18 = torch.fx.symbolic_trace(rn18)
# print(traced_rn18.graph)


interp = ProfilingInterpreter(traced_rn18)
traced_out = interp.run(input)
traced_out = interp.run(input)
traced_out = interp.run(input)
assert torch.allclose(output, traced_out)
# print(interp.summary(True))
# import pdb;pdb.set_trace()
total_time_cost = traced_rn18.time_cost
accum_time =0

# 这里切分的逻辑存在问题：无法处理像residual的结构，只能适用所有的部分都是单输入单输出

def split_gm_into2(gm, time_map, time_accum):
    accum_time =0
    first_gm = copy.deepcopy(gm)
    split_node_name = None
    new_output_name = None
    for node in first_gm.graph.nodes:
        accum_time += time_map.get(node.name,0)
        if accum_time>=time_accum and not split_node_name:
            # jump nodes which has multiple inputs
            if len(node.args)>1:
                continue
            else:
                new_output=first_gm.graph.output(node)
                split_node_name = node.name
                new_output_name = new_output.name

        if node.op=='output' and node.name != new_output_name:
            first_gm.graph.erase_node(node)
    first_gm.graph.eliminate_dead_code()
    first_gm.recompile()

    found = False
    sencond_gm = copy.deepcopy(gm)
    not_needed_nodes = []
    for node in sencond_gm.graph.nodes:
        if split_node_name != node.name and not found:
            # sencond_gm.graph.erase_node(node)
            not_needed_nodes.append(node)
        elif split_node_name == node.name:
            found = True
            new_input = sencond_gm.graph.placeholder(split_node_name)
            node.replace_all_uses_with(new_input)
            sencond_gm.graph.erase_node(node)
            break
    sencond_gm.graph.eliminate_dead_code()

    sencond_gm.recompile()
    return first_gm,sencond_gm

def get_splited_model(gm, partition_id:int, num_partitions:int):
    """
        partition_id: from 1, e.g. if num_partitions=3, partition_id={1,2,3}
    """
    total_time_cost = gm.time_cost
    time_map = {}
    for node in gm.graph.nodes:
        # import pdb;pdb.set_trace()
        time_map[node.name] = node.time_cost
    gms = split_gm_into2(gm, time_map, total_time_cost/num_partitions)
    return gms[partition_id]


gm=get_splited_model(traced_rn18, 1, 2)

print(gm.graph)