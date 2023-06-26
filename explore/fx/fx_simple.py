import torch
import torch.nn as nn
from torch import fx
from torch.fx import symbolic_trace

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, input):
        out = self.fc1(input)
        return self.fc2(out)

module = MyModule()
# 符号追踪这个模块
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# symbolic_traced.graph.print_tabular()

# # 中间表示
# # High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced.graph)

# # 生成代码
# # Code generation - valid Python code
# print(symbolic_traced.code)

def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        print(node.op, node.name, node.target)
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul
        elif node.op == 'call_module':
            import pdb;pdb.set_trace()
            print("module")
            # if node.starget =
        elif node.op == 'output':
            import pdb;pdb.set_trace()
            print('output')

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)

transform(module, torch.fx.Tracer)