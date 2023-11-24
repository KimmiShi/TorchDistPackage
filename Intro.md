# Pytorch 并行训练加速库介绍
Repo: [PyTorch Distributed Training Package](https://github.com/KimmiShi/TorchDistPackage)； 欢迎大家加星~

目前比较受欢迎的加速库例如 [ColossalAI](https://github.com/hpcaitech/ColossalAI), [DeepSpeed](https://www.deepspeed.ai/) 已经比较好用了，为什么还需要自己写一个呢？

主要有以下原因：
1. API方面：需要更灵活的可分割的组件，可以在不同的项目中选用其中一部分
2. 功能方面：补充一些新的实现


因此，本仓库并不着力全部重写已有的功能，而是在作者为大模型训练做分布式加速的过程中遇到部分需要重写的/改进的部分会被整理收集到该仓库。

# 特性介绍

## 1. 灵活的通信组划分
当我们在处理混合并行(DDP,TP,PP,MOE)的时候，避免不了初始化不同的通信组。这里作者对ColossalAI使用较多，但也发现不够灵活，例如我经过分析，发现DDP通信组的通信需求最大，于是想试一下将DDP通信组放在最内层（即尽可能的少跨节点、跨交换机），但初始化的代码似乎已经固定了通信组的顺序（最新的代码作者不保证是否发生了更改），于是这里提供了一套可以灵活改变(DDP,TP,PP）三种通信组拓扑结构的通信组初始化代码。而且还会根据`Model Parallel`的语义（即对同一份数据进行计算的模型并行组）会自动构建MP的通信组，可以使用该特性方便的进行组内数据广播。

代码用法示例：

初始化 DP,TP,PP 三个通信组，并自动初始化`Model Parallel` 通信组。

```py
from torchdistpackage import tpc

# world_size=16
# pp_size=2

dist_config = [('pipe',pp_size), ('tensor',2), ('data',world_size/(2*pp_size))]
tpc.setup_process_groups(dist_config)
```

上面的代码会把DP组放在节点内，pp组（Pipeline Parallel Group）放在跨节点，一个直观的log：
```
group pipe, ranks: [0, 8]
group pipe, ranks: [1, 9]
...
group tensor, ranks: [0, 4]
group tensor, ranks: [8, 12]
...
group data, ranks: [0, 1, 2, 3]
group data, ranks: [4, 5, 6, 7]
...
group model, ranks: [0, 4, 8, 12]
...
```

> 对于

如果想改变通信组的顺序，希望DP组在最外层（跨节点），只需要改变config为：
```
dist_config = [('data',world_size/(2*pp_size)), ('pipe',pp_size), ('tensor',2)]
```

## 2. 灵活的pipeline parallel调度器
> 本章节代码主要参考 [ColossalAI](https://github.com/hpcaitech/ColossalAI)

对于最简单的nn.Sequential类型的模型，ColossalAI的代码已经可以满足需求了；
这里提出的pipeline parallel调度器主要是为了处理非线性结构的模型（即不同pipeline stage的前向代码不一致，或者输入数量不同），例如[CLIP](https://openai.com/research/clip)，我们对model的划分方式是在最后一个stage做text部分以及剩下的image部分，所以就不能直接用原先的代码。

这里做出的改动是：
- 将每个stage的forward, backward函数作为参数传入，由用户定义
- 提供 `forward_backward` 接口，等价替换原先训练代码中的 `fwd+bwd` 部分，不引入 `engine` 等新的结构

使用例子可以参考 [pipeline scheduler 测试代码](./torchdistpackage/parallel/test_pipeline.py)

> 目前仅迁移了1F1B调度器


## 加速ZeRO多卡训练速度

facts:
1. 随着卡数增加，zero的通信开销越来越大
2. 超过一定卡数后，卡数继续增加，zero对显存的削减作用变小

举个例子，例如opt_state=40GB; 8卡的时候每张卡5GB；32卡的时候每张卡1.xGB，对于80GB级别的显卡来说，省下的这4GB显存无关痛痒，但由于通信跨节点了，zero最后一步all-gather(或者broadcast)会变慢很多，尤其是只有一两个IB的机器。

所以解决方案是：只在节点内做ZeRO。

目前提供了[ZeRO1的实现](./torchdistpackage/dist/node_group.py)。