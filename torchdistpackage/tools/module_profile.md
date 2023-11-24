# model profile / flops profile BEST Practice

# FLOPS profile
两种思路：
- 基于module hook
    - 优点：兼容性好；结果可读性好
    - 缺点：存在不支持的op；需要扩展代码
- 基于fx
    - 优点：捕捉的op比较全
    - 缺点：兼容性不好，有些model trace会出问题

对于较大的模型来说，影响FLOPS统计的主要是矩阵乘，其他的一些element-wise的算子没统计到一般也不会对FLOPS结果产生大的影响。

推荐使用 deepspeed profiler，它不仅基于module hook，而且还patch了一些常用的function & method, 使得大多数都能统计到。

推荐版本：https://github.com/microsoft/DeepSpeed/pull/4724

- 增加处理 `F.scaled_dot_product_attention` in transformer models.
- 增加处理 expreesions like `a@b`

如果有一些自定义的kernel，flops比较大不想漏掉，可以通过相同的方式来patch。

# module profile： 以module为粒度统计时间和显存开销，指导grad checkpointing

这里推荐使用 [torchdistpackage.get_model_profile](../../torchdistpackage/tools/module_profile.md)

[测例用法](./test_profile.py)参考

优点：
- 分层级打印submodule的时间和显存消耗
- 可以选择计算 `显存消耗/时间消耗` 排序，以指导`grad checkpointing`的位置

缺陷：
- 分级有时候不太准，和模型的写法有关系。。。

resnet例子：

```

level: 0
root: MEM: 384 MB; Time: 29.402 ms

level: 1
conv1: MEM: 11 MB; Time: 1.1326 ms
bn1: MEM: 51 MB; Time: 0.8379 ms
relu: MEM: 0 MB; Time: 0.1888 ms
maxpool: MEM: 105 MB; Time: 0.6408 ms
layer1: MEM: 103 MB; Time: 7.1421 ms
layer2: MEM: 64 MB; Time: 6.193 ms
......
avgpool: MEM: 2 MB; Time: 0.1494 ms
fc: MEM: 0 MB; Time: 0.1812 ms

level: 2
layer1.0.conv1: MEM: 13 MB; Time: 0.7857 ms
layer1.0.bn1: MEM: 13 MB; Time: 0.3004 ms
layer1.0.relu: MEM: 0 MB; Time: 0.1253 ms
layer1.0.conv2: MEM: 13 MB; Time: 0.5398 ms
layer1.0.bn2: MEM: 13 MB; Time: 0.2897 ms
layer1.1.conv1: MEM: 13 MB; Time: 0.5373 ms
......

```