# varuna
link: https://arxiv.org/pdf/2111.04007.pdf

- 适用低带宽集群
- 适用于利用大集群空闲时间，弹性训练

highlight:
- pipeline parallel: 不同于一般做法的尽可能使用小的pipeline stage数量(减少pipeline bubble)，论文提出随着模型增大，增大pipeline stage数量，并增加micro-batch数量，来减少DDP通信的开销
  - 对于micro-batch-num增大带来的batchsize增大，可能对精度造成损失的问题，作者用16倍大的batchsize训练了一个2.5B的GPT2
- 弹性训练：job morphing 
