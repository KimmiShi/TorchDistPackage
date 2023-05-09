import os
import torch
import torch.nn as nn

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import subprocess
import time

from torchdistpackage import tpc, setup_distributed_slurm,test_comm
from torchdistpackage.parallel.pipeline_helper import partition_uniform
from torchdistpackage.parallel.pipeline_sched import forward_backward

class DummyClsDataset(Dataset):
    def __init__(self, shape, num_samples=1000):
        self.num_samples = num_samples
        self.shape = shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        del idx

        img = torch.randn(self.shape)
        label = torch.randint(0, 10, (1,)).squeeze()

        return img, label

def get_dataloaders(
        batch_size,
        batch_shape,
        train_samples,
        test_samples,
    ) -> Tuple[DataLoader, DataLoader]:
    """ get dataloaders """
    train_data = DummyClsDataset(batch_shape, train_samples)
    sampler = DistributedSampler(train_data, shuffle=True)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=0, # multi-workers?
        pin_memory=True,
        sampler=sampler,
    )

    test_dataloader = DataLoader(
        DummyClsDataset(batch_shape, test_samples),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_dataloader, test_dataloader

def build_seq_model():
    model = nn.Sequential(
          nn.Linear(10,10),
          nn.ReLU(),
          nn.Linear(10,10),
          nn.ReLU(),
          nn.ReLU(),
        )
    return model

# Define some config
BATCH_SIZE = 512
NUM_EPOCHS = 2
NUM_CHUNKS = 2
NUM_MICRO_BATCHES=4


setup_distributed_slurm()
world_size = int(os.environ["SLURM_NTASKS"])
pp_size=2
dist_config = [('pipe',pp_size), ('data',world_size/(pp_size))]
tpc.setup_process_groups(dist_config)

model = build_seq_model().cuda()
model = nn.Sequential(*partition_uniform(model)).cuda()


# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# build dataloader
root = os.environ.get('DATA', './data')
train_dataloader, test_dataloader = get_dataloaders(BATCH_SIZE, [10], 4000, 100)
test_dataloader=None

def pp_fwd_fn(ins):
    input_img = ins
    img_feat = model(input_img)

    if tpc.is_last_in_pipeline_group():
        loss = img_feat.mean()
        return loss

    return img_feat

def pp_bwd_fn(output_obj):
    output_obj.backward()

def local_forward_backward(inp, model):
    inputs = []
    if tpc.is_first_in_pipeline_group():
        inputs.append(inp)

    forward_backward(
        optimizer,
        pp_fwd_fn,
        pp_bwd_fn,
        inputs,
        num_microbatches=NUM_MICRO_BATCHES,
        forward_only=False,
        dtype=torch.float32,
        scatter_gather_tensors=False)



for epoch in range(NUM_EPOCHS):

   for img, label in train_dataloader:
        img = img.cuda()
        label = label.cuda()

        local_forward_backward(img, model)
        optimizer.step()
        print("-------------")