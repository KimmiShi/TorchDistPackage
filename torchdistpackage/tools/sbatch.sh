#!/bin/bash

#SBATCH --partition=DEV
#SBATCH --ntasks=16
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1


srun python torchdistpackage/dist/py_comm_test.py