#!/bin/bash -l
#
#SBATCH --job-name="dqn"
#SBATCH --output=dqn.out
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
# make sure to add your account!
#SBATCH --account=education-me-msc-sc

module load 2023r1
module load python
module load openmpi
module load py-pytorch
module load py-numpy
module load py-scipy
module load py-h5py
module load py-matplotlib

srun python dqn.py