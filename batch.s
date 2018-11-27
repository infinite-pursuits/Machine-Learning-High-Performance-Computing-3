#!/usr/bin/env bash

#SBATCH --job-name=lab3
#SBATCH --output=/scratch/cy1235/HPC/lab3/stdout.log
#SBATCH --error=/scratch/cy1235/HPC/lab3/stderr.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=chhavi@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu

source ~/.bash_profile

module purge
module load cuda/9.0.176
module load cudnn/9.0v7.0.5

pushd /scratch/cy1235/HPC/lab3

make clean
make lab3 && ./lab3
