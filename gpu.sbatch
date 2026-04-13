#!/usr/bin/env bash

#SBATCH --partition=hopper
#SBATCH --job-name=llm-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=00:01:00
#SBATCH --output=gpu.out

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

module load gcc-native/12.3
module load nvidia-mixed/24.3
make train_gpt2cu PRECISION=FP32
./train_gpt2cu \
    -e "gpt2_124M.bin" \
    -b 1 -t 1024 \
    -l 0.0 \
    -s 1 -g 1024 \
    -h 0 -x 3 \
    -f 0 