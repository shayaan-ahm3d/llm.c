#!/usr/bin/env bash

#SBATCH --partition=ampere
#SBATCH --job-name=llm-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --output=cpu.out

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

OMP_NUM_THREADS=8 BLIS_NUM_THREADS=4 OMP_PROC_BIND=true OMP_PLACES=cores ./train_gpt2 -h 0 -x 3 -g 1024 -l 0.0