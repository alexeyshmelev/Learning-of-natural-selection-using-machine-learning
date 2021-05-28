#!/bin/bash
#SBATCH --job-name=tof_rnn
#SBATCH --error=tof_rnn_errors.txt
#SBATCH --output=tof_rnn_results.txt
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1

module load Python/Anaconda_v10.2019
cd ..
cd rnn
srun python rnn_tof.py