#!/bin/bash
#SBATCH --job-name=selam_rnn
#SBATCH --error=rnn_errors.txt
#SBATCH --output=rnn_results.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2

module load Python/Anaconda_v11.2020
cd ..
cd rnn
srun python rnn.py