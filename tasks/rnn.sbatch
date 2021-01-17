#!/bin/bash
#SBATCH --job-name=selam_rnn
#SBATCH --error=rnn_errors.txt
#SBATCH --output=rnn_results.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --deadline=00:06:00

module load Python/Anaconda_v10.2019
cd ..
cd rnn
srun python rnn.py