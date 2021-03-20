#!/bin/bash
#SBATCH --job-name=next_selam
#SBATCH --error=next_gen_selam_errors.txt
#SBATCH --output=next_gen_selam_results.txt
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --deadline=01:20:00

module load Python/Anaconda_v10.2019
module load libs/gsl/2.6
export PATH=$PATH:/home/avshmelev/SELAM-master/SELAM-master/src
cd ..
cd selam
srun python next_gen_selam.py
