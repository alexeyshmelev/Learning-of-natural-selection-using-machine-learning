#!/bin/bash
#SBATCH --job-name=next_gen_10
#SBATCH --error=next_gen_10_errors.txt
#SBATCH --output=next_gen_10_results.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

module load Python/Anaconda_v10.2019
module load libs/gsl/2.6
export PATH=$PATH:/home/avshmelev/SELAM-master/SELAM-master/src
cd ..
cd selam
cd num_10
srun python simulation.py