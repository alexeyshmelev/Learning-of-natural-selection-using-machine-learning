#!/bin/bash
#SBATCH --job-name=next_gen_2
#SBATCH --error=next_gen_2_errors.txt
#SBATCH --output=next_gen_2_results.txt
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

module load Python/Anaconda_v10.2019
module load libs/gsl/2.6
export PATH=$PATH:/home/avshmelev/SELAM-master/SELAM-master/src
cd ..
cd selam
cd num_2
srun python simulation.py