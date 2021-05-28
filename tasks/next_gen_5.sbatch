#!/bin/bash
#SBATCH --job-name=next_gen_5
#SBATCH --error=next_gen_5_errors.txt
#SBATCH --output=next_gen_5_results.txt
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

module load Python/Anaconda_v10.2019
module load libs/gsl/2.6
export PATH=$PATH:/home/avshmelev/SELAM-master/SELAM-master/src
cd ..
cd selam
cd num_5
srun python tof_simulation.py