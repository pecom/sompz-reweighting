#!/bin/bash
#SBATCH --job-name=shear-analysis         # create a short name for your job
#SBATCH -N 1                # node count
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p short-40core
#SBATCH -o /gpfs/home/padari/logs/unrecpz/reweight.%A.out
#SBATCH -e /gpfs/home/padari/logs/unrecpz/reweight.%A.err


module purge
module load shared
module load slurm


# SEAWULF
# module load openmpi/4.0.1
# module load openmpi/gcc13.2/4.1.6

# MILAN
module load openmpi/gcc12.1/4.1.4


module load anaconda/3
# conda activate /gpfs/projects/VonDerLindenGroup/padari/pyenvs/xlens
conda activate /gpfs/projects/VonDerLindenGroup/padari/pyenvs/unrec-pz

cd /gpfs/projects/VonDerLindenGroup/padari/som-pz

python run_som.py

