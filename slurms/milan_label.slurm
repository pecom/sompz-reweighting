#!/bin/bash
#SBATCH --job-name=shear-analysis         # create a short name for your job
#SBATCH -N 2                # node count
#SBATCH -n 80
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p short-40core
#SBATCH -o /gpfs/home/padari/logs/anacal/shear_analysis.%A.out
#SBATCH -e /gpfs/home/padari/logs/anacal/shear_analysis.%A.err


module purge
module load shared
module load slurm
# module load openmpi/4.0.1
# module load openmpi/gcc13.2/4.1.6
module load openmpi/gcc12.1/4.1.4
module load anaconda/3

# conda activate /gpfs/projects/VonDerLindenGroup/padari/pyenvs/xlens
conda activate /gpfs/projects/VonDerLindenGroup/padari/pyenvs/xlens

cd /gpfs/projects/VonDerLindenGroup/padari/som-pz

mpirun python lbl_unrec.py

