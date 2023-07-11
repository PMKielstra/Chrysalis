#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --output=2d_%A_%a.out

module load python
source ./venv/bin/activate
srun python3 main.py --logN $SLURM_ARRAY_TASK_ID 