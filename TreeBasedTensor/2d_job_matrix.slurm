#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --output=out/2d_matrix_%A_%a.out
#SBATCH --account=m4293

eval "$(conda shell.bash hook)"
conda activate venv
srun python3 main.py --verbose --logN $SLURM_ARRAY_TASK_ID --dimens 2 --asMatrix
