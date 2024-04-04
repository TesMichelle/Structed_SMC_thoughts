#!/bin/bash
#SBATCH --job-name="compute_density"
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:0
#SBATCH --output="compute_density/%a.log"
#SBATCH --error="compute_density/%a.err"
#SBATCH --array="0-111"

source activate lamom2
idx=$SLURM_ARRAY_TASK_ID
srun python compute_density.py ${idx}
