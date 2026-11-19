#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --gres gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name="eval_01"
#SBATCH --chdir=.
#SBATCH --output=eval_01_%j.out
#SBATCH --error=eval_01_%j.err

module purge
module load miniforge
module load cuda/12.6
source activate deepLearning

python evaluate_model.py