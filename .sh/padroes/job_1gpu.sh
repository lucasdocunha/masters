#!/bin/bash -l
#SBATCH --job-name=Mestrado_1GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL

source /opt/conda/etc/profile.d/conda.sh
conda activate meu-ambiente

python "${1:-seu_script.py}"
