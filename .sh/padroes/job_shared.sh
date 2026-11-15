#!/bin/bash
#SBATCH --job-name=Mestrado_Shared
#SBATCH --partition=shared
#SBATCH --gres=mps:25
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL

source /opt/conda/etc/profile.d/conda.sh
conda activate meu-ambiente

python "${1:-seu_script.py}"
