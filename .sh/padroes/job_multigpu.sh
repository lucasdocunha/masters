#!/bin/bash -l
#SBATCH --job-name=Mestrado_MultiGPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL

source /opt/conda/etc/profile.d/conda.sh
conda activate meu-ambiente

# Para modelos com device_map="auto" (ex: Hugging Face):
python "${1:-seu_script.py}"

# Para treinamento distribuído (PyTorch DDP / Accelerate), descomente abaixo:
# torchrun --standalone --nproc_per_node=2 "${1:-seu_script.py}"
