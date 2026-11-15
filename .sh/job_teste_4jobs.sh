#!/bin/bash -l
#SBATCH --job-name=Mestrado_Inferencia
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%A_parte%a.out
#SBATCH --error=%x_%A_parte%a.err
#SBATCH --time=72:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL

source /opt/conda/etc/profile.d/conda.sh
conda activate meu-ambiente

echo "Iniciando Parte ${SLURM_ARRAY_TASK_ID} de ${SLURM_ARRAY_TASK_COUNT:-4} na GPU:"
nvidia-smi

python -u /home/lucas/masters/masters-data/teste.py "${SLURM_ARRAY_TASK_ID}" "${SLURM_ARRAY_TASK_COUNT:-4}"
