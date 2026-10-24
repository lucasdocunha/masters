#!/bin/bash
#SBATCH --job-name=ceap_full_p2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --time=120:00:00
#SBATCH --output=logs/cluster_full_p2_%j.out
#SBATCH --error=logs/cluster_full_p2_%j.err

set -e

echo "=== INICIANDO JOB FULL PARTE 2 (2019-2025 - ~635k PDFs) ==="
echo "Node: $(hostname)"
echo "Data: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd /home/lucas.ocunha/masters

PYTHON="/home/lucas.ocunha/masters/.venv/bin/python"

# Execução da clusterização da Parte 2 (2019 a 2025 completo)
$PYTHON scripts/clusterizacao.py \
    --input-dir /media/ssd2/datasets/CEAP \
    --raw-dir data/raw \
    --years 2019 2020 2021 2022 2023 2024 2025 \
    --sample-rate 1.0 \
    --sample-seed 42 \
    --output-prefix data/latent_vectors_full_part2 \
    --dino-model facebook/dinov2-small \
    --vlm-model vidore/colpali-v1.2 \
    --k-doc 5 \
    --k-page 10 \
    --checkpoint-every 5000

echo "=== JOB FULL PARTE 2 FINALIZADO COM SUCESSO: $(date) ==="
