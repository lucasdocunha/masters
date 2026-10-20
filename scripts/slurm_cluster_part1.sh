#!/bin/bash
#SBATCH --job-name=ceap_cluster_p1
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cluster_p1_%j.out
#SBATCH --error=logs/cluster_p1_%j.err

set -e

echo "=== INICIANDO JOB PARTE 1 (2014-2018) ==="
echo "Node: $(hostname)"
echo "Data: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd /home/lucas.ocunha/masters

PYTHON="/home/lucas.ocunha/masters/.venv/bin/python"

# Execução da clusterização da Parte 1 (2014 a 2018 com 5% de amostra)
$PYTHON scripts/clusterizacao.py \
    --input-dir /media/ssd2/datasets/CEAP \
    --raw-dir data/raw \
    --years 2014 2015 2016 2017 2018 \
    --sample-rate 0.05 \
    --sample-seed 42 \
    --output-prefix data/latent_vectors_sample_part1 \
    --dino-model facebook/dinov2-small \
    --vlm-model vidore/colpali-v1.2 \
    --k-doc 5 \
    --k-page 10

echo "=== JOB PARTE 1 FINALIZADO COM SUCESSO: $(date) ==="
