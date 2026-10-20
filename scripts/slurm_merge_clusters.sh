#!/bin/bash
#SBATCH --job-name=ceap_cluster_merge
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cluster_merge_%j.out
#SBATCH --error=logs/cluster_merge_%j.err

set -e

echo "=== INICIANDO CONSOLIDAÇÃO E CLUSTERIZAÇÃO GLOBAL ==="
echo "Node: $(hostname)"
echo "Data: $(date)"

cd /home/lucas.ocunha/masters

PYTHON="/home/lucas.ocunha/masters/.venv/bin/python"

# Unificação das partes e execução do KMeans global
$PYTHON scripts/merge_clusters.py \
    --parts data/latent_vectors_sample_part1 data/latent_vectors_sample_part2 \
    --output-prefix data/latent_vectors_sample_5pct \
    --k-doc 5 \
    --k-page 10

echo "=== CONSOLIDAÇÃO FINALIZADA COM SUCESSO: $(date) ==="
