#!/bin/bash
set -e

cd /home/lucas.ocunha/masters

mkdir -p logs data

echo "Submetendo Jobs de Clusterização no SLURM (Dataset 2014 a 2025 - 1.34 Milhão de PDFs em 2 GPUs)..."

# Submete Parte 1 (GPU 0)
OUT1=$(sbatch scripts/slurm_cluster_part1.sh)
JOB1=$(echo "$OUT1" | awk '{print $4}')
echo "Job Parte 1 (2014-2018 - ~704k docs) submetido com ID: $JOB1"

# Submete Parte 2 (GPU 1)
OUT2=$(sbatch scripts/slurm_cluster_part2.sh)
JOB2=$(echo "$OUT2" | awk '{print $4}')
echo "Job Parte 2 (2019-2025 - ~635k docs) submetido com ID: $JOB2"

# Submete Merge com dependência dos dois
OUT3=$(sbatch --dependency=afterok:${JOB1}:${JOB2} scripts/slurm_merge_clusters.sh)
JOB3=$(echo "$OUT3" | awk '{print $4}')
echo "Job Merge & Cluster Global submetido com ID: $JOB3 (aguardando $JOB1 e $JOB2)"

echo ""
echo "Status atual da fila SLURM:"
squeue -u "$USER"
