#!/bin/bash
#SBATCH --job-name=ceap_ocr_lighton
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=120:00:00
#SBATCH --output=logs/ocr_lighton_%j.out
#SBATCH --error=logs/ocr_lighton_%j.err

set -e

echo "=== INICIANDO JOB OCR LIGHTON-3-4B (DUAL-GPU SLURM) ==="
echo "Node: $(hostname)"
echo "Data: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd /home/lucas.ocunha/masters
mkdir -p logs

PYTHON="/home/lucas.ocunha/masters/.venv_ocr/bin/python"

# Lançar Worker 0 na GPU 0 (particionamento par de ide_documento)
$PYTHON scripts/extract_lightonocr.py \
    --gpu-id 0 \
    --num-gpus 2 \
    --batch-size 16 \
    --dpi 150 > logs/ocr_gpu0_${SLURM_JOB_ID}.log 2>&1 &
PID_GPU0=$!

# Lançar Worker 1 na GPU 1 (particionamento ímpar de ide_documento)
$PYTHON scripts/extract_lightonocr.py \
    --gpu-id 1 \
    --num-gpus 2 \
    --batch-size 16 \
    --dpi 150 > logs/ocr_gpu1_${SLURM_JOB_ID}.log 2>&1 &
PID_GPU1=$!

echo "Workers disparados em background: Worker-0 (PID $PID_GPU0), Worker-1 (PID $PID_GPU1)"

# Aguardar conclusão de ambos os workers
wait $PID_GPU0
wait $PID_GPU1

echo "=== JOB OCR LIGHTON-3-4B FINALIZADO COM SUCESSO: $(date) ==="
