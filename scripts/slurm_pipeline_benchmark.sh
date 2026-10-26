#!/bin/bash
#SBATCH --job-name=bench_vlm_pipelines
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/benchmark_pipelines_%j.out
#SBATCH --error=logs/benchmark_pipelines_%j.err

set -e

echo "=== INICIANDO BENCHMARK DE PIPELINES VLM (1.300 DOCUMENTOS) ==="
echo "Node: $(hostname)"
echo "Data de Início: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

cd /home/lucas.ocunha/masters
mkdir -p logs

PYTHON="/home/lucas.ocunha/masters/.venv_ocr/bin/python"

# -------------------------------------------------------------
# FASE 1: OPÇÃO B - Qwen2.5-VL-3B-Instruct (End-to-End Dual-GPU)
# -------------------------------------------------------------
echo ""
echo "=========================================================="
echo " FASE 1: OPÇÃO B - Qwen2.5-VL-3B-Instruct (End-to-End)"
echo "=========================================================="
echo "Início da Fase 1: $(date)"

$PYTHON scripts/run_pipeline_benchmark.py \
    --method qwenvl \
    --gpu-id 0 \
    --num-gpus 2 > logs/bench_qwenvl_gpu0_${SLURM_JOB_ID}.log 2>&1 &
PID_QWEN_0=$!

$PYTHON scripts/run_pipeline_benchmark.py \
    --method qwenvl \
    --gpu-id 1 \
    --num-gpus 2 > logs/bench_qwenvl_gpu1_${SLURM_JOB_ID}.log 2>&1 &
PID_QWEN_1=$!

wait $PID_QWEN_0
wait $PID_QWEN_1

echo "Fase 1 (Qwen-VL) concluída com sucesso: $(date)"
echo ""
echo "--- Relatório Parcial após Fase 1 ---"
$PYTHON scripts/compare_all_pipelines.py

# -------------------------------------------------------------
# FASE 2: OPÇÃO A - LightOnOCR-3-4B + Qwen (Two-Stage Dual-GPU)
# -------------------------------------------------------------
echo ""
echo "=========================================================="
echo " FASE 2: OPÇÃO A - LightOnOCR + Qwen (Two-Stage)"
echo "=========================================================="
echo "Início da Fase 2: $(date)"

$PYTHON scripts/run_pipeline_benchmark.py \
    --method twostage \
    --gpu-id 0 \
    --num-gpus 2 > logs/bench_twostage_gpu0_${SLURM_JOB_ID}.log 2>&1 &
PID_TWO_0=$!

$PYTHON scripts/run_pipeline_benchmark.py \
    --method twostage \
    --gpu-id 1 \
    --num-gpus 2 > logs/bench_twostage_gpu1_${SLURM_JOB_ID}.log 2>&1 &
PID_TWO_1=$!

wait $PID_TWO_0
wait $PID_TWO_1

echo "Fase 2 (Two-Stage) concluída com sucesso: $(date)"

# -------------------------------------------------------------
# FASE 3: RELATÓRIO FINAL CONSOLIDADO
# -------------------------------------------------------------
echo ""
echo "=========================================================="
echo " FASE 3: RELATÓRIO FINAL CONSOLIDADO DOS 3 PIPELINES"
echo "=========================================================="
$PYTHON scripts/compare_all_pipelines.py

echo "=== BENCHMARK FINALIZADO COM SUCESSO: $(date) ==="
