#!/bin/bash -l
#SBATCH --job-name=Mestrado_Inferencia_NFE
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=saida_%a.out
#SBATCH --error=saida_%a.err
#SBATCH --time=72:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL

source /opt/conda/etc/profile.d/conda.sh
conda activate mestrado

SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-5}"
SCRIPT_PATH="/users/home/lucas.ocunha/research/masters/teste.py"

echo "Iniciando Parte ${SHARD_ID} de ${NUM_SHARDS} na GPU:"
nvidia-smi

# Se passar modelos como argumento no sbatch, usa os passados;
# caso contrário, sobe os 3 modelos em paralelo na mesma H100 (80GB):
if [ "$#" -gt 0 ]; then
    MODELOS=("$@")
else
    MODELOS=(
        "maritaca-ai/S5-nano"
        "Qwen/Qwen3.5-4B"
        "Qwen/Qwen3.5-9B"
    )
fi

PIDS=()

encerrar_filhos() {
    echo "Sinal de interrupção recebido no job! Encaminhando para os ${#PIDS[@]} modelos em paralelo..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait
    exit 0
}

trap encerrar_filhos SIGINT SIGTERM

for MODELO in "${MODELOS[@]}"; do
    echo "======================================================================"
    echo "Subindo em paralelo na GPU: ${MODELO} (Parte ${SHARD_ID}/${NUM_SHARDS})"
    echo "======================================================================"
    python -u "${SCRIPT_PATH}" "${SHARD_ID}" "${NUM_SHARDS}" "${MODELO}" &
    PIDS+=("$!")
done

# Aguarda todos os modelos em paralelo finalizarem
wait
echo "Todos os ${#MODELOS[@]} modelos finalizaram a Parte ${SHARD_ID}/${NUM_SHARDS}!"
