#!/bin/bash -l
#SBATCH --job-name=Mestrado_Inferencia_NFE
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
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

# Se passar um modelo específico no sbatch (ex: sbatch .sh/job_teste_4jobs.sh Qwen/Qwen3.5-4B), roda só ele;
# caso contrário, roda ambos em sequência na mesma alocação:
if [ -n "${1:-}" ]; then
    MODELOS=("$1")
else
    MODELOS=(
        "maritaca-ai/S5-nano"
        "Qwen/Qwen3.5-4B"
    )
fi

for MODELO in "${MODELOS[@]}"; do
    echo "======================================================================"
    echo "Rodando modelo: ${MODELO} (Parte ${SHARD_ID}/${NUM_SHARDS})"
    echo "======================================================================"
    python -u "${SCRIPT_PATH}" "${SHARD_ID}" "${NUM_SHARDS}" "${MODELO}"
done
