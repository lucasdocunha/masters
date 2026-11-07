#!/bin/bash
#SBATCH --job-name=ceap_pdfs
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=120:00:00

# Onde salvar os PDFs e onde está o repositório. Para continuar um job interrompido, rode de novo.
DEST="/caminho/para/pdfs"
REPO="$HOME/masters"

set -e
cd "$REPO"

ls data/raw/Ano-*.json > /dev/null 2>&1 || uv run python -m dados.ceap.download_url
uv run python -m dados.ceap.download_pdfs --path "$DEST"
