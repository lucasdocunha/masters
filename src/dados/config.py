"""Caminhos centrais do projeto.

Todo script/módulo deve importar caminhos daqui em vez de hardcodar diretórios.
Variáveis de ambiente permitem apontar para o armazenamento externo do dataset
e para a área de trabalho (validação manual) sem editar código.
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # Ano-*.json(.zip) e amostra de PDFs
PROCESSED_DIR = DATA_DIR / "processed"  # embeddings, npz, tarefas do Label Studio
MODELS_DIR = DATA_DIR / "models"        # histórico de anotações, cabeças de projeção
LABELING_DIR = DATA_DIR / "labeling"    # decisões aceitas por classe
REPORTS_DIR = DATA_DIR / "reports"      # clusterizações e visualizações (HTML)
EVALUATION_DIR = DATA_DIR / "evaluation"
WEB_DIR = ROOT_DIR / "web"

# Armazenamento externo com os ~1,4 M PDFs (não versionado).
PDF_DIR = Path(os.environ.get("CEAP_PDF_DIR", "/media/ssd2/datasets/CEAP"))
DESKTOP_DIR = Path(os.environ.get("CEAP_DESKTOP_DIR", Path.home() / "Desktop"))
