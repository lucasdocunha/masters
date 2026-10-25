"""
FastAPI ML Backend para Label Studio com aprendizado ativo e retreinamento a cada 100 anotações.
Compatível com a API padrão do Label Studio ML Backend.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

from model import DocumentClassifierManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml_backend.api")

app = FastAPI(title="CEAP Document Classifier - Label Studio ML Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações de ambiente
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
EMBEDDINGS_NPZ = Path(os.environ.get("EMBEDDINGS_NPZ", "/data/latent_vectors_sample_5pct.npz"))
METADATA_JSON = Path(os.environ.get("METADATA_JSON", "/data/latent_vectors_sample_5pct_metadata.json"))
RETRAIN_INTERVAL = int(os.environ.get("RETRAIN_INTERVAL", "50"))

LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://ceap-label-studio:8080")
LABEL_STUDIO_EMAIL = os.environ.get("LABEL_STUDIO_EMAIL", "admin@ceap.com")
LABEL_STUDIO_PASS = os.environ.get("LABEL_STUDIO_PASS", "ceap-admin-2026")

# Instância global do gerenciador
manager = DocumentClassifierManager(
    model_dir=MODEL_DIR,
    embeddings_npz=EMBEDDINGS_NPZ,
    metadata_json=METADATA_JSON,
    retrain_interval=RETRAIN_INTERVAL,
)


def sync_predictions_for_unlabeled_tasks() -> None:
    """Atualiza o model_version no Label Studio e gera predições para todas as tarefas pendentes."""
    logger.info(f"[SYNC PREDICTIONS] Iniciando sincronização automática para modelo {manager.version}...")
    try:
        s = requests.Session()
        login_url = f"{LABEL_STUDIO_URL}/user/login/"
        s.get(login_url, timeout=10)
        csrftoken = s.cookies.get("csrftoken")
        login_res = s.post(
            login_url,
            data={
                "email": LABEL_STUDIO_EMAIL,
                "password": LABEL_STUDIO_PASS,
                "csrfmiddlewaretoken": csrftoken,
            },
            headers={"Referer": login_url},
            timeout=10,
        )
        if login_res.status_code != 200:
            logger.error(f"[SYNC PREDICTIONS] Falha na autenticação do Label Studio: status {login_res.status_code}")
            return

        csrftoken = s.cookies.get("csrftoken")
        headers = {"X-CSRFToken": csrftoken, "Referer": f"{LABEL_STUDIO_URL}/"}

        # 1. Atualizar model_version do Projeto 1 para a nova versão treinada
        p_res = s.patch(
            f"{LABEL_STUDIO_URL}/api/projects/1",
            json={"model_version": manager.version},
            headers=headers,
            timeout=10,
        )
        logger.info(f"[SYNC PREDICTIONS] Project 1 model_version atualizado para '{manager.version}' (status={p_res.status_code})")

        # 2. Disparar a ação retrieve_tasks_predictions no Data Manager do Label Studio
        action_res = s.post(
            f"{LABEL_STUDIO_URL}/api/dm/actions?project=1&id=retrieve_tasks_predictions",
            json={"selectedItems": {"all": True, "included": []}},
            headers=headers,
            timeout=180,
        )
        logger.info(f"[SYNC PREDICTIONS] Ação retrieve_tasks_predictions finalizada (status={action_res.status_code})")
    except Exception as e:
        logger.error(f"[SYNC PREDICTIONS] Erro ao sincronizar predições: {e}", exc_info=True)


def extract_filename_from_url_or_data(data: dict[str, Any]) -> str:
    """Extrai o nome do arquivo (ex: '5453054.png' ou '5453054.pdf') a partir dos campos do Label Studio."""
    # 1. Se houver campo explícito 'doc_name' no data da tarefa, priorizar
    if "doc_name" in data and data["doc_name"]:
        return Path(data["doc_name"]).name

    # 2. Verificar se está nos campos comuns de mídia
    candidate = data.get("image") or data.get("pdf") or data.get("document") or data.get("url") or ""
    if not candidate:
        for k, v in data.items():
            if isinstance(v, str) and (".png" in v.lower() or ".pdf" in v.lower()):
                candidate = v
                break

    # 3. Tratar query string do Label Studio Local Files: /data/local-files/?d=images/5477927.png
    if "?d=" in candidate:
        query_part = candidate.split("?d=")[-1].split("&")[0]
        return Path(query_part).name

    clean = candidate.split("?")[0]
    fname = Path(clean).name
    return fname or candidate


@app.get("/")
@app.get("/health")
def health() -> dict[str, Any]:
    """Healthcheck endpoint para o Label Studio e Docker."""
    return {
        "status": "UP",
        "model_version": manager.version,
        "classes": manager.classes,
        "total_annotations": len(manager.annotations_history),
        "counter_to_next_retrain": f"{manager.new_annotations_counter}/{manager.retrain_interval}",
        "indexed_embeddings": len(manager.embeddings_lookup),
    }


@app.post("/setup")
async def setup(request: Request) -> dict[str, Any]:
    """Endpoint de setup chamado pelo Label Studio ao conectar o ML Backend."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    logger.info(f"Label Studio /setup conectado: {body.get('project', 'unknown')}")
    return {
        "status": "ok",
        "model_version": manager.version,
    }


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    """Endpoint de predição do Label Studio ML Backend.
    
    Recebe lista de tarefas e devolve predições no schema:
    {
      "results": [
        {
          "result": [{ "from_name": "choice", "to_name": "image", "type": "choices", "value": { "choices": [classe] } }],
          "score": 0.95,
          "model_version": "v1"
        }
      ]
    }
    """
    body = await request.json()
    tasks = body.get("tasks", [])
    label_config = body.get("label_config", "")

    # Identificar tag de choice e target no label_config se disponível
    from_name = "choice"
    to_name = "image"

    results = []
    for task in tasks:
        task_data = task.get("data", {})
        fname = extract_filename_from_url_or_data(task_data)
        pred = manager.predict(fname)

        pred_item = {
            "result": [
                {
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "choices",
                    "value": {
                        "choices": [pred["class"]]
                    },
                }
            ],
            "score": pred["score"],
            "model_version": pred["model_version"],
        }
        results.append(pred_item)

    return {"results": results}


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Webhook chamado automaticamente pelo Label Studio ao criar/editar anotações."""
    payload = await request.json()
    action = payload.get("action", "")
    logger.info(f"Webhook recebido: ação='{action}'")

    if action in ["ANNOTATION_CREATED", "ANNOTATION_UPDATED", "TASKS_DONE"]:
        annotation = payload.get("annotation", {})
        task = payload.get("task", {}) or annotation.get("task", {})
        task_data = task.get("data", {})

        doc_name = extract_filename_from_url_or_data(task_data)
        results = annotation.get("result", [])

        chosen_label = None
        for res in results:
            if res.get("type") == "choices":
                choices = res.get("value", {}).get("choices", [])
                if choices:
                    chosen_label = choices[0]
                    break

        if doc_name and chosen_label:
            retrained = manager.add_annotation(doc_name, chosen_label)
            if retrained:
                background_tasks.add_task(sync_predictions_for_unlabeled_tasks)
            return {
                "status": "recorded",
                "doc_name": doc_name,
                "label": chosen_label,
                "retrained": retrained,
                "new_version": manager.version,
                "new_counter": f"{manager.new_annotations_counter}/{manager.retrain_interval}",
            }

    return {"status": "ignored", "reason": f"Ação {action} ou campos vazios"}


@app.post("/train")
def train_manual(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Força retreinamento do modelo independente do contador e sincroniza predições."""
    result = manager.train()
    if result.get("status") == "success":
        background_tasks.add_task(sync_predictions_for_unlabeled_tasks)
    return result


@app.get("/stats")
def stats() -> dict[str, Any]:
    """Retorna detalhes sobre histórico de anotações e modelo atual."""
    return {
        "version": manager.version,
        "train_count": manager.train_count,
        "total_annotations": len(manager.annotations_history),
        "counter_to_next_retrain": manager.new_annotations_counter,
        "retrain_interval": manager.retrain_interval,
        "classes": manager.classes,
        "is_model_loaded": manager.model is not None,
    }
