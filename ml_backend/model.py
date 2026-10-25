"""
Gerenciador do modelo de Machine Learning para classificação de documentos do CEAP.
Utiliza representações densas concatenadas (DINOv2 384-D + ColPali VLM 128-D = 512-D)
e um classificador linear (LogisticRegression) ultraleve com retreino em <0.5s.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("ml_backend.model")

DEFAULT_CLASSES = [
    "nfc",
    "nfe",
    "nfse",
    "recibo",
    "emails",
    "energia",
    "agua",
    "boleto",
    "fatura_cartao_credito",
    "uber",
    "aviao",
    "fatura_internet_celular",
    "invoice",
    "fatura",
]


class DocumentClassifierManager:
    def __init__(
        self,
        model_dir: Path,
        embeddings_npz: Optional[Path] = None,
        metadata_json: Optional[Path] = None,
        retrain_interval: int = 50,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.retrain_interval = retrain_interval
        self.classes = DEFAULT_CLASSES
        self.model: Optional[LogisticRegression] = None
        self.version = "v0-cold-start"
        self.train_count = 0
        self.new_annotations_counter = 0

        # Histórico de anotações: {doc_name: label}
        self.annotations_history: dict[str, str] = {}
        self.history_file = self.model_dir / "annotations_history.json"

        # Mapa em memória {doc_name: embedding_512d}
        self.embeddings_lookup: dict[str, np.ndarray] = {}

        if embeddings_npz and metadata_json:
            self.load_embeddings(embeddings_npz, metadata_json)

        self.load_history()
        self.load_latest_model()

    def load_embeddings(self, npz_path: Path, meta_path: Path) -> None:
        """Carrega e concatena embeddings DINOv2 e VLM em mapa doc_name -> vetor 512-D."""
        npz_path = Path(npz_path)
        meta_path = Path(meta_path)
        if not npz_path.exists() or not meta_path.exists():
            logger.warning(f"Arquivos de embeddings não encontrados: {npz_path} ou {meta_path}")
            return

        logger.info(f"Carregando embeddings de {npz_path}...")
        npz_data = np.load(npz_path)
        vlm = npz_data["vlm_doc_embeddings"]
        dino = npz_data["dino_doc_embeddings"]

        # Concatena: (N, 512)
        combined = np.hstack([dino, vlm]).astype(np.float32)
        # Normalização L2
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        combined /= norms

        logger.info(f"Carregando metadados de {meta_path}...")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        docs = meta.get("documents", [])

        for idx, doc in enumerate(docs):
            if idx >= len(combined):
                break
            name = doc.get("doc_name", "")
            if name:
                self.embeddings_lookup[name] = combined[idx]
                stem = Path(name).stem
                self.embeddings_lookup[stem] = combined[idx]

        logger.info(f"Carregados embeddings para {len(self.embeddings_lookup):,} referências de documentos.")

    def load_history(self) -> None:
        """Carrega anotações prévias do disco se existirem."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.annotations_history = json.load(f)
                logger.info(f"Histórico carregado com {len(self.annotations_history)} anotações.")
            except Exception as e:
                logger.warning(f"Falha ao carregar histórico de anotações: {e}")

    def save_history(self) -> None:
        """Salva anotações persistidas em JSON."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.annotations_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Falha ao salvar histórico: {e}")

    def load_latest_model(self) -> None:
        """Carrega o modelo treinado mais recente do diretório de modelos."""
        model_files = sorted(self.model_dir.glob("model_v*.joblib"))
        if model_files:
            latest = model_files[-1]
            try:
                data = joblib.load(latest)
                self.model = data["model"]
                self.version = data.get("version", latest.stem)
                self.classes = data.get("classes", DEFAULT_CLASSES)
                self.train_count = int(self.version.replace("v", "").split("-")[0]) if "v" in self.version else len(model_files)
                logger.info(f"Modelo carregado com sucesso: {latest.name} ({self.version})")
            except Exception as e:
                logger.warning(f"Erro ao carregar modelo {latest}: {e}")

    def add_annotation(self, doc_name: str, label: str) -> bool:
        """Registra uma anotação e verifica se atingiu o gatilho de retreinamento."""
        stem = Path(doc_name).stem
        self.annotations_history[stem] = label
        self.new_annotations_counter += 1
        self.save_history()

        logger.info(f"Nova anotação: {stem} -> {label} (Acumulado desde último treino: {self.new_annotations_counter}/{self.retrain_interval})")

        if self.new_annotations_counter >= self.retrain_interval:
            logger.info(f"Gatilho de retreinamento atingido ({self.new_annotations_counter} novas anotações). Retreinando...")
            self.train()
            return True
        return False

    def train(self) -> dict[str, Any]:
        """Treina LogisticRegression sobre todas as anotações acumuladas."""
        if not self.annotations_history:
            return {"status": "skipped", "reason": "nenhuma anotação disponível"}

        X = []
        y = []
        for doc_stem, label in self.annotations_history.items():
            emb = self.embeddings_lookup.get(doc_stem)
            if emb is not None:
                X.append(emb)
                y.append(label)

        if len(X) < 5:
            return {"status": "skipped", "reason": f"amostras insuficientes ({len(X)} < 5)"}

        unique_labels = sorted(list(set(y)))
        if len(unique_labels) < 2:
            return {"status": "skipped", "reason": f"mínimo de 2 classes necessárias (atual: {unique_labels})"}

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        clf = LogisticRegression(C=1.0, max_iter=250, class_weight="balanced", random_state=42)
        clf.fit(X, y)

        self.train_count += 1
        self.version = f"v{self.train_count}"
        self.model = clf
        self.classes = list(clf.classes_)
        self.new_annotations_counter = 0

        # Salvar checkpoint do modelo
        model_path = self.model_dir / f"model_{self.version}.joblib"
        joblib.dump({"model": clf, "version": self.version, "classes": self.classes}, model_path)

        train_score = float(clf.score(X, y))
        logger.info(f"[TREINO SUCESSO] Modelo {self.version} treinado em {len(X)} amostras | Acurácia treino: {train_score:.2%}")

        return {
            "status": "success",
            "version": self.version,
            "samples": len(X),
            "classes": self.classes,
            "train_accuracy": train_score,
            "model_path": str(model_path),
        }

    def predict(self, doc_name: str) -> dict[str, Any]:
        """Prediz a classe e probabilidade para um documento."""
        stem = Path(doc_name).stem
        emb = self.embeddings_lookup.get(stem)

        if emb is None:
            # Fallback neutro se embedding não for encontrado
            return {
                "class": "recibo",
                "score": 0.25,
                "model_version": self.version,
                "probabilities": {},
            }

        if self.model is not None:
            probs = self.model.predict_proba([emb])[0]
            best_idx = int(np.argmax(probs))
            best_class = str(self.model.classes_[best_idx])
            best_score = float(probs[best_idx])
            prob_dict = {str(c): float(p) for c, p in zip(self.model.classes_, probs)}
            return {
                "class": best_class,
                "score": best_score,
                "model_version": self.version,
                "probabilities": prob_dict,
            }
        else:
            # Cold-start prior
            return {
                "class": "nfc",
                "score": 0.35,
                "model_version": "v0-cold-start",
                "probabilities": {},
            }
