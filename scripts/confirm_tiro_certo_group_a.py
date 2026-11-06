#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirmação de Alta Precisão das Classes 'Tiro Certo' (Grupo A):
- sem_parar           : 711
- agua                : 348
- passagem_aerea      : 188
- correios            : 1.232
- passagem_rodoviaria : 146
Total: ~2.625 documentos
Critérios: Similaridade métrica ArcFace >= 97% + Validação por Fornecedor Oficial.
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from collections import Counter
import psycopg
import numpy as np

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
HIST_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
EMBS_PATH = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

RULES = {
    "sem_parar": re.compile(
        r'SEM\s*PARAR|CGMP|CENTRO\s*DE\s*GEST[AÃ]O\s*DE\s*MEIOS',
        re.IGNORECASE
    ),
    "agua": re.compile(
        r'COPASA|SABESP|SANEAGO|CAESB|CEDAE|COMPESA|CORSAN|SANEPAR|AGUAS?|SANEAMENTO|CASAN|DESO|CAEMA|EMBASA|DAE\b',
        re.IGNORECASE
    ),
    "passagem_aerea": re.compile(
        r'LATAM|TAM\b|GOL\b|AZUL\b|AVIANCA|AIRLINES|LINHAS\s*A[EÉ]REAS',
        re.IGNORECASE
    ),
    "correios": re.compile(
        r'CORREIO|ECT\b|EMPR.*BRAS.*CORREIO|TELEGRAFO',
        re.IGNORECASE
    ),
    "passagem_rodoviaria": re.compile(
        r'VIA[CÇ][AÃ]O|OURO\s*E\s*PRATA|1001|NOVO\s*HORIZONTE|COMETA|CATARINENSE|PROGRESSO|GUANABARA|GONTINHO|AGUIA\s*BRANCA|RODOVI[AÁ]RI|ITAPEMIRIM|PENHA|SANTA\s*CRUZ',
        re.IGNORECASE
    )
}


def main():
    print("=" * 76)
    print(" CONFIRMAÇÃO AUTOMÁTICA DAS CLASSES 'TIRO CERTO' (GRUPO A)")
    print("=" * 76)
    t0 = time.time()

    # 1. Carregar Histórico e Metadados
    print("\n1. Carregando anotações e espaço latente ArcFace 128-D...")
    with open(HIST_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in history.items()}

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    stems = [d["doc_name"].replace(".pdf", "") for d in docs]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    embs = np.load(EMBS_PATH)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    # 2. Processar cada classe do Grupo A
    print("\n2. Filtrando e validando por similaridade >= 97% + fornecedor...")
    confirmed_by_class = {}
    doubt_by_class = {}
    all_confirmed = []

    for cls, regex in RULES.items():
        c_idxs = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == cls and s in stem_to_idx]
        if not c_idxs:
            continue
        centroid = embs[c_idxs].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        sims = np.dot(embs, centroid)

        cands = [stems[i] for i in range(len(stems)) if sims[i] >= 0.97 and stems[i] not in norm_hist]
        cls_conf = []
        cls_doubt = []

        for s in cands:
            forn = lookup.get(f"{s}.pdf", {}).get("fornecedor", "")
            if regex.search(forn):
                cls_conf.append(s)
                all_confirmed.append((s, cls))
            else:
                cls_doubt.append((s, forn))

        confirmed_by_class[cls] = cls_conf
        doubt_by_class[cls] = cls_doubt

        prev_count = sum(1 for v in norm_hist.values() if v == cls)
        print(f"   • {cls.upper():20s}: {prev_count:5,d} -> {prev_count + len(cls_conf):5,d} (+{len(cls_conf):4,d} confirmados | {len(cls_doubt):3,d} dúvidas)")

    total_new = len(all_confirmed)
    print(f"\n   ✓ Total geral de novos documentos confirmados: {total_new:,}")

    if total_new == 0:
        print("[AVISO] Nenhum documento para confirmar.")
        return

    # 3. Atualizar annotations_history.json
    print(f"\n3. Registrando no histórico ({HIST_FILE.name})...")
    prev_total = len(norm_hist)
    for s, cls in all_confirmed:
        norm_hist[s] = cls

    with open(HIST_FILE, "w", encoding="utf-8") as f:
        json.dump(norm_hist, f, indent=2, ensure_ascii=False)

    new_total = len(norm_hist)
    print(f"   ✓ NOVO TOTAL GERAL DO DATASET: {new_total:,} (+{total_new:,})")

    # 4. Sincronizar PostgreSQL
    print(f"\n4. Sincronizando com PostgreSQL ({DB_URL})...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                doc_rows = []
                for s, cls in all_confirmed:
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    doc_rows.append((doc_id, ano, caminho, "baixado", cls))

                batch_size = 1000
                for i in range(0, len(doc_rows), batch_size):
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE
                        SET classe_predita = EXCLUDED.classe_predita,
                            status = EXCLUDED.status;
                    """, doc_rows[i:i + batch_size])

                conn.commit()
        print("   ✓ PostgreSQL sincronizado com sucesso!")
    except Exception as e:
        print(f"   [AVISO] Erro no PostgreSQL: {e}")

    # 5. Salvar manifestos de dúvidas no Desktop
    p_duvidas = DESKTOP_DIR / "validacao_tiro_certo_duvidas"
    p_duvidas.mkdir(parents=True, exist_ok=True)
    with open(p_duvidas / "casos_ambiguos.json", "w", encoding="utf-8") as f:
        json.dump(doubt_by_class, f, indent=2, ensure_ascii=False)
    print(f"\n5. Casos com fornecedor não mapeado salvos em {p_duvidas.name}/casos_ambiguos.json")

    print("\n" + "=" * 76)
    print(f" SUCESSO! {total_new:,} documentos do Grupo A confirmados em {time.time() - t0:.1f}s.")
    print(f" Dataset total: {new_total:,} documentos rotulados.")
    print("=" * 76)


if __name__ == "__main__":
    main()
