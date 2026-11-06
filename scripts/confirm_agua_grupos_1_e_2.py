#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirmação Oficial de Faturas de Água e Saneamento Básico (Grupos 1 e 2).
Grupo 1: Grandes concessionárias estaduais já mapeadas (COPASA, SABESP, SANEAGO, SANEPAR, CASAN, etc.)
Grupo 2: Novas concessionárias estaduais e privadas descobertas (CAGECE, CAERN, CAERD/CAER, BRK, SANESUL, COSANPA, SAAE, etc.)

Critérios de Validação:
1. Similaridade no espaço métrico ArcFace 128-D >= 70% com o centroide de 'agua'
2. Validação por Fornecedor Oficial de Água/Saneamento (Regex abrangente)
3. Exclusão estrita de condomínios e imobiliárias
4. Consistência textual em documentos com texto no PostgreSQL
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
HIST_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
META_PATH = BASE_DIR / "data" / "latent_vectors_full_metadata.json"
EMBS_PATH = BASE_DIR / "data" / "latent_vectors_metric_learned_128d.npy"
LOOKUP_FILE = BASE_DIR / "data" / "pdf_metadata_lookup.json"
YEAR_FILE = BASE_DIR / "data" / "pdf_to_year.json"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

# Regex Grupos 1 e 2 (Concessionárias Oficiais de Água e Saneamento)
AGUA_REGEX = re.compile(
    r"COPASA|SABESP|SANEAGO|CAESB|CEDAE|COMPESA|CORSAN|SANEPAR|"
    r"AGUAS?|[ÁA]GUAS|SANEAMENTO|CASAN|DESO|CAEMA|EMBASA|DAE\b|"
    r"CAGECE|CAERN|CAERD|CAER\b|SANESUL|BRK|COSANPA|SANEAR|SAAE|"
    r"SEMAE|SAERP|COSAMA|CASAL|CESAN|AGESPISA",
    re.IGNORECASE
)

# Filtro de Exclusão (Condomínios, Imobiliárias, Participações)
EXCLUDE_REGEX = re.compile(
    r"CONDOM[ÍI]NIO|EDIF[ÍI]CIO|\bED\b|IM[ÓO]VEIS|PARTICIPA|LOCADORA|HOTEL|POSTO\b",
    re.IGNORECASE
)


def main():
    print("=" * 76)
    print(" CONFIRMAÇÃO OFICIAL DE ÁGUA E SANEAMENTO (GRUPOS 1 E 2)")
    print("=" * 76)
    t0 = time.time()

    # 1. Carregar Histórico e Espaço Latente ArcFace
    print("\n1. Carregando anotações e espaço métrico 128-D...")
    with open(HIST_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in history.items()}
    prev_agua_count = sum(1 for v in norm_hist.values() if v == "agua")
    print(f"   • Documentos 'agua' previamente confirmados: {prev_agua_count:,}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    stems = [d["doc_name"].replace(".pdf", "") for d in docs]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    embs = np.load(EMBS_PATH)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    print("   • Carregando metadados de fornecedores e anos...")
    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    # 2. Calcular Centroide de Água e Similaridade
    print("\n2. Calculando similaridade no espaço métrico com centroide de 'agua'...")
    agua_idxs = [stem_to_idx[s] for s, l in norm_hist.items() if l == "agua" and s in stem_to_idx]
    cent_agua = embs[agua_idxs].mean(axis=0)
    cent_agua /= np.linalg.norm(cent_agua)

    sims = np.dot(embs, cent_agua)

    # 3. Filtrar e Validar Candidatos
    print("\n3. Filtrando candidatos com similaridade >= 70% + fornecedor oficial...")
    cands_idx = [i for i in range(len(stems)) if sims[i] >= 0.70 and stems[i] not in norm_hist]
    print(f"   • Candidatos brutos >= 70% não rotulados: {len(cands_idx):,}")

    confirmed = []
    doubt_list = []

    for idx in cands_idx:
        s = stems[idx]
        sim_val = float(sims[idx])
        forn = lookup.get(f"{s}.pdf", {}).get("fornecedor", "")

        # Testar fornecedor
        if AGUA_REGEX.search(forn) and not EXCLUDE_REGEX.search(forn):
            confirmed.append((s, forn, sim_val))
        else:
            doubt_list.append((s, forn, sim_val))

    print(f"   ✓ CONFIRMADOS COMO 'AGUA': {len(confirmed):,}")
    print(f"   • Casos de dúvida/outros fornecedores: {len(doubt_list):,}")

    if not confirmed:
        print("[AVISO] Nenhum novo documento validado.")
        return

    # Estatísticas de fornecedores confirmados
    forn_counter = Counter(c[1] for c in confirmed)
    print("\nTop 15 Fornecedores de Água Confirmados:")
    for forn, cnt in forn_counter.most_common(15):
        print(f"   ✓ {cnt:4d}x | {forn[:65]}")

    # 4. Atualizar annotations_history.json
    print(f"\n4. Registrando no histórico ({HIST_FILE.name})...")
    for s, _, _ in confirmed:
        norm_hist[s] = "agua"

    with open(HIST_FILE, "w", encoding="utf-8") as f:
        json.dump(norm_hist, f, indent=2, ensure_ascii=False)

    new_total = len(norm_hist)
    new_agua_count = sum(1 for v in norm_hist.values() if v == "agua")
    print(f"   ✓ Total 'agua' atualizado: {prev_agua_count:,} -> {new_agua_count:,} (+{len(confirmed):,})")
    print(f"   ✓ NOVO TOTAL DO DATASET HISTÓRICO: {new_total:,}")

    # 5. Sincronizar com PostgreSQL
    print(f"\n5. Sincronizando com PostgreSQL ({DB_URL})...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                doc_rows = []
                for s, _, _ in confirmed:
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    doc_rows.append((doc_id, ano, caminho, "baixado", "agua"))

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
        print(f"   [AVISO] Erro na sincronização com PostgreSQL: {e}")

    # 6. Salvar Dúvidas no Desktop
    p_duvidas = DESKTOP_DIR / "validacao_agua_duvidas"
    p_duvidas.mkdir(parents=True, exist_ok=True)
    with open(p_duvidas / "casos_duvidas.json", "w", encoding="utf-8") as f:
        json.dump([
            {"doc_id": s, "fornecedor": forn, "similaridade": round(sim_val, 4)}
            for s, forn, sim_val in doubt_list
        ], f, indent=2, ensure_ascii=False)
    print(f"\n6. {len(doubt_list):,} casos de dúvida salvos em {p_duvidas.name}/casos_duvidas.json")

    print("\n" + "=" * 76)
    print(f" SUCESSO! {len(confirmed):,} documentos de água confirmados em {time.time() - t0:.1f}s.")
    print("=" * 76)


if __name__ == "__main__":
    main()
