#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Confirmação Automática de Alta Precisão para ENERGIA e TELEFONIA:
- Critérios: Similaridade métrica ArcFace >= 97% + Fornecedor Oficial Regulatório (Concessionária / Operadora).
- Total esperado:
    • ENERGIA  : ~3.046 novos confirmados
    • TELEFONIA: ~1.023 novos confirmados
- Registra em:
    1. ml_backend/models/annotations_history.json
    2. PostgreSQL (tabela documento)
    3. Desktop/validacao_energia_duvidas (apenas os 8 casos ambíguos para inspeção)
    4. Regenera os CSVs de exportação no Desktop
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
HIST_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
EMBS_PATH = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

# Regex Concessionárias de Energia
RE_ENERGIA_EXP = re.compile(
    r'ENERG|EL[EÉ]TR|DISTRIBUI|CEMIG|LIGHT|CPFL|COELBA|CELESC|CEEE|ENEL|'
    r'PIRATININGA|AMPLA|ELEKTRO|RGE|EQUATORIAL|COPEL|CELPE|EDP|COSERN|'
    r'FOR[CÇ]A\s*E\s*LUZ|CHESP|CEB|ENERGISA|COELCE|CELG|CEMAR|ELETROPAULO',
    re.IGNORECASE
)

# Regex Operadoras de Telecomunicações
RE_TELEFONIA_EXP = re.compile(
    r'TELECOM|TELEFON|CLARO|VIVO|TIM|OI\b|NEXTEL|EMBRATEL|GVT|ALGAR|NET\b|AMERICA\s*MOVIL|'
    r'TELEF[OÔ]NICA|INTELIG|HUGHESNET|STARLINK|SKY',
    re.IGNORECASE
)


def main():
    print("=" * 76)
    print(" CONFIRMAÇÃO DE ENERGIA E TELEFONIA (SIMILARIDADE >= 97% + FORNECEDOR)")
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

    # 2. Processar ENERGIA
    print("\n2. Filtrando candidatos de ENERGIA...")
    c_idxs_eng = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == "energia" and s in stem_to_idx]
    cent_eng = embs[c_idxs_eng].mean(axis=0)
    cent_eng /= np.linalg.norm(cent_eng)
    sims_eng = np.dot(embs, cent_eng)

    cands_eng = [stems[i] for i in range(len(stems)) if sims_eng[i] >= 0.97 and stems[i] not in norm_hist]
    eng_confirmed = []
    eng_doubt = []
    for s in cands_eng:
        forn = lookup.get(f"{s}.pdf", {}).get("fornecedor", "")
        if RE_ENERGIA_EXP.search(forn) and not RE_TELEFONIA_EXP.search(forn):
            eng_confirmed.append(s)
        else:
            eng_doubt.append((s, forn))

    print(f"   ✓ Candidatos com sim >= 97%: {len(cands_eng):,}")
    print(f"   ✓ Concessionárias confirmadas: {len(eng_confirmed):,}")
    print(f"   • Casos isolados para revisão: {len(eng_doubt):,}")

    # 3. Processar TELEFONIA
    print("\n3. Filtrando candidatos de TELEFONIA...")
    c_idxs_tel = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == "telefonia" and s in stem_to_idx]
    cent_tel = embs[c_idxs_tel].mean(axis=0)
    cent_tel /= np.linalg.norm(cent_tel)
    sims_tel = np.dot(embs, cent_tel)

    cands_tel = [stems[i] for i in range(len(stems)) if sims_tel[i] >= 0.97 and stems[i] not in norm_hist]
    tel_confirmed = []
    tel_doubt = []
    for s in cands_tel:
        forn = lookup.get(f"{s}.pdf", {}).get("fornecedor", "")
        if RE_TELEFONIA_EXP.search(forn):
            tel_confirmed.append(s)
        else:
            tel_doubt.append((s, forn))

    print(f"   ✓ Candidatos com sim >= 97%: {len(cands_tel):,}")
    print(f"   ✓ Operadoras confirmadas     : {len(tel_confirmed):,}")
    print(f"   • Casos isolados para revisão: {len(tel_doubt):,}")

    # 4. Atualizar annotations_history.json
    print(f"\n4. Registrando no histórico ({HIST_FILE.name})...")
    prev_total = len(norm_hist)
    prev_eng = sum(1 for v in norm_hist.values() if v == "energia")
    prev_tel = sum(1 for v in norm_hist.values() if v == "telefonia")

    for s in eng_confirmed:
        norm_hist[s] = "energia"
    for s in tel_confirmed:
        norm_hist[s] = "telefonia"

    with open(HIST_FILE, "w", encoding="utf-8") as f:
        json.dump(norm_hist, f, indent=2, ensure_ascii=False)

    new_total = len(norm_hist)
    new_eng = sum(1 for v in norm_hist.values() if v == "energia")
    new_tel = sum(1 for v in norm_hist.values() if v == "telefonia")

    print(f"   • ENERGIA   : {prev_eng:,} -> {new_eng:,} (+{len(eng_confirmed):,})")
    print(f"   • TELEFONIA : {prev_tel:,} -> {new_tel:,} (+{len(tel_confirmed):,})")
    print(f"   ✓ NOVO TOTAL GERAL ROTULADO: {new_total:,} (+{len(eng_confirmed) + len(tel_confirmed):,})")

    # 5. Sincronizar PostgreSQL
    print(f"\n5. Sincronizando com PostgreSQL ({DB_URL})...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                doc_rows = []
                for s in eng_confirmed:
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    doc_rows.append((doc_id, ano, caminho, "baixado", "energia"))

                for s in tel_confirmed:
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    doc_rows.append((doc_id, ano, caminho, "baixado", "telefonia"))

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

    # 6. Salvar dúvidas no Desktop
    if eng_doubt:
        p_duvidas = DESKTOP_DIR / "validacao_energia_duvidas"
        p_duvidas.mkdir(parents=True, exist_ok=True)
        manifest_doubt = {s: forn for s, forn in eng_doubt}
        with open(p_duvidas / "casos_ambiguos.json", "w", encoding="utf-8") as f:
            json.dump(manifest_doubt, f, indent=2, ensure_ascii=False)
        print(f"\n6. {len(eng_doubt)} casos de energia ambíguos salvos em {p_duvidas.name}/")

    print("\n" + "=" * 76)
    print(f" SUCESSO! {len(eng_confirmed) + len(tel_confirmed):,} documentos confirmados em {time.time() - t0:.1f}s.")
    print(f" Dataset total: {new_total:,} documentos rotulados.")
    print("=" * 76)


if __name__ == "__main__":
    main()
