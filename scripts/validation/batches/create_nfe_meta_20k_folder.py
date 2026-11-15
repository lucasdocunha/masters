#!/usr/bin/env python3
"""
Gera pasta de validação no Desktop para atingir a meta de 20.000 NF-e.
Atuais: 18.064 rotuladas. Faltam: 1.936.
Critérios:
  1. Contém obrigatoriamente a palavra 'DANFE' no texto do documento
  2. Exclusão estrita de NFC-e (Modelo 65 / Consumidor Eletrônica / SAT)
  3. Exclusão estrita de NFS-e (Serviços Municipais / Prefeituras / RPS)
  4. Exclusão de concessionárias de água/luz/telecom
  5. Ordenados pela similaridade de cosseno com o novo centróide de NF-e no espaço ArcFace 128-D

Estrutura:
  Desktop/validacao_nfe_meta_20k/
    ├── lote_01_top_1000/       (Rank 0001 a 1000 | sim 0.999 -> 0.970)
    ├── lote_02_top_1000/       (Rank 1001 a 2000 | sim 0.970 -> 0.850)
    ├── lote_03_reserva_300/    (Rank 2001 a 2300 | sim 0.850 -> 0.650)
    ├── confirmar_validacao.py
    ├── LEIAME.txt
    └── .manifesto_meta_20k.json
"""

import hashlib
import json
import os
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np
import psycopg2
import pymupdf

from dados.config import DESKTOP_DIR, PDF_DIR, ROOT_DIR

DATA_DIR = ROOT_DIR / "data"
BATCHES_DIR = DATA_DIR / "validacao_batches"
TARGET_DIR = BATCHES_DIR / "validacao_nfe_meta_20k"
DESKTOP_LINK = DESKTOP_DIR / "validacao_nfe_meta_20k"
RENDERED_DIR = DATA_DIR / "rendered_pages"
THUMB_DIR = Path.home() / ".cache" / "thumbnails" / "large"
HISTORY_FILE = DATA_DIR / "models" / "annotations_history.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
EMBS_FILE = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
META_FILE = DATA_DIR / "latent_vectors_full_metadata.json"

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

UTILITY_WORDS = [
    "ENERG", "ELETR", "CEEE", "COSERN", "COELBA", "LIGHT", "ENEL", "COPEL",
    "CEMIG", "SANEPAR", "SABESP", "COPASA", "CASAN", "EMBASA", "TELECOM",
    "CLARO", "VIVO", "TIM", "OI S.A", "TAXI", "SINPETAXI", "UBER"
]


def sanitize_filename(name: str, max_len: int = 24) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] if cleaned else "DESCONHECIDO"


def render_page_if_needed(doc_id: str) -> bool:
    png_path = RENDERED_DIR / f"{doc_id}.png"
    if png_path.exists() and png_path.stat().st_size > 0:
        return True
    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        return False
    try:
        doc = pymupdf.open(str(pdf_path))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=100)
        pix.save(str(png_path))
        doc.close()
        return True
    except Exception:
        return False


def setup_thumbnails(symlink_paths: List[Path], png_path: Path) -> None:
    for sp in symlink_paths:
        try:
            uri = "file://" + urllib.parse.quote(str(sp.absolute()))
            h = hashlib.md5(uri.encode("utf-8")).hexdigest()
            t_path = THUMB_DIR / f"{h}.png"
            if not t_path.exists() and png_path.exists():
                t_path.symlink_to(png_path.absolute())
        except Exception:
            pass


def main():
    t0 = time.time()
    print("=" * 70)
    print("PREPARANDO LOTE DE CANDIDATOS DANFE PARA META DE 20.000 NF-E")
    print("=" * 70)

    # 1. Carregar anotações
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        hist = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in hist.items()}
    current_nfe = sum(1 for v in norm_hist.values() if v == "nfe")
    needed = 20000 - current_nfe
    print(f"NF-e já rotuladas: {current_nfe:,}")
    print(f"Necessárias para atingir 20k: {needed:,}")

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stems = [d["doc_name"].replace(".pdf", "") for d in meta["documents"]]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    lookup = {}
    if LOOKUP_FILE.exists():
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    # 2. Carregar espaço métrico ArcFace 128-D e calcular centróide de NF-e
    print("\nCalculando centróide de NF-e no novo espaço métrico ArcFace 128-D...")
    embs = np.load(EMBS_FILE, mmap_mode="r")
    nfe_idxs = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == "nfe" and s in stem_to_idx]
    nfe_cent = np.array(embs[nfe_idxs]).mean(axis=0)
    nfe_cent /= np.linalg.norm(nfe_cent)

    # 3. Consultar documentos com 'DANFE' no PostgreSQL (sem NFC-e e sem NFS-e)
    print("\nConsultando documentos no PostgreSQL com 'DANFE' estrito...")
    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/ceap")
    cur = conn.cursor()
    cur.execute("""
        SELECT ide_documento, texto_completo
        FROM documento_texto_completo
        WHERE texto_completo ILIKE '%DANFE%'
          AND texto_completo NOT ILIKE '%NFC-E%'
          AND texto_completo NOT ILIKE '%CONSUMIDOR ELETR%'
          AND texto_completo NOT ILIKE '%MODELO 65%'
          AND texto_completo NOT ILIKE '%MOD. 65%'
          AND texto_completo NOT ILIKE '%CF-E-SAT%'
          AND texto_completo NOT ILIKE '%NOTA FISCAL DE SERVI%'
          AND texto_completo NOT ILIKE '%NFS-E%'
          AND texto_completo NOT ILIKE '%PREFEITURA MUNICIPAL%'
          AND texto_completo NOT ILIKE '%ENERGIA ELÉTRICA%'
          AND texto_completo NOT ILIKE '%ENERGIA ELETRICA%'
          AND texto_completo NOT ILIKE '%SANEAMENTO%'
          AND texto_completo NOT ILIKE '%TELEFONIA%'
    """)

    candidates = []
    for doc_id, txt in cur.fetchall():
        s = str(doc_id)
        if s not in stem_to_idx or s in norm_hist:
            continue
        info = lookup.get(f"{s}.pdf", {})
        forn = info.get("fornecedor", "").upper()
        if any(w in forn for w in UTILITY_WORDS):
            continue

        idx = stem_to_idx[s]
        v = np.array(embs[idx], dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        sim = float(np.dot(v, nfe_cent))

        candidates.append({
            "doc_id": s,
            "sim": sim,
            "fornecedor": info.get("fornecedor", "N/A"),
            "ano": info.get("ano", 2024),
            "subcota": info.get("subcota", "")
        })

    # Ordenar por similaridade decrescente
    candidates.sort(key=lambda x: -x["sim"])
    print(f"Total de candidatos encontrados com DANFE: {len(candidates):,}")
    print(f"Similaridade máxima: {candidates[0]['sim']:.4f} | Mínima: {candidates[-1]['sim']:.4f}")

    # Selecionar os top 2.300 candidatos
    selected = candidates[:2300]
    selected_ids = [int(c["doc_id"]) for c in selected]

    # Consultar metadados financeiros detalhados (valor liquido)
    print("\nConsultando valores financeiros no PostgreSQL...")
    meta_db = {}
    chunk_size = 5000
    for i in range(0, len(selected_ids), chunk_size):
        chunk = selected_ids[i:i + chunk_size]
        cur.execute("""
            SELECT DISTINCT ON (ide_documento) ide_documento, fornecedor_nome, valor_liquido
            FROM vw_documento_detalhado
            WHERE ide_documento = ANY(%s)
        """, (chunk,))
        for r in cur.fetchall():
            meta_db[str(r[0])] = {
                "fornecedor": r[1] or "",
                "valor": float(r[2]) if r[2] is not None else 0.0
            }
    conn.close()

    # Preencher metadados finais
    for c in selected:
        s = c["doc_id"]
        db_info = meta_db.get(s, {})
        c["fornecedor"] = db_info.get("fornecedor") or c["fornecedor"]
        c["valor"] = db_info.get("valor", 0.0)

    # 4. Dividir em lotes operacionais
    lotes = [
        ("lote_01_top_1000", selected[0:1000]),
        ("lote_02_top_1000", selected[1000:2000]),
        ("lote_03_reserva_300", selected[2000:2300])
    ]

    # Criar link imediato no Desktop
    if DESKTOP_LINK.is_symlink() or DESKTOP_LINK.exists():
        DESKTOP_LINK.unlink()
    DESKTOP_LINK.symlink_to(TARGET_DIR)

    all_manifest = {}
    print("\nCriando pastas e gerando atalhos com miniaturas...")
    for folder_name, items in lotes:
        b_dir = TARGET_DIR / folder_name
        b_dir.mkdir(parents=True, exist_ok=True)
        desktop_b_dir = DESKTOP_LINK / folder_name

        print(f"\n   -> {folder_name}: {len(items):,} documentos (sim {items[0]['sim']:.4f} -> {items[-1]['sim']:.4f})...")
        t_batch = time.time()

        # Renderizar páginas ausentes
        missing = [it["doc_id"] for it in items if not (RENDERED_DIR / f"{it['doc_id']}.png").exists()]
        if missing:
            print(f"      Renderizando {len(missing)} páginas com 32 workers...")
            with ThreadPoolExecutor(max_workers=32) as ex:
                list(ex.map(render_page_if_needed, missing))

        # Links e miniaturas
        thumb_tasks = []
        for rank, it in enumerate(items, 1):
            s = it["doc_id"]
            clean_forn = sanitize_filename(it["fornecedor"])
            val_str = f"R${it['valor']:.2f}".replace(".", ",")
            sim_str = f"{it['sim']:.3f}"
            link_name = f"rank_{rank:04d}_[sim_{sim_str}]_[{clean_forn}]_[{val_str}]_{s}.png"

            target_png = RENDERED_DIR / f"{s}.png"
            batch_link = b_dir / link_name
            desktop_link_file = desktop_b_dir / link_name

            if batch_link.exists() or batch_link.is_symlink():
                batch_link.unlink()
            if target_png.exists():
                batch_link.symlink_to(target_png)

            thumb_tasks.append(([batch_link, desktop_link_file], target_png))

            all_manifest[s] = {
                "doc_id": s,
                "folder": folder_name,
                "rank": rank,
                "filename": link_name,
                "sim": it["sim"],
                "fornecedor": it["fornecedor"],
                "valor": it["valor"]
            }

        # Miniaturas
        with ThreadPoolExecutor(max_workers=32) as ex:
            ex.map(lambda t: setup_thumbnails(t[0], t[1]), thumb_tasks)

        print(f"      [OK] Concluído em {time.time()-t_batch:.1f}s.")

    # 5. Criar script de confirmação
    create_confirm_script(TARGET_DIR, current_nfe)

    # 6. Criar LEIAME.txt
    create_readme(TARGET_DIR, lotes, current_nfe)

    # 7. Salvar manifesto JSON
    with open(TARGET_DIR / ".manifesto_meta_20k.json", "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"PROCESSO COMPLETO EM {time.time()-t0:.1f} SEGUNDOS!")
    print(f"Pasta disponível no Desktop: ~/Desktop/{DESKTOP_LINK.name}")
    print("=" * 70)


def create_confirm_script(target_dir: Path, initial_nfe: int):
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Confirmação dos documentos validados para atingir a meta de 20.000 NF-e.
Varre os arquivos PNG mantidos nesta pasta e adiciona ao annotations_history.json.

Uso:
    python confirmar_validacao.py
\"\"\"

import argparse
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from dados.labeling.labels_csv import write_labels_dir

HISTORY_FILE = PROJECT_DIR / "data" / "models" / "annotations_history.json"
ROTULOS_DIR = PROJECT_DIR / "data" / "rotulos"


def main():
    parser = argparse.ArgumentParser(description="Confirma os candidatos para a meta de 20k.")
    parser.add_argument("--lote", type=str, default=None, help="Confirmar apenas um lote específico")
    args = parser.parse_args()

    search_dir = BASE_DIR / args.lote if args.lote else BASE_DIR

    print("=== CONFIRMAÇÃO DA META DE 20.000 NF-E ===")
    png_files = list(search_dir.rglob("*.png"))
    print(f"Arquivos mantidos encontrados: {{len(png_files):,}}")

    if not png_files:
        print("[AVISO] Nenhum arquivo PNG encontrado.")
        return

    doc_ids = set()
    for p in png_files:
        m = re.search(r'_(\\d+)\\.png$', p.name)
        if m:
            doc_ids.add(m.group(1))

    print(f"IDs válidos identificados: {{len(doc_ids):,}}")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    for d_id in doc_ids:
        if d_id not in history or history[d_id] != "nfe":
            history[d_id] = "nfe"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    total_nfe = sum(1 for v in history.values() if v == "nfe")
    print(f"\\n[SUCESSO] {{added:,}} novas anotações adicionadas como 'nfe'.")
    print(f"          Total histórico de NF-e agora: {{total_nfe:,}} / 20.000!")

    if total_nfe >= 20000:
        print("          >>> PARABÉNS! META DE 20.000 NF-E ATINGIDA COM SUCESSO! <<<")

    print("\\nSincronizando com as partes CSV em data/rotulos/...")
    write_labels_dir(history, ROTULOS_DIR)
    print("[OK] Base de rótulos CSV sincronizada!")
    print("Agora basta me avisar no chat: 'confirmei o lote 20k, pode atualizar o modelo'.")


if __name__ == "__main__":
    main()
"""
    script_path = target_dir / "confirmar_validacao.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    script_path.chmod(0o755)


def create_readme(target_dir: Path, lotes: list, initial_nfe: int):
    needed = 20000 - initial_nfe
    readme = f"""PASTA DE VALIDAÇÃO VISUAL - META 20.000 NF-E (DANFE / MODELO 55)
========================================================================================

Meta do Projeto: Atingir 20.000 documentos rotulados como NF-e.
Situação Atual:  {initial_nfe:,} rotulados
Faltam:          {needed:,} documentos

CRITÉRIOS DE SELEÇÃO:
1. Contêm OBRIGATORIAMENTE o termo "DANFE" no texto extraído.
2. Filtro anti-contaminação: Zero NFC-e (Modelo 65), Zero NFS-e (Serviços), Zero concessionárias.
3. Classificados por ordem estrita de similaridade no espaço métrico ArcFace 128-D recém-treinado.

ESTRUTURA DOS LOTES:
"""
    for folder_name, items in lotes:
        readme += f"- {folder_name:24s}: {len(items):,d} documentos (sim: {items[0]['sim']:.4f} -> {items[-1]['sim']:.4f})\n"

    readme += f"""
COMO CHEGAR AOS 20K:
- Se você validar o 'lote_01_top_1000' (1.000 docs) e o 'lote_02_top_1000' (1.000 docs), 
  você já atinge 20.064 documentos (meta batida!).
- O 'lote_03_reserva_300' serve caso você exclua algum documento duvidoso.

COMO REVISAR:
1. Abra a pasta no Thunar ou visualizador de imagens.
2. Navegue com as setas (<- e ->).
3. Aperte DELETE se encontrar algo que não seja DANFE.
4. Ao concluir, execute:
     cd ~/Desktop/validacao_nfe_meta_20k && ./confirmar_validacao.py
   ou simplesmente me diga no chat:
     "validei o lote 20k, pode aplicar"
"""
    with open(target_dir / "LEIAME.txt", "w", encoding="utf-8") as f:
        f.write(readme)


if __name__ == "__main__":
    main()
