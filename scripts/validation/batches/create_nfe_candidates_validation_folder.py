#!/usr/bin/env python3
"""
Gera a pasta de validação visual para novos candidatos NF-e (Modelo 55 / DANFE).
Total: ~18.000 documentos organizados em subpastas de ~3.000 documentos.

Pastas:
  Desktop/validacao_nfe_novos_candidatos/
    ├── 01_confirmados_sefaz_danfe_2.9k/    (Tier 1: Chave de acesso SEFAZ Mod 55 / Protocolo DANFE)
    ├── 02_alta_confianca_metrica_lote1_3k/ (Tier 2: ArcFace sim 0.999..0.985)
    ├── 03_alta_confianca_metrica_lote2_3k/ (Tier 2: ArcFace sim 0.985..0.975)
    ├── 04_alta_confianca_metrica_lote3_3k/ (Tier 2: ArcFace sim 0.975..0.968)
    ├── 05_alta_confianca_metrica_lote4_3k/ (Tier 2: ArcFace sim 0.968..0.963)
    ├── 06_alta_confianca_metrica_lote5_3.1k/(Tier 2: ArcFace sim 0.963..0.960)
    ├── confirmar_validacao.py
    ├── LEIAME.txt
    └── .candidatos_manifesto.json
"""

import hashlib
import json
import os
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
import pymupdf

from dados.config import DESKTOP_DIR, PDF_DIR, ROOT_DIR

DATA_DIR = ROOT_DIR / "data"
BATCHES_DIR = DATA_DIR / "validacao_batches"
TARGET_DIR = BATCHES_DIR / "validacao_nfe_novos_candidatos"
DESKTOP_LINK = DESKTOP_DIR / "validacao_nfe_novos_candidatos"
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

SEFAZ_WEIGHTS = [4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]


def sanitize_filename(name: str, max_len: int = 22) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] if cleaned else "DESCONHECIDO"


def verify_sefaz_key_mod55(text: str) -> Tuple[bool, str]:
    cands = re.findall(r"(?:[0-9][\s\.\-]*){44}", text)
    for c in cands:
        digits = [int(ch) for ch in c if ch.isdigit()]
        if len(digits) == 44:
            soma = sum(d * w for d, w in zip(digits[:43], SEFAZ_WEIGHTS))
            resto = soma % 11
            dv = 0 if resto in (0, 1) else 11 - resto
            if dv == digits[43] and digits[20] == 5 and digits[21] == 5:
                return True, "".join(str(d) for d in digits)
    return False, ""


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
    print("MINERAÇÃO E ESTRUTURAÇÃO DE CANDIDATOS NF-E (MODELO 55 / DANFE)")
    print("=" * 70)

    # 1. Carregar histórico e metadados
    print("1. Carregando anotações existentes e metadados...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        hist = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in hist.items()}
    print(f"   Anotações existentes: {len(norm_hist):,} documentos (NF-e atuais: {sum(1 for v in norm_hist.values() if v == 'nfe'):,})")

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stems = [d["doc_name"].replace(".pdf", "") for d in meta["documents"]]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    lookup = {}
    if LOOKUP_FILE.exists():
        with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
            lookup = json.load(f)

    # 2. Mineração PostgreSQL (Tier 1: Certificados SEFAZ Mod 55 / DANFE)
    print("\n2. Consultando PostgreSQL para documentos com Chave SEFAZ Mod 55 / Protocolo DANFE...")
    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/ceap")
    cur = conn.cursor("cur_nfe_candidates")
    cur.execute("""
        SELECT ide_documento, texto_completo 
        FROM documento_texto_completo 
        WHERE (texto_completo ILIKE '%DANFE%' 
               OR texto_completo ILIKE '%CHAVE DE ACESSO%'
               OR texto_completo ILIKE '%MODELO 55%'
               OR texto_completo ILIKE '%MOD. 55%')
          AND texto_completo NOT ILIKE '%NFC-E%'
          AND texto_completo NOT ILIKE '%CONSUMIDOR ELETR%'
          AND texto_completo NOT ILIKE '%MODELO 65%'
          AND texto_completo NOT ILIKE '%ENERGIA ELÉTRICA%'
          AND texto_completo NOT ILIKE '%ENERGIA ELETRICA%'
          AND texto_completo NOT ILIKE '%SANEAMENTO%'
          AND texto_completo NOT ILIKE '%TELEFONIA%'
    """)

    cert_pg_docs: Dict[str, dict] = {}
    while True:
        rows = cur.fetchmany(10000)
        if not rows:
            break
        for doc_id, txt in rows:
            s = str(doc_id)
            if s not in stem_to_idx or s in norm_hist:
                continue
            info = lookup.get(f"{s}.pdf", {})
            forn = info.get("fornecedor", "").upper()
            if any(w in forn for w in UTILITY_WORDS):
                continue
            txt_u = txt.upper()
            is_55, key_found = verify_sefaz_key_mod55(txt)
            if not is_55 and "DANFE" in txt_u and ("PROTOCOLO" in txt_u or "NATUREZA DA OPERA" in txt_u):
                is_55 = True
            if is_55:
                cert_pg_docs[s] = {"source": "sefaz_danfe", "key": key_found}

    cur.close()
    print(f"   Tier 1 (Certificados SEFAZ / DANFE): {len(cert_pg_docs):,} documentos")

    # 3. Espaço Métrico ArcFace (Tier 2: Alta similaridade métrica)
    print("\n3. Calculando similaridade no espaço métrico ArcFace 128-D...")
    embs = np.load(EMBS_FILE, mmap_mode="r")

    nfe_idxs = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == "nfe" and s in stem_to_idx]
    nfe_cent = np.array(embs[nfe_idxs]).mean(axis=0)
    nfe_cent /= np.linalg.norm(nfe_cent)

    class_names = ["cupom_fiscal", "nfce", "nfse", "uber", "energia", "agua", "boleto", "telefonia"]
    other_cents = []
    for c in class_names:
        c_idxs = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == c and s in stem_to_idx]
        cent = np.array(embs[c_idxs]).mean(axis=0)
        cent /= np.linalg.norm(cent)
        other_cents.append(cent)
    other_cents = np.array(other_cents)

    metric_docs: Dict[str, dict] = {}
    chunk_size = 100000
    for i in range(0, len(stems), chunk_size):
        chunk = np.array(embs[i:i + chunk_size], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        chunk /= norms
        s_nfe = np.dot(chunk, nfe_cent)
        s_other = np.dot(chunk, other_cents.T).max(axis=1)
        for j, (sn, so) in enumerate(zip(s_nfe, s_other)):
            idx = i + j
            s = stems[idx]
            if s in norm_hist:
                continue
            diff = sn - so
            if sn >= 0.96 and diff >= 0.40:
                info = lookup.get(f"{s}.pdf", {})
                forn = info.get("fornecedor", "").upper()
                if any(w in forn for w in UTILITY_WORDS):
                    continue
                metric_docs[s] = {"sim": float(sn), "diff": float(diff)}

    print(f"   Tier 2 (Métrico >= 0.96 e margem >= 0.40): {len(metric_docs):,} documentos")

    # 4. Consolidar e ordenar
    print("\n4. Consolidando candidatos e consultando detalhes financeiros...")
    all_selected_ids = list(set(cert_pg_docs.keys()).union(set(metric_docs.keys())))
    print(f"   Total combinado único de candidatos: {len(all_selected_ids):,}")

    meta_db = {}
    with conn.cursor() as cur:
        chunk_size = 5000
        for i in range(0, len(all_selected_ids), chunk_size):
            chunk = [int(x) for x in all_selected_ids[i:i + chunk_size]]
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

    # Montar lista Tier 1
    tier1_items = []
    for s, d in cert_pg_docs.items():
        m_info = metric_docs.get(s, {})
        db_info = meta_db.get(s, {})
        forn = db_info.get("fornecedor") or lookup.get(f"{s}.pdf", {}).get("fornecedor", "FORNECEDOR")
        val = db_info.get("valor", 0.0)
        tier1_items.append({
            "doc_id": s,
            "source": "sefaz_danfe",
            "sim": m_info.get("sim", 1.0),
            "diff": m_info.get("diff", 0.6),
            "fornecedor": forn,
            "valor": val,
            "key": d["key"]
        })
    tier1_items.sort(key=lambda x: -x["sim"])

    # Montar lista Tier 2 (apenas os que NÃO estão no Tier 1)
    tier2_stems = [s for s in metric_docs if s not in cert_pg_docs]
    tier2_items = []
    for s in tier2_stems:
        d = metric_docs[s]
        db_info = meta_db.get(s, {})
        forn = db_info.get("fornecedor") or lookup.get(f"{s}.pdf", {}).get("fornecedor", "FORNECEDOR")
        val = db_info.get("valor", 0.0)
        tier2_items.append({
            "doc_id": s,
            "source": "metric",
            "sim": d["sim"],
            "diff": d["diff"],
            "fornecedor": forn,
            "valor": val,
            "key": ""
        })
    tier2_items.sort(key=lambda x: -x["sim"])

    print(f"   Tier 1 final: {len(tier1_items):,} | Tier 2 complementar: {len(tier2_items):,}")

    # 5. Organizar em sub-lotes
    batches = [
        ("01_confirmados_sefaz_danfe_2.9k", tier1_items),
        ("02_alta_confianca_metrica_lote1_3k", tier2_items[0:3000]),
        ("03_alta_confianca_metrica_lote2_3k", tier2_items[3000:6000]),
        ("04_alta_confianca_metrica_lote3_3k", tier2_items[6000:9000]),
        ("05_alta_confianca_metrica_lote4_3k", tier2_items[9000:12000]),
        ("06_alta_confianca_metrica_lote5_3.1k", tier2_items[12000:])
    ]

    # 6. Renderizar e criar links lote por lote
    print("\n5. Criando pastas, renderizando páginas e criando atalhos com miniaturas...")
    desktop_equiv_base = DESKTOP_LINK

    all_manifest = {}
    total_docs = sum(len(items) for _, items in batches)
    processed_count = 0

    for folder_name, items in batches:
        b_dir = TARGET_DIR / folder_name
        b_dir.mkdir(parents=True, exist_ok=True)
        desktop_b_dir = desktop_equiv_base / folder_name

        print(f"\n   -> Processando pasta: {folder_name} ({len(items):,} docs)...")
        t_batch0 = time.time()

        # Renderizar páginas ausentes
        missing_ids = [it["doc_id"] for it in items if not (RENDERED_DIR / f"{it['doc_id']}.png").exists()]
        if missing_ids:
            print(f"      Renderizando {len(missing_ids):,} páginas ausentes com 32 workers...")
            with ThreadPoolExecutor(max_workers=32) as ex:
                list(ex.map(render_page_if_needed, missing_ids))
            print(f"      Renderização concluída em {time.time()-t_batch0:.1f}s.")

        # Criar symlinks e miniaturas
        thumb_tasks = []
        for rank, it in enumerate(items, 1):
            s = it["doc_id"]
            clean_forn = sanitize_filename(it["fornecedor"])
            val_str = f"R${it['valor']:.2f}".replace(".", ",")
            sim_str = f"{it['sim']:.3f}"
            tag = "SEFAZ" if it["source"] == "sefaz_danfe" else "METRICA"
            link_name = f"rank_{rank:04d}_[{tag}]_[sim_{sim_str}]_[{clean_forn}]_[{val_str}]_{s}.png"

            target_png = RENDERED_DIR / f"{s}.png"
            batch_link = b_dir / link_name

            if batch_link.exists() or batch_link.is_symlink():
                batch_link.unlink()
            if target_png.exists():
                batch_link.symlink_to(target_png)

            # Preparar miniaturas para o caminho da pasta real e para o Desktop
            desktop_link_file = desktop_b_dir / link_name
            thumb_tasks.append(([batch_link, desktop_link_file], target_png))

            all_manifest[s] = {
                "doc_id": s,
                "folder": folder_name,
                "rank": rank,
                "filename": link_name,
                "source": it["source"],
                "sim": it["sim"],
                "fornecedor": it["fornecedor"],
                "valor": it["valor"]
            }

        # Criar miniaturas em paralelo
        with ThreadPoolExecutor(max_workers=32) as ex:
            ex.map(lambda t: setup_thumbnails(t[0], t[1]), thumb_tasks)

        processed_count += len(items)
        print(f"      [OK] Pasta {folder_name} pronta ({processed_count:,}/{total_docs:,} total) em {time.time()-t_batch0:.1f}s.")

    # 7. Criar link simbólico no Desktop
    if DESKTOP_LINK.is_symlink() or DESKTOP_LINK.exists():
        DESKTOP_LINK.unlink()
    DESKTOP_LINK.symlink_to(TARGET_DIR)
    print(f"\n[OK] Link no Desktop criado: {DESKTOP_LINK} -> {TARGET_DIR}")

    # 8. Criar script de confirmação
    create_confirm_script(TARGET_DIR)

    # 9. Criar LEIAME.txt
    create_readme(TARGET_DIR, batches)

    # 10. Salvar manifesto JSON
    with open(TARGET_DIR / ".candidatos_manifesto.json", "w", encoding="utf-8") as f:
        json.dump(all_manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"PROCESSO COMPLETO EM {time.time()-t0:.1f} SEGUNDOS!")
    print(f"Pasta disponível no Desktop: ~/Desktop/{DESKTOP_LINK.name}")
    print("=" * 70)


def create_confirm_script(target_dir: Path):
    script_content = """#!/usr/bin/env python3
\"\"\"
Script de confirmação dos candidatos NF-e validados visualmente pelo usuário.
Varre os arquivos PNG que NÃO foram apagados nas pastas do lote,
adiciona ao data/models/annotations_history.json e sincroniza com data/rotulos/.

Uso:
    python confirmar_validacao.py
    python confirmar_validacao.py --pasta 01_confirmados_sefaz_danfe_2.9k
\"\"\"

import argparse
import json
import re
import sys
from pathlib import Path

# Ajustar PYTHONPATH para importar dados
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from dados.labeling.labels_csv import write_labels_dir

HISTORY_FILE = PROJECT_DIR / "data" / "models" / "annotations_history.json"
ROTULOS_DIR = PROJECT_DIR / "data" / "rotulos"


def main():
    parser = argparse.ArgumentParser(description="Confirma anotações NF-e validadas visualmente.")
    parser.add_argument("--pasta", type=str, default=None, help="Nome de uma subpasta específica para confirmar (opcional)")
    args = parser.parse_args()

    current_dir = BASE_DIR
    if args.pasta:
        search_dir = current_dir / args.pasta
        if not search_dir.exists():
            print(f"[ERRO] Pasta {args.pasta} não encontrada.")
            sys.exit(1)
    else:
        search_dir = current_dir

    print(f"=== CONFIRMAÇÃO DE CANDIDATOS NF-E ===")
    print(f"Varrendo: {search_dir}...")

    png_files = list(search_dir.rglob("*.png"))
    print(f"Total de arquivos mantidos encontrados: {len(png_files):,}")

    if not png_files:
        print("[AVISO] Nenhum arquivo PNG encontrado.")
        return

    doc_ids = set()
    for p in png_files:
        m = re.search(r'_(\\d+)\\.png$', p.name)
        if m:
            doc_ids.add(m.group(1))

    print(f"IDs válidos identificados: {len(doc_ids):,}")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    for d_id in doc_ids:
        if d_id not in history or history[d_id] != "nfe":
            history[d_id] = "nfe"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"[OK] {added:,} novas anotações adicionadas como 'nfe' ao annotations_history.json.")
    print(f"     Total histórico de NF-e agora: {sum(1 for v in history.values() if v == 'nfe'):,}")

    print(f"Sincronizando com as partes CSV em data/rotulos/...")
    write_labels_dir(history, ROTULOS_DIR)
    print(f"[OK] CSVs em data/rotulos/ atualizados com sucesso!")
    print(f"Validação concluída! Pode me avisar para prosseguir com o retreino ou atualização do modelo.")


if __name__ == "__main__":
    main()
"""
    script_path = target_dir / "confirmar_validacao.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    script_path.chmod(0o755)


def create_readme(target_dir: Path, batches: list):
    readme_content = f"""PASTAS DE VALIDAÇÃO VISUAL - NOVOS CANDIDATOS NF-E (DANFE / MODELO 55)
========================================================================================

Este lote contém documentos minerados rigorosamente para expansão da classe NF-e:
1. Tier 1 (01_confirmados_sefaz_danfe_2.9k):
   - 2.942 documentos com Chave de Acesso SEFAZ válida (Módulo 11) ou protocolo DANFE oficial.
   - Quase 100% de precisão.
2. Tier 2 (Pastas 02 a 06):
   - Documentos no espaço métrico ArcFace 128-D com similaridade >= 0.96 e margem >= 0.40.
   - Fornecedores típicos: postos de combustíveis (Cascol, Petrobras, Ipiranga, São Bernardo),
     papelarias e informática (Kalunga, WMS, EmySouto, Editora Globo).
   - Filtros de exclusão aplicados contra: energia, água, telefonia, táxi, Uber, NFC-e e NFS-e.

ESTRUTURA DAS PASTAS:
"""
    for folder_name, items in batches:
        readme_content += f"- {folder_name:38s}: {len(items):,d} documentos\n"

    readme_content += """
COMO REVISAR:
1. Abra a pasta correspondente no Thunar ou visualizador de imagens.
2. Navegue com as setas do teclado (← e →).
3. Se encontrar algum documento que NÃO seja NF-e (DANFE Folha A4 Retrato), aperte DELETE.
4. Quando terminar uma pasta (ou todas), basta executar no terminal:
     cd ~/Desktop/validacao_nfe_novos_candidatos
     python confirmar_validacao.py
   ou simplesmente me diga no chat:
     "validei a pasta X, pode aplicar"
"""
    with open(target_dir / "LEIAME.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)


if __name__ == "__main__":
    main()
