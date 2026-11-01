#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Pipeline de Extração OCR e Classificação para os Candidatos de BOLETO (9,199) e NF-e (13,083).
1. Identifica os candidatos não rotulados com similaridade >= 0.95 no espaço métrico ArcFace 128-D.
2. Renderiza em paralelo os PNGs faltantes em data/rendered_pages/.
3. Extração híbrida de alta performance:
   - Texto digital nativo (PyMuPDF) em PDFs com texto limpo.
   - RapidOCR em pool de 24 workers nos documentos escaneados e rasterizados.
4. Classificação fiscal automática e separação de falsos positivos.
5. Persistência completa no PostgreSQL (documento, documento_ocr_pagina, documento_texto_completo).
6. Organização automática em pastas na Área de Trabalho com thumbnails:
   - Desktop/validacao_boletos_9k/
   - Desktop/validacao_nfe_13k/
"""

import os
import sys
import re
import json
import gzip
import time
import signal
import shutil
import hashlib
import urllib.parse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import cv2
import pymupdf
import psycopg
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
EMBS_PATH = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
HIST_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")
THUMB_DIR = Path(os.path.expanduser("~/.cache/thumbnails/large"))
THUMB_DIR.mkdir(parents=True, exist_ok=True)

stop_requested = False


def sig_handler(signum, frame):
    global stop_requested
    print(f"\n[AVISO] Sinal {signum} recebido. Concluindo operações antes de encerrar...", flush=True)
    stop_requested = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

# Regexes Fiscais
RE_BOLETO = re.compile(
    r'RECIBO\s*D[OE]\s*(SACADO|PAGADOR)|FICHA\s*DE\s*COMPENSA[CÇ][AÃ]O|'
    r'LOCAL\s*DE\s*PAGAMENTO|PAG[AÁ]VEL\s*EM\s*QUALQUER\s*BANCO|'
    r'LINHA\s*DIGIT[AÁ]VEL|AUTENTICA[CÇ][AÃ]O\s*MEC[AÂ]NICA|NOSSO\s*N[UÚ]MERO|'
    r'AG[EÊ]NCIA\s*/\s*C[OÓ]DIGO\s*(DO\s*)?BENEFICI[AÁ]RIO|'
    r'BANCO\s*(DO\s*BRASIL|ITAU|ITAÚ|BRADESCO|SANTANDER|CAIXA|SICOOB|SICREDI|SAFRA|BANRISUL)|'
    r'CEDENTE|SACADO|BENEFICI[AÁ]RIO|PAGADOR|VALOR\s*DO\s*DOCUMENTO',
    re.IGNORECASE
)

RE_DANFE = re.compile(
    r'DOCUMENTO\s*AUXILIAR\s*DA\s*NOTA\s*FISCAL\s*ELETR[OÔ]NICA|'
    r'DANFE\b(?![\s\w]*NFC)|CHAVE\s*DE\s*ACESSO|'
    r'BASE\s*DE\s*C[AÁ]LCULO\s*D[OE]\s*ICMS|VALOR\s*DO\s*ICMS|'
    r'INSCRI[ÇC][ÃA]O\s*ESTADUAL\s*DO\s*SUBST|'
    r'PROTOCOLO\s*DE\s*AUTORIZA[ÇC][ÃA]O\s*DE\s*USO|'
    r'DADOS\s*DOS\s*PRODUTOS\s*/\s*SERVI[ÇC]OS',
    re.IGNORECASE
)

RE_NFSE = re.compile(
    r'NOTA\s*FISCAL\s*DE\s*SERVI[ÇC]O|NFS-?E\b|DANFSE|PREFEITURA\s*MUNICIPAL|'
    r'SECRETARIA\s*MUNICIPAL\s*D[AE]\s*(FAZENDA|FINAN[ÇC]AS)|ISSQN|'
    r'IMPOSTO\s*SOBRE\s*SERVI[ÇC]OS|TRIBUTA[ÇC][ÃA]O\s*MUNICIPAL|'
    r'TOMADOR\s*DE\s*SERVI[ÇC]OS|PRESTADOR\s*DE\s*SERVI[ÇC]OS|'
    r'DISCRIMINA[ÇC][ÃA]O\s*DOS\s*SERVI[ÇC]OS|VALOR\s*(TOTAL\s*)?DOS\s*SERVI[ÇC]OS|'
    r'RECIBO\s*PROVIS[OÓ]RIO\s*DE\s*SERVI[ÇC]OS|RPS\b',
    re.IGNORECASE
)

RE_CUPOM = re.compile(
    r'NFC-?E\b|DANFE\s*NFC|CONSUMIDOR\s*ELETR[OÔ]NIC|'
    r'CUP[O0D]M[\s_-]*[FE1][I1L]SCAL|EXTRATO\s*N[ºO\.]|MOD(ELO)?\s*(65|59)|'
    r'C00\b|COO\s*[:\.]|CCF\s*[:\.]|ECF\b',
    re.IGNORECASE
)

engine = None


def init_ocr_worker():
    global engine
    engine = RapidOCR(use_cls=False, det_limit_side_len=640, intra_op_num_threads=1, inter_op_num_threads=1)


def render_worker_task(task):
    doc_name, pdf_path_str, png_path_str = task
    try:
        png_p = Path(png_path_str)
        if png_p.exists():
            return True
        pdf_p = Path(pdf_path_str)
        if not pdf_p.exists():
            return False
        doc = pymupdf.open(str(pdf_p))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=110, alpha=False)
        pix.save(str(png_p))
        doc.close()
        return True
    except Exception:
        return False


def is_valid_digital_text(txt: str) -> bool:
    txt_clean = txt.strip()
    if len(txt_clean) < 35:
        return False
    alnum_count = sum(1 for c in txt_clean if c.isalnum() or c.isspace())
    if (alnum_count / len(txt_clean)) < 0.65:
        return False
    alpha_count = sum(1 for c in txt_clean if c.isalpha())
    if alpha_count < 15:
        return False
    return True


def classify_boleto_doc(txt: str) -> str:
    if not txt or len(txt.strip()) < 10:
        return "outros_sem_marcador"
    if RE_BOLETO.search(txt):
        return "boleto_confirmado"
    if RE_DANFE.search(txt):
        return "danfe_nfe"
    if RE_NFSE.search(txt):
        return "nfse"
    if RE_CUPOM.search(txt):
        return "cupom_nfce"
    return "outros_sem_marcador"


def classify_nfe_doc(txt: str) -> str:
    if not txt or len(txt.strip()) < 10:
        return "outros_sem_marcador"
    if RE_DANFE.search(txt):
        return "nfe_confirmada"
    if RE_CUPOM.search(txt):
        return "cupom_nfce"
    if RE_NFSE.search(txt):
        return "nfse"
    if RE_BOLETO.search(txt):
        return "boleto"
    return "outros_sem_marcador"


def process_ocr_task(task):
    global engine
    doc_id, png_path_str, target_class = task
    t0 = time.time()
    try:
        img = cv2.imread(png_path_str)
        if img is None:
            return doc_id, "", 0, "erro_imagem", 0

        h = img.shape[0]
        crop = img[:int(h * 0.75), :]
        res, _ = engine(crop)
        t_elapsed_ms = int((time.time() - t0) * 1000)

        if not res:
            return doc_id, "", 0, "outros_sem_marcador", t_elapsed_ms

        full_txt = " ".join([line[1] for line in res])
        char_count = len(full_txt)

        if target_class == "boleto":
            cls = classify_boleto_doc(full_txt)
        else:
            cls = classify_nfe_doc(full_txt)

        return doc_id, full_txt, char_count, cls, t_elapsed_ms
    except Exception as e:
        t_elapsed_ms = int((time.time() - t0) * 1000)
        return doc_id, "", 0, f"erro_ocr: {e}", t_elapsed_ms


def setup_thumbnail(image_path: Path):
    try:
        uri = "file://" + urllib.parse.quote(str(image_path.resolve()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_p = THUMB_DIR / f"{h}.png"
        if thumb_p.exists():
            return True
        doc = pymupdf.open(str(image_path))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_p))
        doc.close()
        return True
    except Exception:
        return False


def main():
    t_start = time.time()
    print("=" * 76)
    print(" PIPELINE DE EXTRAÇÃO RAPID OCR & VALIDAÇÃO: BOLETO & NF-E")
    print("=" * 76)

    # 1. Carregar Espaço Métrico e Selecionar Candidatos
    print("\n1. Carregando embeddings do espaço métrico 128-D e metadados...")
    embs = np.load(EMBS_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)
    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)
    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    doc_stems = [Path(d["doc_name"]).stem for d in docs]
    labeled_set = set(history.keys())

    # Calcular centróides
    c_boleto = embs[[i for i, s in enumerate(doc_stems) if history.get(s) == "boleto"]].mean(axis=0)
    c_boleto /= np.linalg.norm(c_boleto)

    c_nfe = embs[[i for i, s in enumerate(doc_stems) if history.get(s) == "nfe"]].mean(axis=0)
    c_nfe /= np.linalg.norm(c_nfe)

    sims_b = np.dot(embs, c_boleto)
    sims_n = np.dot(embs, c_nfe)

    # Identificar candidatos elegíveis (não rotulados e sim >= 0.95)
    boleto_candidates = []
    nfe_candidates = []

    for i, s in enumerate(doc_stems):
        if s in labeled_set:
            continue
        sim_b = float(sims_b[i])
        sim_n = float(sims_n[i])
        dname = docs[i]["doc_name"]
        forn = lookup.get(dname, {}).get("fornecedor", "")

        if sim_b >= 0.95:
            tier = (
                "1_similares_98_a_100pct" if sim_b >= 0.98 else
                "2_similares_97_a_98pct" if sim_b >= 0.97 else
                "3_similares_96_a_97pct" if sim_b >= 0.96 else
                "4_similares_95_a_96pct"
            )
            boleto_candidates.append({
                "doc_id": s,
                "doc_name": dname,
                "similarity": round(sim_b, 4),
                "tier": tier,
                "fornecedor": forn,
                "target_group": "boleto"
            })

        if sim_n >= 0.95:
            tier = (
                "1_similares_98_a_100pct" if sim_n >= 0.98 else
                "2_similares_97_a_98pct" if sim_n >= 0.97 else
                "3_similares_96_a_97pct" if sim_n >= 0.96 else
                "4_similares_95_a_96pct"
            )
            nfe_candidates.append({
                "doc_id": s,
                "doc_name": dname,
                "similarity": round(sim_n, 4),
                "tier": tier,
                "fornecedor": forn,
                "target_group": "nfe"
            })

    print(f"   ✓ Candidatos a Boleto selecionados (>= 0.95): {len(boleto_candidates):,}")
    print(f"   ✓ Candidatos a NF-e selecionados   (>= 0.95): {len(nfe_candidates):,}")

    all_candidates = boleto_candidates + nfe_candidates
    print(f"   Total conjunto de documentos a processar: {len(all_candidates):,}")

    # 2. Renderização em Paralelo das Páginas Faltantes
    print("\n2. Verificando e renderizando páginas pendentes em PNG...")
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    to_render = []
    for cand in all_candidates:
        s = cand["doc_id"]
        png_path = RENDERED_DIR / f"{s}.png"
        if not png_path.exists():
            pdf_path = PDF_DIR / cand["doc_name"]
            to_render.append((cand["doc_name"], str(pdf_path), str(png_path)))

    if to_render:
        print(f"   Renderizando {len(to_render):,} páginas pendentes com 24 workers...")
        t_rend = time.time()
        with ProcessPoolExecutor(max_workers=24) as pool:
            results = list(pool.map(render_worker_task, to_render, chunksize=64))
        print(f"   ✓ {sum(results):,} páginas renderizadas com sucesso em {time.time()-t_rend:.1f}s!")
    else:
        print("   ✓ Todas as páginas já estavam renderizadas em disco.")

    # 3. Fase 1: Análise de Texto Digital Nativo (PyMuPDF)
    print("\n3. Fase 1: Extração rápida de texto digital nativo (PyMuPDF)...")
    results_dict = {}
    to_ocr_tasks = []
    digital_count = 0

    t_fase1 = time.time()
    for cand in all_candidates:
        doc_id = cand["doc_id"]
        dname = cand["doc_name"]
        pdf_path = PDF_DIR / dname
        png_path = RENDERED_DIR / f"{doc_id}.png"
        tgt = cand["target_group"]

        txt_digital = ""
        if pdf_path.exists():
            try:
                doc = pymupdf.open(str(pdf_path))
                if len(doc) > 0:
                    txt_digital = doc[0].get_text()
                doc.close()
            except Exception:
                txt_digital = ""

        if is_valid_digital_text(txt_digital):
            cls = classify_boleto_doc(txt_digital) if tgt == "boleto" else classify_nfe_doc(txt_digital)
            results_dict[doc_id] = {
                "doc_id": doc_id,
                "doc_name": dname,
                "target_group": tgt,
                "tier": cand["tier"],
                "similarity": cand["similarity"],
                "fornecedor": cand["fornecedor"],
                "metodo": "digital_pdf",
                "texto": txt_digital.strip(),
                "num_caracteres": len(txt_digital.strip()),
                "classe": cls,
                "tempo_ms": 1
            }
            digital_count += 1
        else:
            to_ocr_tasks.append((doc_id, str(png_path), tgt))

    print(f"   Fase 1 concluída em {time.time() - t_fase1:.1f}s:")
    print(f"   • Textos digitais processados diretamente: {digital_count:,} ({digital_count/len(all_candidates)*100:.1f}%)")
    print(f"   • Documentos para RapidOCR (escaneados/fontes CID): {len(to_ocr_tasks):,} ({len(to_ocr_tasks)/len(all_candidates)*100:.1f}%)")

    # 4. Fase 2: Execução do RapidOCR nos Documentos Escaneados
    if to_ocr_tasks:
        num_workers = min(24, os.cpu_count() or 16)
        print(f"\n4. Fase 2: Executando RapidOCR em paralelo com {num_workers} workers...")
        ocr_start = time.time()
        processed_ocr = 0

        cands_by_id = {c["doc_id"]: c for c in all_candidates}

        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_ocr_worker) as executor:
            for res in executor.map(process_ocr_task, to_ocr_tasks, chunksize=16):
                if stop_requested:
                    print("\n[INFO] Parada antecipada solicitada. Interrompendo loop do OCR...")
                    break

                doc_id, full_txt, char_count, cls, t_ms = res
                meta_item = cands_by_id.get(doc_id, {})

                results_dict[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": meta_item.get("doc_name", f"{doc_id}.pdf"),
                    "target_group": meta_item.get("target_group", ""),
                    "tier": meta_item.get("tier", ""),
                    "similarity": meta_item.get("similarity", 0.0),
                    "fornecedor": meta_item.get("fornecedor", ""),
                    "metodo": "rapidocr",
                    "texto": full_txt,
                    "num_caracteres": char_count,
                    "classe": cls,
                    "tempo_ms": t_ms
                }
                processed_ocr += 1

                if processed_ocr % 500 == 0:
                    elapsed = time.time() - ocr_start
                    rate = processed_ocr / max(elapsed, 0.1)
                    rem = (len(to_ocr_tasks) - processed_ocr) / max(rate, 0.1)
                    print(f"   [RapidOCR] {processed_ocr:,}/{len(to_ocr_tasks):,} docs ({rate:.1f} doc/s | Restante: {rem/60:.1f} min)")

        print(f"   Fase 2 finalizada em {time.time() - ocr_start:.1f}s!")

    # 5. Sincronização no Banco PostgreSQL
    print(f"\n5. Sincronizando todos os {len(results_dict):,} resultados no PostgreSQL...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                doc_master_rows = []
                page_rows = []
                full_rows = []

                for doc_id, data in results_dict.items():
                    ide = int(doc_id)
                    dname = data["doc_name"]
                    ano = pdf_to_year.get(dname, 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{dname}"
                    txt = (data["texto"] or "").replace("\x00", "")
                    chars = len(txt)
                    t_ms = data["tempo_ms"]

                    doc_master_rows.append((ide, ano, caminho, "baixado", data["classe"]))
                    page_rows.append((ide, 1, "rapidocr", txt, chars, t_ms))
                    full_rows.append((ide, "rapidocr", 1, txt, chars, t_ms))

                batch_size = 2000
                for i in range(0, len(doc_master_rows), batch_size):
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            classe_predita = EXCLUDED.classe_predita;
                    """, doc_master_rows[i:i + batch_size])

                    cur.executemany("""
                        INSERT INTO documento_ocr_pagina 
                            (ide_documento, numero_pagina, modelo, texto_extraido, num_caracteres, tempo_processamento_ms)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento, numero_pagina, modelo) 
                        DO UPDATE SET 
                            texto_extraido = EXCLUDED.texto_extraido,
                            num_caracteres = EXCLUDED.num_caracteres,
                            tempo_processamento_ms = EXCLUDED.tempo_processamento_ms,
                            criado_em = NOW();
                    """, page_rows[i:i + batch_size])

                    cur.executemany("""
                        INSERT INTO documento_texto_completo 
                            (ide_documento, modelo, total_paginas, texto_completo, num_caracteres_total, tempo_total_ms)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento, modelo) 
                        DO UPDATE SET 
                            total_paginas = EXCLUDED.total_paginas,
                            texto_completo = EXCLUDED.texto_completo,
                            num_caracteres_total = EXCLUDED.num_caracteres_total,
                            tempo_total_ms = EXCLUDED.tempo_total_ms,
                            atualizado_em = NOW();
                    """, full_rows[i:i + batch_size])

                conn.commit()
                print(f"   ✓ {len(results_dict):,} registros sincronizados no PostgreSQL!")
    except Exception as e:
        print(f"   [AVISO] Erro PostgreSQL: {e}")

    # 6. Organizar Pastas no Desktop
    print("\n6. Organizando pastas estruturadas no Desktop para validação humana...")

    dest_boleto = DATA_DIR / "validacao_batches" / "validacao_boletos_9k"
    dest_nfe = DATA_DIR / "validacao_batches" / "validacao_nfe_13k"

    for dest in [dest_boleto, dest_nfe]:
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

    # Subpastas para Boletos
    (dest_boleto / "1_confirmados_ocr").mkdir(exist_ok=True)
    (dest_boleto / "2_outros_candidatos").mkdir(exist_ok=True)
    (dest_boleto / "3_falsos_positivos_detectados").mkdir(exist_ok=True)

    # Subpastas para NF-e
    (dest_nfe / "1_confirmados_ocr").mkdir(exist_ok=True)
    (dest_nfe / "2_outros_candidatos").mkdir(exist_ok=True)
    (dest_nfe / "3_falsos_positivos_detectados").mkdir(exist_ok=True)

    # Distribuir arquivos em lotes
    boleto_results = [d for d in results_dict.values() if d["target_group"] == "boleto"]
    nfe_results = [d for d in results_dict.values() if d["target_group"] == "nfe"]

    # Gravar resultados compactados
    with gzip.open(dest_boleto / "rapidocr_results.json.gz", "wt", encoding="utf-8") as f:
        json.dump(boleto_results, f)
    with gzip.open(dest_nfe / "rapidocr_results.json.gz", "wt", encoding="utf-8") as f:
        json.dump(nfe_results, f)

    # Criar hardlinks organizados para Boletos
    for item in boleto_results:
        doc_id = item["doc_id"]
        cls = item["classe"]
        src_png = RENDERED_DIR / f"{doc_id}.png"
        if not src_png.exists():
            continue

        if cls == "boleto_confirmado":
            sub = "1_confirmados_ocr"
        elif cls == "outros_sem_marcador":
            sub = "2_outros_candidatos"
        else:
            sub = "3_falsos_positivos_detectados"

        dst_png = dest_boleto / sub / f"{doc_id}.png"
        try:
            os.link(src_png, dst_png)
        except Exception:
            shutil.copy2(src_png, dst_png)
        setup_thumbnail(dst_png)

    # Criar hardlinks organizados para NF-e
    for item in nfe_results:
        doc_id = item["doc_id"]
        cls = item["classe"]
        src_png = RENDERED_DIR / f"{doc_id}.png"
        if not src_png.exists():
            continue

        if cls == "nfe_confirmada":
            sub = "1_confirmados_ocr"
        elif cls == "outros_sem_marcador":
            sub = "2_outros_candidatos"
        else:
            sub = "3_falsos_positivos_detectados"

        dst_png = dest_nfe / sub / f"{doc_id}.png"
        try:
            os.link(src_png, dst_png)
        except Exception:
            shutil.copy2(src_png, dst_png)
        setup_thumbnail(dst_png)

    # Links no Desktop
    link_b = DESKTOP_DIR / "validacao_boletos_9k"
    if link_b.exists() or link_b.is_symlink():
        link_b.unlink()
    link_b.symlink_to(dest_boleto)

    link_n = DESKTOP_DIR / "validacao_nfe_13k"
    if link_n.exists() or link_n.is_symlink():
        link_n.unlink()
    link_n.symlink_to(dest_nfe)

    # 7. Relatório Final
    b_counts = Counter(d["classe"] for d in boleto_results)
    n_counts = Counter(d["classe"] for d in nfe_results)

    print("\n" + "=" * 76)
    print(" RELATÓRIO ANALÍTICO DA EXTRAÇÃO & CLASSIFICAÇÃO")
    print("=" * 76)
    print("1. RESULTADOS PARA BOLETO (9.199 Candidatos):")
    for cls, cnt in b_counts.most_common():
        print(f"   • {cls:28s}: {cnt:5,} ({cnt / len(boleto_results) * 100:.1f}%)")

    print("\n2. RESULTADOS PARA NF-e (13.083 Candidatos):")
    for cls, cnt in n_counts.most_common():
        print(f"   • {cls:28s}: {cnt:5,} ({cnt / len(nfe_results) * 100:.1f}%)")

    print(f"\nPastas criadas no Desktop:")
    print(f"  • {link_b}")
    print(f"  • {link_n}")
    print(f"Tempo Total: {time.time() - t_start:.1f}s ({(time.time() - t_start)/60:.2f} min)")
    print("=" * 76)


if __name__ == "__main__":
    main()
