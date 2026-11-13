#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Triagem Rápida por OCR para os Candidatos de RECIBO (Cluster 1 e Cluster 2).
1. Carrega os candidatos de manifest_recibo_42k.json.
2. Executa RapidOCR em paralelo (20 workers) nos candidatos selecionados.
3. Classifica em:
   - 1_confirmados_ocr/: palavras-chave inequívocas de táxi, alimentação ou aluguel/condomínio
   - 2_outros_candidatos/: sem marcador textual forte (recibos manuscritos/desbotados)
   - 3_falsos_positivos_detectados/: marcadores evidentes de outras classes (DANFE, NFS-e, SAT)
4. Gera estrutura de triagem e script de confirmação automática.
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
RENDERED_DIR = DATA_DIR / "rendered_pages"
RECIBO_DIR = DATA_DIR / "validacao_batches" / "validacao_recibo"
MANIFEST_PATH = RECIBO_DIR / "manifest_recibo_42k.json"

# Expressões Regulares - Cluster 1: Táxi / Refeição / Pequenos Recibos
RE_C1_TAXI = re.compile(
    r'\bT[AÁ]XI\b|SINPETAXI|COOPERTAXI|UNITAXI|RADIOT[AÁ]XI|R[AÁ]DIO\s*T[AÁ]XI|'
    r'SINDICATO\s*D(OS|E)\s*TAXISTAS|ASSOCIA[CÇ][AÃ]O\s*D(OS|AS)\s*TAXISTAS|MOTORISTA|'
    r'CORRIDA\s*DE\s*T[AÁ]XI|TAL[AÃ]O\s*DE\s*T[AÁ]XI|TRANSPORTE\s*INDIVIDUAL|PLACA\b|'
    r'\b99\s*(POP|TAXI|APP)\b',
    re.IGNORECASE
)

RE_C1_ALIMENTACAO = re.compile(
    r'RESTAURANTE|CHURRASCARIA|PIZZARIA|LANCHONETE|REFEI[CÇ][AÃ]O|ALIMENTA[CÇ][AÃ]O|'
    r'BAR\s*E\s*LANCHES|GRILL|BUFFET|FORNECIMENTO\s*DE\s*ALIMENTOS',
    re.IGNORECASE
)

RE_C1_RECIBO_GENERICO = re.compile(
    r'\bRECIBO\b|RECEBI(MOS)?\s*D[EO]|VALOR\s*DE\s*R\$|IMPORT[AÂ]NCIA\s*DE|'
    r'REFERENTE\s*A|PAGO\s*A|ASSINATURA\s*DO\s*EMITENTE|QUITA[CÇ][AÃ]O',
    re.IGNORECASE
)

# Expressões Regulares - Cluster 2: Imóveis / Locação / Condomínio / Escritório
RE_C2_IMOVEIS = re.compile(
    r'CONDOM[IÍ]NIO|RECIBO\s*DE\s*ALUGUEL|ALUGUEL|TAXA\s*CONDOMINIAL|EDIF[IÍ]CIO|'
    r'SALA\s*COMERCIAL|IM[OÓ]VEL|IMOBILI[AÁ]RI[AO]|ADMINISTRADORA|LOCAT[AÁ]RI[AO]|'
    r'LOCADOR\b|IPTU|TAXA\s*DE\s*MANUTEN[CÇ][AÃ]O',
    re.IGNORECASE
)

RE_C2_LOCACAO_VEICULOS = re.compile(
    r'LOCA[CÇ][AÃ]O\s*DE\s*VE[IÍ]CULOS?|LOCADORA|RENT\s*A\s*CAR|FATURA\s*DE\s*LOCA[CÇ][AÃ]O|'
    r'CONTRATO\s*DE\s*LOCA[CÇ][AÃ]O|FRETAMENTO|DI[AÁ]RIAS?\s*DE\s*LOCA[CÇ][AÃ]O',
    re.IGNORECASE
)

# Falsos Positivos Evidentes (outras classes fiscais)
RE_FALSE_POSITIVES = re.compile(
    r'\bDANFE\b|DOCUMENTO\s*AUXILIAR\s*DA\s*NOTA\s*FISCAL|CHAVE\s*DE\s*ACESSO|\b\d{44}\b|'
    r'\bNFS-?E\b|NOTA\s*FISCAL\s*DE\s*SERVI[CÇ]OS\s*ELETR[OÔ]NICA|NOTA\s*FISCAL\s*ELETR[OÔ]NICA|'
    r'CUPOM\s*FISCAL\s*ELETR[OÔ]NICO|EXTRATO\s*DO\s*SAT|CUPOM\s*MANAFATURADO',
    re.IGNORECASE
)

engine = None


def init_ocr_worker():
    global engine
    engine = RapidOCR(use_cls=False, det_limit_side_len=480, intra_op_num_threads=1, inter_op_num_threads=1)


def process_doc_ocr(task):
    global engine
    stem, png_path_str, cluster = task
    png_p = Path(png_path_str)
    if not png_p.exists():
        return stem, "ERRO_SEM_PNG", "", ""

    try:
        img = cv2.imread(str(png_p))
        if img is None:
            return stem, "ERRO_IMAGEM", "", ""

        # O cabeçalho / corpo do documento de recibo geralmente está na metade superior
        h = img.shape[0]
        crop = img[:int(h * 0.70), :]

        dt_boxes, _ = engine.text_det(crop)
        if dt_boxes is None or len(dt_boxes) == 0:
            return stem, "OUTROS", "SEM_TEXTO", ""

        dt_boxes = sorted(dt_boxes, key=lambda b: np.min(b[:, 1]))[:12]
        img_crops = engine.get_crop_img_list(crop, dt_boxes)

        rec_res, _ = engine.text_rec(img_crops)
        if not rec_res:
            return stem, "OUTROS", "SEM_TEXTO", ""

        full_text = " ".join([r[0] for r in rec_res])

        # Verificar se é falso positivo
        m_fp = RE_FALSE_POSITIVES.search(full_text)
        if m_fp:
            return stem, "FALSO_POSITIVO", m_fp.group(0), full_text[:120]

        # Verificar regras específicas do cluster
        if cluster == "cluster_1_taxi_alimentacao":
            m_taxi = RE_C1_TAXI.search(full_text)
            if m_taxi:
                return stem, "CONFIRMADO_OCR", f"TAXI:{m_taxi.group(0)}", full_text[:120]
            m_ali = RE_C1_ALIMENTACAO.search(full_text)
            if m_ali:
                return stem, "CONFIRMADO_OCR", f"ALIM:{m_ali.group(0)}", full_text[:120]
            m_rec = RE_C1_RECIBO_GENERICO.search(full_text)
            if m_rec:
                return stem, "CONFIRMADO_OCR", f"RECIBO:{m_rec.group(0)}", full_text[:120]

        elif cluster == "cluster_2_imoveis_locacao":
            m_imob = RE_C2_IMOVEIS.search(full_text)
            if m_imob:
                return stem, "CONFIRMADO_OCR", f"IMOVEIS:{m_imob.group(0)}", full_text[:120]
            m_loc = RE_C2_LOCACAO_VEICULOS.search(full_text)
            if m_loc:
                return stem, "CONFIRMADO_OCR", f"LOCACAO:{m_loc.group(0)}", full_text[:120]
            m_rec = RE_C1_RECIBO_GENERICO.search(full_text)
            if m_rec:
                return stem, "CONFIRMADO_OCR", f"RECIBO:{m_rec.group(0)}", full_text[:120]

        return stem, "OUTROS", "SEM_TAG_FORTE", full_text[:120]
    except Exception as e:
        return stem, "ERRO", str(e), ""


def run_triage(limit_per_cluster: int = 1500, only_99pct: bool = False):
    print("=" * 72)
    print(" TRIAGEM OCR RÁPIDA: RECIBOS (CLUSTER 1 & CLUSTER 2)")
    print("=" * 72)
    t0 = time.time()

    if not MANIFEST_PATH.exists():
        print(f"Erro: {MANIFEST_PATH} não encontrado. Aguarde a criação das pastas.")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    c1_records = [r for r in all_records if r["cluster"] == "cluster_1_taxi_alimentacao"]
    c2_records = [r for r in all_records if r["cluster"] == "cluster_2_imoveis_locacao"]

    if only_99pct:
        c1_selected = [r for r in c1_records if r["tier"] == "99pct"]
        c2_selected = [r for r in c2_records if r["tier"] == "99pct"]
    else:
        c1_selected = c1_records[:limit_per_cluster]
        c2_selected = c2_records[:limit_per_cluster]

    print(f"Candidatos selecionados para Triagem OCR:")
    print(f"  • Cluster 1 (Táxi / Refeição) : {len(c1_selected):,} documentos")
    print(f"  • Cluster 2 (Imóveis / Locação): {len(c2_selected):,} documentos")
    print(f"  • Total para processar        : {len(c1_selected) + len(c2_selected):,} documentos")

    tasks = []
    for r in c1_selected:
        png_p = RENDERED_DIR / f"{r['doc_id']}.png"
        tasks.append((r["doc_id"], str(png_p), "cluster_1_taxi_alimentacao"))
    for r in c2_selected:
        png_p = RENDERED_DIR / f"{r['doc_id']}.png"
        tasks.append((r["doc_id"], str(png_p), "cluster_2_imoveis_locacao"))

    print(f"\nIniciando OCR híbrido com 20 workers...")
    results_map = {}
    done = 0
    with ProcessPoolExecutor(max_workers=20, initializer=init_ocr_worker) as pool:
        for res in pool.map(process_doc_ocr, tasks, chunksize=16):
            stem, status, tag, snippet = res
            results_map[stem] = (status, tag, snippet)
            done += 1
            if done % 500 == 0 or done == len(tasks):
                print(f"  [{done:,}/{len(tasks):,}] processados em {time.time()-t0:.1f}s...", flush=True)

    # Organizar pastas de triagem
    def organize_cluster_triage(records, cluster_dir_name, cluster_name_label):
        triage_dir = RECIBO_DIR / cluster_dir_name / "triagem_ocr"
        dir_ok = triage_dir / "1_confirmados_ocr"
        dir_other = triage_dir / "2_outros_candidatos"
        dir_fp = triage_dir / "3_falsos_positivos_detectados"

        dir_ok.mkdir(parents=True, exist_ok=True)
        dir_other.mkdir(parents=True, exist_ok=True)
        dir_fp.mkdir(parents=True, exist_ok=True)

        cnt_ok, cnt_other, cnt_fp = 0, 0, 0
        triage_manifest = []

        for r in records:
            stem = r["doc_id"]
            if stem not in results_map:
                continue
            status, tag, snippet = results_map[stem]
            src_png = RENDERED_DIR / f"{stem}.png"
            if not src_png.exists():
                continue

            sim_val = r["similarity"]
            forn_tag = re.sub(r"[^\w\s-]", "", r.get("fornecedor", "")).strip().replace(" ", "_")[:18]
            fname = f"sim_{sim_val:.3f}_[{tag}]_{forn_tag}_{stem}.png"

            if status == "CONFIRMADO_OCR":
                dst = dir_ok / fname
                cnt_ok += 1
            elif status == "FALSO_POSITIVO":
                dst = dir_fp / fname
                cnt_fp += 1
            else:
                dst = dir_other / fname
                cnt_other += 1

            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src_png)

            triage_manifest.append({
                "doc_id": stem,
                "status": status,
                "tag": tag,
                "similarity": sim_val,
                "fornecedor": r.get("fornecedor", ""),
                "snippet": snippet
            })

        # Salvar confirmação automática nesta pasta
        script_confirmar = triage_dir / "confirmar_apenas_ocr.py"
        with open(script_confirmar, "w", encoding="utf-8") as f:
            f.write(f"""#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
\"\"\"
Confirma os documentos presentes em 1_confirmados_ocr/ como 'recibo'.
\"\"\"
import re
import json
from pathlib import Path

BASE_DIR = Path("/home/lucas.ocunha/masters")
HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
CURRENT_DIR = Path(__file__).resolve().parent
DIR_OK = CURRENT_DIR / "1_confirmados_ocr"

def main():
    pngs = list(DIR_OK.glob("*.png"))
    stems = set()
    for p in pngs:
        m = re.search(r'(\\d{{7,8}})\\.png$', p.name)
        if m:
            stems.add(m.group(1))

    print(f"Confirmando {{len(stems):,}} documentos de 1_confirmados_ocr/...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    added = 0
    for s in stems:
        if s not in history or history[s] != "recibo":
            history[s] = "recibo"
            added += 1

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"✓ {{added:,}} novos recibos adicionados ao histórico!")

if __name__ == "__main__":
    main()
""")
        script_confirmar.chmod(0o755)

        with open(triage_dir / "triage_manifest.json", "w", encoding="utf-8") as f:
            json.dump(triage_manifest, f, indent=2, ensure_ascii=False)

        print(f"\n{cluster_name_label}:")
        print(f"  ├── 1_confirmados_ocr         : {cnt_ok:5,d} documentos ({cnt_ok/len(records)*100:.1f}%)")
        print(f"  ├── 2_outros_candidatos       : {cnt_other:5,d} documentos ({cnt_other/len(records)*100:.1f}%)")
        print(f"  └── 3_falsos_positivos        : {cnt_fp:5,d} documentos ({cnt_fp/len(records)*100:.1f}%)")

    organize_cluster_triage(c1_selected, "1_taxi_e_alimentacao", "Cluster 1 (Táxi / Refeição)")
    organize_cluster_triage(c2_selected, "2_imoveis_e_locacao", "Cluster 2 (Imóveis / Locação)")

    print("\n" + "=" * 72)
    print(" TRIAGEM OCR FINALIZADA COM SUCESSO!")
    print(f"Tempo total: {time.time()-t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triagem rápida por OCR para Recibos")
    parser.add_argument("--limit", type=int, default=1500, help="Limite por cluster (default: 1500)")
    parser.add_argument("--all-99", action="store_true", help="Triar todos os documentos >= 99% de similaridade")
    args = parser.parse_args()
    run_triage(limit_per_cluster=args.limit, only_99pct=args.all_99)
