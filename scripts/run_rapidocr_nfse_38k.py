#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Pipeline de Extração OCR e Classificação Rápida dos 38.964 Candidatos NFS-e (Tiers 1 a 4).
- Processamento híbrido otimizado de alta velocidade:
    1. PDFs digitais com texto nativo limpo: extração via PyMuPDF (100% acurácia, 0ms overhead).
    2. PDFs escaneados, rasterizados ou com fontes corrompidas/CID: OCR via RapidOCR em pool paralelo de 24 workers.
- Classificação fiscal automática calibrada por regexes:
    - nfse_confirmada (Prefeitura, ISSQN, NFS-e, DANFSE, Tributação Municipal, etc.)
    - danfe_nfe (DANFE modelo 55 mercadorias/produtos)
    - cupom_nfce (NFC-e mod 65, SAT mod 59, ECF)
    - boleto (Ficha de compensação, recibo pagador/sacado)
    - concessionaria_utilidade (Energia, telecom, água)
    - outros_sem_marcador
- Persistência:
    - PostgreSQL: tabelas `documento_ocr_pagina` e `documento_texto_completo` (modelo='rapidocr')
    - Arquivo de resultados: `data/validacao_batches/validacao_nfse_38k/rapidocr_results.json.gz`
    - Manifesto enriquecido: `data/validacao_batches/validacao_nfse_38k/manifest_ocr.json.gz`
"""

import os
import sys
import re
import json
import gzip
import time
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import cv2
import pymupdf
import psycopg

# Otimização de threads de baixo nível
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
cv2.setNumThreads(1)

from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
VALIDATION_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_38k"
MANIFEST_FILE = VALIDATION_DIR / "manifest.json.gz"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
RENDERED_DIR = DATA_DIR / "rendered_pages"
RESULTS_FILE = VALIDATION_DIR / "rapidocr_results.json.gz"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

stop_requested = False


def sig_handler(signum, frame):
    global stop_requested
    print(f"\n[AVISO] Sinal de interrupção ({signum}) recebido. Concluindo lote e salvando...", flush=True)
    stop_requested = True


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

# ==============================================================================
# REGEXES FISCAIS CALIBRADAS
# ==============================================================================
RE_NFSE = re.compile(
    r'NOTA\s*FISCAL\s*DE\s*SERVI[ÇC]O|NFS-?E\b|DANFSE|PREFEITURA\s*MUNICIPAL|'
    r'SECRETARIA\s*MUNICIPAL\s*D[AE]\s*(FAZENDA|FINAN[ÇC]AS)|ISSQN|'
    r'IMPOSTO\s*SOBRE\s*SERVI[ÇC]OS|TRIBUTA[ÇC][ÃA]O\s*MUNICIPAL|'
    r'TOMADOR\s*DE\s*SERVI[ÇC]OS|PRESTADOR\s*DE\s*SERVI[ÇC]OS|'
    r'DISCRIMINA[ÇC][ÃA]O\s*DOS\s*SERVI[ÇC]OS|VALOR\s*(TOTAL\s*)?DOS\s*SERVI[ÇC]OS|'
    r'RECIBO\s*PROVIS[OÓ]RIO\s*DE\s*SERVI[ÇC]OS|RPS\b|'
    r'C[OÓ]DIGO\s*DE\s*VERIFICA[ÇC][ÃA]O|AUTENTICIDADE\s*DESTA\s*NOTA',
    re.IGNORECASE
)

RE_DANFE = re.compile(
    r'DOCUMENTO\s*AUXILIAR\s*DA\s*NOTA\s*FISCAL\s*ELETR[OÔ]NICA|'
    r'DANFE\b(?![\s\w]*NFC)|CHAVE\s*DE\s*ACESSO|'
    r'BASE\s*DE\s*C[AÁ]LCULO\s*D[OE]\s*ICMS|VALOR\s*DO\s*ICMS|'
    r'INSCRI[ÇC][ÃA]O\s*ESTADUAL\s*DO\s*SUBST|PROTOCOLO\s*DE\s*AUTORIZA[ÇC][ÃA]O\s*DE\s*USO',
    re.IGNORECASE
)

RE_CUPOM = re.compile(
    r'NFC-?E\b|DANFE\s*NFC|CONSUMIDOR\s*ELETR[OÔ]NIC|'
    r'CUP[O0D]M[\s_-]*[FE1][I1L]SCAL|EXTRATO\s*N[ºO\.]|MOD(ELO)?\s*(65|59)|'
    r'C00\b|COO\s*[:\.]|CCF\s*[:\.]|ECF\b',
    re.IGNORECASE
)

RE_BOLETO = re.compile(
    r'RECIBO\s*D[OE]\s*(SACADO|PAGADOR)|FICHA\s*DE\s*COMPENSA[CÇ][AÃ]O|'
    r'LOCAL\s*DE\s*PAGAMENTO|PAG[AÁ]VEL\s*EM\s*QUALQUER\s*BANCO|'
    r'LINHA\s*DIGIT[AÁ]VEL|AUTENTICA[CÇ][AÃ]O\s*MEC[AÂ]NICA|NOSSO\s*N[UÚ]MERO',
    re.IGNORECASE
)

RE_CONCESSIONARIA = re.compile(
    r'NOTA\s*FISCAL\s*/\s*CONTA\s*DE\s*ENERGIA|ENERGIA\s*EL[ÉI]TRICA|'
    r'CONSUMO\s*ATIVO|CONTA\s*DE\s*[ÁA]GUA|FATURA\s*/\s*DUPLICATA\s*DE\s*TELECOM',
    re.IGNORECASE
)

engine = None


def init_ocr_worker():
    global engine
    engine = RapidOCR(
        use_cls=False,
        det_limit_side_len=640,
        intra_op_num_threads=1,
        inter_op_num_threads=1
    )


def is_valid_digital_text(txt: str) -> bool:
    """Verifica se o texto digital extraído é legítimo ou ruído/fonte CID corrompida."""
    txt_clean = txt.strip()
    if len(txt_clean) < 35:
        return False
    # Verifica proporção de caracteres alfanuméricos normais
    alnum_count = sum(1 for c in txt_clean if c.isalnum() or c.isspace())
    if (alnum_count / len(txt_clean)) < 0.65:
        return False
    # Verifica se contém letras básicas (evita strings puramente de símbolos como !"##$%)
    alpha_count = sum(1 for c in txt_clean if c.isalpha())
    if alpha_count < 15:
        return False
    return True


def classify_text(txt: str) -> str:
    """Classifica o documento a partir do texto extraído segundo prioridades fiscais."""
    if not txt or len(txt.strip()) < 10:
        return "outros_sem_marcador"
    if RE_NFSE.search(txt):
        return "nfse_confirmada"
    if RE_DANFE.search(txt):
        return "danfe_nfe"
    if RE_CUPOM.search(txt):
        return "cupom_nfce"
    if RE_BOLETO.search(txt):
        return "boleto"
    if RE_CONCESSIONARIA.search(txt):
        return "concessionaria_utilidade"
    return "outros_sem_marcador"


def process_ocr_task(task):
    """Executa RapidOCR no recorte superior da página escaneada/renderizada."""
    global engine
    doc_id, png_path_str = task
    t0 = time.time()
    try:
        img = cv2.imread(png_path_str)
        if img is None:
            return doc_id, "", 0, "erro_imagem", 0

        h = img.shape[0]
        # Corta os primeiros 75% da página onde ficam cabeçalho, dados fiscais e descritivo
        crop = img[:int(h * 0.75), :]
        res, _ = engine(crop)
        t_elapsed_ms = int((time.time() - t0) * 1000)

        if not res:
            return doc_id, "", 0, "outros_sem_marcador", t_elapsed_ms

        full_txt = " ".join([line[1] for line in res])
        char_count = len(full_txt)
        cls = classify_text(full_txt)
        return doc_id, full_txt, char_count, cls, t_elapsed_ms
    except Exception as e:
        t_elapsed_ms = int((time.time() - t0) * 1000)
        return doc_id, "", 0, f"erro_ocr: {e}", t_elapsed_ms


def main():
    t_start = time.time()
    print("=" * 76)
    print(" EXTRAÇÃO RAPID OCR E CLASSIFICAÇÃO DOS 38.964 CANDIDATOS NFS-E (TIERS 1-4)")
    print("=" * 76)

    # 1. Carregar Manifesto
    print(f"\n1. Carregando manifesto de candidatos: {MANIFEST_FILE}")
    with gzip.open(MANIFEST_FILE, "rt", encoding="utf-8") as f:
        manifest_items = json.load(f)
    print(f"   Total de candidatos no manifesto: {len(manifest_items):,}")

    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    # 2. Verificar Checkpoint existente
    existing_results = {}
    if RESULTS_FILE.exists():
        try:
            with gzip.open(RESULTS_FILE, "rt", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"   Checkpoint carregado com sucesso: {len(existing_results):,} já processados.")
        except Exception as e:
            print(f"   Aviso ao carregar checkpoint ({e}), iniciando do zero.")

    # 3. Fase 1: Varredura de Texto Digital Rápido (PyMuPDF)
    print("\n2. Fase 1: Análise e extração rápida de texto digital nativo...")
    to_ocr_tasks = []
    digital_count = 0
    corrupted_or_scanned = 0

    t_fase1 = time.time()
    for item in manifest_items:
        doc_id = item["doc_id"]
        if doc_id in existing_results:
            continue

        doc_name = item["doc_name"]
        pdf_path = PDF_DIR / doc_name
        png_path = RENDERED_DIR / f"{doc_id}.png"

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
            cls = classify_text(txt_digital)
            existing_results[doc_id] = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "tier": item["tier"],
                "lote": item["lote"],
                "similarity": item["similarity"],
                "fornecedor": item["fornecedor"],
                "metodo": "digital_pdf",
                "texto": txt_digital.strip(),
                "num_caracteres": len(txt_digital.strip()),
                "classe": cls,
                "tempo_ms": 1
            }
            digital_count += 1
        else:
            corrupted_or_scanned += 1
            to_ocr_tasks.append((doc_id, str(png_path)))

    print(f"   Fase 1 concluída em {time.time() - t_fase1:.1f}s:")
    print(f"   • Textos digitais processados diretamente: {digital_count:,}")
    print(f"   • Documentos pendentes para RapidOCR (escaneados/fontes CID): {len(to_ocr_tasks):,}")

    # 4. Fase 2: Execução Paralela do RapidOCR
    if to_ocr_tasks:
        num_workers = min(24, os.cpu_count() or 16)
        print(f"\n3. Fase 2: Executando RapidOCR em paralelo com {num_workers} workers...")
        ocr_start = time.time()
        processed_ocr = 0

        # Criar mapa auxiliar para metadados
        items_by_id = {item["doc_id"]: item for item in manifest_items}

        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_ocr_worker) as executor:
            for res in executor.map(process_ocr_task, to_ocr_tasks, chunksize=16):
                if stop_requested:
                    print("\n[INFO] Parada antecipada solicitada. Interrompendo loop do OCR...")
                    break

                doc_id, full_txt, char_count, cls, t_ms = res
                meta = items_by_id.get(doc_id, {})

                existing_results[doc_id] = {
                    "doc_id": doc_id,
                    "doc_name": meta.get("doc_name", f"{doc_id}.pdf"),
                    "tier": meta.get("tier", ""),
                    "lote": meta.get("lote", ""),
                    "similarity": meta.get("similarity", 0.0),
                    "fornecedor": meta.get("fornecedor", ""),
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
                    print(f"   [RapidOCR] {processed_ocr:,}/{len(to_ocr_tasks):,} docs processados ({rate:.1f} doc/s | Restante: {rem/60:.1f} min)")

                    # Checkpoint periódico
                    with gzip.open(RESULTS_FILE, "wt", encoding="utf-8") as f_out:
                        json.dump(existing_results, f_out)

        print(f"   Fase 2 finalizada em {time.time() - ocr_start:.1f}s!")

    # 5. Salvar Resultados Consolidados em Disco
    print(f"\n4. Salvando resultados consolidados em {RESULTS_FILE}...")
    with gzip.open(RESULTS_FILE, "wt", encoding="utf-8") as f_out:
        json.dump(existing_results, f_out)
    print(f"   Total gravado: {len(existing_results):,} documentos.")

    # 6. Persistência no Banco PostgreSQL
    print(f"\n5. Sincronizando resultados no banco de dados PostgreSQL ({DB_URL})...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                print("   Inserindo registros mestre na tabela documento (se ausentes)...")
                doc_master_rows = []
                for doc_id, data in existing_results.items():
                    dname = data["doc_name"]
                    ano = pdf_to_year.get(dname, 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{dname}"
                    doc_master_rows.append((int(doc_id), ano, caminho, "baixado", data["classe"]))

                # Inserção em lotes de 2000
                batch_size = 2000
                for i in range(0, len(doc_master_rows), batch_size):
                    chunk = doc_master_rows[i:i + batch_size]
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            classe_predita = EXCLUDED.classe_predita;
                    """, chunk)
                conn.commit()
                print(f"   ✓ {len(doc_master_rows):,} registros na tabela 'documento' sincronizados.")

                print("   Inserindo texto e OCR nas tabelas 'documento_ocr_pagina' e 'documento_texto_completo'...")
                page_rows = []
                full_rows = []
                for doc_id, data in existing_results.items():
                    ide = int(doc_id)
                    txt = (data["texto"] or "").replace("\x00", "")
                    chars = len(txt)
                    t_ms = data["tempo_ms"]
                    page_rows.append((ide, 1, "rapidocr", txt, chars, t_ms))
                    full_rows.append((ide, "rapidocr", 1, txt, chars, t_ms))

                for i in range(0, len(page_rows), batch_size):
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
                print(f"   ✓ {len(page_rows):,} registros inseridos em 'documento_ocr_pagina' e 'documento_texto_completo'.")
    except Exception as e:
        print(f"   [AVISO] Erro ao sincronizar com PostgreSQL: {e}")
        print("   Os dados continuam íntegros no arquivo local JSON.gz.")

    # 7. Relatório Final de Classificação
    total_total = len(existing_results)
    classes_count = Counter(d["classe"] for d in existing_results.values())
    methods_count = Counter(d["metodo"] for d in existing_results.values())

    print("\n" + "=" * 76)
    print(" RELATÓRIO ANALÍTICO DA EXTRAÇÃO E CLASSIFICAÇÃO")
    print("=" * 76)
    print(f"Total de Documentos Analisados: {total_total:,}")
    print("\nOrigem da Extração:")
    for met, cnt in methods_count.most_common():
        print(f"  • {met:15s}: {cnt:6,} ({cnt / total_total * 100:.2f}%)")

    print("\nDistribuição das Classes Fiscais Identificadas:")
    for cls, cnt in classes_count.most_common():
        print(f"  • {cls:25s}: {cnt:6,} ({cnt / total_total * 100:.2f}%)")

    # Distribuição por Tier
    print("\nDetalhamento por Tier de Similaridade:")
    tier_groups = {}
    for d in existing_results.values():
        t = d.get("tier", "indefinido")
        if t not in tier_groups:
            tier_groups[t] = Counter()
        tier_groups[t][d["classe"]] += 1

    for t in sorted(tier_groups.keys()):
        tot_tier = sum(tier_groups[t].values())
        nfse_tier = tier_groups[t].get("nfse_confirmada", 0)
        print(f"  [{t}] - Total: {tot_tier:,} | NFS-e Confirmadas: {nfse_tier:,} ({nfse_tier/tot_tier*100:.2f}%)")
        for c, count in tier_groups[t].most_common():
            if c != "nfse_confirmada":
                print(f"      - {c}: {count:,} ({count/tot_tier*100:.2f}%)")

    print("\nTempo Total de Execução: {:.1f}s ({:.2f} min)".format(time.time() - t_start, (time.time() - t_start) / 60))
    print("=" * 76)


if __name__ == "__main__":
    main()
