#!/usr/bin/env python3
"""
Saneamento e Purificação da Classe 'boleto' no Histórico de Anotações (CEAP).
1. Avalia todos os 1.219 documentos atualmente anotados como 'boleto'.
2. Garante cobertura de texto 100%:
   - Carrega do PostgreSQL (documento_texto_completo).
   - Se faltar, extrai texto digital nativo (PyMuPDF) ou executa RapidOCR com multiprocessing.
   - Salva os textos OCR extraídos no PostgreSQL.
3. Classifica com regras estritas:
   - Boleto Legítimo: Linha digitável (47-48 dígitos) OU Ficha de Compensação com Banco + Agência + Nosso Número.
   - NFS-e: Documentos municipais com termos de NFS-e / Prefeitura -> reclassifica para 'nfse' (ou 'hotelaria').
   - DANFE: Documentos fiscais modelo 55 com Chave de Acesso -> reclassifica para 'nfe'.
   - Recibo de Aluguel/Condomínio: Demonstrativos e recibos imobiliários sem código de barras -> reclassifica para 'recibo'.
   - Comprovante de Pagamento: Comprovante de transação bancária/internet banking -> reclassifica para 'recibo'.
4. Atualiza data/models/annotations_history.json.
5. Salva relatório detalhado em data/evaluation/relatorio_saneamento_boletos.json.
"""

import io
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import psycopg
import pymupdf

from dados.config import DATA_DIR, MODELS_DIR, PDF_DIR, ROOT_DIR

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")
HISTORY_FILE = MODELS_DIR / "annotations_history.json"
REPORT_FILE = DATA_DIR / "evaluation" / "relatorio_saneamento_boletos.json"

# Expressões regulares estritas
RE_LINHA_DIGITAVEL = re.compile(
    r"\b\d{5}[\.\s]?\d{5}\s+\d{5}[\.\s]?\d{6}\s+\d{5}[\.\s]?\d{6}\s+\d\s+\d{10,14}\b"
)
RE_CODIGO_BARRAS_47_48 = re.compile(r"\b\d{47,48}\b")
RE_CODIGO_BANCO_COMPENSACAO = re.compile(r"\|\s*\d{3}-\w\s*\|")

RE_FICHA_COMPENSACAO = re.compile(
    r"FICHA\s*DE\s*COMPENSA[CÇ][AÃ]O|"
    r"RECIBO\s*D[OE]\s*(SACADO|PAGADOR)|"
    r"BLOQUETO\s*DE\s*COBRAN[CÇ]A|"
    r"BOLETO\s*DE\s*COBRAN[CÇ]A|"
    r"COBRAN[CÇ]A\s*BANC[AÁ]RIA",
    re.IGNORECASE,
)

RE_CAMPOS_BANCARIOS = re.compile(
    r"(NOSSO\s*N[UÚ]MERO|"
    r"AG[EÊ]NCIA\s*/\s*C[OÓ]DIGO\s*(DO\s*)?(BENEFICI[AÁ]RIO|CEDENTE)|"
    r"CARTEIRA\s*[:\d]|"
    r"ESPECIE\s*DOC|"
    r"LOCAL\s*DE\s*PAGAMENTO)",
    re.IGNORECASE,
)

RE_BANCOS = re.compile(
    r"BANCO\s*(DO\s*BRASIL|ITAU|ITAÚ|BRADESCO|SANTANDER|CAIXA|SICOOB|SICREDI|SAFRA|BANRISUL|DO\s*NORDESTE|DE\s*BRASILIA|BRB)|"
    r"CAIXA\s*ECON[OÔ]MICA\s*FEDERAL|"
    r"BANCOOB|"
    r"INTERMEDIUM|BANCO\s*INTER",
    re.IGNORECASE,
)

RE_NFSE = re.compile(
    r"NOTA\s*FISCAL\s*(DE\s*SERVI[ÇC]OS?|ELETR[OÔ]NICA\s*DE\s*SERVI[ÇC]OS?)|"
    r"NFS-?E\b|"
    r"PREFEITURA\s*MUNICIPAL|"
    r"SECRETARIA\s*MUNICIPAL\s*D[AE]\s*(FAZENDA|FINAN[ÇC]AS|RECEITA)|"
    r"ISSQN|"
    r"TOMADOR\s*D[OE]\s*SERVI[ÇC]OS?|"
    r"PRESTADOR\s*D[OE]\s*SERVI[ÇC]OS?|"
    r"DISCRIMINA[CÇ][AÃ]O\s*DOS\s*SERVI[ÇC]OS?|"
    r"RECIBO\s*PROVIS[OÓ]RIO\s*DE\s*SERVI[ÇC]OS?|RPS\b",
    re.IGNORECASE,
)

RE_HOTEL = re.compile(
    r"HOTEL|HOTEIS|HOTÉIS|FLAT|POUSADA|HOSPEDAGEM|SUITES|RESORT|"
    r"BLUE\s*TREE|BOURBON|TRANSAMERICA|MERCURE|IBIS|SLAVIERO|WINDSOR|NOVOTEL",
    re.IGNORECASE,
)

RE_DANFE = re.compile(
    r"DOCUMENTO\s*AUXILIAR\s*DA\s*NOTA\s*FISCAL\s*ELETR[OÔ]NICA|"
    r"DANFE\b(?![\s\w]*NFC)|"
    r"CHAVE\s*DE\s*ACESSO\s*[:\s]*\d{4}\s*\d{4}|"
    r"PROTOCOLO\s*DE\s*AUTORIZA[CÇ][AÃ]O\s*DE\s*USO",
    re.IGNORECASE,
)

RE_ALUGUEL_CONDOMINIO = re.compile(
    r"RECIBO\s*DE\s*ALUGUEL|"
    r"DEMONSTRATIVO\s*DE\s*PAGAMENTO|"
    r"CONDOM[IÍ]NIO\s*(DO\s*EDIF[IÍ]CIO|RESIDENCIAL|EMPRESARIAL|ED\.)|"
    r"TAXA\s*CONDOMINIAL|"
    r"IMOBILI[AÁ]RI[AO]|"
    r"LOCA[CÇ][AÃ]O\s*DE\s*IM[OÓ]VEL|"
    r"INQUILINO|LOCAT[AÁ]RIO|LOCADOR|"
    r"CRECI\s*[:\s]*\d+",
    re.IGNORECASE,
)

RE_COMPROVANTE_PAGTO = re.compile(
    r"COMPROVANTE\s*DE\s*(PAGAMENTO|TRANSFER[EÊ]NCIA|DEP[OÓ]SITO|OPERAC)|"
    r"AUTENTICA[CÇ][AÃ]O\s*BANC[AÁ]RIA|"
    r"PAGAMENTO\s*DE\s*T[IÍ]TULO|"
    r"SISBB\s*-\s*SISTEMA\s*DE\s*INFORMACOES|"
    r"CANAL\s*DE\s*AUTOATENDIMENTO",
    re.IGNORECASE,
)

_engine = None


def init_worker():
    global _engine
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    from rapidocr_onnxruntime import RapidOCR
    _engine = RapidOCR(use_cls=False, det_limit_side_len=640, intra_op_num_threads=1, inter_op_num_threads=1)


def _ocr_single_doc(args_tuple: Tuple[int, str]) -> Tuple[int, str]:
    """Worker de OCR para páginas digitalizadas."""
    global _engine
    doc_id, pdf_path_str = args_tuple
    pdf_p = Path(pdf_path_str)
    if not pdf_p.exists():
        return doc_id, ""

    try:
        doc = pymupdf.open(pdf_p)
        if len(doc) == 0:
            doc.close()
            return doc_id, ""

        # Testar texto digital da 1a página primeiro
        text = doc[0].get_text().strip()
        if len(text) > 40:
            doc.close()
            return doc_id, text

        # Executar RapidOCR na primeira página
        page = doc[0]
        pix = page.get_pixmap(dpi=110, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()

        res, _ = _engine(img)
        if res:
            ocr_text = "\n".join(line[1] for line in res)
            return doc_id, ocr_text
        return doc_id, ""
    except Exception:
        return doc_id, ""


def classify_text(txt: str, subcota: str = "", fornecedor: str = "") -> Tuple[str, str]:
    """
    Classifica o texto em:
    - 'boleto' (boleto real com código de barras/linha digitável ou ficha bancária legítima)
    - 'nfse' ou 'hotelaria' (NFS-e de serviços municipais)
    - 'nfe' (DANFE)
    - 'recibo' (aluguel, condomínio sem boleto, ou comprovante de pagamento)
    """
    if not txt or len(txt.strip()) < 15:
        return "indefinido", "Texto insuficiente para classificação"

    tem_linha = bool(RE_LINHA_DIGITAVEL.search(txt) or RE_CODIGO_BARRAS_47_48.search(txt))
    tem_ficha = bool(RE_FICHA_COMPENSACAO.search(txt))
    tem_banco = bool(RE_BANCOS.search(txt) or RE_CODIGO_BANCO_COMPENSACAO.search(txt))
    tem_campos_banco = bool(RE_CAMPOS_BANCARIOS.search(txt))

    is_nfse = bool(RE_NFSE.search(txt))
    is_danfe = bool(RE_DANFE.search(txt))
    is_aluguel_condo = bool(RE_ALUGUEL_CONDOMINIO.search(txt))
    is_comprovante = bool(RE_COMPROVANTE_PAGTO.search(txt))

    # 1. Checagem de Boleto Real
    # Tem linha digitável / código de barras de 47-48 dígitos
    if tem_linha:
        # Se for DANFE ou NFSE muito clara e a linha for apenas um número qualquer
        if is_danfe and not tem_ficha:
            return "nfe", "DANFE com chave de acesso/modelo 55"
        if is_nfse and not tem_ficha and not tem_banco:
            return "nfse", "NFS-e municipal"
        return "boleto", "Linha digitável / código de barras bancário detectado"

    # Tem Ficha de compensação bancária com Banco + Campos de Agência / Nosso Número
    if tem_ficha and tem_banco and tem_campos_banco:
        if is_danfe and not tem_campos_banco:
            return "nfe", "DANFE fiscal"
        return "boleto", "Ficha de compensação bancária com banco e agência/beneficiário"

    # 2. NFS-e / Hotelaria
    if is_nfse:
        if RE_HOTEL.search(txt) or RE_HOTEL.search(fornecedor) or "HOSPEDAGEM" in subcota.upper():
            return "hotelaria", "NFS-e de Hotelaria/Hospedagem"
        return "nfse", "Nota Fiscal de Serviços Eletrônica (NFS-e municipal)"

    # 3. DANFE / NF-e
    if is_danfe:
        return "nfe", "Documento Auxiliar de NF-e (DANFE)"

    # 4. Recibos de Aluguel e Condomínio (sem boleto)
    if is_aluguel_condo:
        return "recibo", "Recibo / Demonstrativo de Aluguel ou Taxa Condominial sem boleto bancário"

    # 5. Comprovante de Pagamento / Transferência Bancária
    if is_comprovante:
        return "recibo", "Comprovante de pagamento/transferência bancária (não é boleto de cobrança)"

    # Se tiver ficha de compensação com campos bancários mesmo sem linha perfeita
    if tem_ficha and (tem_banco or tem_campos_banco):
        return "boleto", "Ficha de compensação bancária"

    return "recibo", "Documento genérico de cobrança/recibo sem linha digitável"


def main():
    t0 = time.time()
    print("=" * 76)
    print(" SANEAMENTO E RECLASSIFICAÇÃO DA CLASSE 'BOLETO' (CEAP)")
    print("=" * 76)

    # 1. Carregar histórico de anotações
    if not HISTORY_FILE.exists():
        print(f"[ERRO] Arquivo {HISTORY_FILE} não encontrado!")
        sys.exit(1)

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history: Dict = json.load(f)

    boleto_keys = [
        k for k, v in history.items()
        if (v["label"] if isinstance(v, dict) else v) == "boleto"
    ]
    print(f"\n1. Total de documentos atualmente rotulados como 'boleto': {len(boleto_keys):,}")

    # 2. Carregar metadados adicionais de fornecedor/subcota se disponíveis
    meta_lookup = {}
    lookup_file = DATA_DIR / "pdf_metadata_lookup.json"
    if lookup_file.exists():
        with open(lookup_file, "r", encoding="utf-8") as f:
            meta_lookup = json.load(f)

    # 3. Carregar textos do PostgreSQL
    print("\n2. Consultando textos disponíveis no PostgreSQL (ceap)...")
    doc_ids = []
    for k in boleto_keys:
        stem = k[:-4] if k.endswith(".pdf") else k
        if stem.isdigit():
            doc_ids.append(int(stem))

    db_texts: Dict[int, str] = {}
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ide_documento, texto_completo FROM documento_texto_completo WHERE ide_documento = ANY(%s)",
                    (doc_ids,),
                )
                for r in cur.fetchall():
                    if r[1]:
                        db_texts[r[0]] = r[1]
        print(f"   ✓ Encontrados no banco: {len(db_texts):,}/{len(doc_ids):,} ({len(db_texts)/len(doc_ids)*100:.1f}%)")
    except Exception as e:
        print(f"   [AVISO] Erro na consulta ao banco: {e}")

    # 4. OCR em paralelo para os documentos pendentes
    missing_ids = [doc_id for doc_id in doc_ids if doc_id not in db_texts]
    print(f"\n3. Processando OCR para {len(missing_ids):,} documentos pendentes...")

    ocr_tasks = [
        (doc_id, str(PDF_DIR / f"{doc_id}.pdf"))
        for doc_id in missing_ids
    ]

    new_ocr_results = {}
    if ocr_tasks:
        n_workers = min(16, os.cpu_count() or 4)
        print(f"   Iniciando pool de {n_workers} workers RapidOCR com motor compartilhado...")
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as executor:
            for doc_id, txt in executor.map(_ocr_single_doc, ocr_tasks, chunksize=10):
                if txt:
                    new_ocr_results[doc_id] = txt
                    db_texts[doc_id] = txt
        print(f"   ✓ OCR concluído! {len(new_ocr_results):,} textos extraídos.")

        # Persistir no PostgreSQL
        try:
            with psycopg.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    rows = [
                        (doc_id, "rapidocr_sanitize", 1, txt, len(txt), 100)
                        for doc_id, txt in new_ocr_results.items()
                    ]
                    cur.executemany(
                        """
                        INSERT INTO documento_texto_completo 
                            (ide_documento, modelo, total_paginas, texto_completo, num_caracteres_total, tempo_total_ms)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento, modelo) DO UPDATE SET
                            texto_completo = EXCLUDED.texto_completo,
                            num_caracteres_total = EXCLUDED.num_caracteres_total,
                            atualizado_em = NOW();
                        """,
                        rows,
                    )
                conn.commit()
            print(f"   ✓ {len(new_ocr_results):,} textos persistidos no banco de dados.")
        except Exception as e:
            print(f"   [AVISO] Erro ao persistir textos no banco: {e}")

    # 5. Avaliação e Reclassificação
    print("\n4. Aplicando regras estritas de classificação...")
    reclassified_counts = {}
    detailed_audit = []
    kept_boletos = []

    for k in boleto_keys:
        stem = k[:-4] if k.endswith(".pdf") else k
        doc_id = int(stem) if stem.isdigit() else 0
        txt = db_texts.get(doc_id, "")

        meta = meta_lookup.get(f"{stem}.pdf", {})
        subcota = meta.get("subcota", "")
        fornec = meta.get("fornecedor", "")

        nova_classe, motivo = classify_text(txt, subcota=subcota, fornecedor=fornec)

        reclassified_counts[nova_classe] = reclassified_counts.get(nova_classe, 0) + 1

        entry = {
            "key": k,
            "doc_id": doc_id,
            "classe_anterior": "boleto",
            "classe_nova": nova_classe,
            "motivo": motivo,
            "fornecedor": fornec,
            "subcota": subcota,
            "snippet": txt[:140].replace("\n", " ") if txt else "",
        }
        detailed_audit.append(entry)

        if nova_classe == "boleto":
            kept_boletos.append(k)

    print("\nDistribuição pós-saneamento dos 1.219 documentos:")
    for cls, cnt in sorted(reclassified_counts.items(), key=lambda x: -x[1]):
        print(f"  • {cls:16s}: {cnt:,} ({cnt/len(boleto_keys)*100:.1f}%)")

    # 6. Atualizar annotations_history.json
    print(f"\n5. Atualizando histórico em {HISTORY_FILE}...")
    for item in detailed_audit:
        k = item["key"]
        nova_classe = item["classe_nova"]
        if nova_classe == "indefinido":
            # Se indefinido, deixa como recibo
            nova_classe = "recibo"

        if isinstance(history[k], dict):
            history[k]["label"] = nova_classe
        else:
            history[k] = nova_classe

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("   ✓ annotations_history.json salvo com sucesso.")

    # 7. Salvar relatório de avaliação
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_execucao": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_analisado": len(boleto_keys),
                "distribuicao": reclassified_counts,
                "itens": detailed_audit,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"   ✓ Relatório completo salvo em {REPORT_FILE}.")

    # 8. Sincronizar classes corrigidas no PostgreSQL na tabela 'documento'
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows_up = [
                    (it["classe_nova"] if it["classe_nova"] != "indefinido" else "recibo", it["doc_id"])
                    for it in detailed_audit
                    if it["doc_id"] > 0
                ]
                cur.executemany(
                    "UPDATE documento SET classe_predita = %s WHERE ide_documento = %s",
                    rows_up,
                )
            conn.commit()
        print(f"   ✓ {len(rows_up):,} documentos sincronizados na tabela 'documento' do banco.")
    except Exception as e:
        print(f"   [AVISO] Erro na sincronização com PostgreSQL: {e}")

    print("\n" + "=" * 76)
    print(f" SANEAMENTO CONCLUÍDO EM {time.time()-t0:.1f}s")
    print(f" Boletos legítimos preservados: {len(kept_boletos):,}")
    print(f" Falsos positivos purificados: {len(boleto_keys) - len(kept_boletos):,}")
    print("=" * 76)


if __name__ == "__main__":
    main()
