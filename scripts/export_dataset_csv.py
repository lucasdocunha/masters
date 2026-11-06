#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Gera os arquivos .CSV solicitados com todos os documentos do CEAP:
Colunas:
  - doc_id: Identificador numérico do documento
  - caminho_arquivo: Caminho absoluto do arquivo PDF no disco
  - link_documento: URL oficial pública na Câmara dos Deputados
  - arquivo_json_origem: Caminho do arquivo JSON anual de onde veio o registro
  - metadados_json: Objeto JSON compacto com os dados da despesa (fornecedor, CNPJ/CPF, valor, subcota, parlamentar)
  - ano: Ano da despesa
  - classe_mapeada: Classe atribuída no processo de anotação/metric learning (ou 'Não Mapeado')
  - status: 'rotulado' ou 'nao_mapeado'

Saídas geradas:
  1. data/ceap_documentos_completo.csv (~1.40M linhas - universo total)
  2. data/ceap_documentos_rotulados.csv (~61.9k linhas - apenas os já mapeados)
  3. Symlinks convenientes no Desktop do usuário para acesso imediato.
"""

import os
import sys
import glob
import json
import time
import csv
from pathlib import Path

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")

HISTORY_FILE = BASE_DIR / "data" / "models" / "annotations_history.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
URL_CACHE_FILE = DATA_DIR / "stem_to_url.json"

CSV_ALL_FILE = DATA_DIR / "ceap_documentos_completo.csv"
CSV_ROTULADOS_FILE = DATA_DIR / "ceap_documentos_rotulados.csv"


def clean_str(val):
    if val is None:
        return ""
    return str(val).strip().replace("\r", " ").replace("\n", " ")


def main():
    print("=" * 76)
    print(" EXPORTAÇÃO DO DATASET CEAP EM FORMATO .CSV")
    print("=" * 76)
    t_start = time.time()

    # 1. Carregar anotações existentes
    print("\n1. Carregando histórico de anotações...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        raw_annotations = json.load(f)

    # Normalizar chaves para stem sem extensão
    annotations = {}
    for k, v in raw_annotations.items():
        s = str(k).replace(".pdf", "")
        annotations[s] = v
    print(f"   ✓ Total de documentos rotulados carregados: {len(annotations):,}")

    # 2. Carregar stem_to_url e lookup como fallback/referência
    print("\n2. Carregando caches de URL e metadados rápidos...")
    with open(URL_CACHE_FILE, "r", encoding="utf-8") as f:
        stem_to_url = json.load(f)
    print(f"   ✓ URLs mapeadas no cache: {len(stem_to_url):,}")

    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        pdf_lookup = json.load(f)
    print(f"   ✓ Metadados de lookup: {len(pdf_lookup):,}")

    # 3. Listar arquivos presentes fisicamente no SSD
    print("\n3. Verificando arquivos no SSD (/media/ssd2/datasets/CEAP)...")
    disk_files = set(os.listdir(PDF_DIR))
    print(f"   ✓ Total de PDFs encontrados no disco: {len(disk_files):,}")

    # 4. Mapear dados brutos a partir de Ano-*.json
    print("\n4. Indexando dados detalhados dos JSONs brutos da Câmara (2014 a 2026)...")
    raw_files = sorted(RAW_DIR.glob("Ano-*.json"))
    
    # doc_id -> dict com metadados completos
    doc_metadata = {}
    total_raw_records = 0

    for json_p in raw_files:
        ano_str = json_p.stem.replace("Ano-", "")
        t0 = time.time()
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("dados", []) if isinstance(data, dict) else data
        total_raw_records += len(items)

        for item in items:
            doc_id = item.get("idDocumento")
            if not doc_id:
                continue
            doc_id_str = str(doc_id)

            # Manter primeiro registro ou consolidar se não existir
            if doc_id_str not in doc_metadata:
                url = item.get("urlDocumento") or stem_to_url.get(doc_id_str, "")
                ano = item.get("ano") or int(ano_str) if ano_str.isdigit() else 2024
                
                # Resumo compacto dos metadados fiscais em JSON
                meta_compact = {
                    "fornecedor": clean_str(item.get("fornecedor")),
                    "cnpjCPF": clean_str(item.get("cnpjCPF")),
                    "valorDocumento": clean_str(item.get("valorDocumento")),
                    "valorLiquido": clean_str(item.get("valorLiquido")),
                    "subcota": clean_str(item.get("descricao")),
                    "nomeParlamentar": clean_str(item.get("nomeParlamentar")),
                    "siglaUF": clean_str(item.get("siglaUF")),
                    "siglaPartido": clean_str(item.get("siglaPartido")),
                    "dataEmissao": clean_str(item.get("dataEmissao"))
                }

                doc_metadata[doc_id_str] = {
                    "ano": int(ano),
                    "url": url,
                    "arquivo_json": f"data/raw/{json_p.name}",
                    "metadados_json": json.dumps(meta_compact, ensure_ascii=False)
                }

        print(f"   • {json_p.name}: {len(items):,} despesas indexadas em {time.time()-t0:.2f}s")

    print(f"\n   ✓ Total de despesas processadas nos JSONs: {total_raw_records:,}")
    print(f"   ✓ Documentos únicos indexados com metadados: {len(doc_metadata):,}")

    # 5. Gerar os dois arquivos CSV
    print("\n5. Escrevendo arquivos CSV...")

    headers = [
        "doc_id",
        "caminho_arquivo",
        "link_documento",
        "arquivo_json_origem",
        "metadados_json",
        "ano",
        "classe_mapeada",
        "status"
    ]

    # Conjunto unificado de todos os documentos (presentes em disco + anotados + nos metadados)
    all_doc_ids = set()
    # Adiciona a partir do disco
    for fname in disk_files:
        if fname.endswith(".pdf"):
            all_doc_ids.add(fname.replace(".pdf", ""))
    # Adiciona a partir das anotações (garante 100% dos rotulados)
    for stem in annotations:
        all_doc_ids.add(stem)

    sorted_doc_ids = sorted(list(all_doc_ids), key=lambda x: int(x) if x.isdigit() else x)
    print(f"   • Total de documentos a exportar no CSV completo: {len(sorted_doc_ids):,}")

    count_all = 0
    count_rotulados = 0

    with open(CSV_ALL_FILE, "w", newline="", encoding="utf-8") as f_all, \
         open(CSV_ROTULADOS_FILE, "w", newline="", encoding="utf-8") as f_rot:

        w_all = csv.writer(f_all)
        w_rot = csv.writer(f_rot)

        w_all.writerow(headers)
        w_rot.writerow(headers)

        for doc_id_str in sorted_doc_ids:
            # Caminho físico do arquivo
            caminho_pdf = f"/media/ssd2/datasets/CEAP/{doc_id_str}.pdf"

            # Classe mapeada
            classe = annotations.get(doc_id_str, "")
            is_rotulado = bool(classe)
            classe_str = classe if is_rotulado else "Não Mapeado"
            status_str = "rotulado" if is_rotulado else "nao_mapeado"

            # Metadados
            info = doc_metadata.get(doc_id_str)
            if info:
                link = info["url"] or stem_to_url.get(doc_id_str, "")
                arq_json = info["arquivo_json"]
                meta_json = info["metadados_json"]
                ano = info["ano"]
            else:
                # Fallback para lookup
                fname = f"{doc_id_str}.pdf"
                lk = pdf_lookup.get(fname, {})
                ano = lk.get("ano", 2024)
                link = stem_to_url.get(doc_id_str, f"https://www.camara.leg.br/cota-parlamentar/documentos/publ/{ano}/{doc_id_str}.pdf")
                arq_json = f"data/raw/Ano-{ano}.json"
                meta_compact = {
                    "fornecedor": clean_str(lk.get("fornecedor", "")),
                    "subcota": clean_str(lk.get("subcota", "")),
                    "ano": ano
                }
                meta_json = json.dumps(meta_compact, ensure_ascii=False)

            row = [
                doc_id_str,
                caminho_pdf,
                link,
                arq_json,
                meta_json,
                ano,
                classe_str,
                status_str
            ]

            w_all.writerow(row)
            count_all += 1

            if is_rotulado:
                w_rot.writerow(row)
                count_rotulados += 1

    # 6. Criar links no Desktop
    print("\n6. Criando atalhos no Desktop...")
    desktop_all = DESKTOP_DIR / "ceap_documentos_completo.csv"
    desktop_rot = DESKTOP_DIR / "ceap_documentos_rotulados.csv"

    if desktop_all.is_symlink() or desktop_all.exists():
        desktop_all.unlink()
    if desktop_rot.is_symlink() or desktop_rot.exists():
        desktop_rot.unlink()

    desktop_all.symlink_to(CSV_ALL_FILE)
    desktop_rot.symlink_to(CSV_ROTULADOS_FILE)

    sz_all_mb = CSV_ALL_FILE.stat().st_size / (1024 * 1024)
    sz_rot_mb = CSV_ROTULADOS_FILE.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 76)
    print(" EXPORTAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 76)
    print(f"1. CSV Completo (Todos os arquivos):")
    print(f"   • Caminho: {CSV_ALL_FILE}")
    print(f"   • Desktop: {desktop_all}")
    print(f"   • Linhas:  {count_all:,} registros | Tamanho: {sz_all_mb:.1f} MB")
    print(f"\n2. CSV Rotulados (Apenas classes já mapeadas):")
    print(f"   • Caminho: {CSV_ROTULADOS_FILE}")
    print(f"   • Desktop: {desktop_rot}")
    print(f"   • Linhas:  {count_rotulados:,} registros | Tamanho: {sz_rot_mb:.1f} MB")
    print(f"\nTempo total de execução: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
