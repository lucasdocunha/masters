#!/usr/bin/env python3
"""
Prepara e exporta tarefas para rotulação no Label Studio focando em PDFs de página única.
Filtra documentos com num_pages == 1, realiza amostragem estratificada pelos clusters FAISS,
renderiza a 1ª página em PNG em alta resolução (150 DPI) e gera o arquivo JSON de tarefas.
Também gera o arquivo label_studio_config.xml com a taxonomia das 10 classes.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


def render_pdf_page_if_needed(pdf_path: Path, output_png: Path, dpi: int = 150) -> bool:
    """Renderiza a 1ª página do PDF para PNG se ainda não existir."""
    if output_png.exists() and output_png.stat().st_size > 0:
        return True

    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_png))
        doc.close()
        return True
    except Exception as e:
        print(f"    [Aviso] Falha ao renderizar {pdf_path.name}: {e}")
        return False


def generate_label_studio_xml_config(output_file: Path) -> None:
    """Gera o arquivo XML de configuração da interface do Label Studio."""
    xml_content = """<View>
  <Header value="CEAP - Classificação de Documentos (Página Única)" size="5"/>
  <View style="display: flex; gap: 16px; margin-bottom: 12px; background: #f8fafc; padding: 10px; border-radius: 8px; font-size: 13px;">
    <Text name="doc_info" value="Arquivo: $doc_name | Ano: $ano" />
    <Text name="sub_info" value="Subcota Oficial: $subcota_oficial" />
    <Text name="forn_info" value="Fornecedor: $fornecedor_oficial" />
  </View>
  
  <Image name="image" value="$image" maxWidth="920px" style="box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-radius: 6px; border: 1px solid #e2e8f0;"/>
  
  <Header value="Selecione a Classe Documental Correspondente:" size="5" style="margin-top: 16px;"/>
  <Choices name="choice" toName="image" choice="single" showInline="true">
    <Choice value="nfc" hotkey="1" hint="Nota Fiscal de Consumidor (NFC-e / Cupom Fiscal)" />
    <Choice value="nfe" hotkey="2" hint="Nota Fiscal Eletrônica (NF-e / DANFE)" />
    <Choice value="nfse" hotkey="3" hint="Nota Fiscal de Serviços Eletrônica (NFS-e)" />
    <Choice value="recibo" hotkey="4" hint="Recibos em geral (táxi, pedágio, locação, etc.)" />
    <Choice value="emails" hotkey="5" hint="E-mails, trocas de mensagens e comunicações" />
    <Choice value="energia" hotkey="6" hint="Faturas e contas de energia elétrica / luz" />
    <Choice value="agua" hotkey="7" hint="Faturas e contas de água e saneamento" />
    <Choice value="boleto" hotkey="8" hint="Boletos bancários de cobrança e compensação" />
    <Choice value="fatura_cartao_credito" hotkey="9" text="fatura de cartao de credito" hint="Faturas e demonstrativos de cartão de crédito" />
    <Choice value="uber" hotkey="0" hint="Recibos e comprovantes de viagens por aplicativo (Uber, 99, etc.)" />
    <Choice value="aviao" hotkey="a" hint="Passagens aéreas, bilhetes e cartões de embarque" />
    <Choice value="fatura_internet_celular" hotkey="c" text="fatura de internet/celular" hint="Faturas e contas de telefonia celular, fixa e internet" />
    <Choice value="invoice" hotkey="i" hint="Invoices internacionais e recibos em moeda estrangeira" />
    <Choice value="fatura" hotkey="f" hint="Faturas gerais de cobrança, locação ou prestação comercial" />
  </Choices>
</View>
"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"[OK] Configuração XML do Label Studio salva em: {output_file}")


def prepare_tasks(
    metadata_json: Path,
    pdf_dir: Path,
    output_images_dir: Path,
    output_tasks_json: Path,
    meta_lookup_json: Path,
    clusters_npz: Optional[Path] = None,
    n_docs: int = 500,
) -> list[dict[str, Any]]:
    """Filtra PDFs de 1 página e monta a lista de tarefas para o Label Studio."""
    print(f"Carregando metadados de {metadata_json}...")
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    all_docs = meta_data.get("documents", [])

    # Filtro estrito: apenas num_pages == 1
    single_page_docs = [d for d in all_docs if d.get("num_pages") == 1]
    print(f"Total de documentos de 1 página identificados: {len(single_page_docs):,} ({len(single_page_docs)/len(all_docs)*100:.1f}%)")

    # Carregar lookup de subcotas/fornecedores
    meta_lookup = {}
    if meta_lookup_json.exists():
        print(f"Carregando cache de metadados de {meta_lookup_json}...")
        with open(meta_lookup_json, "r", encoding="utf-8") as f:
            meta_lookup = json.load(f)

    # Amostragem (estratificada se houver clusters, ou linear)
    selected_docs = []
    if clusters_npz and clusters_npz.exists():
        print(f"Estratificando amostragem usando clusters de {clusters_npz}...")
        npz = np.load(clusters_npz)
        cluster_labels = npz.get("clusters_doc_vlm")
        if cluster_labels is None:
            cluster_labels = npz.get("clusters_doc_dino")

        if cluster_labels is not None and len(cluster_labels) == len(all_docs):
            cluster_map = defaultdict(list)
            for idx, doc in enumerate(all_docs):
                if doc.get("num_pages") == 1:
                    c_id = int(cluster_labels[idx])
                    cluster_map[c_id].append(doc)

            per_cluster = max(1, n_docs // max(1, len(cluster_map)))
            for c_id in sorted(cluster_map.keys()):
                selected_docs.extend(cluster_map[c_id][:per_cluster])

    if not selected_docs:
        selected_docs = single_page_docs[:n_docs]

    # Limitar ao total solicitado
    selected_docs = selected_docs[:n_docs]
    print(f"Selecionados {len(selected_docs):,} documentos para renderização e empacotamento...")

    output_images_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    rendered_count = 0

    for idx, doc in enumerate(selected_docs):
        doc_name = doc.get("doc_name", "")
        pdf_path = pdf_dir / doc_name
        doc_stem = pdf_path.stem
        png_name = f"{doc_stem}.png"
        out_png = output_images_dir / png_name

        # 1. Renderizar se necessário
        if pdf_path.exists():
            rendered = render_pdf_page_if_needed(pdf_path, out_png, dpi=150)
            if rendered:
                rendered_count += 1

        meta = meta_lookup.get(doc_name, {})
        subcota = meta.get("subcota", "NÃO INFORMADA")
        fornecedor = meta.get("fornecedor", "NÃO INFORMADO")
        ano = meta.get("ano", doc.get("year", "N/A"))

        # Formato de URL local do Label Studio
        image_url = f"/data/local-files/?d=images/{png_name}"
        pdf_url = f"/data/local-files/?d=pdfs/{doc_name}"

        task_item = {
            "id": idx + 1,
            "data": {
                "image": image_url,
                "pdf_url": pdf_url,
                "doc_name": doc_name,
                "doc_id": int(doc.get("doc_id", idx)),
                "num_pages": 1,
                "subcota_oficial": subcota,
                "fornecedor_oficial": fornecedor,
                "ano": ano,
            }
        }
        tasks.append(task_item)

    print(f"Renderizados {rendered_count} PNGs em {output_images_dir}")

    output_tasks_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tasks_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"[OK] Tarefas exportadas com sucesso: {output_tasks_json} ({len(tasks)} tarefas)")

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Prepara tarefas do Label Studio filtrando PDFs de 1 página")
    parser.add_argument("--metadata", type=Path, default=Path("data/latent_vectors_sample_5pct_metadata.json"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("/media/ssd2/datasets/CEAP"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/rendered_pages"))
    parser.add_argument("--output-json", type=Path, default=Path("data/label_studio_tasks_1page.json"))
    parser.add_argument("--meta-lookup", type=Path, default=Path("data/pdf_metadata_lookup.json"))
    parser.add_argument("--clusters-npz", type=Path, default=Path("data/latent_vectors_faiss_k10.npz"))
    parser.add_argument("--output-xml", type=Path, default=Path("data/label_studio_config.xml"))
    parser.add_argument("--n-docs", "-n", type=int, default=500, help="Quantidade de tarefas a preparar")
    args = parser.parse_args()

    # 1. Gerar template XML
    generate_label_studio_xml_config(args.output_xml)

    # 2. Gerar tarefas JSON
    prepare_tasks(
        metadata_json=args.metadata,
        pdf_dir=args.pdf_dir,
        output_images_dir=args.images_dir,
        output_tasks_json=args.output_json,
        meta_lookup_json=args.meta_lookup,
        clusters_npz=args.clusters_npz if args.clusters_npz.exists() else None,
        n_docs=args.n_docs,
    )


if __name__ == "__main__":
    main()
