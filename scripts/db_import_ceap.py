#!/usr/bin/env python3
"""
Módulo de importação e gerenciamento de dados CEAP para o PostgreSQL.
Lê os arquivos brutos (Ano-*.json), os vetores e clusters (.npz / .json),
e realiza a normalização e inserção eficiente no schema relacional.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Generator, Optional

try:
    import psycopg
except ImportError:
    psycopg = None

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

CLUSTERS_TAXONOMY = {
    0: "CONCESSIONARIA_FATURA",
    1: "NFSE_MUNICIPAL",
    2: "CUPOM_TERMICO",
    3: "RECIBO_APP",
    4: "DANFE_A4",
}


def clean_digits(val: Any) -> str:
    """Extrai apenas dígitos de uma string ou número."""
    if not val:
        return ""
    return re.sub(r"\D", "", str(val))


def parse_date(date_str: Any) -> Optional[str]:
    """Normaliza datas para formato ISO (YYYY-MM-DD)."""
    if not date_str:
        return None
    s = str(date_str).strip()
    m_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"
    m_br = re.search(r"(\d{2})/(\d{2})/(\d{4})", s)
    if m_br:
        return f"{m_br.group(3)}-{m_br.group(2)}-{m_br.group(1)}"
    return None


def extract_ceap_records(json_path: Path) -> Generator[dict[str, Any], None, None]:
    """Itera sobre os registros do JSON bruto da CEAP de um ano específico."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("dados", []) if isinstance(data, dict) else data
    for item in items:
        yield item


def build_deputado_record(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    dep_id = item.get("numeroDeputadoID") or item.get("ideCadastro")
    nome = item.get("nomeParlamentar", "").strip()
    if not dep_id or not nome:
        return None
    return {
        "id_deputado": int(dep_id),
        "ide_cadastro": int(item.get("ideCadastro", 0)) if item.get("ideCadastro") else None,
        "nome_parlamentar": nome,
        "cpf": clean_digits(item.get("cpf")) or None,
        "sg_uf": (item.get("siglaUF") or "NA")[:2].upper(),
        "sg_partido": (item.get("siglaPartido") or "")[:30].strip(),
        "legislatura": int(item.get("legislatura", 0)) if item.get("legislatura") else None,
    }


def build_fornecedor_record(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    cnpj_cpf = clean_digits(item.get("cnpjCPF"))
    nome = (item.get("fornecedor") or "").strip()
    if not cnpj_cpf or not nome:
        return None
    tipo_pessoa = "F" if len(cnpj_cpf) <= 11 else "J"
    return {
        "cnpj_cpf": cnpj_cpf[:14],
        "tipo_pessoa": tipo_pessoa,
        "nome_principal": nome,
    }


def build_subcota_record(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    subcota = item.get("numeroSubCota")
    if subcota is None:
        return None
    espec = item.get("numeroEspecificacaoSubCota") or 0
    desc = (item.get("descricao") or "OUTRAS DESPESAS").strip()
    desc_espec = (item.get("descricaoEspecificacao") or "").strip() or None
    return {
        "num_subcota": int(subcota),
        "num_especificacao": int(espec),
        "descricao": desc,
        "descricao_espec": desc_espec,
    }


def build_documento_record(item: dict[str, Any], pdf_dir: Path) -> Optional[dict[str, Any]]:
    doc_id = item.get("idDocumento")
    if not doc_id:
        return None
    url = item.get("urlDocumento") or ""
    filename = f"{doc_id}.pdf"
    file_path = pdf_dir / filename
    ano = int(item.get("ano", 0))
    if not ano:
        return None

    return {
        "ide_documento": int(doc_id),
        "ano": ano,
        "url_origem": url,
        "caminho_arquivo": str(file_path),
        "status": "baixado" if file_path.exists() else "pendente",
        "tamanho_bytes": file_path.stat().st_size if file_path.exists() else None,
    }


def build_despesa_record(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    ano = int(item.get("ano", 0))
    mes = int(item.get("mes", 1))
    dep_id = item.get("numeroDeputadoID") or item.get("ideCadastro")
    subcota = item.get("numeroSubCota")
    if not ano or not dep_id or subcota is None:
        return None

    try:
        vlr_doc = float(item.get("valorDocumento", 0.0) or 0.0)
        vlr_liq = float(item.get("valorLiquido", 0.0) or 0.0)
        vlr_glo = float(item.get("valorGlosa", 0.0) or 0.0)
    except (ValueError, TypeError):
        vlr_doc, vlr_liq, vlr_glo = 0.0, 0.0, 0.0

    cnpj = clean_digits(item.get("cnpjCPF"))[:14] if item.get("cnpjCPF") else None

    return {
        "ide_documento": int(item["idDocumento"]) if item.get("idDocumento") else None,
        "id_deputado": int(dep_id),
        "cnpj_fornecedor": cnpj,
        "num_subcota": int(subcota),
        "num_especificacao": int(item.get("numeroEspecificacaoSubCota") or 0),
        "tipo_documento": int(item.get("tipoDocumento") or 0),
        "numero_documento": str(item.get("numero") or "").strip() or None,
        "data_emissao": parse_date(item.get("dataEmissao")),
        "valor_documento": vlr_doc,
        "valor_glosa": vlr_glo,
        "valor_liquido": vlr_liq,
        "mes": mes,
        "ano": ano,
        "parcela": int(item.get("parcela") or 0),
        "lote": int(item.get("lote") or 0) if item.get("lote") else None,
        "num_ressarcimento": int(item.get("numRessarcimento") or 0) if item.get("numRessarcimento") else None,
        "passageiro": (item.get("passageiro") or "").strip() or None,
        "trecho": (item.get("trecho") or "").strip() or None,
    }


def parse_and_aggregate_year(raw_dir: Path, pdf_dir: Path, year: int) -> dict[str, Any]:
    json_path = raw_dir / f"Ano-{year}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {json_path}")

    deputados = {}
    fornecedores = {}
    subcotas = {}
    documentos = {}
    despesas = []

    for item in extract_ceap_records(json_path):
        dep = build_deputado_record(item)
        if dep:
            deputados[dep["id_deputado"]] = dep

        forn = build_fornecedor_record(item)
        if forn:
            fornecedores[forn["cnpj_cpf"]] = forn

        sub = build_subcota_record(item)
        if sub:
            key = (sub["num_subcota"], sub["num_especificacao"])
            subcotas[key] = sub

        doc = build_documento_record(item, pdf_dir)
        if doc:
            documentos[doc["ide_documento"]] = doc

        desp = build_despesa_record(item)
        if desp:
            despesas.append(desp)

    return {
        "deputados": list(deputados.values()),
        "fornecedores": list(fornecedores.values()),
        "subcotas": list(subcotas.values()),
        "documentos": list(documentos.values()),
        "despesas": despesas,
    }


def populate_year_to_postgres(db_url: str, raw_dir: Path, pdf_dir: Path, year: int) -> None:
    if psycopg is None:
        raise ImportError("psycopg não está instalado. Instale com 'uv add psycopg[binary]'.")

    t0 = time.time()
    print(f"\n[Ano {year}] Carregando e agregando arquivo JSON...", flush=True)
    data = parse_and_aggregate_year(raw_dir, pdf_dir, year)

    print(f"[Ano {year}] Conectando ao PostgreSQL...", flush=True)
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # 1. Deputados
            print(f"  Inserindo {len(data['deputados'])} deputados...", flush=True)
            cur.executemany(
                """
                INSERT INTO deputado (id_deputado, ide_cadastro, nome_parlamentar, cpf, sg_uf, sg_partido, legislatura)
                VALUES (%(id_deputado)s, %(ide_cadastro)s, %(nome_parlamentar)s, %(cpf)s, %(sg_uf)s, %(sg_partido)s, %(legislatura)s)
                ON CONFLICT (id_deputado) DO UPDATE SET
                    sg_partido = EXCLUDED.sg_partido,
                    sg_uf = EXCLUDED.sg_uf;
                """,
                data["deputados"]
            )

            # 2. Fornecedores
            print(f"  Inserindo {len(data['fornecedores'])} fornecedores...", flush=True)
            cur.executemany(
                """
                INSERT INTO fornecedor (cnpj_cpf, tipo_pessoa, nome_principal)
                VALUES (%(cnpj_cpf)s, %(tipo_pessoa)s, %(nome_principal)s)
                ON CONFLICT (cnpj_cpf) DO NOTHING;
                """,
                data["fornecedores"]
            )

            # 3. Subcotas
            print(f"  Inserindo {len(data['subcotas'])} subcotas...", flush=True)
            cur.executemany(
                """
                INSERT INTO subcota (num_subcota, num_especificacao, descricao, descricao_espec)
                VALUES (%(num_subcota)s, %(num_especificacao)s, %(descricao)s, %(descricao_espec)s)
                ON CONFLICT (num_subcota, num_especificacao) DO NOTHING;
                """,
                data["subcotas"]
            )

            # 4. Documentos Físicos
            print(f"  Inserindo {len(data['documentos'])} documentos físicos...", flush=True)
            cur.executemany(
                """
                INSERT INTO documento (ide_documento, ano, url_origem, caminho_arquivo, status, tamanho_bytes)
                VALUES (%(ide_documento)s, %(ano)s, %(url_origem)s, %(caminho_arquivo)s, %(status)s, %(tamanho_bytes)s)
                ON CONFLICT (ide_documento) DO NOTHING;
                """,
                data["documentos"]
            )

            # 5. Despesas (Fato Contábil da CEAP)
            print(f"  Inserindo {len(data['despesas'])} despesas oficiais...", flush=True)
            cur.executemany(
                """
                INSERT INTO despesa (
                    ide_documento, id_deputado, cnpj_fornecedor, num_subcota, num_especificacao,
                    tipo_documento, numero_documento, data_emissao, valor_documento, valor_glosa,
                    valor_liquido, mes, ano, parcela, lote, num_ressarcimento, passageiro, trecho
                )
                VALUES (
                    %(ide_documento)s, %(id_deputado)s, %(cnpj_fornecedor)s, %(num_subcota)s, %(num_especificacao)s,
                    %(tipo_documento)s, %(numero_documento)s, %(data_emissao)s, %(valor_documento)s, %(valor_glosa)s,
                    %(valor_liquido)s, %(mes)s, %(ano)s, %(parcela)s, %(lote)s, %(num_ressarcimento)s,
                    %(passageiro)s, %(trecho)s
                );
                """,
                data["despesas"]
            )

        conn.commit()

    dur = time.time() - t0
    print(f"[Ano {year}] Carga finalizada com sucesso em {dur:.1f}s ({len(data['despesas']):,} despesas inseridas).")


def populate_clusters_and_embeddings(db_url: str, npz_path: Path, meta_path: Path) -> None:
    if psycopg is None:
        raise ImportError("psycopg não está instalado.")
    if np is None:
        raise ImportError("numpy não está instalado.")

    print(f"\nCarregando embeddings e clusters de {npz_path.name}...", flush=True)
    npz_data = np.load(npz_path)
    clusters_vlm = npz_data["clusters_doc_vlm"]
    vlm_embs = npz_data["vlm_doc_embeddings"]

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    docs = meta["documents"]

    total = len(docs)
    print(f"Total de itens para atualizar: {total:,}...", flush=True)

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            batch_doc_updates = []
            batch_emb_inserts = []

            for i, d in enumerate(docs):
                doc_name = d["doc_name"]
                ide_doc = int(doc_name.replace(".pdf", "")) if doc_name.endswith(".pdf") else None
                if not ide_doc:
                    continue

                cid = int(clusters_vlm[i])
                classe = CLUSTERS_TAXONOMY.get(cid, "OUTRO")
                emb_list = vlm_embs[i].tolist()
                emb_str = f"[{','.join(f'{x:.6f}' for x in emb_list)}]"

                caminho = f"/media/ssd2/datasets/CEAP/{doc_name}"
                batch_doc_updates.append((ide_doc, caminho, cid, classe))
                batch_emb_inserts.append((ide_doc, emb_str))

                if len(batch_doc_updates) >= 5000:
                    cur.executemany(
                        """
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, cluster_id, classe_predita, status)
                        VALUES (%s, 0, %s, %s, %s, 'processado')
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            cluster_id = EXCLUDED.cluster_id,
                            classe_predita = EXCLUDED.classe_predita;
                        """,
                        batch_doc_updates
                    )
                    cur.executemany(
                        """
                        INSERT INTO embedding_documento (ide_documento, vetor)
                        VALUES (%s, %s::vector)
                        ON CONFLICT (ide_documento) DO UPDATE SET vetor = EXCLUDED.vetor;
                        """,
                        batch_emb_inserts
                    )
                    conn.commit()
                    batch_doc_updates.clear()
                    batch_emb_inserts.clear()
                    print(f"  Atualizados {i+1:,}/{total:,} documentos com embeddings e clusters...", flush=True)

            if batch_doc_updates:
                cur.executemany(
                    """
                    INSERT INTO documento (ide_documento, ano, caminho_arquivo, cluster_id, classe_predita, status)
                    VALUES (%s, 0, %s, %s, %s, 'processado')
                    ON CONFLICT (ide_documento) DO UPDATE SET
                        cluster_id = EXCLUDED.cluster_id,
                        classe_predita = EXCLUDED.classe_predita;
                    """,
                    batch_doc_updates
                )
                cur.executemany(
                    """
                    INSERT INTO embedding_documento (ide_documento, vetor)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (ide_documento) DO UPDATE SET vetor = EXCLUDED.vetor;
                    """,
                    batch_emb_inserts
                )
                conn.commit()

    print(f"[Sucesso] {total:,} documentos atualizados com clusters e embeddings.")


def main():
    parser = argparse.ArgumentParser(description="Importador e Populador CEAP para PostgreSQL")
    parser.add_argument("--db-url", default=DEFAULT_DB_URL, help="URL de conexão PostgreSQL")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Diretório com Ano-*.json")
    parser.add_argument("--pdf-dir", type=Path, default=Path("/media/ssd2/datasets/CEAP"), help="Diretório de PDFs")
    parser.add_argument("--years", nargs="*", type=int, default=None, help="Anos para importar")
    parser.add_argument("--import-clusters", action="store_true", help="Importa clusters e embeddings do sample consolidado")
    parser.add_argument("--sample-npz", type=Path, default=Path("data/latent_vectors_sample_5pct.npz"), help="NPZ de embeddings")
    parser.add_argument("--sample-meta", type=Path, default=Path("data/latent_vectors_sample_5pct_metadata.json"), help="JSON de metadata")
    args = parser.parse_args()

    years_to_run = args.years if args.years is not None else ([2024] if not args.import_clusters else [])
    for y in years_to_run:
        populate_year_to_postgres(args.db_url, args.raw_dir, args.pdf_dir, y)

    if args.import_clusters and args.sample_npz.exists() and args.sample_meta.exists():
        populate_clusters_and_embeddings(args.db_url, args.sample_npz, args.sample_meta)


if __name__ == "__main__":
    main()
