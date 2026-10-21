#!/usr/bin/env python3
"""
Extração de dados fiscais de PDFs utilizando a API multimodal do Google Gemini (google-genai).
Envia os documentos para o modelo (ex: gemini-2.5-flash) com schema estruturado via Pydantic
e salva o resultado padronizado em '<doc_id>_extraido.json'.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Carregar variáveis de ambiente de .env se existir
load_dotenv()


class ExtracaoDocumentoFiscal(BaseModel):
    fornecedor_nome: Optional[str] = Field(
        default=None,
        description="Razão social ou nome fantasia do estabelecimento emissor do documento fiscal"
    )
    fornecedor_cnpj_cpf: Optional[str] = Field(
        default=None,
        description="CNPJ ou CPF do emissor do documento (formatado ou apenas dígitos)"
    )
    numero_documento: Optional[str] = Field(
        default=None,
        description="Número da nota fiscal, DANFE, cupom fiscal, bilhete ou recibo"
    )
    data_emissao: Optional[str] = Field(
        default=None,
        description="Data de emissão do documento no formato YYYY-MM-DD (ex: 2024-02-07)"
    )
    valor_total: Optional[float] = Field(
        default=None,
        description="Valor total em reais do documento fiscal (ex: 1148.70)"
    )
    descricao_resumida: Optional[str] = Field(
        default=None,
        description="Resumo sucinto dos produtos ou serviços descritos no documento fiscal"
    )
    observacoes: Optional[str] = Field(
        default=None,
        description="Observações relevantes sobre o estado da nota (ex: ilegível, cupom rasurado, etc.)"
    )


PROMPT_SISTEMA = """Você é um especialista em auditoria fiscal e análise de documentos fiscais brasileiros (NF-e, NFC-e, NFS-e, DANFE, cupons fiscais, bilhetes de passagem aérea/terrestre e recibos).
Analise o PDF do documento fiscal em anexo e extraia com máxima fidelidade os campos solicitados no schema JSON:
- Fornecedor (emissor)
- CNPJ ou CPF do emissor
- Número do documento fiscal
- Data de emissão (no formato padronizado YYYY-MM-DD)
- Valor total expresso no documento fiscal (numérico float)
- Breve resumo dos itens/serviços fornecidos
Se algum dado estiver completamente ilegível ou ausente, retorne null no respectivo campo.
"""


def extract_pdf_with_gemini(
    client: genai.Client,
    pdf_path: Path,
    model_name: str = "gemini-3.5-flash-lite",
    max_retries: int = 4,
) -> dict[str, Any]:
    """Lê um PDF, envia para a API do Gemini com schema estruturado e retorna o dicionário."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    config = types.GenerateContentConfig(
        system_instruction=PROMPT_SISTEMA,
        response_mime_type="application/json",
        response_schema=ExtracaoDocumentoFiscal,
        temperature=0.1,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[pdf_part, "Extraia as informações deste documento fiscal conforme o schema estruturado."],
                config=config,
            )

            if response.parsed:
                if isinstance(response.parsed, BaseModel):
                    return response.parsed.model_dump()
                elif isinstance(response.parsed, dict):
                    return response.parsed

            # Fallback para parsing manual do texto
            if response.text:
                try:
                    return json.loads(response.text)
                except Exception:
                    return {"erro_parsing": response.text}

            return {"erro": "Resposta vazia da API"}
        except Exception as e:
            last_error = e
            err_msg = str(e)
            # Se for cota diária esgotada, não adianta tentar em loop
            if "PerDay" in err_msg or "daily" in err_msg.lower():
                raise e

            # Se for limite por minuto (15 RPM) ou instabilidade temporária (503)
            if attempt < max_retries - 1 and any(code in err_msg for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                wait_time = 15 * (attempt + 1)
                print(f" [Limite/Espera temporária detectada. Aguardando {wait_time}s para retry {attempt+1}/{max_retries}...]", end="", flush=True)
                time.sleep(wait_time)
                continue
            raise e

    raise last_error


def run_batch_extraction(
    input_dir: Path,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
    limit: Optional[int] = None,
    filter_year: Optional[int] = None,
    filter_doc_id: Optional[str] = None,
    delay: float = 5.0,
    overwrite: bool = False,
) -> list[Path]:
    """Processa os PDFs selecionados na pasta input_dir e salva os arquivos _extraido.json."""
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Chave de API do Gemini não encontrada!\n"
            "Defina a variável de ambiente GEMINI_API_KEY (ex: export GEMINI_API_KEY='sua-chave')\n"
            "ou crie um arquivo .env contendo: GEMINI_API_KEY=sua-chave\n"
            "ou passe via argumento: --api-key sua-chave"
        )

    client = genai.Client(api_key=resolved_api_key)

    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        docs = manifest.get("documents", [])
    else:
        docs = [{"idDocumento": p.stem, "pdf_path": str(p)} for p in input_dir.glob("*.pdf")]

    # Aplicar filtros
    if filter_year:
        docs = [d for d in docs if d.get("ano") == filter_year]
    if filter_doc_id:
        docs = [d for d in docs if str(d.get("idDocumento")) == str(filter_doc_id)]
    if limit:
        docs = docs[:limit]

    print(f"=== INICIANDO EXTRAÇÃO COM GEMINI ({model_name}) ===")
    print(f"Total de documentos a processar: {len(docs)}")

    processed_files = []

    for idx, item in enumerate(docs, 1):
        doc_id = item["idDocumento"]
        pdf_path = input_dir / f"{doc_id}.pdf"
        out_json_path = input_dir / f"{doc_id}_extraido.json"

        if not pdf_path.exists():
            print(f"[{idx}/{len(docs)}] [Aviso] PDF não encontrado: {pdf_path}")
            continue

        if out_json_path.exists() and not overwrite:
            print(f"[{idx}/{len(docs)}] Doc {doc_id} já extraído anteriormente. Pulando.")
            processed_files.append(out_json_path)
            continue

        print(f"[{idx}/{len(docs)}] Processando doc {doc_id} ({pdf_path.name})...", end="", flush=True)
        t0 = time.time()
        try:
            res = extract_pdf_with_gemini(client, pdf_path, model_name=model_name)
            res["_metadados_extracao"] = {
                "modelo": model_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tempo_segundos": round(time.time() - t0, 2),
                "idDocumento": doc_id,
            }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            print(f" OK ({time.time() - t0:.2f}s) -> Fornecedor: '{res.get('fornecedor_nome')}' | Valor: R$ {res.get('valor_total')}")
            processed_files.append(out_json_path)
        except Exception as e:
            print(f" ERRO: {e}")
            err_str = str(e)
            if "PerDay" in err_str or "daily" in err_str.lower():
                print(f"\n[Aviso de Quota] A cota diária do plano gratuito (RPD) para o modelo '{model_name}' foi atingida.")
                print("O processamento foi pausado para evitar novas falhas. Todas as extrações feitas até agora estão salvas!")
                break

        if delay > 0 and idx < len(docs):
            time.sleep(delay)

    print(f"\n[Concluído] {len(processed_files)} arquivos extraídos salvos em: {input_dir}")
    return processed_files


def main():
    parser = argparse.ArgumentParser(description="Extração multimodal de dados fiscais com Google Gemini")
    parser.add_argument("--input-dir", "-i", type=Path, default=Path("data/pdfs_selecionados"), help="Pasta com PDFs e manifest")
    parser.add_argument("--api-key", help="Chave GEMINI_API_KEY (ou via variável de ambiente / .env)")
    parser.add_argument("--model", default="gemini-3.5-flash-lite", help="Nome do modelo (ex: gemini-3.5-flash-lite, gemini-flash-lite-latest)")
    parser.add_argument("--limit", type=int, help="Limita o número de documentos para teste rápido")
    parser.add_argument("--year", type=int, help="Filtra por um ano específico")
    parser.add_argument("--doc-id", help="Processa apenas um documento específico pelo ID")
    parser.add_argument("--delay", type=float, default=5.0, help="Intervalo em segundos entre chamadas (padrão: 5.0s para respeitar 15 RPM do plano gratuito)")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescreve extrações já existentes")
    args = parser.parse_args()

    run_batch_extraction(
        input_dir=args.input_dir,
        api_key=args.api_key,
        model_name=args.model,
        limit=args.limit,
        filter_year=args.year,
        filter_doc_id=args.doc_id,
        delay=args.delay,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
