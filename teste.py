# pip install -U transformers accelerate pymupdf pillow pandas tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import io
import json
import os
import re
import sys
import time
from pathlib import Path
import urllib.request
import fitz  # PyMuPDF
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForMultimodalLM
import pandas as pd

SHARD_ID = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
NUM_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))


def log(msg: str) -> None:
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm.write(f"[{agora}] [Parte {SHARD_ID + 1}/{NUM_SHARDS}] {msg}")


MODEL = "maritaca-ai/S5-nano"
log(f"Iniciando job | CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")

t_model = time.time()
log(f"Carregando processor e modelo '{MODEL}'...")
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForMultimodalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")
log(f"Modelo carregado em {time.time() - t_model:.1f}s (device: {model.device})")

prompt = """
Você é um especialista em extração de dados de documentos fiscais brasileiros (NF-e / DANFE / NFC-e). Analise a imagem do documento fornecido e extraia exclusivamente os campos solicitados, retornando APENAS um objeto JSON válido, sem markdown (```json) ou texto explicativo.

Regras estritas de extração:
1. Se um campo não estiver visível, estiver ilegível ou não existir no documento, retorne `null` (nunca invente ou estime valores).
2. Mantenha apenas dígitos (sem pontos, barras ou traços) nos campos numéricos de identificação.
3. Atenção à diferença entre Emitente/Remetente (quem emitiu a nota, geralmente no topo à esquerda) e Destinatário/Remetente (o cliente/comprador, no bloco "DESTINATÁRIO / REMETENTE").

Schema JSON esperado e regras por campo:
{
  "numero_nota": "String contendo apenas os dígitos do número da NF-e (ex: '000123456'). Ignore a série.",
  "chave_de_acesso": "String contendo exatamente os 44 dígitos numéricos da chave de acesso, sem espaços.",
  "data_emissao": "String no formato ISO YYYY-MM-DD (ex: '2026-11-15').",
  "valor_total_nota": "Número (float) usando ponto como separador decimal (ex: 1250.50). Corresponde ao campo 'VALOR TOTAL DA NOTA'.",
  "cnpj_emitente": "String com os 14 dígitos numéricos do CNPJ do emissor/fornecedor (cabeçalho da nota).",
  "cnpj_destinatario": "String com os 14 dígitos numéricos do CNPJ do destinatário/cliente (bloco Destinatário/Remetente). Se for CPF, retorne os 11 dígitos."
}
"""


def baixar_imagens(url: str, dpi: int = 150, paginas=None):
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dados = resp.read()

    if dados[:4] == b"%PDF" or url.lower().endswith(".pdf"):
        doc = fitz.open(stream=dados, filetype="pdf")
        idx = paginas or range(len(doc))
        imgs = []
        for i in idx:
            pix = doc[i].get_pixmap(dpi=dpi)
            imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        return imgs, time.time() - t0

    return [Image.open(io.BytesIO(dados)).convert("RGB")], time.time() - t0


def perguntar(imagens, pergunta, pensar=False, max_new_tokens=1024):
    content = [{"type": "image", "image": img} for img in imagens]
    content.append({"type": "text", "text": pergunta})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=pensar,          # False = modo Instruct
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    novos = out[0][inputs["input_ids"].shape[1]:]   # só os tokens gerados
    return processor.decode(novos, skip_special_tokens=True)


def extrair_dict_json(texto: str) -> dict:
    limpo = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto.strip(), flags=re.IGNORECASE)
    try:
        obj = json.loads(limpo)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", limpo, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return {"raw_response": texto.strip()}


csv_path = "data/ceap_nfe_selecionadas.csv"
df_full = pd.read_csv(csv_path)
df = df_full.iloc[SHARD_ID::NUM_SHARDS].copy().reset_index(drop=True)
total_docs = len(df)

out_jsonl = Path(f"resultados_parte_{SHARD_ID}.jsonl")
links_processados = set()
if out_jsonl.exists():
    with out_jsonl.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registro = json.loads(linha)
                links_processados.update(registro.keys())
            except json.JSONDecodeError:
                pass

pendentes = [
    (idx, row.get("doc_id", idx), str(row["link_documento"]))
    for idx, row in df.iterrows()
    if str(row["link_documento"]) not in links_processados
]

log(
    f"CSV '{csv_path}' ({len(df_full)} total) | Parte: {total_docs} docs | "
    f"Já no JSONL: {len(links_processados)} | Pendentes: {len(pendentes)}"
)

t_inicio = time.time()
sucessos = 0
erros = 0

# Prefetch paralelo de PDFs nas CPUs (8 workers) para a GPU nunca esperar download
PREFETCH_WORKERS = 8
MAX_FILA_PREFETCH = 16

with (
    out_jsonl.open("a", encoding="utf-8") as f_out,
    ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool,
    tqdm(
        total=total_docs,
        initial=total_docs - len(pendentes),
        desc=f"Parte {SHARD_ID + 1}/{NUM_SHARDS}",
        unit="doc",
        dynamic_ncols=True,
    ) as pbar,
):
    futuros = {}
    iter_pendentes = iter(pendentes)

    for _ in range(min(MAX_FILA_PREFETCH, len(pendentes))):
        item = next(iter_pendentes)
        futuros[item] = pool.submit(baixar_imagens, item[2])

    for item in pendentes:
        idx, doc_id, link = item
        fut = futuros.pop(item)

        prox = next(iter_pendentes, None)
        if prox is not None:
            futuros[prox] = pool.submit(baixar_imagens, prox[2])

        try:
            imgs, t_dl = fut.result()
            t_inf_start = time.time()
            resposta_str = perguntar(imgs, prompt)
            t_inf = time.time() - t_inf_start

            dados_extraidos = extrair_dict_json(resposta_str)
            linha_jsonl = json.dumps({link: dados_extraidos}, ensure_ascii=False)
            f_out.write(linha_jsonl + "\n")
            f_out.flush()

            sucessos += 1
            pbar.set_postfix(ok=sucessos, err=erros, dl=f"{t_dl:.1f}s", inf=f"{t_inf:.1f}s")
            log(
                f"[{idx + 1}/{total_docs}] doc_id={doc_id} | {len(imgs)} pág(s) | "
                f"dl={t_dl:.1f}s inf={t_inf:.1f}s | {json.dumps(dados_extraidos, ensure_ascii=False)[:120]}"
            )
        except Exception as e:
            erros += 1
            linha_erro = json.dumps({link: {"erro": str(e)}}, ensure_ascii=False)
            f_out.write(linha_erro + "\n")
            f_out.flush()
            pbar.set_postfix(ok=sucessos, err=erros)
            log(f"[{idx + 1}/{total_docs}] ERRO doc_id={doc_id} ({link}): {e}")
        finally:
            pbar.update(1)

tempo_total_min = (time.time() - t_inicio) / 60
log(
    f"FINALIZADO em {tempo_total_min:.1f} min | "
    f"Sucessos: {sucessos} | Já em cache: {len(links_processados)} | Erros: {erros} | "
    f"Salvo em: {out_jsonl}"
)
