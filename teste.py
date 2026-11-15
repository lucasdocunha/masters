# pip install -U transformers accelerate pymupdf pillow pandas
from datetime import datetime
import io
import os
import sys
import time
from pathlib import Path
import urllib.request
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
import pandas as pd


def log(msg: str) -> None:
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{agora}] [Parte {SHARD_ID + 1}/{NUM_SHARDS}] {msg}", flush=True)


SHARD_ID = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
NUM_SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

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


def baixar_imagens(url, dpi=150, paginas=None):
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
        return imgs

    return [Image.open(io.BytesIO(dados)).convert("RGB")]


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


csv_path = "data/ceap_nfe_selecionadas.csv"
df_full = pd.read_csv(csv_path)
df = df_full.iloc[SHARD_ID::NUM_SHARDS].copy().reset_index(drop=True)
total_docs = len(df)
log(f"CSV '{csv_path}' lido ({len(df_full)} total) -> {total_docs} documentos atribuídos a esta parte.")

saida_dir = Path("saida_inferencias") / f"parte_{SHARD_ID}"
t_inicio = time.time()
sucessos = 0
pulados = 0
erros = 0

for idx, row in df.iterrows():
    doc_id = row.get("doc_id", idx)
    lote_dir = saida_dir / f"lote_{idx // 500:04d}"
    lote_dir.mkdir(parents=True, exist_ok=True)
    arq_json = lote_dir / f"{doc_id}.json"

    if arq_json.exists():
        df.loc[idx, "resposta"] = arq_json.read_text(encoding="utf-8")
        pulados += 1
        log(f"[{idx + 1}/{total_docs} | {(idx + 1) / total_docs * 100:5.1f}%] doc_id={doc_id} -> JÁ PROCESSADO (cache)")
        continue

    t_doc = time.time()
    try:
        imgs = baixar_imagens(row["link_documento"])
        t_dl = time.time() - t_doc

        t_inf_start = time.time()
        resposta = perguntar(imgs, prompt)
        t_inf = time.time() - t_inf_start

        arq_json.write_text(resposta, encoding="utf-8")
        df.loc[idx, "resposta"] = resposta
        sucessos += 1

        decorrido = time.time() - t_inicio
        processados_agora = sucessos + erros
        media_s = decorrido / max(processados_agora, 1)
        restantes = total_docs - (idx + 1)
        eta_min = (restantes * media_s) / 60

        resumo_json = " ".join(resposta.split())[:120]
        log(
            f"[{idx + 1}/{total_docs} | {(idx + 1) / total_docs * 100:5.1f}%] "
            f"doc_id={doc_id} | {len(imgs)} pág(s) | dl={t_dl:.1f}s inf={t_inf:.1f}s | "
            f"ETA: {eta_min:.1f} min | {resumo_json}"
        )
    except Exception as e:
        erros += 1
        log(f"[{idx + 1}/{total_docs} | {(idx + 1) / total_docs * 100:5.1f}%] ERRO doc_id={doc_id} ({ time.time() - t_doc:.1f}s): {e}")
        df.loc[idx, "resposta"] = None

out_csv = f"resultados_parte_{SHARD_ID}.csv"
df.to_csv(out_csv, index=False)
tempo_total_min = (time.time() - t_inicio) / 60
log(
    f"FINALIZADO em {tempo_total_min:.1f} min | "
    f"Total: {total_docs} | Sucessos: {sucessos} | Cache: {pulados} | Erros: {erros} | "
    f"Salvo em: {out_csv}"
)


