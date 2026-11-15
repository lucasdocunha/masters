# pip install -U transformers accelerate pymupdf pillow pandas tqdm
import atexit
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import io
import json
import os
import re
import signal
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
MODEL = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MODEL_NAME", "maritaca-ai/S5-nano")
MODEL_SLUG = re.sub(r"[^a-zA-Z0-9._-]+", "_", MODEL.split("/")[-1])

UNIFIED_JSONL = Path(f"resultados_{MODEL_SLUG}.jsonl")
PART_JSONL = Path(f"resultados_{MODEL_SLUG}_parte_{SHARD_ID}.jsonl")
LOCK_FILE = Path(f".resultados_{MODEL_SLUG}.jsonl.lock")


def log(msg: str) -> None:
    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tqdm.write(f"[{agora}] [{MODEL_SLUG} | Parte {SHARD_ID + 1}/{NUM_SHARDS}] {msg}")


def carregar_registros_jsonl(caminho: Path) -> dict[str, dict]:
    registros: dict[str, dict] = {}
    if not caminho.exists():
        return registros
    with caminho.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                obj = json.loads(linha)
                if isinstance(obj, dict):
                    for link, dados in obj.items():
                        if isinstance(dados, dict) and "erro" not in dados:
                            registros[str(link)] = dados
            except json.JSONDecodeError:
                pass
    return registros


def consolidar_todos_jsonl() -> set[str]:
    """Unifica resultados do modelo atual (e migra resultados.jsonl/resultados_parte_*.jsonl legados se S5-nano)."""
    LOCK_FILE.touch(exist_ok=True)
    with LOCK_FILE.open("r+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            unificado = carregar_registros_jsonl(UNIFIED_JSONL)
            arquivos_extras = list(sorted(Path(".").glob(f"resultados_{MODEL_SLUG}_parte_*.jsonl")))

            # Compatibilidade com arquivos da execução anterior do S5-nano
            if MODEL_SLUG == "S5-nano":
                legado_unificado = Path("resultados.jsonl")
                if legado_unificado.exists():
                    arquivos_extras.insert(0, legado_unificado)
                arquivos_extras.extend(sorted(Path(".").glob("resultados_parte_*.jsonl")))

            novos = 0
            for arq in arquivos_extras:
                parciais = carregar_registros_jsonl(arq)
                for link, dados in parciais.items():
                    if link not in unificado:
                        unificado[link] = dados
                        novos += 1

            if novos > 0 or not UNIFIED_JSONL.exists():
                tmp_path = UNIFIED_JSONL.with_suffix(".jsonl.tmp")
                with tmp_path.open("w", encoding="utf-8") as f_out:
                    for link, dados in unificado.items():
                        f_out.write(json.dumps({link: dados}, ensure_ascii=False) + "\n")
                tmp_path.replace(UNIFIED_JSONL)

            if PART_JSONL.exists():
                PART_JSONL.unlink(missing_ok=True)

            return set(unificado.keys())
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def salvar_no_unificado(link: str, dados: dict) -> None:
    """Grava imediatamente no resultados_<modelo>.jsonl único com lock de arquivo entre processos."""
    linha = json.dumps({link: dados}, ensure_ascii=False) + "\n"
    LOCK_FILE.touch(exist_ok=True)
    with LOCK_FILE.open("r+") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with UNIFIED_JSONL.open("a", encoding="utf-8") as f_out:
                f_out.write(linha)
                f_out.flush()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


interrompido = False


def _tratar_sinal(signum, _frame):
    global interrompido
    interrompido = True
    nome_sinal = signal.Signals(signum).name
    log(f"Sinal {nome_sinal} recebido! Consolidando resultados em '{UNIFIED_JSONL}' antes de encerrar...")
    total = len(consolidar_todos_jsonl())
    log(f"Consolidação concluída ({total} documentos únicos em '{UNIFIED_JSONL}'). Encerrando.")
    sys.exit(0)


signal.signal(signal.SIGINT, _tratar_sinal)
signal.signal(signal.SIGTERM, _tratar_sinal)
atexit.register(consolidar_todos_jsonl)

# 1. Consolida quaisquer resultados parciais anteriores e carrega links já feitos para este modelo
links_processados = consolidar_todos_jsonl()

csv_path = "/users/home/lucas.ocunha/research/masters/data/ceap_nfe_todas_20k..csv"
df_full = pd.read_csv(csv_path)
df_shard = df_full.iloc[SHARD_ID::NUM_SHARDS].copy()
total_shard_original = len(df_shard)

# Filtra apenas os que ainda NÃO foram processados por este modelo
df = df_shard[~df_shard["link_documento"].astype(str).isin(links_processados)].copy().reset_index(drop=True)
ja_feitos_nesta_parte = total_shard_original - len(df)
total_pendentes = len(df)

log(
    f"CSV '{csv_path}' ({len(df_full)} total) | "
    f"Já salvos em '{UNIFIED_JSONL}': {len(links_processados)} (desta parte: {ja_feitos_nesta_parte}) | "
    f"Restantes para inferência nesta parte: {total_pendentes}"
)

if total_pendentes == 0:
    log(f"Todos os documentos desta parte já foram processados pelo modelo '{MODEL}'! Encerrando.")
    sys.exit(0)

log(f"Iniciando job | Modelo: {MODEL} | CUDA disponível: {torch.cuda.is_available()}")
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


pendentes = [
    (idx, row.get("doc_id", idx), str(row["link_documento"]))
    for idx, row in df.iterrows()
]

t_inicio = time.time()
sucessos = 0
erros = 0

PREFETCH_WORKERS = 8
MAX_FILA_PREFETCH = 16

with (
    ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool,
    tqdm(
        total=total_pendentes,
        desc=f"{MODEL_SLUG} P{SHARD_ID + 1}/{NUM_SHARDS}",
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
        if interrompido:
            break
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
            salvar_no_unificado(link, dados_extraidos)

            sucessos += 1
            pbar.set_postfix(ok=sucessos, err=erros, dl=f"{t_dl:.1f}s", inf=f"{t_inf:.1f}s")
            log(
                f"[{idx + 1}/{total_pendentes}] doc_id={doc_id} | {len(imgs)} pág(s) | "
                f"dl={t_dl:.1f}s inf={t_inf:.1f}s | {json.dumps(dados_extraidos, ensure_ascii=False)}"
            )
        except Exception as e:
            erros += 1
            pbar.set_postfix(ok=sucessos, err=erros)
            log(f"[{idx + 1}/{total_pendentes}] ERRO doc_id={doc_id} ({link}): {e}")
        finally:
            pbar.update(1)

total_unificado = len(consolidar_todos_jsonl())
tempo_total_min = (time.time() - t_inicio) / 60
log(
    f"FINALIZADO em {tempo_total_min:.1f} min | "
    f"Novos nesta execução: {sucessos} | Erros: {erros} | "
    f"Total acumulado em '{UNIFIED_JSONL}': {total_unificado}"
)
