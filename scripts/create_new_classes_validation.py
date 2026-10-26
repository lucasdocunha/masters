#!/usr/bin/env python3
"""
Gera pastas de validação visual com 100 amostras de alta confiança para as novas classes do CEAP:
- combustivel (Postos de Combustível / Cupons de Abastecimento)
- telefonia (Contas de Telefone / Internet / Telecomunicações)
- hospedagem (Hotéis e Pousadas / Room Folios / Diárias)
- locacao_veiculos (Locadoras de Veículos / Faturas de Locação)
- passagem_aerea (Bilhetes Aéreos / E-tickets / Companhias Aéreas)
- pedagio (Tags e Praças de Pedágio / Sem Parar / ConectCar)
"""

import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import psycopg2
import pymupdf

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
RENDERED_DIR = DATA_DIR / "rendered_pages"
PDF_DIR = Path("/media/ssd2/datasets/CEAP")
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
HISTORY_FILE = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
THUMB_DIR = Path(os.path.expanduser("~/.cache/thumbnails/large"))

RENDERED_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# Definições das classes a serem extraídas
CLASSES = {
    "combustivel": {
        "title": "COMBUSTÍVEL (Postos de Gasolina / Abastecimento)",
        "subcota_desc": "Combustíveis e Lubrificantes",
        "where": """
            d.num_subcota = 3 
            AND (f.nome_principal ILIKE '%POSTO%' OR f.nome_principal ILIKE '%COMBUSTIVEL%' 
                 OR f.nome_principal ILIKE '%PETROBRAS%' OR f.nome_principal ILIKE '%CASCOL%'
                 OR f.nome_principal ILIKE '%AUTO POSTO%' OR f.nome_principal ILIKE '%IPIRANGA%'
                 OR f.nome_principal ILIKE '%SHELL%')
        """
    },
    "telefonia": {
        "title": "TELEFONIA (Contas de Telefone / Internet / Telecom)",
        "subcota_desc": "Telefonia e Telecomunicações",
        "where": """
            (f.nome_principal ILIKE '%TELEFONICA%' OR f.nome_principal ILIKE '%CLARO%' 
             OR f.nome_principal ILIKE '%TIM S.A%' OR f.nome_principal ILIKE '%OI S.A%'
             OR f.nome_principal ILIKE '%ALGAR MULTIMIDIA%' OR f.nome_principal ILIKE '%EMBRATEL%')
            AND f.nome_principal NOT ILIKE '%PARKING%'
        """
    },
    "hospedagem": {
        "title": "HOSPEDAGEM (Hotéis / Pousadas / Room Folio)",
        "subcota_desc": "Hospedagem",
        "where": """
            d.num_subcota = 14 
            AND (f.nome_principal ILIKE '%HOTEL%' OR f.nome_principal ILIKE '%HOTEIS%' 
                 OR f.nome_principal ILIKE '%POUSADA%' OR f.nome_principal ILIKE '%FLAT%')
        """
    },
    "locacao_veiculos": {
        "title": "LOCAÇÃO DE VEÍCULOS (Faturas de Locadoras de Carro)",
        "subcota_desc": "Locação ou Fretamento de Veículos",
        "where": """
            d.num_subcota = 120 
            AND (f.nome_principal ILIKE '%LOCADORA%' OR f.nome_principal ILIKE '%LOCACAO%' 
                 OR f.nome_principal ILIKE '%VEICULOS%' OR f.nome_principal ILIKE '%RENT A CAR%'
                 OR f.nome_principal ILIKE '%PANTANAL%' OR f.nome_principal ILIKE '%MOVIDA%'
                 OR f.nome_principal ILIKE '%LOCALIZA%' OR f.nome_principal ILIKE '%UNIDAS%')
        """
    },
    "passagem_aerea": {
        "title": "PASSAGEM AÉREA (Bilhetes / E-tickets / Localizadores)",
        "subcota_desc": "Passagens Aéreas",
        "where": """
            d.num_subcota IN (9, 998, 999) 
            AND (f.nome_principal ILIKE '%TAM%' OR f.nome_principal ILIKE '%LATAM%' 
                 OR f.nome_principal ILIKE '%GOL%' OR f.nome_principal ILIKE '%AZUL%'
                 OR f.nome_principal ILIKE '%AVIANCA%')
        """
    },
    "pedagio": {
        "title": "PEDÁGIO (Extratos de Tags / Praças de Pedágio)",
        "subcota_desc": "Serviço de Pedágio e Estacionamento",
        "where": """
            (f.nome_principal ILIKE '%SEM PARAR%' OR f.nome_principal ILIKE '%CONECTCAR%'
             OR f.nome_principal ILIKE '%VELOE%' OR f.nome_principal ILIKE '%PEDAGIO%')
        """
    }
}

def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:25]

def render_page(pdf_path: Path, png_path: Path) -> bool:
    if png_path.exists():
        return True
    try:
        doc = pymupdf.open(str(pdf_path))
        if len(doc) == 0:
            doc.close()
            return False
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        pix.save(str(png_path))
        doc.close()
        return True
    except Exception as e:
        print(f"Erro renderizando {pdf_path}: {e}")
        return False

def setup_thumbnail(symlink_path: Path, png_path: Path) -> bool:
    try:
        uri = "file://" + urllib.parse.quote(str(symlink_path.absolute()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_path = THUMB_DIR / f"{h}.png"
        if thumb_path.exists():
            return True
        doc = pymupdf.open(str(png_path))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_path))
        doc.close()
        return True
    except Exception:
        return False

def make_confirm_script(class_key: str, class_title: str) -> str:
    return f"""#!/usr/bin/env python3
\"\"\"
Confirma a validação dos documentos na pasta 'validacao_{class_key}'.
- O que você MANTIVER na pasta é registrado como classe '{class_key}'.
- O que você APAGAR é desconsiderado.
Em seguida, retreina o modelo no ML backend automaticamente!
\"\"\"

import json
import subprocess
import sys
import time
from pathlib import Path
import requests

FOLDER = Path(__file__).resolve().parent
MANIFEST_FILE = FOLDER / ".candidatos.json"
HISTORY_FILE = Path("/home/lucas.ocunha/masters/ml_backend/models/annotations_history.json")
ACCEPTED_FILE = Path("/home/lucas.ocunha/masters/data/accepted_{class_key}.json")
ML_BACKEND_URL = "http://localhost:9090"

if not MANIFEST_FILE.exists():
    print(f"Erro: {{MANIFEST_FILE}} não encontrado.")
    sys.exit(1)

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

current_stems = set(p.stem.split("_")[-1] for p in FOLDER.glob("rank_*.*") if p.suffix in (".png", ".pdf"))
accepted = [m for m in manifest if m["stem"] in current_stems]
rejected = [m for m in manifest if m["stem"] not in current_stems]

print("=== Validação da Classe {class_title} ===")
print(f"Total inicial de candidatos: {{len(manifest):,}}")
print(f"  [OK] Mantidos (Confirmados como {class_key}): {{len(accepted):,}}")
print(f"  [X]  Apagados (Rejeitados/Outros):           {{len(rejected):,}}")

if not accepted:
    print("\\nNenhum arquivo para confirmar.")
    sys.exit(0)

auto_yes = "-y" in sys.argv or "--yes" in sys.argv
if not auto_yes:
    ans = input(f"\\nDeseja gravar os {{len(accepted):,}} documentos como '{class_key}' no modelo e retreinar? [S/n]: ")
    if ans.strip().lower() not in ["", "s", "sim", "y", "yes"]:
        print("Operação cancelada.")
        sys.exit(0)

accepted_stems = [m["stem"] for m in accepted]
ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(ACCEPTED_FILE, "w", encoding="utf-8") as f:
    json.dump({{"total": len(accepted), "stems": accepted_stems}}, f, indent=2)

try:
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    for stem in accepted_stems:
        history[stem] = "{class_key}"
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[OK] Histórico atualizado. Total agora: {{len(history):,}} anotações.")
except PermissionError:
    print("[Aviso] Atualizando histórico via container...")
    subprocess.run([
        "docker", "exec", "ceap-ml-backend", "python3", "-c",
        f\"\"\"
import json
with open('/data/accepted_{class_key}.json') as f:
    stems = set(json.load(f)['stems'])
with open('/app/models/annotations_history.json', 'r') as f:
    h = json.load(f)
for s in stems:
    h[s] = '{class_key}'
with open('/app/models/annotations_history.json', 'w') as f:
    json.dump(h, f, indent=2)
print(f'Histórico atualizado via container: {{len(h)}} anotações.')
\"\"\"
    ], check=True)

# Disparar retreino
print("\\nDisparando retreino no ML backend...")
try:
    subprocess.run(["docker", "restart", "ceap-ml-backend"], check=True)
    time.sleep(3)
    res = requests.post(f"{{ML_BACKEND_URL}}/train", timeout=60)
    if res.status_code == 200:
        data = res.json()
        print(f"[OK SUCESSO] Modelo retreinado: {{data.get('version')}} | Acurácia treino: {{data.get('train_accuracy', 0):.2%}}")
    else:
        print(f"Resposta retreino: {{res.status_code}} - {{res.text}}")
except Exception as e:
    print(f"Aviso no retreino: {{e}}")

print("\\nProcesso concluído com sucesso!")
"""

def main():
    t0 = time.time()
    print("=== CRIAÇÃO DE PASTAS DE VALIDAÇÃO (100 AMOSTRAS POR CLASSE) ===")
    
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    annotated_stems = set(history.keys())
    print(f"Histórico atual de anotações: {len(annotated_stems):,} amostras.")

    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/ceap")
    cur = conn.cursor()

    assigned_in_this_run = set()

    for class_key, class_info in CLASSES.items():
        print(f"\n--- Processando Classe: {class_key.upper()} ---")
        dest_dir = DESKTOP_DIR / f"validacao_{class_key}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Limpar symlinks antigos
        for old_f in dest_dir.glob("rank_*.*"):
            old_f.unlink()

        query = f"""
            SELECT DISTINCT d.ide_documento, f.nome_principal, doc.caminho_arquivo, 
                            d.valor_liquido, d.data_emissao
            FROM despesa d
            JOIN documento doc ON d.ide_documento = doc.ide_documento
            JOIN fornecedor f ON d.cnpj_fornecedor = f.cnpj_cpf
            WHERE doc.status IN ('baixado', 'processado')
              AND ({class_info['where']})
            ORDER BY d.data_emissao DESC NULLS LAST
        """
        cur.execute(query)
        rows = cur.fetchall()

        candidates = []
        for r in rows:
            stem = str(r[0])
            fornecedor = r[1] or "DESCONHECIDO"
            pdf_path = Path(r[2])
            valor = float(r[3]) if r[3] is not None else 0.0
            data_emissao = str(r[4]) if r[4] is not None else ""

            if stem in annotated_stems or stem in assigned_in_this_run:
                continue
            if not pdf_path.exists():
                continue

            candidates.append({
                "stem": stem,
                "ide_documento": r[0],
                "fornecedor": fornecedor,
                "pdf_path": str(pdf_path),
                "valor_liquido": valor,
                "data_emissao": data_emissao,
                "classe_proposta": class_key
            })

            assigned_in_this_run.add(stem)
            if len(candidates) == 100:
                break

        print(f"  -> {len(candidates)} candidatos selecionados para {class_key}.")

        # Renderizar PNGs
        print(f"  -> Verificando/Renderizando páginas...")
        manifest = []
        thumb_tasks = []

        for rank, c in enumerate(candidates, 1):
            stem = c["stem"]
            png_target = RENDERED_DIR / f"{stem}.png"
            if not png_target.exists():
                render_page(Path(c["pdf_path"]), png_target)

            forn_tag = sanitize_filename(c["fornecedor"])
            symlink_name = f"rank_{rank:04d}_[{forn_tag}]_{stem}.png"
            symlink_path = dest_dir / symlink_name

            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(png_target)

            c_info = dict(c)
            c_info["rank"] = rank
            c_info["png_target"] = str(png_target)
            c_info["symlink_name"] = symlink_name
            manifest.append(c_info)
            thumb_tasks.append((symlink_path, png_target))

        # Salvar manifesto .candidatos.json
        with open(dest_dir / ".candidatos.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Salvar script de confirmação
        script_confirmar = dest_dir / "confirmar_validacao.py"
        with open(script_confirmar, "w", encoding="utf-8") as f:
            f.write(make_confirm_script(class_key, class_info["title"]))
        script_confirmar.chmod(0o755)

        # Salvar LEIAME.txt
        with open(dest_dir / "LEIAME.txt", "w", encoding="utf-8") as f:
            f.write(f"""PASTA DE VALIDAÇÃO VISUAL - CLASSE: {class_info['title']}
============================================================

Total de documentos nesta pasta: {len(manifest):,} amostras de alta confiança.

COMO USAR:
1. Abra esta pasta no Gerenciador de Arquivos (Thunar) ou no visualizador de imagens (Ristretto):
   thunar ~/Desktop/validacao_{class_key} &

2. Navegue visualmente pelas imagens:
   - Se o documento for de fato da classe '{class_key}', MANTENHA-O na pasta.
   - Se houver algum documento intruso ou ilegível, APAGUE-O (tecla Del).

3. Para confirmar e aplicar no modelo:
   - Basta me avisar no chat ("validacao {class_key} concluida"), ou
   - Execute no terminal:
     python3 ~/Desktop/validacao_{class_key}/confirmar_validacao.py
""")

        # Pré-gerar miniaturas
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(setup_thumbnail, sym, png) for sym, png in thumb_tasks]
            done_cnt = sum(1 for fut in futures if fut.result())
        print(f"  -> Miniaturas pré-geradas para o Thunar: {done_cnt}/{len(thumb_tasks)}")

    cur.close()
    conn.close()

    print(f"\n[SUCESSO COMPLETO] Todas as 6 pastas foram criadas no Desktop em {time.time()-t0:.2f}s!")

if __name__ == "__main__":
    main()
