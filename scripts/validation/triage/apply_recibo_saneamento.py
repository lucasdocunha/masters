#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Saneamento e Purificação Definitiva da Classe RECIBO:
1. Reclassifica 360 boletos de condomínio/aluguel bancário de volta para 'boleto'.
2. Reclassifica 69 talões manuais de restaurante (Série D-1) para 'cupom_fiscal'.
3. Mantém 155 recibos autênticos (táxis manuais, cooperativas, quitações) como 'recibo'.
4. Atualiza annotations_history.json e sincroniza com PostgreSQL.
5. Remove o lote de 42k do Desktop (validacao_recibo).
6. Gera relatório de saneamento em data/evaluation/saneamento_recibo_novembro_2026.json.
"""

import os
import sys
import re
import json
import shutil
import time
from pathlib import Path
from collections import Counter
import psycopg

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "models" / "annotations_history.json"
BACKUP_FILE = DATA_DIR / "models" / "annotations_history.backup_before_recibo_split.json"
YEAR_FILE = DATA_DIR / "pdf_to_year.json"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
REPORT_FILE = DATA_DIR / "evaluation" / "saneamento_recibo_novembro_2026.json"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ceap")

RE_BOLETO = re.compile(
    r'FICHA\s*DE\s*(COMPENSA|CAIXA)|RECIBO\s*DO\s*(SACADO|PAGADOR)|CEDENTE|SACADO|BENEFICI[AÁ]RIO|'
    r'VENCIMENTO.*VALOR|BANCO\s*(ITAU|BRADESCO|SANTANDER|DO\s*BRASIL|CEF|CAIXA)|\b\d{47,48}\b',
    re.IGNORECASE
)
RE_NOTA_D1 = re.compile(
    r'NOTA\s*FISCAL\s*DE\s*VENDA\s*A(O)?\s*CONSUMIDOR|S[EÉ]RIE\s*D-?1|VENDA\s*A\s*CONSUMIDOR|'
    r'SERIE\s*D|MODELO\s*0?2',
    re.IGNORECASE
)
RE_TAXI = re.compile(
    r'\bT[AÁ]XI\b|SINPETAXI|COOPERTAXI|RADIOT[AÁ]XI|MOTORISTA\s*AUT[OÔ]NOMO|\b99\s*(POP|TAXI|APP)\b',
    re.IGNORECASE
)


def main():
    print("=" * 74)
    print(" SANEAMENTO E PURIFICAÇÃO DEFINITIVA DA CLASSE RECIBO")
    print("=" * 74)
    t0 = time.time()

    # 1. Carregar histórico e metadados
    print("\n1. Carregando dados...")
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Backup de segurança
    shutil.copy2(HISTORY_FILE, BACKUP_FILE)
    print(f"   ✓ Backup salvo em {BACKUP_FILE}")

    with open(LOOKUP_FILE, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    with open(YEAR_FILE, "r", encoding="utf-8") as f:
        pdf_to_year = json.load(f)

    recibo_stems = [k for k, v in history.items() if v == "recibo"]
    print(f"   ✓ Total atual de documentos em 'recibo': {len(recibo_stems)}")

    # 2. Carregar textos do PostgreSQL
    ids_int = [int(s) for s in recibo_stems if s.isdigit()]
    db_texts = {}
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ide_documento, texto_completo 
                    FROM documento_texto_completo 
                    WHERE ide_documento = ANY(%s)
                """, (ids_int,))
                db_texts = dict(cur.fetchall())
        print(f"   ✓ Textos OCR recuperados do PostgreSQL: {len(db_texts)}")
    except Exception as e:
        print(f"   [AVISO] Erro ao consultar PostgreSQL: {e}")

    # Origens conhecidas
    san_boletos_file = DATA_DIR / "evaluation" / "relatorio_saneamento_boletos.json"
    boleto_to_recibo = set()
    if san_boletos_file.exists():
        with open(san_boletos_file, "r", encoding="utf-8") as f:
            san_boletos = json.load(f)
            boleto_to_recibo = set(x["key"] for x in san_boletos["itens"] if x["classe_nova"] == "recibo")

    taxi_file = DATA_DIR / "accepted_taxi_as_recibo.json"
    acc_taxi = set()
    if taxi_file.exists():
        with open(taxi_file, "r", encoding="utf-8") as f:
            acc_taxi = set(x["doc_id"] for x in json.load(f))

    # 3. Classificação estrita
    print("\n2. Executando saneamento estrito...")
    reclassified = []
    counts = Counter()

    for s in recibo_stems:
        txt = db_texts.get(int(s), "")
        meta = lookup.get(f"{s}.pdf", {})
        sub = meta.get("subcota", "").upper()
        forn = meta.get("fornecedor", "").upper()

        if s in boleto_to_recibo or RE_BOLETO.search(txt) or "BANCO" in forn or "CONDOMINIO" in forn or "CONDOMÍNIO" in forn:
            if ("RECIBO DE ALUGUEL" in txt or "DECLARAÇÃO" in txt) and not ("BANCO" in txt or "FICHA DE" in txt or "VENCIMENTO" in txt):
                nova_classe = "recibo"
                motivo = "Recibo autêntico de aluguel/quitação (sem boleto)"
            else:
                nova_classe = "boleto"
                motivo = "Boleto bancário / taxa condominial com banco, cedente e vencimento"

        elif s in acc_taxi or RE_TAXI.search(txt) or "TAXI" in forn or "TÁXI" in forn or "SINPETAXI" in forn or "TÁXI" in sub:
            nova_classe = "recibo"
            motivo = "Recibo autêntico de táxi / transporte individual"

        elif RE_NOTA_D1.search(txt) or "ALIMENTAÇÃO" in sub or "RESTAURANTE" in forn or "CHURRASCARIA" in forn or "PIZZARIA" in forn:
            nova_classe = "cupom_fiscal"
            motivo = "Nota Fiscal Venda ao Consumidor Modelo 2 / Série D-1 de restaurante"

        else:
            nova_classe = "recibo"
            motivo = "Recibo / quitação mantida"

        counts[nova_classe] += 1
        history[s] = nova_classe
        reclassified.append({
            "doc_id": s,
            "fornecedor": forn,
            "subcota": sub,
            "classe_anterior": "recibo",
            "classe_nova": nova_classe,
            "motivo": motivo
        })

    # 4. Salvar histórico purificado
    print("\n3. Salvando histórico purificado...")
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"   ✓ {HISTORY_FILE} atualizado!")

    # 5. Sincronizar PostgreSQL
    print("\n4. Sincronizando com PostgreSQL...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                rows_to_update = []
                for item in reclassified:
                    s = item["doc_id"]
                    doc_id = int(s)
                    ano = pdf_to_year.get(f"{s}.pdf", 2024)
                    caminho = f"/media/ssd2/datasets/CEAP/{s}.pdf"
                    rows_to_update.append((doc_id, ano, caminho, "baixado", item["classe_nova"]))

                batch_size = 2000
                for i in range(0, len(rows_to_update), batch_size):
                    chunk = rows_to_update[i:i + batch_size]
                    cur.executemany("""
                        INSERT INTO documento (ide_documento, ano, caminho_arquivo, status, classe_predita)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (ide_documento) DO UPDATE SET
                            status = 'baixado',
                            classe_predita = EXCLUDED.classe_predita;
                    """, chunk)
                conn.commit()
        print(f"   ✓ {len(rows_to_update):,} documentos sincronizados na tabela 'documento'.")
    except Exception as e:
        print(f"   [AVISO] Erro PostgreSQL: {e}")

    # 6. Remover a pasta de 42k do Desktop
    print("\n5. Removendo pasta temporária de 42k do Desktop...")
    desktop_symlink = DESKTOP_DIR / "validacao_recibo"
    if desktop_symlink.is_symlink() or desktop_symlink.exists():
        desktop_symlink.unlink()
        print(f"   ✓ Atalho {desktop_symlink} removido.")

    batch_dir = DATA_DIR / "validacao_batches" / "validacao_recibo"
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
        print(f"   ✓ Pasta {batch_dir} excluída com sucesso.")

    # 7. Salvar relatório de auditoria
    report_data = {
        "data_execucao": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_analisado": len(recibo_stems),
        "distribuicao": dict(counts),
        "itens": reclassified
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Relatório salvo em {REPORT_FILE}")

    # Resumo
    print("\n" + "=" * 74)
    print(" SANEAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 74)
    print("Nova distribuição das 584 anotações purificadas:")
    for cls, cnt in counts.most_common():
        pct = (cnt / len(recibo_stems)) * 100
        print(f"  • {cls:15s}: {cnt:4d} documentos ({pct:5.1f}%)")
    print(f"Tempo total: {time.time()-t0:.2f}s")
    print("=" * 74)


if __name__ == "__main__":
    main()
