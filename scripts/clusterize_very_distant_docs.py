#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Clusterização e Descoberta de Classes nos 62.241 Documentos Muito Distantes (< 50% de Similaridade).
1. Isola os documentos que possuem texto no PostgreSQL e estão a < 50% dos centróides conhecidos.
2. Executa K-Means nos embeddings 128-D para agrupar em K clusters naturais.
3. Extrai para cada cluster:
   - Principais fornecedores
   - Palavras-chave / n-grams textuais mais frequentes
   - Exemplos de doc_id
   - Sugestão semântica de nova classe documental
4. Salva análise detalhada em data/clusters_muito_distantes.json
"""

import os
import sys
import json
import time
from collections import Counter
from pathlib import Path
import psycopg2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
HIST_PATH = BASE_DIR / "ml_backend" / "models" / "annotations_history.json"
META_PATH = DATA_DIR / "latent_vectors_full_metadata.json"
EMBS_PATH = DATA_DIR / "latent_vectors_metric_learned_128d.npy"
LOOKUP_FILE = DATA_DIR / "pdf_metadata_lookup.json"
OUTPUT_FILE = DATA_DIR / "clusters_muito_distantes.json"
DB_URL = "postgresql://postgres:postgres@localhost:5432/ceap"


def main():
    print("=" * 76)
    print(" CLUSTERIZAÇÃO DOS 62.241 DOCUMENTOS MUITO DISTANTES (< 50% SIM)")
    print("=" * 76)
    t0 = time.time()

    # 1. Carregar histórico e embeddings
    print("\n1. Carregando dados do espaço latente ArcFace 128-D...")
    with open(HIST_PATH) as f:
        hist = json.load(f)
    norm_hist = {k.replace(".pdf", ""): v for k, v in hist.items()}

    with open(META_PATH) as f:
        meta = json.load(f)
    docs = meta["documents"]
    stems = [d["doc_name"].replace(".pdf", "") for d in docs]
    stem_to_idx = {s: i for i, s in enumerate(stems)}

    embs = np.load(EMBS_PATH)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs /= norms

    # Centróides dominantes
    top_classes = ["uber", "cupom_fiscal", "nfse", "nfe", "energia", "correios", "telefonia"]
    top_centroids = []
    for cls in top_classes:
        c_idxs = [stem_to_idx[s] for s, lbl in norm_hist.items() if lbl == cls and s in stem_to_idx]
        cent = embs[c_idxs].mean(axis=0)
        cent /= np.linalg.norm(cent)
        top_centroids.append(cent)
    top_centroids = np.array(top_centroids)

    sims = np.dot(embs, top_centroids.T)
    max_sim = sims.max(axis=1)

    # 2. Consultar do banco os documentos com texto
    print("\n2. Consultando documentos com texto no PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT ide_documento FROM documento_texto_completo")
    docs_with_text = set(str(r[0]) for r in cur.fetchall())

    # Filtrar os muito distantes (< 50%) e não rotulados
    target_stems = [
        s for s in docs_with_text 
        if s in stem_to_idx and s not in norm_hist and max_sim[stem_to_idx[s]] < 0.50
    ]
    print(f"   ✓ Documentos identificados para clusterização: {len(target_stems):,}")

    target_indices = [stem_to_idx[s] for s in target_stems]
    target_vectors = embs[target_indices]

    # 3. Executar K-Means
    K = 12
    print(f"\n3. Executando K-Means (K={K}) nos {len(target_stems):,} vetores 128-D...")
    t_km = time.time()
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(target_vectors)
    print(f"   ✓ K-Means concluído em {time.time() - t_km:.1f}s!")

    # 4. Analisar cada cluster com Fornecedores e Textos
    print("\n4. Carregando metadados de fornecedores e analisando textos...")
    with open(LOOKUP_FILE) as f:
        lookup = json.load(f)

    # Buscar uma amostra de textos para cada cluster
    cluster_to_stems = {k: [] for k in range(K)}
    for s, lbl in zip(target_stems, cluster_labels):
        cluster_to_stems[lbl].append(s)

    cluster_summaries = []

    # Stopwords simples em português
    stopwords_pt = set([
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "nao", "não",
        "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi",
        "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "nos",
        "ja", "já", "eu", "tambem", "também", "so", "só", "pelo", "pela", "ate", "até", "isso",
        "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas",
        "me", "esse", "eles", "estao", "estão", "você", "voce", "tinha", "foram", "essa",
        "num", "nem", "suas", "meu", "as", "minha", "têm", "numa", "pelos", "elas", "havia",
        "seja", "qual", "será", "nosso", "nossa", "nossos", "nossas", "valor", "r$", "total",
        "data", "documento", "número", "numero", "cpf", "cnpj", "nome", "brasil", "df", "sp",
        "rs", "rj", "mg", "pr", "sc", "ce", "ba", "pe", "go", "es", "ma", "pa", "pb", "rn",
        "al", "pi", "to", "se", "ro", "ac", "ap", "rr", "ms", "mt", "emissão", "emissao", "via",
        "codigo", "código", "autenticação", "autenticacao", "página", "pagina", "1", "2", "3", "4"
    ])

    for k in range(K):
        c_stems = cluster_to_stems[k]
        size = len(c_stems)

        # Fornecedores
        forn_counts = Counter()
        for s in c_stems:
            f_name = lookup.get(f"{s}.pdf", {}).get("fornecedor", "NÃO INFORMADO")
            forn_counts[f_name] += 1
        top_forns = forn_counts.most_common(5)

        # Amostra de textos do PostgreSQL para n-grams
        sample_stems = c_stems[:60]
        cur.execute("""
            SELECT texto_completo 
            FROM documento_texto_completo 
            WHERE ide_documento = ANY(%s) 
            LIMIT 60
        """, ([int(s) for s in sample_stems],))
        sample_texts = [r[0] for r in cur.fetchall() if r[0]]

        # Extrair termos mais frequentes
        word_counts = Counter()
        for txt in sample_texts:
            tokens = [w.lower().strip(".,;:()[]{}\"'/-") for w in txt.split() if len(w) >= 4]
            clean_tokens = [w for w in tokens if w not in stopwords_pt and not w.isdigit()]
            word_counts.update(clean_tokens)

        top_words = [w for w, _ in word_counts.most_common(8)]

        # Tentar classificar semanticamente o cluster
        words_joined = " ".join(top_words).lower()
        forns_joined = " ".join([f[0] for f in top_forns]).lower()

        suggested_class = "outros_documentos"
        if any(w in words_joined or w in forns_joined for w in ["locacao", "locação", "aluguel", "condominio", "condomínio", "imovel", "imóvel"]):
            suggested_class = "locacao_imoveis"
        elif any(w in words_joined or w in forns_joined for w in ["estacionamento", "garagem", "veiculos", "veículo", "estac"]):
            suggested_class = "estacionamento"
        elif any(w in words_joined or w in forns_joined for w in ["consultoria", "assessoria", "pesquisa", "juridico", "jurídica", "advocacia"]):
            suggested_class = "consultoria_pesquisa"
        elif any(w in words_joined or w in forns_joined for w in ["hotel", "pousada", "hospedagem", "resort"]):
            suggested_class = "hotelaria_atipica"
        elif any(w in words_joined or w in forns_joined for w in ["combustivel", "combustível", "petroleo", "petróleo", "posto", "auto"]):
            suggested_class = "combustivel_avulso"
        elif any(w in words_joined or w in forns_joined for w in ["taxi", "táxi", "radiotaxi", "sinpetaxi"]):
            suggested_class = "recibo_taxi"
        elif any(w in words_joined or w in forns_joined for w in ["grafica", "gráfica", "impressao", "impressão", "editora", "comunicacao", "comunicação"]):
            suggested_class = "divulgacao_grafica"
        elif any(w in words_joined or w in forns_joined for w in ["aereo", "aéreo", "voo", "aviao", "avião", "aeroporto"]):
            suggested_class = "transporte_aereo"
        elif any(w in words_joined or w in forns_joined for w in ["seguranca", "segurança", "vigilancia", "vigilância"]):
            suggested_class = "seguranca_vigilancia"

        summary = {
            "cluster_id": k,
            "tamanho": size,
            "proporcao_pct": round(size / len(target_stems) * 100, 2),
            "classe_sugerida": suggested_class,
            "top_fornecedores": [{"fornecedor": f[0], "count": f[1]} for f in top_forns],
            "palavras_chave": top_words,
            "exemplos_doc_id": c_stems[:8]
        }
        cluster_summaries.append(summary)

    # Ordenar por tamanho do cluster decrescente
    cluster_summaries.sort(key=lambda x: x["tamanho"], reverse=True)

    # 5. Salvar resultado JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_documentos": len(target_stems),
            "num_clusters": K,
            "clusters": cluster_summaries
        }, f, indent=2, ensure_ascii=False)
    print(f"\n5. Análise detalhada salva em: {OUTPUT_FILE}")

    # 6. Exibir Tabela Consolidada
    print("\n" + "=" * 76)
    print(f" RESUMO DOS {K} CLUSTERS ENCONTRADOS ({len(target_stems):,} DOCUMENTOS)")
    print("=" * 76)
    print(f"{'#':2s} | {'Classe Sugerida':22s} | {'Docs':7s} | {'%':5s} | {'Palavras-Chave':30s}")
    print("-" * 76)
    for idx, c in enumerate(cluster_summaries, 1):
        kw_str = ", ".join(c["palavras_chave"][:4])
        print(f"{idx:2d} | {c['classe_sugerida']:22s} | {c['tamanho']:7,d} | {c['proporcao_pct']:4.1f}% | {kw_str[:30]:30s}")

    print("\n" + "=" * 76)
    print(f" CONCLUÍDO EM {time.time() - t0:.1f} SEGUNDOS!")
    print("=" * 76)


if __name__ == "__main__":
    main()
