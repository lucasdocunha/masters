-- =============================================================================
-- CEAP Document AI & Multimodal Clustering Schema (Lean Senior Edition)
-- PostgreSQL + pgvector + pg_trgm
-- Design enxuto, sem particionamento prematuro, otimizado para consultas diretas.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- -----------------------------------------------------------------------------
-- 1. Base Cadastral e Rótulos Oficiais (Ground Truth da CEAP)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS deputado (
    id_deputado       INTEGER PRIMARY KEY,         -- nuDeputadoId oficial
    ide_cadastro      INTEGER,
    nome_parlamentar  TEXT NOT NULL,
    cpf               CHAR(11),
    sg_uf             CHAR(2),                     -- Estado (SP, RJ, PR, etc.)
    sg_partido        VARCHAR(30),
    legislatura       INTEGER,
    criado_em         TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_deputado_uf_partido ON deputado (sg_uf, sg_partido);
CREATE INDEX IF NOT EXISTS idx_deputado_nome_trgm ON deputado USING gin (nome_parlamentar gin_trgm_ops);

CREATE TABLE IF NOT EXISTS fornecedor (
    cnpj_cpf          VARCHAR(14) PRIMARY KEY,     -- PK natural limpa (apenas dígitos)
    tipo_pessoa       CHAR(1) NOT NULL,            -- 'J' ou 'F'
    nome_principal    TEXT NOT NULL,               -- Razão social / nome canônico
    criado_em         TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_fornecedor_nome_trgm ON fornecedor USING gin (nome_principal gin_trgm_ops);

CREATE TABLE IF NOT EXISTS subcota (
    num_subcota       INTEGER NOT NULL,
    num_especificacao INTEGER NOT NULL DEFAULT 0,
    descricao         TEXT NOT NULL,
    descricao_espec   TEXT,
    PRIMARY KEY (num_subcota, num_especificacao)
);

-- -----------------------------------------------------------------------------
-- 2. Documentos Físicos e Despesas Fiscais
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documento (
    ide_documento     BIGINT PRIMARY KEY,          -- ID oficial da CEAP (ex: 6033600)
    ano               SMALLINT NOT NULL,
    caminho_arquivo   TEXT NOT NULL,
    url_origem        TEXT,
    sha256            CHAR(64),
    tamanho_bytes     BIGINT,
    num_paginas       SMALLINT DEFAULT 1,
    status            VARCHAR(20) DEFAULT 'pendente', -- pendente, baixado, processado, erro
    
    -- Tipologia e Clusterização incorporadas diretamente (Zero JOINs necessários)
    cluster_id        SMALLINT,                    -- 0 a 4 (ColPali VLM)
    dist_centroide    REAL,                        -- Distância ao centroide do cluster
    classe_predita    VARCHAR(30),                 -- 'NFSE', 'DANFE', 'CUPOM', 'UBER', 'FATURA'
    confianca_classe  REAL,
    coord_x           REAL,                        -- UMAP 3D X
    coord_y           REAL,                        -- UMAP 3D Y
    coord_z           REAL,                        -- UMAP 3D Z
    criado_em         TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_documento_ano_cluster ON documento (ano, cluster_id);
CREATE INDEX IF NOT EXISTS idx_documento_classe ON documento (classe_predita);
CREATE INDEX IF NOT EXISTS idx_documento_status ON documento (status);

CREATE TABLE IF NOT EXISTS despesa (
    id_despesa        BIGSERIAL PRIMARY KEY,
    ide_documento     BIGINT REFERENCES documento(ide_documento) ON DELETE SET NULL,
    id_deputado       INTEGER NOT NULL REFERENCES deputado(id_deputado),
    cnpj_fornecedor   VARCHAR(14) REFERENCES fornecedor(cnpj_cpf),
    num_subcota       INTEGER NOT NULL,
    num_especificacao INTEGER NOT NULL DEFAULT 0,
    tipo_documento    SMALLINT,                    -- 0=Nota Fiscal, 1=Recibo, 2=Passagem
    numero_documento  TEXT,
    data_emissao      DATE,
    valor_documento   NUMERIC(12,2) NOT NULL,
    valor_glosa       NUMERIC(12,2) DEFAULT 0.00,
    valor_liquido     NUMERIC(12,2) NOT NULL,
    mes               SMALLINT NOT NULL,           -- 1 a 12
    ano               SMALLINT NOT NULL,
    parcela           INTEGER DEFAULT 0,
    lote              INTEGER,
    num_ressarcimento INTEGER,
    passageiro        TEXT,
    trecho            TEXT,
    FOREIGN KEY (num_subcota, num_especificacao) REFERENCES subcota(num_subcota, num_especificacao)
);
CREATE INDEX IF NOT EXISTS idx_despesa_filtro_ano_mes ON despesa (ano, mes, num_subcota);
CREATE INDEX IF NOT EXISTS idx_despesa_deputado_ano ON despesa (id_deputado, ano);
CREATE INDEX IF NOT EXISTS idx_despesa_fornecedor ON despesa (cnpj_fornecedor);
CREATE INDEX IF NOT EXISTS idx_despesa_doc ON despesa (ide_documento);

-- -----------------------------------------------------------------------------
-- 3. Embeddings Multimodais (pgvector)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embedding_documento (
    ide_documento     BIGINT PRIMARY KEY REFERENCES documento(ide_documento) ON DELETE CASCADE,
    modelo            VARCHAR(50) DEFAULT 'vidore/colpali-v1.2',
    vetor             vector(128) NOT NULL,        -- ColPali (128 dimensões)
    criado_em         TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_emb_doc_hnsw ON embedding_documento 
USING hnsw (vetor vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------------------------
-- 4. Extração Estruturada e Validação (Predições da IA x Ground Truth)
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS extracao_documento (
    id_extracao       BIGSERIAL PRIMARY KEY,
    ide_documento     BIGINT NOT NULL REFERENCES documento(ide_documento) ON DELETE CASCADE,
    modelo            VARCHAR(50) NOT NULL,        -- 'gemini-2.5-flash', 'tesseract', etc.
    
    -- Campos Extraídos pelo Modelo
    cnpj_extraido     VARCHAR(14),
    nome_fornecedor   TEXT,
    numero_documento  TEXT,
    data_emissao      DATE,
    valor_total       NUMERIC(12,2),
    itens_json        JSONB,                       -- Lista de produtos/serviços
    tempo_resposta_ms INTEGER,
    tokens_entrada    INTEGER,
    tokens_saida      INTEGER,
    custo_usd         NUMERIC(8,5),
    
    -- Métricas de Avaliação Automática contra o Ground Truth Oficial
    cnpj_confere      BOOLEAN,
    valor_confere     BOOLEAN,                     -- Diferença <= R$ 0.05
    data_confere      BOOLEAN,
    numero_confere    BOOLEAN,
    sim_fornecedor    REAL,                        -- Jaccard/Trigram similarity
    score_geral       REAL,                        -- Score agregado (0.0 a 1.0)
    status_validacao  VARCHAR(30) DEFAULT 'pendente', -- correto, divergente_valor, etc.
    
    extraido_em       TIMESTAMPTZ DEFAULT now(),
    UNIQUE (ide_documento, modelo)
);
CREATE INDEX IF NOT EXISTS idx_extracao_modelo_score ON extracao_documento (modelo, score_geral);
CREATE INDEX IF NOT EXISTS idx_extracao_doc ON extracao_documento (ide_documento);

-- -----------------------------------------------------------------------------
-- 5. Views Analíticas Prontas para Pesquisa
-- -----------------------------------------------------------------------------

-- Visão 1: Painel Geral Unificado por Documento
CREATE OR REPLACE VIEW vw_documento_detalhado AS
SELECT 
    d.ide_documento,
    d.ano,
    d.num_paginas,
    d.status,
    d.cluster_id,
    d.classe_predita,
    desp.id_despesa,
    desp.mes,
    desp.data_emissao,
    desp.valor_liquido,
    dep.nome_parlamentar,
    dep.sg_uf,
    dep.sg_partido,
    sub.descricao AS subcota_descricao,
    forn.cnpj_cpf AS fornecedor_cnpj,
    forn.nome_principal AS fornecedor_nome,
    ext.modelo AS extrator_modelo,
    ext.score_geral AS extracao_score,
    ext.valor_confere AS extracao_valor_ok,
    ext.cnpj_confere AS extracao_cnpj_ok
FROM documento d
LEFT JOIN despesa desp ON desp.ide_documento = d.ide_documento
LEFT JOIN deputado dep ON dep.id_deputado = desp.id_deputado
LEFT JOIN subcota sub ON sub.num_subcota = desp.num_subcota AND sub.num_especificacao = desp.num_especificacao
LEFT JOIN fornecedor forn ON forn.cnpj_cpf = desp.cnpj_fornecedor
LEFT JOIN extracao_documento ext ON ext.ide_documento = d.ide_documento;

-- Visão 2: Benchmark Comparativo de Modelos de Extração
CREATE OR REPLACE VIEW vw_benchmark_modelos AS
SELECT 
    modelo,
    COUNT(*) AS total_avaliados,
    ROUND(100.0 * COUNT(*) FILTER (WHERE cnpj_confere) / COUNT(*), 2) AS acuracia_cnpj_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE valor_confere) / COUNT(*), 2) AS acuracia_valor_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE data_confere) / COUNT(*), 2) AS acuracia_data_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE numero_confere) / COUNT(*), 2) AS acuracia_numero_pct,
    ROUND(AVG(sim_fornecedor)::numeric * 100, 2) AS media_sim_fornecedor_pct,
    ROUND(AVG(score_geral)::numeric * 100, 2) AS score_geral_pct,
    ROUND(AVG(tempo_resposta_ms)) AS latencia_media_ms,
    ROUND(SUM(custo_usd)::numeric, 4) AS custo_total_usd
FROM extracao_documento
GROUP BY modelo
ORDER BY score_geral_pct DESC;

-- Visão 3: Análise Agregada dos Clusters
CREATE OR REPLACE VIEW vw_analise_clusters AS
SELECT 
    d.cluster_id,
    d.classe_predita,
    COUNT(DISTINCT d.ide_documento) AS total_documentos,
    ROUND(AVG(d.num_paginas), 2) AS media_paginas,
    ROUND(AVG(desp.valor_liquido), 2) AS ticket_medio,
    ROUND(SUM(desp.valor_liquido), 2) AS valor_total_reembolsado,
    MODE() WITHIN GROUP (ORDER BY sub.descricao) AS categoria_predominante,
    MODE() WITHIN GROUP (ORDER BY forn.nome_principal) AS fornecedor_mais_frequente
FROM documento d
JOIN despesa desp ON desp.ide_documento = d.ide_documento
JOIN subcota sub ON sub.num_subcota = desp.num_subcota AND sub.num_especificacao = desp.num_especificacao
LEFT JOIN fornecedor forn ON forn.cnpj_cpf = desp.cnpj_fornecedor
WHERE d.cluster_id IS NOT NULL
GROUP BY d.cluster_id, d.classe_predita
ORDER BY total_documentos DESC;
