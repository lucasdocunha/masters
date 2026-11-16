# Relatório Comparativo de Modelos VLM: S5-nano vs Qwen3.5-4B vs Qwen3.5-9B

**Data da Análise:** 2026-11-16 10:42:09  
**Base Analisada:** 20,232 Notas Fiscais Eletrônicas (NF-e) do CEAP  
**Documentos em Comum (3 Modelos):** 20,071 (99.20%)  

## 1. Cobertura Geral e Taxas de Preenchimento

| Modelo | Docs Processados | Cobertura | Falhas de Parse (Raw) |
| :--- | :---: | :---: | :---: |
| **S5-nano** | 20,128 | 99.49% | 7 (0.03%) |
| **Qwen3.5-4B** | 20,184 | 99.76% | 106 (0.53%) |
| **Qwen3.5-9B** | 20,213 | 99.91% | 215 (1.06%) |

### Preenchimento por Campo (Docs Comuns: 20.071)

| Campo | S5-nano | Qwen3.5-4B | Qwen3.5-9B |
| :--- | :---: | :---: | :---: |
| `numero_nota` | 99.87% | 99.48% | 98.93% |
| `chave_de_acesso` | 98.05% | 98.89% | 98.16% |
| `data_emissao` | 99.93% | 99.47% | 98.93% |
| `valor_total_nota` | 99.95% | 99.48% | 98.92% |
| `cnpj_emitente` | 93.47% | 99.38% | 98.86% |
| `cnpj_destinatario` | 88.86% | 95.17% | 98.33% |

## 2. Matriz de Concordância entre os Modelos

| Campo | Consenso (3/3) | Maioria (≥2/3) | Qwen-4B == Qwen-9B | S5-nano == Qwen-9B | S5-nano == Qwen-4B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`numero_nota`** | **88.61%** | **98.67%** | 92.67% | 92.69% | 90.52% |
| **`chave_de_acesso`** | **0.98%** | **23.19%** | 20.53% | 2.23% | 2.39% |
| **`data_emissao`** | **72.82%** | **98.61%** | 81.11% | 81.32% | 81.82% |
| **`valor_total_nota`** | **87.66%** | **99.5%** | 89.84% | 95.37% | 89.62% |
| **`cnpj_emitente`** | **28.55%** | **67.84%** | 43.5% | 36.59% | 44.85% |
| **`cnpj_destinatario`** | **52.41%** | **88.36%** | 70.23% | 61.9% | 61.05% |

## 3. Validação de Chave de Acesso (SEFAZ Módulo 11)

| Modelo | Chaves Extraídas | Com exatos 44 dígitos | Válidas no Módulo 11 SEFAZ |
| :--- | :---: | :---: | :---: |
| **S5-nano** | 19,680 | 584 (2.91%) | **277 (1.38%)** |
| **Qwen3.5-4B** | 19,848 | 6,878 (34.27%) | **5,952 (29.65%)** |
| **Qwen3.5-9B** | 19,701 | 5,956 (29.67%) | **5,012 (24.97%)** |

## 4. Acurácia Real contra o Ground Truth Oficial da Câmara (CEAP)

| Campo | S5-nano | Qwen3.5-4B | Qwen3.5-9B | **Ensemble (Maioria 2/3)** |
| :--- | :---: | :---: | :---: | :---: |
| **`valor_total_nota`** | 93.29% | 87.75% | 94.78% | **95.27%** |
| **`data_emissao`** | 89.97% | 77.24% | 79.1% | **85.25%** |
| **`cnpj_emitente`** | 46.58% | 53.26% | 55.9% | **62.18%** |

