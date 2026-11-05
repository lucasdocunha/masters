# Relatório Comparativo de Pipelines de Extração Fiscal

Amostra total avaliada: **1300 documentos** do CEAP (anos 2014 a 2026).

## 1. Comparativo Geral de Desempenho e Acurácia

| Métrica / Campo Fiscal | Google Gemini 2.5 Flash Lite | Opção B: Qwen2.5-VL-3B (End-to-End) | Opção A: LightOnOCR + Qwen (Two-Stage) |
| :--- | :---: | :---: | :---: |
| **Documentos Processados** | **1300/1300** | **1300/1300** | **1300/1300** |
| **Tempo Médio / Doc** | **13.99s** *(API)* | **2.90s** *(Local)* | **17.99s** *(Local)* |
| **CNPJ / CPF do Emissor** | **86.0%** | **63.4%** | **78.2%** |
| **Valor Total do Documento** | **95.4%** | **87.9%** | **76.2%** |
| **Data de Emissão** | **90.3%** | **74.2%** | **84.5%** |
| **Número da Nota / Cupom** | **74.2%** | **55.1%** | **64.2%** |
| **Nome / Razão Social** | **81.2%** | **66.9%** | **72.5%** |
| **MATCH PERFEITO (Todos os 5)** | **52.2%** | **25.9%** | **28.4%** |

## 2. Acurácia de Match Perfeito por Ano (Evolução Temporal 2014-2026)

| Ano | Amostra | Gemini 2.5 Flash Lite | Qwen-VL (End-to-End) | LightOnOCR + Qwen (Two-Stage) |
| :---: | :---: | :---: | :---: | :---: |
| 2014 | 100 | 34.0% | 9.0% | 18.0% |
| 2015 | 100 | 39.0% | 12.0% | 13.0% |
| 2016 | 100 | 37.0% | 23.0% | 20.0% |
| 2017 | 100 | 47.0% | 26.0% | 33.0% |
| 2018 | 100 | 38.0% | 16.0% | 20.0% |
| 2019 | 100 | 45.0% | 20.0% | 32.0% |
| 2020 | 100 | 60.0% | 37.0% | 41.0% |
| 2021 | 100 | 59.0% | 32.0% | 35.0% |
| 2022 | 100 | 60.0% | 33.0% | 30.0% |
| 2023 | 100 | 59.0% | 29.0% | 35.0% |
| 2024 | 100 | 67.0% | 33.0% | 29.0% |
| 2025 | 100 | 58.0% | 31.0% | 30.0% |
| 2026 | 100 | 75.0% | 36.0% | 33.0% |

## 3. Conclusões e Síntese Experimental

- **Qwen2.5-VL-3B (End-to-End Local):** Destaca-se pela **altíssima velocidade (2.90s/doc)**, quase 5 vezes mais rápido que a API em nuvem do Gemini (13.99s), com acurácia de 87.9% no Valor Total, mas com taxa de acerto inferior em números de documentos e CNPJ.
- **LightOnOCR + Qwen (Two-Stage Local):** A utilização do LightOnOCR-3-4B para OCR visual completo prévio elevou o acerto de **CNPJ (+14.8%)** e **Data (+10.3%)**, alcançando **28.4% de Match Perfeito**, superando o Qwen2.5-VL direto.
- **Google Gemini 2.5 Flash Lite:** Apresenta a maior acurácia geral (52.2% de Match Perfeito), porém depende de envio de imagens para APIs externas em nuvem e latência de ~14 segundos por documento.