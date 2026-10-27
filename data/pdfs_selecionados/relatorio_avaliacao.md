# Relatório de Avaliação da Extração com Google Gemini

Documentos avaliados: **1300** de **1300** disponíveis na amostra.

## 1. Métricas Globais de Acurácia

| Campo Fiscal | Acertos | Total Avaliado | Acurácia (%) |
| :--- | :--- | :--- | :--- |
| **CNPJ / CPF do Fornecedor** | 1118 | 1300 | **86.0%** |
| **Valor Total do Documento** | 1240 | 1300 | **95.4%** |
| **Data de Emissão** | 1174 | 1300 | **90.3%** |
| **Número da Nota / Documento** | 964 | 1300 | **74.2%** |
| **Nome / Razão Social do Fornecedor** | 1055 | 1300 | **81.2%** |
| **Match Perfeito (Todos os 5 campos)** | 678 | 1300 | **52.2%** |

## 2. Acurácia por Ano

| Ano | Qtd Docs | CNPJ (%) | Valor (%) | Data (%) | Número (%) | Fornecedor (%) | Perfeito (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2014 | 100 | 82% | 99% | 91% | 65% | 72% | 34% |
| 2015 | 100 | 86% | 94% | 91% | 55% | 82% | 39% |
| 2016 | 100 | 88% | 92% | 84% | 60% | 75% | 37% |
| 2017 | 100 | 83% | 93% | 90% | 74% | 77% | 47% |
| 2018 | 100 | 82% | 97% | 88% | 54% | 82% | 38% |
| 2019 | 100 | 84% | 97% | 76% | 74% | 84% | 45% |
| 2020 | 100 | 83% | 96% | 92% | 76% | 86% | 60% |
| 2021 | 100 | 82% | 95% | 94% | 82% | 81% | 59% |
| 2022 | 100 | 88% | 95% | 91% | 82% | 80% | 60% |
| 2023 | 100 | 84% | 94% | 89% | 82% | 80% | 59% |
| 2024 | 100 | 91% | 95% | 98% | 86% | 88% | 67% |
| 2025 | 100 | 93% | 96% | 95% | 84% | 81% | 58% |
| 2026 | 100 | 92% | 97% | 95% | 90% | 87% | 75% |

## 3. Análise de Discrepâncias (622 casos com divergência em pelo menos 1 campo)

| Doc ID | Ano | Campo c/ Falha | Esperado | Extraído |
| :--- | :--- | :--- | :--- | :--- |
| `5454153` | 2014 | **CNPJ** | `088.227.670/0010-8` | `None` |
| `5454153` | 2014 | **Fornecedor** | `NASCENTES DAS GERAIS` | `` |
| `5458368` | 2014 | **Data** | `2014-04-24` | `2014-04-29` |
| `5458368` | 2014 | **Fornecedor** | `FARAILDES FERREIRA SANTOS-ME` | `Bar e Restaurante de Fará - Faraildes Ferreira Santos - ME` |
| `5470705` | 2014 | **Número** | `579480` | `411554` |
| `5471134` | 2014 | **CNPJ** | `432.838.110/0864-9` | `43.283.811/0006-49` |
| `5471726` | 2014 | **CNPJ** | `628.829.640/0016-8` | `62.882.954/0001-68` |
| `5479086` | 2014 | **Número** | `S/N` | `None` |
| `5480086` | 2014 | **CNPJ** | `000.420.440/0018-4` | `00.042.044/0001-24` |
| `5481195` | 2014 | **Número** | `756383` | `720568` |
| `5484068` | 2014 | **Número** | `217` | `None` |
| `5484763` | 2014 | **CNPJ** | `088.797.830/0013-7` | `08874783000137` |
| `5484763` | 2014 | **Número** | `109024` | `104203` |
| `5486428` | 2014 | **Data** | `2014-07-16` | `2014-06-16` |
| `5486682` | 2014 | **CNPJ** | `070.472.510/0017-0` | `07047251029170` |
| `5489897` | 2014 | **Número** | `sn` | `None` |
| `5491655` | 2014 | **Número** | `002822635` | `002928316` |
| `5491655` | 2014 | **Fornecedor** | `CELPE GRUPO NEOENERGIA` | `COMPANHIA ENERGETICA DE PERNAMBUCO - CELPE` |
| `5491921` | 2014 | **Data** | `2014-06-14` | `None` |
| `5491944` | 2014 | **Fornecedor** | `EGR` | `Empresa Gaucha de Rodovias S.A. - EGR` |
| `5492401` | 2014 | **Número** | `5000347` | `None` |
| `5492401` | 2014 | **Fornecedor** | `concessionaria via rondon s/a` | `Concessionaria ViaRondon S/A` |
| `5493483` | 2014 | **Fornecedor** | `POSTO TOSCANO` | `POSTO TOSCANO - DANIELLY MARIA DA COSTA GALVAO` |
| `5494252` | 2014 | **Fornecedor** | `ECT` | `ECT - EMP. BRAS. DE CORREIOS E TELEGRAFOS` |
| `5494607` | 2014 | **Número** | `132767` | `None` |