# 🎯 Global Solution - Motor de Orientação de Habilidades (MOH)

**Disciplina:** Engenharia de Software - Dynamic Programming  
**Professor:** André Marques  
**Data:** Novembro 2025  

---

## Repositório

**Link Repositório:** https://github.com/renan-utida/Global-Solution-2---Dynamic-Programming-MOH-


---

## 📋 Descrição do Projeto

O MOH (Motor de Orientação de Habilidades) é um sistema de otimização que guia profissionais na aquisição estratégica de habilidades para maximizar o valor de carreira e adaptabilidade no mercado de trabalho.

### 🎯 Objetivo Principal
Alcançar a habilidade **S6 - IA Generativa Ética** otimizando:
- ✅ Valor de carreira
- ✅ Tempo de aprendizado (≤ 350 horas)
- ✅ Complexidade cumulativa (≤ 30)

---

## 🏗️ Estrutura do Projeto

```
GS_DynamicProgramming_MOH/
│
├── 📓 GS_MOH_Principal.ipynb         # Notebook principal (orquestração)
├── 📄 README.md                      # Instruções de uso
├── 📄 relatorio_tecnico.pdf          # Relatório final
├── 📄 requirements.txt               # Dependências Python
│
├── 📁 data/
│   └── skills_dataset.json            # Dataset das 12 habilidades
│
├── 📁 src/
│   ├── __init__.py                    # Módulo Python
│   ├── config.py                    # Constantes globais
│   ├── decorators.py                # Decoradores de performance
│   │
│   ├── graph_validation.py          # 🔴 Validação de ciclos e órfãos
│   ├── graph_structures.py          # Grafo + estruturas de dados
│   │
│   ├── desafio1_dp_knapsack.py     # Desafio 1: DP + Monte Carlo
│   ├── desafio2_permutations.py    # Desafio 2: 120 permutações
│   ├── desafio3_greedy.py          # Desafio 3: Guloso vs Ótimo
│   ├── desafio4_sorting.py         # Desafio 4: Merge/Quick Sort
│   ├── desafio5_recommendation.py  # Desafio 5: DP look-ahead
│   │
│   ├── monte_carlo.py               # Simulação estocástica
│   ├── analysis.py                  # Análises estatísticas
│   └── visualization.py             # Gráficos e visualizações
│
├── 📂 tests/
│   ├── test_fase0.py                 
│   ├── test_fase1.py     
│   ├── test_monte_carlo.py            
│   ├── test_desafio1.py   
│   ├── test_desafio2.py   
│   ├── test_desafio3.py          
│   ├── test_desafio4.py       
│   └── test_desafio5.py               
│
└── 📁 outputs/
    ├── desafio1_results.json           # Resultados Desafio 1
    ├── desafio2_results.json           # Resultados Desafio 2
    ├── ...                             # Resultados Desafio 3, 4 e 5
    └── figures/                        # Gráficos salvos
```

---

## 🚀 Setup e Instalação

### 1. Requisitos

- Python 3.9+
- pip ou conda

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Estrutura Criada (FASE 0 - ✅ COMPLETA)

```bash
# Verificar estrutura
ls -la data/
ls -la src/
ls -la outputs/
```

**Arquivos criados:**
- ✅ `data/skills_dataset.json` - Dataset com 12 habilidades
- ✅ `src/config.py` - Constantes e funções de formatação
- ✅ `src/decorators.py` - Decoradores reutilizados
- ✅ `src/__init__.py` - Módulo Python
- ✅ `outputs/` - Diretório para resultados

---

## 📊 Dataset de Habilidades

O projeto utiliza 12 habilidades divididas em:

### 🔵 Habilidades Básicas (sem pré-requisitos)
- **S1** - Programação Básica (Python)
- **S2** - Modelagem de Dados (SQL)
- **S7** - Estruturas em Nuvem (AWS/Azure)
- **H10** - Segurança de Dados
- **H12** - Introdução a IoT

### 🔴 Habilidades Críticas
- **S3** - Algoritmos Avançados (requer S1)
- **S5** - Visualização de Dados (requer S2)
- **S7** - Estruturas em Nuvem
- **S8** - APIs e Microsserviços (requer S1)
- **S9** - DevOps & CI/CD (requer S7, S8)

### 🟢 Objetivo Final
- **S6** - IA Generativa Ética (requer S4)

### 🟣 Avançadas
- **S4** - Fundamentos de Machine Learning (requer S1, S3)
- **H11** - Análise de Big Data (requer S4)

---

## 🎯 Os 5 Desafios

### **Desafio 1 - Caminho de Valor Máximo**
- Algoritmo: DP Knapsack Multidimensional
- Incerteza: Monte Carlo (1000 cenários)
- Restrições: T ≤ 350h, C ≤ 30

### **Desafio 2 - Verificação Crítica**
- Algoritmo: Permutações (5! = 120)
- Validação: Detecção de ciclos e órfãos
- Custo: Tempo de aquisição + espera

### **Desafio 3 - Pivô Mais Rápido**
- Algoritmo: Guloso (V/T) vs Busca Exaustiva
- Contraexemplo: Demonstrar quando guloso falha
- Meta: S ≥ 15 (adaptabilidade)

### **Desafio 4 - Trilhas Paralelas**
- Algoritmo: Merge Sort ou Quick Sort
- Ordenação: Por complexidade C
- Divisão: Sprint A (1-6), Sprint B (7-12)

### **Desafio 5 - Recomendação**
- Algoritmo: DP Look-Ahead (5 anos)
- Cenários: Transições de mercado
- Output: Top 2-3 habilidades

---

## 💻 Uso

### Executar Notebook Principal

```bash
jupyter notebook GS_MOH_Principal.ipynb
```

### Executar Módulos Individualmente

```python
from src.config import *
from src.decorators import *
from src.graph_structures import SkillGraph

# Carregar dataset
import json
with open(SKILLS_DATASET_FILE) as f:
    dataset = json.load(f)

# Criar grafo
graph = SkillGraph(dataset['skills'])
```

---

## 🧪 Testes

```bash
# Executar testes unitários (quando implementados)
pytest tests/

# Validar dataset
python -c "import json; json.load(open('data/skills_dataset.json'))"
```

---

## 📈 Metodologia de Avaliação

| Critério | Pontos |
|----------|--------|
| Modelagem e estruturas (grafos, dicionários, conjuntos) | 20 |
| Implementações corretas | 35 |
| Validação do grafo (ciclos, órfãos), testes e logs | 10 |
| Relatório técnico e análise experimental | 20 |
| Qualidade do código (clareza, modularidade, docstrings) | 15 |
| Implementação no GitHub | +10 |
| **TOTAL** | **110** |

---

## 📝 Próximas Etapas

### FASE 1 - Estruturas de Dados + Validação (próxima)
- [ ] `graph_structures.py` - Grafo direcionado
- [ ] `graph_validation.py` - Detecção de ciclos/órfãos
- [ ] Testes de validação

### FASE 2-6 - Implementação dos Desafios
- [ ] Desafio 1: DP Knapsack + Monte Carlo
- [ ] Desafio 2: 120 Permutações
- [ ] Desafio 3: Guloso vs Ótimo
- [ ] Desafio 4: Merge Sort
- [ ] Desafio 5: DP Recomendação

### FASE 7-10 - Finalização
- [ ] Análise e visualização
- [ ] Notebook principal
- [ ] Relatório técnico
- [ ] Testes e documentação

---

## 👤 Autores

**Renan Dias Utida**  
RM: 558540  

**Camila Pedroza da Cunha** 
RM 558768

Curso: Engenharia de Software - FIAP  
Turma: 2ESPW

---

## 📄 Licença

Este projeto é parte de uma avaliação acadêmica da FIAP.  
Política de integridade: código autoral; referências e bibliotecas citadas.

---

**Status:** FASE 2 ✅ COMPLETA | FASE 3 🔄 EM DESENVOLVIMENTO