# 🎯 Global Solution - Motor de Orientação de Habilidades (MOH)

**Disciplina:** Engenharia de Software - Dynamic Programming  
**Professor:** André Marques  
**Data:** Novembro 2025  

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
├── 📄 README.md                      # Este arquivo
├── 📄 RELATORIO_TECNICO.md          # Relatório técnico completo
├── 📄 requirements.txt               # Dependências Python
│
├── 📁 data/
│   └── skills_dataset.json          # Dataset das 12 habilidades
│
├── 📁 src/
│   ├── __init__.py                  # Módulo Python
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
│   └── visualization.py             # Gráficos e plots
│
├── 📂 tests/
│   ├── test_fase0.py                 
│   ├── test_fase1.py                
│   ├── test_desafio1.py   
│   ├── test_desafio2.py   
│   ├── test_desafio3.py          
│   ├── test_desafio4.py       
│   ├── test_desafio5.py  
│   │
│   └── Não sei se há mais algum arquivo de teste.py             
│
└── 📁 outputs/
    ├── desafio1_results.json           # Resultados Desafio 1
    ├── desafio2_results.json           # Resultados Desafio 2
    ├── ...                             # Resultados Desafio 3, 4 e 5
    └── figures/                        # Gráficos salvos
```

📦 gs-moh-dynamic-programming/
│
├── 📄 README.md                          # Instruções de uso
├── 📄 requirements.txt                   # Dependências
├── 📄 relatorio_tecnico.pdf              # Relatório final
│
├── 📂 data/
│   ├── skills_dataset.json               # Dataset base (12 habilidades)
│   └── market_transitions.json           # Probabilidades (Desafio 5)
│
├── 📂 src/
│   │
│   ├── 📄 config.py                      # Constantes globais + formatação
│   ├── 📄 decorators.py                  # @measure_performance, @validate_inputs
│   │
│   ├── 📄 graph_structures.py            # CRÍTICO - Grafo + Validação
│   │   ├── class SkillGraph              # Grafo direcionado ponderado
│   │   ├── detect_cycles()               # DFS para ciclos
│   │   ├── find_orphan_nodes()           # Nós com pré-reqs inválidos
│   │   ├── topological_sort()            # Ordenação topológica
│   │   └── validate_graph()              # Valida antes de otimizar
│   │
│   ├── 📄 challenge1_max_value.py        # Desafio 1: DP Multidimensional
│   │   ├── knapsack_2d_dp()              # Knapsack com T e C
│   │   ├── monte_carlo_uncertainty()     # 1000 cenários V~Uniforme
│   │   ├── deterministic_solution()      # Sem incerteza
│   │   └── compare_solutions()           # E[V], std, comparação
│   │
│   ├── 📄 challenge2_critical_path.py    # Desafio 2: Permutações + Validação
│   │   ├── enumerate_permutations()      # 5! = 120 permutações
│   │   ├── calculate_total_cost()        # Tempo + Espera pré-reqs
│   │   ├── find_top_3_orders()           # 3 melhores
│   │   └── analyze_heuristics()          # Justificativa
│   │
│   ├── 📄 challenge3_greedy_pivot.py     # Desafio 3: Greedy vs Ótimo
│   │   ├── greedy_by_ratio()             # Guloso V/T
│   │   ├── exhaustive_search()           # Busca exaustiva (ótimo)
│   │   ├── generate_counterexample()     # Contraexemplo
│   │   └── complexity_analysis()         # Discussão Big-O
│   │
│   ├── 📄 challenge4_sorting.py          # Desafio 4: Merge/Quick Sort
│   │   ├── merge_sort()                  # Implementação própria
│   │   ├── quick_sort()                  # Implementação própria
│   │   ├── divide_sprints()              # Sprint A + B
│   │   └── compare_with_native()         # Benchmark vs sorted()
│   │
│   ├── 📄 challenge5_recommendation.py   # Desafio 5: DP Horizonte
│   │   ├── dp_finite_horizon()           # DP com look-ahead
│   │   ├── simulate_market_transitions() # Probabilidades de cenário
│   │   └── recommend_top_skills()        # 2-3 habilidades
│   │
│   ├── 📄 analysis.py                    # Análises comparativas
│   │   ├── complexity_analysis()         # Big-O de cada desafio
│   │   ├── experimental_results()        # Tempos medidos
│   │   └── generate_metrics_table()      # Tabelas para relatório
│   │
│   └── 📄 visualization.py               # Gráficos e visualizações
│       ├── plot_graph_structure()        # Visualiza grafo de habilidades
│       ├── plot_monte_carlo_distribution() # Histograma E[V]
│       ├── plot_time_vs_input_size()     # Performance experimental
│       ├── plot_permutations_cost()      # Top 3 vs médio
│       └── create_dashboard()            # Dashboard consolidado
│
├── 📂 tests/
│   ├── test_graph_validation.py          # Testa ciclos, órfãos
│   ├── test_challenge1.py                # Testa DP multidimensional
│   ├── test_challenge2.py                # Testa permutações
│   ├── test_challenge3.py                # Testa greedy vs ótimo
│   ├── test_challenge4.py                # Testa sorting
│   └── test_challenge5.py                # Testa recomendações
│
├── 📂 notebooks/
│   └── main_execution.ipynb              # Notebook principal (orquestração)
│
└── 📂 results/
    ├── challenge1_results.json           # Resultados Desafio 1
    ├── challenge2_results.json           # Resultados Desafio 2
    ├── challenge3_results.json           # Resultados Desafio 3
    ├── challenge4_results.json           # Resultados Desafio 4
    ├── challenge5_results.json           # Resultados Desafio 5
    └── figures/                          # Gráficos salvos

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