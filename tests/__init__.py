"""
Testes automatizados para o projeto MOH (Motor de Orientação de Habilidades)

Este pacote contém todos os testes unitários e de integração do projeto.

Estrutura:
    test_fase0.py - Validação do setup inicial
    test_fase1.py - Testes de estruturas e validação de grafo
    test_desafio1.py - Testes do Desafio 1 (DP Knapsack + Monte Carlo)
    test_desafio2.py - Testes do Desafio 2 (120 Permutações)
    test_desafio3.py - Testes do Desafio 3 (Guloso vs Ótimo)
    test_desafio4.py - Testes do Desafio 4 (Merge Sort)
    test_desafio5.py - Testes do Desafio 5 (Recomendação DP)

Cobertura:
    ✅ Fase 0: Setup e configuração
    ✅ Fase 1: Estruturas de dados e validação (30 pontos)
    ✅ Desafio 1: DP Knapsack + Monte Carlo (7 testes)
    ✅ Desafio 2: Permutações (6 testes)
    ✅ Desafio 3: Guloso vs Ótimo (7 testes)
    ✅ Desafio 4: Merge Sort (6 testes)
    ✅ Desafio 5: Recomendação (7 testes)
    
    Total: 30+ testes unitários

Uso:
    # Rodar todos os testes
    pytest tests/
    
    # Rodar teste específico
    pytest tests/test_desafio1.py
    
    # Rodar com verbose
    pytest tests/ -v
    
    # Rodar com coverage
    pytest tests/ --cov=src --cov-report=html
    
    # Rodar apenas testes que falharam
    pytest tests/ --lf

Convenções:
    - Cada teste deve começar com test_
    - Usar fixtures quando possível
    - Testes devem ser independentes
    - Usar asserts claros com mensagens
    - Documentar casos de teste complexos
"""

__version__ = '1.0.0'
__author__ = 'Renan Dias Utida, Camila Pedroza da Cunha'

# Imports compartilhados (se necessário)
import sys
from pathlib import Path

# Adiciona o diretório src ao path (para imports relativos)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fixtures ou utilidades compartilhadas podem ir aqui
# Por exemplo:
# import pytest
# 
# @pytest.fixture
# def sample_graph():
#     from src.graph_structures import build_graph_from_file
#     return build_graph_from_file('data/skills_dataset.json')

__all__ = []  # Não exportamos nada por padrão em testes