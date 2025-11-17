"""
Testes automatizados para o projeto MOH (Motor de Orientação de Habilidades)

Este pacote contém todos os testes unitários e de integração do projeto.

Estrutura:
    test_fase0.py - Validação do setup inicial
    test_graph_structures.py - Testes do grafo de habilidades
    test_graph_validation.py - Testes de detecção de ciclos e órfãos
    test_desafio1.py - Testes do Desafio 1 (DP Knapsack)
    test_desafio2.py - Testes do Desafio 2 (Permutações)
    test_desafio3.py - Testes do Desafio 3 (Guloso)
    test_desafio4.py - Testes do Desafio 4 (Sorting)
    test_desafio5.py - Testes do Desafio 5 (Recomendação)

Uso:
    # Rodar todos os testes
    pytest tests/
    
    # Rodar teste específico
    pytest tests/test_fase0.py
    
    # Rodar com coverage
    pytest tests/ --cov=src --cov-report=html
"""

__version__ = '1.0.0'

# Você pode adicionar imports compartilhados aqui se quiser
# Por exemplo:
# from .test_utils import *