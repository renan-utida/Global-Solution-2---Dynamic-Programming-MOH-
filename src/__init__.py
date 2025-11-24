"""
Módulo src - Motor de Orientação de Habilidades (MOH)
Global Solution - Dynamic Programming

Este módulo contém todos os componentes necessários para resolver
os 5 desafios da Global Solution.

Estrutura:
    config.py - Constantes e configurações globais
    decorators.py - Decoradores reutilizáveis (@measure_performance, etc)
    
    graph_structures.py - Estruturas de dados (SkillGraph)
    graph_validation.py - Validação de ciclos e órfãos
    
    desafio1_dp_knapsack.py - Desafio 1: DP Knapsack + Monte Carlo
    desafio2_permutations.py - Desafio 2: 120 Permutações
    desafio3_greedy.py - Desafio 3: Guloso vs Ótimo
    desafio4_sorting.py - Desafio 4: Merge Sort
    desafio5_recommendation.py - Desafio 5: Recomendação DP
    
    monte_carlo.py - Simulação Monte Carlo
    analysis.py - Análises estatísticas
    visualization.py - Gráficos e visualizações

Uso típico:
    >>> from src.graph_structures import build_graph_from_file
    >>> from src.desafio1_dp_knapsack import solve_complete as solve_d1
    >>> 
    >>> graph = build_graph_from_file('data/skills_dataset.json')
    >>> resultado = solve_d1(graph)
"""

__version__ = '1.0.0'
__author__ = 'Renan Dias Utida, Camila Pedroza da Cunha'
__course__ = 'Engenharia de Software - FIAP'

# Importações principais para acesso fácil
from .config import *
from .decorators import *

# Importações de estruturas
try:
    from .graph_structures import SkillGraph, build_graph_from_file
    from .graph_validation import validate_graph, detect_cycles, find_orphan_nodes
except ImportError:
    pass  # Permite importação parcial durante desenvolvimento

try:
    from .desafio1_dp_knapsack import (
        solve_complete as solve_desafio1,
        save_desafio1_results,
        run_desafio1_complete
    )
    from .desafio2_permutations import (
        solve_complete as solve_desafio2,
        save_desafio2_results,
        run_desafio2_complete
    )
    from .desafio3_greedy import (
        solve_complete as solve_desafio3,
        save_desafio3_results,
        run_desafio3_complete 
    )
    from .desafio4_sorting import (
        solve_complete as solve_desafio4,
        save_desafio4_results, 
        run_desafio4_complete 
    )
    from .desafio5_recommendation import (
        solve_complete as solve_desafio5,
        save_desafio5_results,
        run_desafio5_complete
    )
except ImportError:
    pass

# Importações de análise
try:
    from .analysis import (
        compare_deterministic_vs_stochastic,
        analyze_permutations_costs,
        compare_greedy_vs_optimal,
        generate_summary_report
    )
    from .visualization import (
        plot_monte_carlo_distribution,
        plot_permutations_comparison,
        plot_greedy_vs_optimal,
        plot_market_scenarios,
        create_dashboard,
        plot_graph_by_levels
    )
except ImportError:
    pass  # Permite importação parcial durante desenvolvimento

__all__ = [
    # Módulos base
    'config',
    'decorators',
    
    # Estruturas
    'SkillGraph',
    'build_graph_from_file',
    'validate_graph',
    'detect_cycles',
    'find_orphan_nodes',
    
    # Soluções dos desafios
    'solve_desafio1',
    'solve_desafio2',
    'solve_desafio3',
    'solve_desafio4',
    'solve_desafio5',
    
    # Funções completas (run_complete)
    'run_desafio1_complete',
    'run_desafio2_complete',
    'run_desafio3_complete',
    'run_desafio4_complete',
    'run_desafio5_complete',
    
    # Funções de salvamento
    'save_desafio1_results',
    'save_desafio2_results',
    'save_desafio3_results',
    'save_desafio4_results',
    'save_desafio5_results',
    
    # Análises
    'compare_deterministic_vs_stochastic',
    'analyze_permutations_costs',
    'compare_greedy_vs_optimal',
    'generate_summary_report',
    
    # Visualizações
    'plot_graph_by_levels',
    'plot_graph_structure',
    'plot_monte_carlo_distribution',
    'plot_permutations_comparison',
    'plot_greedy_vs_optimal',
    'plot_market_scenarios',
    'create_dashboard',
]