"""
Desafio 1 - Caminho de Valor Máximo

Implementa o algoritmo de Programação Dinâmica (DP) Knapsack Multidimensional
para encontrar a sequência ótima de habilidades que maximiza o valor total,
respeitando as restrições de:
- Tempo total ≤ 350 horas
- Complexidade cumulativa ≤ 30
- Pré-requisitos obrigatórios

Além disso, implementa simulação Monte Carlo para lidar com incerteza
nos valores (V ~ Uniforme[V-10%, V+10%]).

Algoritmo:
    DP[i][t][c] = max valor usando skills[0:i] com tempo t e complexidade c
    
    Recorrência:
    DP[i][t][c] = max(
        DP[i-1][t][c],                       # Não pega skill i
        DP[i-1][t-T[i]][c-C[i]] + V[i]      # Pega skill i (se satisfaz pré-reqs)
    )

Complexidade:
    O(n × T × C) onde n = número de skills, T = max_time, C = max_complexity
    Para este problema: O(12 × 350 × 30) = O(126,000)
"""

from typing import Dict, List, Tuple, Set, Any, Optional
import numpy as np
from dataclasses import dataclass

from src.graph_structures import SkillGraph, build_graph_from_file
from src.monte_carlo import (
    generate_scenarios,
    run_monte_carlo,
    compare_deterministic_vs_stochastic,
    print_monte_carlo_summary,
    MonteCarloResult
)
from src.decorators import measure_performance


@dataclass
class KnapsackSolution:
    """
    Solução do problema de knapsack.
    
    Attributes:
        path: Lista de IDs das habilidades selecionadas
        total_value: Valor total acumulado
        total_time: Tempo total gasto
        total_complexity: Complexidade total acumulada
        reaches_target: Se o caminho alcança S6 (IA Generativa Ética)
        skill_details: Detalhes de cada habilidade no caminho
    """
    path: List[str]
    total_value: float
    total_time: int
    total_complexity: int
    reaches_target: bool
    skill_details: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'path': self.path,
            'total_value': self.total_value,
            'total_time': self.total_time,
            'total_complexity': self.total_complexity,
            'reaches_target': self.reaches_target,
            'skill_details': self.skill_details,
            'path_formatted': ' → '.join(self.path)
        }
    
    def __str__(self) -> str:
        """Representação string formatada."""
        target_status = "✅ ALCANÇA S6" if self.reaches_target else "❌ NÃO ALCANÇA S6"
        return (
            f"Knapsack Solution:\n"
            f"  Caminho: {' → '.join(self.path)}\n"
            f"  Valor total: {self.total_value:.2f}\n"
            f"  Tempo total: {self.total_time}h / 350h\n"
            f"  Complexidade: {self.total_complexity} / 30\n"
            f"  {target_status}"
        )


def dp_knapsack_2d(
    graph: SkillGraph,
    max_time: int = 350,
    max_complexity: int = 30,
    target_skill: str = 'S6',
    respect_prerequisites: bool = True
) -> KnapsackSolution:
    """
    Resolve o problema de Knapsack Multidimensional (2D) usando Programação Dinâmica.
    
    Encontra a sequência de habilidades que maximiza o valor total, respeitando:
    - Tempo total ≤ max_time
    - Complexidade cumulativa ≤ max_complexity
    - Pré-requisitos obrigatórios
    
    Args:
        graph: Grafo de habilidades
        max_time: Tempo máximo disponível (horas)
        max_complexity: Complexidade máxima acumulada
        target_skill: Habilidade objetivo (padrão: 'S6')
        respect_prerequisites: Se True, respeita ordem topológica
    
    Returns:
        KnapsackSolution: Solução ótima encontrada
    
    Complexity:
        O(n × T × C) onde n = número de skills, T = max_time, C = max_complexity
        Para este problema: O(12 × 350 × 30) = O(126,000)
    
    Algorithm:
        1. Ordena habilidades topologicamente (se respect_prerequisites=True)
        2. Inicializa tabela DP 3D: DP[i][t][c] = max valor
        3. Para cada habilidade i:
           Para cada tempo t:
               Para cada complexidade c:
                   Decide: pegar ou não pegar skill i
        4. Backtrack para reconstruir caminho ótimo
    
    Examples:
        >>> from src.graph_structures import build_graph_from_file
        >>> from src.config import SKILLS_DATASET_FILE
        >>> 
        >>> graph = build_graph_from_file(SKILLS_DATASET_FILE)
        >>> solution = dp_knapsack_2d(graph, max_time=350, max_complexity=30)
        >>> 
        >>> print(solution.path)
        >>> print(f"Valor: {solution.total_value}")
    """
    # Ordena habilidades topologicamente se necessário
    if respect_prerequisites:
        try:
            skill_order = graph.topological_sort()
        except ValueError:
            # Se há ciclo, usa ordem arbitrária
            skill_order = list(graph.nodes)
    else:
        skill_order = list(graph.nodes)
    
    n = len(skill_order)
    
    # Tabela DP: DP[i][t][c] = max valor usando skills[0:i] com tempo t e complexidade c
    # Inicializa com -infinito (impossível)
    DP = np.full((n + 1, max_time + 1, max_complexity + 1), -np.inf, dtype=float)
    
    # Caso base: sem skills, valor = 0
    DP[0, :, :] = 0
    
    # Tabela para backtracking: guarda se pegou ou não a skill i
    taken = np.zeros((n + 1, max_time + 1, max_complexity + 1), dtype=bool)
    
    # Mapa de índices: skill_id -> índice em skill_order
    skill_index = {skill_order[i]: i for i in range(n)}
    
    # Preenche tabela DP
    for i in range(1, n + 1):
        skill_id = skill_order[i - 1]
        metadata = graph.get_metadata(skill_id)
        
        skill_time = metadata['tempo_horas']
        skill_value = metadata['valor']
        skill_complexity = metadata['complexidade']
        prereqs = metadata.get('pre_requisitos', [])
        
        for t in range(max_time + 1):
            for c in range(max_complexity + 1):
                # Opção 1: NÃO pegar skill i
                DP[i, t, c] = DP[i - 1, t, c]
                
                # Opção 2: PEGAR skill i (se couber E pré-reqs satisfeitos)
                can_take = True
                
                # Verifica pré-requisitos (devem ter índice menor que i)
                if respect_prerequisites and prereqs:
                    for prereq in prereqs:
                        if prereq in skill_index:
                            prereq_idx = skill_index[prereq]
                            if prereq_idx >= i - 1:  # Pré-req vem depois na ordem
                                can_take = False
                                break
                        else:
                            # Pré-req não existe no grafo
                            can_take = False
                            break
                
                if can_take and t >= skill_time and c >= skill_complexity:
                    value_if_taken = DP[i - 1, t - skill_time, c - skill_complexity] + skill_value
                    
                    # Se pegar é melhor, atualiza
                    if value_if_taken > DP[i, t, c]:
                        DP[i, t, c] = value_if_taken
                        taken[i, t, c] = True
    
    # Valor ótimo
    optimal_value = DP[n, max_time, max_complexity]
    
    # Backtracking para reconstruir caminho
    path = []
    t_remaining = max_time
    c_remaining = max_complexity
    
    for i in range(n, 0, -1):
        if taken[i, t_remaining, c_remaining]:
            skill_id = skill_order[i - 1]
            path.append(skill_id)
            
            metadata = graph.get_metadata(skill_id)
            t_remaining -= metadata['tempo_horas']
            c_remaining -= metadata['complexidade']
    
    # Inverte para ordem correta
    path = path[::-1]
    
    # Calcula totais reais (pode ser diferente se houve arredondamento)
    total_time = 0
    total_complexity = 0
    total_value = 0
    skill_details = []
    
    for skill_id in path:
        metadata = graph.get_metadata(skill_id)
        total_time += metadata['tempo_horas']
        total_complexity += metadata['complexidade']
        total_value += metadata['valor']
        
        skill_details.append({
            'skill_id': skill_id,
            'nome': metadata['nome'],
            'tempo': metadata['tempo_horas'],
            'valor': metadata['valor'],
            'complexidade': metadata['complexidade']
        })
    
    # Verifica se alcança o objetivo
    reaches_target = target_skill in path
    
    return KnapsackSolution(
        path=path,
        total_value=total_value,
        total_time=total_time,
        total_complexity=total_complexity,
        reaches_target=reaches_target,
        skill_details=skill_details
    )


def validate_solution(solution: KnapsackSolution, graph: SkillGraph) -> Dict[str, Any]:
    """
    Valida uma solução do knapsack.
    
    Verifica:
    - Pré-requisitos são satisfeitos
    - Não excede limites de tempo e complexidade
    - Cálculos de totais estão corretos
    
    Args:
        solution: Solução a validar
        graph: Grafo de habilidades
    
    Returns:
        Dict com resultado da validação
    """
    issues = []
    acquired = set()
    
    # Valida pré-requisitos
    for skill_id in solution.path:
        metadata = graph.get_metadata(skill_id)
        prereqs = metadata.get('pre_requisitos', [])
        
        for prereq in prereqs:
            if prereq not in acquired:
                issues.append(f"Pré-requisito {prereq} de {skill_id} não foi adquirido antes")
        
        acquired.add(skill_id)
    
    # Valida limites
    if solution.total_time > 350:
        issues.append(f"Tempo excede limite: {solution.total_time} > 350")
    
    if solution.total_complexity > 30:
        issues.append(f"Complexidade excede limite: {solution.total_complexity} > 30")
    
    # Recalcula totais para validar
    calc_time = sum(graph.get_metadata(sid)['tempo_horas'] for sid in solution.path)
    calc_complexity = sum(graph.get_metadata(sid)['complexidade'] for sid in solution.path)
    calc_value = sum(graph.get_metadata(sid)['valor'] for sid in solution.path)
    
    if abs(calc_time - solution.total_time) > 0.01:
        issues.append(f"Tempo calculado ({calc_time}) diferente do reportado ({solution.total_time})")
    
    if abs(calc_complexity - solution.total_complexity) > 0.01:
        issues.append(f"Complexidade calculada ({calc_complexity}) diferente da reportada ({solution.total_complexity})")
    
    if abs(calc_value - solution.total_value) > 0.01:
        issues.append(f"Valor calculado ({calc_value}) diferente do reportado ({solution.total_value})")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


@measure_performance
def solve_deterministic(
    graph: SkillGraph,
    max_time: int = 350,
    max_complexity: int = 30,
    target_skill: str = 'S6'
) -> Dict[str, Any]:
    """
    Resolve o problema de forma determinística (sem incerteza).
    
    Args:
        graph: Grafo de habilidades
        max_time: Tempo máximo (horas)
        max_complexity: Complexidade máxima
        target_skill: Habilidade objetivo
    
    Returns:
        Dict com solução determinística completa
    """
    solution = dp_knapsack_2d(graph, max_time, max_complexity, target_skill)
    validation = validate_solution(solution, graph)
    
    return {
        'solution': solution.to_dict(),
        'validation': validation,
        'algorithm': 'DP Knapsack 2D',
        'constraints': {
            'max_time': max_time,
            'max_complexity': max_complexity,
            'target_skill': target_skill
        }
    }


def dp_knapsack_wrapper(skills_dict: Dict[str, Dict[str, Any]], **kwargs) -> float:
    """
    Wrapper para usar com Monte Carlo.
    
    Args:
        skills_dict: Dicionário de habilidades (pode ter valores modificados)
        **kwargs: max_time, max_complexity, etc.
    
    Returns:
        float: Valor total da solução ótima
    """
    from src.graph_structures import build_graph_from_dict
    
    # Constrói grafo a partir do dicionário
    graph = build_graph_from_dict(skills_dict)
    
    # Resolve DP
    solution = dp_knapsack_2d(
        graph,
        max_time=kwargs.get('max_time', 350),
        max_complexity=kwargs.get('max_complexity', 30),
        target_skill=kwargs.get('target_skill', 'S6')
    )
    
    return solution.total_value


@measure_performance
def solve_stochastic(
    graph: SkillGraph,
    n_scenarios: int = 1000,
    uncertainty: float = 0.10,
    max_time: int = 350,
    max_complexity: int = 30,
    target_skill: str = 'S6',
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Resolve o problema com incerteza usando Monte Carlo.
    
    Simula V ~ Uniforme[V-10%, V+10%] em 1000 cenários.
    
    Args:
        graph: Grafo de habilidades
        n_scenarios: Número de cenários estocásticos
        uncertainty: Percentual de incerteza (0.10 = ±10%)
        max_time: Tempo máximo
        max_complexity: Complexidade máxima
        target_skill: Habilidade objetivo
        seed: Seed para reprodutibilidade
    
    Returns:
        Dict com resultados da simulação Monte Carlo
    """
    # Extrai dicionário de skills do grafo
    skills_dict = {}
    for skill_id in graph.nodes:
        metadata = graph.get_metadata(skill_id)
        skills_dict[skill_id] = metadata
    
    # Gera cenários estocásticos
    print(f"Gerando {n_scenarios} cenários com incerteza ±{uncertainty * 100}%...")
    scenarios = generate_scenarios(
        skills_dict,
        n_scenarios=n_scenarios,
        uncertainty=uncertainty,
        seed=seed
    )
    
    # Executa Monte Carlo
    print(f"Executando simulação Monte Carlo...")
    mc_result = run_monte_carlo(
        dp_knapsack_wrapper,
        scenarios,
        max_time=max_time,
        max_complexity=max_complexity,
        target_skill=target_skill
    )
    
    return {
        'monte_carlo_result': mc_result,
        'n_scenarios': n_scenarios,
        'uncertainty': uncertainty,
        'algorithm': 'DP Knapsack 2D + Monte Carlo',
        'constraints': {
            'max_time': max_time,
            'max_complexity': max_complexity,
            'target_skill': target_skill
        }
    }


@measure_performance
def solve_complete(
    graph: SkillGraph,
    max_time: int = 350,
    max_complexity: int = 30,
    target_skill: str = 'S6',
    n_scenarios: int = 1000,
    uncertainty: float = 0.10,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Resolve o problema COMPLETO: determinístico + estocástico + comparação.
    
    Esta é a função principal do Desafio 1.
    
    Args:
        graph: Grafo de habilidades
        max_time: Tempo máximo
        max_complexity: Complexidade máxima
        target_skill: Habilidade objetivo
        n_scenarios: Número de cenários Monte Carlo
        uncertainty: Incerteza percentual
        seed: Seed para reprodutibilidade
    
    Returns:
        Dict com TODOS os resultados do Desafio 1
    """
    print("\n" + "=" * 70)
    print("🎯 DESAFIO 1 - CAMINHO DE VALOR MÁXIMO")
    print("=" * 70)
    
    # 1. Solução Determinística
    print("\n📊 FASE 1: Solução Determinística (sem incerteza)")
    print("-" * 70)
    deterministic_result = solve_deterministic(graph, max_time, max_complexity, target_skill)
    # Remove path_formatted antes de criar o objeto
    solution_dict = deterministic_result['solution'].copy()
    solution_dict.pop('path_formatted', None)
    det_solution = KnapsackSolution(**solution_dict)
    
    print(f"\n✅ Solução determinística:")
    print(f"   Caminho: {' → '.join(det_solution.path)}")
    print(f"   Valor: {det_solution.total_value:.2f}")
    print(f"   Tempo: {det_solution.total_time}h / {max_time}h")
    print(f"   Complexidade: {det_solution.total_complexity} / {max_complexity}")
    print(f"   Alcança S6: {'✅ SIM' if det_solution.reaches_target else '❌ NÃO'}")
    
    # 2. Solução Estocástica
    print(f"\n📊 FASE 2: Solução Estocástica (Monte Carlo - {n_scenarios} cenários)")
    print("-" * 70)
    stochastic_result = solve_stochastic(
        graph, n_scenarios, uncertainty, max_time, max_complexity, target_skill, seed
    )
    mc_result = stochastic_result['monte_carlo_result']
    
    print(f"\n✅ Solução estocástica:")
    print(f"   E[Valor] = {mc_result.expected_value:.2f} ± {mc_result.std_deviation:.2f}")
    print(f"   Range: [{mc_result.min_value:.2f}, {mc_result.max_value:.2f}]")
    print(f"   Mediana: {mc_result.median:.2f}")
    print(f"   IC 95%: [{mc_result.confidence_interval_95[0]:.2f}, {mc_result.confidence_interval_95[1]:.2f}]")
    
    # 3. Comparação
    print(f"\n📊 FASE 3: Comparação Determinístico vs Estocástico")
    print("-" * 70)
    comparison = compare_deterministic_vs_stochastic(det_solution.total_value, mc_result)
    
    print(f"\n📈 Análise comparativa:")
    print(f"   Determinístico: {comparison['deterministic_value']:.2f}")
    print(f"   Estocástico (E[X]): {comparison['stochastic_expected']:.2f} ± {comparison['stochastic_std']:.2f}")
    print(f"   Diferença: {comparison['difference']:.2f}")
    print(f"   Erro relativo: {comparison['relative_error_percent']:.2f}%")
    print(f"   Determinístico dentro IC 95%: {'✅ SIM' if comparison['deterministic_within_95ci'] else '❌ NÃO'}")
    
    print(f"\n💬 {comparison['interpretation']}")
    
    # Resultado completo
    return {
        'deterministic': deterministic_result,
        'stochastic': stochastic_result,
        'comparison': comparison,
        'summary': {
            'deterministic_value': det_solution.total_value,
            'deterministic_path': det_solution.path,
            'stochastic_expected': mc_result.expected_value,
            'stochastic_std': mc_result.std_deviation,
            'reaches_target': det_solution.reaches_target
        }
    }


def print_solution_details(solution: KnapsackSolution) -> None:
    """
    Imprime detalhes completos de uma solução.
    
    Args:
        solution: Solução a imprimir
    """
    print("\n" + "=" * 70)
    print("📋 DETALHES DA SOLUÇÃO")
    print("=" * 70)
    
    print(f"\n🎯 Caminho: {' → '.join(solution.path)}")
    print(f"\n📊 Totais:")
    print(f"   • Valor: {solution.total_value:.2f}")
    print(f"   • Tempo: {solution.total_time}h / 350h ({solution.total_time/350*100:.1f}%)")
    print(f"   • Complexidade: {solution.total_complexity} / 30 ({solution.total_complexity/30*100:.1f}%)")
    print(f"   • Alcança S6: {'✅ SIM' if solution.reaches_target else '❌ NÃO'}")
    
    print(f"\n📝 Habilidades no caminho:")
    for i, detail in enumerate(solution.skill_details, 1):
        print(f"   {i}. {detail['skill_id']} - {detail['nome']}")
        print(f"      Tempo: {detail['tempo']}h | Valor: {detail['valor']} | Complexidade: {detail['complexidade']}")
    
    print("=" * 70)