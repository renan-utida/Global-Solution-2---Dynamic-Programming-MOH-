"""
Desafio 2 - Verificação Crítica

Enumera todas as 120 permutações das 5 Habilidades Críticas (S3, S5, S7, S8, S9)
e calcula o custo total de cada ordem, considerando:
- Tempo de aquisição de cada habilidade
- Tempo de espera para pré-requisitos serem satisfeitos

CRÍTICO: Valida o grafo ANTES de calcular custos!

Habilidades Críticas:
    S3: Algoritmos Avançados (requer S1)
    S5: Visualização de Dados (requer S2)
    S7: Estruturas em Nuvem (sem pré-reqs)
    S8: APIs e Microsserviços (requer S1)
    S9: DevOps & CI/CD (requer S7, S8)

Permutações: 5! = 120

Complexidade:
    O(n!) × O(cálculo de custo) = O(120 × n × p)
    onde n = tamanho da ordem, p = pré-requisitos médios
"""

from typing import List, Dict, Tuple, Set, Any
from itertools import permutations
from dataclasses import dataclass
import numpy as np

from src.graph_structures import SkillGraph, build_graph_from_file
from src.graph_validation import validate_graph, ensure_valid_graph
from src.decorators import measure_performance


# Habilidades Críticas
CRITICAL_SKILLS = ['S3', 'S5', 'S7', 'S8', 'S9']


@dataclass
class OrderCost:
    """
    Custo de uma ordem específica de habilidades.
    
    Attributes:
        order: Lista de IDs na ordem de aquisição
        total_cost: Custo total (tempo + esperas)
        acquisition_time: Tempo de aquisição das habilidades
        waiting_time: Tempo de espera por pré-requisitos
        details: Detalhes de cada habilidade na ordem
    """
    order: List[str]
    total_cost: float
    acquisition_time: float
    waiting_time: float
    details: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'order': self.order,
            'order_formatted': ' → '.join(self.order),
            'total_cost': self.total_cost,
            'acquisition_time': self.acquisition_time,
            'waiting_time': self.waiting_time,
            'details': self.details
        }
    
    def __str__(self) -> str:
        """Representação string formatada."""
        return (
            f"Order: {' → '.join(self.order)}\n"
            f"  Total Cost: {self.total_cost:.0f}h\n"
            f"  Acquisition: {self.acquisition_time:.0f}h\n"
            f"  Waiting: {self.waiting_time:.0f}h"
        )


def calculate_order_cost(order: List[str], graph: SkillGraph) -> OrderCost:
    """
    Calcula o custo total de uma ordem de aquisição de habilidades.
    
    Custo = Tempo de aquisição + Tempo de espera por pré-requisitos
    
    Algoritmo:
        1. Começa com conjunto vazio de habilidades adquiridas
        2. Para cada habilidade na ordem:
           a. Verifica quais pré-requisitos ainda não foram adquiridos
           b. Adiciona tempo de espera (soma dos tempos dos pré-reqs faltantes)
           c. Adiciona tempo de aquisição da habilidade
           d. Marca habilidade como adquirida
        3. Retorna custo total
    
    Args:
        order: Lista de IDs das habilidades na ordem de aquisição
        graph: Grafo de habilidades com metadados
    
    Returns:
        OrderCost: Objeto com custo total e detalhes
    
    Complexity:
        O(n × p) onde n = tamanho da ordem, p = pré-requisitos médios
    
    Examples:
        >>> order = ['S7', 'S3', 'S8', 'S5', 'S9']
        >>> cost = calculate_order_cost(order, graph)
        >>> 
        >>> # S7: 70h (sem pré-reqs)
        >>> # S3: 100h + espera por S1 (80h) = 180h
        >>> # S8: 90h (S1 já satisfeito por S3)
        >>> # S5: 40h + espera por S2 (60h) = 100h
        >>> # S9: 110h (S7 e S8 já satisfeitos)
        >>> # Total: 70 + 180 + 90 + 100 + 110 = 550h
    """
    acquired = set()  # Habilidades já adquiridas
    total_cost = 0
    acquisition_time = 0
    waiting_time = 0
    details = []
    
    for skill_id in order:
        metadata = graph.get_metadata(skill_id)
        skill_time = metadata['tempo_horas']
        prereqs = metadata.get('pre_requisitos', [])
        
        # Verifica quais pré-requisitos ainda não foram adquiridos
        missing_prereqs = [p for p in prereqs if p not in acquired]
        
        # Calcula tempo de espera (soma dos tempos dos pré-reqs faltantes)
        wait_time = 0
        for prereq in missing_prereqs:
            if prereq in graph:
                prereq_metadata = graph.get_metadata(prereq)
                wait_time += prereq_metadata['tempo_horas']
        
        # Custo desta habilidade = tempo de aquisição + tempo de espera
        skill_cost = skill_time + wait_time
        
        # Atualiza totais
        total_cost += skill_cost
        acquisition_time += skill_time
        waiting_time += wait_time
        
        # Adiciona detalhes
        details.append({
            'skill_id': skill_id,
            'nome': metadata['nome'],
            'acquisition_time': skill_time,
            'waiting_time': wait_time,
            'missing_prereqs': missing_prereqs,
            'cost': skill_cost,
            'cumulative_cost': total_cost
        })
        
        # Marca como adquirida
        acquired.add(skill_id)
    
    return OrderCost(
        order=order,
        total_cost=total_cost,
        acquisition_time=acquisition_time,
        waiting_time=waiting_time,
        details=details
    )


def generate_all_permutations(skills: List[str]) -> List[List[str]]:
    """
    Gera todas as permutações possíveis de uma lista de habilidades.
    
    Args:
        skills: Lista de IDs das habilidades
    
    Returns:
        List[List[str]]: Lista de todas as permutações
    
    Complexity:
        O(n!) onde n = número de habilidades
        Para 5 habilidades: 5! = 120 permutações
    
    Examples:
        >>> skills = ['S3', 'S5', 'S7']
        >>> perms = generate_all_permutations(skills)
        >>> len(perms)
        6  # 3! = 6
    """
    return [list(perm) for perm in permutations(skills)]


@measure_performance
def validate_before_compute(graph: SkillGraph) -> Dict[str, Any]:
    """
    Valida o grafo ANTES de calcular permutações.
    
    Esta função é CRÍTICA e obrigatória pelo enunciado!
    
    Args:
        graph: Grafo de habilidades
    
    Returns:
        Dict com resultado da validação
    
    Raises:
        ValueError: Se o grafo é inválido (ciclos ou órfãos)
    
    Examples:
        >>> graph = build_graph_from_file(SKILLS_DATASET_FILE)
        >>> try:
        ...     validation = validate_before_compute(graph)
        ...     print("✅ Grafo válido, pode prosseguir")
        ... except ValueError as e:
        ...     print(f"❌ ERRO: {e}")
        ...     # NÃO prosseguir com cálculos!
    """
    print("\n🔍 VALIDAÇÃO CRÍTICA DO GRAFO (Obrigatória)")
    print("-" * 70)
    
    # Usa função de validação do módulo graph_validation
    validation_result = validate_graph(graph)
    
    if validation_result['valid']:
        print("✅ Grafo válido! Pode prosseguir com cálculos de permutações.")
    else:
        print("❌ GRAFO INVÁLIDO! NÃO pode prosseguir!")
        print(f"\nErro: {validation_result['error_msg']}")
        
        if validation_result['cycles']:
            print(f"\n🔴 Ciclos detectados: {len(validation_result['cycles'])}")
            for cycle in validation_result['cycles']:
                print(f"   {' → '.join(cycle)}")
        
        if validation_result['orphans']:
            print(f"\n🔴 Nós órfãos detectados: {len(validation_result['orphans'])}")
            print(f"   {validation_result['orphans']}")
        
        # Lança exceção para interromper execução
        ensure_valid_graph(graph)
    
    return validation_result


def calculate_all_permutations_costs(
    graph: SkillGraph,
    skills: List[str] = CRITICAL_SKILLS
) -> List[OrderCost]:
    """
    Calcula o custo de todas as permutações possíveis.
    
    Args:
        graph: Grafo de habilidades
        skills: Lista de habilidades críticas (padrão: S3, S5, S7, S8, S9)
    
    Returns:
        List[OrderCost]: Lista de custos para cada permutação
    
    Complexity:
        O(n! × n × p) onde n = número de skills, p = pré-requisitos médios
        Para 5 skills: O(120 × 5 × 2) = O(1,200)
    """
    print(f"\n📊 Calculando custos de {len(skills)}! permutações...")
    print(f"   Habilidades: {', '.join(skills)}")
    
    # Gera todas as permutações
    all_perms = generate_all_permutations(skills)
    print(f"   Total de permutações: {len(all_perms)}")
    
    # Calcula custo de cada permutação
    all_costs = []
    for perm in all_perms:
        cost = calculate_order_cost(perm, graph)
        all_costs.append(cost)
    
    # Ordena por custo (menor para maior)
    all_costs.sort(key=lambda x: x.total_cost)
    
    print(f"✅ Custos calculados para {len(all_costs)} permutações")
    
    return all_costs


def find_top_n_orders(
    all_costs: List[OrderCost],
    n: int = 3,
    best: bool = True
) -> List[OrderCost]:
    """
    Encontra as N melhores (ou piores) ordens.
    
    Args:
        all_costs: Lista de custos (já ordenada)
        n: Número de ordens a retornar
        best: Se True, retorna as melhores (menor custo); se False, as piores
    
    Returns:
        List[OrderCost]: Top N ordens
    
    Examples:
        >>> top_3 = find_top_n_orders(all_costs, n=3, best=True)
        >>> worst_3 = find_top_n_orders(all_costs, n=3, best=False)
    """
    if best:
        return all_costs[:n]
    else:
        return all_costs[-n:][::-1]


def calculate_average_cost(costs: List[OrderCost]) -> float:
    """
    Calcula o custo médio de uma lista de ordens.
    
    Args:
        costs: Lista de custos
    
    Returns:
        float: Custo médio
    """
    return np.mean([c.total_cost for c in costs])


def analyze_heuristics(
    all_costs: List[OrderCost],
    graph: SkillGraph
) -> Dict[str, Any]:
    """
    Analisa heurísticas observadas nas melhores ordens.
    
    Investiga:
    1. Habilidades sem pré-requisitos aparecem primeiro?
    2. Habilidades com muitos dependentes aparecem cedo?
    3. Qual é o padrão comum nas melhores ordens?
    
    Args:
        all_costs: Lista de custos de todas as permutações
        graph: Grafo de habilidades
    
    Returns:
        Dict com análises heurísticas
    
    Examples:
        >>> heuristics = analyze_heuristics(all_costs, graph)
        >>> print(heuristics['patterns'])
    """
    top_10 = all_costs[:10]
    
    # Análise 1: Primeira posição
    first_positions = [cost.order[0] for cost in top_10]
    first_position_freq = {skill: first_positions.count(skill) for skill in set(first_positions)}
    
    # Análise 2: Habilidades sem pré-requisitos
    no_prereqs = []
    for skill in CRITICAL_SKILLS:
        metadata = graph.get_metadata(skill)
        prereqs = metadata.get('pre_requisitos', [])
        if len(prereqs) == 0:
            no_prereqs.append(skill)
    
    # Análise 3: Posição média de cada habilidade nas top 10
    position_stats = {skill: [] for skill in CRITICAL_SKILLS}
    for cost in top_10:
        for pos, skill in enumerate(cost.order):
            position_stats[skill].append(pos)
    
    avg_positions = {skill: np.mean(positions) for skill, positions in position_stats.items()}
    
    # Análise 4: Pré-requisitos
    prereqs_info = {}
    for skill in CRITICAL_SKILLS:
        metadata = graph.get_metadata(skill)
        prereqs = metadata.get('pre_requisitos', [])
        prereqs_info[skill] = {
            'prereqs': prereqs,
            'num_prereqs': len(prereqs),
            'avg_position': avg_positions[skill]
        }
    
    # Padrões identificados
    patterns = []
    
    # Padrão 1: Habilidades sem pré-reqs aparecem primeiro
    if no_prereqs:
        early_skills = [s for s, pos in avg_positions.items() if pos < 2]
        if any(s in no_prereqs for s in early_skills):
            patterns.append(
                f"Habilidades sem pré-requisitos ({', '.join(no_prereqs)}) "
                f"tendem a aparecer nas primeiras posições"
            )
    
    # Padrão 2: Habilidades com muitos pré-reqs aparecem por último
    many_prereqs = [s for s, info in prereqs_info.items() if info['num_prereqs'] >= 2]
    if many_prereqs:
        late_skills = [s for s, pos in avg_positions.items() if pos > 3]
        if any(s in many_prereqs for s in late_skills):
            patterns.append(
                f"Habilidades com múltiplos pré-requisitos ({', '.join(many_prereqs)}) "
                f"tendem a aparecer nas últimas posições"
            )
    
    # Padrão 3: Ordem mais comum
    most_common_first = max(first_position_freq, key=first_position_freq.get)
    patterns.append(
        f"Habilidade mais comum na primeira posição: {most_common_first} "
        f"({first_position_freq[most_common_first]}/{len(top_10)} vezes)"
    )
    
    return {
        'first_position_frequency': first_position_freq,
        'no_prereq_skills': no_prereqs,
        'avg_positions': avg_positions,
        'prereqs_info': prereqs_info,
        'patterns': patterns,
        'recommendation': (
            "Recomendação: Priorize habilidades sem pré-requisitos nas primeiras posições "
            "para minimizar tempo de espera acumulado."
        )
    }


@measure_performance
def solve_complete(
    graph: SkillGraph,
    skills: List[str] = CRITICAL_SKILLS
) -> Dict[str, Any]:
    """
    Resolve o Desafio 2 COMPLETO:
    1. Valida grafo
    2. Gera todas as permutações
    3. Calcula custos
    4. Encontra top 3 melhores e piores
    5. Analisa heurísticas
    
    Esta é a função principal do Desafio 2.
    
    Args:
        graph: Grafo de habilidades
        skills: Lista de habilidades críticas
    
    Returns:
        Dict com TODOS os resultados do Desafio 2
    """
    print("\n" + "=" * 70)
    print("🎯 DESAFIO 2 - VERIFICAÇÃO CRÍTICA")
    print("=" * 70)
    
    # 1. VALIDAÇÃO CRÍTICA (obrigatória)
    try:
        validation = validate_before_compute(graph)
    except ValueError as e:
        print(f"\n❌ ERRO CRÍTICO: Validação falhou!")
        print(f"   {e}")
        print("\n🚫 INTERROMPENDO EXECUÇÃO (conforme enunciado)")
        raise
    
    # 2. Calcula custos de todas as permutações
    print("\n📊 FASE 1: Cálculo de Custos")
    print("-" * 70)
    all_costs = calculate_all_permutations_costs(graph, skills)
    
    # 3. Encontra top 3 melhores e piores
    print("\n📊 FASE 2: Identificação das Melhores e Piores Ordens")
    print("-" * 70)
    top_3_best = find_top_n_orders(all_costs, n=3, best=True)
    top_3_worst = find_top_n_orders(all_costs, n=3, best=False)
    
    print(f"\n✅ Top 3 MELHORES ordens:")
    for i, cost in enumerate(top_3_best, 1):
        print(f"   {i}. {' → '.join(cost.order)}")
        print(f"      Custo: {cost.total_cost:.0f}h (Aquisição: {cost.acquisition_time:.0f}h, Espera: {cost.waiting_time:.0f}h)")
    
    print(f"\n❌ Top 3 PIORES ordens:")
    for i, cost in enumerate(top_3_worst, 1):
        print(f"   {i}. {' → '.join(cost.order)}")
        print(f"      Custo: {cost.total_cost:.0f}h (Aquisição: {cost.acquisition_time:.0f}h, Espera: {cost.waiting_time:.0f}h)")
    
    # 4. Calcula custo médio das top 3
    avg_top_3 = calculate_average_cost(top_3_best)
    avg_all = calculate_average_cost(all_costs)
    
    print(f"\n📊 Estatísticas:")
    print(f"   Custo médio (todas): {avg_all:.0f}h")
    print(f"   Custo médio (top 3): {avg_top_3:.0f}h")
    print(f"   Economia: {avg_all - avg_top_3:.0f}h ({(avg_all - avg_top_3) / avg_all * 100:.1f}%)")
    
    # 5. Análise de heurísticas
    print("\n📊 FASE 3: Análise de Heurísticas")
    print("-" * 70)
    heuristics = analyze_heuristics(all_costs, graph)
    
    print(f"\n🔍 Padrões identificados:")
    for i, pattern in enumerate(heuristics['patterns'], 1):
        print(f"   {i}. {pattern}")
    
    print(f"\n💡 {heuristics['recommendation']}")
    
    return {
        'validation': validation,
        'all_costs': all_costs,
        'top_3_best': [c.to_dict() for c in top_3_best],
        'top_3_worst': [c.to_dict() for c in top_3_worst],
        'statistics': {
            'avg_all': avg_all,
            'avg_top_3': avg_top_3,
            'best_cost': top_3_best[0].total_cost,
            'worst_cost': top_3_worst[0].total_cost,
            'cost_range': top_3_worst[0].total_cost - top_3_best[0].total_cost
        },
        'heuristics': heuristics
    }


def print_order_details(cost: OrderCost) -> None:
    """
    Imprime detalhes completos de uma ordem.
    
    Args:
        cost: Custo da ordem
    """
    print("\n" + "=" * 70)
    print("📋 DETALHES DA ORDEM")
    print("=" * 70)
    
    print(f"\n🎯 Ordem: {' → '.join(cost.order)}")
    print(f"\n📊 Resumo:")
    print(f"   • Custo total: {cost.total_cost:.0f}h")
    print(f"   • Tempo de aquisição: {cost.acquisition_time:.0f}h")
    print(f"   • Tempo de espera: {cost.waiting_time:.0f}h")
    
    print(f"\n📝 Detalhamento por habilidade:")
    for i, detail in enumerate(cost.details, 1):
        print(f"\n   {i}. {detail['skill_id']} - {detail['nome']}")
        print(f"      Tempo de aquisição: {detail['acquisition_time']}h")
        if detail['missing_prereqs']:
            print(f"      Pré-requisitos faltantes: {', '.join(detail['missing_prereqs'])}")
            print(f"      Tempo de espera: {detail['waiting_time']}h")
        else:
            print(f"      ✅ Sem pré-requisitos faltantes")
        print(f"      Custo desta habilidade: {detail['cost']}h")
        print(f"      Custo acumulado: {detail['cumulative_cost']}h")
    
    print("=" * 70)