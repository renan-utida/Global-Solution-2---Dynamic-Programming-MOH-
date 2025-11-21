"""
Desafio 5 - Recomendação de Próximas Habilidades

Objetivo: Dado um perfil atual e horizonte de 5 anos, sugerir as próximas 2-3
habilidades que maximizam o valor esperado, considerando transições de mercado.

Cenários de Mercado (próximos 5 anos):
    1. IA em Alta (35%): Boost em IA Generativa, ML, Big Data
    2. Cloud First (30%): Boost em Cloud, DevOps, APIs
    3. Data Driven (20%): Boost em Dados, SQL, BI, Big Data
    4. Segurança Crítica (15%): Boost em Segurança, Cloud, DevOps

Algoritmo:
    DP com Look-Ahead em horizonte finito (5 anos)
    
    Estado: (skills_adquiridas, ano_atual)
    Ação: Escolher próxima habilidade
    Transição: Probabilidades de cenário de mercado
    
    V(S, t) = max_{a} [ E[Valor(a) | cenários] + V(S ∪ {a}, t+1) ]
    
    Onde:
    - S = conjunto de skills já adquiridas
    - t = ano atual (0 a 5)
    - a = próxima skill a adquirir
    - E[Valor(a) | cenários] = valor esperado considerando probabilidades

Complexidade:
    O(n × 2^n × h) onde n = skills disponíveis, h = horizonte
    Para n=12, h=5: O(12 × 4096 × 5) ≈ O(245,760)
    
    Na prática, usamos aproximações:
    - Limitar profundidade de busca
    - Considerar apenas k próximas melhores skills
    - Usar heurística gulosa com look-ahead
"""

from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass
import numpy as np
from itertools import combinations

from src.graph_structures import SkillGraph, build_graph_from_file
from src.decorators import measure_performance
from src.config import MARKET_SCENARIOS


# Horizonte de planejamento
RECOMMENDATION_HORIZON_YEARS = 5

# Número de habilidades a recomendar
N_RECOMMENDATIONS = 3


@dataclass
class MarketScenario:
    """
    Cenário de mercado futuro.
    
    Attributes:
        name: Nome do cenário
        probability: Probabilidade de ocorrência (0-1)
        boosts: Multiplicadores de valor por skill {skill_id: multiplier}
        description: Descrição do cenário
    """
    name: str
    probability: float
    boosts: Dict[str, float]
    description: str
    
    def apply_boost(self, skill_id: str, base_value: float) -> float:
        """
        Aplica boost do cenário ao valor base.
        
        Args:
            skill_id: ID da habilidade
            base_value: Valor base
        
        Returns:
            float: Valor com boost aplicado
        """
        multiplier = self.boosts.get(skill_id, 1.0)
        return base_value * multiplier


@dataclass
class Recommendation:
    """
    Recomendação de habilidades.
    
    Attributes:
        skills_recommended: Lista de IDs das habilidades recomendadas
        expected_value: Valor esperado total
        expected_value_per_scenario: Valor esperado por cenário
        reasoning: Justificativa da recomendação
        details: Detalhes de cada habilidade recomendada
    """
    skills_recommended: List[str]
    expected_value: float
    expected_value_per_scenario: Dict[str, float]
    reasoning: str
    details: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'skills_recommended': self.skills_recommended,
            'expected_value': self.expected_value,
            'expected_value_per_scenario': self.expected_value_per_scenario,
            'reasoning': self.reasoning,
            'details': self.details
        }
    
    def __str__(self) -> str:
        """Representação string formatada."""
        skills_str = ' → '.join(self.skills_recommended)
        return (
            f"Recommendation:\n"
            f"  Skills: {skills_str}\n"
            f"  E[Valor] = {self.expected_value:.2f}\n"
            f"  {self.reasoning}"
        )


def load_market_scenarios() -> List[MarketScenario]:
    """
    Carrega cenários de mercado da configuração.
    
    Returns:
        List[MarketScenario]: Lista de cenários
    """
    scenarios = []
    
    for scenario_name, scenario_data in MARKET_SCENARIOS.items():
        scenario = MarketScenario(
            name=scenario_name,
            probability=scenario_data['prob'],
            boosts=scenario_data['boost'],
            description=f"Cenário: {scenario_name.replace('_', ' ').title()}"
        )
        scenarios.append(scenario)
    
    return scenarios


def calculate_expected_value(
    skill_id: str,
    base_value: float,
    scenarios: List[MarketScenario]
) -> Tuple[float, Dict[str, float]]:
    """
    Calcula valor esperado de uma habilidade considerando cenários de mercado.
    
    E[Valor] = Σ P(cenário) × Valor(skill | cenário)
    
    Args:
        skill_id: ID da habilidade
        base_value: Valor base da habilidade
        scenarios: Lista de cenários de mercado
    
    Returns:
        Tuple[float, Dict]: (valor_esperado, valores_por_cenario)
    
    Examples:
        >>> expected, per_scenario = calculate_expected_value('S6', 10, scenarios)
        >>> # Se S6 tem boost em "ia_em_alta" (35%, +25%):
        >>> # E[V] = 0.35 × 12.5 + 0.30 × 10 + 0.20 × 10 + 0.15 × 10 = 10.375
    """
    values_per_scenario = {}
    expected_value = 0.0
    
    for scenario in scenarios:
        # Aplica boost do cenário
        boosted_value = scenario.apply_boost(skill_id, base_value)
        values_per_scenario[scenario.name] = boosted_value
        
        # Pondera pela probabilidade
        expected_value += scenario.probability * boosted_value
    
    return expected_value, values_per_scenario


def greedy_recommendation_with_lookahead(
    graph: SkillGraph,
    current_skills: Set[str],
    n_recommendations: int = N_RECOMMENDATIONS,
    scenarios: Optional[List[MarketScenario]] = None,
    lookahead_depth: int = 2
) -> Recommendation:
    """
    Recomenda habilidades usando algoritmo guloso com look-ahead.
    
    Algoritmo:
        1. Para cada skill disponível (não adquirida):
           a. Calcula valor esperado imediato (considerando cenários)
           b. Simula aquisição e calcula valor das próximas k skills
           c. Score = valor_imediato + α × valor_futuro
        2. Seleciona as top n_recommendations skills com maior score
    
    Args:
        graph: Grafo de habilidades
        current_skills: Conjunto de skills já adquiridas
        n_recommendations: Número de skills a recomendar
        scenarios: Lista de cenários de mercado
        lookahead_depth: Profundidade do look-ahead (1-3)
    
    Returns:
        Recommendation: Recomendação completa
    
    Complexity:
        O(n × k × m) onde:
        - n = skills disponíveis
        - k = lookahead_depth
        - m = número de cenários
    """
    if scenarios is None:
        scenarios = load_market_scenarios()
    
    # Skills disponíveis (não adquiridas e com pré-reqs satisfeitos)
    available_skills = []
    for skill_id in graph.nodes:
        if skill_id in current_skills:
            continue
        
        # Verifica pré-requisitos
        prereqs = graph.get_prerequisites(skill_id)
        if all(prereq in current_skills for prereq in prereqs):
            available_skills.append(skill_id)
    
    if not available_skills:
        return Recommendation(
            skills_recommended=[],
            expected_value=0.0,
            expected_value_per_scenario={},
            reasoning="Nenhuma habilidade disponível (pré-requisitos não satisfeitos)",
            details=[]
        )
    
    # Calcula score para cada skill disponível
    skill_scores = []
    
    for skill_id in available_skills:
        metadata = graph.get_metadata(skill_id)
        base_value = metadata['valor']
        
        # Valor esperado imediato
        immediate_value, values_per_scenario = calculate_expected_value(
            skill_id, base_value, scenarios
        )
        
        # Look-ahead: simula aquisição e calcula valor futuro
        future_value = 0.0
        if lookahead_depth > 0:
            # Simula conjunto com skill adquirida
            temp_acquired = current_skills | {skill_id}
            
            # Calcula valor esperado das próximas skills acessíveis
            future_available = []
            for next_skill_id in graph.nodes:
                if next_skill_id in temp_acquired:
                    continue
                
                next_prereqs = graph.get_prerequisites(next_skill_id)
                if all(prereq in temp_acquired for prereq in next_prereqs):
                    next_metadata = graph.get_metadata(next_skill_id)
                    next_base_value = next_metadata['valor']
                    next_expected, _ = calculate_expected_value(
                        next_skill_id, next_base_value, scenarios
                    )
                    future_available.append(next_expected)
            
            # Valor futuro = média das top k skills futuras
            if future_available:
                future_available.sort(reverse=True)
                top_k = min(lookahead_depth, len(future_available))
                future_value = sum(future_available[:top_k]) / top_k
        
        # Score combinado: valor imediato + fator de desconto × valor futuro
        discount_factor = 0.3  # Peso do futuro vs presente
        total_score = immediate_value + discount_factor * future_value
        
        skill_scores.append({
            'skill_id': skill_id,
            'nome': metadata['nome'],
            'immediate_value': immediate_value,
            'future_value': future_value,
            'total_score': total_score,
            'values_per_scenario': values_per_scenario,
            'base_value': base_value,
            'tempo_horas': metadata['tempo_horas']
        })
    
    # Ordena por score total (decrescente)
    skill_scores.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Seleciona top n_recommendations
    top_recommendations = skill_scores[:n_recommendations]
    
    # Calcula valor esperado total
    total_expected_value = sum(rec['immediate_value'] for rec in top_recommendations)
    
    # Agrega valores por cenário
    aggregated_per_scenario = {scenario.name: 0.0 for scenario in scenarios}
    for rec in top_recommendations:
        for scenario_name, value in rec['values_per_scenario'].items():
            aggregated_per_scenario[scenario_name] += value
    
    # Gera justificativa
    reasoning = generate_reasoning(top_recommendations, scenarios)
    
    # Detalhes
    details = []
    for rec in top_recommendations:
        details.append({
            'skill_id': rec['skill_id'],
            'nome': rec['nome'],
            'immediate_value': rec['immediate_value'],
            'future_value': rec['future_value'],
            'total_score': rec['total_score'],
            'base_value': rec['base_value'],
            'tempo_horas': rec['tempo_horas'],
            'values_per_scenario': rec['values_per_scenario']
        })
    
    return Recommendation(
        skills_recommended=[rec['skill_id'] for rec in top_recommendations],
        expected_value=total_expected_value,
        expected_value_per_scenario=aggregated_per_scenario,
        reasoning=reasoning,
        details=details
    )


def generate_reasoning(
    recommendations: List[Dict[str, Any]],
    scenarios: List[MarketScenario]
) -> str:
    """
    Gera justificativa textual para as recomendações.
    
    Args:
        recommendations: Lista de recomendações
        scenarios: Lista de cenários
    
    Returns:
        str: Justificativa formatada
    """
    if not recommendations:
        return "Nenhuma recomendação disponível."
    
    lines = ["Recomendação baseada em:"]
    
    # Identifica cenário mais provável
    most_likely_scenario = max(scenarios, key=lambda s: s.probability)
    lines.append(f"• Cenário mais provável: {most_likely_scenario.name} ({most_likely_scenario.probability*100:.0f}%)")
    
    # Analisa recomendações
    skill_names = [rec['nome'] for rec in recommendations]
    lines.append(f"• Skills recomendadas alinham com tendências de mercado")
    
    # Identifica skills com maior boost
    best_boosts = []
    for rec in recommendations:
        for scenario in scenarios:
            if rec['skill_id'] in scenario.boosts:
                boost = scenario.boosts[rec['skill_id']]
                if boost > 1.1:  # Boost significativo (>10%)
                    best_boosts.append(
                        f"{rec['skill_id']} (+{(boost-1)*100:.0f}% em {scenario.name})"
                    )
    
    if best_boosts:
        lines.append(f"• Boosts identificados: {', '.join(best_boosts[:3])}")
    
    return '\n'.join(lines)


def dp_recommendation_exhaustive(
    graph: SkillGraph,
    current_skills: Set[str],
    n_recommendations: int = N_RECOMMENDATIONS,
    scenarios: Optional[List[MarketScenario]] = None,
    max_combinations: int = 1000
) -> Recommendation:
    """
    Recomenda habilidades usando DP exaustivo (força bruta otimizada).
    
    Enumera todas as combinações possíveis de n habilidades e escolhe
    a que maximiza o valor esperado.
    
    Args:
        graph: Grafo de habilidades
        current_skills: Conjunto de skills já adquiridas
        n_recommendations: Número de skills a recomendar
        scenarios: Lista de cenários
        max_combinations: Limite de combinações a avaliar
    
    Returns:
        Recommendation: Recomendação ótima
    
    Complexity:
        O(C(n, k) × m) onde:
        - C(n, k) = combinações de n skills tomadas k a k
        - m = número de cenários
        
        Para n=10, k=3: C(10, 3) = 120 combinações
    """
    if scenarios is None:
        scenarios = load_market_scenarios()
    
    # Skills disponíveis
    available_skills = []
    for skill_id in graph.nodes:
        if skill_id in current_skills:
            continue
        
        prereqs = graph.get_prerequisites(skill_id)
        if all(prereq in current_skills for prereq in prereqs):
            available_skills.append(skill_id)
    
    if len(available_skills) < n_recommendations:
        # Não há skills suficientes, usa guloso
        return greedy_recommendation_with_lookahead(
            graph, current_skills, n_recommendations, scenarios, lookahead_depth=1
        )
    
    # Enumera todas as combinações de n_recommendations skills
    best_combination = None
    best_expected_value = -float('inf')
    best_per_scenario = {}
    
    combinations_evaluated = 0
    
    for combination in combinations(available_skills, n_recommendations):
        if combinations_evaluated >= max_combinations:
            break
        
        combinations_evaluated += 1
        
        # Calcula valor esperado desta combinação
        total_expected = 0.0
        per_scenario = {scenario.name: 0.0 for scenario in scenarios}
        
        valid_combination = True
        for skill_id in combination:
            # Verifica se todos os skills na combinação têm pré-reqs satisfeitos
            # considerando os outros skills da combinação
            temp_acquired = current_skills | set(combination[:combination.index(skill_id)])
            prereqs = graph.get_prerequisites(skill_id)
            
            if not all(prereq in temp_acquired for prereq in prereqs):
                valid_combination = False
                break
            
            metadata = graph.get_metadata(skill_id)
            base_value = metadata['valor']
            expected, values_per_scenario = calculate_expected_value(
                skill_id, base_value, scenarios
            )
            
            total_expected += expected
            for scenario_name, value in values_per_scenario.items():
                per_scenario[scenario_name] += value
        
        if not valid_combination:
            continue
        
        # Atualiza melhor combinação
        if total_expected > best_expected_value:
            best_expected_value = total_expected
            best_combination = combination
            best_per_scenario = per_scenario
    
    if best_combination is None:
        # Fallback para guloso
        return greedy_recommendation_with_lookahead(
            graph, current_skills, n_recommendations, scenarios, lookahead_depth=1
        )
    
    # Constrói resultado
    details = []
    for skill_id in best_combination:
        metadata = graph.get_metadata(skill_id)
        base_value = metadata['valor']
        expected, values_per_scenario = calculate_expected_value(
            skill_id, base_value, scenarios
        )
        
        details.append({
            'skill_id': skill_id,
            'nome': metadata['nome'],
            'immediate_value': expected,
            'future_value': 0.0,
            'total_score': expected,
            'base_value': base_value,
            'tempo_horas': metadata['tempo_horas'],
            'values_per_scenario': values_per_scenario
        })
    
    reasoning = generate_reasoning(details, scenarios)
    
    return Recommendation(
        skills_recommended=list(best_combination),
        expected_value=best_expected_value,
        expected_value_per_scenario=best_per_scenario,
        reasoning=reasoning,
        details=details
    )


def compare_recommendation_methods(
    graph: SkillGraph,
    current_skills: Set[str],
    n_recommendations: int = N_RECOMMENDATIONS,
    scenarios: Optional[List[MarketScenario]] = None
) -> Dict[str, Any]:
    """
    Compara diferentes métodos de recomendação.
    
    Args:
        graph: Grafo de habilidades
        current_skills: Skills já adquiridas
        n_recommendations: Número de recomendações
        scenarios: Cenários de mercado
    
    Returns:
        Dict com comparação dos métodos
    """
    if scenarios is None:
        scenarios = load_market_scenarios()
    
    # Método 1: Guloso simples (sem look-ahead)
    greedy_simple = greedy_recommendation_with_lookahead(
        graph, current_skills, n_recommendations, scenarios, lookahead_depth=0
    )
    
    # Método 2: Guloso com look-ahead
    greedy_lookahead = greedy_recommendation_with_lookahead(
        graph, current_skills, n_recommendations, scenarios, lookahead_depth=2
    )
    
    # Método 3: DP exaustivo
    dp_exhaustive = dp_recommendation_exhaustive(
        graph, current_skills, n_recommendations, scenarios
    )
    
    return {
        'greedy_simple': greedy_simple.to_dict(),
        'greedy_lookahead': greedy_lookahead.to_dict(),
        'dp_exhaustive': dp_exhaustive.to_dict(),
        'comparison': {
            'greedy_simple_value': greedy_simple.expected_value,
            'greedy_lookahead_value': greedy_lookahead.expected_value,
            'dp_exhaustive_value': dp_exhaustive.expected_value,
            'best_method': max([
                ('greedy_simple', greedy_simple.expected_value),
                ('greedy_lookahead', greedy_lookahead.expected_value),
                ('dp_exhaustive', dp_exhaustive.expected_value)
            ], key=lambda x: x[1])[0]
        }
    }


@measure_performance
def solve_complete(
    graph: SkillGraph,
    current_skills: Set[str],
    n_recommendations: int = N_RECOMMENDATIONS,
    horizon_years: int = RECOMMENDATION_HORIZON_YEARS
) -> Dict[str, Any]:
    """
    Resolve o Desafio 5 COMPLETO:
    1. Carrega cenários de mercado
    2. Calcula recomendações com look-ahead
    3. Compara diferentes métodos
    4. Analisa impacto de cada cenário
    
    Esta é a função principal do Desafio 5.
    
    Args:
        graph: Grafo de habilidades
        current_skills: Skills já adquiridas pelo profissional
        n_recommendations: Número de skills a recomendar (2-3)
        horizon_years: Horizonte de planejamento (anos)
    
    Returns:
        Dict com TODOS os resultados do Desafio 5
    """
    print("\n" + "=" * 70)
    print("🎯 DESAFIO 5 - RECOMENDAÇÃO DE PRÓXIMAS HABILIDADES")
    print("=" * 70)
    
    print(f"\nObjetivo: Recomendar {n_recommendations} habilidades")
    print(f"Horizonte: {horizon_years} anos")
    print(f"Skills atuais: {', '.join(sorted(current_skills)) if current_skills else 'Nenhuma'}")
    
    # 1. Carrega cenários de mercado
    print("\n📊 FASE 1: Cenários de Mercado")
    print("-" * 70)
    
    scenarios = load_market_scenarios()
    
    print(f"\n📈 {len(scenarios)} cenários carregados:")
    for scenario in scenarios:
        print(f"   • {scenario.name}: {scenario.probability*100:.0f}% de probabilidade")
        boost_skills = [f"{sid}(+{(mult-1)*100:.0f}%)" 
                       for sid, mult in scenario.boosts.items() if mult > 1.0]
        if boost_skills:
            print(f"     Boosts: {', '.join(boost_skills[:3])}")
    
    # 2. Recomendação com look-ahead
    print("\n📊 FASE 2: Recomendação com Look-Ahead")
    print("-" * 70)
    
    recommendation = greedy_recommendation_with_lookahead(
        graph, current_skills, n_recommendations, scenarios, lookahead_depth=2
    )
    
    print(f"\n✅ Recomendação:")
    print(f"   Skills: {' → '.join(recommendation.skills_recommended)}")
    print(f"   E[Valor] = {recommendation.expected_value:.2f}")
    
    print(f"\n📋 Detalhes:")
    for i, detail in enumerate(recommendation.details, 1):
        print(f"\n   {i}. {detail['skill_id']} - {detail['nome']}")
        print(f"      Valor base: {detail['base_value']}")
        print(f"      E[Valor]: {detail['immediate_value']:.2f}")
        print(f"      Tempo: {detail['tempo_horas']}h")
        
        # Mostra valores por cenário
        best_scenario = max(detail['values_per_scenario'].items(), key=lambda x: x[1])
        print(f"      Melhor em: {best_scenario[0]} (V={best_scenario[1]:.2f})")
    
    print(f"\n💡 {recommendation.reasoning}")
    
    # 3. Comparação de métodos
    print("\n📊 FASE 3: Comparação de Métodos")
    print("-" * 70)
    
    comparison = compare_recommendation_methods(
        graph, current_skills, n_recommendations, scenarios
    )
    
    print(f"\n📈 Comparação:")
    print(f"   Guloso simples:     E[V] = {comparison['comparison']['greedy_simple_value']:.2f}")
    print(f"   Guloso + lookahead: E[V] = {comparison['comparison']['greedy_lookahead_value']:.2f}")
    print(f"   DP exaustivo:       E[V] = {comparison['comparison']['dp_exhaustive_value']:.2f}")
    print(f"   Melhor método: {comparison['comparison']['best_method']}")
    
    # 4. Análise por cenário
    print("\n📊 FASE 4: Análise por Cenário de Mercado")
    print("-" * 70)
    
    print(f"\n💰 Valor esperado por cenário:")
    for scenario_name, value in recommendation.expected_value_per_scenario.items():
        scenario_obj = next(s for s in scenarios if s.name == scenario_name)
        print(f"   • {scenario_name}: {value:.2f} (prob={scenario_obj.probability*100:.0f}%)")
    
    return {
        'recommendation': recommendation.to_dict(),
        'scenarios': [
            {
                'name': s.name,
                'probability': s.probability,
                'boosts': s.boosts
            } for s in scenarios
        ],
        'comparison': comparison,
        'horizon_years': horizon_years,
        'n_recommendations': n_recommendations,
        'current_skills': list(current_skills)
    }


def print_recommendation_details(recommendation: Recommendation) -> None:
    """
    Imprime detalhes completos de uma recomendação.
    
    Args:
        recommendation: Recomendação a imprimir
    """
    print("\n" + "=" * 70)
    print("📋 DETALHES DA RECOMENDAÇÃO")
    print("=" * 70)
    
    print(f"\n🎯 Habilidades recomendadas: {' → '.join(recommendation.skills_recommended)}")
    print(f"\n📊 Valor esperado total: {recommendation.expected_value:.2f}")
    
    print(f"\n💡 Justificativa:")
    for line in recommendation.reasoning.split('\n'):
        print(f"   {line}")
    
    print(f"\n📝 Detalhamento:")
    for i, detail in enumerate(recommendation.details, 1):
        print(f"\n   {i}. {detail['skill_id']} - {detail['nome']}")
        print(f"      • Valor base: {detail['base_value']}")
        print(f"      • Valor esperado: {detail['immediate_value']:.2f}")
        print(f"      • Valor futuro: {detail['future_value']:.2f}")
        print(f"      • Score total: {detail['total_score']:.2f}")
        print(f"      • Tempo: {detail['tempo_horas']}h")
    
    print("=" * 70)