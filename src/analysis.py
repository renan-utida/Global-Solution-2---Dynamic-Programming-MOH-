"""
Análises Estatísticas para o Motor de Orientação de Habilidades (MOH)

Este módulo fornece funções para análise comparativa dos resultados dos 5 desafios:
1. Comparação determinístico vs estocástico (Desafio 1)
2. Análise de custos de permutações (Desafio 2)
3. Comparação guloso vs ótimo (Desafio 3)
4. Análise de complexidade de algoritmos (Desafio 4)
5. Análise de cenários de mercado (Desafio 5)

Funções principais:
- compare_all_algorithms: Compara todos os algoritmos usados
- analyze_value_distributions: Analisa distribuições de valores
- calculate_metrics: Calcula métricas agregadas
- generate_summary_report: Gera relatório consolidado
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class ComparisonMetrics:
    """
    Métricas de comparação entre dois métodos.
    
    Attributes:
        method_a_name: Nome do método A
        method_b_name: Nome do método B
        mean_a: Média do método A
        mean_b: Média do método B
        std_a: Desvio padrão do método A
        std_b: Desvio padrão do método B
        difference: Diferença absoluta (A - B)
        relative_difference: Diferença relativa (%)
        p_value: P-valor do teste estatístico
        is_significant: Se a diferença é estatisticamente significativa
    """
    method_a_name: str
    method_b_name: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    relative_difference: float
    p_value: float
    is_significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'method_a': self.method_a_name,
            'method_b': self.method_b_name,
            'mean_a': self.mean_a,
            'mean_b': self.mean_b,
            'std_a': self.std_a,
            'std_b': self.std_b,
            'difference': self.difference,
            'relative_difference': self.relative_difference,
            'p_value': self.p_value,
            'is_significant': self.is_significant
        }


def compare_deterministic_vs_stochastic(
    deterministic_value: float,
    stochastic_values: List[float]
) -> Dict[str, Any]:
    """
    Compara solução determinística com distribuição estocástica.
    
    Args:
        deterministic_value: Valor da solução determinística
        stochastic_values: Lista de valores da simulação Monte Carlo
    
    Returns:
        Dict com métricas de comparação
    """
    stochastic_array = np.array(stochastic_values)
    
    # Estatísticas da distribuição estocástica
    mean_stochastic = np.mean(stochastic_array)
    std_stochastic = np.std(stochastic_array)
    min_stochastic = np.min(stochastic_array)
    max_stochastic = np.max(stochastic_array)
    median_stochastic = np.median(stochastic_array)
    
    # Diferenças
    difference = deterministic_value - mean_stochastic
    relative_diff = (difference / mean_stochastic * 100) if mean_stochastic != 0 else 0
    
    # Intervalo de confiança 95%
    ci_95 = stats.t.interval(
        0.95,
        len(stochastic_array) - 1,
        loc=mean_stochastic,
        scale=stats.sem(stochastic_array)
    )
    
    # Verifica se determinístico está dentro do IC
    within_ci = ci_95[0] <= deterministic_value <= ci_95[1]
    
    # Z-score: quantos desvios padrão o determinístico está da média
    z_score = (deterministic_value - mean_stochastic) / std_stochastic if std_stochastic > 0 else 0
    
    # Interpretação
    if abs(relative_diff) < 5:
        interpretation = "Métodos concordam muito bem (diferença < 5%)"
    elif abs(relative_diff) < 10:
        interpretation = "Diferença pequena e aceitável (diferença < 10%)"
    else:
        interpretation = "Diferença significativa entre os métodos"
    
    return {
        'deterministic_value': deterministic_value,
        'stochastic_mean': mean_stochastic,
        'stochastic_std': std_stochastic,
        'stochastic_min': min_stochastic,
        'stochastic_max': max_stochastic,
        'stochastic_median': median_stochastic,
        'difference': difference,
        'relative_difference_percent': relative_diff,
        'confidence_interval_95': ci_95,
        'within_ci_95': within_ci,
        'z_score': z_score,
        'interpretation': interpretation
    }


def analyze_permutations_costs(
    all_costs: List[float],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Analisa distribuição de custos das permutações (Desafio 2).
    
    Args:
        all_costs: Lista com custos de todas as permutações
        top_n: Número de melhores/piores a destacar
    
    Returns:
        Dict com análise estatística
    """
    costs_array = np.array(all_costs)
    
    # Estatísticas descritivas
    mean_cost = np.mean(costs_array)
    std_cost = np.std(costs_array)
    min_cost = np.min(costs_array)
    max_cost = np.max(costs_array)
    median_cost = np.median(costs_array)
    
    # Percentis
    percentiles = {
        'p10': np.percentile(costs_array, 10),
        'p25': np.percentile(costs_array, 25),
        'p50': np.percentile(costs_array, 50),
        'p75': np.percentile(costs_array, 75),
        'p90': np.percentile(costs_array, 90)
    }
    
    # Range e IQR
    cost_range = max_cost - min_cost
    iqr = percentiles['p75'] - percentiles['p25']
    
    # Coeficiente de variação
    cv = (std_cost / mean_cost * 100) if mean_cost > 0 else 0
    
    # Economia ao escolher melhor vs pior
    savings = max_cost - min_cost
    savings_percent = (savings / max_cost * 100) if max_cost > 0 else 0
    
    # Economia ao escolher melhor vs média
    savings_vs_avg = mean_cost - min_cost
    savings_vs_avg_percent = (savings_vs_avg / mean_cost * 100) if mean_cost > 0 else 0
    
    return {
        'n_permutations': len(all_costs),
        'mean': mean_cost,
        'std': std_cost,
        'min': min_cost,
        'max': max_cost,
        'median': median_cost,
        'range': cost_range,
        'iqr': iqr,
        'cv_percent': cv,
        'percentiles': percentiles,
        'savings': {
            'best_vs_worst': savings,
            'best_vs_worst_percent': savings_percent,
            'best_vs_avg': savings_vs_avg,
            'best_vs_avg_percent': savings_vs_avg_percent
        }
    }


def compare_greedy_vs_optimal(
    greedy_value: float,
    greedy_time: float,
    optimal_value: float,
    optimal_time: float
) -> Dict[str, Any]:
    """
    Compara algoritmo guloso vs ótimo (Desafio 3).
    
    Args:
        greedy_value: Valor da solução gulosa
        greedy_time: Tempo da solução gulosa
        optimal_value: Valor da solução ótima
        optimal_time: Tempo da solução ótima
    
    Returns:
        Dict com comparação
    """
    # Diferenças de valor
    value_diff = optimal_value - greedy_value
    value_diff_percent = (value_diff / optimal_value * 100) if optimal_value > 0 else 0
    
    # Diferenças de tempo
    time_diff = optimal_time - greedy_time
    time_diff_percent = (time_diff / optimal_time * 100) if optimal_time > 0 else 0
    
    # Eficiências (valor/tempo)
    greedy_efficiency = greedy_value / greedy_time if greedy_time > 0 else 0
    optimal_efficiency = optimal_value / optimal_time if optimal_time > 0 else 0
    
    # Qualidade da solução gulosa
    quality = (greedy_value / optimal_value * 100) if optimal_value > 0 else 0
    
    # Interpretação
    if value_diff == 0:
        interpretation = "Guloso encontrou a solução ótima!"
    elif quality >= 95:
        interpretation = f"Guloso é excelente (≥95% do ótimo, {quality:.1f}%)"
    elif quality >= 90:
        interpretation = f"Guloso é muito bom (≥90% do ótimo, {quality:.1f}%)"
    elif quality >= 80:
        interpretation = f"Guloso é bom (≥80% do ótimo, {quality:.1f}%)"
    else:
        interpretation = f"Guloso é subótimo ({quality:.1f}% do ótimo)"
    
    return {
        'greedy': {
            'value': greedy_value,
            'time': greedy_time,
            'efficiency': greedy_efficiency
        },
        'optimal': {
            'value': optimal_value,
            'time': optimal_time,
            'efficiency': optimal_efficiency
        },
        'differences': {
            'value': value_diff,
            'value_percent': value_diff_percent,
            'time': time_diff,
            'time_percent': time_diff_percent
        },
        'greedy_quality_percent': quality,
        'greedy_is_optimal': value_diff == 0,
        'interpretation': interpretation
    }


def analyze_sorting_performance(
    merge_sort_time: float,
    merge_sort_comparisons: int,
    native_sort_time: float,
    n_elements: int
) -> Dict[str, Any]:
    """
    Analisa performance do Merge Sort vs sort nativo (Desafio 4).
    
    Args:
        merge_sort_time: Tempo do Merge Sort (segundos)
        merge_sort_comparisons: Número de comparações do Merge Sort
        native_sort_time: Tempo do sort nativo (segundos)
        n_elements: Número de elementos ordenados
    
    Returns:
        Dict com análise de performance
    """
    # Razão de tempo
    time_ratio = merge_sort_time / native_sort_time if native_sort_time > 0 else float('inf')
    
    # Comparações teóricas esperadas: O(n log n)
    theoretical_comparisons = n_elements * np.log2(n_elements)
    
    # Comparação com teórico
    comparison_efficiency = (merge_sort_comparisons / theoretical_comparisons * 100) if theoretical_comparisons > 0 else 0
    
    # Interpretação
    if time_ratio < 2:
        time_interpretation = f"Merge Sort competitivo ({time_ratio:.2f}x mais lento)"
    elif time_ratio < 5:
        time_interpretation = f"Merge Sort razoável ({time_ratio:.2f}x mais lento)"
    else:
        time_interpretation = f"Merge Sort significativamente mais lento ({time_ratio:.2f}x)"
    
    return {
        'merge_sort': {
            'time_ms': merge_sort_time * 1000,
            'comparisons': merge_sort_comparisons,
            'comparisons_per_element': merge_sort_comparisons / n_elements
        },
        'native_sort': {
            'time_ms': native_sort_time * 1000
        },
        'comparison': {
            'time_ratio': time_ratio,
            'theoretical_comparisons': theoretical_comparisons,
            'actual_vs_theoretical_percent': comparison_efficiency
        },
        'interpretation': time_interpretation
    }


def analyze_market_scenarios(
    expected_values_per_scenario: Dict[str, float],
    scenarios_probabilities: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analisa impacto dos cenários de mercado (Desafio 5).
    
    Args:
        expected_values_per_scenario: Valor esperado em cada cenário
        scenarios_probabilities: Probabilidade de cada cenário
    
    Returns:
        Dict com análise de cenários
    """
    # Valor esperado total
    total_expected_value = sum(
        expected_values_per_scenario[scenario] * scenarios_probabilities[scenario]
        for scenario in expected_values_per_scenario.keys()
    )
    
    # Cenário mais valioso
    best_scenario = max(expected_values_per_scenario.items(), key=lambda x: x[1])
    
    # Cenário menos valioso
    worst_scenario = min(expected_values_per_scenario.items(), key=lambda x: x[1])
    
    # Cenário mais provável
    most_likely_scenario = max(scenarios_probabilities.items(), key=lambda x: x[1])
    
    # Range de valores
    value_range = best_scenario[1] - worst_scenario[1]
    
    # Contribuição ponderada de cada cenário
    contributions = {}
    for scenario in expected_values_per_scenario.keys():
        contribution = expected_values_per_scenario[scenario] * scenarios_probabilities[scenario]
        contribution_percent = (contribution / total_expected_value * 100) if total_expected_value > 0 else 0
        contributions[scenario] = {
            'value': expected_values_per_scenario[scenario],
            'probability': scenarios_probabilities[scenario],
            'contribution': contribution,
            'contribution_percent': contribution_percent
        }
    
    return {
        'total_expected_value': total_expected_value,
        'best_scenario': {
            'name': best_scenario[0],
            'value': best_scenario[1]
        },
        'worst_scenario': {
            'name': worst_scenario[0],
            'value': worst_scenario[1]
        },
        'most_likely_scenario': {
            'name': most_likely_scenario[0],
            'probability': most_likely_scenario[1]
        },
        'value_range': value_range,
        'contributions': contributions
    }


def compare_all_algorithms() -> Dict[str, Any]:
    """
    Compara complexidades de todos os algoritmos usados no projeto.
    
    Returns:
        Dict com comparação de complexidades
    """
    algorithms = {
        'DP Knapsack 2D': {
            'desafio': 1,
            'time_complexity': 'O(n × T × C)',
            'space_complexity': 'O(n × T × C)',
            'tipo': 'Programação Dinâmica',
            'garantia': 'Solução ótima',
            'practical': 'Muito eficiente (n=12, T=350, C=30)'
        },
        'Monte Carlo': {
            'desafio': 1,
            'time_complexity': 'O(N × (n × T × C))',
            'space_complexity': 'O(N)',
            'tipo': 'Simulação Estocástica',
            'garantia': 'Aproximação probabilística',
            'practical': 'Escalável (N=1000 cenários)'
        },
        'Permutations': {
            'desafio': 2,
            'time_complexity': 'O(n! × p)',
            'space_complexity': 'O(n!)',
            'tipo': 'Enumeração Completa',
            'garantia': 'Todas as soluções',
            'practical': 'Viável para n≤10 (5!=120)'
        },
        'Greedy (V/T)': {
            'desafio': 3,
            'time_complexity': 'O(n log n)',
            'space_complexity': 'O(n)',
            'tipo': 'Guloso',
            'garantia': 'Não garante ótimo',
            'practical': 'Muito rápido'
        },
        'Exhaustive Search': {
            'desafio': 3,
            'time_complexity': 'O(2^n × n)',
            'space_complexity': 'O(n)',
            'tipo': 'Força Bruta',
            'garantia': 'Solução ótima',
            'practical': 'Viável para n≤20'
        },
        'Merge Sort': {
            'desafio': 4,
            'time_complexity': 'O(n log n)',
            'space_complexity': 'O(n)',
            'tipo': 'Dividir e Conquistar',
            'garantia': 'Sempre O(n log n)',
            'practical': 'Estável e previsível'
        },
        'Greedy + Look-ahead': {
            'desafio': 5,
            'time_complexity': 'O(n × k × m)',
            'space_complexity': 'O(n)',
            'tipo': 'Guloso com Previsão',
            'garantia': 'Heurística melhorada',
            'practical': 'Bom compromisso qualidade/tempo'
        },
        'DP Recommendation': {
            'desafio': 5,
            'time_complexity': 'O(C(n,k) × m)',
            'space_complexity': 'O(n)',
            'tipo': 'Programação Dinâmica',
            'garantia': 'Solução ótima',
            'practical': 'Viável para k≤5'
        }
    }
    
    return algorithms


def calculate_aggregate_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula métricas agregadas de todos os desafios.
    
    Args:
        results: Dicionário com resultados de todos os desafios
    
    Returns:
        Dict com métricas agregadas
    """
    metrics = {}
    
    # Desafio 1: DP Knapsack + Monte Carlo
    if 'desafio1' in results:
        d1 = results['desafio1']
        metrics['desafio1'] = {
            'deterministic_value': d1.get('deterministic', {}).get('solution', {}).get('total_value', 0),
            'stochastic_mean': d1.get('stochastic', {}).get('monte_carlo_result', {}).get('expected_value', 0),
            'stochastic_std': d1.get('stochastic', {}).get('monte_carlo_result', {}).get('std_deviation', 0)
        }
    
    # Desafio 2: Permutações
    if 'desafio2' in results:
        d2 = results['desafio2']
        metrics['desafio2'] = {
            'best_cost': d2.get('statistics', {}).get('best_cost', 0),
            'worst_cost': d2.get('statistics', {}).get('worst_cost', 0),
            'avg_cost': d2.get('statistics', {}).get('avg_all', 0),
            'n_permutations': len(d2.get('all_costs', []))
        }
    
    # Desafio 3: Guloso vs Ótimo
    if 'desafio3' in results:
        d3 = results['desafio3']
        metrics['desafio3'] = {
            'greedy_value': d3.get('greedy', {}).get('total_value', 0),
            'optimal_value': d3.get('optimal', {}).get('total_value', 0),
            'greedy_is_optimal': d3.get('comparison', {}).get('greedy_is_optimal', False)
        }
    
    # Desafio 4: Merge Sort
    if 'desafio4' in results:
        d4 = results['desafio4']
        metrics['desafio4'] = {
            'merge_sort_time': d4.get('merge_sort_result', {}).get('execution_time', 0),
            'native_sort_time': d4.get('native_sort_result', {}).get('execution_time', 0),
            'time_ratio': d4.get('comparison', {}).get('time_ratio', 0)
        }
    
    # Desafio 5: Recomendação
    if 'desafio5' in results:
        d5 = results['desafio5']
        metrics['desafio5'] = {
            'expected_value': d5.get('recommendation', {}).get('expected_value', 0),
            'n_skills_recommended': len(d5.get('recommendation', {}).get('skills_recommended', []))
        }
    
    return metrics


def generate_summary_report(results: Dict[str, Any]) -> str:
    """
    Gera relatório textual consolidado de todos os desafios.
    
    Args:
        results: Dicionário com resultados de todos os desafios
    
    Returns:
        str: Relatório formatado
    """
    lines = []
    lines.append("=" * 70)
    lines.append("RELATÓRIO CONSOLIDADO - MOTOR DE ORIENTAÇÃO DE HABILIDADES (MOH)")
    lines.append("=" * 70)
    
    # Desafio 1
    if 'desafio1' in results:
        lines.append("\n📊 DESAFIO 1 - Caminho de Valor Máximo")
        lines.append("-" * 70)
        d1 = results['desafio1']
        det_value = d1.get('deterministic', {}).get('solution', {}).get('total_value', 0)
        sto_mean = d1.get('stochastic', {}).get('monte_carlo_result', {}).get('expected_value', 0)
        sto_std = d1.get('stochastic', {}).get('monte_carlo_result', {}).get('std_deviation', 0)
        
        lines.append(f"Determinístico: Valor = {det_value:.2f}")
        lines.append(f"Estocástico:    E[Valor] = {sto_mean:.2f} ± {sto_std:.2f}")
        diff = abs(det_value - sto_mean)
        lines.append(f"Diferença:      {diff:.2f} ({diff/sto_mean*100:.1f}%)")
    
    # Desafio 2
    if 'desafio2' in results:
        lines.append("\n📊 DESAFIO 2 - Verificação Crítica")
        lines.append("-" * 70)
        d2 = results['desafio2']
        best = d2.get('statistics', {}).get('best_cost', 0)
        worst = d2.get('statistics', {}).get('worst_cost', 0)
        avg = d2.get('statistics', {}).get('avg_all', 0)
        
        lines.append(f"Melhor custo:  {best:.0f}h")
        lines.append(f"Pior custo:    {worst:.0f}h")
        lines.append(f"Custo médio:   {avg:.0f}h")
        lines.append(f"Economia:      {worst - best:.0f}h ({(worst-best)/worst*100:.1f}%)")
    
    # Desafio 3
    if 'desafio3' in results:
        lines.append("\n📊 DESAFIO 3 - Pivô Mais Rápido")
        lines.append("-" * 70)
        d3 = results['desafio3']
        greedy = d3.get('greedy', {}).get('total_value', 0)
        optimal = d3.get('optimal', {}).get('total_value', 0)
        is_optimal = d3.get('comparison', {}).get('greedy_is_optimal', False)
        
        lines.append(f"Guloso:  Valor = {greedy:.2f}")
        lines.append(f"Ótimo:   Valor = {optimal:.2f}")
        lines.append(f"Guloso é ótimo: {'✅ SIM' if is_optimal else '❌ NÃO'}")
        quality = (greedy / optimal * 100) if optimal > 0 else 0
        lines.append(f"Qualidade: {quality:.1f}% do ótimo")
    
    # Desafio 4
    if 'desafio4' in results:
        lines.append("\n📊 DESAFIO 4 - Trilhas Paralelas")
        lines.append("-" * 70)
        d4 = results['desafio4']
        merge_time = d4.get('merge_sort_result', {}).get('execution_time', 0) * 1000
        native_time = d4.get('native_sort_result', {}).get('execution_time', 0) * 1000
        ratio = d4.get('comparison', {}).get('time_ratio', 0)
        
        lines.append(f"Merge Sort:  {merge_time:.3f} ms")
        lines.append(f"Native Sort: {native_time:.3f} ms")
        lines.append(f"Razão:       {ratio:.2f}x")
    
    # Desafio 5
    if 'desafio5' in results:
        lines.append("\n📊 DESAFIO 5 - Recomendação")
        lines.append("-" * 70)
        d5 = results['desafio5']
        skills = d5.get('recommendation', {}).get('skills_recommended', [])
        expected = d5.get('recommendation', {}).get('expected_value', 0)
        
        lines.append(f"Skills recomendadas: {', '.join(skills)}")
        lines.append(f"E[Valor]:            {expected:.2f}")
    
    lines.append("\n" + "=" * 70)
    
    return '\n'.join(lines)


def export_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Exporta métricas para arquivo JSON.
    
    Args:
        metrics: Dicionário com métricas
        filepath: Caminho do arquivo de saída
    """
    import json
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)