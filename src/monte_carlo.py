"""
Simulação Monte Carlo para análise de incerteza

Este módulo implementa simulação Monte Carlo para lidar com incerteza
nos valores das habilidades. É usado principalmente no Desafio 1 para
simular V ~ Uniforme[V-10%, V+10%] em 1000 cenários.

Funções principais:
    - simulate_value_with_uncertainty: Simula valor com variação aleatória
    - generate_scenarios: Gera múltiplos cenários estocásticos
    - run_monte_carlo: Executa simulação MC completa
    - calculate_statistics: Calcula estatísticas dos resultados

Aplicações:
    - Desafio 1: Maximizar E[Valor total] sob incerteza
    - Desafio 5: Simular cenários de mercado
"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class MonteCarloResult:
    """
    Resultado de uma simulação Monte Carlo.
    
    Attributes:
        expected_value: Valor esperado (média)
        std_deviation: Desvio padrão
        min_value: Valor mínimo observado
        max_value: Valor máximo observado
        median: Mediana
        percentile_25: 25º percentil (Q1)
        percentile_75: 75º percentil (Q3)
        confidence_interval_95: Intervalo de confiança 95% (lower, upper)
        all_results: Lista com todos os resultados
        n_scenarios: Número de cenários simulados
    """
    expected_value: float
    std_deviation: float
    min_value: float
    max_value: float
    median: float
    percentile_25: float
    percentile_75: float
    confidence_interval_95: Tuple[float, float]
    all_results: List[float]
    n_scenarios: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'expected_value': self.expected_value,
            'std_deviation': self.std_deviation,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'median': self.median,
            'percentile_25': self.percentile_25,
            'percentile_75': self.percentile_75,
            'confidence_interval_95': {
                'lower': self.confidence_interval_95[0],
                'upper': self.confidence_interval_95[1]
            },
            'n_scenarios': self.n_scenarios
        }
    
    def __str__(self) -> str:
        """Representação string formatada."""
        return (
            f"Monte Carlo Results ({self.n_scenarios} scenarios):\n"
            f"  E[X] = {self.expected_value:.2f} ± {self.std_deviation:.2f}\n"
            f"  Range: [{self.min_value:.2f}, {self.max_value:.2f}]\n"
            f"  Median: {self.median:.2f}\n"
            f"  95% CI: [{self.confidence_interval_95[0]:.2f}, {self.confidence_interval_95[1]:.2f}]"
        )


def simulate_value_with_uncertainty(
    base_value: float,
    uncertainty: float = 0.10,
    distribution: str = 'uniform',
    seed: Optional[int] = None
) -> float:
    """
    Simula um valor com incerteza usando distribuição especificada.
    
    Para o Desafio 1: V ~ Uniforme[V-10%, V+10%]
    
    Args:
        base_value: Valor base (nominal)
        uncertainty: Percentual de incerteza (0.10 = ±10%)
        distribution: Tipo de distribuição ('uniform', 'normal', 'triangular')
        seed: Seed para reprodutibilidade (opcional)
    
    Returns:
        float: Valor simulado
    
    Examples:
        >>> # Simula valor com ±10% de variação uniforme
        >>> simulated = simulate_value_with_uncertainty(100, uncertainty=0.10)
        >>> # Resultado: valor entre 90 e 110
        
        >>> # Com distribuição normal (mais concentrado no centro)
        >>> simulated = simulate_value_with_uncertainty(100, uncertainty=0.10, distribution='normal')
    """
    if seed is not None:
        np.random.seed(seed)
    
    delta = base_value * uncertainty
    
    if distribution == 'uniform':
        # Uniforme: probabilidade igual em todo intervalo
        lower = base_value - delta
        upper = base_value + delta
        return np.random.uniform(lower, upper)
    
    elif distribution == 'normal':
        # Normal: mais concentrado no centro
        # σ tal que 95% dos valores estejam em [base-delta, base+delta]
        sigma = delta / 2  # Aproximadamente 2σ = range
        return np.random.normal(base_value, sigma)
    
    elif distribution == 'triangular':
        # Triangular: pico no valor base
        lower = base_value - delta
        upper = base_value + delta
        return np.random.triangular(lower, base_value, upper)
    
    else:
        raise ValueError(f"Distribuição '{distribution}' não suportada. Use 'uniform', 'normal' ou 'triangular'.")


def generate_scenarios(
    skills_dict: Dict[str, Dict[str, Any]],
    n_scenarios: int = 1000,
    uncertainty: float = 0.10,
    distribution: str = 'uniform',
    vary_time: bool = False,
    seed: Optional[int] = None
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Gera múltiplos cenários estocásticos do dataset de habilidades.
    
    Cada cenário é uma versão do dataset com valores perturbados.
    Por padrão, apenas 'valor' varia; tempo e complexidade ficam fixos.
    
    Args:
        skills_dict: Dicionário original de habilidades
        n_scenarios: Número de cenários a gerar (padrão: 1000)
        uncertainty: Percentual de incerteza (padrão: 0.10 = ±10%)
        distribution: Tipo de distribuição ('uniform', 'normal', 'triangular')
        vary_time: Se True, também varia tempo (não recomendado para Desafio 1)
        seed: Seed para reprodutibilidade
    
    Returns:
        List[Dict]: Lista de cenários, onde cada cenário é um skills_dict modificado
    
    Complexity:
        O(n_scenarios × n_skills)
    
    Examples:
        >>> from src.graph_structures import load_skills_from_json
        >>> from src.config import SKILLS_DATASET_FILE
        >>> 
        >>> skills = load_skills_from_json(SKILLS_DATASET_FILE)
        >>> scenarios = generate_scenarios(skills, n_scenarios=1000)
        >>> 
        >>> # Cada cenário tem valores ligeiramente diferentes
        >>> print(scenarios[0]['S1']['valor'])  # Ex: 3.15
        >>> print(scenarios[1]['S1']['valor'])  # Ex: 2.87
    """
    if seed is not None:
        np.random.seed(seed)
    
    scenarios = []
    
    for scenario_idx in range(n_scenarios):
        # Copia o dataset original
        scenario = {}
        
        for skill_id, metadata in skills_dict.items():
            # Copia metadados
            scenario_metadata = metadata.copy()
            
            # Simula valor com incerteza
            base_value = metadata['valor']
            simulated_value = simulate_value_with_uncertainty(
                base_value,
                uncertainty=uncertainty,
                distribution=distribution,
                seed=None  # Não fixar seed aqui para garantir variação
            )
            scenario_metadata['valor'] = simulated_value
            
            # Opcionalmente varia tempo também
            if vary_time:
                base_time = metadata['tempo_horas']
                simulated_time = simulate_value_with_uncertainty(
                    base_time,
                    uncertainty=uncertainty,
                    distribution=distribution,
                    seed=None
                )
                scenario_metadata['tempo_horas'] = max(1, int(simulated_time))  # Garante >= 1
            
            scenario[skill_id] = scenario_metadata
        
        scenarios.append(scenario)
    
    return scenarios


def run_monte_carlo(
    optimization_function: Callable,
    scenarios: List[Dict[str, Dict[str, Any]]],
    **kwargs
) -> MonteCarloResult:
    """
    Executa simulação Monte Carlo completa.
    
    Para cada cenário:
    1. Executa a função de otimização
    2. Coleta o resultado
    3. Calcula estatísticas agregadas
    
    Args:
        optimization_function: Função que recebe skills_dict e retorna resultado numérico
                              Assinatura: f(skills_dict, **kwargs) -> float ou Dict com 'value'
        scenarios: Lista de cenários gerados por generate_scenarios()
        **kwargs: Argumentos adicionais para optimization_function
    
    Returns:
        MonteCarloResult: Objeto com todas as estatísticas
    
    Examples:
        >>> def my_optimization(skills_dict, max_time):
        ...     # Sua lógica de otimização aqui
        ...     return total_value  # ou {'value': total_value, ...}
        
        >>> scenarios = generate_scenarios(skills, n_scenarios=1000)
        >>> result = run_monte_carlo(my_optimization, scenarios, max_time=350)
        >>> 
        >>> print(result.expected_value)  # E[Valor total]
        >>> print(result.std_deviation)   # σ
    """
    results = []
    
    for scenario in scenarios:
        # Executa otimização neste cenário
        result = optimization_function(scenario, **kwargs)
        
        # Extrai valor numérico
        if isinstance(result, dict):
            value = result.get('value', result.get('total_value', 0))
        else:
            value = float(result)
        
        results.append(value)
    
    # Calcula estatísticas
    return calculate_statistics(results)


def calculate_statistics(results: List[float]) -> MonteCarloResult:
    """
    Calcula estatísticas descritivas de uma lista de resultados.
    
    Args:
        results: Lista de valores numéricos
    
    Returns:
        MonteCarloResult: Objeto com todas as estatísticas
    
    Complexity:
        O(n log n) devido à ordenação para percentis
    
    Examples:
        >>> results = [100, 105, 95, 110, 90, 102, 98]
        >>> stats = calculate_statistics(results)
        >>> print(stats.expected_value)  # Média
        >>> print(stats.std_deviation)   # Desvio padrão
    """
    results_array = np.array(results)
    
    # Estatísticas básicas
    expected_value = float(np.mean(results_array))
    std_deviation = float(np.std(results_array, ddof=1))  # ddof=1 para amostra
    min_value = float(np.min(results_array))
    max_value = float(np.max(results_array))
    median = float(np.median(results_array))
    
    # Percentis
    percentile_25 = float(np.percentile(results_array, 25))
    percentile_75 = float(np.percentile(results_array, 75))
    
    # Intervalo de confiança 95% (assumindo distribuição normal)
    # CI = μ ± 1.96 × (σ / √n)
    n = len(results_array)
    margin_of_error = 1.96 * (std_deviation / np.sqrt(n))
    ci_lower = expected_value - margin_of_error
    ci_upper = expected_value + margin_of_error
    
    return MonteCarloResult(
        expected_value=expected_value,
        std_deviation=std_deviation,
        min_value=min_value,
        max_value=max_value,
        median=median,
        percentile_25=percentile_25,
        percentile_75=percentile_75,
        confidence_interval_95=(ci_lower, ci_upper),
        all_results=results,
        n_scenarios=n
    )


def compare_deterministic_vs_stochastic(
    deterministic_result: float,
    stochastic_result: MonteCarloResult
) -> Dict[str, Any]:
    """
    Compara resultado determinístico vs estocástico.
    
    Útil para o Desafio 1, onde precisamos comparar:
    - Solução determinística (sem incerteza)
    - Solução estocástica (com Monte Carlo)
    
    Args:
        deterministic_result: Resultado da otimização determinística
        stochastic_result: Resultado da simulação Monte Carlo
    
    Returns:
        Dict com comparação detalhada
    
    Examples:
        >>> deterministic = 30.0
        >>> stochastic = run_monte_carlo(...)
        >>> comparison = compare_deterministic_vs_stochastic(deterministic, stochastic)
        >>> 
        >>> print(comparison['difference'])
        >>> print(comparison['relative_error'])
    """
    expected = stochastic_result.expected_value
    
    # Diferença absoluta
    difference = expected - deterministic_result
    
    # Erro relativo percentual
    if deterministic_result != 0:
        relative_error = (difference / deterministic_result) * 100
    else:
        relative_error = float('inf') if difference != 0 else 0
    
    # Verifica se determinístico está dentro do IC 95%
    ci_lower, ci_upper = stochastic_result.confidence_interval_95
    within_ci = ci_lower <= deterministic_result <= ci_upper
    
    return {
        'deterministic_value': deterministic_result,
        'stochastic_expected': expected,
        'stochastic_std': stochastic_result.std_deviation,
        'difference': difference,
        'relative_error_percent': relative_error,
        'deterministic_within_95ci': within_ci,
        'confidence_interval_95': stochastic_result.confidence_interval_95,
        'interpretation': (
            f"O valor determinístico ({deterministic_result:.2f}) "
            f"{'está' if within_ci else 'NÃO está'} dentro do IC 95% "
            f"[{ci_lower:.2f}, {ci_upper:.2f}] do valor esperado estocástico "
            f"({expected:.2f} ± {stochastic_result.std_deviation:.2f})."
        )
    }


def analyze_sensitivity(
    optimization_function: Callable,
    base_scenario: Dict[str, Dict[str, Any]],
    skill_to_vary: str,
    variation_range: Tuple[float, float],
    n_points: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Análise de sensibilidade: varia um parâmetro e observa o impacto.
    
    Útil para entender quais habilidades têm maior impacto na solução.
    
    Args:
        optimization_function: Função de otimização
        base_scenario: Cenário base
        skill_to_vary: ID da habilidade a variar
        variation_range: (min_multiplier, max_multiplier) - ex: (0.5, 1.5) = 50% a 150%
        n_points: Número de pontos a testar
        **kwargs: Argumentos para optimization_function
    
    Returns:
        Dict com curva de sensibilidade
    
    Examples:
        >>> # Analisa impacto de variar valor de S1
        >>> sensitivity = analyze_sensitivity(
        ...     my_opt_func,
        ...     base_scenario,
        ...     skill_to_vary='S1',
        ...     variation_range=(0.5, 1.5),
        ...     n_points=20
        ... )
        >>> 
        >>> # Plota curva
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(sensitivity['multipliers'], sensitivity['results'])
    """
    base_value = base_scenario[skill_to_vary]['valor']
    
    multipliers = np.linspace(variation_range[0], variation_range[1], n_points)
    results = []
    
    for multiplier in multipliers:
        # Cria cenário modificado
        modified_scenario = {k: v.copy() for k, v in base_scenario.items()}
        modified_scenario[skill_to_vary]['valor'] = base_value * multiplier
        
        # Executa otimização
        result = optimization_function(modified_scenario, **kwargs)
        
        if isinstance(result, dict):
            value = result.get('value', result.get('total_value', 0))
        else:
            value = float(result)
        
        results.append(value)
    
    return {
        'skill_id': skill_to_vary,
        'base_value': base_value,
        'multipliers': multipliers.tolist(),
        'results': results,
        'max_impact': max(results) - min(results),
        'relative_impact': (max(results) - min(results)) / min(results) if min(results) > 0 else float('inf')
    }


def bootstrap_confidence_interval(
    results: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calcula intervalo de confiança usando bootstrap (mais robusto que normal).
    
    Bootstrap: reamostragem com reposição para estimar distribuição.
    
    Args:
        results: Lista de resultados originais
        confidence_level: Nível de confiança (0.95 = 95%)
        n_bootstrap: Número de amostras bootstrap
        seed: Seed para reprodutibilidade
    
    Returns:
        Tuple[float, float]: (lower_bound, upper_bound)
    
    Complexity:
        O(n_bootstrap × n)
    
    Examples:
        >>> results = [100, 105, 95, 110, 90]
        >>> ci_lower, ci_upper = bootstrap_confidence_interval(results)
        >>> print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    """
    if seed is not None:
        np.random.seed(seed)
    
    results_array = np.array(results)
    n = len(results_array)
    
    # Gera amostras bootstrap
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Reamostragem com reposição
        sample = np.random.choice(results_array, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calcula percentis
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return float(ci_lower), float(ci_upper)


def save_monte_carlo_results(
    result: MonteCarloResult,
    filepath: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Salva resultados Monte Carlo em arquivo JSON.
    
    Args:
        result: Resultado da simulação
        filepath: Caminho do arquivo de saída
        additional_info: Informações adicionais a incluir (opcional)
    
    Examples:
        >>> result = run_monte_carlo(...)
        >>> save_monte_carlo_results(
        ...     result,
        ...     'outputs/desafio1_monte_carlo.json',
        ...     additional_info={'max_time': 350, 'max_complexity': 30}
        ... )
    """
    data = result.to_dict()
    
    # Remove lista completa de resultados para economizar espaço
    # (mantém apenas estatísticas agregadas)
    if 'all_results' in data:
        del data['all_results']
    
    if additional_info:
        data['additional_info'] = additional_info
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_monte_carlo_summary(result: MonteCarloResult) -> None:
    """
    Imprime resumo formatado dos resultados Monte Carlo.
    
    Args:
        result: Resultado da simulação
    
    Examples:
        >>> result = run_monte_carlo(...)
        >>> print_monte_carlo_summary(result)
    """
    print("\n" + "=" * 70)
    print("📊 RESULTADOS DA SIMULAÇÃO MONTE CARLO")
    print("=" * 70)
    
    print(f"\n🎲 Número de cenários: {result.n_scenarios:,}")
    
    print(f"\n📈 Estatísticas Descritivas:")
    print(f"   • Valor Esperado (E[X]): {result.expected_value:.2f}")
    print(f"   • Desvio Padrão (σ):     {result.std_deviation:.2f}")
    print(f"   • Mediana:               {result.median:.2f}")
    
    print(f"\n📉 Range:")
    print(f"   • Mínimo:                {result.min_value:.2f}")
    print(f"   • Máximo:                {result.max_value:.2f}")
    print(f"   • Amplitude:             {result.max_value - result.min_value:.2f}")
    
    print(f"\n📊 Percentis:")
    print(f"   • Q1 (25%):              {result.percentile_25:.2f}")
    print(f"   • Q2 (50% - Mediana):    {result.median:.2f}")
    print(f"   • Q3 (75%):              {result.percentile_75:.2f}")
    print(f"   • IQR (Q3 - Q1):         {result.percentile_75 - result.percentile_25:.2f}")
    
    ci_lower, ci_upper = result.confidence_interval_95
    print(f"\n🎯 Intervalo de Confiança 95%:")
    print(f"   • [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"   • Margem de erro: ±{(ci_upper - ci_lower) / 2:.2f}")
    
    # Coeficiente de variação
    cv = (result.std_deviation / result.expected_value) * 100 if result.expected_value > 0 else 0
    print(f"\n📐 Coeficiente de Variação: {cv:.2f}%")
    
    print("=" * 70)


# Atalhos para uso comum
def quick_monte_carlo(
    optimization_function: Callable,
    skills_dict: Dict[str, Dict[str, Any]],
    n_scenarios: int = 1000,
    uncertainty: float = 0.10,
    seed: Optional[int] = None,
    **kwargs
) -> MonteCarloResult:
    """
    Atalho para executar Monte Carlo completo em uma única chamada.
    
    Args:
        optimization_function: Função de otimização
        skills_dict: Dataset de habilidades
        n_scenarios: Número de cenários (padrão: 1000)
        uncertainty: Incerteza percentual (padrão: 0.10 = ±10%)
        seed: Seed para reprodutibilidade
        **kwargs: Argumentos para optimization_function
    
    Returns:
        MonteCarloResult: Resultado completo
    
    Examples:
        >>> result = quick_monte_carlo(my_opt_func, skills, n_scenarios=1000, max_time=350)
        >>> print(result)
    """
    scenarios = generate_scenarios(
        skills_dict,
        n_scenarios=n_scenarios,
        uncertainty=uncertainty,
        seed=seed
    )
    
    return run_monte_carlo(optimization_function, scenarios, **kwargs)