"""
Testes para o módulo monte_carlo.py

Valida:
- Simulação de incerteza
- Geração de cenários
- Cálculo de estatísticas
- Comparação determinístico vs estocástico
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.monte_carlo import (
    simulate_value_with_uncertainty,
    generate_scenarios,
    calculate_statistics,
    run_monte_carlo,
    compare_deterministic_vs_stochastic,
    MonteCarloResult,
    quick_monte_carlo,
    print_monte_carlo_summary
)
from src.graph_structures import load_skills_from_json
from src.config import SKILLS_DATASET_FILE


def test_simulate_value_with_uncertainty():
    """Testa simulação de valor com incerteza."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 1: Simulação de Valor com Incerteza")
    print("=" * 70)
    
    base_value = 100
    uncertainty = 0.10  # ±10%
    n_samples = 10000
    
    print(f"\nValor base: {base_value}")
    print(f"Incerteza: ±{uncertainty * 100}%")
    print(f"Range esperado: [{base_value * (1 - uncertainty)}, {base_value * (1 + uncertainty)}]")
    print(f"Amostras: {n_samples:,}")
    
    # Gera amostras
    samples = [
        simulate_value_with_uncertainty(base_value, uncertainty, distribution='uniform', seed=None)
        for _ in range(n_samples)
    ]
    
    # Calcula estatísticas
    mean = np.mean(samples)
    std = np.std(samples)
    min_val = np.min(samples)
    max_val = np.max(samples)
    
    print(f"\n📊 Resultados:")
    print(f"   • Média: {mean:.2f} (esperado: ~{base_value})")
    print(f"   • Desvio padrão: {std:.2f}")
    print(f"   • Mínimo: {min_val:.2f} (esperado: ~{base_value * 0.9})")
    print(f"   • Máximo: {max_val:.2f} (esperado: ~{base_value * 1.1})")
    
    # Validações
    assert 90 <= min_val <= 92, f"Mínimo fora do esperado: {min_val}"
    assert 108 <= max_val <= 110, f"Máximo fora do esperado: {max_val}"
    assert 98 <= mean <= 102, f"Média fora do esperado: {mean}"
    
    print("\n✅ Teste 1: PASSOU - Distribuição uniforme correta")
    return True


def test_generate_scenarios():
    """Testa geração de cenários estocásticos."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 2: Geração de Cenários")
    print("=" * 70)
    
    # Dataset simplificado para teste
    skills_dict = {
        'S1': {'nome': 'Python', 'tempo_horas': 80, 'valor': 3, 'complexidade': 4, 'pre_requisitos': []},
        'S3': {'nome': 'Algoritmos', 'tempo_horas': 100, 'valor': 7, 'complexidade': 8, 'pre_requisitos': ['S1']}
    }
    
    n_scenarios = 100
    
    print(f"Gerando {n_scenarios} cenários...")
    print(f"Valor base S1: {skills_dict['S1']['valor']}")
    print(f"Valor base S3: {skills_dict['S3']['valor']}")
    
    scenarios = generate_scenarios(skills_dict, n_scenarios=n_scenarios, uncertainty=0.10, seed=42)
    
    print(f"\n✅ Cenários gerados: {len(scenarios)}")
    
    # Valida estrutura
    assert len(scenarios) == n_scenarios
    assert 'S1' in scenarios[0]
    assert 'S3' in scenarios[0]
    
    # Coleta valores de S1 em todos os cenários
    s1_values = [scenario['S1']['valor'] for scenario in scenarios]
    
    print(f"\n📊 Variação em S1:")
    print(f"   • Mínimo: {min(s1_values):.2f}")
    print(f"   • Máximo: {max(s1_values):.2f}")
    print(f"   • Média: {np.mean(s1_values):.2f}")
    
    # Valida que valores variam
    assert len(set(s1_values)) > 50, "Valores não estão variando suficientemente"
    assert all(2.7 <= v <= 3.3 for v in s1_values), "Valores fora do range ±10%"
    
    # Valida que tempo não varia (por padrão)
    s1_times = [scenario['S1']['tempo_horas'] for scenario in scenarios]
    assert all(t == 80 for t in s1_times), "Tempo não deveria variar por padrão"
    
    print("\n✅ Teste 2: PASSOU - Cenários gerados corretamente")
    return True


def test_calculate_statistics():
    """Testa cálculo de estatísticas."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 3: Cálculo de Estatísticas")
    print("=" * 70)
    
    # Dados de teste conhecidos
    results = [100, 105, 95, 110, 90, 102, 98, 107, 93, 101]
    
    print(f"Dados de teste: {results}")
    
    stats = calculate_statistics(results)
    
    print(f"\n📊 Estatísticas calculadas:")
    print(f"   • Média: {stats.expected_value:.2f}")
    print(f"   • Desvio padrão: {stats.std_deviation:.2f}")
    print(f"   • Mínimo: {stats.min_value:.2f}")
    print(f"   • Máximo: {stats.max_value:.2f}")
    print(f"   • Mediana: {stats.median:.2f}")
    print(f"   • Q1: {stats.percentile_25:.2f}")
    print(f"   • Q3: {stats.percentile_75:.2f}")
    
    # Validações
    assert isinstance(stats, MonteCarloResult)
    assert stats.n_scenarios == 10
    assert 98 <= stats.expected_value <= 102  # Média próxima de 100
    assert stats.min_value == 90
    assert stats.max_value == 110
    
    # Valida intervalo de confiança
    ci_lower, ci_upper = stats.confidence_interval_95
    print(f"   • IC 95%: [{ci_lower:.2f}, {ci_upper:.2f}]")
    assert ci_lower < stats.expected_value < ci_upper
    
    print("\n✅ Teste 3: PASSOU - Estatísticas corretas")
    return True


def test_run_monte_carlo():
    """Testa execução completa de Monte Carlo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 4: Execução Monte Carlo Completa")
    print("=" * 70)
    
    # Função de otimização simples para teste
    def simple_optimization(skills_dict):
        """Soma todos os valores (otimização trivial)."""
        return sum(skill['valor'] for skill in skills_dict.values())
    
    # Dataset simplificado
    skills_dict = {
        'S1': {'nome': 'Python', 'tempo_horas': 80, 'valor': 3, 'complexidade': 4, 'pre_requisitos': []},
        'S2': {'nome': 'SQL', 'tempo_horas': 60, 'valor': 4, 'complexidade': 3, 'pre_requisitos': []},
    }
    
    print("Função de otimização: soma de todos os valores")
    print(f"Valor determinístico: {simple_optimization(skills_dict)}")
    
    # Gera cenários
    n_scenarios = 500
    print(f"\nGerando {n_scenarios} cenários...")
    scenarios = generate_scenarios(skills_dict, n_scenarios=n_scenarios, uncertainty=0.10, seed=42)
    
    # Executa Monte Carlo
    print("Executando Monte Carlo...")
    result = run_monte_carlo(simple_optimization, scenarios)
    
    print(f"\n📊 Resultado:")
    print(f"   • E[Soma]: {result.expected_value:.2f}")
    print(f"   • σ: {result.std_deviation:.2f}")
    print(f"   • Range: [{result.min_value:.2f}, {result.max_value:.2f}]")
    
    # Validações
    assert result.n_scenarios == n_scenarios
    # Esperado: ~7 (3 + 4)
    assert 6.5 <= result.expected_value <= 7.5
    
    print("\n✅ Teste 4: PASSOU - Monte Carlo executado com sucesso")
    return True


def test_compare_deterministic_vs_stochastic():
    """Testa comparação determinístico vs estocástico."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 5: Comparação Determinístico vs Estocástico")
    print("=" * 70)
    
    # Resultado determinístico
    deterministic = 30.0
    
    # Resultado estocástico (simulado)
    stochastic_results = [29, 31, 30, 32, 28, 30, 31, 29, 30, 31]
    stochastic = calculate_statistics(stochastic_results)
    
    print(f"Valor determinístico: {deterministic}")
    print(f"Valor estocástico (E[X]): {stochastic.expected_value:.2f} ± {stochastic.std_deviation:.2f}")
    
    # Compara
    comparison = compare_deterministic_vs_stochastic(deterministic, stochastic)
    
    print(f"\n📊 Comparação:")
    print(f"   • Diferença: {comparison['difference']:.2f}")
    print(f"   • Erro relativo: {comparison['relative_error_percent']:.2f}%")
    print(f"   • Dentro do IC 95%: {comparison['deterministic_within_95ci']}")
    
    print(f"\n💬 Interpretação:")
    print(f"   {comparison['interpretation']}")
    
    # Validações
    assert 'difference' in comparison
    assert 'relative_error_percent' in comparison
    assert 'deterministic_within_95ci' in comparison
    
    print("\n✅ Teste 5: PASSOU - Comparação realizada")
    return True


def test_with_real_dataset():
    """Testa com dataset real de 12 habilidades."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 6: Monte Carlo com Dataset Real")
    print("=" * 70)
    
    print(f"Carregando dataset de: {SKILLS_DATASET_FILE}")
    skills = load_skills_from_json(SKILLS_DATASET_FILE)
    
    print(f"✅ Dataset carregado: {len(skills)} habilidades")
    
    # Função de otimização simples: soma todos os valores
    def sum_all_values(skills_dict):
        return sum(skill['valor'] for skill in skills_dict.values())
    
    deterministic = sum_all_values(skills)
    print(f"\nValor determinístico (soma): {deterministic}")
    
    # Executa Monte Carlo (poucos cenários para teste rápido)
    print("\nExecutando Monte Carlo (100 cenários)...")
    result = quick_monte_carlo(
        sum_all_values,
        skills,
        n_scenarios=100,
        uncertainty=0.10,
        seed=42
    )
    
    print_monte_carlo_summary(result)
    
    # Validações
    assert result.n_scenarios == 100
    # Valor esperado deve ser próximo do determinístico
    assert abs(result.expected_value - deterministic) < 5
    
    print("\n✅ Teste 6: PASSOU - Monte Carlo funciona com dataset real")
    return True


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DE monte_carlo.py")
    print("=" * 70)
    
    tests = [
        ("Simulação de Valor com Incerteza", test_simulate_value_with_uncertainty),
        ("Geração de Cenários", test_generate_scenarios),
        ("Cálculo de Estatísticas", test_calculate_statistics),
        ("Execução Monte Carlo Completa", test_run_monte_carlo),
        ("Comparação Determinístico vs Estocástico", test_compare_deterministic_vs_stochastic),
        ("Monte Carlo com Dataset Real", test_with_real_dataset),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO em '{test_name}': {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES - monte_carlo.py")
    print("=" * 70)
    
    total_tests = len(tests)
    passed_tests = sum(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSOU" if results[i] else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Resultados: {passed_tests}/{total_tests} testes passaram")
    
    if passed_tests == total_tests:
        print("\n🎉 monte_carlo.py VALIDADO COM SUCESSO!")
        print("✅ Pronto para usar no Desafio 1")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} teste(s) falharam.")
        return 1


if __name__ == '__main__':
    sys.exit(main())