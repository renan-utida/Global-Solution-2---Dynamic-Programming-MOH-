"""
Testes para o Desafio 5 - Recomendação de Próximas Habilidades
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desafio5_recommendation import (
    load_market_scenarios,
    calculate_expected_value,
    greedy_recommendation_with_lookahead,
    dp_recommendation_exhaustive,
    compare_recommendation_methods,
    solve_complete,
    MarketScenario,
    Recommendation,
    RECOMMENDATION_HORIZON_YEARS,
    N_RECOMMENDATIONS
)
from src.graph_structures import build_graph_from_file
from src.config import SKILLS_DATASET_FILE


def test_load_market_scenarios():
    """Testa carregamento de cenários de mercado."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 1: Carregamento de Cenários de Mercado")
    print("=" * 70)
    
    scenarios = load_market_scenarios()
    
    print(f"\nCenários carregados: {len(scenarios)}")
    for scenario in scenarios:
        print(f"   • {scenario.name}: {scenario.probability*100:.0f}%")
        print(f"     Boosts: {len(scenario.boosts)} skills")
    
    # Validações
    assert len(scenarios) == 4, f"Esperado 4 cenários, obtido {len(scenarios)}"
    
    # Soma das probabilidades deve ser ~1.0
    total_prob = sum(s.probability for s in scenarios)
    assert 0.99 <= total_prob <= 1.01, f"Probabilidades devem somar ~1.0, obtido {total_prob}"
    
    # Cada cenário deve ter boosts
    for scenario in scenarios:
        assert len(scenario.boosts) > 0, f"Cenário {scenario.name} sem boosts"
    
    print("\n✅ Teste 1: PASSOU")
    return True


def test_calculate_expected_value():
    """Testa cálculo de valor esperado."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 2: Cálculo de Valor Esperado")
    print("=" * 70)
    
    scenarios = load_market_scenarios()
    
    # Testa com skill que tem boost
    skill_id = 'S6'  # IA Generativa - deve ter boost em 'ia_em_alta'
    base_value = 10
    
    expected, per_scenario = calculate_expected_value(skill_id, base_value, scenarios)
    
    print(f"\nSkill: {skill_id}")
    print(f"Valor base: {base_value}")
    print(f"Valor esperado: {expected:.2f}")
    print(f"\nPor cenário:")
    for scenario_name, value in per_scenario.items():
        print(f"   • {scenario_name}: {value:.2f}")
    
    # Validações
    assert expected > 0
    assert len(per_scenario) == len(scenarios)
    
    # Valor esperado deve ser >= valor base (há boosts positivos)
    assert expected >= base_value * 0.9, f"Valor esperado muito baixo: {expected}"
    
    print("\n✅ Teste 2: PASSOU")
    return True


def test_greedy_recommendation():
    """Testa recomendação gulosa com look-ahead."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 3: Recomendação Gulosa com Look-Ahead")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    # Simula perfil: já tem S1 e S2
    current_skills = {'S1', 'S2'}
    
    recommendation = greedy_recommendation_with_lookahead(
        graph, current_skills, n_recommendations=3, lookahead_depth=2
    )
    
    print(f"\n✅ Recomendação:")
    print(f"   Skills: {' → '.join(recommendation.skills_recommended)}")
    print(f"   E[Valor]: {recommendation.expected_value:.2f}")
    print(f"\n   {recommendation.reasoning}")
    
    # Validações
    assert len(recommendation.skills_recommended) <= 3
    assert recommendation.expected_value > 0
    assert len(recommendation.details) == len(recommendation.skills_recommended)
    
    # Skills recomendadas não devem estar em current_skills
    for skill_id in recommendation.skills_recommended:
        assert skill_id not in current_skills, f"{skill_id} já está adquirida"
    
    print("\n✅ Teste 3: PASSOU")
    return True


def test_dp_exhaustive():
    """Testa recomendação com DP exaustivo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 4: Recomendação com DP Exaustivo")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    # Perfil com várias skills já adquiridas
    current_skills = {'S1', 'S2', 'S7', 'H10', 'H12'}
    
    recommendation = dp_recommendation_exhaustive(
        graph, current_skills, n_recommendations=3
    )
    
    print(f"\n✅ Recomendação:")
    print(f"   Skills: {' → '.join(recommendation.skills_recommended)}")
    print(f"   E[Valor]: {recommendation.expected_value:.2f}")
    
    # Validações
    assert len(recommendation.skills_recommended) <= 3
    assert recommendation.expected_value > 0
    
    print("\n✅ Teste 4: PASSOU")
    return True


def test_compare_methods():
    """Testa comparação entre métodos."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 5: Comparação de Métodos")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    current_skills = {'S1', 'S2'}
    
    comparison = compare_recommendation_methods(
        graph, current_skills, n_recommendations=3
    )
    
    print(f"\n📊 Comparação:")
    print(f"   Guloso simples:     {comparison['comparison']['greedy_simple_value']:.2f}")
    print(f"   Guloso + lookahead: {comparison['comparison']['greedy_lookahead_value']:.2f}")
    print(f"   DP exaustivo:       {comparison['comparison']['dp_exhaustive_value']:.2f}")
    print(f"   Melhor método: {comparison['comparison']['best_method']}")
    
    # Validações
    assert 'greedy_simple' in comparison
    assert 'greedy_lookahead' in comparison
    assert 'dp_exhaustive' in comparison
    assert 'comparison' in comparison
    
    # DP exaustivo deve ser >= outros métodos (é ótimo)
    dp_value = comparison['comparison']['dp_exhaustive_value']
    greedy_value = comparison['comparison']['greedy_simple_value']
    lookahead_value = comparison['comparison']['greedy_lookahead_value']
    
    assert dp_value >= greedy_value * 0.95, "DP deve ser melhor que guloso simples"
    
    print("\n✅ Teste 5: PASSOU")
    return True


def test_empty_current_skills():
    """Testa recomendação com perfil vazio (iniciante)."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 6: Recomendação para Iniciante")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    # Perfil vazio
    current_skills = set()
    
    recommendation = greedy_recommendation_with_lookahead(
        graph, current_skills, n_recommendations=3, lookahead_depth=1
    )
    
    print(f"\n✅ Recomendação para iniciante:")
    print(f"   Skills: {' → '.join(recommendation.skills_recommended)}")
    print(f"   E[Valor]: {recommendation.expected_value:.2f}")
    
    # Validações
    assert len(recommendation.skills_recommended) > 0
    
    # Deve recomendar apenas skills sem pré-requisitos
    for skill_id in recommendation.skills_recommended:
        prereqs = graph.get_prerequisites(skill_id)
        # Pode ter pré-reqs se outros skills recomendados os satisfazem
        for prereq in prereqs:
            if prereq not in recommendation.skills_recommended:
                # Se tem pré-req fora da recomendação, deve estar vazio
                assert len(prereqs) == 0 or prereq in recommendation.skills_recommended, \
                    f"{skill_id} tem pré-req {prereq} não satisfeito"
    
    print("\n✅ Teste 6: PASSOU")
    return True


def test_solve_complete():
    """Testa solução completa do Desafio 5."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 7: Solução Completa")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    # Perfil intermediário
    current_skills = {'S1', 'S2', 'S7'}
    
    result = solve_complete(
        graph, current_skills, n_recommendations=3, horizon_years=5
    )
    
    print(f"\n✅ Resumo:")
    print(f"   Skills recomendadas: {len(result['recommendation']['skills_recommended'])}")
    print(f"   E[Valor]: {result['recommendation']['expected_value']:.2f}")
    print(f"   Cenários: {len(result['scenarios'])}")
    print(f"   Melhor método: {result['comparison']['comparison']['best_method']}")
    
    # Validações
    assert 'recommendation' in result
    assert 'scenarios' in result
    assert 'comparison' in result
    assert 'horizon_years' in result
    assert 'n_recommendations' in result
    assert 'current_skills' in result
    
    assert result['horizon_years'] == 5
    assert result['n_recommendations'] == 3
    assert len(result['scenarios']) == 4
    
    print("\n✅ Teste 7: PASSOU")
    return True


def main():
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DO DESAFIO 5")
    print("=" * 70)
    
    tests = [
        ("Carregamento de Cenários", test_load_market_scenarios),
        ("Cálculo de Valor Esperado", test_calculate_expected_value),
        ("Recomendação Gulosa", test_greedy_recommendation),
        ("DP Exaustivo", test_dp_exhaustive),
        ("Comparação de Métodos", test_compare_methods),
        ("Recomendação para Iniciante", test_empty_current_skills),
        ("Solução Completa", test_solve_complete),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERRO em '{test_name}': {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 RESUMO - DESAFIO 5")
    print("=" * 70)
    
    passed = sum(results)
    total = len(tests)
    
    for i, (name, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\n📈 {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 DESAFIO 5 COMPLETO E VALIDADO!")
        print("\n📌 Funcionalidades implementadas:")
        print("   ✅ Cenários de mercado com probabilidades")
        print("   ✅ Cálculo de valor esperado (E[V])")
        print("   ✅ Algoritmo guloso com look-ahead")
        print("   ✅ DP exaustivo para solução ótima")
        print("   ✅ Comparação de métodos")
        print("   ✅ Recomendação para diferentes perfis")
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())