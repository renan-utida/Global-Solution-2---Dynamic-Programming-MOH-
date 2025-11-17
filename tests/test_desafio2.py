"""
Testes para o Desafio 2 - Verificação Crítica (120 Permutações)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desafio2_permutations import (
    generate_all_permutations,
    calculate_order_cost,
    validate_before_compute,
    calculate_all_permutations_costs,
    find_top_n_orders,
    analyze_heuristics,
    solve_complete,
    CRITICAL_SKILLS
)
from src.graph_structures import build_graph_from_file
from src.config import SKILLS_DATASET_FILE


def test_generate_permutations():
    """Testa geração de permutações."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 1: Geração de Permutações")
    print("=" * 70)
    
    # Teste com 3 skills (3! = 6)
    test_skills = ['S3', 'S5', 'S7']
    perms = generate_all_permutations(test_skills)
    
    print(f"\nSkills: {test_skills}")
    print(f"Permutações geradas: {len(perms)}")
    print(f"Primeiras 3:")
    for i, perm in enumerate(perms[:3], 1):
        print(f"   {i}. {' → '.join(perm)}")
    
    assert len(perms) == 6, f"Esperado 6 permutações, obtido {len(perms)}"
    
    # Teste com 5 skills (5! = 120)
    perms_5 = generate_all_permutations(CRITICAL_SKILLS)
    assert len(perms_5) == 120, f"Esperado 120 permutações, obtido {len(perms_5)}"
    
    print(f"\n✅ 5 skills críticas: {len(perms_5)} permutações")
    print("\n✅ Teste 1: PASSOU")
    return True


def test_calculate_order_cost():
    """Testa cálculo de custo de uma ordem."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 2: Cálculo de Custo de Ordem")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    # Ordem de teste: S7 primeiro (sem pré-reqs)
    order = ['S7', 'S3', 'S8', 'S5', 'S9']
    
    print(f"\nOrdem de teste: {' → '.join(order)}")
    
    cost = calculate_order_cost(order, graph)
    
    print(f"\n📊 Resultado:")
    print(f"   Custo total: {cost.total_cost:.0f}h")
    print(f"   Tempo aquisição: {cost.acquisition_time:.0f}h")
    print(f"   Tempo espera: {cost.waiting_time:.0f}h")
    
    # Validações
    assert cost.total_cost > 0
    assert cost.acquisition_time > 0
    assert len(cost.details) == 5
    
    print("\n✅ Teste 2: PASSOU")
    return True


def test_validate_before_compute():
    """Testa validação crítica do grafo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 3: Validação Crítica")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    try:
        validation = validate_before_compute(graph)
        print("\n✅ Validação passou")
        assert validation['valid'] == True
    except ValueError as e:
        print(f"\n❌ Validação falhou: {e}")
        return False
    
    print("\n✅ Teste 3: PASSOU")
    return True


def test_solve_complete():
    """Testa solução completa do Desafio 2."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 4: Solução Completa")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    result = solve_complete(graph)
    
    print(f"\n✅ Resumo:")
    print(f"   Total de permutações: {len(result['all_costs'])}")
    print(f"   Melhor custo: {result['statistics']['best_cost']:.0f}h")
    print(f"   Pior custo: {result['statistics']['worst_cost']:.0f}h")
    print(f"   Custo médio: {result['statistics']['avg_all']:.0f}h")
    
    # Validações
    assert len(result['all_costs']) == 120
    assert len(result['top_3_best']) == 3
    assert len(result['top_3_worst']) == 3
    assert 'heuristics' in result
    
    print("\n✅ Teste 4: PASSOU")
    return True


def main():
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DO DESAFIO 2")
    print("=" * 70)
    
    tests = [
        ("Geração de Permutações", test_generate_permutations),
        ("Cálculo de Custo", test_calculate_order_cost),
        ("Validação Crítica", test_validate_before_compute),
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
    print("📊 RESUMO - DESAFIO 2")
    print("=" * 70)
    
    passed = sum(results)
    total = len(tests)
    
    for i, (name, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\n📈 {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 DESAFIO 2 COMPLETO E VALIDADO!")
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())