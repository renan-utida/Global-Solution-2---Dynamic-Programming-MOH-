"""
Testes para o Desafio 4 - Trilhas Paralelas (Merge Sort)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desafio4_sorting import (
    merge_sort,
    divide_into_sprints,
    sort_skills_merge,
    sort_skills_native,
    compare_with_native_sort,
    analyze_complexity,
    solve_complete
)
from src.graph_structures import build_graph_from_file
from src.config import SKILLS_DATASET_FILE


def test_merge_sort_basic():
    """Testa Merge Sort básico."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 1: Merge Sort Básico")
    print("=" * 70)
    
    # Teste com dados simples
    test_data = [
        {'id': 'A', 'complexidade': 5},
        {'id': 'B', 'complexidade': 2},
        {'id': 'C', 'complexidade': 8},
        {'id': 'D', 'complexidade': 1}
    ]
    
    sorted_data = merge_sort(test_data, key='complexidade')
    
    print(f"\nAntes:  {[d['complexidade'] for d in test_data]}")
    print(f"Depois: {[d['complexidade'] for d in sorted_data]}")
    
    # Valida ordenação
    for i in range(len(sorted_data) - 1):
        assert sorted_data[i]['complexidade'] <= sorted_data[i+1]['complexidade']
    
    print("\n✅ Teste 1: PASSOU")
    return True


def test_divide_into_sprints():
    """Testa divisão em sprints."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 2: Divisão em Sprints")
    print("=" * 70)
    
    # Cria lista de 12 habilidades
    skills = [{'id': f'S{i}', 'complexidade': i} for i in range(1, 13)]
    
    sprint_a, sprint_b = divide_into_sprints(skills)
    
    print(f"\nTotal: {len(skills)} habilidades")
    print(f"Sprint A: {len(sprint_a)} habilidades")
    print(f"Sprint B: {len(sprint_b)} habilidades")
    
    # Validações
    assert len(sprint_a) == 6
    assert len(sprint_b) == 6
    assert len(sprint_a) + len(sprint_b) == len(skills)
    
    print("\n✅ Teste 2: PASSOU")
    return True


def test_sort_skills_merge():
    """Testa ordenação completa com Merge Sort."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 3: Ordenação Completa")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    result = sort_skills_merge(graph, key='complexidade')
    
    print(f"\n✅ Resultado:")
    print(f"   Algoritmo: {result.algorithm}")
    print(f"   Tempo: {result.execution_time * 1000:.3f} ms")
    print(f"   Comparações: {result.comparisons}")
    print(f"   Skills ordenadas: {len(result.sorted_skills)}")
    print(f"   Sprint A: {len(result.sprint_a)}")
    print(f"   Sprint B: {len(result.sprint_b)}")
    
    # Validações
    assert len(result.sorted_skills) == 12
    assert len(result.sprint_a) == 6
    assert len(result.sprint_b) == 6
    assert result.comparisons > 0
    
    # Valida ordenação
    for i in range(len(result.sorted_skills) - 1):
        assert result.sorted_skills[i]['complexidade'] <= result.sorted_skills[i+1]['complexidade']
    
    print("\n✅ Teste 3: PASSOU")
    return True


def test_compare_with_native():
    """Testa comparação com sort nativo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 4: Comparação com Sort Nativo")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    result = compare_with_native_sort(graph, key='complexidade')
    
    print(f"\n📊 Comparação:")
    print(f"   Merge Sort:  {result['merge_sort']['execution_time'] * 1000:.3f} ms")
    print(f"   Native Sort: {result['native_sort']['execution_time'] * 1000:.3f} ms")
    print(f"   Razão: {result['comparison']['time_ratio']:.2f}x")
    print(f"   Resultados idênticos: {'✅' if result['comparison']['results_match'] else '❌'}")
    
    # Validações
    assert result['comparison']['results_match'] == True
    assert 'merge_sort' in result
    assert 'native_sort' in result
    
    print("\n✅ Teste 4: PASSOU")
    return True


def test_analyze_complexity():
    """Testa análise de complexidade."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 5: Análise de Complexidade")
    print("=" * 70)
    
    complexity = analyze_complexity()
    
    print(f"\n⏱️  Complexidade Merge Sort:")
    print(f"   Melhor:  {complexity['merge_sort']['time_complexity']['best']}")
    print(f"   Médio:   {complexity['merge_sort']['time_complexity']['average']}")
    print(f"   Pior:    {complexity['merge_sort']['time_complexity']['worst']}")
    
    # Validações
    assert complexity['merge_sort']['time_complexity']['best'] == 'O(n log n)'
    assert complexity['merge_sort']['time_complexity']['average'] == 'O(n log n)'
    assert complexity['merge_sort']['time_complexity']['worst'] == 'O(n log n)'
    
    print("\n✅ Teste 5: PASSOU")
    return True


def test_solve_complete():
    """Testa solução completa."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 6: Solução Completa")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    result = solve_complete(graph, key='complexidade')
    
    print(f"\n✅ Resumo:")
    print(f"   Merge Sort: {result['merge_sort_result']['execution_time'] * 1000:.3f} ms")
    print(f"   Native Sort: {result['native_sort_result']['execution_time'] * 1000:.3f} ms")
    print(f"   Razão: {result['comparison']['time_ratio']:.2f}x")
    print(f"   Resultados OK: {'✅' if result['comparison']['results_match'] else '❌'}")
    
    # Validações
    assert 'merge_sort_result' in result
    assert 'native_sort_result' in result
    assert 'comparison' in result
    assert 'complexity_analysis' in result
    assert result['comparison']['results_match'] == True
    
    print("\n✅ Teste 6: PASSOU")
    return True


def main():
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DO DESAFIO 4")
    print("=" * 70)
    
    tests = [
        ("Merge Sort Básico", test_merge_sort_basic),
        ("Divisão em Sprints", test_divide_into_sprints),
        ("Ordenação Completa", test_sort_skills_merge),
        ("Comparação com Native", test_compare_with_native),
        ("Análise de Complexidade", test_analyze_complexity),
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
    print("📊 RESUMO - DESAFIO 4")
    print("=" * 70)
    
    passed = sum(results)
    total = len(tests)
    
    for i, (name, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\n📈 {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 DESAFIO 4 COMPLETO E VALIDADO!")
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())