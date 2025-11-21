"""
Testes para o Desafio 3 - Pivô Mais Rápido (Guloso vs Ótimo)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desafio3_greedy import (
    greedy_selection,
    exhaustive_search,
    create_counterexample,
    compare_greedy_vs_optimal,
    analyze_complexity,
    solve_complete,
    BASIC_SKILLS,
    MIN_ADAPTABILITY_TARGET
)
from src.graph_structures import build_graph_from_file
from src.config import SKILLS_DATASET_FILE


def test_greedy_selection():
    """Testa algoritmo guloso."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 1: Algoritmo Guloso")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    solution = greedy_selection(graph, BASIC_SKILLS, MIN_ADAPTABILITY_TARGET)
    
    print(f"\n✅ Solução Gulosa:")
    print(f"   Skills: {' + '.join(solution.skills_selected)}")
    print(f"   Valor: {solution.total_value}")
    print(f"   Tempo: {solution.total_time}h")
    print(f"   Atinge meta: {'✅' if solution.meets_target() else '❌'}")
    
    # Validações
    assert solution.total_value > 0
    assert solution.total_time > 0
    assert len(solution.skills_selected) > 0
    assert solution.algorithm == 'Greedy'
    
    print("\n✅ Teste 1: PASSOU")
    return True


def test_exhaustive_search():
    """Testa busca exaustiva."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 2: Busca Exaustiva")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    solution = exhaustive_search(graph, BASIC_SKILLS, MIN_ADAPTABILITY_TARGET)
    
    print(f"\n✅ Solução Ótima:")
    print(f"   Skills: {' + '.join(solution.skills_selected)}")
    print(f"   Valor: {solution.total_value}")
    print(f"   Tempo: {solution.total_time}h")
    print(f"   Atinge meta: {'✅' if solution.meets_target() else '❌'}")
    
    # Validações
    assert solution.total_value > 0
    assert solution.total_time > 0
    assert len(solution.skills_selected) > 0
    assert solution.algorithm == 'Exhaustive'
    
    print("\n✅ Teste 2: PASSOU")
    return True


def test_create_counterexample():
    """Testa criação de contraexemplo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 3: Contraexemplo")
    print("=" * 70)
    
    counterexample = create_counterexample()
    
    print(f"\n{counterexample['explanation']}")
    
    # Validações
    assert 'greedy' in counterexample
    assert 'optimal' in counterexample
    assert counterexample['greedy']['meets_target']
    assert counterexample['optimal']['meets_target']
    
    # O contraexemplo deve mostrar que guloso não é ótimo
    print(f"\n✅ Guloso é ótimo: {'❌ NÃO' if not counterexample['greedy_is_optimal'] else '✅ SIM'}")
    
    print("\n✅ Teste 3: PASSOU")
    return True


def test_compare_greedy_vs_optimal():
    """Testa comparação entre guloso e ótimo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 4: Comparação Guloso vs Ótimo")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    result = compare_greedy_vs_optimal(graph, BASIC_SKILLS, MIN_ADAPTABILITY_TARGET)
    
    print(f"\n📊 Comparação:")
    print(f"   Guloso: {result['greedy']['total_value']} ({result['greedy']['total_time']}h)")
    print(f"   Ótimo: {result['optimal']['total_value']} ({result['optimal']['total_time']}h)")
    print(f"   Guloso é ótimo: {'✅' if result['comparison']['greedy_is_optimal'] else '❌'}")
    
    # Validações
    assert 'greedy' in result
    assert 'optimal' in result
    assert 'comparison' in result
    
    print("\n✅ Teste 4: PASSOU")
    return True


def test_analyze_complexity():
    """Testa análise de complexidade."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 5: Análise de Complexidade")
    print("=" * 70)
    
    complexity = analyze_complexity()
    
    print(f"\n⏱️  Complexidade:")
    print(f"   Guloso: {complexity['greedy']['time_complexity']}")
    print(f"   Exaustivo: {complexity['exhaustive']['time_complexity']}")
    
    print(f"\n📊 Para n=5:")
    print(f"   Guloso: {complexity['comparison']['n=5']['greedy']}")
    print(f"   Exaustivo: {complexity['comparison']['n=5']['exhaustive']}")
    print(f"   Razão: {complexity['comparison']['n=5']['ratio']}")
    
    # Validações
    assert complexity['greedy']['time_complexity'] == 'O(n log n)'
    assert complexity['exhaustive']['time_complexity'] == 'O(2^n × n)'
    
    print("\n✅ Teste 5: PASSOU")
    return True


def test_solve_complete():
    """Testa solução completa."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 6: Solução Completa")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    result = solve_complete(graph, BASIC_SKILLS, MIN_ADAPTABILITY_TARGET)
    
    print(f"\n✅ Resumo:")
    print(f"   Guloso: {result['greedy']['total_value']} ({result['greedy']['total_time']}h)")
    print(f"   Ótimo: {result['optimal']['total_value']} ({result['optimal']['total_time']}h)")
    print(f"   Diferença: {result['comparison']['value_difference']}")
    
    # Validações
    assert 'greedy' in result
    assert 'optimal' in result
    assert 'comparison' in result
    assert 'counterexample' in result
    assert 'complexity_analysis' in result
    
    print("\n✅ Teste 6: PASSOU")
    return True


def main():
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DO DESAFIO 3")
    print("=" * 70)
    
    tests = [
        ("Algoritmo Guloso", test_greedy_selection),
        ("Busca Exaustiva", test_exhaustive_search),
        ("Contraexemplo", test_create_counterexample),
        ("Comparação", test_compare_greedy_vs_optimal),
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
    print("📊 RESUMO - DESAFIO 3")
    print("=" * 70)
    
    passed = sum(results)
    total = len(tests)
    
    for i, (name, _) in enumerate(tests):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")
    
    print(f"\n📈 {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 DESAFIO 3 COMPLETO E VALIDADO!")
        return 0
    return 1


if __name__ == '__main__':
    sys.exit(main())