"""
Script de teste rápido para graph_structures.py
Valida as funcionalidades básicas da classe SkillGraph
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_structures import (
    SkillGraph,
    build_graph_from_file,
    print_graph_summary
)
from src.config import SKILLS_DATASET_FILE


def test_basic_operations():
    """Testa operações básicas do grafo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 1: Operações Básicas")
    print("=" * 70)
    
    graph = SkillGraph()
    
    # Adiciona nós
    graph.add_node('S1', {
        'nome': 'Python',
        'tempo_horas': 80,
        'valor': 3,
        'complexidade': 4,
        'pre_requisitos': []
    })
    
    graph.add_node('S3', {
        'nome': 'Algoritmos',
        'tempo_horas': 100,
        'valor': 7,
        'complexidade': 8,
        'pre_requisitos': ['S1']
    })
    
    print("✅ Nós adicionados: S1, S3")
    
    # Adiciona aresta
    graph.add_edge('S1', 'S3', weight=80)
    print("✅ Aresta adicionada: S1 → S3")
    
    # Testa métodos
    assert 'S1' in graph
    assert 'S3' in graph
    print("✅ Verificação de nós: OK")
    
    neighbors = graph.get_neighbors('S1')
    assert neighbors == ['S3']
    print(f"✅ Vizinhos de S1: {neighbors}")
    
    prereqs = graph.get_prerequisites('S3')
    assert prereqs == ['S1']
    print(f"✅ Pré-requisitos de S3: {prereqs}")
    
    metadata = graph.get_metadata('S1')
    assert metadata['nome'] == 'Python'
    print(f"✅ Metadata de S1: {metadata['nome']}")
    
    print("\n✅ Teste 1: PASSOU")
    return True


def test_topological_sort():
    """Testa ordenação topológica."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 2: Ordenação Topológica (Kahn's Algorithm)")
    print("=" * 70)
    
    graph = SkillGraph()
    
    # Cria um pequeno grafo: S1 → S3 → S4
    #                           ↓
    #                           S8
    graph.add_node('S1', {'nome': 'Python', 'tempo_horas': 80, 'valor': 3, 'complexidade': 4, 'pre_requisitos': []})
    graph.add_node('S3', {'nome': 'Algoritmos', 'tempo_horas': 100, 'valor': 7, 'complexidade': 8, 'pre_requisitos': ['S1']})
    graph.add_node('S4', {'nome': 'ML', 'tempo_horas': 120, 'valor': 8, 'complexidade': 9, 'pre_requisitos': ['S1', 'S3']})
    graph.add_node('S8', {'nome': 'APIs', 'tempo_horas': 90, 'valor': 6, 'complexidade': 6, 'pre_requisitos': ['S1']})
    
    graph.add_edge('S1', 'S3')
    graph.add_edge('S1', 'S8')
    graph.add_edge('S1', 'S4')
    graph.add_edge('S3', 'S4')
    
    print("Grafo criado:")
    print("   S1 → S3 → S4")
    print("   S1 → S8")
    print("   S1 → S4")
    
    topo_order = graph.topological_sort()
    print(f"\n✅ Ordenação topológica: {' → '.join(topo_order)}")
    
    # Valida propriedade: S1 antes de S3, S3 antes de S4
    assert topo_order.index('S1') < topo_order.index('S3')
    assert topo_order.index('S3') < topo_order.index('S4')
    assert topo_order.index('S1') < topo_order.index('S8')
    
    print("✅ Propriedades topológicas validadas")
    print("\n✅ Teste 2: PASSOU")
    return True


def test_load_from_file():
    """Testa carregamento do dataset completo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 3: Carregamento do Dataset Completo")
    print("=" * 70)
    
    try:
        graph = build_graph_from_file(SKILLS_DATASET_FILE)
        print(f"✅ Dataset carregado: {len(graph)} habilidades")
        
        # Valida que todas as 12 habilidades foram carregadas
        expected_skills = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'H10', 'H11', 'H12']
        for skill_id in expected_skills:
            assert skill_id in graph
        print(f"✅ Todas as 12 habilidades presentes: {', '.join(expected_skills)}")
        
        # Testa ordenação topológica
        topo_order = graph.topological_sort()
        print(f"✅ Ordenação topológica bem-sucedida: {len(topo_order)} habilidades")
        
        # Valida algumas propriedades esperadas
        # S1 deve vir antes de S3, S4, S8
        assert topo_order.index('S1') < topo_order.index('S3')
        assert topo_order.index('S1') < topo_order.index('S4')
        assert topo_order.index('S1') < topo_order.index('S8')
        
        # S3 deve vir antes de S4
        assert topo_order.index('S3') < topo_order.index('S4')
        
        # S4 deve vir antes de S6
        assert topo_order.index('S4') < topo_order.index('S6')
        
        print("✅ Propriedades de pré-requisitos validadas")
        
        # Habilidades básicas
        basic_skills = graph.get_basic_skills()
        print(f"\n📋 Habilidades básicas (sem pré-requisitos): {len(basic_skills)}")
        for skill_id in sorted(basic_skills):
            metadata = graph.get_metadata(skill_id)
            print(f"   • {skill_id} - {metadata['nome']}")
        
        print("\n✅ Teste 3: PASSOU")
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def test_graph_properties():
    """Testa propriedades do grafo."""
    print("\n" + "=" * 70)
    print("🧪 TESTE 4: Propriedades do Grafo")
    print("=" * 70)
    
    graph = build_graph_from_file(SKILLS_DATASET_FILE)
    
    # Testa graus de entrada e saída
    print("\n📊 Graus de entrada e saída:")
    
    # S1 deve ter muitos dependentes
    s1_out_degree = graph.get_out_degree('S1')
    print(f"   • S1 (Python): {s1_out_degree} dependentes")
    assert s1_out_degree > 0
    
    # S6 não deve ter dependentes (objetivo final)
    s6_out_degree = graph.get_out_degree('S6')
    print(f"   • S6 (IA Generativa): {s6_out_degree} dependentes")
    assert s6_out_degree == 0
    
    # S9 deve ter pré-requisitos (S7 e S8)
    s9_in_degree = graph.get_in_degree('S9')
    print(f"   • S9 (DevOps): {s9_in_degree} pré-requisitos")
    assert s9_in_degree == 2
    
    print("\n✅ Teste 4: PASSOU")
    return True


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DE graph_structures.py")
    print("=" * 70)
    
    tests = [
        ("Operações Básicas", test_basic_operations),
        ("Ordenação Topológica", test_topological_sort),
        ("Carregamento do Dataset", test_load_from_file),
        ("Propriedades do Grafo", test_graph_properties)
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
    
    # Resumo final
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES")
    print("=" * 70)
    
    total_tests = len(tests)
    passed_tests = sum(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSOU" if results[i] else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Resultados: {passed_tests}/{total_tests} testes passaram")
    
    if passed_tests == total_tests:
        print("\n🎉 FASE 1.1 COMPLETA E VALIDADA!")
        print("✅ graph_structures.py funcionando perfeitamente")
        print("✅ Pronto para FASE 1.2: graph_validation.py")
        
        # Mostra resumo do grafo
        print("\n")
        graph = build_graph_from_file(SKILLS_DATASET_FILE)
        print_graph_summary(graph)
        
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} teste(s) falharam.")
        return 1


if __name__ == '__main__':
    sys.exit(main())