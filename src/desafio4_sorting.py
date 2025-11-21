"""
Desafio 4 - Trilhas Paralelas

Objetivo: Ordenar as 12 habilidades por Complexidade C usando Merge Sort
implementado DO ZERO (sem usar sorted() ou .sort()).

Após ordenação, dividir em:
- Sprint A: habilidades 1-6 (menos complexas)
- Sprint B: habilidades 7-12 (mais complexas)

Análise inclui:
- Complexidade teórica (melhor, médio, pior caso)
- Comparação experimental com sort nativo
- Justificativa da escolha do algoritmo

Por que Merge Sort?
1. O(n log n) GARANTIDO em todos os casos
2. Estável (mantém ordem relativa de elementos iguais)
3. Ótimo para dados externos (divide e conquista)
4. Previsível (sem pior caso O(n²) como Quick Sort)

Complexidade:
    Tempo: O(n log n) - melhor, médio e pior
    Espaço: O(n) - precisa de array auxiliar
"""

from typing import List, Dict, Any, Callable, Tuple
import time
from dataclasses import dataclass

from src.graph_structures import SkillGraph, build_graph_from_file
from src.decorators import measure_performance


@dataclass
class SortedResult:
    """
    Resultado de ordenação.
    
    Attributes:
        sorted_skills: Lista de habilidades ordenadas
        algorithm: Nome do algoritmo usado
        execution_time: Tempo de execução (segundos)
        comparisons: Número de comparações realizadas
        sprint_a: Habilidades do Sprint A (1-6)
        sprint_b: Habilidades do Sprint B (7-12)
    """
    sorted_skills: List[Dict[str, Any]]
    algorithm: str
    execution_time: float
    comparisons: int
    sprint_a: List[Dict[str, Any]]
    sprint_b: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'sorted_skills': self.sorted_skills,
            'algorithm': self.algorithm,
            'execution_time': self.execution_time,
            'comparisons': self.comparisons,
            'sprint_a': self.sprint_a,
            'sprint_b': self.sprint_b
        }


class ComparisonCounter:
    """Contador de comparações para análise."""
    
    def __init__(self):
        self.count = 0
    
    def reset(self):
        """Reseta o contador."""
        self.count = 0
    
    def compare(self, a: Any, b: Any, key: Callable = None) -> bool:
        """
        Compara dois elementos e incrementa contador.
        
        Args:
            a: Primeiro elemento
            b: Segundo elemento
            key: Função para extrair chave de comparação
        
        Returns:
            bool: True se a <= b
        """
        self.count += 1
        
        if key:
            return key(a) <= key(b)
        return a <= b


# Instância global do contador
comparison_counter = ComparisonCounter()


def merge_sort(
    arr: List[Dict[str, Any]],
    key: str = 'complexidade',
    counter: ComparisonCounter = None
) -> List[Dict[str, Any]]:
    """
    Implementa Merge Sort DO ZERO (sem usar sorted() ou .sort()).
    
    Algoritmo:
        1. Divide: Divide o array ao meio recursivamente
        2. Conquista: Ordena cada metade recursivamente
        3. Combina: Mescla as duas metades ordenadas
    
    Args:
        arr: Lista de dicionários a ordenar
        key: Chave do dicionário para ordenação
        counter: Contador de comparações (opcional)
    
    Returns:
        List[Dict[str, Any]]: Lista ordenada
    
    Complexity:
        Tempo: O(n log n) - melhor, médio e pior
        Espaço: O(n) - array auxiliar
    
    Examples:
        >>> skills = [{'id': 'S1', 'complexidade': 5}, {'id': 'S2', 'complexidade': 3}]
        >>> sorted_skills = merge_sort(skills, key='complexidade')
        >>> sorted_skills[0]['complexidade']
        3
    """
    # Caso base: array com 0 ou 1 elemento já está ordenado
    if len(arr) <= 1:
        return arr.copy()
    
    # Divide ao meio
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Conquista: ordena cada metade recursivamente
    left_sorted = merge_sort(left_half, key, counter)
    right_sorted = merge_sort(right_half, key, counter)
    
    # Combina: mescla as duas metades ordenadas
    return merge(left_sorted, right_sorted, key, counter)


def merge(
    left: List[Dict[str, Any]],
    right: List[Dict[str, Any]],
    key: str,
    counter: ComparisonCounter = None
) -> List[Dict[str, Any]]:
    """
    Mescla duas listas ordenadas em uma única lista ordenada.
    
    Args:
        left: Lista ordenada à esquerda
        right: Lista ordenada à direita
        key: Chave para comparação
        counter: Contador de comparações
    
    Returns:
        List[Dict[str, Any]]: Lista mesclada e ordenada
    
    Complexity:
        O(n) onde n = len(left) + len(right)
    """
    result = []
    i = j = 0
    
    # Mescla enquanto houver elementos em ambas as listas
    while i < len(left) and j < len(right):
        # Incrementa contador de comparações
        if counter:
            counter.count += 1
        
        if left[i][key] <= right[j][key]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Adiciona elementos restantes (se houver)
    while i < len(left):
        result.append(left[i])
        i += 1
    
    while j < len(right):
        result.append(right[j])
        j += 1
    
    return result


def divide_into_sprints(
    sorted_skills: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Divide habilidades ordenadas em dois sprints.
    
    Sprint A: habilidades 1-6 (menos complexas)
    Sprint B: habilidades 7-12 (mais complexas)
    
    Args:
        sorted_skills: Lista de habilidades ordenadas por complexidade
    
    Returns:
        Tuple[List, List]: (sprint_a, sprint_b)
    
    Examples:
        >>> skills = [...12 habilidades ordenadas...]
        >>> sprint_a, sprint_b = divide_into_sprints(skills)
        >>> len(sprint_a)
        6
        >>> len(sprint_b)
        6
    """
    mid = len(sorted_skills) // 2
    sprint_a = sorted_skills[:mid]
    sprint_b = sorted_skills[mid:]
    
    return sprint_a, sprint_b


def sort_skills_merge(
    graph: SkillGraph,
    key: str = 'complexidade'
) -> SortedResult:
    """
    Ordena habilidades usando Merge Sort e divide em sprints.
    
    Args:
        graph: Grafo de habilidades
        key: Chave para ordenação
    
    Returns:
        SortedResult: Resultado completo da ordenação
    """
    # Extrai lista de habilidades com metadados
    skills_list = []
    for skill_id in graph.nodes:
        metadata = graph.get_metadata(skill_id)
        skills_list.append({
            'skill_id': skill_id,
            'nome': metadata['nome'],
            'complexidade': metadata['complexidade'],
            'tempo_horas': metadata['tempo_horas'],
            'valor': metadata['valor']
        })
    
    # Reseta contador de comparações
    comparison_counter.reset()
    
    # Mede tempo de execução
    start_time = time.perf_counter()
    sorted_skills = merge_sort(skills_list, key=key, counter=comparison_counter)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    # Divide em sprints
    sprint_a, sprint_b = divide_into_sprints(sorted_skills)
    
    return SortedResult(
        sorted_skills=sorted_skills,
        algorithm='Merge Sort',
        execution_time=execution_time,
        comparisons=comparison_counter.count,
        sprint_a=sprint_a,
        sprint_b=sprint_b
    )


def sort_skills_native(
    graph: SkillGraph,
    key: str = 'complexidade'
) -> SortedResult:
    """
    Ordena habilidades usando sort nativo do Python (baseline).
    
    Args:
        graph: Grafo de habilidades
        key: Chave para ordenação
    
    Returns:
        SortedResult: Resultado completo da ordenação
    """
    # Extrai lista de habilidades
    skills_list = []
    for skill_id in graph.nodes:
        metadata = graph.get_metadata(skill_id)
        skills_list.append({
            'skill_id': skill_id,
            'nome': metadata['nome'],
            'complexidade': metadata['complexidade'],
            'tempo_horas': metadata['tempo_horas'],
            'valor': metadata['valor']
        })
    
    # Mede tempo de execução
    start_time = time.perf_counter()
    sorted_skills = sorted(skills_list, key=lambda x: x[key])
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    
    # Divide em sprints
    sprint_a, sprint_b = divide_into_sprints(sorted_skills)
    
    return SortedResult(
        sorted_skills=sorted_skills,
        algorithm='Python Native Sort (Timsort)',
        execution_time=execution_time,
        comparisons=-1,  # Não contamos para sort nativo
        sprint_a=sprint_a,
        sprint_b=sprint_b
    )


def compare_with_native_sort(
    graph: SkillGraph,
    key: str = 'complexidade'
) -> Dict[str, Any]:
    """
    Compara Merge Sort implementado com sort nativo do Python.
    
    Args:
        graph: Grafo de habilidades
        key: Chave para ordenação
    
    Returns:
        Dict com comparação detalhada
    """
    # Ordena com Merge Sort
    merge_result = sort_skills_merge(graph, key)
    
    # Ordena com sort nativo
    native_result = sort_skills_native(graph, key)
    
    # Compara resultados
    return {
        'merge_sort': merge_result.to_dict(),
        'native_sort': native_result.to_dict(),
        'comparison': {
            'time_ratio': merge_result.execution_time / native_result.execution_time if native_result.execution_time > 0 else float('inf'),
            'merge_time': merge_result.execution_time,
            'native_time': native_result.execution_time,
            'merge_comparisons': merge_result.comparisons,
            'results_match': merge_result.sorted_skills == native_result.sorted_skills
        }
    }


def analyze_complexity() -> Dict[str, Any]:
    """
    Analisa a complexidade teórica do Merge Sort.
    
    Returns:
        Dict com análise de complexidade
    """
    return {
        'merge_sort': {
            'time_complexity': {
                'best': 'O(n log n)',
                'average': 'O(n log n)',
                'worst': 'O(n log n)',
                'explanation': (
                    'Merge Sort sempre divide o array ao meio (log n níveis) '
                    'e mescla em O(n) por nível, resultando em O(n log n) '
                    'independentemente da distribuição dos dados.'
                )
            },
            'space_complexity': {
                'auxiliary': 'O(n)',
                'explanation': (
                    'Precisa de um array auxiliar de tamanho n para mesclar '
                    'as sublistas ordenadas.'
                )
            },
            'stability': 'Estável',
            'adaptive': 'Não adaptativo (tempo fixo independente da ordem inicial)',
            'in_place': 'Não (requer espaço auxiliar O(n))'
        },
        'timsort': {
            'description': 'Sort nativo do Python (híbrido de Merge + Insertion)',
            'time_complexity': {
                'best': 'O(n)',
                'average': 'O(n log n)',
                'worst': 'O(n log n)'
            },
            'space_complexity': {
                'auxiliary': 'O(n)'
            },
            'stability': 'Estável',
            'adaptive': 'Adaptativo (aproveita ordem parcial)'
        },
        'justification': (
            'Escolhemos Merge Sort porque:\n'
            '1. Complexidade O(n log n) GARANTIDA (sem pior caso O(n²))\n'
            '2. Estável (mantém ordem relativa)\n'
            '3. Previsível (sempre O(n log n))\n'
            '4. Didático (algoritmo clássico de dividir e conquistar)\n'
            '5. Bom para dados grandes (divide e conquista)\n\n'
            'Quick Sort seria mais rápido na prática (melhor constante), '
            'mas tem pior caso O(n²) se pivô mal escolhido.'
        )
    }


@measure_performance
def solve_complete(
    graph: SkillGraph,
    key: str = 'complexidade'
) -> Dict[str, Any]:
    """
    Resolve o Desafio 4 COMPLETO:
    1. Ordena com Merge Sort implementado
    2. Divide em Sprint A e B
    3. Compara com sort nativo
    4. Analisa complexidade
    
    Esta é a função principal do Desafio 4.
    
    Args:
        graph: Grafo de habilidades
        key: Chave para ordenação (padrão: complexidade)
    
    Returns:
        Dict com TODOS os resultados do Desafio 4
    """
    print("\n" + "=" * 70)
    print("🎯 DESAFIO 4 - TRILHAS PARALELAS")
    print("=" * 70)
    
    print(f"\nObjetivo: Ordenar 12 habilidades por {key.upper()}")
    print("Algoritmo: Merge Sort (implementado do zero)")
    
    # 1. Ordenação com Merge Sort
    print("\n📊 FASE 1: Ordenação com Merge Sort")
    print("-" * 70)
    
    merge_result = sort_skills_merge(graph, key)
    
    print(f"\n✅ Ordenação completa:")
    print(f"   Algoritmo: {merge_result.algorithm}")
    print(f"   Tempo: {merge_result.execution_time * 1000:.3f} ms")
    print(f"   Comparações: {merge_result.comparisons}")
    
    print(f"\n📝 Habilidades ordenadas por {key}:")
    for i, skill in enumerate(merge_result.sorted_skills, 1):
        print(f"   {i:2d}. {skill['skill_id']:4s} - {skill['nome']:40s} (C={skill['complexidade']})")
    
    # 2. Divisão em Sprints
    print("\n📊 FASE 2: Divisão em Sprints")
    print("-" * 70)
    
    print(f"\n🏃 Sprint A (habilidades 1-6 - menos complexas):")
    for i, skill in enumerate(merge_result.sprint_a, 1):
        print(f"   {i}. {skill['skill_id']} - {skill['nome']} (C={skill['complexidade']})")
    
    print(f"\n🏃 Sprint B (habilidades 7-12 - mais complexas):")
    for i, skill in enumerate(merge_result.sprint_b, 1):
        print(f"   {i+6}. {skill['skill_id']} - {skill['nome']} (C={skill['complexidade']})")
    
    # 3. Comparação com sort nativo
    print("\n📊 FASE 3: Comparação com Sort Nativo")
    print("-" * 70)
    
    native_result = sort_skills_native(graph, key)
    
    print(f"\n📈 Benchmark:")
    print(f"   Merge Sort:  {merge_result.execution_time * 1000:.3f} ms ({merge_result.comparisons} comparações)")
    print(f"   Native Sort: {native_result.execution_time * 1000:.3f} ms (Timsort)")
    
    ratio = merge_result.execution_time / native_result.execution_time if native_result.execution_time > 0 else float('inf')
    print(f"   Razão: {ratio:.2f}x {'mais lento' if ratio > 1 else 'mais rápido'}")
    
    results_match = merge_result.sorted_skills == native_result.sorted_skills
    print(f"   Resultados idênticos: {'✅ SIM' if results_match else '❌ NÃO'}")
    
    # 4. Análise de Complexidade
    print("\n📊 FASE 4: Análise de Complexidade")
    print("-" * 70)
    
    complexity = analyze_complexity()
    
    print(f"\n⏱️  Complexidade Temporal do Merge Sort:")
    print(f"   Melhor caso:  {complexity['merge_sort']['time_complexity']['best']}")
    print(f"   Caso médio:   {complexity['merge_sort']['time_complexity']['average']}")
    print(f"   Pior caso:    {complexity['merge_sort']['time_complexity']['worst']}")
    
    print(f"\n💾 Complexidade Espacial:")
    print(f"   Auxiliar: {complexity['merge_sort']['space_complexity']['auxiliary']}")
    
    print(f"\n📋 Propriedades:")
    print(f"   Estabilidade: {complexity['merge_sort']['stability']}")
    print(f"   Adaptativo: {complexity['merge_sort']['adaptive']}")
    print(f"   In-place: {complexity['merge_sort']['in_place']}")
    
    print(f"\n💡 Justificativa:")
    for line in complexity['justification'].split('\n'):
        if line.strip():
            print(f"   {line}")
    
    return {
        'merge_sort_result': merge_result.to_dict(),
        'native_sort_result': native_result.to_dict(),
        'comparison': {
            'time_ratio': ratio,
            'results_match': results_match
        },
        'complexity_analysis': complexity
    }


def print_sprint_details(sprint: List[Dict[str, Any]], name: str) -> None:
    """
    Imprime detalhes de um sprint.
    
    Args:
        sprint: Lista de habilidades do sprint
        name: Nome do sprint (A ou B)
    """
    print(f"\n{'=' * 70}")
    print(f"🏃 SPRINT {name}")
    print(f"{'=' * 70}")
    
    total_time = sum(skill['tempo_horas'] for skill in sprint)
    total_value = sum(skill['valor'] for skill in sprint)
    avg_complexity = sum(skill['complexidade'] for skill in sprint) / len(sprint)
    
    print(f"\n📊 Resumo:")
    print(f"   Total de habilidades: {len(sprint)}")
    print(f"   Tempo total: {total_time}h")
    print(f"   Valor total: {total_value}")
    print(f"   Complexidade média: {avg_complexity:.1f}")
    
    print(f"\n📝 Habilidades:")
    for i, skill in enumerate(sprint, 1):
        print(f"   {i}. {skill['skill_id']} - {skill['nome']}")
        print(f"      Tempo: {skill['tempo_horas']}h | Valor: {skill['valor']} | Complexidade: {skill['complexidade']}")
    
    print("=" * 70)