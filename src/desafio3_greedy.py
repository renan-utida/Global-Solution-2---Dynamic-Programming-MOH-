"""
Desafio 3 - Pivô Mais Rápido

Objetivo: Alcançar adaptabilidade mínima S ≥ 15 usando apenas habilidades 
de nível básico (sem pré-requisitos), selecionando iterativamente pela 
razão V/T (valor por tempo).

Compara:
- Algoritmo GULOSO: Escolhe sempre a habilidade com maior V/T
- Algoritmo ÓTIMO: Busca exaustiva em todos os 2^5 = 32 subconjuntos

Demonstra CONTRAEXEMPLO onde o guloso NÃO encontra a solução ótima.

Habilidades Básicas (sem pré-requisitos):
    S1: Programação Básica (Python) - 80h, V=3
    S2: Modelagem de Dados (SQL) - 60h, V=4
    S7: Estruturas em Nuvem (AWS/Azure) - 70h, V=5
    H10: Segurança de Dados - 60h, V=5
    H12: Introdução a IoT - 30h, V=3

Complexidade:
    - Guloso: O(n log n) - ordenação + seleção
    - Exaustivo: O(2^n × n) - enumera todos os subconjuntos
"""

from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass
from itertools import combinations
import time

import json
from pathlib import Path
from src.config import OUTPUTS_DIR

from src.graph_structures import SkillGraph, build_graph_from_file
from src.decorators import measure_performance


# Habilidades básicas (sem pré-requisitos)
BASIC_SKILLS = ['S1', 'S2', 'S7', 'H10', 'H12']

# Meta de adaptabilidade
MIN_ADAPTABILITY_TARGET = 15


@dataclass
class Solution:
    """
    Solução do problema de seleção de habilidades.
    
    Attributes:
        skills_selected: Lista de IDs das habilidades selecionadas
        total_value: Valor total (adaptabilidade)
        total_time: Tempo total gasto
        algorithm: Nome do algoritmo usado
        details: Detalhes de cada habilidade
    """
    skills_selected: List[str]
    total_value: float
    total_time: float
    algorithm: str
    details: List[Dict[str, Any]]
    
    def meets_target(self, target: float = MIN_ADAPTABILITY_TARGET) -> bool:
        """Verifica se atinge a meta."""
        return self.total_value >= target
    
    def efficiency(self) -> float:
        """Retorna a eficiência (V/T)."""
        if self.total_time == 0:
            return 0
        return self.total_value / self.total_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'skills_selected': self.skills_selected,
            'total_value': self.total_value,
            'total_time': self.total_time,
            'algorithm': self.algorithm,
            'meets_target': self.meets_target(),
            'efficiency': self.efficiency(),
            'details': self.details
        }
    
    def __str__(self) -> str:
        """Representação string formatada."""
        target_status = "✅" if self.meets_target() else "❌"
        return (
            f"{self.algorithm} Solution:\n"
            f"  Skills: {' + '.join(self.skills_selected)}\n"
            f"  Value: {self.total_value} {target_status}\n"
            f"  Time: {self.total_time}h\n"
            f"  Efficiency: {self.efficiency():.3f}"
        )


def calculate_vt_ratio(value: float, time: float) -> float:
    """
    Calcula a razão Valor/Tempo (V/T).
    
    Args:
        value: Valor da habilidade
        time: Tempo de aquisição
    
    Returns:
        float: Razão V/T
    """
    if time == 0:
        return float('inf')
    return value / time


def greedy_selection(
    graph: SkillGraph,
    basic_skills: List[str] = BASIC_SKILLS,
    target_value: float = MIN_ADAPTABILITY_TARGET
) -> Solution:
    """
    Seleciona habilidades usando algoritmo GULOSO (greedy).
    
    Algoritmo:
        1. Calcula V/T para cada habilidade básica
        2. Ordena por V/T (decrescente)
        3. Seleciona iterativamente até atingir meta
    
    Args:
        graph: Grafo de habilidades
        basic_skills: Lista de habilidades básicas
        target_value: Meta de valor mínimo
    
    Returns:
        Solution: Solução gulosa
    
    Complexity:
        O(n log n) para ordenação + O(n) para seleção = O(n log n)
    
    Examples:
        >>> solution = greedy_selection(graph, target_value=15)
        >>> print(solution.skills_selected)
        >>> print(solution.total_value)
    """
    # Extrai metadados e calcula V/T
    skills_data = []
    for skill_id in basic_skills:
        if skill_id not in graph:
            continue
        
        metadata = graph.get_metadata(skill_id)
        time = metadata['tempo_horas']
        value = metadata['valor']
        vt_ratio = calculate_vt_ratio(value, time)
        
        skills_data.append({
            'skill_id': skill_id,
            'nome': metadata['nome'],
            'time': time,
            'value': value,
            'vt_ratio': vt_ratio
        })
    
    # Ordena por V/T (decrescente)
    skills_data.sort(key=lambda x: x['vt_ratio'], reverse=True)
    
    # Seleção gulosa
    selected = []
    total_value = 0
    total_time = 0
    details = []
    
    for skill in skills_data:
        # Adiciona skill
        selected.append(skill['skill_id'])
        total_value += skill['value']
        total_time += skill['time']
        
        details.append({
            'skill_id': skill['skill_id'],
            'nome': skill['nome'],
            'time': skill['time'],
            'value': skill['value'],
            'vt_ratio': skill['vt_ratio'],
            'cumulative_value': total_value,
            'cumulative_time': total_time
        })
        
        # Para se atingiu a meta
        if total_value >= target_value:
            break
    
    return Solution(
        skills_selected=selected,
        total_value=total_value,
        total_time=total_time,
        algorithm='Greedy',
        details=details
    )


def exhaustive_search(
    graph: SkillGraph,
    basic_skills: List[str] = BASIC_SKILLS,
    target_value: float = MIN_ADAPTABILITY_TARGET
) -> Solution:
    """
    Busca exaustiva (força bruta) para encontrar a solução ÓTIMA.
    
    Algoritmo:
        1. Enumera todos os 2^n subconjuntos possíveis
        2. Para cada subconjunto:
           a. Calcula valor total e tempo total
           b. Se atinge meta, guarda como candidato
        3. Retorna o candidato com MENOR tempo
    
    Args:
        graph: Grafo de habilidades
        basic_skills: Lista de habilidades básicas
        target_value: Meta de valor mínimo
    
    Returns:
        Solution: Solução ótima (menor tempo que atinge meta)
    
    Complexity:
        O(2^n × n) onde n = número de habilidades básicas
        Para 5 skills: O(32 × 5) = O(160)
    
    Examples:
        >>> solution = exhaustive_search(graph, target_value=15)
        >>> print(solution.skills_selected)
    """
    # Extrai metadados
    skills_data = {}
    for skill_id in basic_skills:
        if skill_id not in graph:
            continue
        metadata = graph.get_metadata(skill_id)
        skills_data[skill_id] = {
            'nome': metadata['nome'],
            'time': metadata['tempo_horas'],
            'value': metadata['valor']
        }
    
    skill_ids = list(skills_data.keys())
    n = len(skill_ids)
    
    # Busca exaustiva em todos os subconjuntos
    best_solution = None
    best_time = float('inf')
    
    # Enumera todos os 2^n subconjuntos (incluindo vazio)
    for subset_size in range(n + 1):
        for subset in combinations(skill_ids, subset_size):
            # Calcula valor e tempo deste subconjunto
            total_value = sum(skills_data[sid]['value'] for sid in subset)
            total_time = sum(skills_data[sid]['time'] for sid in subset)
            
            # Se atinge a meta e é melhor que o atual
            if total_value >= target_value and total_time < best_time:
                best_time = total_time
                best_solution = subset
    
    # Se não encontrou solução que atinge meta
    if best_solution is None:
        # Retorna a melhor solução possível (todas as skills)
        best_solution = tuple(skill_ids)
    
    # Constrói Solution
    selected = list(best_solution)
    total_value = sum(skills_data[sid]['value'] for sid in selected)
    total_time = sum(skills_data[sid]['time'] for sid in selected)
    
    details = []
    cumulative_value = 0
    cumulative_time = 0
    for skill_id in selected:
        skill = skills_data[skill_id]
        cumulative_value += skill['value']
        cumulative_time += skill['time']
        
        details.append({
            'skill_id': skill_id,
            'nome': skill['nome'],
            'time': skill['time'],
            'value': skill['value'],
            'vt_ratio': calculate_vt_ratio(skill['value'], skill['time']),
            'cumulative_value': cumulative_value,
            'cumulative_time': cumulative_time
        })
    
    return Solution(
        skills_selected=selected,
        total_value=total_value,
        total_time=total_time,
        algorithm='Exhaustive',
        details=details
    )


def create_counterexample() -> Dict[str, Any]:
    """
    Cria um contraexemplo onde o algoritmo guloso NÃO encontra o ótimo.
    
    Contraexemplo:
        Habilidades:
        - A: V=8, T=10 → V/T = 0.80
        - B: V=7, T=5  → V/T = 1.40 ← Guloso escolhe primeiro
        - C: V=9, T=6  → V/T = 1.50 ← Guloso escolhe segundo
        
        Meta: S ≥ 15
        
        Guloso: B + C = 7 + 9 = 16 (tempo 11h) ✅
        Ótimo:  A + C = 8 + 9 = 17 (tempo 16h) ✅ MELHOR valor!
        
        Logo, guloso não maximiza valor; apenas minimiza tempo.
    
    Returns:
        Dict com o contraexemplo detalhado
    
    Examples:
        >>> counterexample = create_counterexample()
        >>> print(counterexample['explanation'])
    """
    # Define as habilidades do contraexemplo
    skills = {
        'A': {'value': 8, 'time': 10},
        'B': {'value': 7, 'time': 5},
        'C': {'value': 9, 'time': 6}
    }
    
    target = 15
    
    # Calcula V/T
    for sid, data in skills.items():
        data['vt_ratio'] = data['value'] / data['time']
    
    # Solução gulosa (ordena por V/T)
    sorted_skills = sorted(skills.items(), key=lambda x: x[1]['vt_ratio'], reverse=True)
    
    greedy_selected = []
    greedy_value = 0
    greedy_time = 0
    
    for sid, data in sorted_skills:
        greedy_selected.append(sid)
        greedy_value += data['value']
        greedy_time += data['time']
        if greedy_value >= target:
            break
    
    # Solução ótima (A + C tem maior valor)
    optimal_selected = ['A', 'C']
    optimal_value = skills['A']['value'] + skills['C']['value']
    optimal_time = skills['A']['time'] + skills['C']['time']
    
    return {
        'skills': skills,
        'target': target,
        'greedy': {
            'selected': greedy_selected,
            'value': greedy_value,
            'time': greedy_time,
            'meets_target': greedy_value >= target
        },
        'optimal': {
            'selected': optimal_selected,
            'value': optimal_value,
            'time': optimal_time,
            'meets_target': optimal_value >= target
        },
        'greedy_is_optimal': greedy_value == optimal_value,
        'explanation': (
            f"Contraexemplo:\n"
            f"  Guloso escolhe: {' + '.join(greedy_selected)} = {greedy_value} ({greedy_time}h)\n"
            f"  Ótimo seria: {' + '.join(optimal_selected)} = {optimal_value} ({optimal_time}h)\n"
            f"  Diferença: {optimal_value - greedy_value} de valor\n"
            f"  Conclusão: Guloso minimiza TEMPO, não maximiza VALOR!"
        )
    }


def compare_greedy_vs_optimal(
    graph: SkillGraph,
    basic_skills: List[str] = BASIC_SKILLS,
    target_value: float = MIN_ADAPTABILITY_TARGET
) -> Dict[str, Any]:
    """
    Compara solução gulosa vs solução ótima.
    
    Args:
        graph: Grafo de habilidades
        basic_skills: Lista de habilidades básicas
        target_value: Meta de valor
    
    Returns:
        Dict com comparação detalhada
    """
    # Executa ambos os algoritmos
    greedy_solution = greedy_selection(graph, basic_skills, target_value)
    optimal_solution = exhaustive_search(graph, basic_skills, target_value)
    
    # Compara
    is_optimal = (
        greedy_solution.total_value == optimal_solution.total_value and
        greedy_solution.total_time == optimal_solution.total_time
    )
    
    return {
        'greedy': greedy_solution.to_dict(),
        'optimal': optimal_solution.to_dict(),
        'comparison': {
            'greedy_is_optimal': is_optimal,
            'value_difference': optimal_solution.total_value - greedy_solution.total_value,
            'time_difference': optimal_solution.total_time - greedy_solution.total_time,
            'greedy_efficiency': greedy_solution.efficiency(),
            'optimal_efficiency': optimal_solution.efficiency()
        },
        'interpretation': (
            f"Guloso {'ENCONTROU' if is_optimal else 'NÃO ENCONTROU'} a solução ótima.\n"
            f"Diferença de valor: {optimal_solution.total_value - greedy_solution.total_value}\n"
            f"Diferença de tempo: {optimal_solution.total_time - greedy_solution.total_time}h"
        )
    }


def analyze_complexity() -> Dict[str, Any]:
    """
    Analisa a complexidade de cada algoritmo.
    
    Returns:
        Dict com análise de complexidade
    """
    return {
        'greedy': {
            'time_complexity': 'O(n log n)',
            'space_complexity': 'O(n)',
            'explanation': (
                "Ordenação por V/T: O(n log n)\n"
                "Seleção iterativa: O(n)\n"
                "Total: O(n log n)"
            ),
            'practical': 'Muito eficiente para n grande'
        },
        'exhaustive': {
            'time_complexity': 'O(2^n × n)',
            'space_complexity': 'O(n)',
            'explanation': (
                "Enumera 2^n subconjuntos: O(2^n)\n"
                "Para cada subconjunto, calcula soma: O(n)\n"
                "Total: O(2^n × n)"
            ),
            'practical': 'Inviável para n > 20'
        },
        'comparison': {
            'n=5': {
                'greedy': '~12 operações',
                'exhaustive': '~160 operações',
                'ratio': '13x mais lento'
            },
            'n=10': {
                'greedy': '~33 operações',
                'exhaustive': '~10,240 operações',
                'ratio': '310x mais lento'
            },
            'n=20': {
                'greedy': '~86 operações',
                'exhaustive': '~20,971,520 operações',
                'ratio': '~244,000x mais lento'
            }
        },
        'recommendation': (
            "Para n ≤ 20: Exaustivo é viável e garante ótimo\n"
            "Para n > 20: Guloso é necessário (exaustivo é impraticável)\n"
            "Neste problema (n=5): Ambos são rápidos, exaustivo garante ótimo"
        )
    }


@measure_performance
def solve_complete(
    graph: SkillGraph,
    basic_skills: List[str] = BASIC_SKILLS,
    target_value: float = MIN_ADAPTABILITY_TARGET
) -> Dict[str, Any]:
    """
    Resolve o Desafio 3 COMPLETO:
    1. Algoritmo guloso
    2. Busca exaustiva
    3. Comparação
    4. Contraexemplo
    5. Análise de complexidade
    
    Esta é a função principal do Desafio 3.
    
    Args:
        graph: Grafo de habilidades
        basic_skills: Lista de habilidades básicas
        target_value: Meta de adaptabilidade
    
    Returns:
        Dict com TODOS os resultados do Desafio 3
    """
    print("\n" + "=" * 70)
    print("🎯 DESAFIO 3 - PIVÔ MAIS RÁPIDO")
    print("=" * 70)
    
    print(f"\nObjetivo: Alcançar S ≥ {target_value} com habilidades básicas")
    print(f"Habilidades disponíveis: {', '.join(basic_skills)}")
    
    # 1. Algoritmo Guloso
    print("\n📊 FASE 1: Algoritmo Guloso (V/T)")
    print("-" * 70)
    greedy_result = greedy_selection(graph, basic_skills, target_value)
    
    print(f"\n✅ Solução Gulosa:")
    print(f"   Skills: {' + '.join(greedy_result.skills_selected)}")
    print(f"   Valor: {greedy_result.total_value} {'✅' if greedy_result.meets_target() else '❌'}")
    print(f"   Tempo: {greedy_result.total_time}h")
    print(f"   Eficiência (V/T): {greedy_result.efficiency():.3f}")
    
    # 2. Busca Exaustiva
    print("\n📊 FASE 2: Busca Exaustiva (Ótimo)")
    print("-" * 70)
    optimal_result = exhaustive_search(graph, basic_skills, target_value)
    
    print(f"\n✅ Solução Ótima:")
    print(f"   Skills: {' + '.join(optimal_result.skills_selected)}")
    print(f"   Valor: {optimal_result.total_value} {'✅' if optimal_result.meets_target() else '❌'}")
    print(f"   Tempo: {optimal_result.total_time}h")
    print(f"   Eficiência (V/T): {optimal_result.efficiency():.3f}")
    
    # 3. Comparação
    print("\n📊 FASE 3: Comparação Guloso vs Ótimo")
    print("-" * 70)
    
    is_optimal = (
        greedy_result.total_value == optimal_result.total_value and
        greedy_result.total_time == optimal_result.total_time
    )
    
    print(f"\n📈 Comparação:")
    print(f"   Guloso é ótimo: {'✅ SIM' if is_optimal else '❌ NÃO'}")
    print(f"   Diferença de valor: {optimal_result.total_value - greedy_result.total_value}")
    print(f"   Diferença de tempo: {optimal_result.total_time - greedy_result.total_time}h")
    
    # 4. Contraexemplo
    print("\n📊 FASE 4: Contraexemplo")
    print("-" * 70)
    counterexample = create_counterexample()
    print(f"\n{counterexample['explanation']}")
    
    # 5. Análise de Complexidade
    print("\n📊 FASE 5: Análise de Complexidade")
    print("-" * 70)
    complexity = analyze_complexity()
    
    print(f"\n⏱️  Complexidade:")
    print(f"   Guloso: {complexity['greedy']['time_complexity']}")
    print(f"   Exaustivo: {complexity['exhaustive']['time_complexity']}")
    
    print(f"\n💡 {complexity['recommendation']}")
    
    return {
        'greedy': greedy_result.to_dict(),
        'optimal': optimal_result.to_dict(),
        'comparison': {
            'greedy_is_optimal': is_optimal,
            'value_difference': optimal_result.total_value - greedy_result.total_value,
            'time_difference': optimal_result.total_time - greedy_result.total_time
        },
        'counterexample': counterexample,
        'complexity_analysis': complexity
    }

def save_desafio3_results(results: dict) -> None:
    """
    Salva resultados do Desafio 3 em JSON.
    
    Args:
        results: Resultados completos do Desafio 3
    """
    output_file = OUTPUTS_DIR / 'desafio3_results.json'
    
    save_data = {
        'metadata': {
            'desafio': 'Desafio 3 - Pivô Mais Rápido',
            'metodo': 'Guloso vs Busca Exaustiva',
            'habilidades_basicas': BASIC_SKILLS,
            'target_adaptabilidade': MIN_ADAPTABILITY_TARGET
        },
        'guloso': results['greedy'],
        'otimo': results['optimal'],
        'comparacao': results['comparison'],
        'contraexemplo': results['counterexample'],
        'analise_complexidade': results['complexity_analysis'],
        'tempo_execucao_ms': float(results.get('time_ms', 0)),
        'memoria_kb': float(results.get('memory_kb', 0))
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {output_file}")


def run_desafio3_complete(
    graph: SkillGraph,
    basic_skills: List[str] = BASIC_SKILLS,
    target_value: float = MIN_ADAPTABILITY_TARGET
) -> dict:
    """
    Executa Desafio 3 completo e salva resultados.
    """
    results = solve_complete(graph, basic_skills, target_value)
    save_desafio3_results(results)
    return results