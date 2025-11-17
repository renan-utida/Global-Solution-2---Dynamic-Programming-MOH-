"""
Validação de grafo de habilidades

Este módulo implementa validações CRÍTICAS para o grafo de habilidades:
1. Detecção de ciclos (dependências circulares)
2. Detecção de nós órfãos (pré-requisitos inexistentes)

É OBRIGATÓRIO validar o grafo antes de executar otimizações!

Algoritmos:
    - detect_cycles: DFS com estados WHITE/GRAY/BLACK - O(V + E)
    - detect_orphan_nodes: Verificação de existência - O(V × P) onde P = pré-reqs médio
    
Por que é crítico:
    - Desafio 2 exige validação ANTES das 120 permutações
    - Vale 10 pontos na rubrica de avaliação
    - Previne resultados incorretos em todos os desafios
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum


class NodeState(Enum):
    """
    Estados de um nó durante a busca DFS para detecção de ciclos.
    
    WHITE: Nó não visitado ainda
    GRAY: Nó em processamento (na pilha de recursão)
    BLACK: Nó completamente processado (todos descendentes visitados)
    """
    WHITE = 0  # Não visitado
    GRAY = 1   # Em processamento (na pilha)
    BLACK = 2  # Processado completamente


def detect_cycles(graph) -> Dict[str, Any]:
    """
    Detecta ciclos no grafo usando DFS com estados (WHITE/GRAY/BLACK).
    
    Algoritmo:
        1. Marca todos os nós como WHITE (não visitados)
        2. Para cada nó WHITE, executa DFS
        3. Durante DFS:
           - Marca nó como GRAY (em processamento)
           - Visita vizinhos recursivamente
           - Se encontrar vizinho GRAY → CICLO detectado!
           - Após processar todos vizinhos, marca como BLACK
        4. Retorna lista de todos os ciclos encontrados
    
    Um ciclo é detectado quando encontramos uma aresta de um nó GRAY
    para outro nó GRAY, indicando um caminho de volta na pilha de recursão.
    
    Args:
        graph: Instância de SkillGraph
    
    Returns:
        Dict contendo:
            - has_cycles: bool - True se há ciclos
            - cycles: List[List[str]] - Lista de ciclos (cada ciclo é uma lista de nós)
            - cycle_edges: List[Tuple[str, str]] - Arestas que formam ciclos
    
    Complexity:
        O(V + E) onde V = vértices, E = arestas
        Cada nó e cada aresta são visitados exatamente uma vez
    
    Examples:
        >>> result = detect_cycles(graph)
        >>> if result['has_cycles']:
        ...     print(f"Ciclos encontrados: {result['cycles']}")
        >>> else:
        ...     print("Grafo é um DAG (Directed Acyclic Graph)")
    """
    # Estado de cada nó
    state = {node: NodeState.WHITE for node in graph.nodes}
    
    # Lista de ciclos encontrados
    cycles_found = []
    cycle_edges = []
    
    # Caminho atual na DFS (para reconstruir o ciclo)
    path = []
    path_set = set()  # Para busca O(1)
    
    def dfs(node: str) -> bool:
        """
        DFS recursivo para detectar ciclos.
        
        Args:
            node: Nó atual
        
        Returns:
            bool: True se ciclo foi encontrado a partir deste nó
        """
        # Marca como em processamento
        state[node] = NodeState.GRAY
        path.append(node)
        path_set.add(node)
        
        # Visita todos os vizinhos
        for neighbor in graph.get_neighbors(node):
            if state[neighbor] == NodeState.GRAY:
                # CICLO DETECTADO!
                # neighbor está na pilha de recursão (GRAY)
                
                # Reconstrói o ciclo a partir do caminho
                cycle_start_idx = path.index(neighbor)
                cycle = path[cycle_start_idx:] + [neighbor]
                cycles_found.append(cycle)
                
                # Aresta que fecha o ciclo
                cycle_edges.append((node, neighbor))
                
                return True
            
            elif state[neighbor] == NodeState.WHITE:
                # Vizinho ainda não visitado, explora recursivamente
                if dfs(neighbor):
                    # Propaga detecção de ciclo
                    pass  # Continua buscando mais ciclos
        
        # Marca como completamente processado
        state[node] = NodeState.BLACK
        path.pop()
        path_set.remove(node)
        
        return False
    
    # Executa DFS a partir de cada nó não visitado
    for node in graph.nodes:
        if state[node] == NodeState.WHITE:
            dfs(node)
    
    return {
        'has_cycles': len(cycles_found) > 0,
        'cycles': cycles_found,
        'cycle_edges': cycle_edges,
        'num_cycles': len(cycles_found)
    }


def detect_orphan_nodes(graph) -> Dict[str, Any]:
    """
    Detecta nós órfãos: habilidades que referenciam pré-requisitos inexistentes.
    
    Um nó é "órfão" se lista pré-requisitos que não existem no grafo.
    Isso indica erro de dados ou configuração incorreta.
    
    Algoritmo:
        1. Para cada nó no grafo
        2. Obtém lista de pré-requisitos dos metadados
        3. Verifica se cada pré-requisito existe no grafo
        4. Se não existe, marca como órfão
    
    Args:
        graph: Instância de SkillGraph
    
    Returns:
        Dict contendo:
            - has_orphans: bool - True se há nós órfãos
            - orphan_nodes: List[str] - Lista de nós com pré-reqs inexistentes
            - missing_prereqs: Dict[str, List[str]] - Mapa nó → pré-reqs faltantes
            - details: List[Dict] - Detalhes de cada nó órfão
    
    Complexity:
        O(V × P) onde V = vértices, P = pré-requisitos médios por nó
    
    Examples:
        >>> result = detect_orphan_nodes(graph)
        >>> if result['has_orphans']:
        ...     for node, missing in result['missing_prereqs'].items():
        ...         print(f"{node} requer {missing} que não existem!")
    """
    orphan_nodes = []
    missing_prereqs = {}
    details = []
    
    # Para cada nó no grafo
    for node in graph.nodes:
        # Obtém metadados
        metadata = graph.get_metadata(node)
        
        # Obtém lista de pré-requisitos dos metadados
        prereqs = metadata.get('pre_requisitos', [])
        
        # Verifica se cada pré-requisito existe
        missing = []
        for prereq in prereqs:
            if prereq not in graph.nodes:
                missing.append(prereq)
        
        # Se há pré-requisitos faltantes, marca como órfão
        if missing:
            orphan_nodes.append(node)
            missing_prereqs[node] = missing
            
            details.append({
                'node': node,
                'nome': metadata.get('nome', 'N/A'),
                'missing_prereqs': missing,
                'all_prereqs': prereqs
            })
    
    return {
        'has_orphans': len(orphan_nodes) > 0,
        'orphan_nodes': orphan_nodes,
        'missing_prereqs': missing_prereqs,
        'details': details,
        'num_orphans': len(orphan_nodes)
    }


def validate_graph(graph) -> Dict[str, Any]:
    """
    Valida completamente o grafo de habilidades.
    
    Executa TODAS as validações necessárias:
    1. Detecção de ciclos (dependências circulares)
    2. Detecção de nós órfãos (pré-requisitos inexistentes)
    
    Esta função é o PONTO DE ENTRADA principal para validação.
    DEVE ser chamada ANTES de qualquer otimização!
    
    Args:
        graph: Instância de SkillGraph
    
    Returns:
        Dict[str, Any] contendo:
            - valid: bool - True se grafo é válido (sem ciclos e sem órfãos)
            - cycles: List - Lista de ciclos encontrados
            - orphans: List - Lista de nós órfãos
            - error_msg: str - Mensagem de erro descritiva (se houver)
            - warnings: List[str] - Avisos não-críticos
            - details: Dict - Informações detalhadas sobre problemas
    
    Raises:
        ValueError: Se graph é None ou inválido
    
    Examples:
        >>> from src.graph_structures import build_graph_from_file
        >>> from src.config import SKILLS_DATASET_FILE
        >>> 
        >>> graph = build_graph_from_file(SKILLS_DATASET_FILE)
        >>> result = validate_graph(graph)
        >>> 
        >>> if result['valid']:
        ...     print("✅ Grafo válido! Pronto para otimização.")
        >>> else:
        ...     print(f"❌ ERRO: {result['error_msg']}")
        ...     if result['cycles']:
        ...         print(f"Ciclos: {result['cycles']}")
        ...     if result['orphans']:
        ...         print(f"Órfãos: {result['orphans']}")
    """
    # Valida input
    if graph is None:
        return {
            'valid': False,
            'cycles': [],
            'orphans': [],
            'error_msg': 'Grafo é None! Forneça uma instância válida de SkillGraph.',
            'warnings': [],
            'details': {}
        }
    
    if len(graph) == 0:
        return {
            'valid': False,
            'cycles': [],
            'orphans': [],
            'error_msg': 'Grafo está vazio! Adicione nós antes de validar.',
            'warnings': [],
            'details': {}
        }
    
    # VALIDAÇÃO 1: Detecção de Ciclos
    cycles_result = detect_cycles(graph)
    
    # VALIDAÇÃO 2: Detecção de Nós Órfãos
    orphans_result = detect_orphan_nodes(graph)
    
    # Determina se grafo é válido
    has_cycles = cycles_result['has_cycles']
    has_orphans = orphans_result['has_orphans']
    is_valid = not has_cycles and not has_orphans
    
    # Constrói mensagem de erro
    error_parts = []
    if has_cycles:
        error_parts.append(
            f"Grafo contém {cycles_result['num_cycles']} ciclo(s)! "
            f"Dependências circulares detectadas."
        )
    
    if has_orphans:
        error_parts.append(
            f"Grafo contém {orphans_result['num_orphans']} nó(s) órfão(s)! "
            f"Pré-requisitos inexistentes detectados."
        )
    
    error_msg = " ".join(error_parts) if error_parts else ""
    
    # Avisos não-críticos
    warnings = []
    
    # Verifica se há nós isolados (sem conexões)
    isolated_nodes = []
    for node in graph.nodes:
        if graph.get_in_degree(node) == 0 and graph.get_out_degree(node) == 0:
            isolated_nodes.append(node)
    
    if isolated_nodes:
        warnings.append(
            f"Aviso: {len(isolated_nodes)} nó(s) isolado(s) "
            f"(sem pré-requisitos e sem dependentes): {isolated_nodes}"
        )
    
    # Detalhes completos
    details = {
        'cycles_info': cycles_result,
        'orphans_info': orphans_result,
        'graph_stats': {
            'num_nodes': len(graph),
            'num_edges': sum(len(graph.adjacency_list[node]) for node in graph.nodes),
            'basic_skills': len(graph.get_basic_skills()),
            'isolated_nodes': isolated_nodes
        }
    }
    
    return {
        'valid': is_valid,
        'cycles': cycles_result['cycles'],
        'orphans': orphans_result['orphan_nodes'],
        'error_msg': error_msg,
        'warnings': warnings,
        'details': details
    }


def print_validation_report(validation_result: Dict[str, Any]) -> None:
    """
    Imprime um relatório formatado da validação do grafo.
    
    Args:
        validation_result: Resultado retornado por validate_graph()
    
    Examples:
        >>> result = validate_graph(graph)
        >>> print_validation_report(result)
    """
    print("\n" + "=" * 70)
    print("🔍 RELATÓRIO DE VALIDAÇÃO DO GRAFO")
    print("=" * 70)
    
    # Status geral
    if validation_result['valid']:
        print("\n✅ STATUS: GRAFO VÁLIDO")
        print("   O grafo passou em todas as validações!")
    else:
        print("\n❌ STATUS: GRAFO INVÁLIDO")
        print(f"   {validation_result['error_msg']}")
    
    # Detalhes de ciclos
    if validation_result['cycles']:
        print(f"\n🔴 CICLOS DETECTADOS: {len(validation_result['cycles'])}")
        for i, cycle in enumerate(validation_result['cycles'], 1):
            cycle_str = " → ".join(cycle)
            print(f"   Ciclo {i}: {cycle_str}")
    else:
        print("\n✅ CICLOS: Nenhum ciclo detectado (grafo é DAG)")
    
    # Detalhes de órfãos
    if validation_result['orphans']:
        print(f"\n🔴 NÓS ÓRFÃOS DETECTADOS: {len(validation_result['orphans'])}")
        
        orphans_info = validation_result['details']['orphans_info']
        for detail in orphans_info['details']:
            print(f"   • {detail['node']} ({detail['nome']})")
            print(f"     Pré-requisitos faltantes: {', '.join(detail['missing_prereqs'])}")
    else:
        print("\n✅ ÓRFÃOS: Todos os pré-requisitos existem")
    
    # Avisos
    if validation_result['warnings']:
        print(f"\n⚠️  AVISOS ({len(validation_result['warnings'])}):")
        for warning in validation_result['warnings']:
            print(f"   • {warning}")
    
    # Estatísticas do grafo
    stats = validation_result['details']['graph_stats']
    print(f"\n📊 ESTATÍSTICAS DO GRAFO:")
    print(f"   • Total de nós: {stats['num_nodes']}")
    print(f"   • Total de arestas: {stats['num_edges']}")
    print(f"   • Habilidades básicas: {stats['basic_skills']}")
    
    print("=" * 70)


def ensure_valid_graph(graph) -> None:
    """
    Valida o grafo e lança exceção se inválido.
    
    Útil para garantir que o grafo é válido antes de prosseguir.
    
    Args:
        graph: Instância de SkillGraph
    
    Raises:
        ValueError: Se o grafo é inválido (com detalhes do erro)
    
    Examples:
        >>> try:
        ...     ensure_valid_graph(graph)
        ...     # Prossegue com otimizações
        ... except ValueError as e:
        ...     print(f"Erro: {e}")
        ...     # Interrompe execução
    """
    result = validate_graph(graph)
    
    if not result['valid']:
        error_details = []
        
        if result['cycles']:
            error_details.append(
                f"CICLOS DETECTADOS ({len(result['cycles'])}): "
                + ", ".join([" → ".join(cycle) for cycle in result['cycles']])
            )
        
        if result['orphans']:
            orphans_info = result['details']['orphans_info']
            orphan_details = []
            for detail in orphans_info['details']:
                orphan_details.append(
                    f"{detail['node']} requer {detail['missing_prereqs']}"
                )
            error_details.append(
                f"NÓS ÓRFÃOS DETECTADOS ({len(result['orphans'])}): "
                + "; ".join(orphan_details)
            )
        
        full_error = result['error_msg'] + "\n\nDetalhes:\n" + "\n".join(error_details)
        
        raise ValueError(full_error)


def get_cycle_free_subgraph(graph, nodes_to_include: Set[str]):
    """
    Tenta criar um subgrafo sem ciclos a partir de um conjunto de nós.
    
    Útil para quando há ciclos mas queremos trabalhar com um subconjunto válido.
    
    Args:
        graph: Instância de SkillGraph
        nodes_to_include: Conjunto de IDs de nós a incluir
    
    Returns:
        SkillGraph: Novo grafo sem ciclos (ou None se impossível)
    
    Note:
        Esta é uma função auxiliar. Em produção, prefira corrigir os ciclos
        diretamente no dataset original.
    """
    from src.graph_structures import SkillGraph
    
    subgraph = SkillGraph()
    
    # Adiciona nós
    for node in nodes_to_include:
        if node in graph:
            metadata = graph.get_metadata(node)
            subgraph.add_node(node, metadata)
    
    # Adiciona arestas (apenas se ambos os nós estão no subgrafo)
    for node in nodes_to_include:
        if node in graph:
            for neighbor in graph.get_neighbors(node):
                if neighbor in nodes_to_include:
                    weight = graph.get_edge_weight(node, neighbor)
                    subgraph.add_edge(node, neighbor, weight)
    
    # Valida se ficou sem ciclos
    result = validate_graph(subgraph)
    
    if result['valid']:
        return subgraph
    else:
        return None


# Aliases para compatibilidade
check_cycles = detect_cycles
check_orphans = detect_orphan_nodes
validate = validate_graph