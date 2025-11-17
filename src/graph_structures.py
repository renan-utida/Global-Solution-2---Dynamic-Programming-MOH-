"""
Estruturas de dados para o grafo de habilidades

Este módulo implementa um grafo direcionado ponderado para representar
as habilidades e suas dependências (pré-requisitos).

Estruturas principais:
    - SkillGraph: Grafo direcionado com nós (habilidades) e arestas (pré-requisitos)
    - Dicionário de metadados: Informações completas de cada habilidade
    - Listas de adjacência: Representação eficiente do grafo

Complexidades:
    - add_node: O(1)
    - add_edge: O(1)
    - get_neighbors: O(1)
    - topological_sort: O(V + E) onde V = vértices, E = arestas
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict, deque


class SkillGraph:
    """
    Grafo direcionado ponderado para representar habilidades e pré-requisitos.
    
    Estrutura:
        - Nós: Habilidades (ex: S1, S2, ..., H12)
        - Arestas: Pré-requisitos direcionados (S1 → S3 significa "S1 é pré-req de S3")
        - Pesos: Tempo/custo da aresta (opcional, pode usar tempo da habilidade)
    
    Representação interna:
        - adjacency_list: Dict[str, List[str]] - Lista de adjacências
        - reverse_adjacency_list: Dict[str, List[str]] - Lista reversa (para pré-requisitos)
        - nodes_metadata: Dict[str, Dict] - Metadados completos de cada nó
        - edge_weights: Dict[Tuple[str, str], float] - Pesos das arestas
    
    Examples:
        >>> graph = SkillGraph()
        >>> graph.add_node('S1', {'nome': 'Python', 'tempo': 80, 'valor': 3})
        >>> graph.add_node('S3', {'nome': 'Algoritmos', 'tempo': 100, 'valor': 7})
        >>> graph.add_edge('S1', 'S3', weight=80)  # S1 é pré-req de S3
        >>> graph.get_neighbors('S1')
        ['S3']
        >>> graph.get_prerequisites('S3')
        ['S1']
    """
    
    def __init__(self):
        """Inicializa um grafo vazio."""
        # Lista de adjacências: skill_id → [dependentes]
        # Ex: 'S1' → ['S3', 'S4', 'S8'] (S3, S4, S8 dependem de S1)
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        
        # Lista reversa: skill_id → [pré-requisitos]
        # Ex: 'S3' → ['S1'] (S3 requer S1)
        self.reverse_adjacency_list: Dict[str, List[str]] = defaultdict(list)
        
        # Metadados dos nós
        self.nodes_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Pesos das arestas (opcional)
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        
        # Conjunto de todos os nós para busca O(1)
        self.nodes: Set[str] = set()
    
    def add_node(self, skill_id: str, metadata: Dict[str, Any]) -> None:
        """
        Adiciona um nó (habilidade) ao grafo.
        
        Args:
            skill_id: ID único da habilidade (ex: 'S1', 'H10')
            metadata: Dicionário com informações da habilidade:
                - nome: str
                - tempo_horas: int
                - valor: int (1-10)
                - complexidade: int (1-10)
                - pre_requisitos: List[str]
                - categoria: str (opcional)
                - descricao: str (opcional)
        
        Raises:
            ValueError: Se skill_id já existe
        
        Complexity:
            O(1)
        
        Examples:
            >>> graph.add_node('S1', {
            ...     'nome': 'Programação Básica (Python)',
            ...     'tempo_horas': 80,
            ...     'valor': 3,
            ...     'complexidade': 4,
            ...     'pre_requisitos': []
            ... })
        """
        if skill_id in self.nodes:
            raise ValueError(f"Nó '{skill_id}' já existe no grafo")
        
        self.nodes.add(skill_id)
        self.nodes_metadata[skill_id] = metadata
        
        # Inicializa listas de adjacência (mesmo que vazias)
        if skill_id not in self.adjacency_list:
            self.adjacency_list[skill_id] = []
        if skill_id not in self.reverse_adjacency_list:
            self.reverse_adjacency_list[skill_id] = []
    
    def add_edge(self, from_id: str, to_id: str, weight: Optional[float] = None) -> None:
        """
        Adiciona uma aresta direcionada (pré-requisito) ao grafo.
        
        Semântica: from_id → to_id significa "from_id é pré-requisito de to_id"
        
        Args:
            from_id: ID da habilidade pré-requisito
            to_id: ID da habilidade dependente
            weight: Peso da aresta (opcional, pode ser o tempo da habilidade)
        
        Raises:
            ValueError: Se from_id ou to_id não existem no grafo
        
        Complexity:
            O(1)
        
        Examples:
            >>> graph.add_edge('S1', 'S3')  # S1 é pré-requisito de S3
            >>> graph.add_edge('S1', 'S4', weight=80)  # Com peso
        """
        if from_id not in self.nodes:
            raise ValueError(f"Nó de origem '{from_id}' não existe no grafo")
        if to_id not in self.nodes:
            raise ValueError(f"Nó de destino '{to_id}' não existe no grafo")
        
        # Adiciona na lista de adjacências (from_id → to_id)
        if to_id not in self.adjacency_list[from_id]:
            self.adjacency_list[from_id].append(to_id)
        
        # Adiciona na lista reversa (to_id ← from_id)
        if from_id not in self.reverse_adjacency_list[to_id]:
            self.reverse_adjacency_list[to_id].append(from_id)
        
        # Armazena peso se fornecido
        if weight is not None:
            self.edge_weights[(from_id, to_id)] = weight
    
    def has_node(self, skill_id: str) -> bool:
        """
        Verifica se um nó existe no grafo.
        
        Args:
            skill_id: ID da habilidade
        
        Returns:
            bool: True se o nó existe, False caso contrário
        
        Complexity:
            O(1)
        """
        return skill_id in self.nodes
    
    def get_neighbors(self, skill_id: str) -> List[str]:
        """
        Retorna lista de habilidades que dependem de skill_id.
        
        Em outras palavras: retorna as habilidades que têm skill_id como pré-requisito.
        
        Args:
            skill_id: ID da habilidade
        
        Returns:
            List[str]: Lista de IDs das habilidades dependentes
        
        Raises:
            ValueError: Se skill_id não existe no grafo
        
        Complexity:
            O(1)
        
        Examples:
            >>> graph.get_neighbors('S1')
            ['S3', 'S4', 'S8']  # S3, S4, S8 dependem de S1
        """
        if skill_id not in self.nodes:
            raise ValueError(f"Nó '{skill_id}' não existe no grafo")
        
        return self.adjacency_list[skill_id].copy()
    
    def get_prerequisites(self, skill_id: str) -> List[str]:
        """
        Retorna lista de pré-requisitos de skill_id.
        
        Args:
            skill_id: ID da habilidade
        
        Returns:
            List[str]: Lista de IDs dos pré-requisitos
        
        Raises:
            ValueError: Se skill_id não existe no grafo
        
        Complexity:
            O(1)
        
        Examples:
            >>> graph.get_prerequisites('S3')
            ['S1']  # S3 requer S1
            >>> graph.get_prerequisites('S9')
            ['S7', 'S8']  # S9 requer S7 e S8
        """
        if skill_id not in self.nodes:
            raise ValueError(f"Nó '{skill_id}' não existe no grafo")
        
        return self.reverse_adjacency_list[skill_id].copy()
    
    def get_metadata(self, skill_id: str) -> Dict[str, Any]:
        """
        Retorna os metadados completos de uma habilidade.
        
        Args:
            skill_id: ID da habilidade
        
        Returns:
            Dict[str, Any]: Dicionário com todos os metadados
        
        Raises:
            ValueError: Se skill_id não existe no grafo
        
        Complexity:
            O(1)
        
        Examples:
            >>> metadata = graph.get_metadata('S1')
            >>> print(metadata['nome'])
            'Programação Básica (Python)'
            >>> print(metadata['tempo_horas'])
            80
        """
        if skill_id not in self.nodes:
            raise ValueError(f"Nó '{skill_id}' não existe no grafo")
        
        return self.nodes_metadata[skill_id].copy()
    
    def get_edge_weight(self, from_id: str, to_id: str) -> Optional[float]:
        """
        Retorna o peso de uma aresta, se existir.
        
        Args:
            from_id: ID da habilidade origem
            to_id: ID da habilidade destino
        
        Returns:
            Optional[float]: Peso da aresta, ou None se não houver peso definido
        """
        return self.edge_weights.get((from_id, to_id))
    
    def get_in_degree(self, skill_id: str) -> int:
        """
        Retorna o grau de entrada de um nó (número de pré-requisitos).
        
        Args:
            skill_id: ID da habilidade
        
        Returns:
            int: Número de pré-requisitos
        
        Complexity:
            O(1)
        """
        if skill_id not in self.nodes:
            raise ValueError(f"Nó '{skill_id}' não existe no grafo")
        
        return len(self.reverse_adjacency_list[skill_id])
    
    def get_out_degree(self, skill_id: str) -> int:
        """
        Retorna o grau de saída de um nó (número de dependentes).
        
        Args:
            skill_id: ID da habilidade
        
        Returns:
            int: Número de habilidades que dependem desta
        
        Complexity:
            O(1)
        """
        if skill_id not in self.nodes:
            raise ValueError(f"Nó '{skill_id}' não existe no grafo")
        
        return len(self.adjacency_list[skill_id])
    
    def topological_sort(self) -> List[str]:
        """
        Retorna uma ordenação topológica do grafo usando Kahn's Algorithm.
        
        Ordenação topológica: ordem linear dos nós tal que para toda aresta (u, v),
        u aparece antes de v na ordenação.
        
        Em termos de habilidades: pré-requisitos sempre aparecem antes de suas dependentes.
        
        Algorithm (Kahn):
            1. Calcula grau de entrada de todos os nós
            2. Adiciona todos os nós com grau 0 a uma fila
            3. Enquanto fila não estiver vazia:
                a. Remove nó da fila e adiciona ao resultado
                b. Para cada vizinho do nó, decrementa seu grau de entrada
                c. Se grau de entrada chegar a 0, adiciona vizinho à fila
            4. Se resultado tem todos os nós, retorna; senão, há ciclo
        
        Returns:
            List[str]: Lista de IDs em ordem topológica
        
        Raises:
            ValueError: Se o grafo contém ciclos (não é DAG)
        
        Complexity:
            O(V + E) onde V = número de vértices, E = número de arestas
        
        Examples:
            >>> order = graph.topological_sort()
            >>> print(order)
            ['S1', 'S2', 'S7', 'H10', 'H12', 'S3', 'S5', 'S8', 'S4', 'S9', 'H11', 'S6']
        """
        # Copia dos graus de entrada para não modificar o grafo
        in_degree = {node: self.get_in_degree(node) for node in self.nodes}
        
        # Fila com nós de grau 0 (sem pré-requisitos)
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        
        # Resultado da ordenação
        topo_order = []
        
        while queue:
            # Remove nó da fila
            current = queue.popleft()
            topo_order.append(current)
            
            # Para cada vizinho (dependente)
            for neighbor in self.adjacency_list[current]:
                # Decrementa grau de entrada
                in_degree[neighbor] -= 1
                
                # Se grau chegou a 0, adiciona à fila
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Verifica se todos os nós foram processados
        if len(topo_order) != len(self.nodes):
            # Se não, há ciclo no grafo
            missing_nodes = set(self.nodes) - set(topo_order)
            raise ValueError(
                f"Grafo contém ciclo! Ordenação topológica impossível. "
                f"Nós não processados: {missing_nodes}"
            )
        
        return topo_order
    
    def get_all_paths(self, start: str, end: str) -> List[List[str]]:
        """
        Retorna todos os caminhos possíveis de start até end (sem ciclos).
        
        Usa DFS para encontrar todos os caminhos.
        
        Args:
            start: ID da habilidade inicial
            end: ID da habilidade final
        
        Returns:
            List[List[str]]: Lista de caminhos, onde cada caminho é uma lista de IDs
        
        Complexity:
            O(V! / (V-L)!) no pior caso, onde L = tamanho do caminho
        """
        if start not in self.nodes or end not in self.nodes:
            return []
        
        all_paths = []
        
        def dfs(current: str, target: str, path: List[str], visited: Set[str]):
            """DFS recursivo para encontrar todos os caminhos."""
            if current == target:
                all_paths.append(path.copy())
                return
            
            visited.add(current)
            
            for neighbor in self.adjacency_list[current]:
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, target, path, visited)
                    path.pop()
            
            visited.remove(current)
        
        dfs(start, end, [start], set())
        return all_paths
    
    def get_basic_skills(self) -> List[str]:
        """
        Retorna lista de habilidades básicas (sem pré-requisitos).
        
        Returns:
            List[str]: IDs das habilidades com grau de entrada 0
        """
        return [skill_id for skill_id in self.nodes if self.get_in_degree(skill_id) == 0]
    
    def __len__(self) -> int:
        """Retorna número de nós no grafo."""
        return len(self.nodes)
    
    def __contains__(self, skill_id: str) -> bool:
        """Permite usar 'in' para verificar se nó existe."""
        return skill_id in self.nodes
    
    def __repr__(self) -> str:
        """Representação string do grafo."""
        return f"SkillGraph(nodes={len(self.nodes)}, edges={sum(len(v) for v in self.adjacency_list.values())})"


def load_skills_from_json(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """
    Carrega o dataset de habilidades de um arquivo JSON.
    
    Args:
        filepath: Caminho para o arquivo JSON
    
    Returns:
        Dict[str, Dict]: Dicionário de habilidades no formato:
            {
                'S1': {
                    'nome': 'Programação Básica (Python)',
                    'tempo_horas': 80,
                    'valor': 3,
                    'complexidade': 4,
                    'pre_requisitos': []
                },
                ...
            }
    
    Raises:
        FileNotFoundError: Se arquivo não existe
        json.JSONDecodeError: Se JSON é inválido
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('skills', {})


def build_graph_from_dict(skills_dict: Dict[str, Dict[str, Any]]) -> SkillGraph:
    """
    Constrói um SkillGraph a partir de um dicionário de habilidades.
    
    Args:
        skills_dict: Dicionário de habilidades (formato load_skills_from_json)
    
    Returns:
        SkillGraph: Grafo completo com todos os nós e arestas
    
    Examples:
        >>> skills = load_skills_from_json('data/skills_dataset.json')
        >>> graph = build_graph_from_dict(skills)
        >>> len(graph)
        12
    """
    graph = SkillGraph()
    
    # FASE 1: Adiciona todos os nós
    for skill_id, metadata in skills_dict.items():
        graph.add_node(skill_id, metadata)
    
    # FASE 2: Adiciona todas as arestas (pré-requisitos)
    for skill_id, metadata in skills_dict.items():
        pre_reqs = metadata.get('pre_requisitos', [])
        
        for pre_req_id in pre_reqs:
            # Verifica se pré-requisito existe
            if pre_req_id in graph:
                # Adiciona aresta: pre_req → skill
                # Peso = tempo do pré-requisito
                weight = skills_dict[pre_req_id]['tempo_horas']
                graph.add_edge(pre_req_id, skill_id, weight=weight)
    
    return graph


def build_graph_from_file(filepath: Path) -> SkillGraph:
    """
    Constrói um SkillGraph diretamente de um arquivo JSON.
    
    Combina load_skills_from_json e build_graph_from_dict.
    
    Args:
        filepath: Caminho para o arquivo JSON
    
    Returns:
        SkillGraph: Grafo completo
    
    Examples:
        >>> from pathlib import Path
        >>> graph = build_graph_from_file(Path('data/skills_dataset.json'))
        >>> print(graph)
        SkillGraph(nodes=12, edges=...)
    """
    skills_dict = load_skills_from_json(filepath)
    return build_graph_from_dict(skills_dict)


# Funções auxiliares para análise do grafo

def print_graph_summary(graph: SkillGraph) -> None:
    """
    Imprime um resumo detalhado do grafo.
    
    Args:
        graph: Grafo de habilidades
    """
    print("=" * 70)
    print("RESUMO DO GRAFO DE HABILIDADES")
    print("=" * 70)
    
    print(f"\n📊 Estatísticas Gerais:")
    print(f"   • Total de habilidades: {len(graph)}")
    print(f"   • Total de pré-requisitos: {sum(len(v) for v in graph.adjacency_list.values())}")
    
    # Habilidades básicas
    basic = graph.get_basic_skills()
    print(f"\n🔵 Habilidades Básicas (sem pré-requisitos): {len(basic)}")
    for skill_id in basic:
        metadata = graph.get_metadata(skill_id)
        print(f"   • {skill_id} - {metadata['nome']}")
    
    # Ordenação topológica
    try:
        topo_order = graph.topological_sort()
        print(f"\n📋 Ordenação Topológica:")
        print(f"   {' → '.join(topo_order)}")
    except ValueError as e:
        print(f"\n❌ Erro na ordenação topológica: {e}")
    
    print("=" * 70)