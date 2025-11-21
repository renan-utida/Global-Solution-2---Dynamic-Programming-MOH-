"""
Configurações e constantes globais para o Motor de Orientação de Habilidades (MOH)
Global Solution - Dynamic Programming
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Suprime warnings desnecessários
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURAÇÃO DE ESTILO PARA GRÁFICOS
# ============================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# CAMINHOS DO PROJETO
# ============================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
SRC_DIR = PROJECT_ROOT / 'src'

# Cria diretórios se não existirem
OUTPUTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Arquivo de dataset
SKILLS_DATASET_FILE = DATA_DIR / 'skills_dataset.json'

# ============================================
# PARÂMETROS DO PROBLEMA - DESAFIO 1
# ============================================

# Restrições do knapsack multidimensional
MAX_TIME_HOURS = 350        # Tempo máximo total (horas)
MAX_COMPLEXITY = 30         # Complexidade cumulativa máxima

# Objetivo final
TARGET_SKILL = 'S6'         # IA Generativa Ética

# Simulação Monte Carlo
N_MONTE_CARLO_SCENARIOS = 1000  # Número de cenários estocásticos
UNCERTAINTY_PERCENTAGE = 0.10    # ±10% de variação no valor

# ============================================
# PARÂMETROS DO PROBLEMA - DESAFIO 2
# ============================================

# Habilidades críticas (5 habilidades)
CRITICAL_SKILLS = ['S3', 'S5', 'S7', 'S8', 'S9']
N_CRITICAL_PERMUTATIONS = 120  # 5! = 120

# ============================================
# PARÂMETROS DO PROBLEMA - DESAFIO 3
# ============================================

# Habilidades básicas (sem pré-requisitos)
BASIC_SKILLS = ['S1', 'S2', 'S7', 'H10', 'H12']

# Meta de adaptabilidade mínima
MIN_ADAPTABILITY_TARGET = 15  # S ≥ 15

# ============================================
# PARÂMETROS DO PROBLEMA - DESAFIO 4
# ============================================

# Algoritmo de ordenação
SORTING_ALGORITHM = 'merge_sort'  # Opções: 'merge_sort', 'quick_sort'

# Divisão em sprints
SPRINT_A_SIZE = 6  # Habilidades 1-6
SPRINT_B_SIZE = 6  # Habilidades 7-12

# ============================================
# PARÂMETROS DO PROBLEMA - DESAFIO 5
# ============================================

# Horizonte de recomendação
RECOMMENDATION_HORIZON_YEARS = 5

# Número de habilidades a recomendar
N_RECOMMENDATIONS = 3  # Top 2-3 habilidades

# Cenários de mercado (probabilidades)
MARKET_SCENARIOS = {
    'ia_em_alta': {
        'prob': 0.35,
        'boost': {'S4': 1.20, 'S6': 1.25, 'H11': 1.15}  # +20%, +25%, +15%
    },
    'cloud_first': {
        'prob': 0.30,
        'boost': {'S7': 1.15, 'S9': 1.20, 'S8': 1.10}
    },
    'data_driven': {
        'prob': 0.20,
        'boost': {'S2': 1.10, 'S5': 1.15, 'H11': 1.20}
    },
    'seguranca_critica': {
        'prob': 0.15,
        'boost': {'H10': 1.25, 'S7': 1.10, 'S9': 1.10}
    }
}

# ============================================
# SEED GLOBAL PARA REPRODUTIBILIDADE
# ============================================

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# ============================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# ============================================

FIGURE_DPI = 300          # DPI para salvamento de figuras
FIGURE_FORMAT = 'png'     # Formato padrão
HIST_BINS = 50            # Bins para histogramas

# Cores para os métodos
COLOR_DETERMINISTIC = '#3498db'  # Azul
COLOR_STOCHASTIC = '#e74c3c'     # Vermelho
COLOR_OPTIMAL = '#2ecc71'        # Verde
COLOR_GREEDY = '#f39c12'         # Laranja
COLOR_BASELINE = '#95a5a6'       # Cinza

# Cores para categorias de habilidades
COLOR_BASIC = '#3498db'      # Azul
COLOR_CRITICAL = '#e74c3c'   # Vermelho
COLOR_ADVANCED = '#9b59b6'   # Roxo
COLOR_TARGET = '#2ecc71'     # Verde

# ============================================
# CONFIGURAÇÕES DE LOGS E DEBUG
# ============================================

VERBOSE = True            # Exibe logs detalhados
DEBUG_MODE = False        # Modo debug (mais informações)
LOG_PERFORMANCE = True    # Loga métricas de performance

# ============================================
# FUNÇÕES AUXILIARES DE FORMATAÇÃO
# ============================================

def print_header(title: str, symbol: str = "=", width: int = 70) -> None:
    """
    Imprime um cabeçalho formatado.
    
    Args:
        title: Título do cabeçalho
        symbol: Símbolo para a linha
        width: Largura da linha
    
    Returns:
        None
    """
    print("\n" + symbol * width)
    print(title)
    print(symbol * width)


def format_hours(hours: float) -> str:
    """
    Formata horas para exibição.
    
    Args:
        hours: Número de horas
    
    Returns:
        str: Horas formatadas (ex: "80h", "120.5h")
    """
    if hours == int(hours):
        return f"{int(hours)}h"
    return f"{hours:.1f}h"


def format_value(value: float) -> str:
    """
    Formata valor (escala 1-10) para exibição.
    
    Args:
        value: Valor numérico
    
    Returns:
        str: Valor formatado (ex: "8.0", "9.5")
    """
    return f"{value:.1f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Formata valor como percentual.
    
    Args:
        value: Valor decimal (ex: 0.10 para 10%)
        decimals: Número de casas decimais
    
    Returns:
        str: Percentual formatado (ex: "+10.00%", "-5.50%")
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def format_skill_name(skill_id: str, skill_name: str) -> str:
    """
    Formata nome da habilidade para exibição.
    
    Args:
        skill_id: ID da habilidade (ex: "S1")
        skill_name: Nome da habilidade
    
    Returns:
        str: Formatado como "S1 - Programação Básica (Python)"
    """
    return f"{skill_id} - {skill_name}"


def format_path(path: list) -> str:
    """
    Formata caminho de habilidades para exibição.
    
    Args:
        path: Lista de IDs de habilidades
    
    Returns:
        str: Caminho formatado (ex: "S1 → S3 → S4 → S6")
    """
    return " → ".join(path)


def format_constraint(current: float, maximum: float, unit: str = "") -> str:
    """
    Formata restrição para exibição.
    
    Args:
        current: Valor atual
        maximum: Valor máximo
        unit: Unidade (opcional, ex: "h")
    
    Returns:
        str: Formatado como "280h / 350h (80%)"
    """
    percentage = (current / maximum * 100) if maximum > 0 else 0
    return f"{current:.0f}{unit} / {maximum:.0f}{unit} ({percentage:.0f}%)"


# ============================================
# INFORMAÇÕES DE VALIDAÇÃO
# ============================================

VALIDATION_INFO = {
    'ciclos': {
        'severidade': 'CRÍTICA',
        'acao': 'Interromper execução',
        'mensagem': 'Grafo contém ciclos! Pré-requisitos formam dependência circular.'
    },
    'orfaos': {
        'severidade': 'CRÍTICA',
        'acao': 'Interromper execução',
        'mensagem': 'Grafo contém nós órfãos! Pré-requisitos inexistentes detectados.'
    }
}

# ============================================
# EXPORTAÇÃO DE CONSTANTES IMPORTANTES
# ============================================

__all__ = [
    # Constantes principais
    'MAX_TIME_HOURS',
    'MAX_COMPLEXITY',
    'TARGET_SKILL',
    'CRITICAL_SKILLS',
    'BASIC_SKILLS',
    
    # Funções de formatação
    'print_header',
    'format_hours',
    'format_value',
    'format_percentage',
    'format_skill_name',
    'format_path',
    'format_constraint',
    
    # Mensagens
    'WELCOME_MESSAGE',
    'PROBLEM_DESCRIPTION',
    
    # Caminhos
    'DATA_DIR',
    'OUTPUTS_DIR',
    'SKILLS_DATASET_FILE',
]