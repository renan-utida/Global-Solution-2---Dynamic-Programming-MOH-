"""
Decorators para medição de performance e validação
"""

import time
import tracemalloc
from functools import wraps
from typing import Any, Callable, Dict, Optional


def format_time(seconds: float) -> str:
    """
    Formata o tempo de execução para exibição legível.
    
    Args:
        seconds: Tempo em segundos
    
    Returns:
        str: Tempo formatado (μs, ms, s ou min)
    """
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"


def measure_performance(func: Callable) -> Callable:
    """
    Decorator para medir tempo de execução e uso de memória.
    
    Adiciona ao resultado do retorno:
    - time_ms: Tempo de execução em milissegundos
    - memory_kb: Pico de memória em KB
    - time_formatted: Tempo formatado para exibição
    
    Args:
        func: Função a ser decorada
    
    Returns:
        Callable: Função decorada com medições de performance
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        # Exibe início da execução
        print(f"\n🔍 Executando: {func.__name__}")
        print("-" * 70)
        
        # Inicia medições
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Executa função
        result = func(*args, **kwargs)
        
        # Finaliza medições
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calcula métricas
        time_seconds = end_time - start_time
        time_ms = time_seconds * 1000
        memory_kb = peak / 1024
        time_formatted = format_time(time_seconds)
        
        # Adiciona métricas ao resultado
        if isinstance(result, dict):
            result['time_ms'] = time_ms
            result['memory_kb'] = memory_kb
            result['time_formatted'] = time_formatted
        else:
            # Se não for dict, encapsula
            result = {
                'data': result,
                'time_ms': time_ms,
                'memory_kb': memory_kb,
                'time_formatted': time_formatted
            }
        
        # Exibe métricas formatadas
        print(f"\n📊 Performance:")
        print(f"   ⏱️  Tempo: {time_formatted}")
        print(f"   💾 Memória: {memory_kb:.2f} KB")
        
        return result
    
    return wrapper


def validate_inputs(**validators) -> Callable:
    """
    Decorator para validar inputs de funções com validadores customizados.
    
    Args:
        **validators: Dicionário de validadores por parâmetro
            - key: nome do parâmetro
            - value: função validadora que retorna (bool, str)
    
    Returns:
        Callable: Decorator configurado
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Valida cada parâmetro
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    value = kwargs[param_name]
                    is_valid, error_msg = validator(value)
                    if not is_valid:
                        raise ValueError(f"Validação falhou para '{param_name}': {error_msg}")
            
            # Executa função
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_graph_inputs(func: Callable) -> Callable:
    """
    Decorator específico para validar inputs de funções que trabalham com grafos.
    
    Verifica:
    - Grafo não é None
    - Skill IDs existem no grafo
    - Parâmetros numéricos são válidos
    
    Args:
        func: Função a ser decorada
    
    Returns:
        Callable: Função decorada com validação
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Valida grafo
        if 'graph' in kwargs and kwargs['graph'] is None:
            raise ValueError("Grafo não pode ser None")
        
        # Valida skill_id
        if 'skill_id' in kwargs:
            skill_id = kwargs['skill_id']
            graph = kwargs.get('graph')
            if graph is not None and not hasattr(graph, 'has_node'):
                raise TypeError("Grafo deve ter método 'has_node'")
        
        # Valida max_time
        if 'max_time' in kwargs:
            max_time = kwargs['max_time']
            if max_time <= 0:
                raise ValueError(f"max_time deve ser > 0, recebido: {max_time}")
        
        # Valida max_complexity
        if 'max_complexity' in kwargs:
            max_complexity = kwargs['max_complexity']
            if max_complexity <= 0:
                raise ValueError(f"max_complexity deve ser > 0, recebido: {max_complexity}")
        
        # Executa função
        return func(*args, **kwargs)
    
    return wrapper


def log_execution(verbose: bool = True) -> Callable:
    """
    Decorator para logar execução de funções.
    
    Args:
        verbose: Se True, exibe logs detalhados
    
    Returns:
        Callable: Decorator configurado
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if verbose:
                func_name = func.__name__
                print(f"\n🚀 Iniciando: {func_name}")
                
                # Exibe argumentos se houver
                if args:
                    print(f"   📥 Args: {args}")
                if kwargs:
                    print(f"   📥 Kwargs: {kwargs}")
            
            # Executa função
            result = func(*args, **kwargs)
            
            if verbose:
                print(f"   ✅ {func_name} concluído com sucesso")
            
            return result
        return wrapper
    return decorator


def cache_results(maxsize: int = 128) -> Callable:
    """
    Decorator para cachear resultados de funções (memoização).
    
    Útil para funções puras com chamadas repetidas.
    
    Args:
        maxsize: Tamanho máximo do cache
    
    Returns:
        Callable: Decorator configurado
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache_hits, cache_misses
            
            # Cria chave do cache
            # Converte kwargs em tupla ordenada para ser hashable
            kwargs_tuple = tuple(sorted(kwargs.items()))
            cache_key = (args, kwargs_tuple)
            
            # Verifica se está no cache
            if cache_key in cache:
                cache_hits += 1
                return cache[cache_key]
            
            # Executa função
            cache_misses += 1
            result = func(*args, **kwargs)
            
            # Armazena no cache (limitado ao maxsize)
            if len(cache) >= maxsize:
                # Remove item mais antigo (FIFO)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[cache_key] = result
            return result
        
        # Adiciona métodos auxiliares
        wrapper.cache_info = lambda: {
            'hits': cache_hits,
            'misses': cache_misses,
            'size': len(cache),
            'maxsize': maxsize
        }
        wrapper.cache_clear = lambda: cache.clear()
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """
    Decorator para tentar novamente em caso de falha.
    
    Args:
        max_attempts: Número máximo de tentativas
        delay: Tempo de espera entre tentativas (segundos)
    
    Returns:
        Callable: Decorator configurado
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        print(f"❌ Falha após {max_attempts} tentativas: {e}")
                        raise
                    print(f"⚠️  Tentativa {attempts} falhou: {e}")
                    print(f"   Tentando novamente em {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator


# ============================================
# VALIDADORES PRÉ-DEFINIDOS
# ============================================

def validate_positive(value: float) -> tuple[bool, str]:
    """Valida se valor é positivo."""
    if value <= 0:
        return False, f"Valor deve ser > 0, recebido: {value}"
    return True, ""


def validate_non_negative(value: float) -> tuple[bool, str]:
    """Valida se valor é não-negativo."""
    if value < 0:
        return False, f"Valor deve ser ≥ 0, recebido: {value}"
    return True, ""


def validate_in_range(min_val: float, max_val: float):
    """Retorna validador de intervalo."""
    def validator(value: float) -> tuple[bool, str]:
        if not (min_val <= value <= max_val):
            return False, f"Valor deve estar entre {min_val} e {max_val}, recebido: {value}"
        return True, ""
    return validator


def validate_not_empty(value: Any) -> tuple[bool, str]:
    """Valida se valor não é vazio."""
    if not value:
        return False, "Valor não pode ser vazio"
    return True, ""


# ============================================
# EXPORTAÇÃO
# ============================================

__all__ = [
    'measure_performance',
    'validate_inputs',
    'validate_graph_inputs',
    'log_execution',
    'cache_results',
    'retry_on_failure',
    'format_time',
    # Validadores
    'validate_positive',
    'validate_non_negative',
    'validate_in_range',
    'validate_not_empty',
]