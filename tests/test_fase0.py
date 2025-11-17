#!/usr/bin/env python3
"""
Script de teste para validar setup da FASE 0
Global Solution - Motor de Orientação de Habilidades (MOH)

Este script verifica se todos os componentes da FASE 0 foram criados corretamente.
"""

import sys
import json
from pathlib import Path


def print_header(title: str) -> None:
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_directory_structure() -> bool:
    """Testa se a estrutura de diretórios foi criada."""
    print_header("🗂️  TESTE 1: Estrutura de Diretórios")
    
    required_dirs = [
        Path('data'),
        Path('src'),
        Path('outputs'),
        Path('tests')
    ]
    
    all_exist = True
    for directory in required_dirs:
        exists = directory.exists() and directory.is_dir()
        status = "✅" if exists else "❌"
        print(f"{status} {directory}/")
        all_exist = all_exist and exists
    
    return all_exist


def test_dataset() -> bool:
    """Testa se o dataset foi criado corretamente."""
    print_header("📊 TESTE 2: Dataset JSON")
    
    dataset_file = Path('data/skills_dataset.json')
    
    if not dataset_file.exists():
        print("❌ Arquivo skills_dataset.json não encontrado!")
        return False
    
    print("✅ Arquivo skills_dataset.json encontrado")
    
    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print("✅ JSON válido")
        
        # Valida estrutura
        assert 'skills' in dataset, "Chave 'skills' não encontrada"
        assert len(dataset['skills']) == 12, f"Esperado 12 habilidades, encontrado {len(dataset['skills'])}"
        
        # Valida habilidades específicas
        required_skills = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'H10', 'H11', 'H12']
        for skill_id in required_skills:
            assert skill_id in dataset['skills'], f"Habilidade {skill_id} não encontrada"
        
        print(f"✅ 12 habilidades validadas: {', '.join(required_skills)}")
        
        # Valida campos obrigatórios
        required_fields = ['nome', 'tempo_horas', 'valor', 'complexidade', 'pre_requisitos']
        sample_skill = dataset['skills']['S1']
        for field in required_fields:
            assert field in sample_skill, f"Campo '{field}' não encontrado"
        
        print(f"✅ Campos obrigatórios validados")
        
        # Valida S6 (objetivo final)
        s6 = dataset['skills']['S6']
        assert s6['nome'] == 'IA Generativa Ética', "S6 não é 'IA Generativa Ética'"
        assert s6['valor'] == 10, "S6 deve ter valor 10"
        assert s6['complexidade'] == 10, "S6 deve ter complexidade 10"
        
        print("✅ S6 (objetivo final) validado")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao decodificar JSON: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Validação falhou: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def test_config_module() -> bool:
    """Testa se o módulo config.py foi criado."""
    print_header("⚙️  TESTE 3: Módulo config.py")
    
    config_file = Path('src/config.py')
    
    if not config_file.exists():
        print("❌ Arquivo config.py não encontrado!")
        return False
    
    print("✅ Arquivo config.py encontrado")
    
    try:
        # Tenta importar
        sys.path.insert(0, str(Path.cwd()))
        from src import config
        
        print("✅ Módulo importado com sucesso")
        
        # Valida constantes principais
        required_constants = [
            'MAX_TIME_HOURS',
            'MAX_COMPLEXITY',
            'TARGET_SKILL',
            'CRITICAL_SKILLS',
            'BASIC_SKILLS'
        ]
        
        for const in required_constants:
            assert hasattr(config, const), f"Constante '{const}' não encontrada"
        
        print(f"✅ Constantes principais validadas")
        
        # Valida valores
        assert config.MAX_TIME_HOURS == 350, "MAX_TIME_HOURS deve ser 350"
        assert config.MAX_COMPLEXITY == 30, "MAX_COMPLEXITY deve ser 30"
        assert config.TARGET_SKILL == 'S6', "TARGET_SKILL deve ser 'S6'"
        assert len(config.CRITICAL_SKILLS) == 5, "Deve haver 5 habilidades críticas"
        
        print("✅ Valores das constantes validados")
        
        # Valida funções de formatação
        required_functions = [
            'print_header',
            'format_hours',
            'format_value',
            'format_percentage',
            'format_skill_name',
            'format_path'
        ]
        
        for func_name in required_functions:
            assert hasattr(config, func_name), f"Função '{func_name}' não encontrada"
        
        print("✅ Funções de formatação validadas")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro ao importar módulo: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Validação falhou: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def test_decorators_module() -> bool:
    """Testa se o módulo decorators.py foi criado."""
    print_header("🎨 TESTE 4: Módulo decorators.py")
    
    decorators_file = Path('src/decorators.py')
    
    if not decorators_file.exists():
        print("❌ Arquivo decorators.py não encontrado!")
        return False
    
    print("✅ Arquivo decorators.py encontrado")
    
    try:
        # Tenta importar
        from src import decorators
        
        print("✅ Módulo importado com sucesso")
        
        # Valida decorators
        required_decorators = [
            'measure_performance',
            'validate_inputs',
            'validate_graph_inputs',
            'log_execution',
            'cache_results'
        ]
        
        for decorator_name in required_decorators:
            assert hasattr(decorators, decorator_name), f"Decorator '{decorator_name}' não encontrado"
        
        print(f"✅ Decorators principais validados")
        
        # Testa measure_performance
        @decorators.measure_performance
        def dummy_function():
            return {'result': 42}
        
        result = dummy_function()
        assert 'time_ms' in result, "measure_performance não adicionou 'time_ms'"
        assert 'memory_kb' in result, "measure_performance não adicionou 'memory_kb'"
        
        print("✅ @measure_performance testado e funcionando")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro ao importar módulo: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Validação falhou: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False


def main():
    """Executa todos os testes."""
    print("\n" + "=" * 70)
    print("🧪 VALIDAÇÃO DA FASE 0 - SETUP INICIAL")
    print("=" * 70)
    
    tests = [
        ("Estrutura de Diretórios", test_directory_structure),
        ("Dataset JSON", test_dataset),
        ("Módulo config.py", test_config_module),
        ("Módulo decorators.py", test_decorators_module)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERRO CRÍTICO em '{test_name}': {e}")
            results.append(False)
    
    # Resumo final
    print_header("📊 RESUMO DOS TESTES")
    
    total_tests = len(tests)
    passed_tests = sum(results)
    failed_tests = total_tests - passed_tests
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSOU" if results[i] else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Resultados: {passed_tests}/{total_tests} testes passaram")
    
    if passed_tests == total_tests:
        print("\n🎉 FASE 0 COMPLETA E VALIDADA COM SUCESSO!")
        print("✅ Pronto para iniciar a FASE 1: Estruturas de Dados + Validação")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} teste(s) falharam. Corrija os erros antes de prosseguir.")
        return 1


if __name__ == '__main__':
    sys.exit(main())