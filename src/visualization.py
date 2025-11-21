"""
Visualizações para o Motor de Orientação de Habilidades (MOH)

Este módulo fornece funções para criar gráficos e visualizações dos resultados:
1. Histogramas e distribuições (Monte Carlo)
2. Gráficos de barras comparativos (Permutações, Guloso vs Ótimo)
3. Box plots (Análise de dispersão)
4. Gráficos de linha (Evolução de valores)
5. Heatmaps (Cenários de mercado)
6. Dashboard consolidado

Funções principais:
- plot_monte_carlo_distribution: Histograma da simulação MC
- plot_permutations_comparison: Comparação de custos
- plot_greedy_vs_optimal: Comparação de algoritmos
- plot_market_scenarios: Análise de cenários
- create_dashboard: Dashboard completo
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Configurações globais
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Cores consistentes
COLORS = {
    'deterministic': '#3498db',  # Azul
    'stochastic': '#e74c3c',     # Vermelho
    'optimal': '#2ecc71',        # Verde
    'greedy': '#f39c12',         # Laranja
    'baseline': '#95a5a6',       # Cinza
    'primary': '#3498db',
    'secondary': '#9b59b6'       # Roxo
}


def setup_plot_style(figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Configura estilo padrão para plots.
    
    Args:
        figsize: Tamanho da figura (largura, altura)
    
    Returns:
        Tuple[Figure, Axes]: Figura e eixos do matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')
    return fig, ax


def plot_monte_carlo_distribution(
    values: List[float],
    deterministic_value: Optional[float] = None,
    title: str = "Distribuição Monte Carlo",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plota histograma da distribuição Monte Carlo.
    
    Args:
        values: Lista de valores da simulação
        deterministic_value: Valor determinístico (opcional)
        title: Título do gráfico
        save_path: Caminho para salvar (opcional)
    
    Returns:
        Figure: Figura do matplotlib
    """
    fig, ax = setup_plot_style(figsize=(12, 6))
    
    values_array = np.array(values)
    mean_val = np.mean(values_array)
    std_val = np.std(values_array)
    
    # Histograma
    n, bins, patches = ax.hist(
        values_array,
        bins=50,
        color=COLORS['stochastic'],
        alpha=0.7,
        edgecolor='black',
        label='Monte Carlo'
    )
    
    # Linha vertical da média
    ax.axvline(
        mean_val,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'E[X] = {mean_val:.2f}'
    )
    
    # Área do IC 95%
    ci_95 = (mean_val - 1.96*std_val, mean_val + 1.96*std_val)
    ax.axvspan(ci_95[0], ci_95[1], alpha=0.2, color='yellow', label='IC 95%')
    
    # Linha do determinístico (se fornecido)
    if deterministic_value is not None:
        ax.axvline(
            deterministic_value,
            color=COLORS['deterministic'],
            linestyle='-',
            linewidth=2,
            label=f'Determinístico = {deterministic_value:.2f}'
        )
    
    ax.set_xlabel('Valor', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    
    # Estatísticas no canto
    stats_text = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nn = {len(values)}'
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_permutations_comparison(
    costs: List[float],
    top_n: int = 10,
    title: str = "Comparação de Custos - Permutações",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plota comparação de custos das permutações.
    
    Args:
        costs: Lista de custos (ordenada, melhor primeiro)
        top_n: Número de melhores/piores a destacar
        title: Título do gráfico
        save_path: Caminho para salvar
    
    Returns:
        Figure: Figura do matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Top N melhores vs Top N piores
    best_n = costs[:top_n]
    worst_n = costs[-top_n:][::-1]
    
    x = np.arange(top_n)
    width = 0.35
    
    ax1.bar(x - width/2, best_n, width, label='Melhores', color=COLORS['optimal'], alpha=0.8)
    ax1.bar(x + width/2, worst_n, width, label='Piores', color=COLORS['stochastic'], alpha=0.8)
    
    ax1.set_xlabel('Posição', fontsize=12)
    ax1.set_ylabel('Custo (horas)', fontsize=12)
    ax1.set_title(f'Top {top_n} Melhores vs Piores', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i+1}' for i in range(top_n)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Distribuição completa (box plot + histograma)
    ax2_twin = ax2.twinx()
    
    # Box plot
    bp = ax2.boxplot(
        costs,
        vert=True,
        patch_artist=True,
        widths=0.3,
        positions=[0.3]
    )
    bp['boxes'][0].set_facecolor(COLORS['primary'])
    bp['boxes'][0].set_alpha(0.6)
    
    # Histograma horizontal
    ax2_twin.hist(costs, bins=30, orientation='horizontal', alpha=0.5, color=COLORS['secondary'])
    
    ax2.set_ylabel('Custo (horas)', fontsize=12)
    ax2_twin.set_xlabel('Frequência', fontsize=12)
    ax2.set_title('Distribuição Completa', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Estatísticas
    mean_cost = np.mean(costs)
    median_cost = np.median(costs)
    ax2.axhline(mean_cost, color='red', linestyle='--', label=f'Média: {mean_cost:.0f}h')
    ax2.axhline(median_cost, color='green', linestyle=':', label=f'Mediana: {median_cost:.0f}h')
    ax2.legend(loc='upper right')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_greedy_vs_optimal(
    greedy_value: float,
    greedy_time: float,
    optimal_value: float,
    optimal_time: float,
    title: str = "Guloso vs Ótimo",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plota comparação entre algoritmo guloso e ótimo.
    
    Args:
        greedy_value: Valor da solução gulosa
        greedy_time: Tempo da solução gulosa
        optimal_value: Valor da solução ótima
        optimal_time: Tempo da solução ótima
        title: Título do gráfico
        save_path: Caminho para salvar
    
    Returns:
        Figure: Figura do matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['Guloso', 'Ótimo']
    values = [greedy_value, optimal_value]
    times = [greedy_time, optimal_time]
    colors = [COLORS['greedy'], COLORS['optimal']]
    
    # Gráfico 1: Valores
    bars1 = ax1.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Valor (Adaptabilidade)', fontsize=12)
    ax1.set_title('Comparação de Valores', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores nas barras
    for bar, val in zip(bars1, values):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    # Gráfico 2: Tempos
    bars2 = ax2.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Tempo (horas)', fontsize=12)
    ax2.set_title('Comparação de Tempos', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores nas barras
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{time:.1f}h',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    # Qualidade da solução gulosa
    quality = (greedy_value / optimal_value * 100) if optimal_value > 0 else 0
    fig.text(
        0.5, 0.02,
        f'Qualidade do Guloso: {quality:.1f}% do Ótimo',
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3)
    )
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_market_scenarios(
    expected_values_per_scenario: Dict[str, float],
    scenarios_probabilities: Dict[str, float],
    title: str = "Análise de Cenários de Mercado",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plota análise de cenários de mercado.
    
    Args:
        expected_values_per_scenario: Valor esperado em cada cenário
        scenarios_probabilities: Probabilidade de cada cenário
        title: Título do gráfico
        save_path: Caminho para salvar
    
    Returns:
        Figure: Figura do matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = list(expected_values_per_scenario.keys())
    values = list(expected_values_per_scenario.values())
    probs = [scenarios_probabilities[s] for s in scenarios]
    
    # Gráfico 1: Valores por cenário (barras com probabilidades)
    bars = ax1.bar(
        scenarios,
        values,
        color=[plt.cm.viridis(p) for p in probs],
        alpha=0.8,
        edgecolor='black'
    )
    
    ax1.set_ylabel('Valor Esperado', fontsize=12)
    ax1.set_xlabel('Cenário', fontsize=12)
    ax1.set_title('Valor Esperado por Cenário', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores e probabilidades nas barras
    for bar, val, prob in zip(bars, values, probs):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}\n({prob*100:.0f}%)',
            ha='center', va='bottom', fontsize=9
        )
    
    # Gráfico 2: Contribuição ponderada (valor × probabilidade)
    contributions = [v * p for v, p in zip(values, probs)]
    total_contribution = sum(contributions)
    
    colors_contrib = sns.color_palette("Set2", len(scenarios))
    wedges, texts, autotexts = ax2.pie(
        contributions,
        labels=scenarios,
        autopct='%1.1f%%',
        colors=colors_contrib,
        startangle=90,
        explode=[0.05 if c == max(contributions) else 0 for c in contributions]
    )
    
    # Melhora legibilidade
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax2.set_title('Contribuição Ponderada', fontsize=12, fontweight='bold')
    
    # Valor esperado total
    fig.text(
        0.5, 0.02,
        f'E[Valor Total] = {total_contribution:.2f}',
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_algorithms_complexity_comparison(
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plota comparação de complexidades dos algoritmos.
    
    Args:
        save_path: Caminho para salvar
    
    Returns:
        Figure: Figura do matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Dados dos algoritmos
    algorithms = [
        'DP Knapsack\nO(n×T×C)',
        'Monte Carlo\nO(N×n×T×C)',
        'Permutations\nO(n!)',
        'Greedy\nO(n log n)',
        'Exhaustive\nO(2^n)',
        'Merge Sort\nO(n log n)',
        'Look-ahead\nO(n×k×m)',
        'DP Rec.\nO(C(n,k)×m)'
    ]
    
    desafios = [1, 1, 2, 3, 3, 4, 5, 5]
    
    # Complexidade aproximada (log scale) para n=12
    complexities_log = [
        np.log10(12 * 350 * 30),       # DP Knapsack
        np.log10(1000 * 12 * 350 * 30), # Monte Carlo
        np.log10(np.math.factorial(5)),  # Permutations (n=5)
        np.log10(12 * np.log2(12)),     # Greedy
        np.log10(2**12 * 12),           # Exhaustive (n=12)
        np.log10(12 * np.log2(12)),     # Merge Sort
        np.log10(12 * 2 * 4),           # Look-ahead
        np.log10(220 * 4)               # DP Rec (C(12,3) × 4)
    ]
    
    colors_desafio = {1: '#3498db', 2: '#e74c3c', 3: '#2ecc71', 4: '#f39c12', 5: '#9b59b6'}
    bar_colors = [colors_desafio[d] for d in desafios]
    
    bars = ax.barh(algorithms, complexities_log, color=bar_colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Complexidade (log₁₀ operações)', fontsize=12)
    ax.set_title('Comparação de Complexidades dos Algoritmos', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legenda dos desafios
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_desafio[i], label=f'Desafio {i}', alpha=0.8)
        for i in sorted(colors_desafio.keys())
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_dashboard(
    results: Dict[str, Any],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Cria dashboard consolidado com todos os resultados.
    
    Args:
        results: Dicionário com resultados de todos os desafios
        save_path: Caminho para salvar
    
    Returns:
        Figure: Figura do matplotlib
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Título principal
    fig.suptitle(
        'DASHBOARD - Motor de Orientação de Habilidades (MOH)',
        fontsize=16,
        fontweight='bold'
    )
    
    # Painel 1: Desafio 1 - Monte Carlo
    if 'desafio1' in results and 'stochastic' in results['desafio1']:
        ax1 = fig.add_subplot(gs[0, :2])
        mc_result = results['desafio1']['stochastic'].get('monte_carlo_result', {})
        values = mc_result.get('all_results', [])
        if values:
            mean_val = mc_result.get('expected_value', np.mean(values))
            ax1.hist(values, bins=30, color=COLORS['stochastic'], alpha=0.7, edgecolor='black')
            ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'E[X]={mean_val:.2f}')
            ax1.set_title('Desafio 1: Distribuição Monte Carlo', fontweight='bold')
            ax1.set_xlabel('Valor')
            ax1.set_ylabel('Frequência')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # Painel 2: Desafio 2 - Estatísticas
    if 'desafio2' in results:
        ax2 = fig.add_subplot(gs[0, 2])
        d2 = results['desafio2'].get('statistics', {})
        metrics = ['Melhor', 'Média', 'Pior']
        values_d2 = [
            d2.get('best_cost', 0),
            d2.get('avg_all', 0),
            d2.get('worst_cost', 0)
        ]
        ax2.bar(metrics, values_d2, color=[COLORS['optimal'], COLORS['primary'], COLORS['stochastic']], alpha=0.8)
        ax2.set_title('Desafio 2: Custos', fontweight='bold')
        ax2.set_ylabel('Horas')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(values_d2):
            ax2.text(i, v, f'{v:.0f}h', ha='center', va='bottom', fontweight='bold')
    
    # Painel 3: Desafio 3 - Guloso vs Ótimo
    if 'desafio3' in results:
        ax3 = fig.add_subplot(gs[1, 0])
        d3 = results['desafio3']
        methods = ['Guloso', 'Ótimo']
        values_d3 = [
            d3.get('greedy', {}).get('total_value', 0),
            d3.get('optimal', {}).get('total_value', 0)
        ]
        ax3.bar(methods, values_d3, color=[COLORS['greedy'], COLORS['optimal']], alpha=0.8)
        ax3.set_title('Desafio 3: Guloso vs Ótimo', fontweight='bold')
        ax3.set_ylabel('Valor')
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(values_d3):
            ax3.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Painel 4: Desafio 4 - Tempos
    if 'desafio4' in results:
        ax4 = fig.add_subplot(gs[1, 1])
        d4 = results['desafio4']
        methods_d4 = ['Merge\nSort', 'Native\nSort']
        times_d4 = [
            d4.get('merge_sort_result', {}).get('execution_time', 0) * 1000,
            d4.get('native_sort_result', {}).get('execution_time', 0) * 1000
        ]
        ax4.bar(methods_d4, times_d4, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
        ax4.set_title('Desafio 4: Tempos de Ordenação', fontweight='bold')
        ax4.set_ylabel('Milissegundos')
        ax4.grid(True, alpha=0.3, axis='y')
        for i, t in enumerate(times_d4):
            ax4.text(i, t, f'{t:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Painel 5: Desafio 5 - Cenários
    if 'desafio5' in results:
        ax5 = fig.add_subplot(gs[1, 2])
        d5 = results['desafio5']
        per_scenario = d5.get('recommendation', {}).get('expected_value_per_scenario', {})
        if per_scenario:
            scenarios_names = list(per_scenario.keys())
            scenarios_values = list(per_scenario.values())
            colors_pie = sns.color_palette("Set2", len(scenarios_names))
            ax5.pie(scenarios_values, labels=scenarios_names, autopct='%1.0f%%', colors=colors_pie, startangle=90)
            ax5.set_title('Desafio 5: Cenários', fontweight='bold')
    
    # Painel 6: Resumo Geral
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = "RESUMO GERAL\n" + "="*60 + "\n\n"
    
    if 'desafio1' in results:
        d1_val = results['desafio1'].get('deterministic', {}).get('solution', {}).get('total_value', 0)
        summary_text += f"✓ Desafio 1: Valor Máximo = {d1_val:.2f}\n"
    
    if 'desafio2' in results:
        d2_best = results['desafio2'].get('statistics', {}).get('best_cost', 0)
        summary_text += f"✓ Desafio 2: Melhor Custo = {d2_best:.0f}h\n"
    
    if 'desafio3' in results:
        d3_opt = results['desafio3'].get('optimal', {}).get('total_value', 0)
        summary_text += f"✓ Desafio 3: Valor Ótimo = {d3_opt:.2f}\n"
    
    if 'desafio4' in results:
        d4_ratio = results['desafio4'].get('comparison', {}).get('time_ratio', 0)
        summary_text += f"✓ Desafio 4: Merge Sort = {d4_ratio:.2f}x Native Sort\n"
    
    if 'desafio5' in results:
        d5_skills = results['desafio5'].get('recommendation', {}).get('skills_recommended', [])
        summary_text += f"✓ Desafio 5: Recomendações = {', '.join(d5_skills)}\n"
    
    ax6.text(
        0.1, 0.5, summary_text,
        fontsize=11,
        verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3)
    )
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_all_plots(
    results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Salva todos os gráficos em um diretório.
    
    Args:
        results: Dicionário com resultados de todos os desafios
        output_dir: Diretório de saída
    """
    output_dir.mkdir(exist_ok=True)
    
    print(f"Salvando gráficos em: {output_dir}")
    
    # Desafio 1
    if 'desafio1' in results and 'stochastic' in results['desafio1']:
        mc_result = results['desafio1']['stochastic'].get('monte_carlo_result', {})
        values = mc_result.get('all_results', [])
        if values:
            det_val = results['desafio1'].get('deterministic', {}).get('solution', {}).get('total_value')
            plot_monte_carlo_distribution(
                values, det_val,
                save_path=output_dir / 'desafio1_monte_carlo.png'
            )
            plt.close()
    
    # Desafio 2
    if 'desafio2' in results:
        costs = [c.total_cost if hasattr(c, 'total_cost') else c for c in results['desafio2'].get('all_costs', [])]
        if costs:
            plot_permutations_comparison(
                costs,
                save_path=output_dir / 'desafio2_permutations.png'
            )
            plt.close()
    
    # Desafio 3
    if 'desafio3' in results:
        d3 = results['desafio3']
        plot_greedy_vs_optimal(
            d3.get('greedy', {}).get('total_value', 0),
            d3.get('greedy', {}).get('total_time', 0),
            d3.get('optimal', {}).get('total_value', 0),
            d3.get('optimal', {}).get('total_time', 0),
            save_path=output_dir / 'desafio3_greedy_vs_optimal.png'
        )
        plt.close()
    
    # Desafio 5
    if 'desafio5' in results:
        d5 = results['desafio5']
        per_scenario = d5.get('recommendation', {}).get('expected_value_per_scenario', {})
        scenarios = d5.get('scenarios', [])
        probs = {s['name']: s['probability'] for s in scenarios}
        if per_scenario and probs:
            plot_market_scenarios(
                per_scenario, probs,
                save_path=output_dir / 'desafio5_market_scenarios.png'
            )
            plt.close()
    
    # Complexidades
    plot_algorithms_complexity_comparison(
        save_path=output_dir / 'algorithms_complexity.png'
    )
    plt.close()
    
    # Dashboard
    create_dashboard(results, save_path=output_dir / 'dashboard_completo.png')
    plt.close()
    
    print(f"✅ Gráficos salvos com sucesso!")