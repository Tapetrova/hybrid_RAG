#!/usr/bin/env python3
"""
Create publication-ready charts for RAC and Consistency Score
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

# Disable interactive mode
matplotlib.use('Agg')

# Set larger font sizes globally
plt.rcParams.update({'font.size': 18})

def load_metrics():
    """Load the corrected RAC metrics"""
    # Find the latest corrected metrics file
    import glob
    import os
    
    files = glob.glob('rac_corrected_metrics_*.json')
    if not files:
        print("Error: No corrected metrics file found. Run calculate_rac_corrected.py first.")
        return None
    
    latest_file = max(files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def create_rac_bar_chart(metrics):
    """Create a professional RAC comparison bar chart"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    methods_order = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    method_names = {
        'base_llm': 'BASE LLM',
        'vector_rag': 'VECTOR RAG',
        'graph_rag': 'GRAPH RAG',
        'hybrid_ahs': 'HYBRID AHS'
    }
    
    # RAC values
    rac_values = [metrics[m]['RAC'] for m in methods_order]
    
    # Colors - matching the style from other charts
    colors = ['#E57373', '#66BB6A', '#42A5F5', '#9C27B0']
    
    # Create bars
    x_pos = np.arange(len(methods_order))
    bars = ax.bar(x_pos, rac_values, color=colors, edgecolor='black', 
                  linewidth=2.5, width=0.6, alpha=0.9)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, rac_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=22, fontweight='bold')
    
    # Add a subtle pattern to the highest bar
    max_idx = rac_values.index(max(rac_values))
    bars[max_idx].set_hatch('//')
    bars[max_idx].set_edgecolor('gold')
    bars[max_idx].set_linewidth(3)
    
    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_names[m] for m in methods_order], 
                       fontsize=20, fontweight='bold')
    ax.set_ylabel('RAC Score', fontsize=24, fontweight='bold')
    ax.set_title('Robustness Across Categories (RAC)', 
                 fontsize=26, fontweight='bold', pad=25)
    
    # Set y-axis limits and ticks
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{i:.1f}' for i in np.arange(0, 1.1, 0.1)], fontsize=18)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Add horizontal reference lines
    ax.axhline(y=0.9, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.axhline(y=0.85, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Add annotation for the best performer
    if max_idx >= 0:
        ax.annotate('Best Robustness', 
                   xy=(x_pos[max_idx], rac_values[max_idx] + 0.01),
                   xytext=(x_pos[max_idx], rac_values[max_idx] + 0.08),
                   ha='center', fontsize=16, fontweight='bold',
                   color='darkgreen',
                   arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    
    plt.tight_layout()
    return fig

def create_consistency_chart(metrics):
    """Create a professional Consistency Score comparison chart"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    methods_order = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    method_names = {
        'base_llm': 'BASE LLM',
        'vector_rag': 'VECTOR RAG',
        'graph_rag': 'GRAPH RAG',
        'hybrid_ahs': 'HYBRID AHS'
    }
    
    # Consistency values
    consistency_values = [metrics[m]['consistency_score'] for m in methods_order]
    
    # Colors
    colors = ['#E57373', '#66BB6A', '#42A5F5', '#9C27B0']
    
    # Create bars
    x_pos = np.arange(len(methods_order))
    bars = ax.bar(x_pos, consistency_values, color=colors, edgecolor='black', 
                  linewidth=2.5, width=0.6, alpha=0.9)
    
    # Add value labels
    for bar, val in zip(bars, consistency_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=22, fontweight='bold')
    
    # Highlight most consistent
    max_idx = consistency_values.index(max(consistency_values))
    bars[max_idx].set_hatch('\\\\')
    bars[max_idx].set_edgecolor('darkblue')
    bars[max_idx].set_linewidth(3)
    
    # Customize axes
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_names[m] for m in methods_order], 
                       fontsize=20, fontweight='bold')
    ax.set_ylabel('Consistency Score', fontsize=24, fontweight='bold')
    ax.set_title('Performance Consistency Across Categories', 
                 fontsize=26, fontweight='bold', pad=25)
    
    # Set y-axis limits
    ax.set_ylim(0.94, 1.0)
    ax.set_yticks(np.arange(0.94, 1.01, 0.01))
    ax.set_yticklabels([f'{i:.2f}' for i in np.arange(0.94, 1.01, 0.01)], fontsize=18)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Add annotation
    ax.annotate('Most Consistent', 
               xy=(x_pos[max_idx], consistency_values[max_idx] + 0.002),
               xytext=(x_pos[max_idx], consistency_values[max_idx] + 0.01),
               ha='center', fontsize=16, fontweight='bold',
               color='darkblue',
               arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))
    
    # Add subtitle explaining the metric
    ax.text(0.5, -0.15, 'Consistency Score = 1 - CV (Coefficient of Variation)\nHigher values indicate more stable performance across question categories',
            ha='center', transform=ax.transAxes, fontsize=14, style='italic')
    
    plt.tight_layout()
    return fig

def create_combined_rac_components(metrics):
    """Create a combined view of RAC components"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    methods_order = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    method_names = {
        'base_llm': 'BASE LLM',
        'vector_rag': 'VECTOR RAG',
        'graph_rag': 'GRAPH RAG',
        'hybrid_ahs': 'HYBRID AHS'
    }
    
    colors = ['#E57373', '#66BB6A', '#42A5F5', '#9C27B0']
    x_pos = np.arange(len(methods_order))
    
    # Left panel: Stacked bar showing RAC components
    consistency_contrib = [0.6 * metrics[m]['consistency_score'] for m in methods_order]
    worst_case_contrib = [0.4 * metrics[m]['worst_case']/100 for m in methods_order]
    
    bars1 = ax1.bar(x_pos, consistency_contrib, color='lightblue', 
                    edgecolor='black', linewidth=2, label='Consistency (60%)', alpha=0.8)
    bars2 = ax1.bar(x_pos, worst_case_contrib, bottom=consistency_contrib,
                    color='lightcoral', edgecolor='black', linewidth=2, 
                    label='Worst-Case (40%)', alpha=0.8)
    
    # Add total RAC values on top
    for i, method in enumerate(methods_order):
        total = consistency_contrib[i] + worst_case_contrib[i]
        ax1.text(i, total + 0.01, f'RAC\n{total:.3f}', 
                ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([method_names[m] for m in methods_order], 
                        fontsize=18, fontweight='bold')
    ax1.set_ylabel('RAC Score Components', fontsize=20, fontweight='bold')
    ax1.set_title('RAC Score Decomposition', fontsize=22, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=16, loc='upper left')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # Right panel: Scatter plot of Consistency vs Worst-Case
    consistency_scores = [metrics[m]['consistency_score'] for m in methods_order]
    worst_case_scores = [metrics[m]['worst_case'] for m in methods_order]
    rac_scores = [metrics[m]['RAC'] for m in methods_order]
    
    # Create scatter plot with size based on RAC
    for i, (method, color) in enumerate(zip(methods_order, colors)):
        size = 2000 * rac_scores[i]  # Size proportional to RAC
        ax2.scatter(consistency_scores[i], worst_case_scores[i], 
                   s=size, c=[color], edgecolor='black', linewidth=2.5,
                   alpha=0.7, label=method_names[method])
        
        # Add labels
        ax2.annotate(f'{method_names[method]}\nRAC={rac_scores[i]:.3f}', 
                    (consistency_scores[i], worst_case_scores[i]),
                    xytext=(8, 8), textcoords='offset points', 
                    fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Consistency Score', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Worst-Case Performance (%)', fontsize=20, fontweight='bold')
    ax2.set_title('RAC Components Distribution', fontsize=22, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.955, 1.0)
    ax2.set_ylim(65, 92)
    
    # Add diagonal lines for constant RAC values
    for rac_val in [0.85, 0.90, 0.95]:
        x_line = np.linspace(0.955, 1.0, 100)
        y_line = (rac_val - 0.6 * x_line) / 0.4 * 100
        ax2.plot(x_line, y_line, '--', alpha=0.3, color='gray', linewidth=1)
        ax2.text(0.958, (rac_val - 0.6 * 0.958) / 0.4 * 100, f'RAC={rac_val}', 
                fontsize=10, color='gray', rotation=-45)
    
    plt.suptitle('RAC (Robustness Across Categories) Analysis', 
                 fontsize=24, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_performance_stability_radar(metrics):
    """Create a radar chart showing multiple stability metrics"""
    
    from math import pi
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    methods_order = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    method_names = {
        'base_llm': 'BASE LLM',
        'vector_rag': 'VECTOR RAG',
        'graph_rag': 'GRAPH RAG',
        'hybrid_ahs': 'HYBRID AHS'
    }
    
    # Metrics to show (all normalized to 0-1 scale)
    categories = ['RAC Score', 'Consistency', 'Worst-Case\n(normalized)', 
                 'Overall WFAS\n(normalized)', 'Best-Case\n(normalized)']
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Colors
    colors = ['#E57373', '#66BB6A', '#42A5F5', '#9C27B0']
    
    # Plot for each method
    for idx, (method, color) in enumerate(zip(methods_order, colors)):
        m = metrics[method]
        values = [
            m['RAC'],
            m['consistency_score'],
            m['worst_case'] / 100,  # Normalize to 0-1
            m['overall_wfas'] / 100,  # Normalize to 0-1
            m['best_case'] / 100  # Normalize to 0-1
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=method_names[method],
                color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
    ax.grid(True, linewidth=1, alpha=0.5)
    
    # Add title and legend
    plt.title('Multi-Dimensional Robustness Comparison', 
             fontsize=22, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), 
              fontsize=16, framealpha=0.95)
    
    plt.tight_layout()
    return fig

def main():
    print("🎯 Creating RAC and Consistency Score visualizations")
    print("="*60)
    
    # Load metrics
    metrics = load_metrics()
    if not metrics:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create charts
    print("\n1. Creating RAC bar chart...")
    fig1 = create_rac_bar_chart(metrics)
    fig1.savefig(f'rac_bar_chart_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: rac_bar_chart_{timestamp}.png")
    
    print("\n2. Creating Consistency Score chart...")
    fig2 = create_consistency_chart(metrics)
    fig2.savefig(f'consistency_score_chart_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: consistency_score_chart_{timestamp}.png")
    
    print("\n3. Creating combined RAC components chart...")
    fig3 = create_combined_rac_components(metrics)
    fig3.savefig(f'rac_components_combined_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: rac_components_combined_{timestamp}.png")
    
    print("\n4. Creating performance stability radar chart...")
    fig4 = create_performance_stability_radar(metrics)
    fig4.savefig(f'performance_stability_radar_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: performance_stability_radar_{timestamp}.png")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF METRICS:")
    print("="*60)
    
    methods_order = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    method_names = {
        'base_llm': 'BASE LLM',
        'vector_rag': 'VECTOR RAG',
        'graph_rag': 'GRAPH RAG',
        'hybrid_ahs': 'HYBRID AHS'
    }
    
    print("\nRAC Scores (sorted):")
    sorted_methods = sorted([(m, metrics[m]['RAC']) for m in methods_order], 
                           key=lambda x: x[1], reverse=True)
    for method, rac in sorted_methods:
        print(f"  {method_names[method]:12} : {rac:.3f}")
    
    print("\nConsistency Scores (sorted):")
    sorted_consistency = sorted([(m, metrics[m]['consistency_score']) for m in methods_order], 
                               key=lambda x: x[1], reverse=True)
    for method, cons in sorted_consistency:
        print(f"  {method_names[method]:12} : {cons:.3f}")
    
    print("\n✅ All charts created successfully!")

if __name__ == "__main__":
    main()