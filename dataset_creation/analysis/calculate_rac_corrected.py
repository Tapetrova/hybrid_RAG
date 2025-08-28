#!/usr/bin/env python3
"""
Calculate Corrected Robustness Across Categories (RAC) metric using proper WFAS values
"""

import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_and_calculate_wfas():
    """Load hallucination results and calculate proper WFAS scores"""
    
    # Load the dataset for category information
    with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data/apqc_auto.json', 'r') as f:
        dataset = json.load(f)
    
    # Create question_id to category mapping
    question_categories = {q['id']: q['category'] for q in dataset['questions']}
    
    # Load the full hallucination results with claim-level data
    with open('hallucination_FULL_API_706_results_20250821_231422.json', 'r') as f:
        hall_data = json.load(f)
    
    methods = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    categories = ['factual', 'causal', 'diagnostic', 'comparative']
    
    # Initialize metrics storage
    wfas_by_category = {method: {cat: [] for cat in categories} for method in methods}
    overall_metrics = {method: {'supported': 0, 'contradicted': 0, 'unverifiable': 0, 'total': 0} 
                      for method in methods}
    
    # Process each result
    for result in hall_data['results']:
        qid = result['question_id']
        category = question_categories.get(qid, 'unknown')
        
        if category in categories:
            for method in methods:
                if method in result.get('metrics', {}):
                    metrics = result['metrics'][method]
                    if metrics.get('total_claims', 0) > 0:
                        # Calculate WFAS for this question
                        supported = metrics.get('supported', 0)
                        contradicted = metrics.get('contradicted', 0)
                        unverifiable = metrics.get('unverifiable', 0)
                        total = metrics['total_claims']
                        
                        # WFAS = 100 - (2.5*contradicted + 1*unverifiable)/(3.5*total) * 100
                        whr = ((2.5 * contradicted + 1 * unverifiable) / (3.5 * total)) * 100
                        wfas = 100 - whr
                        
                        wfas_by_category[method][category].append(wfas)
                        
                        # Accumulate for overall
                        overall_metrics[method]['supported'] += supported
                        overall_metrics[method]['contradicted'] += contradicted
                        overall_metrics[method]['unverifiable'] += unverifiable
                        overall_metrics[method]['total'] += total
    
    # Calculate mean WFAS for each category and overall
    category_wfas = {method: {} for method in methods}
    overall_wfas = {}
    
    for method in methods:
        # Category-specific WFAS
        for cat in categories:
            if wfas_by_category[method][cat]:
                category_wfas[method][cat] = np.mean(wfas_by_category[method][cat])
            else:
                category_wfas[method][cat] = 0
        
        # Overall WFAS
        m = overall_metrics[method]
        if m['total'] > 0:
            whr = ((2.5 * m['contradicted'] + 1 * m['unverifiable']) / (3.5 * m['total'])) * 100
            overall_wfas[method] = 100 - whr
        else:
            overall_wfas[method] = 0
    
    return category_wfas, overall_wfas

def calculate_rac_metrics(category_wfas, overall_wfas):
    """Calculate RAC and related robustness metrics"""
    
    methods = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    categories = ['factual', 'causal', 'diagnostic', 'comparative']
    
    robustness_metrics = {}
    
    for method in methods:
        # Get WFAS scores for each category
        wfas_scores = [category_wfas[method][cat] for cat in categories]
        wfas_array = np.array(wfas_scores)
        
        # Calculate robustness metrics
        mean_wfas = np.mean(wfas_array)
        std_dev = np.std(wfas_array)
        cv = std_dev / mean_wfas if mean_wfas > 0 else float('inf')
        
        robustness_metrics[method] = {
            'overall_wfas': overall_wfas[method],
            'mean_wfas': mean_wfas,
            'std_dev': std_dev,
            'coefficient_of_variation': cv,
            'min_max_range': np.max(wfas_array) - np.min(wfas_array),
            'worst_case': np.min(wfas_array),
            'best_case': np.max(wfas_array),
            'consistency_score': 1 - cv if cv != float('inf') else 0,
            'category_scores': {cat: category_wfas[method][cat] for cat in categories}
        }
        
        # Calculate RAC = 0.6 * consistency_score + 0.4 * (worst_case / 100)
        metrics = robustness_metrics[method]
        metrics['RAC'] = (0.6 * metrics['consistency_score'] + 
                         0.4 * (metrics['worst_case'] / 100))
        
        # Alternative: Harmonic mean of category scores
        scores = list(metrics['category_scores'].values())
        if all(s > 0 for s in scores):
            metrics['RAC_harmonic'] = len(scores) / sum(1/s for s in scores)
        else:
            metrics['RAC_harmonic'] = 0
    
    return robustness_metrics

def create_comparison_tables(metrics):
    """Create comparison tables for RAC metrics"""
    
    methods_order = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    method_names = {
        'base_llm': 'BASE LLM',
        'vector_rag': 'VECTOR RAG',
        'graph_rag': 'GRAPH RAG',
        'hybrid_ahs': 'HYBRID AHS'
    }
    
    # Main comparison table
    data = []
    for method in methods_order:
        m = metrics[method]
        data.append({
            'Method': method_names[method],
            'Overall WFAS': f"{m['overall_wfas']:.1f}%",
            'Mean WFAS': f"{m['mean_wfas']:.1f}%",
            'Std Dev': f"{m['std_dev']:.1f}",
            'CV': f"{m['coefficient_of_variation']:.3f}",
            'Range': f"{m['min_max_range']:.1f}",
            'Worst': f"{m['worst_case']:.1f}%",
            'Best': f"{m['best_case']:.1f}%",
            'Consistency': f"{m['consistency_score']:.3f}",
            'RAC': f"{m['RAC']:.3f}",
            'RAC-H': f"{m['RAC_harmonic']:.1f}"
        })
    
    df_main = pd.DataFrame(data)
    
    # Category scores table
    cat_data = []
    for method in methods_order:
        row = {'Method': method_names[method]}
        for cat in ['factual', 'causal', 'diagnostic', 'comparative']:
            row[cat.capitalize()] = f"{metrics[method]['category_scores'][cat]:.1f}"
        cat_data.append(row)
    
    df_cat = pd.DataFrame(cat_data)
    
    return df_main, df_cat

def create_rac_visualizations(metrics):
    """Create visualizations for RAC metrics"""
    
    methods_order = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
    method_names = {
        'base_llm': 'BASE LLM',
        'vector_rag': 'VECTOR RAG',
        'graph_rag': 'GRAPH RAG',
        'hybrid_ahs': 'HYBRID AHS'
    }
    
    # Set style
    plt.rcParams.update({'font.size': 14})
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. RAC Comparison Bar Chart
    ax1 = plt.subplot(2, 3, 1)
    rac_values = [metrics[m]['RAC'] for m in methods_order]
    colors = ['#E57373', '#66BB6A', '#42A5F5', '#9C27B0']
    bars = ax1.bar(range(len(methods_order)), rac_values, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, rac_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xticks(range(len(methods_order)))
    ax1.set_xticklabels([method_names[m] for m in methods_order], rotation=45, ha='right')
    ax1.set_ylabel('RAC Score', fontweight='bold')
    ax1.set_title('Robustness Across Categories (RAC)', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. WFAS by Category Heatmap
    ax2 = plt.subplot(2, 3, 2)
    categories = ['factual', 'causal', 'diagnostic', 'comparative']
    wfas_matrix = []
    for method in methods_order:
        wfas_matrix.append([metrics[method]['category_scores'][cat] for cat in categories])
    
    im = ax2.imshow(wfas_matrix, cmap='RdYlGn', vmin=60, vmax=100, aspect='auto')
    
    # Add text annotations
    for i, method in enumerate(methods_order):
        for j, cat in enumerate(categories):
            val = metrics[method]['category_scores'][cat]
            text = ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                          color='white' if val < 80 else 'black', fontweight='bold')
    
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right')
    ax2.set_yticks(range(len(methods_order)))
    ax2.set_yticklabels([method_names[m] for m in methods_order])
    ax2.set_title('WFAS by Category (%)', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('WFAS (%)', rotation=270, labelpad=20)
    
    # 3. Consistency vs Worst-Case Performance
    ax3 = plt.subplot(2, 3, 3)
    consistency_scores = [metrics[m]['consistency_score'] for m in methods_order]
    worst_case_scores = [metrics[m]['worst_case'] for m in methods_order]
    
    for i, (method, color) in enumerate(zip(methods_order, colors)):
        ax3.scatter(consistency_scores[i], worst_case_scores[i], 
                   s=300, c=color, edgecolor='black', linewidth=2,
                   label=method_names[method], alpha=0.8)
        # Add method label next to point
        ax3.annotate(method_names[method], 
                    (consistency_scores[i], worst_case_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax3.set_xlabel('Consistency Score', fontweight='bold')
    ax3.set_ylabel('Worst-Case Performance (%)', fontweight='bold')
    ax3.set_title('RAC Components Analysis', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right')
    
    # 4. Overall WFAS Comparison
    ax4 = plt.subplot(2, 3, 4)
    overall_values = [metrics[m]['overall_wfas'] for m in methods_order]
    bars = ax4.bar(range(len(methods_order)), overall_values, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, overall_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xticks(range(len(methods_order)))
    ax4.set_xticklabels([method_names[m] for m in methods_order], rotation=45, ha='right')
    ax4.set_ylabel('WFAS (%)', fontweight='bold')
    ax4.set_title('Overall WFAS Performance', fontsize=16, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Variance Analysis
    ax5 = plt.subplot(2, 3, 5)
    std_devs = [metrics[m]['std_dev'] for m in methods_order]
    bars = ax5.bar(range(len(methods_order)), std_devs, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, std_devs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax5.set_xticks(range(len(methods_order)))
    ax5.set_xticklabels([method_names[m] for m in methods_order], rotation=45, ha='right')
    ax5.set_ylabel('Standard Deviation', fontweight='bold')
    ax5.set_title('Performance Variability Across Categories', fontsize=16, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. RAC Components Breakdown
    ax6 = plt.subplot(2, 3, 6)
    width = 0.35
    x = np.arange(len(methods_order))
    
    consistency_contribution = [0.6 * metrics[m]['consistency_score'] for m in methods_order]
    worst_case_contribution = [0.4 * metrics[m]['worst_case']/100 for m in methods_order]
    
    bars1 = ax6.bar(x - width/2, consistency_contribution, width, label='Consistency (60%)', 
                    color='skyblue', edgecolor='black')
    bars2 = ax6.bar(x + width/2, worst_case_contribution, width, label='Worst-Case (40%)', 
                    color='lightcoral', edgecolor='black')
    
    ax6.set_xticks(x)
    ax6.set_xticklabels([method_names[m] for m in methods_order], rotation=45, ha='right')
    ax6.set_ylabel('RAC Contribution', fontweight='bold')
    ax6.set_title('RAC Score Decomposition', fontsize=16, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('RAC (Robustness Across Categories) Analysis - Corrected WFAS', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def save_results(metrics, df_main, df_cat, fig):
    """Save all results to files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics as JSON
    with open(f'rac_corrected_metrics_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save tables as CSV
    df_main.to_csv(f'rac_corrected_comparison_{timestamp}.csv', index=False)
    df_cat.to_csv(f'rac_corrected_categories_{timestamp}.csv', index=False)
    
    # Save visualization
    fig.savefig(f'rac_corrected_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # Save text report
    with open(f'rac_corrected_analysis_{timestamp}.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("CORRECTED RAC (Robustness Across Categories) ANALYSIS\n")
        f.write("Using Weighted FAS (WFAS) with 2.5:1 weights\n")
        f.write("="*80 + "\n\n")
        
        f.write("Definition: RAC = 0.6 * Consistency_Score + 0.4 * (Worst_Case_Performance / 100)\n")
        f.write("Where Consistency_Score = 1 - Coefficient_of_Variation\n\n")
        
        f.write("MAIN COMPARISON TABLE:\n")
        f.write("-"*80 + "\n")
        f.write(df_main.to_string(index=False))
        f.write("\n\n")
        
        f.write("WFAS BY CATEGORY:\n")
        f.write("-"*80 + "\n")
        f.write(df_cat.to_string(index=False))
        f.write("\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*80 + "\n")
        
        # Sort by RAC score
        sorted_methods = sorted(metrics.items(), key=lambda x: x[1]['RAC'], reverse=True)
        
        f.write(f"1. Best RAC Score: {sorted_methods[0][0].upper()} = {sorted_methods[0][1]['RAC']:.3f}\n")
        f.write(f"2. Most Consistent: {min(metrics.items(), key=lambda x: x[1]['coefficient_of_variation'])[0].upper()}\n")
        f.write(f"3. Best Worst-Case: {max(metrics.items(), key=lambda x: x[1]['worst_case'])[0].upper()}\n")
        f.write(f"4. Highest Overall WFAS: {max(metrics.items(), key=lambda x: x[1]['overall_wfas'])[0].upper()}\n")
    
    print(f"\n✅ Corrected RAC analysis saved:")
    print(f"   - rac_corrected_metrics_{timestamp}.json")
    print(f"   - rac_corrected_comparison_{timestamp}.csv")
    print(f"   - rac_corrected_categories_{timestamp}.csv")
    print(f"   - rac_corrected_analysis_{timestamp}.png")
    print(f"   - rac_corrected_analysis_{timestamp}.txt")
    
    return timestamp

def main():
    print("🎯 Calculating CORRECTED RAC with proper WFAS values")
    print("="*80)
    
    # Step 1: Calculate proper WFAS scores
    print("\n1. Calculating WFAS with 2.5:1 weights...")
    category_wfas, overall_wfas = load_and_calculate_wfas()
    
    print("\nOverall WFAS scores:")
    for method in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
        print(f"   {method:12}: {overall_wfas[method]:.1f}%")
    
    # Step 2: Calculate RAC metrics
    print("\n2. Calculating RAC metrics...")
    metrics = calculate_rac_metrics(category_wfas, overall_wfas)
    
    # Step 3: Create comparison tables
    print("\n3. Creating comparison tables...")
    df_main, df_cat = create_comparison_tables(metrics)
    
    print("\n" + "="*80)
    print("CORRECTED RAC RESULTS:")
    print("="*80)
    print(df_main.to_string(index=False))
    
    print("\n" + "-"*80)
    print("WFAS BY CATEGORY:")
    print("-"*80)
    print(df_cat.to_string(index=False))
    
    # Step 4: Create visualizations
    print("\n4. Creating visualizations...")
    fig = create_rac_visualizations(metrics)
    
    # Step 5: Save results
    print("\n5. Saving results...")
    timestamp = save_results(metrics, df_main, df_cat, fig)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    
    # Sort methods by RAC score
    sorted_by_rac = sorted(metrics.items(), key=lambda x: x[1]['RAC'], reverse=True)
    
    print("\nRAC Rankings (higher is better):")
    for i, (method, m) in enumerate(sorted_by_rac, 1):
        print(f"{i}. {method.upper():12} - RAC: {m['RAC']:.3f}, "
              f"Overall WFAS: {m['overall_wfas']:.1f}%, "
              f"Consistency: {m['consistency_score']:.3f}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()