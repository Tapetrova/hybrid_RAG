#!/usr/bin/env python3
"""
Create a scientific bar chart for category distribution with classification uncertainty
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

# Disable interactive mode
matplotlib.use('Agg')

# Set publication-quality parameters
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.5,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 22,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def calculate_category_stats_with_uncertainty():
    """Calculate category distribution with classification uncertainty"""
    
    # Load the dataset
    with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data/apqc_auto.json', 'r') as f:
        data = json.load(f)
    
    questions = data['questions']
    total_questions = len(questions)
    
    # Count categories
    categories = {}
    for q in questions:
        cat = q.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    # Based on our validation: 90.7% accuracy with κ=0.852
    # This means ~9.3% misclassification rate
    misclassification_rate = 0.093
    
    # Calculate expected counts and uncertainty
    category_stats = {}
    
    for cat, count in categories.items():
        # Observed count
        observed = count
        
        # Expected true count considering misclassification
        # Using binomial confidence interval
        proportion = count / total_questions
        
        # Standard error for proportion
        se_proportion = np.sqrt(proportion * (1 - proportion) / total_questions)
        
        # Add classification uncertainty (based on 9.3% error rate)
        classification_se = np.sqrt(count * misclassification_rate * (1 - misclassification_rate))
        
        # Combined uncertainty (classification + sampling)
        total_se = np.sqrt((se_proportion * total_questions)**2 + classification_se**2)
        
        # 95% confidence interval
        ci_95 = 1.96 * total_se
        
        category_stats[cat] = {
            'count': observed,
            'percentage': proportion * 100,
            'se': total_se,
            'ci_lower': max(0, observed - ci_95),
            'ci_upper': observed + ci_95,
            'ci_95': ci_95
        }
    
    return category_stats, total_questions

def create_bar_chart(category_stats, total_questions):
    """Create professional bar chart with error bars"""
    
    # Order categories by count
    ordered_cats = sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    categories = [cat.capitalize() for cat, _ in ordered_cats]
    counts = [stats['count'] for _, stats in ordered_cats]
    errors = [stats['ci_95'] for _, stats in ordered_cats]
    percentages = [stats['percentage'] for _, stats in ordered_cats]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for categories - professional palette
    colors = ['#2E7D32', '#1565C0', '#E65100', '#6A1B9A']  # Green, Blue, Orange, Purple
    
    # Left panel: Absolute counts with error bars
    x_pos = np.arange(len(categories))
    bars1 = ax1.bar(x_pos, counts, yerr=errors, capsize=8, 
                    color=colors, alpha=0.8, edgecolor='black', 
                    linewidth=2, error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels on bars
    for bar, count, err in zip(bars1, counts, errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + err + 5,
                f'{int(count)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=14)
    
    ax1.set_xlabel('Question Category', fontweight='bold')
    ax1.set_ylabel('Number of Questions', fontweight='bold')
    ax1.set_title('(a) Absolute Distribution', fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, rotation=0)
    ax1.set_ylim(0, max(counts) * 1.15)
    
    # Add total N annotation
    ax1.text(0.02, 0.98, f'N = {total_questions}', transform=ax1.transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Right panel: Percentages with error bars
    percentage_errors = [(err/total_questions) * 100 for err in errors]
    bars2 = ax2.bar(x_pos, percentages, yerr=percentage_errors, capsize=8,
                    color=colors, alpha=0.8, edgecolor='black',
                    linewidth=2, error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add percentage labels
    for bar, pct, err in zip(bars2, percentages, percentage_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + err + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=14)
    
    ax2.set_xlabel('Question Category', fontweight='bold')
    ax2.set_ylabel('Percentage of Dataset (%)', fontweight='bold')
    ax2.set_title('(b) Relative Distribution', fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=0)
    ax2.set_ylim(0, max(percentages) * 1.15)
    
    # Add classification accuracy note
    ax2.text(0.98, 0.02, 'Error bars: 95% CI\n(including 9.3% classification uncertainty)',
            transform=ax2.transAxes, fontsize=11, ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Overall title
    fig.suptitle('Question Category Distribution with Classification Uncertainty', 
                fontsize=22, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig

def create_stacked_chart_with_uncertainty(category_stats, total_questions):
    """Create a horizontal stacked bar chart showing distribution and uncertainty"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Order categories
    ordered_cats = sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # Calculate positions
    positions = []
    widths = []
    colors_list = []
    labels = []
    
    color_map = {
        'factual': '#2E7D32',      # Green
        'diagnostic': '#1565C0',    # Blue  
        'causal': '#E65100',        # Orange
        'comparative': '#6A1B9A'    # Purple
    }
    
    current_pos = 0
    for cat, stats in ordered_cats:
        width = stats['percentage']
        positions.append(current_pos + width/2)
        widths.append(width)
        colors_list.append(color_map.get(cat, '#666666'))
        labels.append(cat.capitalize())
        current_pos += width
    
    # Create horizontal bars
    y_pos = 0.5
    height = 0.4
    
    for i, (pos, width, color, label) in enumerate(zip(positions, widths, colors_list, labels)):
        # Main bar
        rect = ax.barh(y_pos, width, height=height, left=pos-width/2,
                      color=color, alpha=0.8, edgecolor='black', linewidth=2,
                      label=label)
        
        # Add percentage text
        if width > 5:  # Only show text if segment is large enough
            ax.text(pos, y_pos, f'{label}\n{width:.1f}%\n(n={ordered_cats[i][1]["count"]})',
                   ha='center', va='center', fontweight='bold', fontsize=12,
                   color='white' if width > 15 else 'black')
    
    # Add uncertainty bands
    current_pos = 0
    for cat, stats in ordered_cats:
        width = stats['percentage']
        uncertainty = (stats['ci_95'] / total_questions) * 100
        
        # Add uncertainty band
        ax.barh(y_pos, uncertainty*2, height=height*0.1, 
               left=current_pos + width/2 - uncertainty,
               color='red', alpha=0.5)
        
        current_pos += width
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Percentage of Dataset (%)', fontweight='bold', fontsize=18)
    ax.set_yticks([])
    ax.set_title('Question Category Distribution (N=706)', fontweight='bold', fontsize=20, pad=20)
    
    # Add legend
    ax.text(50, -0.15, 'Red bands indicate classification uncertainty (±9.3%)',
           ha='center', transform=ax.transData, fontsize=12, style='italic')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

def create_professional_vertical_chart(category_stats, total_questions):
    """Create a single professional vertical bar chart"""
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Order categories
    ordered_cats = sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    
    categories = []
    counts = []
    errors = []
    percentages = []
    
    for cat, stats in ordered_cats:
        categories.append(cat.capitalize())
        counts.append(stats['count'])
        errors.append(stats['ci_95'])
        percentages.append(stats['percentage'])
    
    # Professional color scheme
    colors = ['#2E7D32', '#1565C0', '#E65100', '#6A1B9A']
    
    x_pos = np.arange(len(categories))
    width = 0.6
    
    # Create bars with error bars
    bars = ax.bar(x_pos, counts, width=width, yerr=errors, capsize=10,
                  color=colors, alpha=0.85, edgecolor='black', linewidth=2.5,
                  error_kw={'linewidth': 2.5, 'ecolor': 'black', 'capthick': 2.5})
    
    # Add value labels on top of bars
    for i, (bar, count, err, pct) in enumerate(zip(bars, counts, errors, percentages)):
        height = bar.get_height()
        # Count on top
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 8,
               f'n={int(count)}', ha='center', va='bottom',
               fontweight='bold', fontsize=15)
        # Percentage inside bar (if space allows)
        if height > 50:
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{pct:.1f}%', ha='center', va='center',
                   fontweight='bold', fontsize=16, color='white')
    
    # Customize axes
    ax.set_xlabel('Question Category', fontweight='bold', fontsize=20)
    ax.set_ylabel('Number of Questions', fontweight='bold', fontsize=20)
    ax.set_title('Question Category Distribution', 
                fontweight='bold', fontsize=22, pad=25)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=18, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.18)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    ax.set_axisbelow(True)
    
    # Add only total N as simple text (no box) in top right corner
    ax.text(0.98, 0.97, f'N = {total_questions}', transform=ax.transAxes, fontsize=14,
           verticalalignment='top', horizontalalignment='right', fontweight='bold')
    
    # Remove note - cleaner look
    
    # Make the plot cleaner
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    print("Creating scientific category distribution charts...")
    
    # Calculate statistics with uncertainty
    category_stats, total_questions = calculate_category_stats_with_uncertainty()
    
    # Print summary
    print("\nCategory Distribution with Classification Uncertainty:")
    print("="*60)
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"{cat.capitalize():12} : {stats['count']:3d} ({stats['percentage']:5.1f}% ± {(stats['ci_95']/total_questions*100):.1f}%)")
    
    # Create visualizations
    print("\nGenerating charts...")
    
    # 1. Double panel chart (absolute + percentage)
    fig1 = create_bar_chart(category_stats, total_questions)
    fig1.savefig('category_distribution_scientific_double.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: category_distribution_scientific_double.png")
    
    # 2. Stacked horizontal chart
    fig2 = create_stacked_chart_with_uncertainty(category_stats, total_questions)
    fig2.savefig('category_distribution_stacked.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: category_distribution_stacked.png")
    
    # 3. Professional vertical chart (main one to replace pie chart)
    fig3 = create_professional_vertical_chart(category_stats, total_questions)
    fig3.savefig('chart1_category_distribution_scientific.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: chart1_category_distribution_scientific.png")
    
    print("\n✅ All scientific charts created successfully!")
    print("\nRecommendation: Use 'chart1_category_distribution_scientific.png' to replace the pie chart")

if __name__ == "__main__":
    main()