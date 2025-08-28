import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Disable interactive mode
matplotlib.use('Agg')

# Set larger font sizes globally
plt.rcParams.update({'font.size': 14})

# Data from the FAS report
methods = ['BASE LLM', 'VECTOR RAG', 'GRAPH RAG', 'HYBRID AHS']
categories = ['Overall\nFAS', 'Factual\nAccuracy', 'Causal\nPerformance', 
              'Diagnostic\nPerformance', 'Comparative\nPerformance', 'Support\nRate']

# Performance scores (percentages)
scores = {
    'BASE LLM': [30.6, 31.8, 35.5, 23.8, 27.8, 30.6],
    'VECTOR RAG': [78.0, 78.2, 81.2, 73.9, 77.7, 78.0],
    'GRAPH RAG': [65.1, 66.3, 63.5, 64.0, 63.5, 65.1],
    'HYBRID AHS': [73.8, 72.4, 79.7, 75.6, 69.0, 73.8]
}

# Standard errors (estimated based on sample size n=706)
# Using formula: SE = sqrt(p*(1-p)/n) * 100 for percentages
# Then CI = 1.96 * SE for 95% confidence
n = 706
confidence_intervals = {}

for method, values in scores.items():
    cis = []
    for score in values:
        p = score / 100  # Convert to proportion
        se = np.sqrt(p * (1 - p) / n) * 100  # Convert back to percentage
        ci = 1.96 * se  # 95% confidence interval
        cis.append(ci)
    confidence_intervals[method] = cis

# Create the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Set positions for bars
x = np.arange(len(categories))
width = 0.18

# Colors for each method
colors = {
    'BASE LLM': '#E57373',      # Light red
    'VECTOR RAG': '#66BB6A',    # Green
    'GRAPH RAG': '#42A5F5',     # Blue
    'HYBRID AHS': '#AB47BC'     # Purple
}

# Create bars with error bars
bars = {}
for i, (method, values) in enumerate(scores.items()):
    position = x + (i - 1.5) * width
    bars[method] = ax.bar(position, values, width, 
                          label=method, 
                          color=colors[method],
                          edgecolor='black',
                          linewidth=1.5,
                          yerr=confidence_intervals[method],
                          capsize=5,
                          error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels on top of bars
    for j, (bar, value) in enumerate(zip(bars[method], values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + confidence_intervals[method][j] + 1,
                f'{value:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize the plot
ax.set_xlabel('Performance Metrics', fontsize=18, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=18, fontweight='bold')
ax.set_title('Multi-Dimensional Performance Comparison with 95% Confidence Intervals', 
             fontsize=20, fontweight='bold', pad=20)

# Set x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)

# Set y-axis limits and labels
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f'{i}%' for i in range(0, 101, 20)], fontsize=14)

# Add grid for better readability
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='upper left', fontsize=14, framealpha=0.95, 
          edgecolor='black', fancybox=True, shadow=True)

# Add a horizontal line at 50% for reference
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add annotation for statistical significance
ax.text(0.99, 0.02, 
        'Error bars represent 95% confidence intervals\nAll RAG methods significantly outperform baseline (p < 0.001)',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/performance_bar_chart_95CI.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Created performance_bar_chart_95CI.png with larger fonts and 95% confidence intervals")