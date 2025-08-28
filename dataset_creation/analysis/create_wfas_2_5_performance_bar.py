import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Disable interactive mode
matplotlib.use('Agg')

# Set larger font sizes globally
plt.rcParams.update({'font.size': 20})

# Category-specific data
category_scores = {
    'causal': {
        'base_llm': {'supported': 35.5, 'contradicted': 12.0, 'unverifiable': 52.5},
        'vector_rag': {'supported': 81.2, 'contradicted': 6.5, 'unverifiable': 12.3},
        'graph_rag': {'supported': 63.5, 'contradicted': 8.0, 'unverifiable': 28.5},
        'hybrid_ahs': {'supported': 79.7, 'contradicted': 5.2, 'unverifiable': 15.1}
    },
    'comparative': {
        'base_llm': {'supported': 27.8, 'contradicted': 18.2, 'unverifiable': 54.0},
        'vector_rag': {'supported': 77.7, 'contradicted': 9.1, 'unverifiable': 13.2},
        'graph_rag': {'supported': 63.5, 'contradicted': 10.4, 'unverifiable': 26.1},
        'hybrid_ahs': {'supported': 69.0, 'contradicted': 9.1, 'unverifiable': 21.9}
    },
    'diagnostic': {
        'base_llm': {'supported': 23.8, 'contradicted': 17.8, 'unverifiable': 58.4},
        'vector_rag': {'supported': 73.9, 'contradicted': 10.2, 'unverifiable': 15.9},
        'graph_rag': {'supported': 64.0, 'contradicted': 8.5, 'unverifiable': 27.5},
        'hybrid_ahs': {'supported': 75.6, 'contradicted': 6.8, 'unverifiable': 17.6}
    },
    'factual': {
        'base_llm': {'supported': 31.8, 'contradicted': 15.3, 'unverifiable': 52.9},
        'vector_rag': {'supported': 78.2, 'contradicted': 8.2, 'unverifiable': 13.6},
        'graph_rag': {'supported': 66.3, 'contradicted': 8.2, 'unverifiable': 25.5},
        'hybrid_ahs': {'supported': 72.4, 'contradicted': 6.1, 'unverifiable': 21.5}
    }
}

# Overall scores
overall_scores = {
    'base_llm': {'supported': 30.6, 'contradicted': 15.7, 'unverifiable': 53.6},
    'vector_rag': {'supported': 78.0, 'contradicted': 8.4, 'unverifiable': 13.6},
    'graph_rag': {'supported': 65.1, 'contradicted': 8.6, 'unverifiable': 26.3},
    'hybrid_ahs': {'supported': 73.8, 'contradicted': 6.4, 'unverifiable': 19.8}
}

# Calculate WFAS using 2.5:1 weight scheme
def calculate_wfas_2_5(data):
    whr = (2.5 * data['contradicted'] + 1 * data['unverifiable']) / 3.5
    return 100 - whr

# Calculate support rate (same as supported percentage)
def calculate_support_rate(data):
    return data['supported']

methods = ['BASE LLM', 'VECTOR RAG', 'GRAPH RAG', 'HYBRID AHS']
method_keys = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
categories = [r'$\mathbf{Overall\,FAS}$', 'Factual\nQuestions', 'Causal\nQuestions', 
              'Diagnostic\nQuestions', 'Comparative\nQuestions']

# Calculate scores for each method and category
scores = {}
for i, method in enumerate(methods):
    method_key = method_keys[i]
    scores[method] = []
    
    # Overall WFAS
    scores[method].append(calculate_wfas_2_5(overall_scores[method_key]))
    
    # Category-specific WFAS
    scores[method].append(calculate_wfas_2_5(category_scores['factual'][method_key]))
    scores[method].append(calculate_wfas_2_5(category_scores['causal'][method_key]))
    scores[method].append(calculate_wfas_2_5(category_scores['diagnostic'][method_key]))
    scores[method].append(calculate_wfas_2_5(category_scores['comparative'][method_key]))

# Standard errors (estimated based on sample size n=706)
n = 706
confidence_intervals = {}

for method in methods:
    cis = []
    for score in scores[method]:
        p = score / 100  # Convert to proportion
        se = np.sqrt(p * (1 - p) / n) * 100  # Convert back to percentage
        ci = 1.96 * se  # 95% confidence interval
        cis.append(ci)
    confidence_intervals[method] = cis

# Create the plot
fig, ax = plt.subplots(figsize=(20, 12))

# Set positions for bars
x = np.arange(len(categories))
width = 0.20

# Colors for each method - adjusted for better distinction
colors = {
    'BASE LLM': '#E57373',      # Light red
    'VECTOR RAG': '#66BB6A',    # Green
    'GRAPH RAG': '#42A5F5',     # Blue
    'HYBRID AHS': '#9C27B0'     # Purple
}

# Create bars with error bars
bars = {}
for i, method in enumerate(methods):
    position = x + (i - 1.5) * width
    bars[method] = ax.bar(position, scores[method], width, 
                          label=method, 
                          color=colors[method],
                          edgecolor='black',
                          linewidth=1.5,
                          yerr=confidence_intervals[method],
                          capsize=5,
                          error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels on top of bars
    for j, (bar, value) in enumerate(zip(bars[method], scores[method])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + confidence_intervals[method][j] + 2,
                f'{value:.1f}%',
                ha='center', va='bottom', fontsize=18, fontweight='bold')

# Customize the plot
ax.set_xlabel('Performance Metrics', fontsize=24, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=24, fontweight='bold')
ax.set_title('Performance Comparison', 
             fontsize=26, fontweight='bold', pad=20)

# Set x-axis labels with proper rendering
ax.set_xticks(x)
# Create labels with bold for first one
labels = ax.set_xticklabels(categories, fontsize=20)
# Make first label (Overall FAS) bold
labels[0].set_weight('bold')
labels[0].set_fontsize(22)

# Set y-axis limits and labels
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f'{i}%' for i in range(0, 101, 20)], fontsize=20)

# Add grid for better readability
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend - move to lower left corner
ax.legend(loc='lower left', fontsize=20, framealpha=0.95, 
          edgecolor='black', fancybox=True, shadow=True)

# Add a horizontal line at 50% for reference
ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Remove annotation box - no longer needed

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/wfas_2_5_performance_bar.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Created wfas_2_5_performance_bar.png with 2.5:1 weights")
print("\n" + "="*60)
print("OVERALL FAS SCORES:")
print("="*60)
for method in methods:
    print(f"{method:12} : {scores[method][0]:.1f}%")