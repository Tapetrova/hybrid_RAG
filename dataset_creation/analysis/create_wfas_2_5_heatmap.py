import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# Disable interactive mode
matplotlib.use('Agg')

# Set style for publication-quality figure
sns.set_style("white")
plt.rcParams.update({'font.size': 14})

# Category-specific data with all components
category_data = {
    'factual': {
        'base_llm': {'supported': 31.8, 'contradicted': 15.3, 'unverifiable': 52.9},
        'vector_rag': {'supported': 78.2, 'contradicted': 8.2, 'unverifiable': 13.6},
        'graph_rag': {'supported': 66.3, 'contradicted': 8.2, 'unverifiable': 25.5},
        'hybrid_ahs': {'supported': 72.4, 'contradicted': 6.1, 'unverifiable': 21.5}
    },
    'causal': {
        'base_llm': {'supported': 35.5, 'contradicted': 12.0, 'unverifiable': 52.5},
        'vector_rag': {'supported': 81.2, 'contradicted': 6.5, 'unverifiable': 12.3},
        'graph_rag': {'supported': 63.5, 'contradicted': 8.0, 'unverifiable': 28.5},
        'hybrid_ahs': {'supported': 79.7, 'contradicted': 5.2, 'unverifiable': 15.1}
    },
    'diagnostic': {
        'base_llm': {'supported': 23.8, 'contradicted': 17.8, 'unverifiable': 58.4},
        'vector_rag': {'supported': 73.9, 'contradicted': 10.2, 'unverifiable': 15.9},
        'graph_rag': {'supported': 64.0, 'contradicted': 8.5, 'unverifiable': 27.5},
        'hybrid_ahs': {'supported': 75.6, 'contradicted': 6.8, 'unverifiable': 17.6}
    },
    'comparative': {
        'base_llm': {'supported': 27.8, 'contradicted': 18.2, 'unverifiable': 54.0},
        'vector_rag': {'supported': 77.7, 'contradicted': 9.1, 'unverifiable': 13.2},
        'graph_rag': {'supported': 63.5, 'contradicted': 10.4, 'unverifiable': 26.1},
        'hybrid_ahs': {'supported': 69.0, 'contradicted': 9.1, 'unverifiable': 21.9}
    }
}

# Calculate WFAS using 2.5:1 weight scheme
def calculate_wfas_2_5(data):
    whr = (2.5 * data['contradicted'] + 1 * data['unverifiable']) / 3.5
    return 100 - whr

categories = ['Factual', 'Causal', 'Diagnostic', 'Comparative']
methods = ['BASE LLM', 'VECTOR RAG', 'GRAPH RAG', 'HYBRID AHS']
method_keys = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']

# Calculate WFAS matrix (rows: categories, columns: methods)
wfas_data = np.zeros((len(categories), len(methods)))

for i, cat in enumerate(['factual', 'causal', 'diagnostic', 'comparative']):
    for j, method_key in enumerate(method_keys):
        wfas_data[i, j] = calculate_wfas_2_5(category_data[cat][method_key])

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Create heatmap with better color scheme
im = ax.imshow(wfas_data, cmap='RdYlGn', aspect='auto', vmin=60, vmax=95)

# Set ticks
ax.set_xticks(np.arange(len(methods)))
ax.set_yticks(np.arange(len(categories)))

# Set labels
ax.set_xticklabels(methods, fontsize=16, fontweight='bold')
ax.set_yticklabels(categories, fontsize=16)

# Rotate the tick labels for better fit
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Factual Accuracy Score (%)', rotation=90, va="bottom", 
                   fontsize=16, fontweight='bold')
cbar.ax.tick_params(labelsize=14)

# Add text annotations
for i in range(len(categories)):
    for j in range(len(methods)):
        text = ax.text(j, i, f'{wfas_data[i, j]:.1f}',
                      ha="center", va="center", color="black",
                      fontsize=18, fontweight='bold')
        
        # Add white background for better readability
        text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', 
                          edgecolor='none', alpha=0.7))

# Add grid for better readability
ax.set_xticks(np.arange(len(methods))-.5, minor=True)
ax.set_yticks(np.arange(len(categories))-.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=1.5)
ax.tick_params(which="minor", size=0)

# Remove spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Set title
ax.set_title('FAS Heatmap: Methods × Categories', 
             fontsize=20, fontweight='bold', pad=20)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/wfas_2_5_heatmap.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Created wfas_2_5_heatmap.png with 2.5:1 weights")
print("\n" + "="*60)
print("FAS SCORES BY CATEGORY:")
print("="*60)

for i, cat in enumerate(categories):
    print(f"\n{cat}:")
    for j, method in enumerate(methods):
        print(f"  {method:12} : {wfas_data[i, j]:.1f}%")
    
# Find best performer for each category
print("\n" + "="*60)
print("BEST PERFORMER PER CATEGORY:")
print("="*60)
for i, cat in enumerate(categories):
    best_idx = np.argmax(wfas_data[i, :])
    best_method = methods[best_idx]
    best_score = wfas_data[i, best_idx]
    print(f"{cat:12} : {best_method} ({best_score:.1f}%)")