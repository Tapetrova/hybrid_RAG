import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# Disable interactive mode
matplotlib.use('Agg')

# Set style for publication-quality figure
sns.set_style("white")
plt.rcParams.update({'font.size': 14})

# Data from the FAS report (methods × categories)
categories = ['Factual', 'Causal', 'Diagnostic', 'Comparative']
methods = ['BASE LLM', 'VECTOR RAG', 'GRAPH RAG', 'HYBRID AHS']

# FAS scores matrix (rows: categories, columns: methods)
fas_data = np.array([
    [31.8, 78.2, 66.3, 72.4],  # Factual
    [35.5, 81.2, 63.5, 79.7],  # Causal
    [23.8, 73.9, 64.0, 75.6],  # Diagnostic
    [27.8, 77.7, 63.5, 69.0]   # Comparative
])

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Create heatmap
im = ax.imshow(fas_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

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
        text = ax.text(j, i, f'{fas_data[i, j]:.1f}',
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
ax.set_title('FAS Heatmap: Methods × Categories\nHigher values indicate better factual accuracy', 
             fontsize=20, fontweight='bold', pad=20)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/fas_heatmap_clean.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Created fas_heatmap_clean.png - Clean heatmap showing FAS scores for methods × categories")