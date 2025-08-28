import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Disable interactive mode
matplotlib.use('Agg')

# Set larger font sizes globally
plt.rcParams.update({'font.size': 16})

# Original data
methods_data = {
    'BASE LLM': {
        'supported': 30.6,
        'contradicted': 15.7,
        'unverifiable': 53.6
    },
    'VECTOR RAG': {
        'supported': 78.0,
        'contradicted': 8.4,
        'unverifiable': 13.6
    },
    'GRAPH RAG': {
        'supported': 65.1,
        'contradicted': 8.6,
        'unverifiable': 26.3
    },
    'HYBRID AHS': {
        'supported': 73.8,
        'contradicted': 6.4,
        'unverifiable': 19.8
    }
}

# Calculate WFAS using 2.5:1 weight scheme
def calculate_wfas(data):
    whr = (2.5 * data['contradicted'] + 1 * data['unverifiable']) / 3.5
    return 100 - whr

# Colors for each category
colors = {
    'supported': '#4CAF50',      # Green
    'contradicted': '#C62828',   # Dark Red (2.5x weight)
    'unverifiable': '#FFA726'     # Orange (1x weight)
}

# Create figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
axes = axes.flatten()

for idx, (method, data) in enumerate(methods_data.items()):
    ax = axes[idx]
    
    # Calculate WFAS
    wfas = calculate_wfas(data)
    
    # Prepare data for pie chart
    sizes = [data['supported'], data['contradicted'], data['unverifiable']]
    labels = [
        f"Supported\n{data['supported']:.1f}%",
        f"Contradicted\n{data['contradicted']:.1f}%", 
        f"Unverifiable\n{data['unverifiable']:.1f}%"
    ]
    pie_colors = [colors['supported'], colors['contradicted'], colors['unverifiable']]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                       autopct='', startangle=90,
                                       textprops={'fontsize': 14, 'weight': 'bold'})
    
    # Add title with method name and FAS score
    ax.set_title(f'{method}\nFAS: {wfas:.1f}%', 
                 fontsize=22, fontweight='bold', pad=20)
    
    # Make text more readable
    for text in texts:
        text.set_fontsize(14)
        text.set_weight('bold')
    
    # Add white edge to wedges for better separation
    for wedge in wedges:
        wedge.set_edgecolor('white')
        wedge.set_linewidth(3)

# Add main title
fig.suptitle('Claim Classification Breakdown', 
             fontsize=26, fontweight='bold', y=1.02)

# Remove formula text - just add empty space
# fig.text removed

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/wfas_2_5_hallucination_4_charts.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Created wfas_2_5_hallucination_4_charts.png with 2.5:1 weights")

# Create individual charts
for method, data in methods_data.items():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate WFAS
    wfas = calculate_wfas(data)
    
    # Prepare data for pie chart
    sizes = [data['supported'], data['contradicted'], data['unverifiable']]
    labels = [
        f"Supported\n{data['supported']:.1f}%",
        f"Contradicted\n{data['contradicted']:.1f}%",
        f"Unverifiable\n{data['unverifiable']:.1f}%"
    ]
    pie_colors = [colors['supported'], colors['contradicted'], colors['unverifiable']]
    
    # Create pie chart with explode for emphasis
    explode = (0.05, 0.15, 0.05)  # Emphasize contradicted more
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                       autopct='', startangle=90,
                                       explode=explode,
                                       textprops={'fontsize': 18, 'weight': 'bold'},
                                       shadow=True)
    
    # Add title with method name and FAS score
    ax.set_title(f'{method}\nFactual Accuracy Score: {wfas:.1f}%', 
                 fontsize=24, fontweight='bold', pad=30)
    
    # Make text more readable
    for text in texts:
        text.set_fontsize(18)
        text.set_weight('bold')
    
    # Add white edge to wedges for better separation
    for wedge in wedges:
        wedge.set_edgecolor('white')
        wedge.set_linewidth(3)
    
    # Save individual chart
    filename = f'wfas_2_5_hallucination_{method.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(f'/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/{filename}', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {filename}")

print("\n" + "="*60)
print("FAS SCORES:")
print("="*60)
for method, data in methods_data.items():
    wfas = calculate_wfas(data)
    print(f"{method:12} : {wfas:.1f}%")