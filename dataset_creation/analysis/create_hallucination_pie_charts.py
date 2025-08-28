import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Disable interactive mode
matplotlib.use('Agg')

# Set larger font sizes globally
plt.rcParams.update({'font.size': 16})

# Data from the hallucination breakdown
methods_data = {
    'BASE LLM': {
        'supported': 30.6,
        'contradicted': 15.7,
        'unverifiable': 53.6,
        'FAS': 30.6
    },
    'VECTOR RAG': {
        'supported': 78.0,
        'contradicted': 8.4,
        'unverifiable': 13.6,
        'FAS': 78.0
    },
    'GRAPH RAG': {
        'supported': 65.1,
        'contradicted': 8.6,
        'unverifiable': 26.3,
        'FAS': 65.1
    },
    'HYBRID AHS': {
        'supported': 73.8,
        'contradicted': 6.4,
        'unverifiable': 19.8,
        'FAS': 73.8
    }
}

# Colors for each category
colors = {
    'supported': '#4CAF50',      # Green
    'contradicted': '#F44336',   # Red
    'unverifiable': '#9E9E9E'    # Gray
}

# Create figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
axes = axes.flatten()

for idx, (method, data) in enumerate(methods_data.items()):
    ax = axes[idx]
    
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
    ax.set_title(f'{method}\nFAS: {data["FAS"]:.1f}%', 
                 fontsize=20, fontweight='bold', pad=20)
    
    # Make text more readable
    for text in texts:
        text.set_fontsize(14)
        text.set_weight('bold')
    
    # Add white edge to wedges for better separation
    for wedge in wedges:
        wedge.set_edgecolor('white')
        wedge.set_linewidth(2)

# Add main title
fig.suptitle('Claim Classification Breakdown by Method', 
             fontsize=24, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/hallucination_4_charts.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("Created hallucination_4_charts.png with 4 separate pie charts")

# Now create individual charts
for method, data in methods_data.items():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Prepare data for pie chart
    sizes = [data['supported'], data['contradicted'], data['unverifiable']]
    labels = [
        f"Supported\n{data['supported']:.1f}%",
        f"Contradicted\n{data['contradicted']:.1f}%",
        f"Unverifiable\n{data['unverifiable']:.1f}%"
    ]
    pie_colors = [colors['supported'], colors['contradicted'], colors['unverifiable']]
    
    # Create pie chart with explode for emphasis
    explode = (0.05, 0.05, 0.05)  # Slightly separate all slices
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors,
                                       autopct='', startangle=90,
                                       explode=explode,
                                       textprops={'fontsize': 18, 'weight': 'bold'},
                                       shadow=True)
    
    # Add title with method name and FAS score
    ax.set_title(f'{method}\nFactual Accuracy Score: {data["FAS"]:.1f}%', 
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
    filename = f'hallucination_{method.lower().replace(" ", "_")}.png'
    plt.tight_layout()
    plt.savefig(f'/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/{filename}', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {filename}")