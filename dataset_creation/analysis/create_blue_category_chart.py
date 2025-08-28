import json
import matplotlib.pyplot as plt
import matplotlib

# Disable interactive mode
matplotlib.use('Agg')

# Load the dataset
with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data/apqc_auto.json', 'r') as f:
    data = json.load(f)

questions = data['questions']

# Question Category Distribution (Pie Chart in blue shades)
categories = {}
for q in questions:
    cat = q.get('category', 'unknown')
    categories[cat] = categories.get(cat, 0) + 1

# Sort categories for consistent display
sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
labels = [f'{cat.capitalize()}\n({count})' for cat, count in sorted_cats]
sizes = [count for cat, count in sorted_cats]

# Blue color palette (from dark to light)
colors = ['#0D47A1', '#1976D2', '#42A5F5', '#90CAF9']  # Dark blue to light blue

wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                   autopct='%1.1f%%', startangle=90,
                                   textprops={'fontsize': 24, 'fontweight': 'bold'})

# Make percentage text bold and white for better contrast
for autotext in autotexts:
    autotext.set_weight('bold')
    autotext.set_fontsize(24)
    autotext.set_color('white')

# Increase label font size (Factual, Comparative, etc.)
for text in texts:
    text.set_fontsize(22)
    text.set_fontweight('bold')

ax.set_title('Question Category Distribution', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/chart1_category_distribution_blue.png', 
            dpi=300, bbox_inches='tight')
print("Created chart1_category_distribution_blue.png with blue color scheme")