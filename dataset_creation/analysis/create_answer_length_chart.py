import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Disable interactive mode
matplotlib.use('Agg')

# Set larger font sizes globally
plt.rcParams.update({'font.size': 18})

# Load the dataset
with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data/apqc_auto.json', 'r') as f:
    data = json.load(f)

questions = data['questions']

# Gold Answer Length Distribution (Histogram)
answer_lengths = [len(q['answer'].split()) for q in questions]

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
n, bins, patches = ax.hist(answer_lengths, bins=40, color='#4CAF50', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)

# Add mean line
mean_answer = np.mean(answer_lengths)
ax.axvline(mean_answer, color='red', linestyle='--', linewidth=3, 
           label=f'Mean: {mean_answer:.1f} words')

ax.set_xlabel('Answer Length (words)', fontsize=24, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=24, fontweight='bold')
ax.set_title('Gold Answer Length Distribution', fontsize=26, fontweight='bold', pad=25)
ax.legend(fontsize=20, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/chart3_answer_length.png', 
            dpi=300, bbox_inches='tight')
print("Created chart3_answer_length.png without statistics box")