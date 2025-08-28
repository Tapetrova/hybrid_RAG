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

# Question Length Distribution (Histogram)
question_lengths = [len(q['question'].split()) for q in questions]

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
n, bins, patches = ax.hist(question_lengths, bins=30, color='#2196F3', alpha=0.8, edgecolor='navy', linewidth=1.5)

# Add mean line
mean_length = np.mean(question_lengths)
ax.axvline(mean_length, color='red', linestyle='--', linewidth=3, 
           label=f'Mean: {mean_length:.1f} words')

ax.set_xlabel('Question Length (words)', fontsize=24, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=24, fontweight='bold')
ax.set_title('Question Length Distribution', fontsize=26, fontweight='bold', pad=25)
ax.legend(fontsize=20, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/chart2_question_length.png', 
            dpi=300, bbox_inches='tight')
print("Created chart2_question_length.png without statistics box")