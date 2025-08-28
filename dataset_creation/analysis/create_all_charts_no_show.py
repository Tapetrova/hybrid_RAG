import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Disable interactive mode
matplotlib.use('Agg')

# Load the dataset
with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data/apqc_auto.json', 'r') as f:
    data = json.load(f)

questions = data['questions']

# 2. Question Length Distribution (Histogram)
question_lengths = [len(q['question'].split()) for q in questions]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
n, bins, patches = ax.hist(question_lengths, bins=30, color='#2196F3', alpha=0.8, edgecolor='navy')

# Add mean line
mean_length = np.mean(question_lengths)
ax.axvline(mean_length, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_length:.1f} words')

ax.set_xlabel('Question Length (words)', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Question Length Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Add statistics text
stats_text = f'Total Questions: {len(questions)}\nMean: {mean_length:.1f} words\nMedian: {np.median(question_lengths):.1f} words\nRange: {min(question_lengths)}-{max(question_lengths)} words'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/chart2_question_length.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("Created chart2_question_length.png")

# 3. Gold Answer Length Distribution (Histogram)
answer_lengths = [len(q['answer'].split()) for q in questions]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
n, bins, patches = ax.hist(answer_lengths, bins=40, color='#4CAF50', alpha=0.8, edgecolor='darkgreen')

# Add mean line
mean_answer = np.mean(answer_lengths)
ax.axvline(mean_answer, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_answer:.1f} words')

ax.set_xlabel('Answer Length (words)', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Gold Answer Length Distribution', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

# Add statistics text
stats_text = f'Total Answers: {len(questions)}\nMean: {mean_answer:.1f} words\nMedian: {np.median(answer_lengths):.1f} words\nRange: {min(answer_lengths)}-{max(answer_lengths)} words'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/chart3_answer_length.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("Created chart3_answer_length.png")

print("\nAll charts have been created successfully!")