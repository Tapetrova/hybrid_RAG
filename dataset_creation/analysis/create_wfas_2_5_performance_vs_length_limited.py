import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats

# Disable interactive mode
matplotlib.use('Agg')

# Set larger font sizes globally
plt.rcParams.update({'font.size': 14})

# Load the hallucination results with actual FAS scores
with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/hallucination_FULL_API_706_results_20250821_231422.json', 'r') as f:
    hall_data = json.load(f)

# Load the dataset to get question lengths
with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data/apqc_auto.json', 'r') as f:
    dataset = json.load(f)

# Create a mapping of question_id to question length
question_lengths = {}
for q in dataset['questions']:
    question_lengths[q['id']] = len(q['question'].split())

# Organize scores by question length bins - EXCLUDING >30 words
length_bins = [
    (0, 10, '<10'),
    (10, 15, '10-15'),
    (15, 20, '15-20'),
    (20, 30, '20-30')
    # Removed (30, 100, '>30') bin
]

# Initialize data structure for each method and bin
methods = ['hybrid_ahs', 'vector_rag', 'graph_rag', 'base_llm']
method_names = {
    'base_llm': 'BASE LLM',
    'vector_rag': 'VECTOR RAG', 
    'graph_rag': 'GRAPH RAG',
    'hybrid_ahs': 'HYBRID AHS'
}

# Collect scores by length bin for each method
performance_by_length = {method: {bin_label: [] for _, _, bin_label in length_bins} 
                         for method in methods}

# Process hallucination results
for result in hall_data['results']:
    qid = result['question_id']
    if qid in question_lengths:
        q_length = question_lengths[qid]
        
        # Find appropriate bin
        for min_len, max_len, bin_label in length_bins:
            if min_len <= q_length < max_len:
                # Calculate WFAS for each method using 2.5:1 weight scheme
                for method in methods:
                    if method in result.get('metrics', {}):
                        metrics = result['metrics'][method]
                        if metrics.get('total_claims', 0) > 0:
                            # Calculate weighted hallucination rate with 2.5:1 ratio
                            supported = metrics.get('supported', 0)
                            contradicted = metrics.get('contradicted', 0)
                            unverifiable = metrics.get('unverifiable', 0)
                            total = metrics['total_claims']
                            
                            # WFAS = 100 - (2.5*contradicted + 1*unverifiable)/(3.5*total) * 100
                            whr = ((2.5 * contradicted + 1 * unverifiable) / (3.5 * total)) * 100
                            wfas = 100 - whr
                            
                            performance_by_length[method][bin_label].append(wfas)
                break

# Calculate mean and std error for each bin
bin_labels = [label for _, _, label in length_bins]
mean_scores = {method: [] for method in methods}
std_errors = {method: [] for method in methods}

for method in methods:
    for bin_label in bin_labels:
        scores = performance_by_length[method][bin_label]
        if scores:
            mean_scores[method].append(np.mean(scores))
            # Standard error = std / sqrt(n)
            std_errors[method].append(np.std(scores) / np.sqrt(len(scores)))
        else:
            mean_scores[method].append(0)
            std_errors[method].append(0)

# Create the plot with adjusted size
fig, ax = plt.subplots(figsize=(12, 10))

# Colors for each method - more vibrant for better visibility
colors = {
    'base_llm': '#E53935',      # Bright red
    'vector_rag': '#43A047',    # Bright green
    'graph_rag': '#1E88E5',     # Bright blue
    'hybrid_ahs': '#8E24AA'     # Bright purple
}

# Plot lines with error bars
x_pos = np.arange(len(bin_labels))

for method in methods:
    ax.errorbar(x_pos, mean_scores[method], yerr=std_errors[method],
                label=method_names[method], color=colors[method],
                marker='o', markersize=12, linewidth=3, capsize=6,
                capthick=2.5, elinewidth=2.5, alpha=0.9)

# Customize the plot
ax.set_xlabel('Question Length (words)', fontsize=18, fontweight='bold')
ax.set_ylabel('Factual Accuracy Score (%)', fontsize=18, fontweight='bold')
ax.set_title('Performance vs Question Length (Limited Range)', 
             fontsize=20, fontweight='bold', pad=20)

# Set x-axis labels with larger font
ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels, fontsize=16, fontweight='bold')

# Set y-axis limits to 0-100% with more granular ticks
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))
ax.set_yticklabels([f'{i}%' for i in range(0, 101, 10)], fontsize=14)

# Add minor grid for better readability
ax.grid(True, axis='y', alpha=0.3, linestyle='--', which='major')
ax.grid(True, axis='y', alpha=0.15, linestyle=':', which='minor')
ax.minorticks_on()
ax.set_axisbelow(True)

# Add legend with larger font
ax.legend(loc='lower left', fontsize=15, framealpha=0.95,
          edgecolor='black', fancybox=True, shadow=True, 
          borderpad=1, columnspacing=1)

# Add horizontal reference lines
ax.axhline(y=90, color='green', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(y=80, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(y=70, color='red', linestyle=':', linewidth=1, alpha=0.5)

# Removed yellow annotation box for best performer

# Annotate degradation for long questions
worst_long_idx = 3  # Index for 20-30 words
degradations = {}
for method in methods:
    if mean_scores[method][0] > 0 and mean_scores[method][worst_long_idx] > 0:
        degradation = mean_scores[method][0] - mean_scores[method][worst_long_idx]
        degradations[method_names[method]] = degradation

# Removed orange annotation box about degradation

# Calculate and display summary statistics
print("\n" + "="*60)
print("PERFORMANCE vs QUESTION LENGTH (LIMITED TO 20-30 WORDS)")
print("="*60)
print("\nDetailed statistics (FAS with 2.5:1 weights):")
print("-" * 60)

for method in methods:
    print(f"\n{method_names[method]}:")
    for i, bin_label in enumerate(bin_labels):
        n_samples = len(performance_by_length[method][bin_label])
        if n_samples > 0:
            mean = mean_scores[method][i]
            se = std_errors[method][i]
            print(f"  {bin_label} words: {mean:.1f}% ± {se:.1f}% (n={n_samples})")
        else:
            print(f"  {bin_label} words: No data")
    
    # Calculate degradation
    if mean_scores[method][0] > 0 and mean_scores[method][-1] > 0:
        degradation = mean_scores[method][0] - mean_scores[method][-1]
        print(f"  Degradation (<10 to 20-30): -{degradation:.1f} percentage points")

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/wfas_2_5_performance_vs_length_limited.png',
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\nCreated wfas_2_5_performance_vs_length_limited.png")
print("Chart shows performance up to 20-30 word questions only")
print("Y-axis scale: 0-100%")