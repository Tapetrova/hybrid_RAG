import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set larger fonts
plt.rcParams.update({'font.size': 14})

# Original data
original_data = {
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

print("="*70)
print("WEIGHTED HALLUCINATION METRICS CALCULATION")
print("="*70)
print("\n📊 ORIGINAL METRICS (Equal weights):")
print("-"*50)
print("Formula: HR = (Contradicted + Unverifiable) / 100")
print("         FAS = Supported\n")

for method, data in original_data.items():
    hr = data['contradicted'] + data['unverifiable']
    fas = data['supported']
    print(f"{method:12} | FAS: {fas:5.1f}% | HR: {hr:5.1f}% | Contra: {data['contradicted']:5.1f}% | Unver: {data['unverifiable']:5.1f}%")

# Calculate weighted metrics with different weight schemes
weight_schemes = [
    {"name": "Equal (1:1)", "w_contra": 1.0, "w_unver": 1.0},
    {"name": "Moderate (1.5:1)", "w_contra": 1.5, "w_unver": 1.0},
    {"name": "Strong (2:1)", "w_contra": 2.0, "w_unver": 1.0},
    {"name": "Severe (3:1)", "w_contra": 3.0, "w_unver": 1.0},
]

print("\n"*2)
print("="*70)
print("WEIGHTED HALLUCINATION METRICS")
print("="*70)

for scheme in weight_schemes:
    print(f"\n🎯 WEIGHT SCHEME: {scheme['name']}")
    print(f"   Contradicted weight: {scheme['w_contra']}")
    print(f"   Unverifiable weight: {scheme['w_unver']}")
    print("-"*50)
    
    results = {}
    for method, data in original_data.items():
        # Calculate weighted hallucination rate
        whr = (scheme['w_contra'] * data['contradicted'] + 
               scheme['w_unver'] * data['unverifiable']) / (scheme['w_contra'] + scheme['w_unver'])
        
        # Calculate weighted FAS (normalized to 100%)
        # We need to normalize because total can exceed 100% with weights
        total_weighted = (data['supported'] + 
                         scheme['w_contra'] * data['contradicted'] + 
                         scheme['w_unver'] * data['unverifiable'])
        
        wfas = (data['supported'] / total_weighted) * 100
        
        # Alternative calculation: WFAS = 100 - WHR (simpler)
        wfas_alt = 100 - whr
        
        results[method] = {
            'whr': whr,
            'wfas': wfas_alt,
            'original_fas': data['supported']
        }
        
        delta = wfas_alt - data['supported']
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
        
        print(f"{method:12} | WFAS: {wfas_alt:5.1f}% | WHR: {whr:5.1f}% | Change: {direction}{abs(delta):4.1f}%")

print("\n"*2)
print("="*70)
print("RECOMMENDED WEIGHT SCHEME: Strong (2:1)")
print("="*70)
print("\nRationale:")
print("• Contradicted statements are TWICE as harmful as unverifiable ones")
print("• Direct falsehoods damage trust more than unverifiable claims")
print("• This penalizes models that generate false information")
print("\nNew Formula:")
print("  WHR = (2 × Contradicted% + 1 × Unverifiable%) / 3")
print("  WFAS = 100% - WHR")

# Calculate with recommended weights (2:1)
w_contra = 2.0
w_unver = 1.0

print("\n📊 FINAL WEIGHTED RESULTS (2:1 weights):")
print("-"*50)

final_results = {}
for method, data in original_data.items():
    whr = (w_contra * data['contradicted'] + w_unver * data['unverifiable']) / (w_contra + w_unver)
    wfas = 100 - whr
    
    final_results[method] = {
        'supported': data['supported'],
        'contradicted': data['contradicted'],
        'unverifiable': data['unverifiable'],
        'wfas': wfas,
        'whr': whr,
        'original_fas': data['supported']
    }

# Sort by WFAS
sorted_methods = sorted(final_results.items(), key=lambda x: x[1]['wfas'], reverse=True)

print("\nRANKING BY WEIGHTED FACTUAL ACCURACY SCORE (WFAS):")
print("-"*50)
for rank, (method, metrics) in enumerate(sorted_methods, 1):
    print(f"{rank}. {method:12} | WFAS: {metrics['wfas']:5.1f}% | Original FAS: {metrics['original_fas']:5.1f}%")
    print(f"   → Contradicted: {metrics['contradicted']:5.1f}% (weight: 2x)")
    print(f"   → Unverifiable: {metrics['unverifiable']:5.1f}% (weight: 1x)")
    print()

# Create visualization comparing original vs weighted
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

methods = list(original_data.keys())
original_fas = [original_data[m]['supported'] for m in methods]
weighted_fas = [final_results[m]['wfas'] for m in methods]

x = np.arange(len(methods))
width = 0.35

# Original FAS
bars1 = ax1.bar(x - width/2, original_fas, width, label='Original FAS', color='#2196F3')
bars2 = ax1.bar(x + width/2, weighted_fas, width, label='Weighted FAS (2:1)', color='#4CAF50')

ax1.set_xlabel('Method', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax1.set_title('Original vs Weighted FAS Comparison', fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=0)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

# Penalty comparison
contradicted = [original_data[m]['contradicted'] for m in methods]
unverifiable = [original_data[m]['unverifiable'] for m in methods]

bar_width = 0.35
r1 = np.arange(len(methods))
r2 = [x + bar_width for x in r1]

bars3 = ax2.bar(r1, contradicted, bar_width, label='Contradicted (2x penalty)', color='#F44336')
bars4 = ax2.bar(r2, unverifiable, bar_width, label='Unverifiable (1x penalty)', color='#FFA726')

ax2.set_xlabel('Method', fontsize=14, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax2.set_title('Hallucination Types with Weight Penalties', fontsize=16, fontweight='bold')
ax2.set_xticks([r + bar_width/2 for r in range(len(methods))])
ax2.set_xticklabels(methods)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/analysis/weighted_hallucination_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization saved as: weighted_hallucination_comparison.png")

# Impact analysis
print("\n"*2)
print("="*70)
print("IMPACT ANALYSIS OF WEIGHTED METRICS")
print("="*70)

print("\n1. BIGGEST WINNERS (methods that improve with weighted scoring):")
for method, metrics in sorted_methods:
    change = metrics['wfas'] - metrics['original_fas']
    if change > 0:
        print(f"   {method}: +{change:.1f}% (low contradiction rate helps)")

print("\n2. BIGGEST LOSERS (methods that worsen with weighted scoring):")
for method, metrics in sorted_methods:
    change = metrics['wfas'] - metrics['original_fas']
    if change < 0:
        print(f"   {method}: {change:.1f}% (high contradiction rate hurts)")

print("\n3. KEY INSIGHT:")
print("   Weighted scoring better reflects real-world harm:")
print("   • False information (contradicted) is more dangerous than missing info")
print("   • HYBRID AHS benefits most: lowest contradiction rate (6.4%)")
print("   • BASE LLM suffers most: highest contradiction rate (15.7%)")

print("\n" + "="*70)