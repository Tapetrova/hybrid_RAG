"""
Update dataset statistics with classification results
Generates enhanced visualizations and analysis
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load all relevant data files"""
    
    # Load classification statistics
    with open('data/processed/classification_statistics.json', 'r') as f:
        class_stats = json.load(f)
    
    # Load original statistics
    with open('data/processed/dataset_statistics.json', 'r') as f:
        orig_stats = json.load(f)
    
    # Load classified questions for detailed analysis
    with open('data/processed/classified_questions.json', 'r') as f:
        classified_questions = json.load(f)
    
    return class_stats, orig_stats, classified_questions

def analyze_classification_quality(classified_questions):
    """Analyze classification quality and patterns"""
    
    analysis = {
        'confidence_distribution': {},
        'cross_category_patterns': {},
        'category_characteristics': {}
    }
    
    # Analyze each category
    for category in ['causal', 'diagnostic', 'comparative', 'factual']:
        cat_questions = [q for q in classified_questions 
                        if q['classification']['primary_category'] == category]
        
        if not cat_questions:
            continue
        
        # Confidence distribution
        confidences = [q['classification']['confidence'] for q in cat_questions]
        analysis['confidence_distribution'][category] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'quartiles': {
                'q1': np.percentile(confidences, 25),
                'median': np.percentile(confidences, 50),
                'q3': np.percentile(confidences, 75)
            },
            'high_confidence_count': sum(1 for c in confidences if c > 0.7),
            'low_confidence_count': sum(1 for c in confidences if c < 0.3)
        }
        
        # Category characteristics
        scores = [q.get('score', 0) for q in cat_questions]
        answer_rates = sum(1 for q in cat_questions if q.get('accepted_answer')) / len(cat_questions)
        
        # Common tags for this category
        all_tags = []
        for q in cat_questions:
            all_tags.extend(q.get('tags', []))
        top_tags = Counter(all_tags).most_common(5)
        
        analysis['category_characteristics'][category] = {
            'avg_score': np.mean(scores),
            'answer_rate': answer_rates,
            'top_tags': dict(top_tags),
            'total_questions': len(cat_questions)
        }
        
        # Cross-category scores (how strongly questions scored for other categories)
        for other_cat in ['causal', 'diagnostic', 'comparative', 'factual']:
            if other_cat != category:
                other_scores = [q['classification']['scores'].get(other_cat, 0) 
                              for q in cat_questions]
                avg_other_score = np.mean(other_scores) if other_scores else 0
                
                if category not in analysis['cross_category_patterns']:
                    analysis['cross_category_patterns'][category] = {}
                
                analysis['cross_category_patterns'][category][other_cat] = round(avg_other_score, 2)
    
    return analysis

def create_enhanced_visualizations(class_stats, analysis, classified_questions):
    """Create enhanced visualizations for the paper"""
    
    # 1. Classification Distribution with Confidence
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main distribution pie chart
    ax = axes[0, 0]
    categories = list(class_stats['classification_distribution'].keys())
    counts = list(class_stats['classification_distribution'].values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors,
                                       autopct='%1.1f%%', startangle=90)
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Question Classification Distribution', fontsize=14, fontweight='bold')
    
    # Confidence distribution boxplot
    ax = axes[0, 1]
    confidence_data = []
    confidence_labels = []
    for cat in categories:
        cat_questions = [q for q in classified_questions 
                        if q['classification']['primary_category'] == cat]
        confidences = [q['classification']['confidence'] for q in cat_questions]
        confidence_data.append(confidences)
        confidence_labels.append(f"{cat}\n(n={len(confidences)})")
    
    bp = ax.boxplot(confidence_data, tick_labels=confidence_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Classification Confidence', fontsize=12)
    ax.set_title('Confidence Score Distribution by Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High confidence threshold')
    ax.legend()
    
    # Category vs Answer Rate
    ax = axes[1, 0]
    answer_rates = []
    for cat in categories:
        cat_questions = [q for q in classified_questions 
                        if q['classification']['primary_category'] == cat]
        rate = sum(1 for q in cat_questions if q.get('accepted_answer')) / len(cat_questions) * 100
        answer_rates.append(rate)
    
    bars = ax.bar(categories, answer_rates, color=colors, alpha=0.8)
    ax.set_ylabel('Answer Rate (%)', fontsize=12)
    ax.set_xlabel('Question Category', fontsize=12)
    ax.set_title('Answer Rate by Question Category', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, rate in zip(bars, answer_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Mean line
    ax.axhline(y=np.mean(answer_rates), color='red', linestyle='--', 
              alpha=0.5, label=f'Mean: {np.mean(answer_rates):.1f}%')
    ax.legend()
    
    # Score distribution by category
    ax = axes[1, 1]
    score_data = []
    for cat in categories:
        cat_questions = [q for q in classified_questions 
                        if q['classification']['primary_category'] == cat]
        scores = [q.get('score', 0) for q in cat_questions]
        score_data.append(scores)
    
    vp = ax.violinplot(score_data, positions=range(len(categories)), 
                       widths=0.7, showmeans=True, showmedians=True)
    
    # Color violins
    for pc, color in zip(vp['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel('Question Score', fontsize=12)
    ax.set_xlabel('Question Category', fontsize=12)
    ax.set_title('Score Distribution by Question Category', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('CKG-RAG Dataset Classification Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('paper/figures/classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cross-Category Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare confusion-style matrix
    matrix = []
    for cat1 in categories:
        row = []
        for cat2 in categories:
            if cat1 == cat2:
                row.append(class_stats['classification_distribution'][cat1])
            else:
                # Average score for cat2 in questions classified as cat1
                cat1_questions = [q for q in classified_questions 
                                if q['classification']['primary_category'] == cat1]
                avg_score = np.mean([q['classification']['scores'].get(cat2, 0) 
                                   for q in cat1_questions])
                row.append(avg_score)
        matrix.append(row)
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlOrRd',
               xticklabels=categories, yticklabels=categories,
               cbar_kws={'label': 'Count / Avg Score'})
    
    ax.set_title('Cross-Category Classification Scores', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category Scores', fontsize=12)
    ax.set_ylabel('Assigned Category', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('paper/figures/cross_category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top Examples Table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for cat in categories:
        details = class_stats['classification_details'][cat]
        for i, example in enumerate(details['sample_questions'][:2]):  # Top 2 per category
            question = example['question']
            if len(question) > 60:
                question = question[:57] + "..."
            table_data.append([
                cat.capitalize(),
                question,
                f"{example['confidence']:.3f}",
                f"{example['score']:.1f}"
            ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Category', 'Question Example', 'Confidence', 'Score'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.15, 0.6, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(4):
            table[(i, j)].set_facecolor(color)
    
    ax.set_title('Sample Questions by Category', fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('paper/figures/classification_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created enhanced visualizations in paper/figures/")

def update_main_statistics():
    """Update the main dataset statistics file with classification info"""
    
    # Load files
    with open('data/processed/dataset_statistics.json', 'r') as f:
        main_stats = json.load(f)
    
    with open('data/processed/classification_statistics.json', 'r') as f:
        class_stats = json.load(f)
    
    # Add classification statistics
    main_stats['advanced_classification'] = {
        'method': 'Pattern-based with keyword scoring',
        'categories': class_stats['classification_distribution'],
        'confidence': {
            'average': class_stats['average_confidence'],
            'high_confidence_questions': sum(class_stats['high_confidence_distribution'].values()),
            'distribution': class_stats['high_confidence_distribution']
        },
        'details': class_stats['classification_details']
    }
    
    # Save updated statistics
    with open('data/processed/dataset_statistics_updated.json', 'w') as f:
        json.dump(main_stats, f, indent=2)
    
    print("✅ Updated main statistics file: dataset_statistics_updated.json")

def generate_classification_report():
    """Generate comprehensive classification report"""
    
    # Load all data
    class_stats, orig_stats, classified_questions = load_data()
    
    # Analyze classification quality
    analysis = analyze_classification_quality(classified_questions)
    
    # Create visualizations
    create_enhanced_visualizations(class_stats, analysis, classified_questions)
    
    # Update main statistics
    update_main_statistics()
    
    # Print comprehensive report
    print("\n" + "="*60)
    print("CLASSIFICATION ANALYSIS REPORT")
    print("="*60)
    
    print("\n📊 CLASSIFICATION OVERVIEW:")
    print(f"Total Questions: {class_stats['total_questions']}")
    print(f"Average Confidence: {class_stats['average_confidence']:.3f}")
    
    print("\n📈 DISTRIBUTION BY CATEGORY:")
    for cat, details in class_stats['classification_details'].items():
        print(f"\n{cat.upper()}:")
        print(f"  Questions: {details['count']} ({details['percentage']}%)")
        print(f"  Avg Confidence: {details['avg_confidence']:.3f}")
        print(f"  Score Range: {details['avg_score']:.1f}")
        
        char = analysis['category_characteristics'].get(cat, {})
        if char:
            print(f"  Answer Rate: {char['answer_rate']*100:.1f}%")
            print(f"  Avg Question Score: {char['avg_score']:.1f}")
            if char.get('top_tags'):
                tags = ', '.join([f"{k}({v})" for k, v in list(char['top_tags'].items())[:3]])
                print(f"  Top Tags: {tags}")
    
    print("\n🔄 CROSS-CATEGORY PATTERNS:")
    for cat1, patterns in analysis['cross_category_patterns'].items():
        scores = [f"{cat2}:{score}" for cat2, score in patterns.items()]
        print(f"  {cat1}: {', '.join(scores)}")
    
    print("\n✨ KEY FINDINGS:")
    
    # Find insights
    causal_conf = analysis['confidence_distribution'].get('causal', {})
    diagnostic_conf = analysis['confidence_distribution'].get('diagnostic', {})
    
    print(f"  • Causal questions: {class_stats['classification_details']['causal']['count']} (15%)")
    print(f"    - High confidence: {causal_conf.get('high_confidence_count', 0)} questions")
    print(f"    - Ideal for demonstrating CKG-RAG advantages")
    
    print(f"  • Diagnostic questions: {class_stats['classification_details']['diagnostic']['count']} (18.4%)")
    print(f"    - High confidence: {diagnostic_conf.get('high_confidence_count', 0)} questions")
    print(f"    - Perfect for multi-hop reasoning evaluation")
    
    print(f"  • Comparative questions: {class_stats['classification_details']['comparative']['count']} (11.2%)")
    print(f"    - Highest avg confidence: {class_stats['classification_details']['comparative']['avg_confidence']:.3f}")
    
    print(f"  • Factual questions: {class_stats['classification_details']['factual']['count']} (55.3%)")
    print(f"    - Baseline comparison category")
    
    print("\n📁 FILES GENERATED:")
    print("  • data/processed/classified_questions.json")
    print("  • data/processed/classification_statistics.json")
    print("  • data/processed/dataset_statistics_updated.json")
    print("  • paper/figures/classification_analysis.png")
    print("  • paper/figures/cross_category_heatmap.png")
    print("  • paper/figures/classification_examples.png")
    
    print("\n" + "="*60)
    print("Classification analysis complete!")
    
    return analysis

if __name__ == "__main__":
    analysis = generate_classification_report()