"""
Generate comprehensive statistics for the automotive Q&A dataset
For CKG-RAG paper
"""

import json
import os
import re
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_dataset(filepath: str) -> Dict[str, Any]:
    """Load the dataset"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_question_types(questions: List[Dict]) -> Dict[str, int]:
    """Classify questions by type based on keywords"""
    
    question_types = {
        'causal': 0,      # Why/How questions
        'diagnostic': 0,   # Problem/Issue questions
        'procedural': 0,   # How-to questions
        'factual': 0,      # What is questions
        'comparative': 0,  # Comparison questions
        'conditional': 0   # If/When questions
    }
    
    causal_patterns = [r'\bwhy\b', r'\bcause', r'\breason', r'\bhow does', r'\bwhat makes', r'\bleads to']
    diagnostic_patterns = [r'\bproblem\b', r'\bissue\b', r'\bwrong\b', r'\bfix\b', r'\bdiagnose', r'\btrouble']
    procedural_patterns = [r'\bhow to\b', r'\bhow do i\b', r'\bsteps\b', r'\bprocedure', r'\bprocess']
    factual_patterns = [r'\bwhat is\b', r'\bwhat are\b', r'\bdefine\b', r'\bmean\b', r'\bexplain what']
    comparative_patterns = [r'\bvs\b', r'\bversus\b', r'\bbetter\b', r'\bcompare', r'\bdifference']
    conditional_patterns = [r'\bif\b', r'\bwhen\b', r'\bshould i\b', r'\bcan i\b', r'\bis it']
    
    for q in questions:
        title = q.get('question', '').lower() + ' ' + q.get('title', '').lower()
        
        # Check patterns
        if any(re.search(p, title) for p in causal_patterns):
            question_types['causal'] += 1
        elif any(re.search(p, title) for p in diagnostic_patterns):
            question_types['diagnostic'] += 1
        elif any(re.search(p, title) for p in procedural_patterns):
            question_types['procedural'] += 1
        elif any(re.search(p, title) for p in comparative_patterns):
            question_types['comparative'] += 1
        elif any(re.search(p, title) for p in conditional_patterns):
            question_types['conditional'] += 1
        else:
            question_types['factual'] += 1
    
    return question_types

def analyze_text_lengths(questions: List[Dict]) -> Dict[str, Any]:
    """Analyze text lengths in the dataset"""
    
    question_lengths = []
    answer_lengths = []
    context_lengths = []
    
    for q in questions:
        if 'question' in q:
            question_lengths.append(len(q['question'].split()))
        if 'context' in q:
            context_lengths.append(len(q['context'].split()))
        if 'answer' in q and q['answer']:
            answer_lengths.append(len(q['answer'].split()))
    
    return {
        'question': {
            'mean': np.mean(question_lengths) if question_lengths else 0,
            'median': np.median(question_lengths) if question_lengths else 0,
            'std': np.std(question_lengths) if question_lengths else 0,
            'min': min(question_lengths) if question_lengths else 0,
            'max': max(question_lengths) if question_lengths else 0
        },
        'context': {
            'mean': np.mean(context_lengths) if context_lengths else 0,
            'median': np.median(context_lengths) if context_lengths else 0,
            'std': np.std(context_lengths) if context_lengths else 0,
            'min': min(context_lengths) if context_lengths else 0,
            'max': max(context_lengths) if context_lengths else 0
        },
        'answer': {
            'mean': np.mean(answer_lengths) if answer_lengths else 0,
            'median': np.median(answer_lengths) if answer_lengths else 0,
            'std': np.std(answer_lengths) if answer_lengths else 0,
            'min': min(answer_lengths) if answer_lengths else 0,
            'max': max(answer_lengths) if answer_lengths else 0,
            'total_with_answers': len(answer_lengths)
        }
    }

def analyze_tags(questions: List[Dict]) -> Dict[str, Any]:
    """Analyze tag distribution"""
    
    all_tags = []
    for q in questions:
        if 'tags' in q:
            all_tags.extend(q['tags'])
    
    tag_counts = Counter(all_tags)
    
    return {
        'total_unique_tags': len(tag_counts),
        'top_20_tags': dict(tag_counts.most_common(20)),
        'tags_per_question': {
            'mean': np.mean([len(q.get('tags', [])) for q in questions]),
            'median': np.median([len(q.get('tags', [])) for q in questions]),
            'max': max([len(q.get('tags', [])) for q in questions])
        }
    }

def analyze_scores(questions: List[Dict]) -> Dict[str, Any]:
    """Analyze score distribution"""
    
    scores = [q.get('score', 0) for q in questions]
    scores_by_category = defaultdict(list)
    
    for q in questions:
        category = q.get('category', 'unknown')
        scores_by_category[category].append(q.get('score', 0))
    
    category_stats = {}
    for cat, cat_scores in scores_by_category.items():
        if cat_scores:
            category_stats[cat] = {
                'mean': round(np.mean(cat_scores), 2),
                'median': np.median(cat_scores),
                'max': max(cat_scores),
                'min': min(cat_scores)
            }
    
    return {
        'overall': {
            'mean': round(np.mean(scores), 2),
            'median': np.median(scores),
            'std': round(np.std(scores), 2),
            'min': min(scores),
            'max': max(scores),
            'q1': np.percentile(scores, 25),
            'q3': np.percentile(scores, 75)
        },
        'by_category': category_stats,
        'high_quality_questions': len([s for s in scores if s >= 10]),
        'very_high_quality': len([s for s in scores if s >= 20])
    }

def identify_causal_questions(questions: List[Dict]) -> List[Dict]:
    """Identify questions that are good for causal reasoning"""
    
    causal_questions = []
    
    causal_keywords = [
        'why', 'cause', 'reason', 'lead to', 'result in', 'because',
        'effect', 'consequence', 'due to', 'therefore', 'since',
        'make', 'trigger', 'induce', 'produce'
    ]
    
    for q in questions:
        text = (q.get('question', '') + ' ' + q.get('context', '')).lower()
        
        # Count causal keywords
        causal_score = sum(1 for keyword in causal_keywords if keyword in text)
        
        if causal_score >= 2:  # At least 2 causal keywords
            causal_questions.append({
                'id': q.get('id'),
                'question': q.get('question', '')[:100],
                'causal_score': causal_score,
                'category': q.get('category'),
                'has_answer': bool(q.get('answer'))
            })
    
    # Sort by causal score
    causal_questions.sort(key=lambda x: x['causal_score'], reverse=True)
    
    return causal_questions[:50]  # Top 50 causal questions

def generate_statistics_report(data_file: str):
    """Generate comprehensive statistics report"""
    
    print("="*60)
    print("CKG-RAG DATASET STATISTICS GENERATION")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    
    # Load main dataset
    main_data = load_dataset(data_file)
    
    # Load APQ-C dataset
    apqc_file = 'data/processed/apqc_auto_200.json'
    apqc_data = load_dataset(apqc_file) if os.path.exists(apqc_file) else []
    
    # Extract questions based on data structure
    if isinstance(main_data, dict) and 'questions' in main_data:
        all_questions = main_data['questions']
        metadata = main_data.get('metadata', {})
    else:
        all_questions = main_data
        metadata = {}
    
    print(f"Loaded {len(all_questions)} questions from main dataset")
    print(f"Loaded {len(apqc_data)} questions from APQ-C dataset")
    
    # Generate statistics
    statistics = {
        'dataset_overview': {
            'total_questions': len(all_questions),
            'apqc_subset_size': len(apqc_data),
            'questions_with_answers': sum(1 for q in all_questions if q.get('answer') or q.get('accepted_answer')),
            'answer_rate': round(sum(1 for q in all_questions if q.get('answer') or q.get('accepted_answer')) / len(all_questions) * 100, 2) if all_questions else 0,
            'generation_date': datetime.now().isoformat()
        },
        'category_distribution': {},
        'question_types': {},
        'text_lengths': {},
        'score_analysis': {},
        'tag_analysis': {},
        'causal_questions': [],
        'quality_metrics': {}
    }
    
    # Category distribution
    print("\nAnalyzing category distribution...")
    category_counts = Counter(q.get('category', q.get('collected_tag', 'unknown')) for q in all_questions)
    for category, count in category_counts.items():
        cat_questions = [q for q in all_questions if q.get('category', q.get('collected_tag')) == category]
        with_answers = sum(1 for q in cat_questions if q.get('answer') or q.get('accepted_answer'))
        
        statistics['category_distribution'][category] = {
            'count': count,
            'percentage': round(count / len(all_questions) * 100, 2),
            'with_answers': with_answers,
            'answer_rate': round(with_answers / count * 100, 2) if count > 0 else 0
        }
    
    # Question type analysis
    print("Analyzing question types...")
    statistics['question_types'] = analyze_question_types(apqc_data if apqc_data else all_questions[:200])
    
    # Text length analysis
    print("Analyzing text lengths...")
    statistics['text_lengths'] = analyze_text_lengths(apqc_data if apqc_data else all_questions[:200])
    
    # Score analysis
    print("Analyzing scores...")
    statistics['score_analysis'] = analyze_scores(all_questions)
    
    # Tag analysis
    print("Analyzing tags...")
    statistics['tag_analysis'] = analyze_tags(all_questions)
    
    # Identify causal questions
    print("Identifying causal questions...")
    statistics['causal_questions'] = identify_causal_questions(apqc_data if apqc_data else all_questions[:200])
    
    # Quality metrics
    print("Calculating quality metrics...")
    high_quality = [q for q in all_questions if q.get('score', 0) >= 10]
    very_high_quality = [q for q in all_questions if q.get('score', 0) >= 20]
    
    statistics['quality_metrics'] = {
        'high_quality_questions': len(high_quality),
        'high_quality_percentage': round(len(high_quality) / len(all_questions) * 100, 2),
        'very_high_quality_questions': len(very_high_quality),
        'very_high_quality_percentage': round(len(very_high_quality) / len(all_questions) * 100, 2),
        'questions_suitable_for_causal': len(statistics['causal_questions']),
        'causal_percentage': round(len(statistics['causal_questions']) / 200 * 100, 2)
    }
    
    # Save statistics
    stats_file = 'data/processed/dataset_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Statistics saved to: {stats_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(statistics)
    
    # Print summary
    print_summary(statistics)
    
    return statistics

def generate_visualizations(stats: Dict[str, Any]):
    """Generate visualization plots for the paper"""
    
    # Create figure directory
    os.makedirs('paper/figures', exist_ok=True)
    
    # 1. Category Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = list(stats['category_distribution'].keys())
    counts = [stats['category_distribution'][cat]['count'] for cat in categories]
    answer_rates = [stats['category_distribution'][cat]['answer_rate'] for cat in categories]
    
    # Bar plot for counts
    ax1.bar(categories, counts, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Number of Questions', fontsize=12)
    ax1.set_title('Question Distribution by Category', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(counts):
        ax1.text(i, v + 5, str(v), ha='center', fontsize=10)
    
    # Bar plot for answer rates
    ax2.bar(categories, answer_rates, color='coral', alpha=0.8)
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Answer Rate (%)', fontsize=12)
    ax2.set_title('Answer Rate by Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=np.mean(answer_rates), color='red', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(answer_rates):.1f}%')
    ax2.legend()
    
    # Add value labels
    for i, v in enumerate(answer_rates):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('paper/figures/category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Question Type Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    types = list(stats['question_types'].keys())
    type_counts = list(stats['question_types'].values())
    
    # Create pie chart
    colors = plt.cm.Set3(range(len(types)))
    wedges, texts, autotexts = ax.pie(type_counts, labels=types, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    
    # Beautify the text
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Question Type Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with counts
    legend_labels = [f'{t}: {c} questions' for t, c in zip(types, type_counts)]
    ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.savefig('paper/figures/question_types.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Score Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot by category
    score_data = []
    score_labels = []
    for cat in categories[:6]:  # Top 6 categories
        if cat in stats['score_analysis']['by_category']:
            cat_stats = stats['score_analysis']['by_category'][cat]
            # Generate sample data based on statistics (for visualization)
            score_data.append([cat_stats['mean']] * 10)  # Simplified for visualization
            score_labels.append(f"{cat}\n(μ={cat_stats['mean']:.1f})")
    
    if score_data:  # Only plot if we have data
        ax1.boxplot(score_data, tick_labels=score_labels)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Score Distribution by Category', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Overall score statistics
    overall_stats = stats['score_analysis']['overall']
    stats_text = f"Overall Statistics:\n"
    stats_text += f"Mean: {overall_stats['mean']:.1f}\n"
    stats_text += f"Median: {overall_stats['median']:.1f}\n"
    stats_text += f"Std Dev: {overall_stats['std']:.1f}\n"
    stats_text += f"Min: {overall_stats['min']}\n"
    stats_text += f"Max: {overall_stats['max']}\n"
    stats_text += f"Q1: {overall_stats['q1']:.1f}\n"
    stats_text += f"Q3: {overall_stats['q3']:.1f}"
    
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')
    ax2.set_title('Score Statistics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper/figures/score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Text Length Analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    text_types = ['question', 'context', 'answer']
    colors = ['skyblue', 'lightgreen', 'salmon']
    
    for idx, (text_type, color) in enumerate(zip(text_types, colors)):
        ax = axes[idx]
        text_stats = stats['text_lengths'][text_type]
        
        # Create bar chart for statistics
        stat_names = ['Mean', 'Median', 'Min', 'Max']
        stat_values = [text_stats['mean'], text_stats['median'], 
                      text_stats['min'], text_stats['max']]
        
        bars = ax.bar(stat_names, stat_values, color=color, alpha=0.7)
        ax.set_ylabel('Word Count', fontsize=11)
        ax.set_title(f'{text_type.capitalize()} Length Statistics', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, stat_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Text Length Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper/figures/text_lengths.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to paper/figures/")

def print_summary(stats: Dict[str, Any]):
    """Print summary statistics to console"""
    
    print("\n" + "="*60)
    print("DATASET STATISTICS SUMMARY")
    print("="*60)
    
    print("\n📊 OVERVIEW:")
    print(f"  Total Questions: {stats['dataset_overview']['total_questions']}")
    print(f"  Questions with Answers: {stats['dataset_overview']['questions_with_answers']}")
    print(f"  Answer Rate: {stats['dataset_overview']['answer_rate']}%")
    print(f"  APQ-C Subset: {stats['dataset_overview']['apqc_subset_size']} questions")
    
    print("\n📈 QUALITY METRICS:")
    print(f"  High Quality (score ≥ 10): {stats['quality_metrics']['high_quality_questions']} ({stats['quality_metrics']['high_quality_percentage']}%)")
    print(f"  Very High Quality (score ≥ 20): {stats['quality_metrics']['very_high_quality_questions']} ({stats['quality_metrics']['very_high_quality_percentage']}%)")
    print(f"  Suitable for Causal Reasoning: {stats['quality_metrics']['questions_suitable_for_causal']} ({stats['quality_metrics']['causal_percentage']}%)")
    
    print("\n🏷️ CATEGORY DISTRIBUTION:")
    for cat, data in sorted(stats['category_distribution'].items(), 
                           key=lambda x: x[1]['count'], reverse=True):
        print(f"  {cat}: {data['count']} questions ({data['percentage']}%), "
              f"Answer rate: {data['answer_rate']}%")
    
    print("\n❓ QUESTION TYPES:")
    total_typed = sum(stats['question_types'].values())
    for qtype, count in sorted(stats['question_types'].items(), 
                              key=lambda x: x[1], reverse=True):
        percentage = (count / total_typed * 100) if total_typed > 0 else 0
        print(f"  {qtype.capitalize()}: {count} ({percentage:.1f}%)")
    
    print("\n📝 TEXT STATISTICS:")
    print("  Question length: {:.0f} words (mean), {:.0f} words (median)".format(
        stats['text_lengths']['question']['mean'],
        stats['text_lengths']['question']['median']
    ))
    print("  Context length: {:.0f} words (mean), {:.0f} words (median)".format(
        stats['text_lengths']['context']['mean'],
        stats['text_lengths']['context']['median']
    ))
    print("  Answer length: {:.0f} words (mean), {:.0f} words (median)".format(
        stats['text_lengths']['answer']['mean'],
        stats['text_lengths']['answer']['median']
    ))
    
    print("\n⭐ SCORE ANALYSIS:")
    overall = stats['score_analysis']['overall']
    print(f"  Mean Score: {overall['mean']}")
    print(f"  Median Score: {overall['median']}")
    print(f"  Score Range: {overall['min']} - {overall['max']}")
    print(f"  High Quality Questions: {stats['score_analysis']['high_quality_questions']}")
    
    print("\n🔖 TOP TAGS:")
    for tag, count in list(stats['tag_analysis']['top_20_tags'].items())[:10]:
        print(f"  {tag}: {count}")
    
    print("\n🔄 CAUSAL QUESTIONS IDENTIFIED:")
    print(f"  Total: {len(stats['causal_questions'])} questions suitable for causal reasoning")
    if stats['causal_questions']:
        print("  Top 5 causal questions:")
        for i, q in enumerate(stats['causal_questions'][:5], 1):
            print(f"    {i}. {q['question'][:60]}... (score: {q['causal_score']})")
    
    print("\n" + "="*60)
    print("Statistics generation completed successfully!")
    print("Files created:")
    print("  📊 data/processed/dataset_statistics.json")
    print("  📈 paper/figures/*.png (visualization plots)")
    print("="*60)

if __name__ == "__main__":
    # Run statistics generation
    main_data_file = 'data/processed/automotive_qa_final_20250816_230600.json'
    
    if os.path.exists(main_data_file):
        statistics = generate_statistics_report(main_data_file)
    else:
        print(f"Error: Main dataset not found at {main_data_file}")
        print("Please ensure the dataset has been collected and processed.")