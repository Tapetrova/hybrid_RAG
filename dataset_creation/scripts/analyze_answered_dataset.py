"""
Comprehensive analysis of the cleaned dataset (questions with answers only)
"""

import json
import os
from collections import Counter
import statistics
from datetime import datetime

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def analyze_comprehensive(data):
    """Perform comprehensive analysis of the dataset"""
    questions = data['questions'] if isinstance(data, dict) and 'questions' in data else data
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE DATASET ANALYSIS")
    print(f"{'='*80}")
    
    # Basic counts
    total = len(questions)
    print(f"\nTotal Questions with Answers: {total}")
    
    # Category analysis
    categories = []
    category_scores = {}
    for q in questions:
        cat = q.get('category', 'unknown')
        categories.append(cat)
        if cat not in category_scores:
            category_scores[cat] = []
        if 'metadata' in q and 'score' in q['metadata']:
            category_scores[cat].append(q['metadata']['score'])
    
    category_counts = Counter(categories)
    print(f"\n{'='*50}")
    print("CATEGORY DISTRIBUTION")
    print(f"{'='*50}")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        avg_score = statistics.mean(category_scores[cat]) if category_scores[cat] else 0
        print(f"{cat:20s}: {count:4d} ({percentage:5.1f}%) - Avg Score: {avg_score:.1f}")
    
    # Tag analysis
    all_tags = []
    collected_tags = []
    for q in questions:
        if 'metadata' in q:
            if 'tags' in q['metadata'] and isinstance(q['metadata']['tags'], list):
                all_tags.extend(q['metadata']['tags'])
            if 'collected_tag' in q['metadata']:
                collected_tags.append(q['metadata']['collected_tag'])
    
    tag_counts = Counter(all_tags)
    collected_tag_counts = Counter(collected_tags)
    
    print(f"\n{'='*50}")
    print("TAG ANALYSIS")
    print(f"{'='*50}")
    print(f"Total unique tags: {len(tag_counts)}")
    print(f"\nTop 15 Most Common Tags:")
    for tag, count in tag_counts.most_common(15):
        print(f"  {tag:30s}: {count:4d}")
    
    print(f"\n{'='*50}")
    print("SOURCE TAGS (Collection Categories)")
    print(f"{'='*50}")
    for tag, count in sorted(collected_tag_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"{tag:20s}: {count:4d} ({percentage:5.1f}%)")
    
    # Score analysis
    scores = []
    for q in questions:
        if 'metadata' in q and 'score' in q['metadata']:
            scores.append(q['metadata']['score'])
    
    if scores:
        print(f"\n{'='*50}")
        print("QUALITY METRICS (Stack Exchange Scores)")
        print(f"{'='*50}")
        print(f"Average score: {statistics.mean(scores):.1f}")
        print(f"Median score: {statistics.median(scores):.1f}")
        print(f"Max score: {max(scores)}")
        print(f"Min score: {min(scores)}")
        print(f"High quality (score ≥ 10): {len([s for s in scores if s >= 10])} ({len([s for s in scores if s >= 10])/total*100:.1f}%)")
        print(f"Very high quality (score ≥ 20): {len([s for s in scores if s >= 20])} ({len([s for s in scores if s >= 20])/total*100:.1f}%)")
        print(f"Exceptional (score ≥ 50): {len([s for s in scores if s >= 50])} ({len([s for s in scores if s >= 50])/total*100:.1f}%)")
    
    # Text length analysis
    question_lengths = []
    answer_lengths = []
    context_lengths = []
    
    for q in questions:
        if 'question' in q:
            question_lengths.append(len(q['question'].split()))
        if 'answer' in q:
            answer_lengths.append(len(q['answer'].split()))
        if 'context' in q:
            context_lengths.append(len(q['context'].split()))
    
    print(f"\n{'='*50}")
    print("TEXT LENGTH STATISTICS")
    print(f"{'='*50}")
    print(f"Questions:")
    print(f"  Average: {statistics.mean(question_lengths):.1f} words")
    print(f"  Median: {statistics.median(question_lengths):.1f} words")
    print(f"  Max: {max(question_lengths)} words")
    print(f"  Min: {min(question_lengths)} words")
    
    print(f"\nAnswers:")
    print(f"  Average: {statistics.mean(answer_lengths):.1f} words")
    print(f"  Median: {statistics.median(answer_lengths):.1f} words")
    print(f"  Max: {max(answer_lengths)} words")
    print(f"  Min: {min(answer_lengths)} words")
    
    if context_lengths:
        print(f"\nContext:")
        print(f"  Average: {statistics.mean(context_lengths):.1f} words")
        print(f"  Median: {statistics.median(context_lengths):.1f} words")
        print(f"  Max: {max(context_lengths)} words")
        print(f"  Min: {min(context_lengths)} words")
    
    # Classification confidence analysis
    confidences = []
    for q in questions:
        if 'classification_confidence' in q:
            confidences.append(q['classification_confidence'])
    
    if confidences:
        print(f"\n{'='*50}")
        print("CLASSIFICATION CONFIDENCE")
        print(f"{'='*50}")
        print(f"Average confidence: {statistics.mean(confidences):.3f}")
        print(f"High confidence (≥ 0.8): {len([c for c in confidences if c >= 0.8])} ({len([c for c in confidences if c >= 0.8])/len(confidences)*100:.1f}%)")
        print(f"Medium confidence (0.5-0.8): {len([c for c in confidences if 0.5 <= c < 0.8])} ({len([c for c in confidences if 0.5 <= c < 0.8])/len(confidences)*100:.1f}%)")
        print(f"Low confidence (< 0.5): {len([c for c in confidences if c < 0.5])} ({len([c for c in confidences if c < 0.5])/len(confidences)*100:.1f}%)")
    
    # Return summary statistics
    return {
        'total_questions': total,
        'categories': dict(category_counts),
        'top_tags': dict(tag_counts.most_common(20)),
        'collected_tags': dict(collected_tag_counts),
        'score_stats': {
            'mean': statistics.mean(scores) if scores else 0,
            'median': statistics.median(scores) if scores else 0,
            'max': max(scores) if scores else 0,
            'min': min(scores) if scores else 0,
            'high_quality_count': len([s for s in scores if s >= 10]) if scores else 0,
            'very_high_quality_count': len([s for s in scores if s >= 20]) if scores else 0
        },
        'text_stats': {
            'avg_question_words': statistics.mean(question_lengths) if question_lengths else 0,
            'avg_answer_words': statistics.mean(answer_lengths) if answer_lengths else 0,
            'avg_context_words': statistics.mean(context_lengths) if context_lengths else 0
        }
    }

def main():
    base_dir = '/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data'
    
    # Load main dataset
    main_file = os.path.join(base_dir, 'apqc_auto.json')
    data = load_json(main_file)
    
    # Perform analysis
    stats = analyze_comprehensive(data)
    
    # Save statistics
    stats_file = os.path.join(base_dir, 'final_statistics_answered_only.json')
    save_json(stats, stats_file)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"Statistics saved to: {stats_file}")
    print(f"{'='*80}")
    
    # Generate summary report
    report = f"""
# AUTOMOTIVE Q&A DATASET - FINAL STATISTICS
## Questions with Answers Only

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Questions**: {stats['total_questions']}
- **Source**: Stack Exchange Mechanics (mechanics.stackexchange.com)
- **Data Type**: Questions with verified answers only
- **Average Question Score**: {stats['score_stats']['mean']:.1f}

## Category Distribution
"""
    for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_questions']) * 100
        report += f"- **{cat}**: {count} ({percentage:.1f}%)\n"
    
    report += f"""
## Quality Metrics
- High Quality (score ≥ 10): {stats['score_stats']['high_quality_count']} questions
- Very High Quality (score ≥ 20): {stats['score_stats']['very_high_quality_count']} questions
- Maximum Score: {stats['score_stats']['max']}

## Text Statistics
- Average Question Length: {stats['text_stats']['avg_question_words']:.1f} words
- Average Answer Length: {stats['text_stats']['avg_answer_words']:.1f} words
- Average Context Length: {stats['text_stats']['avg_context_words']:.1f} words

## Collection Categories
"""
    for tag, count in sorted(stats['collected_tags'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_questions']) * 100
        report += f"- {tag}: {count} ({percentage:.1f}%)\n"
    
    # Save report
    report_file = os.path.join(base_dir, '..', 'documentation', 'FINAL_ANALYSIS_REPORT.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    main()