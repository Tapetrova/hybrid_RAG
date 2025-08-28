"""
Script to clean dataset and keep only questions with answers
Also generates comprehensive statistics for the cleaned dataset
"""

import json
import os
from datetime import datetime
from collections import Counter
import statistics

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def filter_questions_with_answers(data):
    """Filter to keep only questions that have answers"""
    if isinstance(data, dict) and 'questions' in data:
        # Handle format with 'questions' key
        filtered_questions = [q for q in data['questions'] if q.get('answer') and q['answer'].strip()]
        return {'questions': filtered_questions, 'metadata': data.get('metadata', {})}
    elif isinstance(data, list):
        # Handle list format
        return [q for q in data if q.get('answer') and q['answer'].strip()]
    else:
        # Unknown format, return as is
        return data

def analyze_dataset(data):
    """Generate comprehensive statistics for the dataset"""
    questions = data['questions'] if isinstance(data, dict) and 'questions' in data else data
    
    if not questions:
        return {}
    
    # Basic statistics
    total_questions = len(questions)
    
    # Category distribution
    categories = [q.get('category', 'unknown') for q in questions]
    category_counts = Counter(categories)
    
    # Tags analysis
    all_tags = []
    for q in questions:
        if 'tags' in q and isinstance(q['tags'], list):
            all_tags.extend(q['tags'])
    tag_counts = Counter(all_tags)
    
    # Score statistics
    scores = [q.get('score', 0) for q in questions if 'score' in q]
    avg_score = statistics.mean(scores) if scores else 0
    median_score = statistics.median(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    
    # Text length statistics
    question_lengths = [len(q.get('question', '').split()) for q in questions]
    answer_lengths = [len(q.get('answer', '').split()) for q in questions]
    context_lengths = [len(q.get('context', '').split()) for q in questions if 'context' in q]
    
    # Question type analysis (if available)
    question_types = [q.get('question_type', 'unclassified') for q in questions if 'question_type' in q]
    type_counts = Counter(question_types)
    
    # View and answer count statistics
    view_counts = [q.get('view_count', 0) for q in questions if 'view_count' in q]
    answer_counts = [q.get('answer_count', 0) for q in questions if 'answer_count' in q]
    
    stats = {
        'total_questions_with_answers': total_questions,
        'categories': {
            'distribution': dict(category_counts),
            'total_categories': len(category_counts)
        },
        'tags': {
            'top_20_tags': dict(tag_counts.most_common(20)),
            'total_unique_tags': len(tag_counts)
        },
        'scores': {
            'average': round(avg_score, 2),
            'median': median_score,
            'max': max_score,
            'min': min_score,
            'high_quality_count': len([s for s in scores if s >= 10]),
            'very_high_quality_count': len([s for s in scores if s >= 20])
        },
        'text_statistics': {
            'avg_question_words': round(statistics.mean(question_lengths), 1) if question_lengths else 0,
            'avg_answer_words': round(statistics.mean(answer_lengths), 1) if answer_lengths else 0,
            'avg_context_words': round(statistics.mean(context_lengths), 1) if context_lengths else 0,
            'max_answer_words': max(answer_lengths) if answer_lengths else 0,
            'min_answer_words': min(answer_lengths) if answer_lengths else 0
        },
        'question_types': dict(type_counts) if type_counts else {},
        'engagement': {
            'avg_views': round(statistics.mean(view_counts), 1) if view_counts else 0,
            'avg_answer_count': round(statistics.mean(answer_counts), 1) if answer_counts else 0,
            'max_views': max(view_counts) if view_counts else 0
        }
    }
    
    return stats

def main():
    base_dir = '/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data'
    
    # Files to process
    files_to_clean = [
        'apqc_auto.json',
        'apqc_auto_causal.json',
        'apqc_auto_diagnostic.json',
        'apqc_auto_high_confidence.json',
        'processed/automotive_qa_final_20250816_230600.json',
        'processed/classified_questions.json'
    ]
    
    all_stats = {}
    
    print("=" * 80)
    print("CLEANING DATASETS - KEEPING ONLY QUESTIONS WITH ANSWERS")
    print("=" * 80)
    
    for file_path in files_to_clean:
        full_path = os.path.join(base_dir, file_path)
        
        if not os.path.exists(full_path):
            print(f"Skipping {file_path} - file not found")
            continue
            
        print(f"\nProcessing: {file_path}")
        print("-" * 40)
        
        # Load data
        data = load_json(full_path)
        
        # Count before filtering
        if isinstance(data, dict) and 'questions' in data:
            before_count = len(data['questions'])
        elif isinstance(data, list):
            before_count = len(data)
        else:
            before_count = 0
        
        # Filter to keep only questions with answers
        filtered_data = filter_questions_with_answers(data)
        
        # Count after filtering
        if isinstance(filtered_data, dict) and 'questions' in filtered_data:
            after_count = len(filtered_data['questions'])
        elif isinstance(filtered_data, list):
            after_count = len(filtered_data)
        else:
            after_count = 0
        
        # Save cleaned data (overwrite original)
        save_json(filtered_data, full_path)
        
        # Generate statistics
        stats = analyze_dataset(filtered_data)
        all_stats[file_path] = stats
        
        print(f"Before: {before_count} questions")
        print(f"After: {after_count} questions (with answers)")
        print(f"Removed: {before_count - after_count} questions without answers")
        
        if stats:
            print(f"Categories: {len(stats['categories']['distribution'])}")
            print(f"Average score: {stats['scores']['average']}")
            print(f"Average answer length: {stats['text_statistics']['avg_answer_words']} words")
    
    # Save comprehensive statistics
    stats_file = os.path.join(base_dir, 'dataset_statistics_answered_only.json')
    save_json(all_stats, stats_file)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF MAIN DATASET (apqc_auto.json)")
    print("=" * 80)
    
    if 'apqc_auto.json' in all_stats:
        main_stats = all_stats['apqc_auto.json']
        print(f"\nTotal Questions with Answers: {main_stats['total_questions_with_answers']}")
        print(f"\nCategory Distribution:")
        for cat, count in sorted(main_stats['categories']['distribution'].items(), 
                                 key=lambda x: x[1], reverse=True):
            percentage = (count / main_stats['total_questions_with_answers']) * 100
            print(f"  - {cat}: {count} ({percentage:.1f}%)")
        
        print(f"\nQuality Metrics:")
        print(f"  - Average Score: {main_stats['scores']['average']}")
        print(f"  - High Quality (score ≥ 10): {main_stats['scores']['high_quality_count']}")
        print(f"  - Very High Quality (score ≥ 20): {main_stats['scores']['very_high_quality_count']}")
        
        print(f"\nText Statistics:")
        print(f"  - Avg Question Length: {main_stats['text_statistics']['avg_question_words']} words")
        print(f"  - Avg Answer Length: {main_stats['text_statistics']['avg_answer_words']} words")
        print(f"  - Avg Context Length: {main_stats['text_statistics']['avg_context_words']} words")
        
        print(f"\nEngagement Metrics:")
        print(f"  - Average Views: {main_stats['engagement']['avg_views']}")
        print(f"  - Average Answer Count: {main_stats['engagement']['avg_answer_count']}")
    
    print("\n" + "=" * 80)
    print("DATASET CLEANING COMPLETE!")
    print(f"Statistics saved to: {stats_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()