"""
Create final APQ-C dataset for CKG-RAG paper
Removes duplicates using cosine similarity and creates clean dataset
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

class DatasetFinalizer:
    """
    Finalize dataset by removing duplicates and creating APQ-C format
    """
    
    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize with similarity threshold for duplicate detection
        
        Args:
            similarity_threshold: Cosine similarity threshold (default 0.9)
        """
        self.similarity_threshold = similarity_threshold
        print("Loading sentence transformer model...")
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Model loaded successfully")
        
    def compute_embeddings(self, questions: List[Dict]) -> np.ndarray:
        """
        Compute embeddings for all questions
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Numpy array of embeddings
        """
        print(f"\nComputing embeddings for {len(questions)} questions...")
        
        # Prepare texts for encoding
        texts = []
        for q in questions:
            # Combine title and body for better representation
            title = q.get('title', q.get('question', ''))
            body = q.get('body', q.get('context', ''))[:500]  # Limit body length
            combined_text = f"{title} {body}"
            texts.append(combined_text)
        
        # Batch encode for efficiency
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 256 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} questions...")
        
        return np.array(all_embeddings)
    
    def find_duplicates(self, questions: List[Dict], embeddings: np.ndarray) -> Dict[int, List[int]]:
        """
        Find duplicate questions using cosine similarity
        
        Args:
            questions: List of questions
            embeddings: Question embeddings
            
        Returns:
            Dictionary mapping question index to list of duplicate indices
        """
        print(f"\nFinding duplicates (threshold: {self.similarity_threshold})...")
        
        # Compute similarity matrix
        print("  Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find duplicates
        duplicates = defaultdict(list)
        processed = set()
        
        for i in range(len(questions)):
            if i in processed:
                continue
                
            # Find all questions similar to this one
            similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
            
            # Exclude self
            similar_indices = [idx for idx in similar_indices if idx != i]
            
            if similar_indices:
                # Mark all as processed
                for idx in similar_indices:
                    processed.add(idx)
                    duplicates[i].append(idx)
        
        print(f"  Found {len(duplicates)} groups of duplicates")
        print(f"  Total duplicate questions to remove: {sum(len(v) for v in duplicates.values())}")
        
        return duplicates
    
    def select_best_from_duplicates(self, questions: List[Dict], duplicate_groups: Dict[int, List[int]]) -> List[int]:
        """
        Select the best question from each duplicate group
        
        Args:
            questions: List of questions
            duplicate_groups: Groups of duplicate questions
            
        Returns:
            List of indices to keep
        """
        indices_to_keep = []
        indices_to_remove = set()
        
        # Process duplicate groups
        for main_idx, duplicate_indices in duplicate_groups.items():
            group = [main_idx] + duplicate_indices
            
            # Select best based on: 1) has answer, 2) score, 3) length
            best_idx = main_idx
            best_score = self._calculate_quality_score(questions[main_idx])
            
            for idx in duplicate_indices:
                score = self._calculate_quality_score(questions[idx])
                if score > best_score:
                    best_idx = idx
                    best_score = score
            
            indices_to_keep.append(best_idx)
            
            # Mark others for removal
            for idx in group:
                if idx != best_idx:
                    indices_to_remove.add(idx)
        
        # Add all non-duplicate questions
        for i in range(len(questions)):
            if i not in indices_to_remove and i not in [idx for group in duplicate_groups.values() for idx in group]:
                if i not in duplicate_groups:  # Not a main index of duplicate group
                    indices_to_keep.append(i)
        
        return sorted(list(set(indices_to_keep)))
    
    def _calculate_quality_score(self, question: Dict) -> float:
        """Calculate quality score for a question"""
        score = 0.0
        
        # Has answer
        if question.get('accepted_answer') or question.get('answer'):
            score += 10.0
        
        # Question score
        score += min(question.get('score', 0), 20) / 2  # Cap at 20, divide by 2
        
        # Has classification
        if 'classification' in question:
            score += 2.0
            # High confidence classification
            if question['classification'].get('confidence', 0) > 0.7:
                score += 3.0
        
        # Text length (prefer moderate length)
        text_length = len(question.get('title', '') + question.get('body', question.get('context', '')))
        if 100 < text_length < 1000:
            score += 2.0
        
        return score
    
    def create_apqc_format(self, question: Dict) -> Dict:
        """
        Convert question to APQ-C format
        
        Args:
            question: Original question dictionary
            
        Returns:
            APQ-C formatted question
        """
        # Extract components
        question_text = question.get('title', question.get('question', ''))
        context = question.get('body', question.get('context', ''))
        
        # Get answer
        answer = ''
        if question.get('accepted_answer'):
            if isinstance(question['accepted_answer'], dict):
                answer = question['accepted_answer'].get('body', '')
            else:
                answer = question['accepted_answer']
        elif question.get('answer'):
            answer = question['answer']
        
        # Clean HTML if present
        answer = self._clean_html(answer)
        context = self._clean_html(context)
        
        # Get classification
        category = 'factual'  # default
        confidence = 0.0
        if 'classification' in question:
            category = question['classification'].get('primary_category', 'factual')
            confidence = question['classification'].get('confidence', 0.0)
        
        # Create APQ-C format
        apqc_question = {
            'id': f"auto_{question.get('question_id', question.get('id', 'unknown'))}",
            'question': question_text,
            'context': context[:1000],  # Limit context length
            'answer': answer[:2000],     # Limit answer length
            'category': category,
            'classification_confidence': round(confidence, 3),
            'metadata': {
                'source': 'stack_exchange',
                'score': question.get('score', 0),
                'tags': question.get('tags', []),
                'has_accepted_answer': bool(question.get('accepted_answer')),
                'collected_tag': question.get('collected_tag', question.get('category', 'unknown'))
            }
        }
        
        return apqc_question
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        if not text:
            return ''
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Decode HTML entities
        text = text.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&#39;', "'").replace('&nbsp;', ' ')
        
        return text.strip()
    
    def process_dataset(self, input_file: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Process dataset: remove duplicates and convert to APQ-C format
        
        Args:
            input_file: Path to input JSON file
            
        Returns:
            Tuple of (processed questions, statistics)
        """
        # Load data
        print(f"\nLoading data from {input_file}...")
        with open(input_file, 'r') as f:
            questions = json.load(f)
        
        print(f"Loaded {len(questions)} questions")
        
        # Compute embeddings
        embeddings = self.compute_embeddings(questions)
        
        # Find duplicates
        duplicate_groups = self.find_duplicates(questions, embeddings)
        
        # Select best questions
        indices_to_keep = self.select_best_from_duplicates(questions, duplicate_groups)
        
        print(f"\nKeeping {len(indices_to_keep)} unique questions")
        print(f"Removed {len(questions) - len(indices_to_keep)} duplicates")
        
        # Filter questions
        unique_questions = [questions[i] for i in indices_to_keep]
        
        # Convert to APQ-C format
        print("\nConverting to APQ-C format...")
        apqc_questions = []
        for q in unique_questions:
            apqc_q = self.create_apqc_format(q)
            apqc_questions.append(apqc_q)
        
        # Calculate statistics
        stats = self.calculate_statistics(apqc_questions, len(questions), len(duplicate_groups))
        
        return apqc_questions, stats
    
    def calculate_statistics(self, questions: List[Dict], original_count: int, duplicate_groups: int) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        
        stats = {
            'total_questions': len(questions),
            'original_count': original_count,
            'duplicates_removed': original_count - len(questions),
            'duplicate_groups_found': duplicate_groups,
            'duplicate_removal_rate': (original_count - len(questions)) / original_count * 100 if original_count > 0 else 0,
            'categories': defaultdict(int),
            'with_answers': 0,
            'avg_question_length': 0,
            'avg_context_length': 0,
            'avg_answer_length': 0,
            'avg_score': 0,
            'high_confidence_classifications': 0
        }
        
        # Calculate detailed statistics
        question_lengths = []
        context_lengths = []
        answer_lengths = []
        scores = []
        
        for q in questions:
            # Category distribution
            stats['categories'][q['category']] += 1
            
            # Answer statistics
            if q['answer']:
                stats['with_answers'] += 1
                answer_lengths.append(len(q['answer'].split()))
            
            # Length statistics
            question_lengths.append(len(q['question'].split()))
            context_lengths.append(len(q['context'].split()))
            
            # Score statistics
            scores.append(q['metadata']['score'])
            
            # Classification confidence
            if q['classification_confidence'] > 0.7:
                stats['high_confidence_classifications'] += 1
        
        # Calculate averages
        stats['avg_question_length'] = np.mean(question_lengths) if question_lengths else 0
        stats['avg_context_length'] = np.mean(context_lengths) if context_lengths else 0
        stats['avg_answer_length'] = np.mean(answer_lengths) if answer_lengths else 0
        stats['avg_score'] = np.mean(scores) if scores else 0
        stats['answer_rate'] = stats['with_answers'] / len(questions) * 100 if questions else 0
        
        # Convert defaultdict to regular dict
        stats['categories'] = dict(stats['categories'])
        
        return stats


def main():
    """Main function to create final dataset"""
    
    print("="*60)
    print("CREATING FINAL APQ-C DATASET")
    print("="*60)
    
    # Initialize finalizer with more aggressive threshold to remove near-duplicates
    finalizer = DatasetFinalizer(similarity_threshold=0.8)
    
    # Process classified questions
    input_file = 'data/processed/classified_questions.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    # Process dataset
    apqc_questions, stats = finalizer.process_dataset(input_file)
    
    # Save final dataset
    output_file = 'data/apqc_auto.json'
    os.makedirs('data', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'dataset_name': 'APQ-C Automotive',
                'version': '1.0',
                'creation_date': datetime.now().isoformat(),
                'total_questions': len(apqc_questions),
                'categories': stats['categories'],
                'answer_rate': round(stats['answer_rate'], 2),
                'duplicate_removal': {
                    'threshold': 0.9,
                    'removed': stats['duplicates_removed']
                }
            },
            'questions': apqc_questions
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved final dataset to {output_file}")
    
    # Save detailed statistics
    stats_file = 'data/apqc_auto_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved statistics to {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    print(f"Total questions: {stats['total_questions']}")
    print(f"Duplicates removed: {stats['duplicates_removed']} ({stats['duplicate_removal_rate']:.1f}%)")
    print(f"Questions with answers: {stats['with_answers']} ({stats['answer_rate']:.1f}%)")
    print(f"Average score: {stats['avg_score']:.1f}")
    
    print("\nCategory distribution:")
    for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total_questions'] * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print("\nText statistics (words):")
    print(f"  Avg question length: {stats['avg_question_length']:.1f}")
    print(f"  Avg context length: {stats['avg_context_length']:.1f}")
    print(f"  Avg answer length: {stats['avg_answer_length']:.1f}")
    
    print(f"\nHigh confidence classifications: {stats['high_confidence_classifications']} ({stats['high_confidence_classifications']/stats['total_questions']*100:.1f}%)")
    
    # Create subset datasets for experiments
    create_experiment_subsets(apqc_questions)
    
    return apqc_questions, stats


def create_experiment_subsets(questions: List[Dict]):
    """Create subset datasets for specific experiments"""
    
    print("\nCreating experiment subsets...")
    
    # Causal subset
    causal_questions = [q for q in questions if q['category'] == 'causal']
    with open('data/apqc_auto_causal.json', 'w') as f:
        json.dump(causal_questions, f, indent=2)
    print(f"  Created causal subset: {len(causal_questions)} questions")
    
    # Diagnostic subset
    diagnostic_questions = [q for q in questions if q['category'] == 'diagnostic']
    with open('data/apqc_auto_diagnostic.json', 'w') as f:
        json.dump(diagnostic_questions, f, indent=2)
    print(f"  Created diagnostic subset: {len(diagnostic_questions)} questions")
    
    # High confidence subset
    high_conf_questions = [q for q in questions if q['classification_confidence'] > 0.7]
    with open('data/apqc_auto_high_confidence.json', 'w') as f:
        json.dump(high_conf_questions, f, indent=2)
    print(f"  Created high confidence subset: {len(high_conf_questions)} questions")
    
    # Questions with answers subset
    with_answers = [q for q in questions if q['answer']]
    with open('data/apqc_auto_with_answers.json', 'w') as f:
        json.dump(with_answers, f, indent=2)
    print(f"  Created with-answers subset: {len(with_answers)} questions")


if __name__ == "__main__":
    apqc_questions, stats = main()