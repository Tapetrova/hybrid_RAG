"""
Advanced Question Classification for CKG-RAG Paper
Classifies automotive questions into: causal, diagnostic, comparative, factual
Ensures minimum 50 questions per category for balanced evaluation
"""

import json
import re
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

class AutomotiveQueryClassifier:
    """
    Advanced classifier for automotive questions
    Based on agent_flow QueryClassifier architecture
    """
    
    def __init__(self):
        self.initialize_patterns()
        self.initialize_keywords()
        
    def initialize_patterns(self):
        """Initialize regex patterns for each question type"""
        
        self.patterns = {
            'causal': [
                # Direct causal questions
                r'\bwhy\s+(does|do|is|are|did|was|were|has|have|had|will|would|can|could|should)\b',
                r'\bwhat\s+causes?\b',
                r'\bwhat\s+(is|are)\s+the\s+(cause|reason)',
                r'\b(cause|reason)\s+(of|for|behind)\b',
                r'\bhow\s+does\s+\w+\s+(affect|impact|influence|cause)',
                r'\bleads?\s+to\b',
                r'\bresults?\s+in\b',
                r'\bdue\s+to\b',
                r'\bbecause\s+of\b',
                
                # Mechanism questions
                r'\bhow\s+does\s+\w+\s+work',
                r'\bwhat\s+makes?\b',
                r'\bwhat\s+happens?\s+(when|if)\b',
                r'\bexplain\s+(why|how)\b',
                
                # Effect questions
                r'\bwhat\s+(is|are)\s+the\s+effects?\b',
                r'\bwhat\s+happens?\s+to\b',
                r'\bconsequences?\s+of\b',
                r'\bimpact\s+of\b',
            ],
            
            'diagnostic': [
                # Problem identification
                r'\b(problem|issue|trouble|fault|defect|malfunction)\s+with\b',
                r'\bwhat\s+(is|could\s+be)\s+(wrong|the\s+problem)\b',
                r'\bwhy\s+(is|does)\s+my\s+\w+\s+not\s+working',
                r'\b(diagnose|troubleshoot|debug|fix|repair|solve)\b',
                
                # Symptom questions
                r'\b(symptom|sign|indication)\s+of\b',
                r'\bmeans?\s+when\b',
                r'\bindicates?\b',
                r'\b(normal|abnormal|unusual|strange|weird)\s+(sound|noise|behavior|smell|vibration)',
                
                # Failure questions
                r'\b(fail|failure|failed|failing|broken|broke|breaking)\b',
                r'\b(won\'t|wont|doesn\'t|doesnt|can\'t|cant)\s+(start|work|turn|move|shift)',
                r'\b(dead|died|dying)\s+\w+',
                
                # Check/test questions
                r'\bhow\s+(do\s+I|to)\s+(check|test|verify|inspect)\b',
                r'\b(bad|faulty|defective|worn|damaged)\s+\w+',
            ],
            
            'comparative': [
                # Direct comparisons
                r'\b(vs\.?|versus|compared?\s+to|against)\b',
                r'\b(better|worse|best|worst)\s+(than|for)\b',
                r'\b(difference|differences?)\s+between\b',
                r'\bwhich\s+(is|are|one|type|kind|brand)\b',
                
                # Choice questions
                r'\bshould\s+I\s+(use|choose|buy|get|select)\b',
                r'\b(or|either)\s+\w+\s+or\b',
                r'\b(pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)\b',
                r'\b(prefer|recommend|suggestion|advice)\b',
                
                # Performance comparisons
                r'\b(more|less|most|least)\s+(efficient|effective|reliable|durable)',
                r'\b(higher|lower|faster|slower|longer|shorter)\s+than\b',
                r'\bcompare\s+\w+\s+(with|to)\b',
                
                # Alternative questions
                r'\b(alternative|substitute|replacement|instead\s+of)\b',
                r'\b(same\s+as|similar\s+to|like|equivalent)\b',
            ],
            
            'factual': [
                # Definition questions
                r'\bwhat\s+(is|are)\s+(a|an|the)?\s*\w+\s*\?',
                r'\bdefine\s+\w+',
                r'\b(definition|meaning)\s+of\b',
                r'\bwhat\s+does\s+\w+\s+mean\b',
                
                # Specification questions
                r'\b(specification|spec|requirement|standard)\s+(for|of)\b',
                r'\bhow\s+(much|many|long|often|far|big|small)\b',
                r'\b(size|capacity|volume|weight|dimension|measurement)\s+of\b',
                
                # Component questions
                r'\bwhat\s+(type|kind|brand|model)\s+of\b',
                r'\b(part\s+number|model\s+number|code)\b',
                r'\b(location|position|where)\s+(is|are)\b',
                
                # Procedure questions (simple)
                r'\bhow\s+(do\s+I|to)\s+(change|replace|install|remove)\b',
                r'\b(steps?|procedure|process|method)\s+(to|for)\b',
                r'\b(interval|frequency|schedule)\s+for\b',
            ]
        }
    
    def initialize_keywords(self):
        """Initialize keyword sets for scoring"""
        
        self.keywords = {
            'causal': {
                'primary': ['why', 'cause', 'reason', 'because', 'lead', 'result', 'effect', 
                          'consequence', 'due', 'since', 'therefore', 'thus', 'hence'],
                'secondary': ['make', 'create', 'produce', 'trigger', 'induce', 'generate',
                            'contribute', 'influence', 'affect', 'impact', 'determine'],
                'context': ['mechanism', 'process', 'relationship', 'connection', 'link']
            },
            
            'diagnostic': {
                'primary': ['problem', 'issue', 'wrong', 'fail', 'broken', 'fix', 'repair',
                          'diagnose', 'troubleshoot', 'fault', 'defect', 'malfunction'],
                'secondary': ['symptom', 'sign', 'indicate', 'mean', 'check', 'test', 'verify',
                            'inspect', 'bad', 'faulty', 'worn', 'damaged', 'dead'],
                'context': ['noise', 'sound', 'smell', 'vibration', 'leak', 'smoke', 'warning']
            },
            
            'comparative': {
                'primary': ['vs', 'versus', 'compare', 'difference', 'better', 'worse', 'best',
                          'which', 'choose', 'prefer', 'recommend', 'alternative'],
                'secondary': ['advantage', 'disadvantage', 'pro', 'con', 'benefit', 'drawback',
                            'superior', 'inferior', 'optimal', 'ideal', 'suitable'],
                'context': ['option', 'choice', 'selection', 'decision', 'consideration']
            },
            
            'factual': {
                'primary': ['what', 'define', 'specification', 'how much', 'how many', 'when',
                          'where', 'location', 'type', 'kind', 'model', 'size'],
                'secondary': ['mean', 'capacity', 'volume', 'weight', 'dimension', 'interval',
                            'frequency', 'schedule', 'procedure', 'step', 'method'],
                'context': ['information', 'detail', 'data', 'fact', 'figure', 'number']
            }
        }
    
    def calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate pattern matching score"""
        score = 0.0
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            score += len(matches) * 2.0  # 2 points per pattern match
        
        return score
    
    def calculate_keyword_score(self, text: str, keywords: Dict[str, List[str]]) -> float:
        """Calculate keyword presence score"""
        score = 0.0
        text_lower = text.lower()
        words = text_lower.split()
        
        # Primary keywords (3 points each)
        for keyword in keywords.get('primary', []):
            if keyword in text_lower:
                score += 3.0
        
        # Secondary keywords (2 points each)
        for keyword in keywords.get('secondary', []):
            if keyword in text_lower:
                score += 2.0
        
        # Context keywords (1 point each)
        for keyword in keywords.get('context', []):
            if keyword in text_lower:
                score += 1.0
        
        return score
    
    def classify_question(self, question: Dict[str, Any]) -> Tuple[str, Dict[str, float]]:
        """
        Classify a single question
        Returns: (predicted_category, scores_dict)
        """
        
        # Combine question title and context for analysis
        text = question.get('question', '') + ' ' + question.get('title', '') + ' ' + question.get('context', '')[:200]
        
        scores = {}
        
        # Calculate scores for each category
        for category in ['causal', 'diagnostic', 'comparative', 'factual']:
            pattern_score = self.calculate_pattern_score(text, self.patterns[category])
            keyword_score = self.calculate_keyword_score(text, self.keywords[category])
            
            # Weighted combination
            total_score = (pattern_score * 0.7) + (keyword_score * 0.3)
            
            # Apply category-specific boosts
            if category == 'causal':
                # Boost for "why" at the beginning
                if re.match(r'^\s*why\b', text.lower()):
                    total_score += 5.0
                # Boost for causal chains
                if 'because' in text.lower() and ('cause' in text.lower() or 'lead' in text.lower()):
                    total_score += 3.0
            
            elif category == 'diagnostic':
                # Boost for personal car problems
                if re.search(r'\bmy\s+(car|vehicle|engine|brake)', text.lower()):
                    total_score += 3.0
                # Boost for symptom descriptions
                if re.search(r'\b(noise|sound|smell|vibration|leak)', text.lower()):
                    total_score += 2.0
            
            elif category == 'comparative':
                # Boost for explicit comparisons
                if re.search(r'\bvs\.?\b|\bversus\b', text.lower()):
                    total_score += 5.0
                # Boost for choice questions
                if re.search(r'\bwhich\s+(is|should|would)\b', text.lower()):
                    total_score += 3.0
            
            elif category == 'factual':
                # Boost for simple "what is" questions
                if re.match(r'^\s*what\s+(is|are)\b', text.lower()):
                    total_score += 3.0
                # Boost for specification questions
                if 'specification' in text.lower() or 'spec' in text.lower():
                    total_score += 2.0
            
            scores[category] = total_score
        
        # Determine category with highest score
        if max(scores.values()) == 0:
            # Default to factual if no patterns match
            predicted_category = 'factual'
        else:
            predicted_category = max(scores, key=scores.get)
        
        # Apply confidence threshold
        confidence = scores[predicted_category] / (sum(scores.values()) + 0.001)
        
        # If low confidence, use secondary classification
        if confidence < 0.3:
            predicted_category = self.secondary_classification(text, scores)
        
        return predicted_category, scores
    
    def secondary_classification(self, text: str, scores: Dict[str, float]) -> str:
        """Secondary classification for ambiguous cases"""
        
        text_lower = text.lower()
        
        # Question word analysis
        if text_lower.startswith('why'):
            return 'causal'
        elif text_lower.startswith('how to'):
            return 'factual'  # Procedural questions are factual
        elif re.match(r'^(is|are|can|could|should|would|will)', text_lower):
            if 'problem' in text_lower or 'issue' in text_lower:
                return 'diagnostic'
            else:
                return 'factual'
        elif 'vs' in text_lower or 'versus' in text_lower or 'better' in text_lower:
            return 'comparative'
        elif 'problem' in text_lower or 'wrong' in text_lower or 'fix' in text_lower:
            return 'diagnostic'
        
        # Default to factual
        return 'factual'
    
    def ensure_minimum_per_category(self, 
                                   classified_questions: List[Dict],
                                   min_per_category: int = 50) -> List[Dict]:
        """Ensure minimum questions per category by reclassifying if needed"""
        
        # Count current distribution
        category_counts = Counter(q['classification']['primary_category'] for q in classified_questions)
        
        print("\nInitial distribution:")
        for cat, count in category_counts.items():
            print(f"  {cat}: {count}")
        
        # Find categories that need more questions
        needed = {}
        for category in ['causal', 'diagnostic', 'comparative', 'factual']:
            current_count = category_counts.get(category, 0)
            if current_count < min_per_category:
                needed[category] = min_per_category - current_count
        
        if not needed:
            print("\nAll categories have minimum required questions!")
            return classified_questions
        
        print(f"\nCategories needing more questions: {needed}")
        
        # Reclassify borderline cases
        for category, need_count in needed.items():
            print(f"\nFinding {need_count} more {category} questions...")
            
            # Sort by secondary score for this category
            candidates = []
            for q in classified_questions:
                if q['classification']['primary_category'] != category:
                    score = q['classification']['scores'].get(category, 0)
                    if score > 0:
                        candidates.append((q, score))
            
            # Sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Reclassify top candidates
            reclassified = 0
            for candidate, score in candidates[:need_count]:
                old_category = candidate['classification']['primary_category']
                
                # Only reclassify if the category has excess
                if category_counts[old_category] > min_per_category:
                    candidate['classification']['primary_category'] = category
                    candidate['classification']['reclassified'] = True
                    candidate['classification']['original_category'] = old_category
                    category_counts[old_category] -= 1
                    category_counts[category] += 1
                    reclassified += 1
                    
                    if reclassified >= need_count:
                        break
            
            print(f"  Reclassified {reclassified} questions to {category}")
        
        return classified_questions


def classify_dataset():
    """Main function to classify the automotive dataset"""
    
    print("="*60)
    print("AUTOMOTIVE QUESTION CLASSIFICATION")
    print("="*60)
    
    # Initialize classifier
    classifier = AutomotiveQueryClassifier()
    
    # Load dataset
    dataset_file = 'data/processed/automotive_qa_final_20250816_230600.json'
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset not found at {dataset_file}")
        return
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Extract questions
    if isinstance(data, dict) and 'questions' in data:
        questions = data['questions']
    else:
        questions = data
    
    print(f"\nLoaded {len(questions)} questions for classification")
    
    # Classify each question
    classified_questions = []
    classification_stats = defaultdict(int)
    
    print("\nClassifying questions...")
    for i, question in enumerate(questions):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(questions)} questions...")
        
        # Classify
        category, scores = classifier.classify_question(question)
        
        # Add classification to question
        question['classification'] = {
            'primary_category': category,
            'scores': scores,
            'confidence': scores[category] / (sum(scores.values()) + 0.001)
        }
        
        classified_questions.append(question)
        classification_stats[category] += 1
    
    print(f"\nClassified all {len(questions)} questions")
    
    # Ensure minimum per category
    print("\nEnsuring minimum 50 questions per category...")
    classified_questions = classifier.ensure_minimum_per_category(classified_questions, min_per_category=50)
    
    # Final statistics
    final_stats = defaultdict(int)
    high_confidence_stats = defaultdict(int)
    
    for q in classified_questions:
        category = q['classification']['primary_category']
        final_stats[category] += 1
        
        if q['classification']['confidence'] > 0.5:
            high_confidence_stats[category] += 1
    
    # Save classified questions
    output_file = 'data/processed/classified_questions.json'
    with open(output_file, 'w') as f:
        json.dump(classified_questions, f, indent=2)
    
    print(f"\n✅ Saved classified questions to {output_file}")
    
    # Generate detailed statistics
    stats = {
        'total_questions': len(classified_questions),
        'classification_distribution': dict(final_stats),
        'high_confidence_distribution': dict(high_confidence_stats),
        'average_confidence': np.mean([q['classification']['confidence'] for q in classified_questions]),
        'classification_details': {}
    }
    
    # Detailed stats per category
    for category in ['causal', 'diagnostic', 'comparative', 'factual']:
        cat_questions = [q for q in classified_questions if q['classification']['primary_category'] == category]
        
        if cat_questions:
            confidences = [q['classification']['confidence'] for q in cat_questions]
            scores = [q['classification']['scores'][category] for q in cat_questions]
            
            # Sample questions
            sample_questions = []
            for q in sorted(cat_questions, key=lambda x: x['classification']['confidence'], reverse=True)[:5]:
                sample_questions.append({
                    'question': q.get('question', q.get('title', ''))[:100],
                    'confidence': round(q['classification']['confidence'], 3),
                    'score': round(q['classification']['scores'][category], 2)
                })
            
            stats['classification_details'][category] = {
                'count': len(cat_questions),
                'percentage': round(len(cat_questions) / len(classified_questions) * 100, 2),
                'avg_confidence': round(np.mean(confidences), 3),
                'min_confidence': round(min(confidences), 3),
                'max_confidence': round(max(confidences), 3),
                'avg_score': round(np.mean(scores), 2),
                'sample_questions': sample_questions
            }
    
    # Save statistics
    stats_file = 'data/processed/classification_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved classification statistics to {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"\nTotal Questions: {stats['total_questions']}")
    print(f"Average Confidence: {stats['average_confidence']:.3f}")
    
    print("\nFinal Distribution:")
    for category in ['causal', 'diagnostic', 'comparative', 'factual']:
        details = stats['classification_details'].get(category, {})
        print(f"\n{category.upper()}:")
        print(f"  Count: {details.get('count', 0)} ({details.get('percentage', 0)}%)")
        print(f"  Avg Confidence: {details.get('avg_confidence', 0)}")
        print(f"  Confidence Range: {details.get('min_confidence', 0)} - {details.get('max_confidence', 0)}")
        
        if details.get('sample_questions'):
            print(f"  Top Example: \"{details['sample_questions'][0]['question']}...\"")
    
    return classified_questions, stats


if __name__ == "__main__":
    classified_questions, stats = classify_dataset()