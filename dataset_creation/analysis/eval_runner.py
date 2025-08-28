#!/usr/bin/env python3
"""
Evaluation runner for automotive Q&A dataset
Direct OpenAI API integration with client-side hybrid fusion
"""

import argparse
import json
import os
import hashlib
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml
from tqdm import tqdm
from openai import OpenAI

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use environment variables


class EvaluationRunner:
    def __init__(self, config_path: str):
        """Initialize evaluation runner with config"""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Check for OpenAI API key
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Please set: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # Set random seed
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load dataset
        dataset_path = self.config_path.parent / self.config['dataset_path']
        if not dataset_path.exists():
            print(f"Error: Dataset not found: {dataset_path}")
            sys.exit(1)
            
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        # Validate dataset structure
        if not isinstance(data, dict) or 'questions' not in data:
            print("Error: Dataset must have top-level structure: {'questions': [...]}")
            print(f"Found keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            sys.exit(1)
            
        self.questions = data['questions']
        
        # Validate each question has required fields
        required_fields = ['id', 'question', 'context', 'answer', 'category']
        missing_fields_count = 0
        for i, q in enumerate(self.questions[:10]):  # Check first 10 for quick validation
            missing = [field for field in required_fields if field not in q]
            if missing:
                print(f"Warning: Question {i} missing fields: {missing}")
                missing_fields_count += 1
        
        if missing_fields_count > 0:
            print(f"Warning: {missing_fields_count} questions missing required fields")
            print("Required fields: id, question, context, answer, category")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Calculate category distribution
        category_counts = {}
        for q in self.questions:
            cat = q.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Print dataset statistics
        print("\n" + "="*60)
        print("DATASET LOADED SUCCESSFULLY")
        print("="*60)
        print(f"Total questions: {len(self.questions)}")
        print(f"Expected: 706 questions")
        if len(self.questions) != 706:
            print(f"WARNING: Expected 706 questions, found {len(self.questions)}")
        
        print("\nCategory distribution:")
        for cat in sorted(category_counts.keys()):
            count = category_counts[cat]
            percentage = (count / len(self.questions)) * 100
            print(f"  {cat:15s}: {count:4d} ({percentage:5.1f}%)")
        
        print("="*60 + "\n")
        
        # Initialize results storage
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.all_predictions = []
        
    def truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate text to max chars, preserving word boundaries"""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # Try to break at last space
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # If space is reasonably close to end
            truncated = truncated[:last_space]
        return truncated + "..."
    
    def retrieve_vector(self, query: str) -> List[Dict]:
        """Retrieve from vector store via Knowledge Manager"""
        try:
            url = self.config['services']['knowledge_manager_vector_url']
            payload = {"query": query, "top_k": self.config['retrieval']['top_k']}
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                hits = data.get('hits', [])
                
                # Format results with exact keys required
                results = []
                max_chars = self.config['retrieval']['max_chars_per_passage']
                for hit in hits:
                    result = {
                        "source_id": hit.get("source_id", hit.get("id", "unknown")),
                        "text": hit.get("text", "")[:max_chars],
                        "score": float(hit.get("score", 0.0)),
                        "type": "vector"
                    }
                    results.append(result)
                return results
            else:
                print(f"Vector retrieval failed: {response.status_code}")
                return []
        except (requests.ConnectionError, requests.Timeout) as e:
            # Service not available - return empty
            return []
        except Exception as e:
            print(f"Vector retrieval error: {e}")
            return []
    
    def retrieve_graph(self, query: str, community_detection: bool = True) -> List[Dict]:
        """Retrieve from graph store via Knowledge Manager"""
        try:
            url = self.config['services']['knowledge_manager_graph_url']
            payload = {
                "query": query, 
                "top_k": self.config['retrieval']['top_k'],
                "community_detection": community_detection
            }
            
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                hits = data.get('hits', [])
                
                # Format results with exact keys required
                results = []
                max_chars = self.config['retrieval']['max_chars_per_passage']
                for hit in hits:
                    result = {
                        "source_id": hit.get("source_id", hit.get("id", "unknown")),
                        "text": hit.get("text", "")[:max_chars],
                        "score": float(hit.get("score", 0.0)),
                        "type": "graph"
                    }
                    results.append(result)
                return results
            else:
                print(f"Graph retrieval failed: {response.status_code}")
                return []
        except (requests.ConnectionError, requests.Timeout) as e:
            # Service not available - return empty
            return []
        except Exception as e:
            print(f"Graph retrieval error: {e}")
            return []
    
    def get_hybrid_weights(self, category: str) -> Dict[str, float]:
        """Get hybrid fusion weights for a given category"""
        w = self.config["hybrid_weights"]
        return w.get(category, w["default"])
    
    def minmax_norm(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max normalization"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are equal, return 0.5 for each
        if max_score == min_score:
            return [0.5] * len(scores)
        
        # Normalize to [0, 1]
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def fuse_retrievals(self, vector_hits: List[Dict], graph_hits: List[Dict], 
                       weights: Dict[str, float]) -> List[Dict]:
        """
        Fuse vector and graph search results using weighted combination
        
        Args:
            vector_hits: List of vector search results
            graph_hits: List of graph search results  
            weights: Dict with 'vector' and 'graph' weights
            
        Returns:
            List of fused results sorted by score, each with keys:
            source_id, text, score (fused), type='hybrid'
        """
        top_k = self.config['retrieval']['top_k']
        
        # Extract and normalize vector scores
        vector_scores = [h["score"] for h in vector_hits] if vector_hits else []
        vector_norm = self.minmax_norm(vector_scores) if vector_scores else []
        
        # Extract and normalize graph scores
        graph_scores = [h["score"] for h in graph_hits] if graph_hits else []
        graph_norm = self.minmax_norm(graph_scores) if graph_scores else []
        
        # Build fusion map: source_id -> {text, vector_score, graph_score}
        fusion_map = {}
        
        # Add vector hits
        for i, hit in enumerate(vector_hits):
            source_id = hit["source_id"]
            fusion_map[source_id] = {
                "text": hit["text"],
                "vector_score": vector_norm[i],
                "graph_score": 0.0
            }
        
        # Add/update with graph hits
        for i, hit in enumerate(graph_hits):
            source_id = hit["source_id"]
            if source_id in fusion_map:
                # Same source_id in both - update graph score
                fusion_map[source_id]["graph_score"] = graph_norm[i]
            else:
                # Only in graph
                fusion_map[source_id] = {
                    "text": hit["text"],
                    "vector_score": 0.0,
                    "graph_score": graph_norm[i]
                }
        
        # Calculate fused scores
        fused_results = []
        w_vector = weights.get("vector", 0.5)
        w_graph = weights.get("graph", 0.5)
        
        for source_id, data in fusion_map.items():
            fused_score = (w_vector * data["vector_score"] + 
                          w_graph * data["graph_score"])
            
            fused_results.append({
                "source_id": source_id,
                "text": data["text"],
                "score": fused_score,
                "type": "hybrid"
            })
        
        # Sort by fused score descending
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_k results
        return fused_results[:top_k]
    
    def build_context(self, retrieved: List[Dict]) -> str:
        """Build context string from retrieved passages"""
        if not retrieved:
            return ""
        
        passages = []
        total_chars = 0
        max_total = self.config['retrieval']['max_context_chars_total']
        
        for i, hit in enumerate(retrieved):
            text = hit.get('text', '')
            if total_chars + len(text) > max_total:
                # Truncate this passage to fit
                remaining = max_total - total_chars
                if remaining > 100:  # Only include if meaningful amount left
                    text = self.truncate_text(text, remaining)
                    passages.append(f"[{i+1}] {text}")
                break
            passages.append(f"[{i+1}] {text}")
            total_chars += len(text)
        
        return "\n\n".join(passages)
    
    def generate_answer(self, question: str, context: str = "") -> str:
        """Generate answer using OpenAI with Step 6.1 specification"""
        # System prompt as specified
        system_prompt = ("You are an automotive assistant. Use the provided context if relevant. "
                        "If the context is insufficient, say what's missing. "
                        "Be concise and technically accurate.")
        
        # Build user content with exact format
        if context:
            user_content = f"""QUESTION:
{question}

CONTEXT (may be partial):
---BEGIN---
{context}
---END---"""
        else:
            # For base_llm mode - no context
            user_content = f"""QUESTION:
{question}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['claims_llm']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.config['claims_llm']['temperature'],
                max_tokens=self.config['claims_llm']['max_tokens']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Failed to generate answer."
    
    def extract_claims(self, answer_text: str) -> List[Dict]:
        """Extract claims from answer using Step 6.2 specification"""
        # System prompt as specified
        system_prompt = """You are an information extraction assistant.
Task: Extract atomic, verifiable claims from the assistant's answer.
- Split into minimal factual or causal statements (<= 25 words each).
- No hedging. No style. Claims must be checkable.
- Label each claim: causal | diagnostic | factual | other.
Return only JSON:
{"claims":[{"text":"...", "type":"causal|diagnostic|factual|other"}]}"""
        
        # Try extraction with retry
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config['claims_llm']['model'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": answer_text}
                    ],
                    temperature=self.config['claims_llm']['temperature'],
                    max_tokens=self.config['claims_llm']['max_tokens'],
                    response_format={"type": "json_object"}
                )
                
                data = json.loads(response.choices[0].message.content)
                claims = data.get('claims', [])
                
                # Validate claims
                valid_claims = []
                for claim in claims:
                    if isinstance(claim, dict) and "text" in claim and "type" in claim:
                        if claim["type"] not in ["causal", "diagnostic", "factual", "other"]:
                            claim["type"] = "other"
                        valid_claims.append(claim)
                
                return valid_claims
                
            except (json.JSONDecodeError, KeyError) as e:
                if attempt == max_retries - 1:
                    print(f"Failed to extract claims after {max_retries} attempts: {e}")
                    return []
                time.sleep(0.5)  # Brief pause before retry
        
        return []
    
    def judge_claim(self, claim: str, gold_answer: str, retrieved_texts: List[str]) -> Dict:
        """Judge claim using Step 6.3 specification"""
        # Build support corpus S with size limit
        corpus_parts = []
        if gold_answer:
            corpus_parts.append(gold_answer)
        corpus_parts.extend(retrieved_texts)
        
        # Cap by max_context_chars_total
        max_chars = self.config['retrieval']['max_context_chars_total']
        support_corpus = ""
        char_count = 0
        
        for part in corpus_parts:
            if char_count + len(part) <= max_chars:
                support_corpus += part + "\n\n"
                char_count += len(part) + 2
            else:
                remaining = max_chars - char_count
                if remaining > 100:
                    support_corpus += part[:remaining] + "..."
                break
        
        support_corpus = support_corpus.strip()
        
        # System prompt as specified
        system_prompt = """You are a strict fact-checking judge.
Given:
- CLAIM: <claim>
- SUPPORT CORPUS S: (gold answer + retrieved passages provided to the model)

Decide one label:
- "supported": S contains explicit or strongly entailed evidence;
- "contradicted": S contradicts the claim;
- "unverifiable": S lacks both evidence and contradiction.

Return only JSON:
{"label":"supported|contradicted|unverifiable","rationale":"...","evidence_spans":["...","..."]}"""
        
        user_content = f"""CLAIM: {claim}

SUPPORT CORPUS S:
{support_corpus}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config['judge_llm']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.config['judge_llm']['temperature'],
                max_tokens=self.config['judge_llm']['max_tokens'],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            # Validate response
            label = data.get('label', 'unverifiable')
            if label not in ['supported', 'contradicted', 'unverifiable']:
                label = 'unverifiable'
            
            rationale = data.get('rationale', '')
            if len(rationale) > 280:
                rationale = rationale[:277] + '...'
            
            evidence_spans = data.get('evidence_spans', [])
            if not isinstance(evidence_spans, list):
                evidence_spans = []
            
            return {
                'claim': claim,
                'label': label,
                'rationale': rationale,
                'evidence_spans': evidence_spans
            }
        except Exception as e:
            print(f"Error judging claim: {e}")
            return {
                'claim': claim,
                'label': 'unverifiable',
                'rationale': 'Judgment failed',
                'evidence_spans': []
            }
    
    def calculate_metrics(self, claims: List[Dict], judgments: List[Dict]) -> Dict:
        """Calculate hallucination metrics"""
        total = len(claims)
        if total == 0:
            return {
                'claims_total': 0,
                'contradicted': 0,
                'unverifiable': 0,
                'HR': 0.0,
                'HR_contra': 0.0,
                'HR_unver': 0.0,
                'causal_claims': 0,
                'CHR': 0.0,
                'GAC': None
            }
        
        # Count labels
        contradicted = sum(1 for j in judgments if j['label'] == 'contradicted')
        unverifiable = sum(1 for j in judgments if j['label'] == 'unverifiable')
        
        # Causal claims
        causal_claims = [c for c in claims if c.get('type') == 'causal']
        causal_judgments = [j for c, j in zip(claims, judgments) if c.get('type') == 'causal']
        
        causal_total = len(causal_claims)
        causal_bad = 0
        if causal_total > 0:
            causal_bad = sum(1 for j in causal_judgments if j['label'] in ['contradicted', 'unverifiable'])
        
        return {
            'claims_total': total,
            'contradicted': contradicted,
            'unverifiable': unverifiable,
            'HR': (contradicted + unverifiable) / total,
            'HR_contra': contradicted / total,
            'HR_unver': unverifiable / total,
            'causal_claims': causal_total,
            'CHR': causal_bad / causal_total if causal_total > 0 else 0.0,
            'GAC': None  # Disabled
        }
    
    def evaluate_question(self, question: Dict, mode: str) -> Dict:
        """Evaluate a single question with specified mode - Step 7 pipeline"""
        q_id = question['id']
        q_text = question['question']
        q_context = question.get('context', '')
        full_question = f"{q_text}\n{q_context}" if q_context else q_text
        category = question.get('category', 'factual')
        gold_answer = question.get('answer', '')
        
        # Initialize result with all required fields
        result = {
            'id': q_id,
            'category': category,
            'mode': mode,
            'question': full_question,  # Full question with context
            'gold_answer': gold_answer,
            'answer_model': self.config['claims_llm']['model'],
            'answer_text': '',
            'retrieved': [],
            'claims': [],
            'claim_judgments': [],
            'gac': [],  # Empty - GAC disabled
            'metrics_sample': {},
            'routing': {},
            'ts': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Step 7: Per-mode pipeline
        context_blocks = []  # List of text blocks for context
        
        if mode == 'base_llm':
            # No retrieval, empty context
            context_blocks = []
            result['routing'] = {
                'used_vector': False,
                'used_graph': False,
                'weights': {'graph': 0.0, 'vector': 0.0}
            }
        
        elif mode == 'vector_rag':
            # Vector search only
            hits_v = self.retrieve_vector(full_question)
            result['retrieved'] = hits_v
            context_blocks = [h["text"] for h in hits_v]
            result['routing'] = {
                'used_vector': True,
                'used_graph': False,
                'weights': {'graph': 0.0, 'vector': 1.0}
            }
        
        elif mode == 'graph_rag':
            # Graph search with community detection
            hits_g = self.retrieve_graph(full_question, community_detection=True)
            result['retrieved'] = hits_g
            context_blocks = [h["text"] for h in hits_g]
            result['routing'] = {
                'used_vector': False,
                'used_graph': True,
                'weights': {'graph': 1.0, 'vector': 0.0},
                'graph_settings': {'community_detection': True}
            }
        
        elif mode == 'hybrid_ahs':
            # Hybrid fusion
            hits_v = self.retrieve_vector(full_question)
            hits_g = self.retrieve_graph(full_question, community_detection=True)
            w = self.get_hybrid_weights(category)
            
            # Fuse with top_k from config
            top_k = self.config['retrieval']['top_k']
            hits = self.fuse_retrievals(hits_v, hits_g, w)
            result['retrieved'] = hits
            context_blocks = [h["text"] for h in hits]
            result['routing'] = {
                'used_vector': True,
                'used_graph': True,
                'weights': w
            }
        
        # Generate answer with context blocks
        context = self.build_context(result['retrieved']) if result['retrieved'] else ""
        result['answer_text'] = self.generate_answer(full_question, context)
        
        # Extract claims
        result['claims'] = self.extract_claims(result['answer_text'])
        
        # Judge claims against gold + retrieved texts
        result['claim_judgments'] = [
            self.judge_claim(claim['text'], gold_answer, context_blocks) 
            for claim in result['claims']
        ]
        
        # Calculate per-sample metrics
        result['metrics_sample'] = self.calculate_metrics(
            result['claims'], 
            result['claim_judgments']
        )
        
        return result
    
    def run_evaluation(self, modes: List[str], limit: Optional[int] = None):
        """Run evaluation for specified modes and questions"""
        # Apply limit if specified
        questions_to_eval = self.questions[:limit] if limit else self.questions
        
        print(f"\nEvaluating {len(questions_to_eval)} questions across {len(modes)} modes")
        print(f"Modes: {', '.join(modes)}")
        
        # Output files
        predictions_file = self.output_dir / f"predictions_{self.timestamp}.jsonl"
        
        # Progress bar
        total_evals = len(questions_to_eval) * len(modes)
        pbar = tqdm(total=total_evals, desc="Evaluating")
        
        # Run evaluations
        with open(predictions_file, 'w') as f:
            for question in questions_to_eval:
                for mode in modes:
                    try:
                        result = self.evaluate_question(question, mode)
                        f.write(json.dumps(result) + '\n')
                        f.flush()
                        self.all_predictions.append(result)
                        pbar.update(1)
                        
                        # Rate limiting
                        time.sleep(0.1)  # Be nice to APIs
                        
                    except Exception as e:
                        print(f"\nError on {question['id']} / {mode}: {e}")
                        pbar.update(1)
        
        pbar.close()
        
        # Generate summaries
        self.generate_summaries()
        
        print(f"\nEvaluation complete!")
        print(f"Predictions: {predictions_file}")
        
        # Run sanity checks
        self.run_sanity_checks()
        
    def generate_summaries(self):
        """Generate CSV and MD summaries from predictions"""
        if not self.all_predictions:
            print("No predictions to summarize")
            return
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.all_predictions)
        
        # CSV metrics
        metrics_csv = self.output_dir / f"metrics_{self.timestamp}.csv"
        
        rows = []
        for mode in df['mode'].unique():
            mode_df = df[df['mode'] == mode]
            
            # Overall metrics
            all_metrics = [m for m in mode_df['metrics_sample'] if m]
            if all_metrics:
                overall = self._aggregate_metrics(all_metrics)
                rows.append({
                    'mode': mode,
                    'split': 'overall',
                    'HR': f"{overall['HR']:.3f}",
                    'HR_contra': f"{overall['HR_contra']:.3f}",
                    'HR_unver': f"{overall['HR_unver']:.3f}",
                    'CHR': f"{overall['CHR']:.3f}",
                    'GAC': '',
                    'n_samples': len(mode_df)
                })
            
            # Per category
            for category in ['causal', 'diagnostic', 'factual', 'comparative']:
                cat_df = mode_df[mode_df['category'] == category]
                if len(cat_df) > 0:
                    cat_metrics = [m for m in cat_df['metrics_sample'] if m]
                    if cat_metrics:
                        cat_agg = self._aggregate_metrics(cat_metrics)
                        rows.append({
                            'mode': mode,
                            'split': f"{category}_q",
                            'HR': f"{cat_agg['HR']:.3f}",
                            'HR_contra': f"{cat_agg['HR_contra']:.3f}",
                            'HR_unver': f"{cat_agg['HR_unver']:.3f}",
                            'CHR': f"{cat_agg['CHR']:.3f}",
                            'GAC': '',
                            'n_samples': len(cat_df)
                        })
        
        # Save CSV
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)
        
        # MD summary
        metrics_md = self.output_dir / f"metrics_{self.timestamp}.md"
        with open(metrics_md, 'w') as f:
            f.write(f"# Evaluation Results - {self.timestamp}\n\n")
            f.write("## Configuration\n")
            f.write(f"- Dataset: {self.config['dataset_path']}\n")
            f.write(f"- Questions evaluated: {len(df['id'].unique())}\n")
            f.write(f"- Modes: {', '.join(df['mode'].unique())}\n")
            f.write(f"- Model: {self.config['claims_llm']['model']}\n\n")
            
            f.write("## Overall Metrics\n\n")
            f.write("| Mode | HR | HR_contra | HR_unver | CHR | Samples |\n")
            f.write("|------|-----|-----------|----------|-----|--------|\n")
            
            for row in rows:
                if row['split'] == 'overall':
                    f.write(f"| {row['mode']} | {row['HR']} | {row['HR_contra']} | ")
                    f.write(f"{row['HR_unver']} | {row['CHR']} | {row['n_samples']} |\n")
            
            f.write("\n## Notes\n")
            f.write("- HR: Hallucination Rate (contradicted + unverifiable)\n")
            f.write("- CHR: Causal Hallucination Rate\n")
            f.write("- GAC: Graph Alignment Consistency (disabled)\n")
        
        print(f"Metrics CSV: {metrics_csv}")
        print(f"Summary: {metrics_md}")
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across samples"""
        if not metrics_list:
            return {'HR': 0, 'HR_contra': 0, 'HR_unver': 0, 'CHR': 0}
        
        total_claims = sum(m.get('claims_total', 0) for m in metrics_list)
        if total_claims == 0:
            return {'HR': 0, 'HR_contra': 0, 'HR_unver': 0, 'CHR': 0}
        
        total_contra = sum(m.get('contradicted', 0) for m in metrics_list)
        total_unver = sum(m.get('unverifiable', 0) for m in metrics_list)
        
        # CHR - average across samples that have causal claims
        chr_values = [m.get('CHR', 0) for m in metrics_list if m.get('causal_claims', 0) > 0]
        
        return {
            'HR': (total_contra + total_unver) / total_claims,
            'HR_contra': total_contra / total_claims,
            'HR_unver': total_unver / total_claims,
            'CHR': np.mean(chr_values) if chr_values else 0.0
        }
    
    def run_sanity_checks(self):
        """Run sanity checks on the evaluation results"""
        metrics_csv = self.output_dir / f"metrics_{self.timestamp}.csv"
        
        if not metrics_csv.exists():
            print("\n⚠️ Metrics CSV not found, skipping sanity checks")
            return
        
        print("\n" + "="*60)
        print("RUNNING SANITY CHECKS")
        print("="*60)
        
        # Import sanity check module
        from sanity_checks import run_sanity_checks
        
        # Run checks
        passed = run_sanity_checks(metrics_csv)
        
        if not passed:
            print("\n⚠️ IMPORTANT: Sanity checks detected potential issues!")
            print("   Please review before proceeding with full evaluation.")
            print("   Consider checking:")
            print("   - Are Knowledge Manager services running?")
            print("   - Is retrieval returning results?")
            print("   - Are the model configurations correct?")


def main():
    parser = argparse.ArgumentParser(description='Evaluate automotive Q&A dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--run', type=str, default='all', help='Run mode (all or specific)')
    parser.add_argument('--modes', type=str, help='Comma-separated list of modes to run')
    parser.add_argument('--limit', type=int, help='Limit number of questions to evaluate')
    parser.add_argument('--ablations', type=str, default='off', help='Ablations (on/off)')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(args.config)
    
    # Determine modes to run
    if args.modes:
        modes = [m.strip() for m in args.modes.split(',')]
    elif args.run == 'all':
        modes = runner.config['modes']
    else:
        modes = [args.run]
    
    # Validate modes
    valid_modes = runner.config['modes']
    for mode in modes:
        if mode not in valid_modes:
            print(f"Error: Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}")
            sys.exit(1)
    
    # Run evaluation
    runner.run_evaluation(modes, args.limit)
    
    # Append to README
    readme_path = Path(__file__).parent.parent / 'README.md'
    with open(readme_path, 'a') as f:
        f.write(f"\n\n### Evaluation Run - {runner.timestamp}\n")
        f.write(f"- Config: {args.config}\n")
        f.write(f"- Modes: {', '.join(modes)}\n")
        f.write(f"- Questions: {args.limit if args.limit else len(runner.questions)}\n")
        f.write(f"- Outputs: {runner.output_dir}\n")
        f.write(f"- Completed: {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()