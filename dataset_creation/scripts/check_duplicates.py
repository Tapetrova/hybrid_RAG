"""
Check for duplicates with different thresholds to understand the data
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def analyze_similarity_distribution():
    """Analyze similarity distribution in the dataset"""
    
    print("Loading data...")
    with open('data/processed/classified_questions.json', 'r') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    
    # Load model
    print("Loading model...")
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Prepare texts (sample for faster analysis)
    print("Preparing texts...")
    texts = []
    sample_size = min(500, len(questions))  # Analyze first 500 for speed
    
    for q in questions[:sample_size]:
        title = q.get('title', q.get('question', ''))
        body = q.get('body', q.get('context', ''))[:300]
        combined = f"{title} {body}"
        texts.append(combined)
    
    # Compute embeddings
    print(f"Computing embeddings for {len(texts)} questions...")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    
    # Extract upper triangle (excluding diagonal)
    upper_triangle = []
    for i in range(len(sim_matrix)):
        for j in range(i+1, len(sim_matrix)):
            upper_triangle.append(sim_matrix[i][j])
    
    upper_triangle = np.array(upper_triangle)
    
    # Statistics
    print("\n" + "="*60)
    print("SIMILARITY DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\nTotal comparisons: {len(upper_triangle)}")
    print(f"Mean similarity: {np.mean(upper_triangle):.4f}")
    print(f"Median similarity: {np.median(upper_triangle):.4f}")
    print(f"Max similarity: {np.max(upper_triangle):.4f}")
    print(f"Min similarity: {np.min(upper_triangle):.4f}")
    print(f"Std deviation: {np.std(upper_triangle):.4f}")
    
    # Check different thresholds
    thresholds = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    print("\nDuplicates at different thresholds:")
    
    for threshold in thresholds:
        count = np.sum(upper_triangle > threshold)
        percentage = count / len(upper_triangle) * 100
        print(f"  > {threshold}: {count} pairs ({percentage:.2f}%)")
    
    # Find most similar pairs
    print("\nTop 10 most similar pairs:")
    
    # Get indices of top similarities
    top_indices = np.argpartition(upper_triangle, -10)[-10:]
    top_similarities = upper_triangle[top_indices]
    
    # Sort them
    sorted_indices = np.argsort(top_similarities)[::-1]
    
    # Convert back to matrix indices
    pair_count = 0
    for idx in sorted_indices:
        orig_idx = top_indices[idx]
        
        # Convert linear index back to i,j
        current_idx = 0
        found = False
        for i in range(len(sim_matrix)):
            for j in range(i+1, len(sim_matrix)):
                if current_idx == orig_idx:
                    similarity = sim_matrix[i][j]
                    print(f"\n  Pair {pair_count + 1}: Similarity = {similarity:.4f}")
                    print(f"    Q1 [{i}]: {texts[i][:100]}...")
                    print(f"    Q2 [{j}]: {texts[j][:100]}...")
                    found = True
                    pair_count += 1
                    break
                current_idx += 1
            if found:
                break
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(upper_triangle, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0.9, color='red', linestyle='--', label='Threshold = 0.9')
    plt.axvline(x=np.mean(upper_triangle), color='green', linestyle='--', label=f'Mean = {np.mean(upper_triangle):.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Pairs')
    plt.title(f'Similarity Distribution (n={sample_size} questions)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('paper/figures/similarity_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved histogram to paper/figures/similarity_distribution.png")
    
    return upper_triangle, sim_matrix, texts, questions[:sample_size]

def find_and_remove_duplicates(threshold=0.85):
    """Find and remove duplicates with a lower threshold"""
    
    print(f"\nFinding duplicates with threshold {threshold}...")
    
    with open('data/processed/classified_questions.json', 'r') as f:
        questions = json.load(f)
    
    encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Compute all embeddings
    texts = []
    for q in questions:
        title = q.get('title', q.get('question', ''))
        body = q.get('body', q.get('context', ''))[:300]
        texts.append(f"{title} {body}")
    
    print(f"Computing embeddings for {len(texts)} questions...")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    
    # Find duplicates
    sim_matrix = cosine_similarity(embeddings)
    
    duplicates_found = []
    processed = set()
    
    for i in range(len(questions)):
        if i in processed:
            continue
            
        similar = []
        for j in range(i+1, len(questions)):
            if sim_matrix[i][j] > threshold:
                similar.append((j, sim_matrix[i][j]))
                processed.add(j)
        
        if similar:
            duplicates_found.append({
                'main_idx': i,
                'main_question': questions[i].get('title', '')[:100],
                'duplicates': similar
            })
    
    print(f"\nFound {len(duplicates_found)} groups of duplicates")
    print(f"Total duplicates to remove: {sum(len(d['duplicates']) for d in duplicates_found)}")
    
    if duplicates_found:
        print("\nFirst 5 duplicate groups:")
        for group in duplicates_found[:5]:
            print(f"\n  Main [{group['main_idx']}]: {group['main_question']}")
            for dup_idx, sim in group['duplicates'][:2]:
                print(f"    Dup [{dup_idx}] (sim={sim:.3f}): {questions[dup_idx].get('title', '')[:80]}")
    
    return duplicates_found

if __name__ == "__main__":
    # Analyze similarity distribution
    similarities, matrix, texts, sample_questions = analyze_similarity_distribution()
    
    # Try finding duplicates with lower threshold
    duplicates_085 = find_and_remove_duplicates(0.85)
    duplicates_080 = find_and_remove_duplicates(0.80)