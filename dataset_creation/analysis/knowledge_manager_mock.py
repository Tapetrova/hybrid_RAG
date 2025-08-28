#!/usr/bin/env python3
"""
Minimal Knowledge Manager mock server for evaluation
Provides vector and graph search endpoints with sample automotive data
"""

from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Sample automotive knowledge base
AUTOMOTIVE_KNOWLEDGE = [
    {
        "id": "auto_fact_1",
        "text": "Modern vehicles use electronic fuel injection systems which provide precise fuel metering and better efficiency compared to carburetors. The ECU controls the injectors based on sensor inputs.",
        "score": 0.95,
        "type": "factual"
    },
    {
        "id": "auto_fact_2", 
        "text": "Brake pads should be replaced when they reach 3-4mm thickness. Rotors may need resurfacing if warped or scored, but don't always require replacement with new pads.",
        "score": 0.92,
        "type": "factual"
    },
    {
        "id": "auto_causal_1",
        "text": "Engine knocking occurs due to premature combustion of the air-fuel mixture. This can be caused by low octane fuel, carbon deposits, or incorrect ignition timing.",
        "score": 0.89,
        "type": "causal"
    },
    {
        "id": "auto_diag_1",
        "text": "A check engine light can indicate issues ranging from a loose gas cap to serious engine problems. Use an OBD-II scanner to read diagnostic trouble codes for specific issues.",
        "score": 0.87,
        "type": "diagnostic"
    },
    {
        "id": "auto_fact_3",
        "text": "Synthetic oil typically lasts 7,500-10,000 miles between changes, while conventional oil should be changed every 3,000-5,000 miles depending on driving conditions.",
        "score": 0.90,
        "type": "factual"
    },
    {
        "id": "auto_causal_2",
        "text": "Neutral is placed between 1st and 2nd gear on motorcycles for safety - allowing quick shifts to first when stopping and preventing accidental neutral selection at high speeds.",
        "score": 0.88,
        "type": "causal"
    }
]

@app.route('/vector/search', methods=['POST'])
def vector_search():
    """Simulate vector similarity search"""
    data = request.json
    query = data.get('query', '')
    top_k = min(data.get('top_k', 4), len(AUTOMOTIVE_KNOWLEDGE))
    
    # Return random subset simulating relevance
    results = random.sample(AUTOMOTIVE_KNOWLEDGE, min(top_k, len(AUTOMOTIVE_KNOWLEDGE)))
    
    # Add some noise to scores
    for r in results:
        r['score'] = r['score'] * random.uniform(0.85, 1.0)
    
    return jsonify({
        "results": results,
        "query": query,
        "status": "success"
    })

@app.route('/graph/search', methods=['POST'])
def graph_search():
    """Simulate graph-based search with relationships"""
    data = request.json
    query = data.get('query', '')
    top_k = min(data.get('top_k', 4), len(AUTOMOTIVE_KNOWLEDGE))
    
    # For graph search, prefer causal and diagnostic content
    weighted_knowledge = []
    for item in AUTOMOTIVE_KNOWLEDGE:
        weight = 1.2 if item['type'] in ['causal', 'diagnostic'] else 1.0
        weighted_knowledge.append((item, weight))
    
    # Sort by weight and select top_k
    weighted_knowledge.sort(key=lambda x: x[1] * x[0]['score'], reverse=True)
    results = [item[0].copy() for item in weighted_knowledge[:top_k]]
    
    # Adjust scores
    for r in results:
        r['score'] = r['score'] * random.uniform(0.80, 0.95)
    
    return jsonify({
        "results": results,
        "query": query,
        "status": "success"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("Starting Knowledge Manager Mock Server on port 8098...")
    print("Endpoints available:")
    print("  - POST /vector/search")
    print("  - POST /graph/search")
    print("  - GET /health")
    app.run(host='0.0.0.0', port=8098, debug=False)