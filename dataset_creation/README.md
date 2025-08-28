# Automotive Q&A Dataset for CKG-RAG Research

This directory contains the automotive Q&A dataset used for the CKG-RAG (Causal Knowledge Graph - Retrieval-Augmented Generation) research project. The dataset consists of high-quality automotive questions and answers collected from Stack Exchange.

## 📊 Dataset Overview

**IMPORTANT**: This dataset contains **ONLY questions with verified answers**. Questions without answers have been removed to ensure data quality for the research.

- **Total Questions with Answers**: 706
- **Source**: Stack Exchange Mechanics (mechanics.stackexchange.com)
- **Collection Method**: Stack Exchange API
- **Language**: English
- **Domain**: Automotive maintenance, repair, and theory
- **Average Stack Exchange Score**: 11.6 (community-validated quality)
  - Score = Upvotes - Downvotes from expert community
  - Typical Stack Exchange average: 3-5, our dataset: **11.6** (exceptionally high)
  - Score ranges: 0-2 (poor), 3-5 (average), 6-10 (good), 11-20 (high quality), 20+ (excellent)
  - Our dataset: 46% high quality (≥10), 11% excellent (≥20), max score: 106

## 🌐 Data Source

The data was collected from **Stack Exchange Mechanics** (mechanics.stackexchange.com), a community-driven Q&A platform for automotive professionals and enthusiasts. Stack Exchange provides:
- Expert-verified answers
- Community voting system ensuring quality
- Detailed technical discussions
- Real-world automotive problems and solutions

### Collection Process
1. Used Stack Exchange API to query automotive-related tags
2. Collected questions from 9 major automotive categories
3. Filtered for questions with accepted or high-quality answers
4. Removed all questions without answers
5. Processed and structured the data for research use

## 📈 Dataset Statistics

### Category Distribution
| Category | Count | Percentage | Avg Score |
|----------|-------|------------|-----------|
| **Factual** | 391 | 55.4% | 11.3 |
| **Causal** | 120 | 17.0% | 14.6 |
| **Diagnostic** | 118 | 16.7% | 10.5 |
| **Comparative** | 77 | 10.9% | 10.3 |

### Automotive Topics Coverage
| Topic | Questions | Percentage |
|-------|-----------|------------|
| **Engine** | 143 | 20.3% |
| **Brakes** | 112 | 15.9% |
| **Electrical** | 111 | 15.7% |
| **Exhaust** | 65 | 9.2% |
| **Transmission** | 63 | 8.9% |
| **Oil** | 56 | 7.9% |
| **Air-conditioning** | 54 | 7.6% |
| **Battery** | 52 | 7.4% |
| **Coolant** | 50 | 7.1% |

### Quality Metrics
- **High Quality** (score ≥ 10): 326 questions (46.2%)
- **Very High Quality** (score ≥ 20): 78 questions (11.0%)
- **Exceptional** (score ≥ 50): 5 questions (0.7%)
- **Maximum Score**: 106
- **Median Score**: 9.0

### Text Statistics
| Component | Average | Median | Max | Min |
|-----------|---------|--------|-----|-----|
| **Question Title** | 9.8 words | 9.0 words | 28 words | 2 words |
| **Question Context** | 112.3 words | 106.0 words | 205 words | 11 words |
| **Answer** | 190.6 words | 176.5 words | 402 words | 9 words |

## 📁 Directory Structure

```
dataset_creation/
├── scripts/               # Data processing scripts
│   ├── collect_automotive_questions_final.py  # Data collection
│   ├── clean_dataset_answered_only.py        # Filter answered questions
│   ├── analyze_answered_dataset.py           # Generate statistics
│   └── classify_questions.py                 # Classify by type
├── data/                  # Dataset files
│   ├── apqc_auto.json                       # Main dataset (706 Q&A pairs)
│   ├── apqc_auto_causal.json               # Causal questions subset
│   ├── apqc_auto_diagnostic.json           # Diagnostic questions subset
│   ├── apqc_auto_high_confidence.json      # High-quality subset
│   ├── raw/                                # Original data
│   └── processed/                          # Processed versions
└── documentation/        # Reports and documentation
```

## 📝 Data Format

Each entry in the dataset follows this structure:

```json
{
  "id": "auto_XXXXX",
  "question": "Question title",
  "context": "Detailed question description",
  "answer": "Expert-provided answer",
  "category": "Question type (factual/causal/diagnostic/comparative)",
  "classification_confidence": 0.95,
  "metadata": {
    "source": "stack_exchange",
    "score": 15,
    "tags": ["engine", "oil", "maintenance"],
    "has_accepted_answer": true,
    "collected_tag": "engine"
  }
}
```

## 🔍 Question Categories

Questions are classified into four main types:

1. **Factual** (55.4%) - Questions about facts, definitions, procedures
   - Example: "What is the proper oil viscosity for my engine?"

2. **Causal** (17.0%) - Questions about cause-effect relationships
   - Example: "Why does my engine overheat when idling?"

3. **Diagnostic** (16.7%) - Questions about troubleshooting problems
   - Example: "Car won't start, battery is good, what could be wrong?"

4. **Comparative** (10.9%) - Questions comparing options or alternatives
   - Example: "Synthetic vs conventional oil: which is better?"

## 🚀 Usage

### Loading the Dataset
```python
import json

# Load main dataset
with open('data/apqc_auto.json', 'r') as f:
    data = json.load(f)
    questions = data['questions']

# Each question has both question and answer
for q in questions[:5]:
    print(f"Q: {q['question']}")
    print(f"A: {q['answer'][:200]}...")  # First 200 chars
```

### Running Analysis
```bash
# Analyze the dataset
python scripts/analyze_answered_dataset.py

# Clean dataset (remove unanswered)
python scripts/clean_dataset_answered_only.py
```

## 📜 License

The dataset is derived from Stack Exchange content under the **CC BY-SA 4.0** license. Stack Exchange requires attribution when using their data.

### Attribution
This dataset contains content from Stack Exchange Mechanics (mechanics.stackexchange.com), which is licensed under CC BY-SA 4.0. Individual posts are owned by their respective authors.

## 🔬 Research Context

This dataset was created for research on reducing hallucinations in Large Language Models (LLMs) using Graph-based Retrieval-Augmented Generation (GraphRAG) in the automotive domain. The presence of verified answers is crucial for:
- Training reliable Q&A systems
- Evaluating answer quality
- Building knowledge graphs with verified information
- Comparing different RAG approaches

## 📊 Key Insights

- **High Quality**: Average score of 11.6 indicates community-validated content
- **Comprehensive Coverage**: Covers all major automotive systems
- **Expert Answers**: All questions have verified answers from the community
- **Real-world Relevance**: Questions from actual car owners and mechanics
- **Causal Reasoning**: 17% of questions involve causal relationships, ideal for knowledge graph construction

## 📮 Contact

For questions about this dataset or its use in research, please contact the CKG-RAG research team.

---
*Last updated: Dataset cleaned to include only answered questions*

## Evaluation Setup Log

### Date: 2024-08-20

#### Configuration Created
- **Location**: `dataset_creation/analysis/`
- **Files Created**:
  - `eval_config.yaml` - Configuration for evaluation
  - `eval_runner.py` - Main evaluation script

#### Implementation Details
- **LLM Integration**: Direct OpenAI API (gpt-4o-mini)
- **Retrieval**: Client-side hybrid fusion (hybrid_ahs)
  - Vector search via Knowledge Manager API
  - Graph search via Knowledge Manager API
  - Weighted fusion based on question category
- **Structured Output**: OpenAI JSON mode for claims extraction and judging

#### Evaluation Modes
1. **base_llm** - No retrieval, pure LLM
2. **vector_rag** - Vector retrieval only
3. **graph_rag** - Graph retrieval only  
4. **hybrid_ahs** - Client-side weighted fusion

#### Metrics Implemented
- **HR** (Hallucination Rate): contradicted + unverifiable claims
- **HR_contra**: contradicted claims only
- **HR_unver**: unverifiable claims only
- **CHR** (Causal HR): HR for causal claims
- **GAC**: Disabled (no KG path-check endpoint available)

#### Usage
```bash
# Test run with 25 questions
python analysis/eval_runner.py --limit 25

# Full evaluation with all 706 questions
python analysis/eval_runner.py
```

#### Output Files (in `analysis/outputs/`)
- `predictions_YYYYMMDD_HHMM.jsonl` - Detailed results per question
- `metrics_YYYYMMDD_HHMM.csv` - Aggregated metrics
- `metrics_YYYYMMDD_HHMM.md` - Summary report

#### Hybrid Weights Configuration
- **Causal questions**: graph=0.7, vector=0.3
- **Diagnostic questions**: graph=0.7, vector=0.3
- **Factual questions**: graph=0.3, vector=0.7
- **Comparative questions**: graph=0.4, vector=0.6

### Evaluation Run - 20250820_2246
- Config: eval_config.yaml
- Modes: base_llm
- Questions: 25
- Outputs: outputs
- Completed: 2025-08-20T23:03:28.507897
