# Hybrid RAG: Adaptive Hallucination Suppression in Retrieval-Augmented Generation for Domain-Specific Q&A

## Overview

Implementation and experimental code for the paper "Hybrid RAG: Adaptive Hallucination Suppression in Retrieval-Augmented Generation for Domain-Specific Q&A" (under review).

This repository contains the Hybrid-AHS approach for reducing hallucinations in automotive domain Q&A systems.

## Key Features

- **Hybrid RAG Architecture**: Combines vector and graph-based retrieval methods
- **Weighted Factual Accuracy Score (WFAS)**: Novel evaluation metric with differential weighting for hallucination types
- **Robustness Across Categories (RAC)**: Metric for measuring consistency across different question types
- **APQC Automotive Dataset**: 706 expert-validated automotive Q&A pairs

## Project Structure

```
├── dataset_creation/
│   ├── data/
│   │   └── apqc_auto.json         # 706 automotive Q&A pairs
│   ├── analysis/                  # Core analysis scripts
│   │   ├── calculate_rac_corrected.py        # RAC metric calculation
│   │   ├── create_wfas_2_5_*.py             # WFAS visualizations
│   │   ├── create_rac_cs_charts.py          # RAC charts
│   │   ├── hallucination_full_api_706.py    # Hallucination analysis
│   │   └── validate_category_classifier*.py  # Classification validation
│   ├── scripts/                   # Dataset processing
│   └── documentation/             # Dataset documentation
├── graph-rag-main/                # Graph RAG implementation
└── requirements.txt               # Python dependencies
```

## Key Results

- **Overall WFAS**: Hybrid-AHS achieves 89.6% weighted factual accuracy
- **RAC Score**: 0.942 (highest robustness across question categories)
- **Classification Accuracy**: 90.7% (κ = 0.852) for category classifier

## Requirements

- Python 3.9+
- OpenAI API key (for hallucination evaluation)
- ~8GB RAM for processing
- ~$50-100 in API credits for full evaluation (706 questions × 4 methods)

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Tapetrova/hybrid_RAG.git
cd hybrid_RAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Running the Experiments

### Important Note
The repository contains the code implementation but **not the evaluation results**. To reproduce the paper's findings, you need to run the full evaluation pipeline, which requires:
- OpenAI API access for GPT-4 evaluation
- Implementing your own RAG methods or using existing ones
- ~24-48 hours for complete evaluation of all 706 questions

### Step 1: Dataset Analysis
```bash
cd dataset_creation/analysis
python comprehensive_analysis.py
```
This analyzes the APQC automotive dataset (works without API).

### Step 2: Run Hallucination Evaluation
```bash
python hallucination_full_api_706.py
```
**Note**: This requires:
- OpenAI API key configured
- Implementation of 4 methods (BASE_LLM, VECTOR_RAG, GRAPH_RAG, HYBRID_AHS)
- Generates: `hallucination_FULL_API_706_results_*.json`

### Step 3: Calculate Metrics
```bash
python calculate_rac_corrected.py
```
This calculates RAC and WFAS metrics from evaluation results.

### Step 4: Generate Visualizations
```bash
python generate_paper_figures.py
```
Creates all paper figures (requires completion of Step 2-3).

## Expected Outputs

After successful execution:
- `comprehensive_analysis_report.json` - Dataset statistics
- `hallucination_FULL_API_706_results_*.json` - Evaluation results
- `rac_corrected_metrics_*.json` - RAC metrics
- `*.png` - Paper figures

## Troubleshooting

1. **FileNotFoundError for JSON files**: Run the evaluation pipeline first (Step 2)
2. **API errors**: Check your OpenAI API key and credits
3. **Memory errors**: Reduce batch size in evaluation scripts

## Citation

Paper submitted for review: "Hybrid RAG: Adaptive Hallucination Suppression in Retrieval-Augmented Generation for Domain-Specific Q&A"

## Status

Under review