# Automotive Q&A Hallucination Analysis

## Overview
This directory contains the complete experimental analysis of hallucination detection in Retrieval-Augmented Generation (RAG) systems for automotive question-answering. The experiments evaluate four different approaches on 706 automotive Q&A pairs from the APQC automotive dataset.

## Experimental Setup

### Dataset
- **Source**: APQC Automotive Q&A Dataset (`apqc_auto.json`)
- **Size**: 706 question-answer pairs
- **Categories**: 
  - Factual (392 questions)
  - Causal (120 questions) 
  - Diagnostic (117 questions)
  - Comparative (77 questions)

### Methods Evaluated
1. **Base LLM** - GPT-4o-mini without any external context (baseline)
2. **Vector RAG** - Web-augmented generation using Tavily API for factual retrieval
3. **Graph RAG** - Causal/relational search with "why/cause" query augmentation
4. **Hybrid AHS** - Adaptive Hybrid System combining vector and graph approaches

### Metrics
- **Hallucination Rate (HR)**: Percentage of claims that are contradicted or unverifiable
- **Factual Accuracy Score (FAS)**: 1 - HR (higher is better, used for publication)
- **Support Rate**: Percentage of claims supported by evidence
- **Statistical Significance**: p-values, Cohen's d, 95% confidence intervals

## Results Summary

### Main Findings (N=706)
| Method | FAS (%) | 95% CI | p-value vs baseline | Cohen's d |
|--------|---------|---------|---------------------|-----------|
| **Vector RAG** | 78.0% | [75.8, 80.2] | <0.001 | 1.606 |
| **Hybrid AHS** | 73.8% | [71.4, 76.2] | <0.001 | 1.405 |
| **Graph RAG** | 65.1% | [62.6, 67.6] | <0.001 | 1.076 |
| **Base LLM** | 30.6% | [28.5, 32.8] | - | - |

### Key Improvements
- Vector RAG reduces hallucinations by **68%** compared to baseline
- All RAG methods show statistically significant improvements (p < 0.001)
- Large effect sizes (Cohen's d > 1.0) confirm practical significance

## Project Structure

```
analysis/
├── README.md                          # This file
│
├── Evaluation Scripts/
│   ├── eval_full_706.py              # Full dataset evaluation (706 questions)
│   ├── evaluate_correctly.py         # Corrected evaluation against gold answers
│   └── evaluate_rag_hallucination.py # Initial 100-question pilot study
│
├── Hallucination Analysis/
│   ├── hallucination_full_api_706.py # Complete API-based hallucination analysis
│   ├── analyze_full_706_hallucination.py # Analysis on full dataset
│   └── fast_hallucination_analysis_706.py # Heuristic-based quick analysis
│
├── Statistical Analysis/
│   ├── calculate_fas_metrics.py      # Factual Accuracy Score calculation
│   ├── statistical_analysis_fas.py   # Comprehensive statistical tests
│   └── calculate_factual_accuracy_score.py # FAS with visualizations
│
├── Results (JSON)/
│   ├── eval_FULL_706_results_*.json  # Complete evaluation results
│   ├── hallucination_FULL_API_706_results_*.json # API analysis results
│   ├── factual_accuracy_score_*.json # FAS analysis results
│   └── fas_statistical_report_*.json # Statistical significance report
│
├── Results (CSV)/
│   ├── fas_descriptive_stats_*.csv   # Descriptive statistics table
│   ├── fas_pairwise_comparisons_*.csv # Pairwise comparison p-values
│   └── fas_category_stats_*.csv      # Category-wise performance
│
└── Logs/
    ├── evaluation_706.log             # Full evaluation execution log
    └── hallucination_api_706.log      # API analysis execution log
```

## Key Scripts

### 1. Full Evaluation (`eval_full_706.py`)
- Processes all 706 questions through 4 RAG methods
- Uses real APIs (OpenAI GPT-4o-mini + Tavily)
- Implements checkpoint system for resumption
- Runtime: ~5.3 hours

### 2. Hallucination Analysis (`hallucination_full_api_706.py`)
- Extracts 3-5 claims per answer using GPT-4o-mini
- Judges each claim as supported/contradicted/unverifiable
- Calculates HR metrics for all methods
- Runtime: ~6 hours

### 3. Statistical Analysis (`statistical_analysis_fas.py`)
- Performs rigorous statistical tests:
  - T-tests and Mann-Whitney U tests
  - Kruskal-Wallis ANOVA
  - Bootstrap confidence intervals (10,000 iterations)
  - Cohen's d effect sizes
- Generates publication-ready LaTeX tables

### 4. FAS Calculation (`calculate_fas_metrics.py`)
- Converts HR to Factual Accuracy Score (FAS = 1 - HR)
- More intuitive metric for publication (higher = better)
- Provides ready-to-use formulations for papers

## Reproduction Instructions

### Prerequisites
```bash
pip install openai tavily scipy pandas numpy python-dotenv
```

### Environment Variables
Create `.env` file:
```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

### Running Experiments

1. **Full Evaluation** (Warning: ~$15-20 in API costs)
```bash
python eval_full_706.py
```

2. **Hallucination Analysis** (Warning: ~$2-3 in API costs)
```bash
python hallucination_full_api_706.py
```

3. **Statistical Analysis** (No API costs)
```bash
python statistical_analysis_fas.py
```

4. **FAS Calculation** (No API costs)
```bash
python calculate_fas_metrics.py
```

## Publication Materials

### LaTeX Tables
The statistical analysis generates publication-ready LaTeX tables:
- Table 1: Descriptive Statistics (Mean, CI, Median)
- Table 2: Pairwise Comparisons with Significance

### Recommended Citation Format
```
We evaluated four retrieval approaches on 706 automotive Q&A pairs,
measuring Factual Accuracy Score (FAS = 1 - Hallucination Rate).
Vector RAG achieved the highest FAS of 78.0% (95% CI: [75.8, 80.2]),
representing a 155% improvement over the baseline (p < 0.001, Cohen's d = 1.61).
```

### Key Findings for Paper
1. **Baseline validation**: GPT-4o-mini without context achieves only 30.6% factual accuracy
2. **Best method**: Vector RAG (web-augmented) reduces hallucinations by 68%
3. **Statistical rigor**: All improvements are statistically significant (p < 0.001)
4. **Effect sizes**: Large Cohen's d values (>1.0) confirm practical significance
5. **Category analysis**: Methods show consistent improvements across all question types

## Cost Estimates
- Full evaluation (706 questions × 4 methods): ~$15-20
- Hallucination analysis (claim extraction + judging): ~$2-3
- Total experiment cost: ~$20-25

## Runtime Estimates
- Full evaluation: ~5-6 hours
- Hallucination analysis: ~6-7 hours  
- Statistical analysis: <1 minute

## Contact
For questions about reproduction or methodology, please refer to the paper or contact the authors.

## License
This research code is provided for academic purposes. Please cite our paper if you use this code or data.

---
*Last updated: August 22, 2025*