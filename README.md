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
- See individual `requirements.txt` files in subdirectories

## Citation

Paper submitted for review: "Hybrid RAG: Adaptive Hallucination Suppression in Retrieval-Augmented Generation for Domain-Specific Q&A"

## Status

Under review