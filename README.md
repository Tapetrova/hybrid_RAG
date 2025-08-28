# Hybrid RAG: Adaptive Hallucination Suppression in Retrieval-Augmented Generation for Domain-Specific Q&A

## Overview

This repository contains the implementation and experimental code for the Hybrid Adaptive Hallucination Suppression (Hybrid-AHS) approach for automotive domain Q&A systems.

## Key Features

- **Hybrid RAG Architecture**: Combines vector and graph-based retrieval methods
- **Weighted Factual Accuracy Score (WFAS)**: Novel evaluation metric with differential weighting for hallucination types
- **Robustness Across Categories (RAC)**: Metric for measuring consistency across different question types
- **APQC Automotive Dataset**: 706 expert-validated automotive Q&A pairs

## Project Structure

```
├── dataset_creation/       # Dataset processing and analysis
│   ├── analysis/          # Statistical analysis and visualizations
│   └── data/             # APQC automotive dataset
├── graph-rag-main/        # Graph RAG implementation
│   ├── apps/             # Application modules
│   └── libs/             # Utility libraries
└── analyze_categories.py  # Category analysis script
```

## Key Results

- **Overall WFAS**: Hybrid-AHS achieves 89.6% weighted factual accuracy
- **RAC Score**: 0.942 (highest robustness across question categories)
- **Classification Accuracy**: 90.7% (κ = 0.852) for category classifier

## Requirements

- Python 3.9+
- See individual `requirements.txt` files in subdirectories

## Citation

*Paper currently under review*

## Author

Tapetrova

## License

This project is private and proprietary. All rights reserved.

## Contact

For questions or collaboration inquiries, please contact through GitHub.