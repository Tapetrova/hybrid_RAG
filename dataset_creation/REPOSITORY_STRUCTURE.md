# Repository Structure - Automotive Q&A Dataset

## Current Directory Structure

```
/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/
│
├── dataset_creation/                    # Root directory for dataset creation project
│   │
│   ├── README.md                       # Main documentation with dataset overview
│   ├── REPOSITORY_STRUCTURE.md         # This file - repository structure overview
│   │
│   ├── data/                           # Dataset files (questions with answers only)
│   │   ├── apqc_auto.json              # Main dataset - 706 Q&A pairs
│   │   ├── apqc_auto_causal.json       # Causal questions subset - 120 Q&A pairs
│   │   ├── apqc_auto_diagnostic.json   # Diagnostic questions subset - 118 Q&A pairs
│   │   ├── apqc_auto_high_confidence.json  # High-quality subset - 391 Q&A pairs
│   │   ├── final_statistics_answered_only.json  # Dataset statistics
│   │   └── raw/                        # Original unprocessed data
│   │       └── automotive_questions_with_answers_20250816_224408.json
│   │
│   ├── scripts/                        # Python scripts for data processing
│   │   ├── collect_automotive_questions_final.py  # Stack Exchange data collection
│   │   ├── clean_dataset_answered_only.py        # Remove questions without answers
│   │   ├── analyze_answered_dataset.py           # Comprehensive dataset analysis
│   │   ├── classify_questions.py                 # Question type classification
│   │   ├── create_final_dataset.py              # Dataset creation and formatting
│   │   ├── generate_dataset_statistics.py       # Statistics generation
│   │   ├── check_duplicates.py                  # Duplicate detection and removal
│   │   └── update_statistics_with_classification.py  # Update stats with classifications
│   │
│   ├── documentation/                  # Project documentation and reports
│   │   ├── DATA_COLLECTION_SUMMARY.md          # Data collection process overview
│   │   ├── DATASET_STATISTICS_REPORT.md        # Detailed statistical analysis
│   │   ├── DATASET_STRUCTURE.md                # Data format documentation
│   │   ├── FINAL_DATASET_REPORT.md            # Complete dataset documentation
│   │   ├── CLASSIFICATION_SUMMARY.md           # Question classification methodology
│   │   ├── FINAL_ANALYSIS_REPORT.md           # Analysis of answered questions
│   │   └── STACK_EXCHANGE_SCORING_EXPLAINED.md # Stack Exchange score system explanation
│   │
│   └── analysis/                       # (Empty) Reserved for future analysis outputs
│
└── graph-rag-main/                     # GraphRAG implementation (separate project)
    ├── README.md
    ├── docker-compose.yaml
    ├── apps/
    │   ├── agent_flow/                 # AI agent service
    │   ├── content_scraper/            # Web scraping service
    │   └── knowledge_manager/          # Knowledge management service
    ├── libs/                           # Shared libraries
    └── research/                       # Research notebooks and experiments
```

## File Descriptions

### Data Files (`/data`)
- **apqc_auto.json** (1.5 MB) - Complete dataset with all 706 automotive Q&A pairs
- **apqc_auto_causal.json** (254 KB) - Subset of causal reasoning questions
- **apqc_auto_diagnostic.json** (239 KB) - Subset of diagnostic/troubleshooting questions
- **apqc_auto_high_confidence.json** (833 KB) - High-quality questions (score ≥ 10)
- **final_statistics_answered_only.json** (1 KB) - Comprehensive dataset statistics

### Scripts (`/scripts`)
1. **Data Collection**
   - `collect_automotive_questions_final.py` - Fetches data from Stack Exchange API

2. **Data Processing**
   - `clean_dataset_answered_only.py` - Filters to keep only answered questions
   - `create_final_dataset.py` - Processes raw data into final format
   - `check_duplicates.py` - Identifies and removes duplicate questions

3. **Analysis & Classification**
   - `analyze_answered_dataset.py` - Generates comprehensive statistics
   - `classify_questions.py` - Categorizes questions by type
   - `generate_dataset_statistics.py` - Creates statistical reports
   - `update_statistics_with_classification.py` - Updates stats with classifications

### Documentation (`/documentation`)
- **DATA_COLLECTION_SUMMARY.md** - Overview of collection methodology
- **DATASET_STATISTICS_REPORT.md** - Detailed statistical breakdown
- **DATASET_STRUCTURE.md** - JSON structure and field descriptions
- **FINAL_DATASET_REPORT.md** - Complete dataset documentation
- **CLASSIFICATION_SUMMARY.md** - Question classification system
- **FINAL_ANALYSIS_REPORT.md** - Analysis results for answered questions
- **STACK_EXCHANGE_SCORING_EXPLAINED.md** - Understanding quality scores

## Dataset Statistics Summary

- **Total Questions**: 706 (all with verified answers)
- **Source**: Stack Exchange Mechanics
- **Average Score**: 11.6 (high quality)
- **Categories**: Factual (55%), Causal (17%), Diagnostic (17%), Comparative (11%)
- **Topics**: Engine, Brakes, Electrical, Transmission, etc.
- **Text Length**: Questions (~10 words), Answers (~191 words)

## Usage

```python
# Load the main dataset
import json

with open('data/apqc_auto.json', 'r') as f:
    dataset = json.load(f)
    questions = dataset['questions']
    
print(f"Total Q&A pairs: {len(questions)}")
```

## Notes

- All questions without answers have been removed
- Dataset contains only high-quality, community-validated content
- Average Stack Exchange score of 11.6 (typical average is 3-5)
- 46% of questions have score ≥ 10 (high quality)
- All data is in English from mechanics.stackexchange.com