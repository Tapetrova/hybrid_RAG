# Dataset Structure - CKG-RAG Paper

## Final Dataset Files

### Main Dataset
- **`data/apqc_auto.json`** (1.9 MB)
  - Complete APQ-C format dataset
  - 1,102 unique automotive questions
  - 4 duplicates removed (cosine similarity > 0.8)
  - Categories: Causal (166), Diagnostic (203), Comparative (124), Factual (609)
  - 706 questions with answers (64% answer rate)

### Experimental Subsets
- **`data/apqc_auto_causal.json`** - 166 causal reasoning questions
- **`data/apqc_auto_diagnostic.json`** - 203 diagnostic problem-solving questions  
- **`data/apqc_auto_high_confidence.json`** - 580 high-confidence classified questions
- **`data/apqc_auto_with_answers.json`** - 706 questions with accepted answers

### Statistics
- **`data/apqc_auto_statistics.json`** - Detailed dataset statistics

### Intermediate Files (Keep for Reproducibility)
- **`data/raw/automotive_questions_with_answers_20250816_224408.json`** - Original 900 questions from Stack Exchange
- **`data/processed/automotive_qa_final_20250816_230600.json`** - Merged 1,106 questions before duplicate removal
- **`data/processed/classified_questions.json`** - 1,106 questions with classifications
- **`data/processed/classification_statistics.json`** - Classification analysis results
- **`data/processed/dataset_statistics_updated.json`** - Updated statistics after classification

## Code Files

### Data Collection & Processing
1. **`collect_automotive_questions_final.py`** - Stack Exchange API data collection
2. **`classify_questions.py`** - Pattern-based question classification
3. **`create_final_dataset.py`** - Duplicate removal and APQ-C formatting
4. **`check_duplicates.py`** - Similarity analysis for duplicate detection

### Statistics & Analysis
5. **`generate_dataset_statistics.py`** - Dataset statistics generation
6. **`update_statistics_with_classification.py`** - Update stats with classification results

### System Implementation (Ready for Next Phase)
7. **`ckg_rag_system.py`** - Complete CKG-RAG system implementation
8. **`build_causal_graph.py`** - Causal knowledge graph builder (Week 2)

## Visualizations

### Paper Figures (`paper/figures/`)
- `category_distribution.png` - Question category distribution
- `classification_analysis.png` - Classification confidence analysis
- `classification_examples.png` - Example classifications
- `cross_category_heatmap.png` - Category correlation matrix
- `question_types.png` - Question type breakdown
- `score_distribution.png` - Stack Exchange score distribution
- `similarity_distribution.png` - Cosine similarity analysis
- `text_lengths.png` - Text length distributions

## Reports
- **`FINAL_DATASET_REPORT.md`** - Comprehensive dataset documentation
- **`DATASET_STRUCTURE.md`** - This file

## Data Quality Summary

✅ **Dataset Ready for Experiments**
- 1,102 unique questions (4 duplicates removed)
- Balanced categories (all >100 questions)
- 64% answer rate (706 questions)
- 52.6% high-confidence classifications
- No synthetic data - all from Stack Exchange
- Clean APQ-C format

## Next Steps
1. Week 2: Build causal knowledge graph from answers
2. Week 3: Implement baseline RAG systems
3. Week 4: Run comparative experiments
4. Week 5: Calculate metrics and analyze results