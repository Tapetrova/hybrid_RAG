# CKG-RAG Data Collection Summary

## Dataset Overview
Successfully collected automotive Q&A data from Stack Exchange Mechanics

### Final Statistics
- **Total Questions**: 1,106 unique questions
- **Questions with Answers**: 708 (64.0% answer rate)
- **Average Score**: 10.6 (indicating high-quality questions)
- **APQ-C Evaluation Set**: 200 questions (balanced across categories)

### Distribution by Category
| Category | Questions | With Answers | Answer Rate |
|----------|-----------|--------------|-------------|
| Brakes | 196 | 112 | 57.1% |
| Electrical | 191 | 112 | 58.6% |
| Engine | 184 | 143 | 77.7% |
| Transmission | 96 | 63 | 65.6% |
| Exhaust | 94 | 66 | 70.2% |
| Air-conditioning | 93 | 54 | 58.1% |
| Coolant | 92 | 50 | 54.3% |
| Oil | 89 | 56 | 62.9% |
| Battery | 71 | 52 | 73.2% |

### File Structure
```
research/
├── data/
│   ├── processed/
│   │   ├── automotive_qa_final_20250816_230600.json (2.8MB) - Full dataset
│   │   ├── apqc_auto_200.json (493KB) - Evaluation subset
│   │   └── automotive_qa_summary_20250816_230600.csv (90KB) - Summary CSV
│   └── raw/
│       └── automotive_questions_with_answers_20250816_224408.json (2.3MB) - Original collection
```

### Data Format (APQ-C)
```json
{
  "id": "auto_XXXXX",
  "question": "Question title",
  "context": "Question body with details",
  "answer": "Accepted answer text",
  "category": "Category tag",
  "score": Score value,
  "tags": ["tag1", "tag2", ...]
}
```

### Collection Details
- **Source**: Stack Exchange Mechanics (mechanics.stackexchange.com)
- **Collection Date**: August 16, 2025
- **Selection Criteria**: Top-voted questions per category
- **Quality Control**: Deduplicated by question_id, sorted by score

### Next Steps
1. ✅ Data collection completed
2. ✅ Data cleaning and deduplication done
3. ✅ APQ-C format dataset prepared
4. Ready for:
   - Building causal knowledge graph
   - Implementing baseline RAG systems
   - Running comparative experiments
   - Evaluating with CCS metric

### Notes
- Rate limiting prevented collecting full 200 per category for some tags
- Prioritized quality (high-score questions) over quantity
- Engine category has highest answer rate (77.7%)
- Dataset suitable for demonstrating causal reasoning advantages