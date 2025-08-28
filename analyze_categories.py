import json

with open('/Users/tapetrova/PycharmProjects/Car_paper_v2/pythonProject/dataset_creation/data/apqc_auto.json', 'r') as f:
    data = json.load(f)

print(f'Total questions: {len(data["questions"])}')

categories = {}
examples = {}

for q in data['questions']:
    cat = q.get('category', 'unknown')
    categories[cat] = categories.get(cat, 0) + 1
    
    # Collect examples with high confidence
    if cat not in examples:
        examples[cat] = []
    if len(examples[cat]) < 3 and q.get('classification_confidence', 0) > 0.8:
        examples[cat].append({
            'id': q['id'],
            'question': q['question'],
            'confidence': q.get('classification_confidence', 0)
        })

print('\nCategories count:')
for k, v in sorted(categories.items()):
    print(f'{k}: {v}')

print('\n\n=== ПРИМЕРЫ ВОПРОСОВ ПО КАТЕГОРИЯМ ===\n')
for cat in sorted(examples.keys()):
    print(f'\n{cat.upper()} (3 примера):')
    print('-' * 50)
    for i, ex in enumerate(examples[cat][:3], 1):
        print(f'{i}. {ex["question"]}')
        print(f'   ID: {ex["id"]}, Confidence: {ex["confidence"]:.3f}\n')
