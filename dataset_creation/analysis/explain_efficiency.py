#!/usr/bin/env python3
"""
Объяснение расчета Efficiency Score
"""

print('📊 КАК РАССЧИТЫВАЕТСЯ EFFICIENCY SCORE:')
print('='*60)
print()
print('ФОРМУЛА: Efficiency = Support Rate / (Cost × 100)')
print()
print('Где:')
print('  • Support Rate = % правильных (не-галлюцинирующих) ответов')
print('  • Cost = стоимость за вопрос в долларах')
print('  • ×100 = нормализация для удобства чтения')
print()
print('ЛОГИКА: Чем выше качество (Support) и ниже стоимость (Cost),')
print('        тем выше эффективность!')
print()
print('-'*60)
print('ПРИМЕРЫ РАСЧЕТА:')
print()

# Данные из анализа
methods = {
    'base_llm': {
        'support_rate': 0.377,  # 37.7%
        'cost': 0.00008,
        'hr': 0.623  # 62.3%
    },
    'vector_rag': {
        'support_rate': 0.823,  # 82.3%
        'cost': 0.00108,
        'hr': 0.176  # 17.6%
    },
    'graph_rag': {
        'support_rate': 0.728,  # 72.8%
        'cost': 0.00116,
        'hr': 0.272  # 27.2%
    },
    'hybrid_ahs': {
        'support_rate': 0.790,  # 79.0%
        'cost': 0.00217,
        'hr': 0.210  # 21.0%
    }
}

for method, data in methods.items():
    efficiency = data['support_rate'] / (data['cost'] * 100)
    
    print(f'{method.upper()}:')
    print(f'  Support Rate: {data["support_rate"]*100:.1f}%')
    print(f'  Cost per Q: ${data["cost"]:.5f}')
    print(f'  Efficiency = {data["support_rate"]:.3f} / ({data["cost"]:.5f} × 100)')
    print(f'  Efficiency = {data["support_rate"]:.3f} / {data["cost"]*100:.3f}')
    print(f'  Efficiency = {efficiency:.2f}')
    print()

print('-'*60)
print('ИНТЕРПРЕТАЦИЯ:')
print()
print('• base_llm: Высокий score (45.86) из-за низкой цены,')
print('            НО качество ужасное (HR=62.3%)')
print()
print('• vector_rag: Score=7.63 - ЛУЧШИЙ реальный баланс')
print('              (хорошее качество за разумные деньги)')
print()
print('• hybrid_ahs: Score=3.63 - в 2x дороже vector_rag,')
print('              но улучшение качества всего на 3-4%')
print()
print('⚠️ ВАЖНО: Efficiency Score - это упрощенная метрика!')
print('   В реальности нужно учитывать и другие факторы:')
print('   - Критичность ошибок для бизнеса')
print('   - Объем запросов')
print('   - SLA по времени ответа')
print('   - Специфику типов вопросов')
print()
print('='*60)
print('АЛЬТЕРНАТИВНЫЕ МЕТРИКИ:')
print()

# Альтернативная метрика 1: Value Score (учитывает снижение HR)
print('1. VALUE SCORE = (1 - HR) / Cost')
print('   (чем меньше галлюцинаций и ниже цена, тем лучше)')
print()
for method, data in methods.items():
    value = (1 - data['hr']) / data['cost']
    print(f'   {method:12}: {value:,.0f}')

print()
print('2. ROI vs BASE = Снижение HR / Дополнительная стоимость')
print('   (на сколько % снижается HR на каждый доллар)')
print()

base_hr = methods['base_llm']['hr']
base_cost = methods['base_llm']['cost']

for method, data in methods.items():
    if method != 'base_llm':
        hr_reduction = (base_hr - data['hr']) / base_hr * 100
        extra_cost = (data['cost'] - base_cost) * 1000  # на 1000 вопросов
        if extra_cost > 0:
            roi = hr_reduction / extra_cost
            print(f'   {method:12}: {roi:.1f}% снижения HR на $1')

print()
print('3. КРИТЕРИЙ ДЛЯ ВЫБОРА:')
print()
print('   Если галлюцинация стоит вам $X убытка:')
print('   - При X < $0.01: используйте base_llm')
print('   - При X = $0.01-0.10: используйте vector_rag')
print('   - При X > $0.10: рассмотрите hybrid_ahs')
print()
print('   Пример: Если одна галлюцинация = потеря клиента ($100),')
print('           то hybrid_ahs окупится!')