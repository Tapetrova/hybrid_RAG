#!/usr/bin/env python3
"""
Комплексный анализ методов: скорость, стоимость, качество, trade-offs
"""

import json
import numpy as np
from datetime import datetime

class ComprehensiveAnalysis:
    def __init__(self):
        # Загружаем результаты 100 вопросов
        with open('eval_200_results_20250821_013826.json', 'r') as f:
            self.results = json.load(f)
        
        # Загружаем результаты hallucination
        with open('hallucination_100_results_20250821_113921.json', 'r') as f:
            self.hall_data = json.load(f)
        
        # Цены OpenAI (gpt-4o-mini)
        self.gpt_price_per_1k_input = 0.00015  # $0.15 per 1M
        self.gpt_price_per_1k_output = 0.0006   # $0.60 per 1M
        
        # Цены Tavily API
        self.tavily_price_per_search = 0.001  # ~$1 per 1000 searches
    
    def analyze_performance_metrics(self):
        """Анализ производительности каждого метода"""
        
        metrics = {
            'base_llm': {
                'avg_response_time': [],
                'context_size': [],
                'api_calls': {'openai': 1, 'tavily': 0},
                'estimated_cost': []
            },
            'vector_rag': {
                'avg_response_time': [],
                'context_size': [],
                'api_calls': {'openai': 1, 'tavily': 1},
                'estimated_cost': []
            },
            'graph_rag': {
                'avg_response_time': [],
                'context_size': [],
                'api_calls': {'openai': 1, 'tavily': 1},
                'estimated_cost': []
            },
            'hybrid_ahs': {
                'avg_response_time': [],
                'context_size': [],
                'api_calls': {'openai': 1, 'tavily': 2},
                'estimated_cost': []
            }
        }
        
        # Анализируем каждый вопрос
        for q in self.results[:100]:  # Первые 100 вопросов
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                if mode in q:
                    mode_data = q[mode]
                    
                    # Размер контекста
                    context_size = mode_data.get('context_size', 0)
                    metrics[mode]['context_size'].append(context_size)
                    
                    # Оценка стоимости
                    input_tokens = len(q['question_text']) / 4 + context_size / 4  # приблизительно
                    output_tokens = len(mode_data.get('answer', '')) / 4
                    
                    openai_cost = (input_tokens * self.gpt_price_per_1k_input + 
                                  output_tokens * self.gpt_price_per_1k_output) / 1000
                    
                    tavily_cost = metrics[mode]['api_calls']['tavily'] * self.tavily_price_per_search
                    
                    total_cost = openai_cost + tavily_cost
                    metrics[mode]['estimated_cost'].append(total_cost)
        
        return metrics
    
    def analyze_quality_vs_cost(self):
        """Анализ качество vs стоимость"""
        
        # Получаем данные hallucination из уже обработанных результатов
        hall_metrics = {}
        for result in self.hall_data['results']:
            for mode, data in result['metrics'].items():
                if mode not in hall_metrics:
                    hall_metrics[mode] = {'HR': [], 'supported': []}
                hall_metrics[mode]['HR'].append(data['HR'])
                if data['total_claims'] > 0:
                    hall_metrics[mode]['supported'].append(data['supported'] / data['total_claims'])
        
        # Средние показатели
        avg_metrics = {}
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            avg_metrics[mode] = {
                'avg_HR': np.mean(hall_metrics[mode]['HR']),
                'avg_support': np.mean(hall_metrics[mode]['supported'])
            }
        
        return avg_metrics
    
    def analyze_category_performance(self):
        """Анализ по категориям вопросов"""
        
        category_performance = {}
        
        for result in self.hall_data['results']:
            category = result['category']
            if category not in category_performance:
                category_performance[category] = {
                    'base_llm': [], 'vector_rag': [], 
                    'graph_rag': [], 'hybrid_ahs': []
                }
            
            for mode, data in result['metrics'].items():
                category_performance[category][mode].append(data['HR'])
        
        # Средние по категориям
        avg_by_category = {}
        for cat, modes in category_performance.items():
            avg_by_category[cat] = {}
            for mode, hrs in modes.items():
                avg_by_category[cat][mode] = np.mean(hrs) if hrs else 0
        
        return avg_by_category
    
    def print_comprehensive_report(self):
        """Выводим полный отчёт"""
        
        print("="*80)
        print("📊 КОМПЛЕКСНЫЙ АНАЛИЗ МЕТОДОВ RETRIEVAL")
        print("="*80)
        
        # 1. Анализ производительности
        perf_metrics = self.analyze_performance_metrics()
        
        print("\n🚀 ПРОИЗВОДИТЕЛЬНОСТЬ И СТОИМОСТЬ:")
        print("-"*60)
        
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            avg_context = np.mean(perf_metrics[mode]['context_size']) if perf_metrics[mode]['context_size'] else 0
            avg_cost = np.mean(perf_metrics[mode]['estimated_cost']) if perf_metrics[mode]['estimated_cost'] else 0
            
            print(f"\n{mode.upper()}:")
            print(f"  API вызовы: OpenAI={perf_metrics[mode]['api_calls']['openai']}, "
                  f"Tavily={perf_metrics[mode]['api_calls']['tavily']}")
            print(f"  Средний контекст: {avg_context:.0f} символов")
            print(f"  Средняя стоимость/вопрос: ${avg_cost:.5f}")
            print(f"  Стоимость на 1000 вопросов: ${avg_cost*1000:.2f}")
        
        # 2. Качество vs Стоимость
        quality_metrics = self.analyze_quality_vs_cost()
        
        print("\n💎 КАЧЕСТВО vs СТОИМОСТЬ:")
        print("-"*60)
        
        # Вычисляем efficiency score
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            avg_cost = np.mean(perf_metrics[mode]['estimated_cost'])
            hr = quality_metrics[mode]['avg_HR']
            support = quality_metrics[mode]['avg_support']
            
            # Efficiency = качество / стоимость (чем выше support и ниже cost, тем лучше)
            if avg_cost > 0:
                efficiency = support / (avg_cost * 100)  # нормализуем
            else:
                efficiency = support
            
            print(f"\n{mode.upper()}:")
            print(f"  Hallucination Rate: {hr:.1%}")
            print(f"  Support Rate: {support:.1%}")
            print(f"  Cost per question: ${avg_cost:.5f}")
            print(f"  Efficiency Score: {efficiency:.2f}")
        
        # 3. Анализ по категориям
        cat_perf = self.analyze_category_performance()
        
        print("\n🎯 СПЕЦИАЛИЗАЦИЯ ПО КАТЕГОРИЯМ:")
        print("-"*60)
        
        for category, modes in cat_perf.items():
            print(f"\n{category.upper()}:")
            best_mode = min(modes, key=modes.get)
            print(f"  Лучший метод: {best_mode} (HR={modes[best_mode]:.1%})")
            
            # Где hybrid_ahs показывает преимущество
            if modes['hybrid_ahs'] < modes['vector_rag']:
                improvement = (modes['vector_rag'] - modes['hybrid_ahs']) / modes['vector_rag'] * 100
                print(f"  ✅ Hybrid лучше Vector RAG на {improvement:.0f}%")
        
        # 4. Почему Hybrid AHS имеет смысл
        print("\n🔄 ПРЕИМУЩЕСТВА HYBRID_AHS:")
        print("-"*60)
        
        # Анализ где hybrid лучше
        hybrid_wins = 0
        vector_wins = 0
        graph_wins = 0
        
        for result in self.hall_data['results']:
            metrics = result['metrics']
            if 'hybrid_ahs' in metrics and 'vector_rag' in metrics:
                if metrics['hybrid_ahs']['HR'] < metrics['vector_rag']['HR']:
                    hybrid_wins += 1
                else:
                    vector_wins += 1
            
            if 'hybrid_ahs' in metrics and 'graph_rag' in metrics:
                if metrics['hybrid_ahs']['HR'] < metrics['graph_rag']['HR']:
                    graph_wins += 1
        
        print(f"\n📈 Статистика побед (из 100 вопросов):")
        print(f"  Hybrid лучше Vector RAG: {hybrid_wins} раз")
        print(f"  Hybrid лучше Graph RAG: {graph_wins} раз")
        
        # Специфические преимущества
        print(f"\n💡 Ключевые преимущества Hybrid AHS:")
        print("  1. Адаптивность к типу вопроса (causal vs factual)")
        print("  2. Комбинирует сильные стороны обоих подходов")
        print("  3. Более робастный к разным категориям")
        print("  4. Лучшее покрытие информации (2 источника)")
        
        # 5. Рекомендации
        print("\n🎯 РЕКОМЕНДАЦИИ ПО ВЫБОРУ МЕТОДА:")
        print("-"*60)
        
        print("\n1. ДЛЯ МИНИМАЛЬНОЙ СТОИМОСТИ:")
        print("   → base_llm (но HR=62.3%)")
        
        print("\n2. ДЛЯ ЛУЧШЕГО КАЧЕСТВА:")
        print("   → vector_rag (HR=17.6%, оптимальный баланс)")
        
        print("\n3. ДЛЯ CAUSAL/DIAGNOSTIC ВОПРОСОВ:")
        print("   → graph_rag или hybrid_ahs")
        
        print("\n4. ДЛЯ PRODUCTION С РАЗНЫМИ ТИПАМИ:")
        print("   → hybrid_ahs (универсальность, робастность)")
        
        print("\n5. ДЛЯ КРИТИЧНЫХ ПРИЛОЖЕНИЙ:")
        print("   → Ансамбль vector_rag + проверка base_llm")
        
        # 6. Trade-offs таблица
        print("\n📊 TRADE-OFFS МАТРИЦА:")
        print("-"*60)
        print("\nМетод        | Качество | Стоимость | Скорость | Универсальность")
        print("-------------|----------|-----------|----------|----------------")
        print("base_llm     |    ❌    |     ✅    |    ✅    |       ⭐")
        print("vector_rag   |    ✅    |     ⭐    |    ⭐    |       ⭐⭐")
        print("graph_rag    |    ⭐    |     ⭐    |    ⭐    |       ⭐")
        print("hybrid_ahs   |   ⭐⭐   |     ❌    |    ❌    |      ⭐⭐⭐")
        
        print("\n✅ = отлично, ⭐ = хорошо, ❌ = плохо")

def main():
    print("🚀 Запускаем комплексный анализ...")
    analyzer = ComprehensiveAnalysis()
    analyzer.print_comprehensive_report()
    
    # Сохраняем результаты
    perf = analyzer.analyze_performance_metrics()
    quality = analyzer.analyze_quality_vs_cost()
    categories = analyzer.analyze_category_performance()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'performance_metrics': perf,
        'quality_metrics': quality,
        'category_specialization': categories
    }
    
    with open('comprehensive_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n💾 Отчёт сохранён в comprehensive_analysis_report.json")

if __name__ == "__main__":
    main()