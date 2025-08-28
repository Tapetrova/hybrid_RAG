#!/usr/bin/env python3
"""
ВДУМЧИВО заполняем Knowledge Manager автомобильными знаниями
Используем Tavily API экономно и только для ключевых концептов
"""

import json
import time
import requests
from pathlib import Path
from typing import List, Dict
from collections import Counter

def extract_key_concepts(dataset_path: str) -> List[str]:
    """Извлекаем ключевые автомобильные концепты из датасета"""
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Собираем все слова из вопросов
    concepts = []
    automotive_keywords = {
        'engine', 'brake', 'transmission', 'oil', 'tire', 'battery', 
        'alternator', 'clutch', 'suspension', 'exhaust', 'fuel', 
        'coolant', 'radiator', 'spark plug', 'filter', 'belt',
        'rotor', 'pad', 'caliper', 'differential', 'axle',
        'carburetor', 'injector', 'turbo', 'supercharger',
        'ECU', 'sensor', 'catalytic converter', 'muffler'
    }
    
    for item in data['questions']:
        question = item['question'].lower()
        for keyword in automotive_keywords:
            if keyword in question:
                concepts.append(keyword)
    
    # Топ-10 самых частых концептов
    concept_counts = Counter(concepts)
    top_concepts = [c[0] for c in concept_counts.most_common(10)]
    
    print(f"Найдено {len(top_concepts)} ключевых концептов:")
    for i, c in enumerate(top_concepts, 1):
        print(f"  {i}. {c} ({concept_counts[c]} упоминаний)")
    
    return top_concepts

def search_with_tavily(query: str, api_key: str) -> Dict:
    """Поиск через Tavily API - ИСПОЛЬЗУЕМ ЭКОНОМНО"""
    
    url = "https://api.tavily.com/search"
    
    payload = {
        "api_key": api_key,
        "query": f"automotive {query} repair maintenance technical",
        "search_depth": "basic",  # basic чтобы экономить кредиты
        "max_results": 3,  # Только 3 результата
        "include_domains": [
            "mechanics.stackexchange.com",
            "reddit.com/r/MechanicAdvice", 
            "reddit.com/r/cars",
            "howstuffworks.com",
            "yourmechanic.com"
        ]
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Tavily API error: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Error calling Tavily: {e}")
        return {}

def populate_knowledge_manager(concepts: List[str], tavily_key: str):
    """Заполняем Knowledge Manager данными"""
    
    print("\n" + "="*60)
    print("НАЧИНАЕМ ВДУМЧИВОЕ ЗАПОЛНЕНИЕ KNOWLEDGE MANAGER")
    print("="*60)
    
    # Ограничиваем до 5 концептов чтобы не тратить много API кредитов
    concepts_to_search = concepts[:5]
    
    print(f"\nБудем искать информацию по {len(concepts_to_search)} концептам:")
    print(", ".join(concepts_to_search))
    
    all_results = []
    
    for i, concept in enumerate(concepts_to_search, 1):
        print(f"\n[{i}/{len(concepts_to_search)}] Поиск: {concept}")
        
        # Поиск через Tavily
        results = search_with_tavily(concept, tavily_key)
        
        if results and 'results' in results:
            print(f"  ✓ Найдено {len(results['results'])} результатов")
            
            # Сохраняем для Knowledge Manager
            for r in results['results']:
                doc = {
                    'concept': concept,
                    'title': r.get('title', ''),
                    'content': r.get('content', ''),
                    'url': r.get('url', ''),
                    'score': r.get('score', 0)
                }
                all_results.append(doc)
        else:
            print(f"  ✗ Нет результатов")
        
        # Пауза между запросами чтобы не перегружать API
        time.sleep(2)
    
    # Сохраняем результаты локально
    output_file = Path("knowledge_base_external.json")
    with open(output_file, 'w') as f:
        json.dump({
            'concepts': concepts_to_search,
            'documents': all_results,
            'total': len(all_results)
        }, f, indent=2)
    
    print(f"\n✅ Сохранено {len(all_results)} документов в {output_file}")
    
    # TODO: Здесь должна быть отправка в Knowledge Manager через API
    # Но сначала нужно разобраться с правильными эндпоинтами
    
    return all_results

def main():
    # Путь к датасету
    dataset_path = "../data/apqc_auto.json"
    
    # API ключи
    tavily_key = "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"
    
    # 1. Извлекаем ключевые концепты
    concepts = extract_key_concepts(dataset_path)
    
    # 2. ВДУМЧИВО заполняем Knowledge Manager
    results = populate_knowledge_manager(concepts, tavily_key)
    
    print("\n" + "="*60)
    print("ИТОГИ:")
    print(f"- Проанализировано концептов: {len(concepts)}")
    print(f"- Использовано для поиска: 5 (экономим API)")
    print(f"- Получено документов: {len(results)}")
    print("="*60)

if __name__ == "__main__":
    main()