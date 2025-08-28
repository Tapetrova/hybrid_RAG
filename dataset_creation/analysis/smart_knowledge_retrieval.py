#!/usr/bin/env python3
"""
УМНЫЙ И ЭКОНОМНЫЙ Knowledge Retrieval
Используем внешние API только когда это действительно нужно!
"""

import json
import hashlib
from typing import Dict, List, Optional
from pathlib import Path

class SmartKnowledgeRetrieval:
    def __init__(self, tavily_key: str, serper_key: str):
        self.tavily_key = tavily_key
        self.serper_key = serper_key
        
        # КЕШ для экономии API вызовов
        self.cache_file = Path("retrieval_cache.json")
        self.cache = self.load_cache()
        
        # Счётчики использования API
        self.api_calls = {
            'tavily': 0,
            'serper': 0,
            'cache_hits': 0
        }
    
    def load_cache(self) -> Dict:
        """Загружаем кеш предыдущих запросов"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_cache(self):
        """Сохраняем кеш"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get_query_hash(self, query: str) -> str:
        """Хеш запроса для кеширования"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def should_use_external_api(self, question: str, category: str) -> bool:
        """
        УМНОЕ РЕШЕНИЕ: нужен ли внешний поиск?
        Экономим токены!
        """
        
        # 1. Для простых фактических вопросов - НЕ НУЖЕН
        if category == 'factual' and len(question.split()) < 10:
            return False
        
        # 2. Для сложных каузальных - НУЖЕН
        if category == 'causal' and any(word in question.lower() for word in ['why', 'cause', 'reason']):
            return True
        
        # 3. Для диагностики сложных проблем - НУЖЕН
        if category == 'diagnostic' and any(word in question.lower() for word in ['diagnose', 'troubleshoot', 'multiple']):
            return True
        
        # 4. Для сравнительных вопросов - МОЖЕТ БЫТЬ ПОЛЕЗЕН
        if category == 'comparative':
            return True
        
        # По умолчанию - НЕ используем (экономим!)
        return False
    
    def retrieve_smart(self, question: str, category: str) -> List[Dict]:
        """
        УМНЫЙ retrieval с экономией API
        """
        
        # 1. Проверяем кеш
        query_hash = self.get_query_hash(question)
        if query_hash in self.cache:
            self.api_calls['cache_hits'] += 1
            print(f"  📦 Кеш хит! Экономим API вызов")
            return self.cache[query_hash]
        
        # 2. Решаем, нужен ли внешний поиск
        if not self.should_use_external_api(question, category):
            print(f"  💡 Внешний поиск не нужен для: {category}")
            return []  # Пустой retrieval - пусть base_llm справляется
        
        # 3. Используем API ТОЛЬКО если действительно нужно
        print(f"  🔍 Используем внешний поиск для: {category}")
        
        # Извлекаем ключевое слово для поиска
        keywords = self.extract_key_term(question)
        
        # Минимальный запрос к Tavily
        results = self.search_minimal(keywords)
        
        # Кешируем
        self.cache[query_hash] = results
        self.save_cache()
        
        return results
    
    def extract_key_term(self, question: str) -> str:
        """Извлекаем ОДНО ключевое слово для экономного поиска"""
        
        # Приоритетные автомобильные термины
        priority_terms = [
            'engine', 'brake', 'transmission', 'clutch', 'battery',
            'alternator', 'radiator', 'suspension', 'exhaust'
        ]
        
        question_lower = question.lower()
        for term in priority_terms:
            if term in question_lower:
                return term
        
        # Если не нашли - берём первое существительное
        words = question.split()
        for word in words:
            if len(word) > 4 and word.isalpha():
                return word
        
        return "car repair"  # Дефолт
    
    def search_minimal(self, keyword: str) -> List[Dict]:
        """
        МИНИМАЛЬНЫЙ поиск - экономим API!
        """
        import requests
        
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.tavily_key,
            "query": f"{keyword} automotive technical",
            "search_depth": "basic",  # Самый дешёвый
            "max_results": 2,  # Только 2 результата!
            "include_answer": False,  # Не нужен сгенерированный ответ
            "include_images": False,  # Не нужны картинки
            "include_domains": ["mechanics.stackexchange.com"]  # Только проверенный источник
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                self.api_calls['tavily'] += 1
                data = response.json()
                
                # Форматируем минимально
                results = []
                for r in data.get('results', [])[:2]:
                    results.append({
                        'text': r.get('content', '')[:300],  # Только 300 символов!
                        'score': 0.8,
                        'source': 'tavily'
                    })
                return results
        except:
            pass
        
        return []
    
    def get_stats(self) -> Dict:
        """Статистика использования API"""
        return {
            'tavily_calls': self.api_calls['tavily'],
            'serper_calls': self.api_calls['serper'],
            'cache_hits': self.api_calls['cache_hits'],
            'cache_size': len(self.cache),
            'money_saved': self.api_calls['cache_hits'] * 0.001  # ~$0.001 per API call
        }

# Интеграция с eval_runner.py
def get_smart_retrieval(question: str, category: str) -> List[Dict]:
    """
    Функция для использования в eval_runner
    """
    retriever = SmartKnowledgeRetrieval(
        tavily_key="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",
        serper_key="4e74a3600041c6fdc32d1a7920f3b32413936ab8"
    )
    
    return retriever.retrieve_smart(question, category)

if __name__ == "__main__":
    # Тест
    retriever = SmartKnowledgeRetrieval(
        tavily_key="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",
        serper_key="4e74a3600041c6fdc32d1a7920f3b32413936ab8"
    )
    
    test_questions = [
        ("What oil should I use?", "factual"),  # Не будет использовать API
        ("Why does engine knock?", "causal"),   # Будет использовать API
        ("How to diagnose multiple misfires?", "diagnostic"),  # Будет использовать API
    ]
    
    for q, cat in test_questions:
        print(f"\nВопрос: {q}")
        results = retriever.retrieve_smart(q, cat)
        print(f"Результатов: {len(results)}")
    
    print("\n" + "="*60)
    print("СТАТИСТИКА:")
    stats = retriever.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("="*60)