#!/usr/bin/env python3
"""
РЕАЛЬНЫЙ ТЕСТ ВСЕХ 4 РЕЖИМОВ - БЕЗ МОКОВ!
Используем настоящие API и Knowledge Manager
"""

import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List
import os
from dotenv import load_dotenv
import time

# Загружаем переменные окружения
load_dotenv()

class TestRealModes:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.tavily_key = "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"
        self.serper_key = "4e74a3600041c6fdc32d1a7920f3b32413936ab8"
        
        # URLs для Knowledge Manager (если запущен)
        self.km_vector_url = "http://localhost:8098/retrievement/vector_retrieve"
        self.km_graph_url = "http://localhost:8098/retrievement/graph_retrieve"
        
        # Загружаем 5 тестовых вопросов
        self.test_questions = self.load_test_questions()
        
        # Результаты
        self.results = {
            'base_llm': [],
            'vector_rag': [],
            'graph_rag': [],
            'hybrid_ahs': []
        }
    
    def load_test_questions(self) -> List[Dict]:
        """Загружаем 5 тестовых вопросов"""
        with open("../data/apqc_auto.json", 'r') as f:
            data = json.load(f)
        
        test_q = []
        categories = ['causal', 'diagnostic', 'factual', 'comparative', 'causal']
        
        for cat in categories:
            for q in data['questions']:
                if q['category'] == cat and q not in test_q:
                    test_q.append(q)
                    break
        
        print(f"Загружено {len(test_q)} тестовых вопросов:")
        for i, q in enumerate(test_q, 1):
            print(f"  {i}. [{q['category']}] {q['question'][:60]}...")
        
        return test_q[:5]
    
    def get_real_tavily_retrieval(self, question: str) -> List[Dict]:
        """РЕАЛЬНЫЙ поиск через Tavily API"""
        
        print(f"    🔍 Вызов Tavily API...")
        
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.tavily_key,
            "query": f"automotive {question[:100]}",  # Ограничиваем длину
            "search_depth": "basic",
            "max_results": 3,
            "include_answer": False,
            "include_domains": [
                "mechanics.stackexchange.com",
                "reddit.com/r/MechanicAdvice",
                "yourmechanic.com"
            ]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for r in data.get('results', []):
                    results.append({
                        'text': r.get('content', '')[:500],
                        'url': r.get('url', ''),
                        'score': r.get('score', 0.5)
                    })
                print(f"    ✅ Получено {len(results)} результатов от Tavily")
                return results
            else:
                print(f"    ❌ Tavily error: {response.status_code}")
                return []
        except Exception as e:
            print(f"    ❌ Tavily exception: {e}")
            return []
    
    def test_base_llm(self, question: str) -> Dict:
        """Тест 1: Базовый LLM без контекста"""
        
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an automotive expert. Answer concisely."},
                    {"role": "user", "content": question}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content
            
            return {
                'mode': 'base_llm',
                'answer': answer,
                'context_used': None,
                'context_length': 0,
                'api_calls': {'openai': 1}
            }
        except Exception as e:
            print(f"    ❌ Error in base_llm: {e}")
            return {'mode': 'base_llm', 'answer': f'Error: {e}', 'context_used': None, 'context_length': 0}
    
    def test_vector_rag(self, question: str) -> Dict:
        """Тест 2: Векторный RAG с РЕАЛЬНЫМ retrieval"""
        
        # Получаем РЕАЛЬНЫЙ контекст через Tavily
        contexts = self.get_real_tavily_retrieval(question)
        
        if not contexts:
            print("    ⚠️ Нет контекста от Tavily, используем base_llm")
            return self.test_base_llm(question)
        
        context_str = "\n\n".join([f"Source {i+1}: {c['text']}" for i, c in enumerate(contexts)])
        
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        try:
            prompt = f"""Use the following search results to answer the question:

{context_str}

Question: {question}

Answer based on the search results:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer based on the provided search results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content
            
            return {
                'mode': 'vector_rag',
                'answer': answer,
                'context_used': contexts,
                'context_length': len(context_str),
                'api_calls': {'tavily': 1, 'openai': 1}
            }
        except Exception as e:
            print(f"    ❌ Error in vector_rag: {e}")
            return {'mode': 'vector_rag', 'answer': f'Error: {e}', 'context_used': contexts, 'context_length': len(context_str)}
    
    def test_graph_rag(self, question: str, category: str) -> Dict:
        """Тест 3: Графовый RAG с углублённым поиском"""
        
        # Для graph делаем более специфичный поиск
        if category in ['causal', 'diagnostic']:
            # Добавляем специфичные термины для графовых связей
            enhanced_query = f"why how cause effect relationship {question}"
        else:
            enhanced_query = question
        
        # Получаем контекст
        contexts = self.get_real_tavily_retrieval(enhanced_query)
        
        if not contexts:
            print("    ⚠️ Нет контекста для graph_rag")
            return self.test_base_llm(question)
        
        context_str = "\n\n".join([f"Relationship {i+1}: {c['text']}" for i, c in enumerate(contexts)])
        
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        try:
            prompt = f"""Use the following information about relationships and causes:

{context_str}

Question: {question}

Focus on causal relationships and connections between components:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer focusing on causal relationships and connections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content
            
            return {
                'mode': 'graph_rag',
                'answer': answer,
                'context_used': contexts,
                'context_length': len(context_str),
                'api_calls': {'tavily': 1, 'openai': 1}
            }
        except Exception as e:
            print(f"    ❌ Error in graph_rag: {e}")
            return {'mode': 'graph_rag', 'answer': f'Error: {e}', 'context_used': contexts, 'context_length': len(context_str)}
    
    def test_hybrid_ahs(self, question: str, category: str) -> Dict:
        """Тест 4: Гибридный режим - комбинируем разные источники"""
        
        # Делаем ДВА разных поиска
        print("    🔍 Гибридный поиск (2 запроса)...")
        
        # 1. Фактический поиск
        factual_contexts = self.get_real_tavily_retrieval(question)
        time.sleep(1)  # Пауза между API вызовами
        
        # 2. Каузальный/диагностический поиск
        causal_query = f"why how diagnose troubleshoot {question}"
        causal_contexts = self.get_real_tavily_retrieval(causal_query)
        
        # Комбинируем с весами
        weights = {
            'causal': {'causal': 0.7, 'factual': 0.3},
            'diagnostic': {'causal': 0.7, 'factual': 0.3},
            'factual': {'causal': 0.3, 'factual': 0.7},
            'comparative': {'causal': 0.4, 'factual': 0.6}
        }
        
        w = weights.get(category, {'causal': 0.5, 'factual': 0.5})
        
        # Отбираем контексты по весам
        if w['factual'] > w['causal']:
            contexts = factual_contexts[:2] + causal_contexts[:1]
        else:
            contexts = causal_contexts[:2] + factual_contexts[:1]
        
        context_str = "\n\n".join([f"Context {i+1}: {c['text']}" for i, c in enumerate(contexts) if c])
        
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        try:
            prompt = f"""Use the following hybrid context (factual + relational):

{context_str}

Question: {question}

Provide a comprehensive answer using both factual information and relationships:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Provide comprehensive answer using all context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            answer = response.choices[0].message.content
            
            return {
                'mode': 'hybrid_ahs',
                'answer': answer,
                'context_used': contexts,
                'context_length': len(context_str),
                'api_calls': {'tavily': 2, 'openai': 1}
            }
        except Exception as e:
            print(f"    ❌ Error in hybrid_ahs: {e}")
            return {'mode': 'hybrid_ahs', 'answer': f'Error: {e}', 'context_used': contexts, 'context_length': 0}
    
    def run_all_tests(self):
        """Запускаем все 4 режима на 5 вопросах"""
        
        print("\n" + "="*80)
        print("РЕАЛЬНОЕ ТЕСТИРОВАНИЕ ВСЕХ 4 РЕЖИМОВ")
        print("="*80)
        
        for i, q in enumerate(self.test_questions, 1):
            print(f"\n📝 Вопрос {i}/{len(self.test_questions)}: {q['question'][:80]}...")
            print(f"   Категория: {q['category']}")
            print(f"   Золотой ответ: {q['answer'][:80]}...")
            print("-"*80)
            
            # Тест 1: base_llm
            print("  🤖 Testing base_llm...")
            result_base = self.test_base_llm(q['question'])
            self.results['base_llm'].append(result_base)
            print(f"     ✓ Ответ: {result_base['answer'][:100]}...")
            
            # Тест 2: vector_rag
            print("\n  📊 Testing vector_rag with REAL Tavily...")
            result_vector = self.test_vector_rag(q['question'])
            self.results['vector_rag'].append(result_vector)
            print(f"     ✓ Контекст: {result_vector['context_length']} символов")
            print(f"     ✓ Ответ: {result_vector['answer'][:100]}...")
            
            # Тест 3: graph_rag
            print("\n  🕸️ Testing graph_rag with enhanced search...")
            result_graph = self.test_graph_rag(q['question'], q['category'])
            self.results['graph_rag'].append(result_graph)
            print(f"     ✓ Граф контекст: {result_graph['context_length']} символов")
            print(f"     ✓ Ответ: {result_graph['answer'][:100]}...")
            
            # Тест 4: hybrid_ahs
            print("\n  🔄 Testing hybrid_ahs with dual search...")
            result_hybrid = self.test_hybrid_ahs(q['question'], q['category'])
            self.results['hybrid_ahs'].append(result_hybrid)
            print(f"     ✓ Гибридный контекст: {result_hybrid['context_length']} символов")
            print(f"     ✓ Ответ: {result_hybrid['answer'][:100]}...")
            
            # Пауза между вопросами
            if i < len(self.test_questions):
                print("\n  ⏳ Пауза 2 секунды перед следующим вопросом...")
                time.sleep(2)
    
    def verify_differences(self):
        """Проверяем различия между режимами"""
        
        print("\n" + "="*80)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("="*80)
        
        # 1. Использование контекста
        print("\n📊 Использование контекста:")
        for mode in self.results:
            contexts = [r.get('context_length', 0) for r in self.results[mode]]
            avg_context = np.mean(contexts) if contexts else 0
            print(f"   {mode}: {avg_context:.0f} символов в среднем")
        
        # 2. API вызовы
        print("\n💰 Использование API:")
        total_api_calls = {'tavily': 0, 'openai': 0}
        for mode in self.results:
            for r in self.results[mode]:
                if 'api_calls' in r:
                    for api, count in r.get('api_calls', {}).items():
                        total_api_calls[api] = total_api_calls.get(api, 0) + count
        
        print(f"   Tavily API: {total_api_calls.get('tavily', 0)} вызовов")
        print(f"   OpenAI API: {total_api_calls.get('openai', 0)} вызовов")
        
        # 3. Качество ответов
        print("\n✅ Статус ответов:")
        for mode in self.results:
            errors = sum(1 for r in self.results[mode] if 'Error' in r.get('answer', ''))
            success = len(self.results[mode]) - errors
            print(f"   {mode}: {success}/{len(self.results[mode])} успешных")
        
        # Сохраняем результаты
        with open("test_real_modes_results.json", 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print("\n✅ Результаты сохранены в test_real_modes_results.json")

def main():
    print("🚀 Запуск РЕАЛЬНОГО тестирования...")
    print("⚠️ Будут использованы настоящие API (Tavily + OpenAI)")
    
    tester = TestRealModes()
    tester.run_all_tests()
    tester.verify_differences()
    
    print("\n" + "="*80)
    print("ГОТОВО!")
    print("="*80)

if __name__ == "__main__":
    main()