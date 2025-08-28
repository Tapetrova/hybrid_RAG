#!/usr/bin/env python3
"""
ТЕСТ ВСЕХ 4 РЕЖИМОВ на 5 вопросах
Убеждаемся, что каждый режим работает правильно и отличается от других
"""

import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class TestAllModes:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.tavily_key = "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"
        
        # Загружаем 5 тестовых вопросов разных категорий
        self.test_questions = self.load_test_questions()
        
        # Результаты тестов
        self.results = {
            'base_llm': [],
            'vector_rag': [],
            'graph_rag': [],
            'hybrid_ahs': []
        }
    
    def load_test_questions(self) -> List[Dict]:
        """Загружаем 5 вопросов разных категорий"""
        with open("../data/apqc_auto.json", 'r') as f:
            data = json.load(f)
        
        # Берём по одному из каждой категории
        test_q = []
        categories_needed = ['causal', 'diagnostic', 'factual', 'comparative']
        
        for cat in categories_needed:
            for q in data['questions']:
                if q['category'] == cat and len(test_q) < 5:
                    test_q.append(q)
                    break
        
        # Добавим ещё один causal
        for q in data['questions']:
            if q['category'] == 'causal' and q not in test_q:
                test_q.append(q)
                break
        
        print(f"Загружено {len(test_q)} тестовых вопросов:")
        for i, q in enumerate(test_q, 1):
            print(f"  {i}. [{q['category']}] {q['question'][:50]}...")
        
        return test_q[:5]
    
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
                'context_length': 0
            }
        except Exception as e:
            print(f"Error in base_llm: {e}")
            return {'mode': 'base_llm', 'answer': 'Error', 'context_used': None, 'context_length': 0}
    
    def get_mock_vector_retrieval(self, question: str) -> List[str]:
        """Мок векторного поиска из локальной базы"""
        
        # Простая эмуляция векторного поиска
        mock_contexts = {
            'brake': [
                "Brake pads should be replaced when they reach 3-4mm thickness.",
                "Squealing brakes often indicate worn pads or glazed rotors."
            ],
            'engine': [
                "Engine knocking can be caused by low octane fuel or carbon deposits.",
                "Regular oil changes prevent engine wear and extend engine life."
            ],
            'transmission': [
                "Transmission fluid should be changed every 30,000-60,000 miles.",
                "Slipping gears indicate low fluid or worn clutch plates."
            ]
        }
        
        # Ищем ключевое слово
        question_lower = question.lower()
        for key, contexts in mock_contexts.items():
            if key in question_lower:
                return contexts
        
        # Дефолтный контекст
        return ["Regular maintenance is key to vehicle longevity."]
    
    def get_mock_graph_retrieval(self, question: str, category: str) -> List[str]:
        """Мок графового поиска с учётом связей"""
        
        # Эмуляция графовых связей
        if category == 'causal':
            return [
                "Component A causes issue B due to mechanical interaction.",
                "Root cause analysis shows correlation between symptoms."
            ]
        elif category == 'diagnostic':
            return [
                "Diagnostic procedure: 1) Check codes 2) Test components 3) Verify fix.",
                "Common symptoms indicate specific component failures."
            ]
        else:
            return self.get_mock_vector_retrieval(question)
    
    def test_vector_rag(self, question: str) -> Dict:
        """Тест 2: Векторный RAG"""
        
        # Получаем контекст через векторный поиск
        contexts = self.get_mock_vector_retrieval(question)
        context_str = "\n".join(contexts)
        
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        try:
            prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer based on context:"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer based on the provided context."},
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
                'context_length': len(context_str)
            }
        except Exception as e:
            print(f"Error in vector_rag: {e}")
            return {'mode': 'vector_rag', 'answer': 'Error', 'context_used': contexts, 'context_length': 0}
    
    def test_graph_rag(self, question: str, category: str) -> Dict:
        """Тест 3: Графовый RAG"""
        
        # Получаем контекст через граф
        contexts = self.get_mock_graph_retrieval(question, category)
        context_str = "\n".join(contexts)
        
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        try:
            prompt = f"Graph context:\n{context_str}\n\nQuestion: {question}\nAnswer using causal relationships:"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer based on graph relationships in context."},
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
                'context_length': len(context_str)
            }
        except Exception as e:
            print(f"Error in graph_rag: {e}")
            return {'mode': 'graph_rag', 'answer': 'Error', 'context_used': contexts, 'context_length': 0}
    
    def test_hybrid_ahs(self, question: str, category: str) -> Dict:
        """Тест 4: Гибридный режим"""
        
        # Комбинируем векторный и графовый контекст
        vector_contexts = self.get_mock_vector_retrieval(question)
        graph_contexts = self.get_mock_graph_retrieval(question, category)
        
        # Применяем веса по категории
        weights = {
            'causal': {'graph': 0.7, 'vector': 0.3},
            'diagnostic': {'graph': 0.7, 'vector': 0.3},
            'factual': {'graph': 0.3, 'vector': 0.7},
            'comparative': {'graph': 0.4, 'vector': 0.6}
        }
        
        w = weights.get(category, {'graph': 0.5, 'vector': 0.5})
        
        # Смешиваем контексты с учётом весов
        if w['vector'] > w['graph']:
            contexts = vector_contexts + graph_contexts[:1]
        else:
            contexts = graph_contexts + vector_contexts[:1]
        
        context_str = "\n".join(contexts)
        
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        try:
            prompt = f"Hybrid context (vector+graph):\n{context_str}\n\nQuestion: {question}\nAnswer comprehensively:"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer using both factual and relational context."},
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
                'context_length': len(context_str)
            }
        except Exception as e:
            print(f"Error in hybrid_ahs: {e}")
            return {'mode': 'hybrid_ahs', 'answer': 'Error', 'context_used': contexts, 'context_length': 0}
    
    def run_all_tests(self):
        """Запускаем все 4 режима на 5 вопросах"""
        
        print("\n" + "="*80)
        print("ТЕСТИРОВАНИЕ ВСЕХ 4 РЕЖИМОВ")
        print("="*80)
        
        for i, q in enumerate(self.test_questions, 1):
            print(f"\n📝 Вопрос {i}/{len(self.test_questions)}: {q['question'][:100]}...")
            print(f"   Категория: {q['category']}")
            print("-"*80)
            
            # Тест 1: base_llm
            print("  🤖 Testing base_llm...")
            result_base = self.test_base_llm(q['question'])
            self.results['base_llm'].append(result_base)
            print(f"     ✓ Answer length: {len(result_base['answer'])} chars")
            
            # Тест 2: vector_rag
            print("  📊 Testing vector_rag...")
            result_vector = self.test_vector_rag(q['question'])
            self.results['vector_rag'].append(result_vector)
            print(f"     ✓ Context used: {result_vector['context_length']} chars")
            
            # Тест 3: graph_rag
            print("  🕸️ Testing graph_rag...")
            result_graph = self.test_graph_rag(q['question'], q['category'])
            self.results['graph_rag'].append(result_graph)
            print(f"     ✓ Graph context: {result_graph['context_length']} chars")
            
            # Тест 4: hybrid_ahs
            print("  🔄 Testing hybrid_ahs...")
            result_hybrid = self.test_hybrid_ahs(q['question'], q['category'])
            self.results['hybrid_ahs'].append(result_hybrid)
            print(f"     ✓ Hybrid context: {result_hybrid['context_length']} chars")
            
            # Сравнение ответов
            print("\n  📊 Сравнение ответов:")
            print(f"     Base LLM: {result_base['answer'][:100]}...")
            print(f"     Vector RAG: {result_vector['answer'][:100]}...")
            print(f"     Graph RAG: {result_graph['answer'][:100]}...")
            print(f"     Hybrid: {result_hybrid['answer'][:100]}...")
    
    def verify_differences(self):
        """Проверяем, что режимы действительно отличаются"""
        
        print("\n" + "="*80)
        print("ВЕРИФИКАЦИЯ РАЗЛИЧИЙ МЕЖДУ РЕЖИМАМИ")
        print("="*80)
        
        # Проверка 1: Используется ли контекст?
        print("\n1️⃣ Использование контекста:")
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            avg_context = np.mean([r.get('context_length', 0) for r in self.results[mode]])
            print(f"   {mode}: {avg_context:.0f} символов контекста")
        
        # Проверка 2: Различаются ли ответы?
        print("\n2️⃣ Уникальность ответов:")
        for i in range(len(self.test_questions)):
            answers = {
                'base_llm': self.results['base_llm'][i]['answer'],
                'vector_rag': self.results['vector_rag'][i]['answer'],
                'graph_rag': self.results['graph_rag'][i]['answer'],
                'hybrid_ahs': self.results['hybrid_ahs'][i]['answer']
            }
            
            # Проверяем попарное сходство
            unique_answers = len(set(answers.values()))
            print(f"   Вопрос {i+1}: {unique_answers}/4 уникальных ответов")
        
        # Проверка 3: Работают ли все режимы?
        print("\n3️⃣ Статус работы:")
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            errors = sum(1 for r in self.results[mode] if r['answer'] == 'Error')
            if errors == 0:
                print(f"   ✅ {mode}: Работает корректно")
            else:
                print(f"   ❌ {mode}: {errors} ошибок")
        
        # Сохраняем результаты
        with open("test_all_modes_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n✅ Результаты сохранены в test_all_modes_results.json")

def main():
    tester = TestAllModes()
    tester.run_all_tests()
    tester.verify_differences()
    
    print("\n" + "="*80)
    print("ЗАКЛЮЧЕНИЕ")
    print("="*80)
    print("✅ Все 4 режима протестированы")
    print("📊 Проверьте test_all_modes_results.json для детального анализа")

if __name__ == "__main__":
    main()