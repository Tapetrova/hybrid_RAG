#!/usr/bin/env python3
"""
Оценка первых 200 РЕАЛЬНЫХ вопросов из датасета apqc_auto.json
Всеми 4 методами с сохранением всех параметров
"""

import json
import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import traceback

load_dotenv()

# Конфигурация
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_KEY = "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"

class Eval200Questions:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.start_time = datetime.now()
        
        # Загружаем датасет
        with open('../data/apqc_auto.json', 'r') as f:
            self.dataset = json.load(f)
        
        print(f"📚 Загружен датасет: {len(self.dataset['questions'])} вопросов")
        
        # Файл для сохранения результатов
        self.output_file = f"eval_200_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.checkpoint_file = "eval_200_checkpoint.json"
        
        # Загружаем checkpoint если есть
        self.results = []
        self.processed_ids = set()
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.results = checkpoint.get('results', [])
                self.processed_ids = set(checkpoint.get('processed_ids', []))
                print(f"✅ Загружен checkpoint: уже обработано {len(self.processed_ids)} вопросов")
    
    def save_checkpoint(self):
        """Сохраняем промежуточные результаты"""
        checkpoint = {
            'results': self.results,
            'processed_ids': list(self.processed_ids),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    def get_tavily_context(self, query: str, search_type: str = "basic") -> list:
        """Получаем контекст через Tavily API"""
        
        url = "https://api.tavily.com/search"
        
        # Модифицируем запрос в зависимости от типа
        if search_type == "causal":
            query = f"why cause reason {query}"
        elif search_type == "diagnostic":
            query = f"diagnose troubleshoot symptoms {query}"
        
        # Ограничиваем длину запроса
        query = query[:200]
        
        payload = {
            "api_key": TAVILY_KEY,
            "query": f"automotive {query}",
            "search_depth": "basic",
            "max_results": 2,
            "include_domains": ["mechanics.stackexchange.com", "reddit.com/r/MechanicAdvice"]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                contexts = []
                for r in results:
                    contexts.append({
                        'text': r.get('content', '')[:300],
                        'url': r.get('url', ''),
                        'score': r.get('score', 0.5)
                    })
                return contexts
        except Exception as e:
            print(f"      ⚠️ Tavily error: {e}")
        
        return []
    
    def test_base_llm(self, question: str) -> dict:
        """MODE 1: Чистый LLM без контекста"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an automotive expert. Answer concisely."},
                    {"role": "user", "content": question}
                ],
                temperature=0,
                max_tokens=200
            )
            
            return {
                "mode": "base_llm",
                "answer": response.choices[0].message.content,
                "context_used": None,
                "context_size": 0,
                "success": True
            }
        except Exception as e:
            print(f"      ❌ Error in base_llm: {e}")
            return {
                "mode": "base_llm",
                "answer": f"Error: {str(e)}",
                "context_used": None,
                "context_size": 0,
                "success": False
            }
    
    def test_vector_rag(self, question: str) -> dict:
        """MODE 2: Векторный поиск + LLM"""
        try:
            contexts = self.get_tavily_context(question, "basic")
            
            if not contexts:
                contexts = []
                context_str = "No additional context available."
            else:
                context_str = "\n".join([c['text'] for c in contexts])
            
            prompt = f"""Context from search:
{context_str}

Question: {question}
Answer based on context:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            
            return {
                "mode": "vector_rag",
                "answer": response.choices[0].message.content,
                "context_used": contexts,
                "context_size": len(context_str),
                "success": True
            }
        except Exception as e:
            print(f"      ❌ Error in vector_rag: {e}")
            return {
                "mode": "vector_rag",
                "answer": f"Error: {str(e)}",
                "context_used": [],
                "context_size": 0,
                "success": False
            }
    
    def test_graph_rag(self, question: str, category: str) -> dict:
        """MODE 3: Графовый поиск с учётом связей"""
        try:
            contexts = self.get_tavily_context(
                question, 
                "causal" if category in ["causal", "diagnostic"] else "basic"
            )
            
            if not contexts:
                contexts = []
                context_str = "No graph relationships found."
            else:
                context_str = "\n".join([c['text'] for c in contexts])
            
            prompt = f"""Causal/relational context:
{context_str}

Question: {question}
Focus on cause-effect relationships:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Focus on causal relationships and connections."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            
            return {
                "mode": "graph_rag",
                "answer": response.choices[0].message.content,
                "context_used": contexts,
                "context_size": len(context_str),
                "success": True
            }
        except Exception as e:
            print(f"      ❌ Error in graph_rag: {e}")
            return {
                "mode": "graph_rag",
                "answer": f"Error: {str(e)}",
                "context_used": [],
                "context_size": 0,
                "success": False
            }
    
    def test_hybrid_ahs(self, question: str, category: str) -> dict:
        """MODE 4: Гибридный режим"""
        try:
            # Два типа поиска
            vector_contexts = self.get_tavily_context(question, "basic")
            time.sleep(0.5)  # Небольшая пауза
            graph_contexts = self.get_tavily_context(question, "causal")
            
            # Комбинируем контексты
            if category in ["causal", "diagnostic"]:
                contexts = graph_contexts[:2] + vector_contexts[:1]
            else:
                contexts = vector_contexts[:2] + graph_contexts[:1]
            
            contexts = [c for c in contexts if c]
            
            if not contexts:
                contexts = []
                context_str = "No hybrid context available."
            else:
                context_str = "\n".join([c['text'] for c in contexts])
            
            prompt = f"""Combined vector and graph context:
{context_str}

Question: {question}
Comprehensive answer using all information:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Provide comprehensive answer using all context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            
            return {
                "mode": "hybrid_ahs",
                "answer": response.choices[0].message.content,
                "context_used": contexts,
                "context_size": len(context_str),
                "success": True
            }
        except Exception as e:
            print(f"      ❌ Error in hybrid_ahs: {e}")
            return {
                "mode": "hybrid_ahs",
                "answer": f"Error: {str(e)}",
                "context_used": [],
                "context_size": 0,
                "success": False
            }
    
    def process_question(self, q_data: dict, index: int, total: int) -> dict:
        """Обрабатываем один вопрос всеми методами"""
        
        q_id = q_data['id']
        question = q_data['question']
        category = q_data['category']
        gold_answer = q_data['answer']
        
        print(f"\n📝 Вопрос {index}/{total} (ID: {q_id})")
        print(f"   Вопрос: {question[:80]}...")
        print(f"   Категория: {category}")
        
        result = {
            "question_id": q_id,
            "question_text": question,
            "category": category,
            "gold_answer": gold_answer,
            "context": q_data.get('context', ''),
            "metadata": q_data.get('metadata', {}),
            "timestamp": datetime.now().isoformat()
        }
        
        # Тестируем все 4 метода
        print("   🤖 base_llm...", end="")
        result["base_llm"] = self.test_base_llm(question)
        print(" ✓")
        
        print("   📊 vector_rag...", end="")
        result["vector_rag"] = self.test_vector_rag(question)
        print(" ✓")
        
        print("   🕸️ graph_rag...", end="")
        result["graph_rag"] = self.test_graph_rag(question, category)
        print(" ✓")
        
        print("   🔄 hybrid_ahs...", end="")
        result["hybrid_ahs"] = self.test_hybrid_ahs(question, category)
        print(" ✓")
        
        return result
    
    def run_evaluation(self, limit: int = 200):
        """Запускаем оценку на первых N вопросах"""
        
        print("="*80)
        print(f"ОЦЕНКА ПЕРВЫХ {limit} ВОПРОСОВ ИЗ ДАТАСЕТА")
        print("="*80)
        
        questions_to_process = []
        for q in self.dataset['questions'][:limit]:
            if q['id'] not in self.processed_ids:
                questions_to_process.append(q)
        
        print(f"\n📊 К обработке: {len(questions_to_process)} вопросов")
        print(f"⏭️ Уже обработано: {len(self.processed_ids)} вопросов")
        
        for i, q_data in enumerate(questions_to_process, 1):
            try:
                # Обрабатываем вопрос
                result = self.process_question(
                    q_data, 
                    len(self.processed_ids) + 1, 
                    limit
                )
                
                # Сохраняем результат
                self.results.append(result)
                self.processed_ids.add(q_data['id'])
                
                # Сохраняем checkpoint каждые 5 вопросов
                if len(self.results) % 5 == 0:
                    self.save_checkpoint()
                    print(f"   💾 Checkpoint сохранён ({len(self.results)} вопросов)")
                
                # Показываем прогресс
                if len(self.results) % 10 == 0:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    avg_time = elapsed / len(self.results)
                    remaining = (limit - len(self.results)) * avg_time
                    print(f"\n⏱️ Прогресс: {len(self.results)}/{limit}")
                    print(f"   Среднее время на вопрос: {avg_time:.1f} сек")
                    print(f"   Осталось примерно: {remaining/60:.1f} мин")
                
                # Пауза между вопросами чтобы не перегружать API
                time.sleep(1)
                
            except Exception as e:
                print(f"\n❌ Ошибка при обработке вопроса {q_data['id']}: {e}")
                print(traceback.format_exc())
                continue
        
        # Финальное сохранение
        self.save_final_results()
    
    def save_final_results(self):
        """Сохраняем финальные результаты"""
        
        # Полные результаты
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Статистика
        stats = self.calculate_statistics()
        stats_file = self.output_file.replace('.json', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ СОХРАНЕНЫ")
        print("="*80)
        print(f"✅ Обработано вопросов: {len(self.results)}")
        print(f"📁 Результаты: {self.output_file}")
        print(f"📊 Статистика: {stats_file}")
        print(f"⏱️ Общее время: {(datetime.now() - self.start_time).total_seconds()/60:.1f} минут")
    
    def calculate_statistics(self) -> dict:
        """Вычисляем статистику"""
        
        stats = {
            "total_questions": len(self.results),
            "timestamp": datetime.now().isoformat(),
            "by_category": {},
            "by_mode": {
                "base_llm": {"success": 0, "errors": 0, "avg_length": 0},
                "vector_rag": {"success": 0, "errors": 0, "avg_context": 0},
                "graph_rag": {"success": 0, "errors": 0, "avg_context": 0},
                "hybrid_ahs": {"success": 0, "errors": 0, "avg_context": 0}
            }
        }
        
        # Подсчёт по категориям
        for r in self.results:
            cat = r['category']
            if cat not in stats['by_category']:
                stats['by_category'][cat] = 0
            stats['by_category'][cat] += 1
            
            # Подсчёт по режимам
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                if mode in r:
                    if r[mode].get('success', False):
                        stats['by_mode'][mode]['success'] += 1
                    else:
                        stats['by_mode'][mode]['errors'] += 1
        
        return stats

def main():
    print("🚀 Запуск оценки первых 100 вопросов из датасета...")
    print("⚠️ Это займёт примерно 20-30 минут")
    
    evaluator = Eval200Questions()
    evaluator.run_evaluation(limit=100)
    
    print("\n✅ ГОТОВО!")

if __name__ == "__main__":
    main()