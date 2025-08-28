#!/usr/bin/env python3
"""
Упрощённая версия для оценки 100 вопросов
"""

import json
import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Конфигурация
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_KEY = "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"

print("🚀 Запуск оценки первых 100 вопросов из датасета...")
print(f"⏰ Начало: {datetime.now().strftime('%H:%M:%S')}")

# Загружаем датасет
with open('../data/apqc_auto.json', 'r') as f:
    dataset = json.load(f)

print(f"📚 Загружено {len(dataset['questions'])} вопросов")

client = OpenAI(api_key=OPENAI_KEY)
results = []

def get_tavily_context(query):
    """Простой запрос к Tavily"""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_KEY,
                "query": f"automotive {query[:100]}",
                "search_depth": "basic",
                "max_results": 2
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return [r.get('content', '')[:200] for r in data.get('results', [])]
    except:
        pass
    return []

# Обрабатываем первые 100 вопросов
for i, q in enumerate(dataset['questions'][:100], 1):
    print(f"\n📝 Вопрос {i}/100 (ID: {q['id']})")
    print(f"   {q['question'][:60]}...")
    
    result = {
        "id": q['id'],
        "question": q['question'],
        "category": q['category'],
        "gold_answer": q['answer']
    }
    
    try:
        # 1. Base LLM
        print("   🤖 base_llm...", end="")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an automotive expert. Answer concisely."},
                {"role": "user", "content": q['question']}
            ],
            temperature=0,
            max_tokens=150
        )
        result['base_llm'] = {
            "answer": response.choices[0].message.content,
            "context_size": 0
        }
        print(" ✓")
        
        # 2. Vector RAG
        print("   📊 vector_rag...", end="")
        contexts = get_tavily_context(q['question'])
        context_str = "\n".join(contexts) if contexts else "No context"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer based on context."},
                {"role": "user", "content": f"Context: {context_str}\n\nQuestion: {q['question']}"}
            ],
            temperature=0,
            max_tokens=150
        )
        result['vector_rag'] = {
            "answer": response.choices[0].message.content,
            "context_size": len(context_str)
        }
        print(" ✓")
        
        # 3. Graph RAG
        print("   🕸️ graph_rag...", end="")
        causal_query = f"why cause {q['question']}" if q['category'] in ['causal', 'diagnostic'] else q['question']
        contexts = get_tavily_context(causal_query)
        context_str = "\n".join(contexts) if contexts else "No context"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Focus on causal relationships."},
                {"role": "user", "content": f"Context: {context_str}\n\nQuestion: {q['question']}"}
            ],
            temperature=0,
            max_tokens=150
        )
        result['graph_rag'] = {
            "answer": response.choices[0].message.content,
            "context_size": len(context_str)
        }
        print(" ✓")
        
        # 4. Hybrid
        print("   🔄 hybrid...", end="")
        contexts1 = get_tavily_context(q['question'])
        time.sleep(0.5)
        contexts2 = get_tavily_context(f"why {q['question']}")
        all_contexts = contexts1[:1] + contexts2[:1]
        context_str = "\n".join(all_contexts) if all_contexts else "No context"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Comprehensive answer."},
                {"role": "user", "content": f"Context: {context_str}\n\nQuestion: {q['question']}"}
            ],
            temperature=0,
            max_tokens=150
        )
        result['hybrid_ahs'] = {
            "answer": response.choices[0].message.content,
            "context_size": len(context_str)
        }
        print(" ✓")
        
        results.append(result)
        
        # Сохраняем каждые 10 вопросов
        if i % 10 == 0:
            with open('eval_100_checkpoint.json', 'w') as f:
                json.dump({"count": i, "results": results}, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Сохранено {i} вопросов")
            print(f"⏱️ Время: {datetime.now().strftime('%H:%M:%S')}")
        
        # Пауза между вопросами
        time.sleep(1)
        
    except Exception as e:
        print(f" ❌ Ошибка: {e}")
        continue

# Сохраняем финальные результаты
output_file = f"eval_100_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("✅ ГОТОВО!")
print(f"📁 Результаты сохранены в {output_file}")
print(f"📊 Обработано вопросов: {len(results)}")
print(f"⏰ Конец: {datetime.now().strftime('%H:%M:%S')}")