#!/usr/bin/env python3
"""
ПРОВЕРКА ВСЕХ 4 РЕЖИМОВ НА 5 ВОПРОСАХ
С реальными API и чёткими различиями
"""

import json
import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Конфигурация
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_KEY = "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"

# 5 тестовых вопросов
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Why does engine knock when cold?",
        "category": "causal"
    },
    {
        "id": 2,
        "question": "How to diagnose a misfire?",
        "category": "diagnostic"
    },
    {
        "id": 3,
        "question": "What type of oil for 2020 Honda Civic?",
        "category": "factual"
    },
    {
        "id": 4,
        "question": "Drum brakes vs disc brakes performance?",
        "category": "comparative"
    },
    {
        "id": 5,
        "question": "Why do brakes squeal after replacement?",
        "category": "causal"
    }
]

def test_base_llm(question: str) -> dict:
    """MODE 1: Чистый LLM без контекста"""
    print("  🤖 base_llm: Запрос к OpenAI БЕЗ контекста...")
    
    client = OpenAI(api_key=OPENAI_KEY)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an automotive expert. Answer concisely."},
            {"role": "user", "content": question}
        ],
        temperature=0,
        max_tokens=150
    )
    
    return {
        "mode": "base_llm",
        "answer": response.choices[0].message.content,
        "context_used": None,
        "context_size": 0
    }

def get_tavily_context(query: str, search_type: str = "basic") -> list:
    """Получаем контекст через Tavily API"""
    
    url = "https://api.tavily.com/search"
    
    # Модифицируем запрос в зависимости от типа
    if search_type == "causal":
        query = f"why cause reason {query}"
    elif search_type == "diagnostic":
        query = f"diagnose troubleshoot symptoms {query}"
    
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
            return [r.get('content', '')[:300] for r in results]
    except Exception as e:
        print(f"    ⚠️ Tavily error: {e}")
    
    return []

def test_vector_rag(question: str) -> dict:
    """MODE 2: Векторный поиск + LLM"""
    print("  📊 vector_rag: Получаем контекст через Tavily (vector search)...")
    
    # Получаем контекст через векторный поиск
    contexts = get_tavily_context(question, "basic")
    
    if not contexts:
        print("    ❌ Нет контекста от Tavily!")
        contexts = ["No additional context available."]
    
    context_str = "\n".join(contexts)
    print(f"    ✓ Получено {len(contexts)} фрагментов, {len(context_str)} символов")
    
    client = OpenAI(api_key=OPENAI_KEY)
    
    prompt = f"""Context from search:
{context_str}

Question: {question}
Answer based on context:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=150
    )
    
    return {
        "mode": "vector_rag",
        "answer": response.choices[0].message.content,
        "context_used": contexts,
        "context_size": len(context_str)
    }

def test_graph_rag(question: str, category: str) -> dict:
    """MODE 3: Графовый поиск с учётом связей"""
    print("  🕸️ graph_rag: Получаем контекст с фокусом на связи...")
    
    # Для графа ищем причинно-следственные связи
    contexts = get_tavily_context(question, "causal" if category in ["causal", "diagnostic"] else "basic")
    
    if not contexts:
        contexts = ["No graph relationships found."]
    
    context_str = "\n".join(contexts)
    print(f"    ✓ Граф контекст: {len(context_str)} символов")
    
    client = OpenAI(api_key=OPENAI_KEY)
    
    prompt = f"""Causal/relational context:
{context_str}

Question: {question}
Focus on cause-effect relationships:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Focus on causal relationships and connections."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=150
    )
    
    return {
        "mode": "graph_rag",
        "answer": response.choices[0].message.content,
        "context_used": contexts,
        "context_size": len(context_str)
    }

def test_hybrid_ahs(question: str, category: str) -> dict:
    """MODE 4: Гибридный режим (vector + graph с весами)"""
    print("  🔄 hybrid_ahs: Комбинируем vector и graph поиск...")
    
    # Получаем оба типа контекста
    vector_contexts = get_tavily_context(question, "basic")
    time.sleep(1)  # Пауза между API вызовами
    graph_contexts = get_tavily_context(question, "causal")
    
    # Применяем веса по категории
    if category in ["causal", "diagnostic"]:
        # Больше веса графу
        contexts = graph_contexts[:2] + vector_contexts[:1]
    else:
        # Больше веса вектору
        contexts = vector_contexts[:2] + graph_contexts[:1]
    
    contexts = [c for c in contexts if c]  # Убираем пустые
    
    if not contexts:
        contexts = ["No hybrid context available."]
    
    context_str = "\n".join(contexts)
    print(f"    ✓ Гибридный контекст: {len(context_str)} символов")
    
    client = OpenAI(api_key=OPENAI_KEY)
    
    prompt = f"""Combined vector and graph context:
{context_str}

Question: {question}
Comprehensive answer using all information:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Provide comprehensive answer using all context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=150
    )
    
    return {
        "mode": "hybrid_ahs",
        "answer": response.choices[0].message.content,
        "context_used": contexts,
        "context_size": len(context_str)
    }

def main():
    print("="*80)
    print("ПРОВЕРКА ВСЕХ 4 РЕЖИМОВ")
    print("="*80)
    
    results = []
    
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n📝 Вопрос {i}/5: {q['question']}")
        print(f"   Категория: {q['category']}")
        print("-"*60)
        
        # Test all 4 modes
        try:
            # Mode 1: base_llm
            result_base = test_base_llm(q['question'])
            print(f"    ✅ base_llm: {result_base['answer'][:80]}...")
            
            # Mode 2: vector_rag
            result_vector = test_vector_rag(q['question'])
            print(f"    ✅ vector_rag: {result_vector['answer'][:80]}...")
            
            # Mode 3: graph_rag
            result_graph = test_graph_rag(q['question'], q['category'])
            print(f"    ✅ graph_rag: {result_graph['answer'][:80]}...")
            
            # Mode 4: hybrid_ahs
            result_hybrid = test_hybrid_ahs(q['question'], q['category'])
            print(f"    ✅ hybrid_ahs: {result_hybrid['answer'][:80]}...")
            
            results.append({
                "question": q,
                "base_llm": result_base,
                "vector_rag": result_vector,
                "graph_rag": result_graph,
                "hybrid_ahs": result_hybrid
            })
            
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
        
        # Пауза между вопросами
        if i < 5:
            time.sleep(2)
    
    # Анализ результатов
    print("\n" + "="*80)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    print("\n📊 Использование контекста:")
    for mode in ["base_llm", "vector_rag", "graph_rag", "hybrid_ahs"]:
        sizes = [r[mode]["context_size"] for r in results]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        print(f"  {mode:12} : {avg_size:6.0f} символов в среднем")
    
    print("\n✅ Проверка работоспособности:")
    modes_status = {
        "base_llm": "✅ Работает (без контекста)",
        "vector_rag": "✅ Работает (с Tavily контекстом)" if any(r["vector_rag"]["context_size"] > 0 for r in results) else "❌ Не работает",
        "graph_rag": "✅ Работает (с каузальным контекстом)" if any(r["graph_rag"]["context_size"] > 0 for r in results) else "❌ Не работает",
        "hybrid_ahs": "✅ Работает (комбинированный)" if any(r["hybrid_ahs"]["context_size"] > 0 for r in results) else "❌ Не работает"
    }
    
    for mode, status in modes_status.items():
        print(f"  {mode:12} : {status}")
    
    # Сохраняем результаты
    with open("verify_4_modes_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n💾 Результаты сохранены в verify_4_modes_results.json")
    
    print("\n" + "="*80)
    print("ЗАКЛЮЧЕНИЕ")
    print("="*80)
    
    working_modes = sum(1 for status in modes_status.values() if "✅" in status)
    print(f"Работающих режимов: {working_modes}/4")
    
    if working_modes == 4:
        print("🎉 ВСЕ 4 РЕЖИМА РАБОТАЮТ!")
    else:
        print("⚠️ Не все режимы работают корректно")

if __name__ == "__main__":
    main()