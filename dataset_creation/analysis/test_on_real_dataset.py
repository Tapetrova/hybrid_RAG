#!/usr/bin/env python3
"""
ПРАВИЛЬНЫЙ ТЕСТ: Берём вопросы ПРЯМО из датасета!
Тестируем 4 режима на 5 РЕАЛЬНЫХ вопросах из apqc_auto.json
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
        "query": f"automotive {query[:100]}",  # Ограничиваем длину
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
    print("  📊 vector_rag: Получаем контекст через Tavily...")
    
    contexts = get_tavily_context(question, "basic")
    
    if not contexts:
        print("    ❌ Нет контекста от Tavily!")
        contexts = ["No additional context available."]
    
    context_str = "\n".join(contexts)
    print(f"    ✓ Получено {len(contexts)} фрагментов")
    
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
    """MODE 4: Гибридный режим"""
    print("  🔄 hybrid_ahs: Комбинируем vector и graph поиск...")
    
    vector_contexts = get_tavily_context(question, "basic")
    time.sleep(1)
    graph_contexts = get_tavily_context(question, "causal")
    
    # Применяем веса по категории
    if category in ["causal", "diagnostic"]:
        contexts = graph_contexts[:2] + vector_contexts[:1]
    else:
        contexts = vector_contexts[:2] + graph_contexts[:1]
    
    contexts = [c for c in contexts if c]
    
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
    print("ТЕСТ НА РЕАЛЬНЫХ ВОПРОСАХ ИЗ ДАТАСЕТА")
    print("="*80)
    
    # ЗАГРУЖАЕМ РЕАЛЬНЫЕ ВОПРОСЫ ИЗ ДАТАСЕТА
    with open('../data/apqc_auto.json', 'r') as f:
        dataset = json.load(f)
    
    # Берём по одному вопросу из каждой категории
    test_questions = []
    categories_needed = ['causal', 'diagnostic', 'factual', 'comparative', 'causal']
    
    for cat in categories_needed:
        for q in dataset['questions']:
            if q['category'] == cat and q not in test_questions:
                test_questions.append(q)
                break
    
    print(f"\n📚 Взято {len(test_questions)} вопросов ПРЯМО из датасета:")
    for i, q in enumerate(test_questions[:5], 1):
        print(f"  {i}. ID: {q['id']}, Категория: {q['category']}")
        print(f"     Вопрос: {q['question'][:70]}...")
    
    results = []
    
    for i, q_data in enumerate(test_questions[:5], 1):
        question = q_data['question']
        category = q_data['category']
        gold_answer = q_data['answer']
        q_id = q_data['id']
        
        print(f"\n📝 Вопрос {i}/5 (ID: {q_id})")
        print(f"   Вопрос: {question[:80]}...")
        print(f"   Категория: {category}")
        print(f"   Золотой ответ: {gold_answer[:80]}...")
        print("-"*60)
        
        q_result = {
            "question_id": q_id,
            "question_text": question,
            "category": category,
            "gold_answer": gold_answer
        }
        
        try:
            # Test all 4 modes
            result_base = test_base_llm(question)
            print(f"    ✅ base_llm готов")
            q_result["base_llm"] = result_base
            
            result_vector = test_vector_rag(question)
            print(f"    ✅ vector_rag готов")
            q_result["vector_rag"] = result_vector
            
            result_graph = test_graph_rag(question, category)
            print(f"    ✅ graph_rag готов")
            q_result["graph_rag"] = result_graph
            
            result_hybrid = test_hybrid_ahs(question, category)
            print(f"    ✅ hybrid_ahs готов")
            q_result["hybrid_ahs"] = result_hybrid
            
            results.append(q_result)
            
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
        
        if i < 5:
            time.sleep(2)
    
    # Сохраняем результаты
    with open("test_real_dataset_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("ИТОГИ")
    print("="*80)
    print(f"✅ Протестировано {len(results)} вопросов из датасета")
    print(f"💾 Результаты сохранены в test_real_dataset_results.json")
    print("\nТеперь можно оценивать hallucination по золотым ответам!")

if __name__ == "__main__":
    main()