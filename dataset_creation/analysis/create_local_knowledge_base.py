#!/usr/bin/env python3
"""
Создаём локальную базу знаний из нашего датасета
БЕЗ внешних API - только наши 706 Q&A
"""

import json
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

class LocalKnowledgeBase:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Лёгкая модель
        self.index = None
        self.documents = []
        
    def build_index(self):
        """Строим векторный индекс из датасета"""
        
        print("Загружаем датасет...")
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        print(f"Создаём индекс из {len(data['questions'])} Q&A пар...")
        
        # Подготавливаем документы
        for item in data['questions']:
            # Комбинируем вопрос и ответ для полного контекста
            doc = {
                'id': item['id'],
                'text': f"Question: {item['question']}\nAnswer: {item['answer']}",
                'category': item['category'],
                'question': item['question'],
                'answer': item['answer']
            }
            self.documents.append(doc)
        
        # Создаём эмбеддинги
        print("Создаём эмбеддинги...")
        texts = [d['text'] for d in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Строим FAISS индекс
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Индекс создан: {len(self.documents)} документов")
        
        # Сохраняем
        self.save_index()
        
    def save_index(self):
        """Сохраняем индекс для повторного использования"""
        
        # Сохраняем FAISS индекс
        faiss.write_index(self.index, "local_kb_index.faiss")
        
        # Сохраняем документы
        with open("local_kb_docs.json", 'w') as f:
            json.dump(self.documents, f)
        
        print("✅ Индекс сохранён")
    
    def load_index(self):
        """Загружаем сохранённый индекс"""
        
        if Path("local_kb_index.faiss").exists():
            self.index = faiss.read_index("local_kb_index.faiss")
            with open("local_kb_docs.json", 'r') as f:
                self.documents = json.load(f)
            print("✅ Индекс загружен")
            return True
        return False
    
    def vector_search(self, query: str, top_k: int = 4) -> List[Dict]:
        """Векторный поиск"""
        
        # Эмбеддинг запроса
        query_embedding = self.model.encode([query])
        
        # Поиск
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'id': doc['id'],
                    'text': doc['answer'][:500],  # Ограничиваем размер
                    'score': float(1.0 / (1.0 + dist)),  # Преобразуем расстояние в score
                    'type': 'vector'
                })
        
        return results
    
    def graph_search(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        Псевдо-графовый поиск
        Находим похожие вопросы и их "соседей" по категории
        """
        
        # Сначала векторный поиск
        initial_results = self.vector_search(query, top_k=2)
        
        results = []
        for res in initial_results:
            # Находим документ
            doc_id = res['id']
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            
            if doc:
                # Добавляем сам документ
                results.append(res)
                
                # Находим "соседей" той же категории
                category = doc['category']
                neighbors = [
                    d for d in self.documents 
                    if d['category'] == category and d['id'] != doc_id
                ][:2]
                
                for n in neighbors:
                    results.append({
                        'id': n['id'],
                        'text': n['answer'][:500],
                        'score': res['score'] * 0.8,  # Меньший score для соседей
                        'type': 'graph'
                    })
        
        return results[:top_k]
    
    def hybrid_search(self, query: str, category: str, top_k: int = 4) -> List[Dict]:
        """Гибридный поиск с весами по категории"""
        
        # Веса из конфига
        weights = {
            'causal': {'graph': 0.7, 'vector': 0.3},
            'diagnostic': {'graph': 0.7, 'vector': 0.3},
            'factual': {'graph': 0.3, 'vector': 0.7},
            'comparative': {'graph': 0.4, 'vector': 0.6}
        }
        
        w = weights.get(category, {'graph': 0.5, 'vector': 0.5})
        
        # Получаем результаты обоих методов
        vector_results = self.vector_search(query, top_k)
        graph_results = self.graph_search(query, top_k)
        
        # Объединяем с весами
        combined = {}
        
        for r in vector_results:
            combined[r['id']] = r['score'] * w['vector']
        
        for r in graph_results:
            if r['id'] in combined:
                combined[r['id']] += r['score'] * w['graph']
            else:
                combined[r['id']] = r['score'] * w['graph']
        
        # Сортируем и возвращаем топ-k
        sorted_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_id, score in sorted_ids:
            doc = next((d for d in self.documents if d['id'] == doc_id), None)
            if doc:
                results.append({
                    'id': doc_id,
                    'text': doc['answer'][:500],
                    'score': score,
                    'type': 'hybrid'
                })
        
        return results

def main():
    # Создаём базу знаний
    kb = LocalKnowledgeBase("../data/apqc_auto.json")
    
    # Проверяем, есть ли сохранённый индекс
    if not kb.load_index():
        kb.build_index()
    
    # Тестируем
    test_query = "Why do brakes squeal?"
    
    print("\n" + "="*60)
    print("ТЕСТ ЛОКАЛЬНОЙ БАЗЫ ЗНАНИЙ")
    print("="*60)
    print(f"Запрос: {test_query}\n")
    
    # Vector search
    print("Vector Search:")
    for r in kb.vector_search(test_query, top_k=2):
        print(f"  Score: {r['score']:.3f} | {r['text'][:100]}...")
    
    # Graph search
    print("\nGraph Search:")
    for r in kb.graph_search(test_query, top_k=2):
        print(f"  Score: {r['score']:.3f} | {r['text'][:100]}...")
    
    # Hybrid search
    print("\nHybrid Search:")
    for r in kb.hybrid_search(test_query, 'causal', top_k=2):
        print(f"  Score: {r['score']:.3f} | {r['text'][:100]}...")

if __name__ == "__main__":
    main()