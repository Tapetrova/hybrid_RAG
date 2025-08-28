#!/usr/bin/env python3
"""
Построение автомобильного графа знаний по принципам:
1. Каузальные связи (причина-следствие)
2. Диагностические связи (симптом-проблема)
3. Иерархические связи (часть-целое)
4. Процедурные связи (шаг1-шаг2)
"""

import json
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import networkx as nx

class AutomotiveKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
        # Типы узлов
        self.node_types = {
            'component': set(),     # Компоненты авто
            'problem': set(),       # Проблемы/неисправности
            'symptom': set(),       # Симптомы
            'action': set(),        # Действия/ремонт
            'cause': set(),         # Причины
            'tool': set(),          # Инструменты
            'fluid': set()          # Жидкости/расходники
        }
        
        # Типы связей
        self.edge_types = {
            'CAUSES': [],           # X вызывает Y
            'INDICATES': [],        # X указывает на Y
            'REQUIRES': [],         # X требует Y
            'PART_OF': [],          # X часть Y
            'PREVENTS': [],         # X предотвращает Y
            'FOLLOWED_BY': [],      # X следует за Y (процедуры)
            'LOCATED_IN': [],       # X находится в Y
            'CONNECTS_TO': []       # X соединён с Y
        }
    
    def extract_entities_from_qa(self, question: str, answer: str, category: str) -> Dict:
        """Извлекаем сущности из пары вопрос-ответ"""
        
        entities = {
            'components': [],
            'problems': [],
            'symptoms': [],
            'actions': [],
            'causes': []
        }
        
        # Паттерны для извлечения сущностей
        component_patterns = [
            r'\b(engine|brake|transmission|battery|alternator|starter|radiator|pump|filter|belt|rotor|pad|caliper|clutch|axle|suspension|exhaust|catalytic converter|muffler|turbo|supercharger|ECU|sensor|thermostat|gasket|piston|camshaft|crankshaft|valve|bearing|seal|hose|wire|fuse|relay)\b',
        ]
        
        problem_patterns = [
            r'\b(leak|noise|vibration|overheating|failure|damage|wear|crack|break|malfunction|issue|problem|fault|error)\b',
        ]
        
        symptom_patterns = [
            r'\b(smoke|smell|sound|grinding|squealing|knocking|ticking|rough idle|misfire|stalling|hesitation|pulling|shimmy|check engine light)\b',
        ]
        
        action_patterns = [
            r'\b(replace|repair|inspect|check|test|diagnose|adjust|clean|flush|bleed|tighten|loosen|remove|install)\b',
        ]
        
        text = (question + " " + answer).lower()
        
        # Извлекаем компоненты
        for pattern in component_patterns:
            matches = re.findall(pattern, text)
            entities['components'].extend(matches)
        
        # Извлекаем проблемы
        for pattern in problem_patterns:
            matches = re.findall(pattern, text)
            entities['problems'].extend(matches)
        
        # Извлекаем симптомы
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text)
            entities['symptoms'].extend(matches)
        
        # Извлекаем действия
        for pattern in action_patterns:
            matches = re.findall(pattern, text)
            entities['actions'].extend(matches)
        
        return entities
    
    def extract_relationships(self, question: str, answer: str, category: str) -> List[Tuple]:
        """Извлекаем связи между сущностями"""
        
        relationships = []
        text = (question + " " + answer).lower()
        
        # Каузальные паттерны
        causal_patterns = [
            (r'(\w+)\s+causes?\s+(\w+)', 'CAUSES'),
            (r'(\w+)\s+leads?\s+to\s+(\w+)', 'CAUSES'),
            (r'(\w+)\s+results?\s+in\s+(\w+)', 'CAUSES'),
            (r'due\s+to\s+(\w+).*?(\w+)', 'CAUSES'),
            (r'because\s+of\s+(\w+).*?(\w+)', 'CAUSES'),
        ]
        
        # Диагностические паттерны
        diagnostic_patterns = [
            (r'(\w+)\s+indicates?\s+(\w+)', 'INDICATES'),
            (r'(\w+)\s+suggests?\s+(\w+)', 'INDICATES'),
            (r'(\w+)\s+means?\s+(\w+)', 'INDICATES'),
            (r'if\s+(\w+).*?then\s+(\w+)', 'INDICATES'),
        ]
        
        # Требования
        requirement_patterns = [
            (r'(\w+)\s+requires?\s+(\w+)', 'REQUIRES'),
            (r'(\w+)\s+needs?\s+(\w+)', 'REQUIRES'),
            (r'must\s+(\w+).*?before\s+(\w+)', 'REQUIRES'),
        ]
        
        # Иерархические
        hierarchy_patterns = [
            (r'(\w+)\s+is\s+part\s+of\s+(\w+)', 'PART_OF'),
            (r'(\w+)\s+in\s+the\s+(\w+)', 'PART_OF'),
            (r'(\w+)\s+within\s+(\w+)', 'PART_OF'),
        ]
        
        all_patterns = (
            causal_patterns + 
            diagnostic_patterns + 
            requirement_patterns + 
            hierarchy_patterns
        )
        
        for pattern, rel_type in all_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    relationships.append((match[0], rel_type, match[1]))
        
        return relationships
    
    def build_category_subgraph(self, category: str) -> Dict:
        """Строим подграф для конкретной категории вопросов"""
        
        if category == 'causal':
            # Причинно-следственные связи
            return {
                'focus': 'CAUSES',
                'weight': 2.0,
                'depth': 3  # Глубина обхода графа
            }
        elif category == 'diagnostic':
            # Диагностические связи
            return {
                'focus': 'INDICATES',
                'weight': 2.0,
                'depth': 2
            }
        elif category == 'factual':
            # Фактические связи
            return {
                'focus': 'PART_OF',
                'weight': 1.5,
                'depth': 1
            }
        elif category == 'comparative':
            # Сравнительные связи
            return {
                'focus': 'DIFFERS_FROM',
                'weight': 1.5,
                'depth': 2
            }
        return {'focus': None, 'weight': 1.0, 'depth': 1}
    
    def add_external_knowledge(self, external_docs: List[Dict]):
        """Добавляем внешние знания из Tavily/Serper"""
        
        for doc in external_docs:
            concept = doc.get('concept', '')
            content = doc.get('content', '')
            
            # Добавляем узел концепта
            self.graph.add_node(
                concept,
                type='component',
                source='external',
                content=content[:500]  # Ограничиваем размер
            )
            
            # Извлекаем связанные концепты
            entities = self.extract_entities_from_qa("", content, "external")
            
            # Добавляем связи
            for comp in entities['components'][:5]:  # Ограничиваем
                if comp != concept:
                    self.graph.add_edge(
                        concept, comp,
                        type='RELATED_TO',
                        weight=0.5
                    )
    
    def compute_graph_metrics(self) -> Dict:
        """Вычисляем метрики графа для оценки качества"""
        
        metrics = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }
        
        # PageRank для важности узлов
        if self.graph.number_of_nodes() > 0:
            pagerank = nx.pagerank(self.graph, max_iter=50)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            metrics['top_concepts'] = top_nodes
        
        return metrics
    
    def get_retrieval_subgraph(self, query: str, mode: str = 'graph') -> List[Dict]:
        """Получаем подграф для retrieval"""
        
        # Извлекаем ключевые концепты из запроса
        entities = self.extract_entities_from_qa(query, "", "query")
        all_entities = (
            entities['components'] + 
            entities['problems'] + 
            entities['symptoms']
        )
        
        retrieved = []
        
        for entity in all_entities[:3]:  # Топ-3 сущности
            if entity in self.graph:
                # Получаем соседей
                neighbors = list(self.graph.neighbors(entity))
                predecessors = list(self.graph.predecessors(entity))
                
                # Формируем контекст
                context = {
                    'entity': entity,
                    'type': self.graph.nodes[entity].get('type', 'unknown'),
                    'content': self.graph.nodes[entity].get('content', ''),
                    'connections': neighbors + predecessors,
                    'score': 0.9  # Базовый скор
                }
                retrieved.append(context)
        
        return retrieved

def main():
    # Загружаем датасет
    with open("../data/apqc_auto.json", 'r') as f:
        dataset = json.load(f)
    
    # Создаём граф
    kg = AutomotiveKnowledgeGraph()
    
    print("="*60)
    print("ПОСТРОЕНИЕ ГРАФА ЗНАНИЙ")
    print("="*60)
    
    # Обрабатываем вопросы
    for i, item in enumerate(dataset['questions'][:100], 1):  # Первые 100 для теста
        question = item['question']
        answer = item['answer']
        category = item['category']
        
        # Извлекаем сущности
        entities = kg.extract_entities_from_qa(question, answer, category)
        
        # Добавляем узлы
        for comp in entities['components']:
            kg.graph.add_node(comp, type='component')
        for prob in entities['problems']:
            kg.graph.add_node(prob, type='problem')
        
        # Извлекаем связи
        relationships = kg.extract_relationships(question, answer, category)
        
        # Добавляем рёбра
        for source, rel_type, target in relationships:
            kg.graph.add_edge(source, target, type=rel_type)
        
        if i % 20 == 0:
            print(f"Обработано {i} вопросов...")
    
    # Добавляем внешние знания (если есть)
    external_file = Path("knowledge_base_external.json")
    if external_file.exists():
        with open(external_file, 'r') as f:
            external_data = json.load(f)
            kg.add_external_knowledge(external_data.get('documents', []))
            print(f"Добавлено {len(external_data.get('documents', []))} внешних документов")
    
    # Вычисляем метрики
    metrics = kg.compute_graph_metrics()
    
    print("\n" + "="*60)
    print("МЕТРИКИ ГРАФА:")
    print(f"- Узлов: {metrics['nodes']}")
    print(f"- Рёбер: {metrics['edges']}")
    print(f"- Плотность: {metrics['density']:.4f}")
    print(f"- Компонент связности: {metrics['components']}")
    print(f"- Средняя степень: {metrics['avg_degree']:.2f}")
    
    if 'top_concepts' in metrics:
        print("\nТоп-10 важных концептов (PageRank):")
        for concept, score in metrics['top_concepts']:
            print(f"  - {concept}: {score:.4f}")
    
    # Сохраняем граф
    nx.write_graphml(kg.graph, "automotive_knowledge_graph.graphml")
    print(f"\n✅ Граф сохранён в automotive_knowledge_graph.graphml")
    
    return kg

if __name__ == "__main__":
    kg = main()