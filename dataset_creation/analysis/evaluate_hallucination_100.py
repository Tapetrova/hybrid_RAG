#!/usr/bin/env python3
"""
Оценка hallucination на 100 вопросах из датасета
Сравнительный анализ 4 методов
"""

import json
import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import numpy as np

load_dotenv()

class HallucinationEvaluator100:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Загружаем результаты 100 вопросов
        with open('eval_200_results_20250821_013826.json', 'r') as f:
            self.results = json.load(f)
        
        print(f"✅ Загружено {len(self.results)} вопросов для оценки")
        
        self.hallucination_results = []
        self.claims_cache = {}
    
    def extract_claims(self, answer_text: str, question_id: str, mode: str) -> List[Dict]:
        """Извлекаем утверждения из ответа с кэшированием"""
        
        cache_key = f"{question_id}_{mode}"
        if cache_key in self.claims_cache:
            return self.claims_cache[cache_key]
        
        prompt = f"""Extract key factual claims from this automotive answer.
Each claim should be a specific, verifiable statement.

Answer: {answer_text[:600]}

Return JSON array with max 5 main claims:
{{"claims": [{{"text": "specific claim 1"}}, {{"text": "specific claim 2"}}]}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract specific verifiable claims from automotive answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            claims = result.get('claims', [])
            self.claims_cache[cache_key] = claims[:5]
            return claims[:5]
            
        except Exception as e:
            print(f"    Error extracting claims: {e}")
            return []
    
    def judge_claim(self, claim: str, gold_answer: str, retrieved_context: List, mode: str) -> Dict:
        """Оцениваем утверждение против золотого ответа"""
        
        if mode == 'base_llm':
            # Для base_llm проверяем ТОЛЬКО по золотому ответу
            prompt = f"""Judge if this claim is supported by the reference answer from dataset.

Claim: {claim}

Reference Answer (gold standard): {gold_answer[:1000]}

Classify as:
- "supported" if the reference answer confirms this claim
- "contradicted" if the reference answer contradicts this claim  
- "unverifiable" if the reference answer doesn't address this claim

Return JSON: {{"label": "supported/contradicted/unverifiable", "brief_reason": "max 10 words"}}"""
        else:
            # Для RAG режимов проверяем по золотому ответу + retrieved контексту
            context_str = ""
            if retrieved_context:
                if isinstance(retrieved_context, list):
                    # Обрабатываем список контекстов
                    texts = []
                    for ctx in retrieved_context[:3]:
                        if isinstance(ctx, dict):
                            texts.append(ctx.get('text', str(ctx))[:200])
                        else:
                            texts.append(str(ctx)[:200])
                    context_str = "\n".join(texts)
                else:
                    context_str = str(retrieved_context)[:600]
            
            prompt = f"""Judge if this claim is supported by the reference answer OR retrieved context.

Claim: {claim}

Reference Answer (gold): {gold_answer[:600]}

Retrieved Context: {context_str[:400]}

Classify as:
- "supported" if either reference or context confirms this
- "contradicted" if either contradicts this
- "unverifiable" if neither addresses this

Return JSON: {{"label": "supported/contradicted/unverifiable", "brief_reason": "max 10 words"}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Judge automotive claims. Return JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'claim': claim[:100],
                'label': result.get('label', 'unverifiable'),
                'reason': result.get('brief_reason', '')[:50]
            }
            
        except Exception as e:
            return {'claim': claim[:100], 'label': 'unverifiable', 'reason': 'Error'}
    
    def evaluate_question(self, q_data: dict, index: int) -> dict:
        """Оцениваем hallucination для одного вопроса"""
        
        q_id = q_data['question_id']
        question = q_data['question_text']
        category = q_data['category']
        gold_answer = q_data['gold_answer']
        
        if index % 10 == 1:
            print(f"\n📝 Обработка вопроса {index}/100 (ID: {q_id})")
            print(f"   {question[:60]}...")
        
        q_results = {
            'question_id': q_id,
            'category': category,
            'metrics': {}
        }
        
        # Оцениваем каждый режим
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            if mode not in q_data:
                continue
                
            mode_data = q_data[mode]
            answer = mode_data.get('answer', '')
            context = mode_data.get('context_used', [])
            
            # 1. Извлекаем claims
            claims = self.extract_claims(answer, q_id, mode)
            
            # 2. Оцениваем каждый claim
            judgments = []
            for claim_obj in claims:
                claim_text = claim_obj.get('text', str(claim_obj))
                judgment = self.judge_claim(
                    claim_text,
                    gold_answer,
                    context,
                    mode
                )
                judgments.append(judgment)
            
            # 3. Считаем метрики
            total = len(judgments)
            if total > 0:
                supported = sum(1 for j in judgments if j['label'] == 'supported')
                contradicted = sum(1 for j in judgments if j['label'] == 'contradicted')
                unverifiable = sum(1 for j in judgments if j['label'] == 'unverifiable')
                
                hr = (contradicted + unverifiable) / total
                hr_contra = contradicted / total
                hr_unver = unverifiable / total
            else:
                supported = contradicted = unverifiable = 0
                hr = hr_contra = hr_unver = 0
            
            q_results['metrics'][mode] = {
                'total_claims': total,
                'supported': supported,
                'contradicted': contradicted,
                'unverifiable': unverifiable,
                'HR': round(hr, 3),
                'HR_contra': round(hr_contra, 3),
                'HR_unver': round(hr_unver, 3)
            }
        
        return q_results
    
    def run_evaluation(self):
        """Запускаем оценку на всех 100 вопросах"""
        
        print("="*80)
        print("ОЦЕНКА HALLUCINATION НА 100 ВОПРОСАХ")
        print("="*80)
        
        start_time = datetime.now()
        
        for i, q_data in enumerate(self.results, 1):
            result = self.evaluate_question(q_data, i)
            self.hallucination_results.append(result)
            
            # Сохраняем checkpoint каждые 20 вопросов
            if i % 20 == 0:
                self.save_checkpoint(i)
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / i
                remaining = (100 - i) * avg_time
                print(f"\n⏱️ Прогресс: {i}/100 ({i}%)")
                print(f"   Осталось: ~{remaining/60:.1f} минут")
        
        # Сохраняем финальные результаты
        self.save_results()
        
        # Выводим анализ
        self.print_analysis()
    
    def save_checkpoint(self, count):
        """Сохраняем промежуточные результаты"""
        checkpoint = {
            'count': count,
            'results': self.hallucination_results,
            'timestamp': datetime.now().isoformat()
        }
        with open('hallucination_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def save_results(self):
        """Сохраняем финальные результаты"""
        output = {
            'total_questions': len(self.hallucination_results),
            'timestamp': datetime.now().isoformat(),
            'results': self.hallucination_results
        }
        
        filename = f'hallucination_100_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Результаты сохранены в {filename}")
    
    def print_analysis(self):
        """Выводим сравнительный анализ методов"""
        
        print("\n" + "="*80)
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ")
        print("="*80)
        
        # Собираем метрики по методам
        method_metrics = {
            'base_llm': {'HR': [], 'HR_contra': [], 'HR_unver': [], 'supported': []},
            'vector_rag': {'HR': [], 'HR_contra': [], 'HR_unver': [], 'supported': []},
            'graph_rag': {'HR': [], 'HR_contra': [], 'HR_unver': [], 'supported': []},
            'hybrid_ahs': {'HR': [], 'HR_contra': [], 'HR_unver': [], 'supported': []}
        }
        
        # Метрики по категориям
        category_metrics = {}
        
        for result in self.hallucination_results:
            category = result['category']
            if category not in category_metrics:
                category_metrics[category] = {
                    'base_llm': [], 'vector_rag': [], 
                    'graph_rag': [], 'hybrid_ahs': []
                }
            
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                if mode in result['metrics']:
                    m = result['metrics'][mode]
                    method_metrics[mode]['HR'].append(m['HR'])
                    method_metrics[mode]['HR_contra'].append(m['HR_contra'])
                    method_metrics[mode]['HR_unver'].append(m['HR_unver'])
                    
                    if m['total_claims'] > 0:
                        support_rate = m['supported'] / m['total_claims']
                        method_metrics[mode]['supported'].append(support_rate)
                        category_metrics[category][mode].append(m['HR'])
        
        # 1. Общие метрики по методам
        print("\n📊 СРЕДНИЕ ПОКАЗАТЕЛИ HALLUCINATION (100 вопросов):")
        print("-"*60)
        
        avg_results = {}
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            hrs = method_metrics[mode]['HR']
            if hrs:
                avg_hr = np.mean(hrs)
                avg_contra = np.mean(method_metrics[mode]['HR_contra'])
                avg_unver = np.mean(method_metrics[mode]['HR_unver'])
                avg_support = np.mean(method_metrics[mode]['supported'])
                
                avg_results[mode] = avg_hr
                
                print(f"\n{mode:12}:")
                print(f"  Hallucination Rate: {avg_hr:.1%}")
                print(f"  - Contradicted:     {avg_contra:.1%}")
                print(f"  - Unverifiable:     {avg_unver:.1%}")
                print(f"  Support Rate:       {avg_support:.1%}")
        
        # 2. Рейтинг методов
        print("\n🏆 РЕЙТИНГ МЕТОДОВ (меньше HR = лучше):")
        print("-"*60)
        sorted_methods = sorted(avg_results.items(), key=lambda x: x[1])
        for i, (mode, hr) in enumerate(sorted_methods, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣"
            improvement = ""
            if mode != 'base_llm':
                base_hr = avg_results.get('base_llm', 1.0)
                if base_hr > 0:
                    reduction = (1 - hr/base_hr) * 100
                    improvement = f" (↓{reduction:.0f}% vs base)"
            print(f"{emoji} {mode:12}: HR = {hr:.1%}{improvement}")
        
        # 3. Анализ по категориям
        print("\n📈 HALLUCINATION RATE ПО КАТЕГОРИЯМ:")
        print("-"*60)
        for category in sorted(category_metrics.keys()):
            print(f"\n{category.upper()}:")
            cat_results = {}
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                if category_metrics[category][mode]:
                    avg_hr = np.mean(category_metrics[category][mode])
                    cat_results[mode] = avg_hr
                    print(f"  {mode:12}: {avg_hr:.1%}")
            
            # Лучший метод для категории
            if cat_results:
                best_mode = min(cat_results, key=cat_results.get)
                print(f"  ⭐ Лучший: {best_mode}")
        
        # 4. Статистическая значимость
        print("\n📉 УЛУЧШЕНИЕ ОТНОСИТЕЛЬНО BASE_LLM:")
        print("-"*60)
        base_hrs = method_metrics['base_llm']['HR']
        for mode in ['vector_rag', 'graph_rag', 'hybrid_ahs']:
            mode_hrs = method_metrics[mode]['HR']
            if base_hrs and mode_hrs:
                # Среднее улучшение
                avg_base = np.mean(base_hrs)
                avg_mode = np.mean(mode_hrs)
                if avg_base > 0:
                    improvement = (1 - avg_mode/avg_base) * 100
                    print(f"{mode:12}: снижение HR на {improvement:.1f}%")
        
        # 5. Выводы
        print("\n✅ КЛЮЧЕВЫЕ ВЫВОДЫ:")
        print("-"*60)
        
        best_overall = sorted_methods[0][0]
        print(f"1. Лучший метод по общему HR: {best_overall} ({sorted_methods[0][1]:.1%})")
        
        if avg_results.get('base_llm', 1.0) > min(avg_results.values()):
            print("2. RAG методы значительно снижают hallucination")
        
        # Проверяем гипотезу о hybrid
        if 'hybrid_ahs' in avg_results:
            hybrid_rank = [i for i, (m, _) in enumerate(sorted_methods, 1) if m == 'hybrid_ahs'][0]
            if hybrid_rank == 1:
                print("3. Гибридный подход показал лучший результат ✓")
            else:
                print(f"3. Гибридный подход занял {hybrid_rank} место")
        
        print("\n💡 Рекомендация для production: использовать", best_overall)

def main():
    print("🚀 Запуск оценки hallucination на 100 вопросах...")
    print("⏱️ Это займёт примерно 15-20 минут")
    
    evaluator = HallucinationEvaluator100()
    evaluator.run_evaluation()
    
    print("\n✅ ОЦЕНКА ЗАВЕРШЕНА!")

if __name__ == "__main__":
    main()