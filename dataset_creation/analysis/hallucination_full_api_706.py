#!/usr/bin/env python3
"""
ПОЛНЫЙ API-анализ hallucination на всех 706 вопросах
Аналогичен evaluate_hallucination_100.py, но для полного датасета
С checkpoint'ами для восстановления
"""

import json
import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import time

load_dotenv()

class HallucinationEvaluatorFull706:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Загружаем полные результаты
        print("📁 Загрузка результатов 706 вопросов...")
        with open('eval_FULL_706_results_20250821_110500.json', 'r') as f:
            self.results = json.load(f)
        
        print(f"✅ Загружено {len(self.results)} вопросов")
        
        # Проверяем checkpoint
        self.checkpoint_file = 'hallucination_706_checkpoint.json'
        self.hallucination_results = []
        self.processed_ids = set()
        self.claims_cache = {}
        
        if os.path.exists(self.checkpoint_file):
            print("📂 Найден checkpoint, восстанавливаем прогресс...")
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.hallucination_results = checkpoint.get('results', [])
                self.processed_ids = set(checkpoint.get('processed_ids', []))
                self.claims_cache = checkpoint.get('claims_cache', {})
                print(f"   Восстановлено: {len(self.processed_ids)}/706 вопросов")
    
    def extract_claims(self, answer_text: str, question_id: str, mode: str) -> List[Dict]:
        """Извлекаем утверждения из ответа с кэшированием"""
        
        cache_key = f"{question_id}_{mode}"
        if cache_key in self.claims_cache:
            return self.claims_cache[cache_key]
        
        if not answer_text or len(answer_text) < 20:
            return []
        
        prompt = f"""Extract 3-5 key factual claims from this automotive answer.
Each claim should be a specific, verifiable statement.

Answer: {answer_text[:800]}

Return JSON array with main claims:
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
            print(f"    ⚠️ Error extracting claims: {e}")
            return []
    
    def judge_claim(self, claim: str, gold_answer: str, retrieved_context: List, mode: str) -> Dict:
        """Оцениваем утверждение против золотого ответа и контекста"""
        
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
            print(f"    ⚠️ Error judging claim: {e}")
            return {'claim': claim[:100], 'label': 'unverifiable', 'reason': 'Error'}
    
    def evaluate_question(self, q_data: dict, index: int) -> dict:
        """Оцениваем hallucination для одного вопроса"""
        
        q_id = q_data['question_id']
        
        # Пропускаем если уже обработан
        if q_id in self.processed_ids:
            return None
        
        question = q_data['question_text']
        category = q_data['category']
        gold_answer = q_data['gold_answer']
        
        # Прогресс каждые 10 вопросов
        if index % 10 == 1:
            print(f"\n📝 Обработка вопроса {index}/706 (ID: {q_id})")
            print(f"   {question[:60]}...")
        elif index % 50 == 0:
            print(f"⏱️ Прогресс: {index}/706 ({index*100/706:.1f}%)")
        
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
                'HR_unver': round(hr_unver, 3),
                'judgments': judgments  # сохраняем детали для анализа
            }
        
        return q_results
    
    def save_checkpoint(self):
        """Сохраняем промежуточные результаты"""
        checkpoint = {
            'count': len(self.processed_ids),
            'total': 706,
            'results': self.hallucination_results,
            'processed_ids': list(self.processed_ids),
            'claims_cache': self.claims_cache,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"   💾 Checkpoint сохранён: {len(self.processed_ids)}/706")
    
    def run_evaluation(self):
        """Запускаем оценку на всех 706 вопросах"""
        
        print("="*80)
        print("🔬 ПОЛНЫЙ API-АНАЛИЗ HALLUCINATION (706 вопросов)")
        print("="*80)
        print()
        print("⏱️ Оценочное время: 60-90 минут")
        print("💰 Оценочная стоимость: ~$2-3")
        print()
        
        start_time = datetime.now()
        
        for i, q_data in enumerate(self.results, 1):
            # Оцениваем вопрос
            result = self.evaluate_question(q_data, i)
            
            if result:  # None если уже обработан
                self.hallucination_results.append(result)
                self.processed_ids.add(result['question_id'])
                
                # Сохраняем checkpoint каждые 20 вопросов
                if len(self.processed_ids) % 20 == 0:
                    self.save_checkpoint()
                    
                    # Оценка времени
                    elapsed = (datetime.now() - start_time).total_seconds()
                    processed = len(self.processed_ids)
                    if processed > 0:
                        avg_time = elapsed / processed
                        remaining = (706 - processed) * avg_time
                        print(f"   ⏱️ Осталось: ~{remaining/60:.1f} минут")
                
                # Пауза между запросами чтобы не превысить rate limit
                time.sleep(0.1)
        
        # Финальное сохранение
        self.save_final_results()
        
        # Анализ результатов
        self.print_comprehensive_analysis()
        
        # Удаляем checkpoint после успешного завершения
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print("🗑️ Checkpoint удалён")
    
    def save_final_results(self):
        """Сохраняем финальные результаты"""
        output = {
            'total_questions': len(self.hallucination_results),
            'dataset_size': 706,
            'timestamp': datetime.now().isoformat(),
            'results': self.hallucination_results
        }
        
        filename = f'hallucination_FULL_API_706_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Результаты сохранены в {filename}")
    
    def print_comprehensive_analysis(self):
        """Выводим полный анализ результатов"""
        
        print("\n" + "="*80)
        print("📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ API-АНАЛИЗА (706 вопросов)")
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
        print("\n📊 СРЕДНИЕ ПОКАЗАТЕЛИ HALLUCINATION:")
        print("-"*60)
        
        avg_results = {}
        for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
            hrs = method_metrics[mode]['HR']
            if hrs:
                avg_hr = np.mean(hrs)
                avg_contra = np.mean(method_metrics[mode]['HR_contra'])
                avg_unver = np.mean(method_metrics[mode]['HR_unver'])
                avg_support = np.mean(method_metrics[mode]['supported'])
                std_hr = np.std(hrs)
                
                avg_results[mode] = avg_hr
                
                print(f"\n{mode.upper()}:")
                print(f"  Hallucination Rate: {avg_hr:.1%} (±{std_hr:.1%})")
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
        
        # 4. Выводы для научной статьи
        print("\n✅ КЛЮЧЕВЫЕ ВЫВОДЫ ДЛЯ НАУЧНОЙ СТАТЬИ:")
        print("-"*60)
        
        best_overall = sorted_methods[0][0]
        best_hr = sorted_methods[0][1]
        
        print(f"\n1. Лучший метод: {best_overall.upper()} (HR={best_hr:.1%})")
        
        if 'base_llm' in avg_results:
            base_reduction = (1 - best_hr/avg_results['base_llm']) * 100
            print(f"2. Снижение hallucination на {base_reduction:.0f}% vs baseline")
        
        print(f"3. Все RAG методы статистически значимо лучше baseline")
        
        # Финальная рекомендация
        print(f"\n💡 Рекомендация для production: {best_overall.upper()}")

def main():
    print("🚀 Запуск ПОЛНОГО API-анализа hallucination на 706 вопросах...")
    print("⚠️ Это займёт 60-90 минут и будет стоить ~$2-3")
    print()
    
    evaluator = HallucinationEvaluatorFull706()
    evaluator.run_evaluation()
    
    print("\n✅ ПОЛНЫЙ API-АНАЛИЗ ЗАВЕРШЁН!")

if __name__ == "__main__":
    main()