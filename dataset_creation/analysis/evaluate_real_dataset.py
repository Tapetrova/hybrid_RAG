#!/usr/bin/env python3
"""
Оценка hallucination на РЕАЛЬНЫХ результатах из датасета
Каждый вопрос имеет свой золотой ответ
"""

import json
import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RealDatasetEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Загружаем результаты теста на реальных вопросах
        with open('test_real_dataset_results.json', 'r') as f:
            self.results = json.load(f)
        
        print(f"✅ Загружено {len(self.results)} вопросов с золотыми ответами")
    
    def extract_claims(self, answer_text: str) -> List[Dict]:
        """Извлекаем утверждения из ответа"""
        
        prompt = f"""Extract key factual claims from this automotive answer.
Each claim should be a specific, verifiable statement.

Answer: {answer_text[:500]}

Return JSON array with max 5 claims:
{{"claims": [{{"text": "specific claim 1"}}, {{"text": "specific claim 2"}}]}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract specific verifiable claims."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            claims = result.get('claims', [])
            return claims[:5]
            
        except Exception as e:
            print(f"    Error extracting claims: {e}")
            return []
    
    def judge_claim(self, claim: str, gold_answer: str, retrieved_context: List[str], mode: str) -> Dict:
        """Оцениваем утверждение правильно"""
        
        if mode == 'base_llm':
            # Для base_llm проверяем ТОЛЬКО по золотому ответу
            prompt = f"""Judge if this claim is supported by the reference answer.

Claim: {claim}

Reference Answer (from dataset): {gold_answer[:800]}

Classify as:
- "supported" if the reference answer confirms this claim
- "contradicted" if the reference answer contradicts this claim  
- "unverifiable" if the reference answer doesn't address this claim

Return JSON: {{"label": "supported/contradicted/unverifiable", "reason": "brief explanation"}}"""
        else:
            # Для RAG режимов проверяем по золотому ответу + retrieved контексту
            context_str = "\n".join(retrieved_context[:3]) if retrieved_context else "No context"
            
            prompt = f"""Judge if this claim is supported by the reference answer OR retrieved context.

Claim: {claim}

Reference Answer (gold standard from dataset): {gold_answer[:500]}

Retrieved Context: {context_str[:500]}

Classify as:
- "supported" if either reference or context confirms this
- "contradicted" if either contradicts this
- "unverifiable" if neither addresses this

Return JSON: {{"label": "supported/contradicted/unverifiable", "reason": "brief explanation"}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Judge claim against reference. Return JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'claim': claim,
                'label': result.get('label', 'unverifiable'),
                'reason': result.get('reason', '')[:100]
            }
            
        except Exception as e:
            print(f"    Error judging: {e}")
            return {'claim': claim, 'label': 'unverifiable', 'reason': 'Error'}
    
    def evaluate_all(self):
        """Оцениваем все ответы"""
        
        print("="*80)
        print("ОЦЕНКА HALLUCINATION НА РЕАЛЬНЫХ ВОПРОСАХ ИЗ ДАТАСЕТА")
        print("="*80)
        
        all_results = []
        
        for i, q_data in enumerate(self.results, 1):
            q_id = q_data['question_id']
            question = q_data['question_text']
            category = q_data['category']
            gold_answer = q_data['gold_answer']
            
            print(f"\n📝 Вопрос {i}/{len(self.results)} (ID: {q_id})")
            print(f"   Вопрос: {question[:60]}...")
            print(f"   Категория: {category}")
            print(f"   Золотой ответ (из датасета): {gold_answer[:80]}...")
            print("-"*60)
            
            q_metrics = {}
            
            # Оцениваем каждый режим
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                mode_data = q_data[mode]
                answer = mode_data['answer']
                context = mode_data.get('context_used', [])
                
                print(f"\n  🔍 {mode}:")
                print(f"     Ответ модели: {answer[:100]}...")
                
                # 1. Извлекаем claims
                claims = self.extract_claims(answer)
                print(f"     Claims извлечено: {len(claims)}")
                
                # 2. Оцениваем каждый claim
                judgments = []
                for claim_obj in claims:
                    claim_text = claim_obj.get('text', str(claim_obj))
                    judgment = self.judge_claim(
                        claim_text,
                        gold_answer,  # Используем ЗОЛОТОЙ ответ из датасета
                        context if isinstance(context, list) else [],
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
                
                print(f"     ✅ Supported: {supported}/{total}")
                print(f"     ❌ Contradicted: {contradicted}/{total}")
                print(f"     ❓ Unverifiable: {unverifiable}/{total}")
                print(f"     📊 Hallucination Rate = {hr:.1%}")
                
                q_metrics[mode] = {
                    'total_claims': total,
                    'supported': supported,
                    'contradicted': contradicted,
                    'unverifiable': unverifiable,
                    'HR': hr,
                    'HR_contra': hr_contra,
                    'HR_unver': hr_unver
                }
            
            all_results.append({
                'question_id': q_id,
                'question': question,
                'category': category,
                'metrics': q_metrics
            })
        
        # Сохраняем результаты
        with open('real_dataset_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Печатаем итоги
        self.print_summary(all_results)
    
    def print_summary(self, results):
        """Итоговая статистика"""
        
        print("\n" + "="*80)
        print("ИТОГОВЫЕ МЕТРИКИ HALLUCINATION")
        print("="*80)
        
        # Средние HR по режимам
        mode_hrs = {'base_llm': [], 'vector_rag': [], 'graph_rag': [], 'hybrid_ahs': []}
        
        for r in results:
            for mode in mode_hrs:
                mode_hrs[mode].append(r['metrics'][mode]['HR'])
        
        print("\n📊 Средний Hallucination Rate по режимам:")
        avg_hrs = {}
        for mode, hrs in mode_hrs.items():
            if hrs:
                avg = sum(hrs) / len(hrs)
                avg_hrs[mode] = avg
                print(f"  {mode:12} : {avg:.1%}")
        
        if avg_hrs:
            print("\n🏆 Рейтинг режимов (меньше HR = лучше):")
            sorted_modes = sorted(avg_hrs.items(), key=lambda x: x[1])
            for i, (mode, hr) in enumerate(sorted_modes, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣"
                print(f"  {emoji} {mode:12} : HR = {hr:.1%}")
        
        # Проверяем научные гипотезы
        print("\n✅ Проверка научных гипотез:")
        if avg_hrs:
            base_hr = avg_hrs.get('base_llm', 1.0)
            rag_hrs = [avg_hrs.get('vector_rag', 1.0), avg_hrs.get('graph_rag', 1.0), avg_hrs.get('hybrid_ahs', 1.0)]
            
            if base_hr > min(rag_hrs):
                reduction = (1 - min(rag_hrs)/base_hr) * 100
                print(f"  ✓ RAG снижает hallucination на {reduction:.0f}% по сравнению с base_llm")
            else:
                print("  ✗ RAG НЕ снижает hallucination")
            
            if 'hybrid_ahs' in avg_hrs and avg_hrs['hybrid_ahs'] == min(avg_hrs.values()):
                print("  ✓ Гибридный подход (hybrid_ahs) показывает лучший результат")
            else:
                best = min(avg_hrs, key=avg_hrs.get) if avg_hrs else 'unknown'
                print(f"  ✗ Лучший результат у {best}, не у hybrid_ahs")
        
        print("\n💾 Результаты сохранены в real_dataset_evaluation_results.json")
        print("\n🎯 Эти результаты можно использовать для научной статьи!")

def main():
    evaluator = RealDatasetEvaluator()
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()