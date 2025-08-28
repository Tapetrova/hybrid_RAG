#!/usr/bin/env python3
"""
Оценка результатов 5 вопросов алгоритмами hallucination detection
Используем claims extraction и judging из eval_runner
"""

import json
import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class HallucinationEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Загружаем результаты 5 вопросов
        with open('verify_4_modes_results.json', 'r') as f:
            self.results = json.load(f)
        
        # Загружаем золотые ответы из датасета
        with open('../data/apqc_auto.json', 'r') as f:
            dataset = json.load(f)
            self.gold_answers = {}
            # Мапим вопросы на золотые ответы
            for q in dataset['questions']:
                if "engine knock when cold" in q['question'].lower():
                    self.gold_answers[1] = q['answer']
                elif "diagnose a misfire" in q['question'].lower():
                    self.gold_answers[2] = q['answer']
                elif "2020 honda civic" in q['question'].lower():
                    self.gold_answers[3] = q['answer']
                elif "drum brakes vs disc" in q['question'].lower():
                    self.gold_answers[4] = q['answer']
                elif "brakes squeal after replacement" in q['question'].lower():
                    self.gold_answers[5] = q['answer']
    
    def extract_claims(self, answer_text: str) -> List[Dict]:
        """Извлекаем утверждения из ответа"""
        
        prompt = f"""Extract factual claims from this answer.
Return a JSON array of claims.

Answer:
{answer_text}

Extract each distinct factual claim as a separate item.
Format: [{{"text": "claim 1", "type": "factual"}}, {{"text": "claim 2", "type": "causal"}}]
Types: factual, causal, diagnostic, comparative"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract claims as JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            claims = result.get('claims', [])
            
            # Если claims не список, пробуем другие ключи
            if not isinstance(claims, list):
                if isinstance(result, list):
                    claims = result
                elif 'extracted_claims' in result:
                    claims = result['extracted_claims']
                else:
                    # Берём все значения, которые являются списками
                    for value in result.values():
                        if isinstance(value, list):
                            claims = value
                            break
            
            return claims[:10]  # Максимум 10 утверждений
            
        except Exception as e:
            print(f"Error extracting claims: {e}")
            return []
    
    def judge_claim(self, claim: str, gold_answer: str, retrieved_texts: List[str]) -> Dict:
        """Оцениваем утверждение"""
        
        context = "\n".join(retrieved_texts) if retrieved_texts else "No additional context"
        
        prompt = f"""Judge if this claim is supported, contradicted, or unverifiable based on the reference answer and context.

Claim: {claim}

Reference Answer: {gold_answer[:500]}

Retrieved Context: {context[:500]}

Return JSON with label and brief rationale:
{{"label": "supported|contradicted|unverifiable", "rationale": "brief explanation"}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Judge claim accuracy. Return JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                'claim': claim,
                'label': result.get('label', 'unverifiable'),
                'rationale': result.get('rationale', '')[:200]
            }
            
        except Exception as e:
            print(f"Error judging claim: {e}")
            return {'claim': claim, 'label': 'unverifiable', 'rationale': 'Error in judging'}
    
    def calculate_metrics(self, claim_judgments: List[Dict]) -> Dict:
        """Вычисляем метрики hallucination"""
        
        total = len(claim_judgments)
        if total == 0:
            return {'HR': 0, 'HR_contra': 0, 'HR_unver': 0, 'claims_total': 0}
        
        contradicted = sum(1 for c in claim_judgments if c['label'] == 'contradicted')
        unverifiable = sum(1 for c in claim_judgments if c['label'] == 'unverifiable')
        
        hr = (contradicted + unverifiable) / total
        hr_contra = contradicted / total
        hr_unver = unverifiable / total
        
        return {
            'HR': round(hr, 3),
            'HR_contra': round(hr_contra, 3),
            'HR_unver': round(hr_unver, 3),
            'claims_total': total,
            'supported': total - contradicted - unverifiable
        }
    
    def evaluate_all(self):
        """Оцениваем все ответы"""
        
        print("="*80)
        print("ОЦЕНКА HALLUCINATION ДЛЯ 5 ВОПРОСОВ")
        print("="*80)
        
        evaluation_results = []
        
        for i, question_data in enumerate(self.results, 1):
            q = question_data['question']
            print(f"\n📝 Вопрос {i}: {q['question'][:60]}...")
            print(f"   Категория: {q['category']}")
            
            # Получаем золотой ответ
            gold_answer = self.gold_answers.get(q['id'], "")
            if not gold_answer:
                print(f"   ⚠️ Золотой ответ не найден")
                gold_answer = "Reference answer not available"
            else:
                print(f"   📚 Золотой ответ: {gold_answer[:80]}...")
            
            print("-"*60)
            
            q_results = {'question': q, 'modes': {}}
            
            # Оцениваем каждый режим
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                mode_data = question_data[mode]
                answer = mode_data['answer']
                context = mode_data.get('context_used', [])
                
                print(f"\n  🔍 {mode}:")
                
                # 1. Извлекаем claims
                claims = self.extract_claims(answer)
                print(f"     Извлечено claims: {len(claims)}")
                
                # 2. Оцениваем каждый claim
                claim_judgments = []
                for claim in claims:
                    judgment = self.judge_claim(
                        claim.get('text', str(claim)), 
                        gold_answer,
                        context if isinstance(context, list) else []
                    )
                    claim_judgments.append(judgment)
                
                # 3. Вычисляем метрики
                metrics = self.calculate_metrics(claim_judgments)
                
                print(f"     ✅ Supported: {metrics['supported']}/{metrics['claims_total']}")
                print(f"     ❌ Contradicted: {int(metrics['HR_contra'] * metrics['claims_total'])}")
                print(f"     ❓ Unverifiable: {int(metrics['HR_unver'] * metrics['claims_total'])}")
                print(f"     📊 HR = {metrics['HR']:.1%}")
                
                q_results['modes'][mode] = {
                    'answer': answer[:200],
                    'claims': claims,
                    'claim_judgments': claim_judgments,
                    'metrics': metrics
                }
            
            evaluation_results.append(q_results)
        
        # Сохраняем результаты
        with open('evaluation_5_questions_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        # Итоговая статистика
        self.print_summary(evaluation_results)
        
        return evaluation_results
    
    def print_summary(self, results):
        """Печатаем итоговую статистику"""
        
        print("\n" + "="*80)
        print("ИТОГОВАЯ СТАТИСТИКА HALLUCINATION")
        print("="*80)
        
        # Собираем метрики по режимам
        mode_metrics = {
            'base_llm': [],
            'vector_rag': [],
            'graph_rag': [],
            'hybrid_ahs': []
        }
        
        for q_result in results:
            for mode, data in q_result['modes'].items():
                mode_metrics[mode].append(data['metrics']['HR'])
        
        print("\n📊 Средний Hallucination Rate (HR) по режимам:")
        print("-"*60)
        
        avg_hrs = {}
        for mode, hrs in mode_metrics.items():
            avg_hr = sum(hrs) / len(hrs) if hrs else 0
            avg_hrs[mode] = avg_hr
            print(f"  {mode:12} : {avg_hr:.1%}")
        
        print("\n🏆 Рейтинг режимов (меньше HR = лучше):")
        sorted_modes = sorted(avg_hrs.items(), key=lambda x: x[1])
        for i, (mode, hr) in enumerate(sorted_modes, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣"
            print(f"  {emoji} {mode:12} : HR = {hr:.1%}")
        
        # Проверяем гипотезы
        print("\n✅ Проверка гипотез:")
        if avg_hrs['base_llm'] > min(avg_hrs['vector_rag'], avg_hrs['graph_rag'], avg_hrs['hybrid_ahs']):
            print("  ✓ RAG режимы снижают hallucination по сравнению с base_llm")
        else:
            print("  ✗ RAG режимы НЕ снижают hallucination")
        
        if avg_hrs['hybrid_ahs'] == min(avg_hrs.values()):
            print("  ✓ Гибридный режим показывает лучший результат")
        else:
            best_mode = min(avg_hrs, key=avg_hrs.get)
            print(f"  ✗ Лучший результат у {best_mode}, не у hybrid")
        
        print("\n💾 Детальные результаты сохранены в evaluation_5_questions_results.json")

def main():
    evaluator = HallucinationEvaluator()
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()