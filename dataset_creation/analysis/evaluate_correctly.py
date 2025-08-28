#!/usr/bin/env python3
"""
ПРАВИЛЬНАЯ оценка hallucination:
- Для base_llm: проверяем по ЗОЛОТОМУ ОТВЕТУ из датасета
- Для RAG режимов: проверяем по ЗОЛОТОМУ ОТВЕТУ + retrieved контексту
"""

import json
import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class CorrectEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Загружаем результаты 5 вопросов
        with open('verify_4_modes_results.json', 'r') as f:
            self.results = json.load(f)
        
        # ПРАВИЛЬНО загружаем золотые ответы
        with open('../data/apqc_auto.json', 'r') as f:
            dataset = json.load(f)
        
        # Создаём мапинг вопрос -> золотой ответ
        self.gold_answers = {}
        
        # Точные вопросы из теста
        test_questions = [
            "Why does engine knock when cold?",
            "How to diagnose a misfire?", 
            "What type of oil for 2020 Honda Civic?",
            "Drum brakes vs disc brakes performance?",
            "Why do brakes squeal after replacement?"
        ]
        
        # Более точный поиск золотых ответов
        for test_q in test_questions:
            found = False
            for item in dataset['questions']:
                q_lower = item['question'].lower()
                test_lower = test_q.lower()
                
                # Проверяем по ключевым словам для каждого вопроса
                if "engine knock" in test_lower and "cold" in test_lower:
                    if ("knock" in q_lower and "cold" in q_lower) or ("engine" in q_lower and "knock" in q_lower):
                        self.gold_answers[test_q] = item['answer']
                        print(f"✓ Найден золотой ответ для: {test_q[:40]}...")
                        print(f"  Соответствующий вопрос в датасете: {item['question'][:60]}...")
                        found = True
                        break
                elif "diagnose" in test_lower and "misfire" in test_lower:
                    if "misfire" in q_lower or ("diagnos" in q_lower and "miss" in q_lower):
                        self.gold_answers[test_q] = item['answer']
                        print(f"✓ Найден золотой ответ для: {test_q[:40]}...")
                        print(f"  Соответствующий вопрос в датасете: {item['question'][:60]}...")
                        found = True
                        break
                elif "2020 honda civic" in test_lower and "oil" in test_lower:
                    if ("civic" in q_lower and "oil" in q_lower) or ("honda" in q_lower and "oil" in q_lower):
                        self.gold_answers[test_q] = item['answer']
                        print(f"✓ Найден золотой ответ для: {test_q[:40]}...")
                        print(f"  Соответствующий вопрос в датасете: {item['question'][:60]}...")
                        found = True
                        break
                elif "drum" in test_lower and "disc" in test_lower and "brake" in test_lower:
                    if ("drum" in q_lower and "disc" in q_lower) or ("drum brake" in q_lower and "disc brake" in q_lower):
                        self.gold_answers[test_q] = item['answer']
                        print(f"✓ Найден золотой ответ для: {test_q[:40]}...")
                        print(f"  Соответствующий вопрос в датасете: {item['question'][:60]}...")
                        found = True
                        break
                elif "squeal" in test_lower and "replacement" in test_lower:
                    if ("squeal" in q_lower) or ("brake" in q_lower and "noise" in q_lower):
                        self.gold_answers[test_q] = item['answer']
                        print(f"✓ Найден золотой ответ для: {test_q[:40]}...")
                        print(f"  Соответствующий вопрос в датасете: {item['question'][:60]}...")
                        found = True
                        break
        
        # Если не нашли точные, берём первые подходящие по категории
        categories = ['causal', 'diagnostic', 'factual', 'comparative', 'causal']
        for i, test_q in enumerate(test_questions):
            if test_q not in self.gold_answers:
                category = categories[i]
                # Берём первый ответ из той же категории
                for item in dataset['questions']:
                    if item['category'] == category and len(item['answer']) > 200:
                        self.gold_answers[test_q] = item['answer']
                        print(f"⚠️ Использую ответ из категории '{category}' для: {test_q[:40]}...")
                        print(f"  Вопрос в датасете: {item['question'][:60]}...")
                        break
    
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
            return claims[:5]  # Максимум 5
            
        except Exception as e:
            print(f"    Error extracting claims: {e}")
            return []
    
    def judge_claim_correctly(self, claim: str, gold_answer: str, retrieved_context: List[str], mode: str) -> Dict:
        """ПРАВИЛЬНАЯ оценка claim"""
        
        # Для base_llm проверяем ТОЛЬКО по золотому ответу
        # Для RAG режимов проверяем по золотому ответу + контексту
        
        if mode == 'base_llm':
            # Проверяем только по золотому ответу
            prompt = f"""Judge if this claim is supported by the reference answer.

Claim: {claim}

Reference Answer: {gold_answer[:800]}

Classify as:
- "supported" if the reference answer confirms this claim
- "contradicted" if the reference answer contradicts this claim  
- "unverifiable" if the reference answer doesn't address this claim

Return JSON: {{"label": "supported/contradicted/unverifiable", "reason": "brief explanation"}}"""
        else:
            # Для RAG проверяем по золотому ответу + retrieved контексту
            context_str = "\n".join(retrieved_context[:3]) if retrieved_context else "No context"
            
            prompt = f"""Judge if this claim is supported by the reference answer OR retrieved context.

Claim: {claim}

Reference Answer: {gold_answer[:500]}

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
        """Оцениваем все ответы ПРАВИЛЬНО"""
        
        print("="*80)
        print("ПРАВИЛЬНАЯ ОЦЕНКА HALLUCINATION")
        print("="*80)
        print(f"\nНайдено золотых ответов: {len(self.gold_answers)}/5\n")
        
        all_results = []
        
        for i, q_data in enumerate(self.results):
            question = q_data['question']['question']
            category = q_data['question']['category']
            
            print(f"\n📝 Вопрос {i+1}: {question[:60]}...")
            print(f"   Категория: {category}")
            
            # Получаем золотой ответ
            gold = self.gold_answers.get(question, "")
            if gold:
                print(f"   ✅ Золотой ответ: {gold[:80]}...")
            else:
                print(f"   ❌ Золотой ответ не найден!")
                gold = "Reference answer not available"
            
            print("-"*60)
            
            q_metrics = {}
            
            # Оцениваем каждый режим
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                answer = q_data[mode]['answer']
                context = q_data[mode].get('context_used', [])
                
                print(f"\n  🔍 {mode}:")
                
                # 1. Извлекаем claims
                claims = self.extract_claims(answer)
                print(f"     Claims извлечено: {len(claims)}")
                
                # 2. Оцениваем каждый claim ПРАВИЛЬНО
                judgments = []
                for claim_obj in claims:
                    claim_text = claim_obj.get('text', str(claim_obj))
                    judgment = self.judge_claim_correctly(
                        claim_text,
                        gold,
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
                print(f"     📊 HR = {hr:.1%}")
                
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
                'question': question,
                'category': category,
                'has_gold': gold != "Reference answer not available",
                'metrics': q_metrics
            })
        
        # Сохраняем результаты
        with open('correct_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Печатаем итоги
        self.print_summary(all_results)
    
    def print_summary(self, results):
        """Итоговая статистика"""
        
        print("\n" + "="*80)
        print("ИТОГОВЫЕ МЕТРИКИ HALLUCINATION (ПРАВИЛЬНАЯ ОЦЕНКА)")
        print("="*80)
        
        # Средние HR по режимам
        mode_hrs = {'base_llm': [], 'vector_rag': [], 'graph_rag': [], 'hybrid_ahs': []}
        
        for r in results:
            if r['has_gold']:  # Учитываем только с золотыми ответами
                for mode in mode_hrs:
                    mode_hrs[mode].append(r['metrics'][mode]['HR'])
        
        print("\n📊 Средний Hallucination Rate (с золотыми ответами):")
        avg_hrs = {}
        for mode, hrs in mode_hrs.items():
            if hrs:
                avg = sum(hrs) / len(hrs)
                avg_hrs[mode] = avg
                print(f"  {mode:12} : {avg:.1%}")
        
        if avg_hrs:
            print("\n🏆 Рейтинг (меньше = лучше):")
            sorted_modes = sorted(avg_hrs.items(), key=lambda x: x[1])
            for i, (mode, hr) in enumerate(sorted_modes, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "4️⃣"
                print(f"  {emoji} {mode:12} : HR = {hr:.1%}")
        
        print("\n💾 Результаты сохранены в correct_evaluation_results.json")

def main():
    evaluator = CorrectEvaluator()
    evaluator.evaluate_all()

if __name__ == "__main__":
    main()