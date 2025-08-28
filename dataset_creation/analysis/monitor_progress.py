#!/usr/bin/env python3
"""
Мониторинг прогресса оценки 100 вопросов
"""

import json
import os
import time
from datetime import datetime

def monitor():
    checkpoint_file = "eval_200_checkpoint.json"
    
    print("="*60)
    print("МОНИТОРИНГ ПРОГРЕССА ОЦЕНКИ")
    print("="*60)
    
    while True:
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                processed = len(data.get('processed_ids', []))
                timestamp = data.get('timestamp', 'unknown')
                
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
                print(f"✅ Обработано: {processed}/100 вопросов")
                print(f"📊 Прогресс: {processed}%")
                print(f"💾 Последнее сохранение: {timestamp}")
                
                if processed >= 100:
                    print("\n🎉 ОЦЕНКА ЗАВЕРШЕНА!")
                    break
                    
            except Exception as e:
                print(f"⚠️ Ошибка чтения checkpoint: {e}")
        else:
            print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
            print("⏳ Ожидаем начала обработки...")
        
        # Проверяем каждые 30 секунд
        time.sleep(30)

if __name__ == "__main__":
    monitor()