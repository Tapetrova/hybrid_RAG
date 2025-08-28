#!/usr/bin/env python3
"""
Monitor Step 10 progress in real-time
"""

import time
import json
from pathlib import Path
from datetime import datetime

def monitor():
    predictions_file = Path("outputs/predictions_20250820_2246.jsonl")
    target = 25
    
    print("=" * 60)
    print("STEP 10 PROGRESS MONITOR")
    print("=" * 60)
    
    last_count = 0
    while True:
        if not predictions_file.exists():
            print(f"\r⏳ Waiting for file to be created...", end="")
            time.sleep(2)
            continue
            
        with open(predictions_file, 'r') as f:
            lines = f.readlines()
            count = len(lines)
        
        if count != last_count:
            # New progress
            if lines:
                last_data = json.loads(lines[-1])
                last_id = last_data['id']
                last_hr = last_data['metrics_sample']['HR']
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Progress: {count}/{target} | Last: {last_id} | HR: {last_hr:.3f}     ")
            last_count = count
            
            if count >= target:
                print("\n✅ Target reached! Evaluation should be finishing up...")
                break
        else:
            print(f"\r⏳ {count}/{target} questions processed... waiting for next", end="")
        
        time.sleep(5)
    
    # Wait for final reports
    print("\nWaiting for final reports generation...")
    time.sleep(10)
    
    # Check for reports
    csv_files = list(Path("outputs").glob("metrics_*2246*.csv"))
    md_files = list(Path("outputs").glob("metrics_*2246*.md"))
    
    if csv_files and md_files:
        print(f"✅ Reports generated: {csv_files[0].name}, {md_files[0].name}")
    else:
        print("⚠️ Reports not found yet")

if __name__ == "__main__":
    monitor()