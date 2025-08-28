#!/usr/bin/env python3
"""
Monitor Step 11 - Full evaluation progress
"""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta

def monitor():
    predictions_file = Path("outputs/predictions_20250820_2310.jsonl")
    total_evaluations = 706 * 4  # 706 questions × 4 modes
    
    print("=" * 70)
    print("STEP 11 - FULL EVALUATION MONITOR")
    print("=" * 70)
    print(f"Total evaluations: {total_evaluations} (706 questions × 4 modes)")
    print(f"Estimated time: ~24-28 hours at current pace")
    print("-" * 70)
    
    start_time = datetime.now()
    last_count = 0
    
    while True:
        if not predictions_file.exists():
            print(f"\r⏳ Waiting for file...", end="")
            time.sleep(5)
            continue
            
        with open(predictions_file, 'r') as f:
            lines = f.readlines()
            count = len(lines)
        
        if count != last_count:
            # Calculate progress and ETA
            elapsed = datetime.now() - start_time
            if count > 0:
                rate = count / elapsed.total_seconds()
                remaining = (total_evaluations - count) / rate if rate > 0 else 0
                eta = datetime.now() + timedelta(seconds=remaining)
                
                # Get last processed item
                if lines:
                    last_data = json.loads(lines[-1])
                    last_id = last_data['id']
                    last_mode = last_data['mode']
                    last_hr = last_data['metrics_sample']['HR']
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Progress: {count}/{total_evaluations} ({count*100/total_evaluations:.1f}%) | "
                          f"Last: {last_id}/{last_mode} | HR: {last_hr:.3f} | "
                          f"ETA: {eta.strftime('%m/%d %H:%M')}     ")
                
            last_count = count
            
            # Save checkpoint info every 100 evaluations
            if count % 100 == 0 and count > 0:
                with open("outputs/checkpoint_info.txt", "w") as f:
                    f.write(f"Checkpoint at {count}/{total_evaluations}\n")
                    f.write(f"Time: {datetime.now()}\n")
                    f.write(f"Elapsed: {elapsed}\n")
                    f.write(f"ETA: {eta}\n")
            
            if count >= total_evaluations:
                print(f"\n✅ All {total_evaluations} evaluations complete!")
                break
        else:
            print(f"\r⏳ {count}/{total_evaluations} processed... waiting", end="")
        
        time.sleep(10)
    
    print("\n" + "=" * 70)
    print("Waiting for final report generation...")
    time.sleep(30)
    
    # Check for final reports
    csv_files = list(Path("outputs").glob("metrics_*2310*.csv"))
    md_files = list(Path("outputs").glob("metrics_*2310*.md"))
    
    if csv_files and md_files:
        print(f"✅ Reports generated: {csv_files[0].name}, {md_files[0].name}")
        
        # Show summary
        with open(csv_files[0], 'r') as f:
            lines = f.readlines()
            print("\nFinal Metrics Summary:")
            print("-" * 70)
            for line in lines[:5]:  # Header + overall metrics
                print(line.strip())
    else:
        print("⚠️ Reports not found yet")

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Evaluation continues in background.")
        print("To check status: wc -l outputs/predictions_20250820_2310.jsonl")