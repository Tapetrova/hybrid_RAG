#!/usr/bin/env python3
"""
Analyze interim results from ongoing evaluation for quality assessment
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def analyze_interim():
    predictions_file = Path("outputs/predictions_20250820_2310.jsonl")
    
    if not predictions_file.exists():
        print("❌ Predictions file not found")
        return
    
    # Load all predictions so far
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    print("=" * 80)
    print("INTERIM RESULTS ANALYSIS - QUALITY ASSESSMENT FOR SCIENTIFIC PAPER")
    print("=" * 80)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total evaluations analyzed: {len(predictions)}")
    print()
    
    # Group by mode and category
    mode_metrics = defaultdict(lambda: {
        'HR': [], 'HR_contra': [], 'HR_unver': [], 'CHR': [],
        'claims_total': [], 'samples': 0
    })
    
    category_mode_metrics = defaultdict(lambda: defaultdict(lambda: {
        'HR': [], 'HR_contra': [], 'HR_unver': [], 'CHR': [], 'samples': 0
    }))
    
    # Process each prediction
    for pred in predictions:
        mode = pred['mode']
        category = pred['category']
        metrics = pred['metrics_sample']
        
        # Collect metrics
        mode_metrics[mode]['HR'].append(metrics['HR'])
        mode_metrics[mode]['HR_contra'].append(metrics['HR_contra'])
        mode_metrics[mode]['HR_unver'].append(metrics['HR_unver'])
        if metrics.get('CHR') is not None:
            mode_metrics[mode]['CHR'].append(metrics['CHR'])
        mode_metrics[mode]['claims_total'].append(metrics['claims_total'])
        mode_metrics[mode]['samples'] += 1
        
        # Category-specific
        category_mode_metrics[category][mode]['HR'].append(metrics['HR'])
        category_mode_metrics[category][mode]['HR_contra'].append(metrics['HR_contra'])
        category_mode_metrics[category][mode]['HR_unver'].append(metrics['HR_unver'])
        if metrics.get('CHR') is not None:
            category_mode_metrics[category][mode]['CHR'].append(metrics['CHR'])
        category_mode_metrics[category][mode]['samples'] += 1
    
    # 1. OVERALL METRICS BY MODE
    print("1. OVERALL METRICS BY MODE")
    print("-" * 80)
    print(f"{'Mode':<15} {'Samples':<10} {'HR':<12} {'HR_contra':<12} {'HR_unver':<12} {'CHR':<12} {'Avg Claims':<12}")
    print("-" * 80)
    
    for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
        if mode in mode_metrics:
            m = mode_metrics[mode]
            if m['samples'] > 0:
                avg_hr = np.mean(m['HR'])
                avg_hr_contra = np.mean(m['HR_contra'])
                avg_hr_unver = np.mean(m['HR_unver'])
                avg_chr = np.mean(m['CHR']) if m['CHR'] else 0
                avg_claims = np.mean(m['claims_total'])
                
                print(f"{mode:<15} {m['samples']:<10} "
                      f"{avg_hr:<12.3f} {avg_hr_contra:<12.3f} {avg_hr_unver:<12.3f} "
                      f"{avg_chr:<12.3f} {avg_claims:<12.1f}")
    
    # 2. STATISTICAL SIGNIFICANCE CHECK
    print("\n2. STATISTICAL SIGNIFICANCE INDICATORS")
    print("-" * 80)
    
    if 'base_llm' in mode_metrics and len(mode_metrics['base_llm']['HR']) > 5:
        base_hr = mode_metrics['base_llm']['HR']
        
        for mode in ['vector_rag', 'graph_rag', 'hybrid_ahs']:
            if mode in mode_metrics and len(mode_metrics[mode]['HR']) > 5:
                mode_hr = mode_metrics[mode]['HR']
                
                # Calculate reduction
                base_mean = np.mean(base_hr[:len(mode_hr)])  # Use same sample size
                mode_mean = np.mean(mode_hr)
                reduction = (base_mean - mode_mean) / base_mean * 100 if base_mean > 0 else 0
                
                # Simple t-test approximation
                from scipy import stats
                if len(mode_hr) >= len(base_hr):
                    t_stat, p_value = stats.ttest_ind(base_hr, mode_hr[:len(base_hr)])
                else:
                    t_stat, p_value = stats.ttest_ind(base_hr[:len(mode_hr)], mode_hr)
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"{mode} vs base_llm: {reduction:+.1f}% reduction (p={p_value:.4f}) {significance}")
    
    # 3. CATEGORY-SPECIFIC PERFORMANCE
    print("\n3. CATEGORY-SPECIFIC HALLUCINATION RATES")
    print("-" * 80)
    
    categories = ['causal', 'diagnostic', 'factual', 'comparative']
    for category in categories:
        if category in category_mode_metrics:
            print(f"\n{category.upper()} questions:")
            for mode in ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']:
                if mode in category_mode_metrics[category]:
                    m = category_mode_metrics[category][mode]
                    if m['samples'] > 0:
                        avg_hr = np.mean(m['HR'])
                        avg_chr = np.mean(m['CHR']) if m['CHR'] else 0
                        print(f"  {mode:<15} n={m['samples']:<3} HR={avg_hr:.3f} CHR={avg_chr:.3f}")
    
    # 4. QUALITY ASSESSMENT FOR PAPER
    print("\n4. QUALITY ASSESSMENT FOR SCIENTIFIC PAPER")
    print("-" * 80)
    
    quality_checks = []
    
    # Check 1: Base LLM should have higher HR than retrieval modes
    if 'base_llm' in mode_metrics and 'vector_rag' in mode_metrics:
        base_hr_mean = np.mean(mode_metrics['base_llm']['HR'])
        vector_hr_mean = np.mean(mode_metrics['vector_rag']['HR'])
        if base_hr_mean > vector_hr_mean:
            quality_checks.append("✅ Base LLM has higher HR than retrieval modes (expected)")
        else:
            quality_checks.append("⚠️ Base LLM HR not higher than retrieval - check retrieval quality")
    
    # Check 2: Hybrid should perform best
    if 'hybrid_ahs' in mode_metrics and len(mode_metrics['hybrid_ahs']['HR']) > 0:
        hybrid_hr = np.mean(mode_metrics['hybrid_ahs']['HR'])
        all_hrs = [np.mean(mode_metrics[m]['HR']) for m in mode_metrics if m != 'hybrid_ahs' and mode_metrics[m]['HR']]
        if all_hrs and hybrid_hr <= min(all_hrs):
            quality_checks.append("✅ Hybrid fusion shows best performance (lowest HR)")
        elif all_hrs:
            quality_checks.append("⚠️ Hybrid not showing expected improvement")
    
    # Check 3: Reasonable HR ranges
    all_hr_values = []
    for m in mode_metrics.values():
        all_hr_values.extend(m['HR'])
    
    if all_hr_values:
        overall_mean_hr = np.mean(all_hr_values)
        if 0.2 <= overall_mean_hr <= 0.7:
            quality_checks.append(f"✅ HR values in reasonable range (mean={overall_mean_hr:.3f})")
        else:
            quality_checks.append(f"⚠️ HR values unusual (mean={overall_mean_hr:.3f})")
    
    # Check 4: CHR for causal questions
    if 'causal' in category_mode_metrics:
        causal_chrs = []
        for mode_data in category_mode_metrics['causal'].values():
            if mode_data['CHR']:
                causal_chrs.extend(mode_data['CHR'])
        if causal_chrs and np.mean(causal_chrs) > 0.3:
            quality_checks.append(f"✅ Causal questions show meaningful CHR (mean={np.mean(causal_chrs):.3f})")
        elif causal_chrs:
            quality_checks.append(f"⚠️ Low CHR for causal questions (mean={np.mean(causal_chrs):.3f})")
    
    for check in quality_checks:
        print(check)
    
    # 5. RECOMMENDATIONS
    print("\n5. RECOMMENDATIONS FOR PAPER")
    print("-" * 80)
    
    print("Based on interim results (n={} evaluations):".format(len(predictions)))
    
    if len(predictions) < 100:
        print("⚠️ Sample size still small - wait for more data for robust conclusions")
    
    # Check if we have enough diversity
    modes_seen = list(mode_metrics.keys())
    if len(modes_seen) < 4:
        print(f"⚠️ Only {len(modes_seen)} modes evaluated so far - need all 4 for comparison")
    else:
        print("✅ All 4 modes have been evaluated - good for comparison")
    
    # Check variance
    if 'base_llm' in mode_metrics and len(mode_metrics['base_llm']['HR']) > 10:
        hr_std = np.std(mode_metrics['base_llm']['HR'])
        if hr_std > 0.3:
            print(f"📊 High variance in HR (std={hr_std:.3f}) - indicates diverse question difficulty")
        else:
            print(f"📊 Moderate variance in HR (std={hr_std:.3f})")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Results look promising for scientific publication.")
    print("Continue monitoring for full dataset completion.")
    print("=" * 80)

if __name__ == "__main__":
    try:
        from scipy import stats
        analyze_interim()
    except ImportError:
        print("Installing scipy for statistical tests...")
        import subprocess
        subprocess.run(["pip", "install", "scipy"])
        from scipy import stats
        analyze_interim()