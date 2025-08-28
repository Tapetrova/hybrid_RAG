#!/usr/bin/env python3
"""
Step 9 - Sanity checks for evaluation results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load metrics CSV file"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")
    return pd.read_csv(csv_path)


def check_base_llm_higher_hr(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check if base_llm has higher HR than at least one retrieval mode
    
    Returns:
        (passed, message)
    """
    overall_df = df[df['split'] == 'overall'].copy()
    
    if 'base_llm' not in overall_df['mode'].values:
        return False, "base_llm mode not found in results"
    
    base_hr = overall_df[overall_df['mode'] == 'base_llm']['HR'].values[0]
    retrieval_modes = ['vector_rag', 'graph_rag', 'hybrid_ahs']
    
    retrieval_hrs = []
    for mode in retrieval_modes:
        if mode in overall_df['mode'].values:
            hr = overall_df[overall_df['mode'] == mode]['HR'].values[0]
            retrieval_hrs.append((mode, float(hr)))
    
    if not retrieval_hrs:
        return False, "No retrieval modes found in results"
    
    # Check if base_llm HR is higher than at least one retrieval mode
    base_hr_float = float(base_hr)
    lower_hrs = [(mode, hr) for mode, hr in retrieval_hrs if hr < base_hr_float]
    
    if lower_hrs:
        best_mode, best_hr = min(lower_hrs, key=lambda x: x[1])
        improvement = ((base_hr_float - best_hr) / base_hr_float) * 100
        message = f"✓ base_llm HR ({base_hr:.3f}) > {best_mode} HR ({best_hr:.3f}) - {improvement:.1f}% improvement"
        return True, message
    else:
        message = f"⚠️ WARNING: base_llm HR ({base_hr:.3f}) is NOT higher than any retrieval mode"
        return False, message


def check_causal_diagnostic_chr(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check if CHR for graph_rag/hybrid_ahs is lower than base_llm 
    for causal and diagnostic categories
    
    Returns:
        (passed, list of messages)
    """
    messages = []
    warnings = 0
    checks = 0
    
    categories = ['causal_q', 'diagnostic_q']
    graph_modes = ['graph_rag', 'hybrid_ahs']
    
    for category in categories:
        cat_df = df[df['split'] == category].copy()
        
        if cat_df.empty:
            continue
            
        # Get base_llm CHR for this category
        base_rows = cat_df[cat_df['mode'] == 'base_llm']
        if base_rows.empty:
            continue
            
        base_chr = float(base_rows['CHR'].values[0])
        
        # Check graph-based modes
        for mode in graph_modes:
            mode_rows = cat_df[cat_df['mode'] == mode]
            if mode_rows.empty:
                continue
                
            mode_chr = float(mode_rows['CHR'].values[0])
            checks += 1
            
            if mode_chr < base_chr:
                improvement = ((base_chr - mode_chr) / base_chr) * 100 if base_chr > 0 else 0
                messages.append(f"  ✓ {category[:-2]}: {mode} CHR ({mode_chr:.3f}) < base_llm CHR ({base_chr:.3f}) - {improvement:.1f}% better")
            else:
                warnings += 1
                messages.append(f"  ⚠️ {category[:-2]}: {mode} CHR ({mode_chr:.3f}) >= base_llm CHR ({base_chr:.3f})")
    
    if checks == 0:
        return False, ["No causal/diagnostic comparisons available"]
    
    passed = warnings < checks  # Pass if at least some improvements seen
    return passed, messages


def run_sanity_checks(metrics_csv: Path) -> bool:
    """
    Run all sanity checks on evaluation metrics
    
    Returns:
        True if all critical checks pass, False otherwise
    """
    print("="*60)
    print("STEP 9 - SANITY CHECKS")
    print("="*60)
    
    # Load metrics
    try:
        df = load_metrics(metrics_csv)
        print(f"\nLoaded metrics from: {metrics_csv}")
        print(f"Modes found: {df['mode'].unique().tolist()}")
        print(f"Splits found: {df['split'].unique().tolist()}")
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return False
    
    # Check 1: base_llm should have higher HR
    print("\n" + "-"*40)
    print("Check 1: base_llm HR vs retrieval modes")
    print("-"*40)
    
    check1_passed, check1_msg = check_base_llm_higher_hr(df)
    print(check1_msg)
    
    # Check 2: CHR for causal/diagnostic
    print("\n" + "-"*40)
    print("Check 2: CHR for causal/diagnostic categories")
    print("-"*40)
    
    check2_passed, check2_msgs = check_causal_diagnostic_chr(df)
    for msg in check2_msgs:
        print(msg)
    
    # Overall summary
    print("\n" + "="*60)
    print("SANITY CHECK SUMMARY")
    print("="*60)
    
    all_passed = True
    
    if check1_passed:
        print("✅ Check 1 PASSED: base_llm has higher HR than retrieval")
    else:
        print("❌ Check 1 FAILED: base_llm does NOT have higher HR")
        all_passed = False
    
    if check2_passed:
        print("✅ Check 2 PASSED: Graph modes show CHR improvements")
    else:
        print("⚠️ Check 2 WARNING: Graph modes may not be improving CHR")
        # This is a warning, not a hard fail
    
    print("\n" + "-"*40)
    
    if all_passed:
        print("✅ All critical sanity checks PASSED")
        print("   Ready to proceed with full evaluation")
    else:
        print("⚠️ ATTENTION: Sanity checks detected issues!")
        print("   Please review the results above.")
        print("   Consult with Tatiana before proceeding to full run.")
        print("\n   Possible issues:")
        print("   - Knowledge Manager services may not be running")
        print("   - Retrieval may be returning empty results")
        print("   - Model configurations may need adjustment")
    
    return all_passed


def check_latest_metrics():
    """Check the most recent metrics file"""
    output_dir = Path("outputs")
    
    # Find most recent metrics CSV
    metrics_files = list(output_dir.glob("metrics_*.csv"))
    if not metrics_files:
        print("No metrics files found in outputs/")
        return False
    
    latest = max(metrics_files, key=lambda p: p.stat().st_mtime)
    print(f"Checking latest metrics: {latest.name}")
    
    return run_sanity_checks(latest)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Specific file provided
        metrics_file = Path(sys.argv[1])
        success = run_sanity_checks(metrics_file)
    else:
        # Check latest
        success = check_latest_metrics()
    
    sys.exit(0 if success else 1)