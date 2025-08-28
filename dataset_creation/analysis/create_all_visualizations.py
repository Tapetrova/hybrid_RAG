#!/usr/bin/env python3
"""
Main script to generate all visualizations for the Hybrid RAG paper
Run this to reproduce all figures
"""

import os
import sys

def main():
    """Generate all visualizations"""
    
    scripts_to_run = [
        'create_scientific_category_chart.py',       # Category distribution
        'create_wfas_2_5_performance_bar.py',        # WFAS performance comparison
        'create_wfas_2_5_heatmap.py',               # WFAS heatmap
        'create_wfas_2_5_performance_vs_length_limited.py',  # Performance vs length
        'create_rac_cs_charts.py',                   # RAC and consistency charts
        'create_publication_visualizations.py'       # All publication figures
    ]
    
    print("="*60)
    print("GENERATING ALL VISUALIZATIONS FOR HYBRID RAG")
    print("="*60)
    
    for script in scripts_to_run:
        if os.path.exists(script):
            print(f"\n► Running {script}...")
            try:
                exec(open(script).read(), {'__name__': '__main__'})
                print(f"  ✓ {script} completed")
            except Exception as e:
                print(f"  ✗ Error in {script}: {e}")
        else:
            print(f"  ⚠ {script} not found")
    
    print("\n" + "="*60)
    print("✅ All visualizations generated!")
    print("Check the current directory for output files")
    print("="*60)

if __name__ == "__main__":
    main()