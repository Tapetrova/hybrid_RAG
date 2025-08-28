#!/usr/bin/env python3
"""
Main entry point for Hybrid RAG analysis
"""

import os

def main():
    print("="*60)
    print("HYBRID RAG - ANALYSIS PIPELINE")
    print("="*60)
    
    os.chdir('dataset_creation/analysis')
    
    print("\n1. Dataset analysis...")
    os.system('python comprehensive_analysis.py')
    
    print("\n2. Hallucination evaluation...")
    os.system('python hallucination_full_api_706.py')
    
    print("\n3. RAC metrics...")
    os.system('python calculate_rac_corrected.py')
    
    print("\n4. Generating figures...")
    os.system('python generate_paper_figures.py')
    
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()