#!/usr/bin/env python3
"""
Main entry point for Hybrid RAG analysis
"""

import os
import sys

def main():
    print("="*60)
    print("HYBRID RAG - MAIN ANALYSIS PIPELINE")
    print("="*60)
    
    # Change to analysis directory
    os.chdir('dataset_creation/analysis')
    
    print("\n1. Running comprehensive dataset analysis...")
    os.system('python comprehensive_analysis.py')
    
    print("\n2. Calculating RAC metrics...")
    os.system('python calculate_rac_corrected.py')
    
    print("\n3. Validating category classifier...")
    os.system('python validate_category_classifier_simple.py')
    
    print("\n4. Generating all visualizations...")
    os.system('python create_all_visualizations.py')
    
    print("\n" + "="*60)
    print("✅ Analysis complete! Check dataset_creation/analysis/ for outputs")
    print("="*60)

if __name__ == "__main__":
    main()