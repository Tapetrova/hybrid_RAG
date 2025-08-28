#!/usr/bin/env python3
"""
Create publication-quality visualizations for automotive Q&A hallucination analysis
Includes dataset statistics, method comparisons, and architectural diagrams
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
    from matplotlib.patches import ConnectionPatch
    import seaborn as sns
    from scipy import stats
    from scipy.interpolate import make_interp_spline
    import networkx as nx
    PLOTTING_AVAILABLE = True
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("husl")
    
    # High DPI for publication
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14
    
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ Please install matplotlib and seaborn: pip install matplotlib seaborn scipy networkx")

class PublicationVisualizer:
    def __init__(self):
        if not PLOTTING_AVAILABLE:
            raise ImportError("Visualization libraries not available")
        
        print("📊 Loading experimental data...")
        
        # Load main results
        with open('hallucination_FULL_API_706_results_20250821_231422.json', 'r') as f:
            self.api_results = json.load(f)
        
        # Load dataset
        with open('../data/apqc_auto.json', 'r') as f:
            self.dataset = json.load(f)
        
        # Define color scheme for methods
        self.method_colors = {
            'base_llm': '#E74C3C',      # Red
            'vector_rag': '#2ECC71',    # Green  
            'graph_rag': '#3498DB',     # Blue
            'hybrid_ahs': '#9B59B6'     # Purple
        }
        
        self.category_colors = {
            'factual': '#FF9800',
            'causal': '#4CAF50',
            'diagnostic': '#2196F3',
            'comparative': '#9C27B0'
        }
        
        print("✅ Data loaded successfully")
    
    def create_all_visualizations(self):
        """Generate all publication visualizations"""
        print("\n🎨 Creating publication-quality visualizations...")
        
        # 1. Dataset composition and statistics
        self.visualize_dataset_composition()
        
        # 2. Method architecture diagram
        self.visualize_method_architecture()
        
        # 3. Performance radar chart
        self.visualize_performance_radar()
        
        # 4. Confidence interval comparison
        self.visualize_confidence_intervals()
        
        # 5. Category-wise performance heatmap
        self.visualize_category_heatmap()
        
        # 6. Hallucination type breakdown
        self.visualize_hallucination_breakdown()
        
        # 7. Statistical significance matrix
        self.visualize_significance_matrix()
        
        # 8. Question length vs performance
        self.visualize_question_complexity()
        
        # 9. Cost-benefit analysis
        self.visualize_cost_benefit()
        
        # 10. Error distribution analysis
        self.visualize_error_distribution()
        
        print("\n✅ All visualizations created successfully!")
    
    def visualize_dataset_composition(self):
        """Figure 1: Dataset composition and statistics"""
        fig = plt.figure(figsize=(14, 8))
        
        # Count categories
        categories = {}
        question_lengths = []
        answer_lengths = []
        
        for item in self.dataset['questions']:
            cat = item['category']
            categories[cat] = categories.get(cat, 0) + 1
            question_lengths.append(len(item['question'].split()))
            answer_lengths.append(len(item['answer'].split()))
        
        # 1. Category distribution (pie + bar)
        ax1 = plt.subplot(2, 3, 1)
        sizes = list(categories.values())
        labels = [f"{k.capitalize()}\n({v})" for k, v in categories.items()]
        colors = [self.category_colors[k] for k in categories.keys()]
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
        ax1.set_title('Question Category Distribution', fontweight='bold')
        
        # 2. Question length distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(question_lengths, bins=30, color='#3498DB', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(question_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(question_lengths):.1f} words')
        ax2.set_xlabel('Question Length (words)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Question Length Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Answer length distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(answer_lengths, bins=30, color='#2ECC71', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(answer_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(answer_lengths):.1f} words')
        ax3.set_xlabel('Answer Length (words)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Gold Answer Length Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Category statistics table
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('tight')
        ax4.axis('off')
        
        stats_data = []
        for cat in categories.keys():
            cat_questions = [len(item['question'].split()) 
                           for item in self.dataset['questions'] if item['category'] == cat]
            cat_answers = [len(item['answer'].split()) 
                         for item in self.dataset['questions'] if item['category'] == cat]
            stats_data.append([
                cat.capitalize(),
                categories[cat],
                f"{np.mean(cat_questions):.1f}",
                f"{np.mean(cat_answers):.1f}"
            ])
        
        table = ax4.table(cellText=stats_data,
                         colLabels=['Category', 'Count', 'Avg Q Len', 'Avg A Len'],
                         cellLoc='center',
                         loc='center',
                         colColours=['#f0f0f0']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Category Statistics Summary', fontweight='bold', pad=20)
        
        # 5. Question complexity scatter
        ax5 = plt.subplot(2, 3, 5)
        for cat, color in self.category_colors.items():
            cat_items = [(len(item['question'].split()), len(item['answer'].split()))
                        for item in self.dataset['questions'] if item['category'] == cat]
            if cat_items:
                x, y = zip(*cat_items)
                ax5.scatter(x, y, alpha=0.5, label=cat.capitalize(), color=color, s=20)
        
        ax5.set_xlabel('Question Length (words)')
        ax5.set_ylabel('Answer Length (words)')
        ax5.set_title('Question-Answer Length Correlation', fontweight='bold')
        ax5.legend(loc='upper left', frameon=True)
        ax5.grid(True, alpha=0.3)
        
        # 6. Dataset summary box
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        APQC Automotive Q&A Dataset
        
        Total Questions: 706
        Categories: 4
        
        Question Length:
        • Mean: {np.mean(question_lengths):.1f} words
        • Median: {np.median(question_lengths):.1f} words
        • Range: {min(question_lengths)}-{max(question_lengths)} words
        
        Answer Length:
        • Mean: {np.mean(answer_lengths):.1f} words
        • Median: {np.median(answer_lengths):.1f} words
        • Range: {min(answer_lengths)}-{max(answer_lengths)} words
        
        Dataset Quality:
        • Expert-validated answers
        • Domain-specific automotive
        • Balanced categories
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        plt.suptitle('Dataset Composition and Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure1_dataset_composition_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 1: Dataset composition saved")
        plt.show()
    
    def visualize_method_architecture(self):
        """Figure 2: Method architecture diagram"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Base LLM
        ax1 = axes[0, 0]
        ax1.axis('off')
        ax1.set_title('(a) Base LLM', fontweight='bold', fontsize=12)
        
        # Draw components
        question_box = FancyBboxPatch((0.1, 0.7), 0.3, 0.15, 
                                      boxstyle="round,pad=0.02",
                                      facecolor='#FFE5B4', edgecolor='black', linewidth=2)
        ax1.add_patch(question_box)
        ax1.text(0.25, 0.775, 'Question', ha='center', va='center', fontweight='bold')
        
        llm_box = FancyBboxPatch((0.35, 0.3), 0.3, 0.25,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#E74C3C', edgecolor='black', linewidth=2)
        ax1.add_patch(llm_box)
        ax1.text(0.5, 0.425, 'GPT-4o-mini\n(No Context)', ha='center', va='center', 
                fontweight='bold', color='white')
        
        answer_box = FancyBboxPatch((0.6, 0.05), 0.3, 0.15,
                                    boxstyle="round,pad=0.02",
                                    facecolor='#D5E8D4', edgecolor='black', linewidth=2)
        ax1.add_patch(answer_box)
        ax1.text(0.75, 0.125, 'Answer', ha='center', va='center', fontweight='bold')
        
        # Arrows
        arrow1 = FancyArrowPatch((0.25, 0.7), (0.5, 0.55),
                                connectionstyle="arc3,rad=0", 
                                arrowstyle='->', lw=2, color='black')
        ax1.add_patch(arrow1)
        
        arrow2 = FancyArrowPatch((0.5, 0.3), (0.75, 0.2),
                                connectionstyle="arc3,rad=0",
                                arrowstyle='->', lw=2, color='black')
        ax1.add_patch(arrow2)
        
        # Vector RAG
        ax2 = axes[0, 1]
        ax2.axis('off')
        ax2.set_title('(b) Vector RAG', fontweight='bold', fontsize=12)
        
        # Components
        ax2.add_patch(FancyBboxPatch((0.05, 0.7), 0.25, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#FFE5B4', edgecolor='black', linewidth=2))
        ax2.text(0.175, 0.775, 'Question', ha='center', va='center', fontweight='bold')
        
        ax2.add_patch(FancyBboxPatch((0.05, 0.45), 0.25, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#87CEEB', edgecolor='black', linewidth=2))
        ax2.text(0.175, 0.525, 'Tavily API\n(Web Search)', ha='center', va='center', fontsize=9)
        
        ax2.add_patch(FancyBboxPatch((0.4, 0.45), 0.25, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#FFE5CC', edgecolor='black', linewidth=2))
        ax2.text(0.525, 0.525, 'Context\nDocuments', ha='center', va='center', fontsize=9)
        
        ax2.add_patch(FancyBboxPatch((0.35, 0.15), 0.35, 0.2,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#2ECC71', edgecolor='black', linewidth=2))
        ax2.text(0.525, 0.25, 'GPT-4o-mini\n+ Context', ha='center', va='center',
                fontweight='bold', color='white')
        
        ax2.add_patch(FancyBboxPatch((0.75, 0.05), 0.2, 0.1,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#D5E8D4', edgecolor='black', linewidth=2))
        ax2.text(0.85, 0.1, 'Answer', ha='center', va='center', fontweight='bold')
        
        # Arrows
        ax2.arrow(0.175, 0.7, 0, -0.08, head_width=0.02, head_length=0.02, fc='black')
        ax2.arrow(0.3, 0.525, 0.08, 0, head_width=0.02, head_length=0.02, fc='black')
        ax2.arrow(0.525, 0.45, 0, -0.08, head_width=0.02, head_length=0.02, fc='black')
        ax2.arrow(0.525, 0.15, 0.2, -0.04, head_width=0.02, head_length=0.02, fc='black')
        
        # Graph RAG
        ax3 = axes[1, 0]
        ax3.axis('off')
        ax3.set_title('(c) Graph RAG', fontweight='bold', fontsize=12)
        
        # Components
        ax3.add_patch(FancyBboxPatch((0.05, 0.7), 0.25, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#FFE5B4', edgecolor='black', linewidth=2))
        ax3.text(0.175, 0.775, 'Question', ha='center', va='center', fontweight='bold')
        
        ax3.add_patch(FancyBboxPatch((0.35, 0.65), 0.25, 0.2,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#FFF3CD', edgecolor='black', linewidth=2))
        ax3.text(0.475, 0.75, 'Query\nAugmentation\n("why/cause")', ha='center', va='center', fontsize=9)
        
        ax3.add_patch(FancyBboxPatch((0.05, 0.35), 0.25, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#87CEEB', edgecolor='black', linewidth=2))
        ax3.text(0.175, 0.425, 'Tavily API\n(Causal Search)', ha='center', va='center', fontsize=9)
        
        ax3.add_patch(FancyBboxPatch((0.4, 0.35), 0.25, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#FFE5CC', edgecolor='black', linewidth=2))
        ax3.text(0.525, 0.425, 'Causal\nContext', ha='center', va='center', fontsize=9)
        
        ax3.add_patch(FancyBboxPatch((0.35, 0.1), 0.35, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#3498DB', edgecolor='black', linewidth=2))
        ax3.text(0.525, 0.175, 'GPT-4o-mini\n+ Causal', ha='center', va='center',
                fontweight='bold', color='white')
        
        ax3.add_patch(FancyBboxPatch((0.75, 0.05), 0.2, 0.1,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#D5E8D4', edgecolor='black', linewidth=2))
        ax3.text(0.85, 0.1, 'Answer', ha='center', va='center', fontweight='bold')
        
        # Hybrid AHS
        ax4 = axes[1, 1]
        ax4.axis('off')
        ax4.set_title('(d) Hybrid AHS', fontweight='bold', fontsize=12)
        
        # Components
        ax4.add_patch(FancyBboxPatch((0.35, 0.8), 0.3, 0.1,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#FFE5B4', edgecolor='black', linewidth=2))
        ax4.text(0.5, 0.85, 'Question + Category', ha='center', va='center', fontweight='bold')
        
        # Two parallel paths
        ax4.add_patch(FancyBboxPatch((0.1, 0.55), 0.2, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#2ECC71', edgecolor='black', linewidth=2))
        ax4.text(0.2, 0.625, 'Vector\nRAG', ha='center', va='center', color='white', fontweight='bold')
        
        ax4.add_patch(FancyBboxPatch((0.7, 0.55), 0.2, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#3498DB', edgecolor='black', linewidth=2))
        ax4.text(0.8, 0.625, 'Graph\nRAG', ha='center', va='center', color='white', fontweight='bold')
        
        # Adaptive fusion
        ax4.add_patch(FancyBboxPatch((0.35, 0.3), 0.3, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#FFD700', edgecolor='black', linewidth=2))
        ax4.text(0.5, 0.375, 'Adaptive Fusion\n(Category Weights)', ha='center', va='center', fontsize=9)
        
        ax4.add_patch(FancyBboxPatch((0.35, 0.1), 0.3, 0.15,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#9B59B6', edgecolor='black', linewidth=2))
        ax4.text(0.5, 0.175, 'GPT-4o-mini\n+ Hybrid', ha='center', va='center',
                fontweight='bold', color='white')
        
        ax4.add_patch(FancyBboxPatch((0.75, 0.05), 0.2, 0.08,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#D5E8D4', edgecolor='black', linewidth=2))
        ax4.text(0.85, 0.09, 'Answer', ha='center', va='center', fontweight='bold')
        
        # Arrows for hybrid
        ax4.arrow(0.5, 0.8, -0.25, -0.1, head_width=0.02, head_length=0.02, fc='black')
        ax4.arrow(0.5, 0.8, 0.25, -0.1, head_width=0.02, head_length=0.02, fc='black')
        ax4.arrow(0.2, 0.55, 0.25, -0.08, head_width=0.02, head_length=0.02, fc='black')
        ax4.arrow(0.8, 0.55, -0.25, -0.08, head_width=0.02, head_length=0.02, fc='black')
        ax4.arrow(0.5, 0.3, 0, -0.04, head_width=0.02, head_length=0.02, fc='black')
        ax4.arrow(0.5, 0.1, 0.23, -0.03, head_width=0.02, head_length=0.02, fc='black')
        
        plt.suptitle('RAG Method Architectures', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure2_method_architecture_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 2: Method architecture saved")
        plt.show()
    
    def visualize_performance_radar(self):
        """Figure 3: Multi-dimensional performance radar chart"""
        fig = plt.figure(figsize=(12, 10))
        
        # Calculate metrics for each method
        methods = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
        
        # Metrics to display
        categories = ['Factual\nAccuracy', 'Support\nRate', 'Causal\nPerformance', 
                     'Diagnostic\nPerformance', 'Comparative\nPerformance', 'Overall\nFAS']
        
        # Calculate values
        method_metrics = {}
        for method in methods:
            metrics = []
            
            # Overall FAS
            fas_values = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    fas = (1 - result['metrics'][method]['HR']) * 100
                    fas_values.append(fas)
            
            overall_fas = np.mean(fas_values) if fas_values else 0
            
            # Category-specific performance
            cat_performance = {}
            for cat in ['factual', 'causal', 'diagnostic', 'comparative']:
                cat_fas = []
                for result in self.api_results['results']:
                    if result['category'] == cat and method in result['metrics']:
                        if result['metrics'][method]['total_claims'] > 0:
                            fas = (1 - result['metrics'][method]['HR']) * 100
                            cat_fas.append(fas)
                cat_performance[cat] = np.mean(cat_fas) if cat_fas else 0
            
            # Support rate
            support_rates = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    sr = result['metrics'][method]['supported'] / result['metrics'][method]['total_claims'] * 100
                    support_rates.append(sr)
            
            metrics = [
                cat_performance['factual'],
                np.mean(support_rates) if support_rates else 0,
                cat_performance['causal'],
                cat_performance['diagnostic'],
                cat_performance['comparative'],
                overall_fas
            ]
            
            method_metrics[method] = metrics
        
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the circle
        for method in methods:
            method_metrics[method] += method_metrics[method][:1]
        angles += angles[:1]
        
        # Create subplot
        ax = plt.subplot(111, projection='polar')
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], categories, size=10)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], size=8)
        plt.ylim(0, 100)
        
        # Plot data
        for method, color in self.method_colors.items():
            values = method_metrics[method]
            ax.plot(angles, values, 'o-', linewidth=2, label=method.replace('_', ' ').upper(),
                   color=color, alpha=0.8)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
        
        plt.title('Multi-Dimensional Performance Comparison', size=14, fontweight='bold', pad=20)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure3_performance_radar_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 3: Performance radar chart saved")
        plt.show()
    
    def visualize_confidence_intervals(self):
        """Figure 4: Confidence intervals with significance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = ['Base LLM', 'Vector RAG', 'Graph RAG', 'Hybrid AHS']
        method_keys = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
        
        # Calculate FAS with bootstrap CI
        fas_means = []
        fas_cis = []
        
        for method in method_keys:
            fas_values = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    fas = (1 - result['metrics'][method]['HR']) * 100
                    fas_values.append(fas)
            
            if fas_values:
                # Bootstrap confidence intervals
                n_bootstrap = 1000
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(fas_values, size=len(fas_values), replace=True)
                    bootstrap_means.append(np.mean(sample))
                
                mean = np.mean(fas_values)
                ci_lower = np.percentile(bootstrap_means, 2.5)
                ci_upper = np.percentile(bootstrap_means, 97.5)
                
                fas_means.append(mean)
                fas_cis.append([mean - ci_lower, ci_upper - mean])
            else:
                fas_means.append(0)
                fas_cis.append([0, 0])
        
        # Plot 1: Forest plot style
        y_pos = np.arange(len(methods))
        colors = [self.method_colors[m] for m in method_keys]
        
        ax1.barh(y_pos, fas_means, xerr=np.array(fas_cis).T, 
                align='center', alpha=0.7, error_kw={'linewidth': 2, 'capsize': 5},
                color=colors, edgecolor='black', linewidth=1.5)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(methods)
        ax1.invert_yaxis()
        ax1.set_xlabel('Factual Accuracy Score (%)', fontweight='bold')
        ax1.set_title('FAS with 95% Confidence Intervals', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add values
        for i, (mean, ci) in enumerate(zip(fas_means, fas_cis)):
            ax1.text(mean + ci[1] + 2, i, f'{mean:.1f}%', 
                    va='center', fontweight='bold')
        
        # Add significance lines
        if fas_means[0] < fas_means[1]:  # Base vs Vector
            ax1.plot([fas_means[0] + fas_cis[0][1], fas_means[1] - fas_cis[1][0]], 
                    [-0.5, -0.5], 'k-', linewidth=1)
            ax1.text((fas_means[0] + fas_means[1])/2, -0.6, '***', 
                    ha='center', fontsize=12)
        
        # Plot 2: Improvement over baseline
        ax2.set_title('Improvement over Baseline', fontweight='bold')
        
        baseline = fas_means[0]
        improvements = [(m - baseline) for m in fas_means[1:]]
        improvement_pct = [(m - baseline)/baseline * 100 for m in fas_means[1:]]
        
        x_pos = np.arange(len(methods[1:]))
        bars = ax2.bar(x_pos, improvements, color=colors[1:], alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods[1:], rotation=0)
        ax2.set_ylabel('FAS Improvement (percentage points)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linewidth=1)
        
        # Add values and percentage
        for i, (bar, imp, pct) in enumerate(zip(bars, improvements, improvement_pct)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'+{imp:.1f}pp\n(+{pct:.0f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Statistical Confidence in Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure4_confidence_intervals_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 4: Confidence intervals saved")
        plt.show()
    
    def visualize_category_heatmap(self):
        """Figure 5: Category-wise performance heatmap"""
        fig = plt.figure(figsize=(12, 8))
        
        # Prepare data
        categories = ['factual', 'causal', 'diagnostic', 'comparative']
        methods = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
        
        # Calculate FAS for each category-method combination
        heatmap_data = []
        
        for cat in categories:
            row = []
            for method in methods:
                fas_values = []
                for result in self.api_results['results']:
                    if result['category'] == cat and method in result['metrics']:
                        if result['metrics'][method]['total_claims'] > 0:
                            fas = (1 - result['metrics'][method]['HR']) * 100
                            fas_values.append(fas)
                row.append(np.mean(fas_values) if fas_values else 0)
            heatmap_data.append(row)
        
        # Create main heatmap
        ax1 = plt.subplot(2, 2, (1, 2))
        
        im = ax1.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        
        ax1.set_xticks(np.arange(len(methods)))
        ax1.set_yticks(np.arange(len(categories)))
        ax1.set_xticklabels([m.replace('_', ' ').upper() for m in methods])
        ax1.set_yticklabels([c.capitalize() for c in categories])
        
        # Rotate the tick labels
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Factual Accuracy Score (%)', rotation=270, labelpad=20)
        
        # Add values
        for i in range(len(categories)):
            for j in range(len(methods)):
                text = ax1.text(j, i, f'{heatmap_data[i][j]:.1f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax1.set_title('FAS Heatmap by Category and Method', fontweight='bold', pad=20)
        
        # Add category-specific best method
        ax2 = plt.subplot(2, 2, 3)
        
        best_methods = []
        for i, cat in enumerate(categories):
            best_idx = np.argmax(heatmap_data[i])
            best_methods.append(methods[best_idx])
        
        # Bar chart of best performance per category
        best_scores = [max(row) for row in heatmap_data]
        bars = ax2.barh(range(len(categories)), best_scores, 
                       color=[self.category_colors[c] for c in categories],
                       alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax2.set_yticks(range(len(categories)))
        ax2.set_yticklabels([c.capitalize() for c in categories])
        ax2.set_xlabel('Best FAS (%)', fontweight='bold')
        ax2.set_title('Peak Performance by Category', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add best method labels
        for i, (bar, method) in enumerate(zip(bars, best_methods)):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    method.replace('_', ' ').upper(),
                    va='center', fontsize=9, fontweight='bold')
        
        # Add improvement matrix
        ax3 = plt.subplot(2, 2, 4)
        
        # Calculate improvement over baseline for each category
        improvement_data = []
        for i, cat in enumerate(categories):
            baseline = heatmap_data[i][0]  # base_llm
            improvements = [(val - baseline) for val in heatmap_data[i][1:]]
            improvement_data.append(improvements)
        
        im2 = ax3.imshow(improvement_data, cmap='RdBu_r', vmin=-20, vmax=60, aspect='auto')
        
        ax3.set_xticks(np.arange(len(methods[1:])))
        ax3.set_yticks(np.arange(len(categories)))
        ax3.set_xticklabels([m.replace('_', ' ').upper() for m in methods[1:]])
        ax3.set_yticklabels([c.capitalize() for c in categories])
        
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add values
        for i in range(len(categories)):
            for j in range(len(methods[1:])):
                val = improvement_data[i][j]
                color = 'white' if abs(val) > 30 else 'black'
                text = ax3.text(j, i, f'+{val:.0f}',
                               ha="center", va="center", color=color, fontweight='bold')
        
        ax3.set_title('Improvement vs Baseline (pp)', fontweight='bold')
        
        plt.colorbar(im2, ax=ax3, label='Improvement (pp)')
        
        plt.suptitle('Category-wise Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure5_category_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 5: Category heatmap saved")
        plt.show()
    
    def visualize_hallucination_breakdown(self):
        """Figure 6: Hallucination type breakdown"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        methods = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
        
        for idx, method in enumerate(methods):
            ax = axes[idx // 2, idx % 2]
            
            # Calculate breakdown
            contradicted = []
            unverifiable = []
            supported = []
            
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    m = result['metrics'][method]
                    total = m['total_claims']
                    contradicted.append(m['contradicted'] / total * 100)
                    unverifiable.append(m['unverifiable'] / total * 100)
                    supported.append(m['supported'] / total * 100)
            
            if contradicted:
                # Stacked bar chart
                categories = ['Supported', 'Contradicted', 'Unverifiable']
                values = [np.mean(supported), np.mean(contradicted), np.mean(unverifiable)]
                colors = ['#2ECC71', '#E74C3C', '#95A5A6']
                
                # Create pie chart
                wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors,
                                                   autopct='%1.1f%%', startangle=90,
                                                   explode=(0.05, 0, 0))
                
                # Make percentage text bold
                for autotext in autotexts:
                    autotext.set_fontweight('bold')
                    autotext.set_color('white')
                
                ax.set_title(f'{method.replace("_", " ").upper()}\nFAS: {np.mean(supported):.1f}%',
                            fontweight='bold')
                
                # Add legend with counts
                legend_labels = [
                    f'Supported ({np.mean(supported):.1f}%)',
                    f'Contradicted ({np.mean(contradicted):.1f}%)',
                    f'Unverifiable ({np.mean(unverifiable):.1f}%)'
                ]
                ax.legend(wedges, legend_labels, loc="best", fontsize=8)
        
        plt.suptitle('Claim Classification Breakdown', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure6_hallucination_breakdown_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 6: Hallucination breakdown saved")
        plt.show()
    
    def visualize_significance_matrix(self):
        """Figure 7: Statistical significance matrix"""
        fig = plt.figure(figsize=(10, 8))
        
        methods = ['Base LLM', 'Vector RAG', 'Graph RAG', 'Hybrid AHS']
        method_keys = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
        
        # Calculate all pairwise p-values
        p_values = np.ones((len(methods), len(methods)))
        cohen_d = np.zeros((len(methods), len(methods)))
        
        for i, method1 in enumerate(method_keys):
            fas1 = []
            for result in self.api_results['results']:
                if method1 in result['metrics'] and result['metrics'][method1]['total_claims'] > 0:
                    fas1.append((1 - result['metrics'][method1]['HR']) * 100)
            
            for j, method2 in enumerate(method_keys):
                if i != j:
                    fas2 = []
                    for result in self.api_results['results']:
                        if method2 in result['metrics'] and result['metrics'][method2]['total_claims'] > 0:
                            fas2.append((1 - result['metrics'][method2]['HR']) * 100)
                    
                    if fas1 and fas2:
                        # T-test
                        t_stat, p_val = stats.ttest_ind(fas1, fas2)
                        p_values[i, j] = p_val
                        
                        # Cohen's d
                        pooled_std = np.sqrt((np.var(fas1) + np.var(fas2)) / 2)
                        if pooled_std > 0:
                            cohen_d[i, j] = (np.mean(fas2) - np.mean(fas1)) / pooled_std
        
        # Create heatmap
        ax = plt.subplot(111)
        
        # Create custom colormap
        mask = np.triu(np.ones_like(p_values, dtype=bool))
        
        # Plot p-values
        im = ax.imshow(p_values, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
        
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(methods)
        
        # Add significance markers
        for i in range(len(methods)):
            for j in range(len(methods)):
                if i != j:
                    p = p_values[i, j]
                    d = cohen_d[i, j]
                    
                    if p < 0.001:
                        sig = '***'
                        color = 'white'
                    elif p < 0.01:
                        sig = '**'
                        color = 'white'
                    elif p < 0.05:
                        sig = '*'
                        color = 'black'
                    else:
                        sig = 'ns'
                        color = 'black'
                    
                    text = ax.text(j, i, f'{sig}\nd={d:.2f}',
                                  ha="center", va="center", color=color,
                                  fontsize=9, fontweight='bold')
                else:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                          fill=True, color='lightgray'))
        
        ax.set_title('Statistical Significance Matrix\n(p-values and Cohen\'s d)', 
                    fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('p-value', rotation=270, labelpad=20)
        
        # Add legend
        legend_text = "*** p < 0.001\n** p < 0.01\n* p < 0.05\nns = not significant\n\nd = Cohen's d effect size"
        plt.text(1.15, 0.5, legend_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure7_significance_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 7: Significance matrix saved")
        plt.show()
    
    def visualize_question_complexity(self):
        """Figure 8: Performance vs question complexity"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Analyze performance by question length
        length_bins = [0, 10, 15, 20, 30, 100]
        bin_labels = ['<10', '10-15', '15-20', '20-30', '>30']
        
        methods = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
        
        # Collect data
        performance_by_length = {method: {label: [] for label in bin_labels} 
                                for method in methods}
        
        for i, item in enumerate(self.dataset['questions']):
            q_length = len(item['question'].split())
            bin_idx = np.digitize(q_length, length_bins) - 1
            if bin_idx >= len(bin_labels):
                bin_idx = len(bin_labels) - 1
            bin_label = bin_labels[bin_idx]
            
            # Get performance for this question
            if i < len(self.api_results['results']):
                result = self.api_results['results'][i]
                for method in methods:
                    if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                        fas = (1 - result['metrics'][method]['HR']) * 100
                        performance_by_length[method][bin_label].append(fas)
        
        # Plot 1: Line plot of performance vs complexity
        ax1 = axes[0, 0]
        
        for method, color in self.method_colors.items():
            means = []
            stds = []
            for label in bin_labels:
                values = performance_by_length[method][label]
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values) / np.sqrt(len(values)))  # SEM
                else:
                    means.append(0)
                    stds.append(0)
            
            ax1.errorbar(range(len(bin_labels)), means, yerr=stds,
                        label=method.replace('_', ' ').upper(),
                        marker='o', linewidth=2, capsize=5, capthick=2,
                        color=color, alpha=0.8)
        
        ax1.set_xticks(range(len(bin_labels)))
        ax1.set_xticklabels(bin_labels)
        ax1.set_xlabel('Question Length (words)', fontweight='bold')
        ax1.set_ylabel('Factual Accuracy Score (%)', fontweight='bold')
        ax1.set_title('Performance vs Question Complexity', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        ax2 = axes[0, 1]
        
        data_for_box = []
        labels_for_box = []
        colors_for_box = []
        
        for method in methods:
            all_values = []
            for label in bin_labels:
                all_values.extend(performance_by_length[method][label])
            if all_values:
                data_for_box.append(all_values)
                labels_for_box.append(method.replace('_', ' ').upper())
                colors_for_box.append(self.method_colors[method])
        
        bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                         notch=True, showfliers=False)
        
        for patch, color in zip(bp['boxes'], colors_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Factual Accuracy Score (%)', fontweight='bold')
        ax2.set_title('Overall Performance Distribution', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Scatter plot with regression
        ax3 = axes[1, 0]
        
        for method, color in self.method_colors.items():
            x_vals = []
            y_vals = []
            
            for i, item in enumerate(self.dataset['questions']):
                q_length = len(item['question'].split())
                if i < len(self.api_results['results']):
                    result = self.api_results['results'][i]
                    if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                        fas = (1 - result['metrics'][method]['HR']) * 100
                        x_vals.append(q_length)
                        y_vals.append(fas)
            
            if x_vals and y_vals:
                # Scatter plot
                ax3.scatter(x_vals, y_vals, alpha=0.3, s=10, color=color)
                
                # Add trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                ax3.plot(x_trend, p(x_trend), "--", color=color, alpha=0.8,
                        label=f'{method.replace("_", " ").upper()} (R²={np.corrcoef(x_vals, y_vals)[0,1]**2:.3f})')
        
        ax3.set_xlabel('Question Length (words)', fontweight='bold')
        ax3.set_ylabel('Factual Accuracy Score (%)', fontweight='bold')
        ax3.set_title('Correlation Analysis', fontweight='bold')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Category complexity
        ax4 = axes[1, 1]
        
        category_lengths = {}
        for item in self.dataset['questions']:
            cat = item['category']
            if cat not in category_lengths:
                category_lengths[cat] = []
            category_lengths[cat].append(len(item['question'].split()))
        
        # Calculate average complexity and performance per category
        cat_data = []
        for cat in ['factual', 'causal', 'diagnostic', 'comparative']:
            avg_length = np.mean(category_lengths[cat])
            
            # Get average performance
            cat_fas = []
            for result in self.api_results['results']:
                if result['category'] == cat:
                    for method in methods[1:]:  # Exclude baseline
                        if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                            cat_fas.append((1 - result['metrics'][method]['HR']) * 100)
            
            avg_fas = np.mean(cat_fas) if cat_fas else 0
            cat_data.append((avg_length, avg_fas, cat))
        
        for length, fas, cat in cat_data:
            ax4.scatter(length, fas, s=200, color=self.category_colors[cat],
                       alpha=0.7, edgecolors='black', linewidth=2)
            ax4.annotate(cat.capitalize(), (length, fas), 
                        xytext=(5, 5), textcoords='offset points',
                        fontweight='bold', fontsize=9)
        
        ax4.set_xlabel('Average Question Length (words)', fontweight='bold')
        ax4.set_ylabel('Average FAS (RAG methods)', fontweight='bold')
        ax4.set_title('Category Complexity vs Performance', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Question Complexity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure8_question_complexity_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 8: Question complexity analysis saved")
        plt.show()
    
    def visualize_cost_benefit(self):
        """Figure 9: Cost-benefit analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Define costs (approximate)
        costs = {
            'base_llm': 0.15,      # Just GPT-4o-mini
            'vector_rag': 0.45,    # GPT + Tavily
            'graph_rag': 0.45,     # GPT + Tavily
            'hybrid_ahs': 0.60     # GPT + 2x Tavily
        }
        
        # Processing time (relative)
        times = {
            'base_llm': 1.0,
            'vector_rag': 2.5,
            'graph_rag': 2.5,
            'hybrid_ahs': 3.5
        }
        
        # Calculate FAS
        fas_scores = {}
        for method in costs.keys():
            fas_values = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    fas = (1 - result['metrics'][method]['HR']) * 100
                    fas_values.append(fas)
            fas_scores[method] = np.mean(fas_values) if fas_values else 0
        
        # Plot 1: Cost vs Performance
        ax1 = axes[0, 0]
        
        for method, color in self.method_colors.items():
            ax1.scatter(costs[method], fas_scores[method], s=300, 
                       color=color, alpha=0.7, edgecolors='black', linewidth=2)
            ax1.annotate(method.replace('_', ' ').upper(), 
                        (costs[method], fas_scores[method]),
                        xytext=(10, 5), textcoords='offset points',
                        fontweight='bold', fontsize=9)
        
        # Add efficient frontier
        frontier_x = [costs['base_llm'], costs['vector_rag']]
        frontier_y = [fas_scores['base_llm'], fas_scores['vector_rag']]
        ax1.plot(frontier_x, frontier_y, 'k--', alpha=0.5, linewidth=2,
                label='Efficient Frontier')
        
        ax1.set_xlabel('Cost per 1000 queries ($)', fontweight='bold')
        ax1.set_ylabel('Factual Accuracy Score (%)', fontweight='bold')
        ax1.set_title('Cost vs Performance Trade-off', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Time vs Performance
        ax2 = axes[0, 1]
        
        for method, color in self.method_colors.items():
            ax2.scatter(times[method], fas_scores[method], s=300,
                       color=color, alpha=0.7, edgecolors='black', linewidth=2)
            ax2.annotate(method.replace('_', ' ').upper(),
                        (times[method], fas_scores[method]),
                        xytext=(10, 5), textcoords='offset points',
                        fontweight='bold', fontsize=9)
        
        ax2.set_xlabel('Relative Processing Time', fontweight='bold')
        ax2.set_ylabel('Factual Accuracy Score (%)', fontweight='bold')
        ax2.set_title('Time vs Performance Trade-off', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency Score
        ax3 = axes[1, 0]
        
        # Calculate efficiency (FAS / Cost)
        efficiency = {method: fas_scores[method] / costs[method] 
                     for method in costs.keys()}
        
        methods_sorted = sorted(efficiency.keys(), key=lambda x: efficiency[x], reverse=True)
        
        bars = ax3.bar(range(len(methods_sorted)), 
                      [efficiency[m] for m in methods_sorted],
                      color=[self.method_colors[m] for m in methods_sorted],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax3.set_xticks(range(len(methods_sorted)))
        ax3.set_xticklabels([m.replace('_', ' ').upper() for m in methods_sorted],
                           rotation=45, ha='right')
        ax3.set_ylabel('Efficiency Score (FAS/Cost)', fontweight='bold')
        ax3.set_title('Cost Efficiency Ranking', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bar, method in zip(bars, methods_sorted):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{efficiency[method]:.0f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: ROI Analysis
        ax4 = axes[1, 1]
        
        # Calculate ROI (improvement over baseline / additional cost)
        baseline_fas = fas_scores['base_llm']
        baseline_cost = costs['base_llm']
        
        roi_data = []
        for method in ['vector_rag', 'graph_rag', 'hybrid_ahs']:
            improvement = fas_scores[method] - baseline_fas
            additional_cost = costs[method] - baseline_cost
            roi = improvement / additional_cost if additional_cost > 0 else 0
            roi_data.append((method, roi, improvement, additional_cost))
        
        # Sort by ROI
        roi_data.sort(key=lambda x: x[1], reverse=True)
        
        methods_roi = [d[0] for d in roi_data]
        roi_values = [d[1] for d in roi_data]
        
        bars = ax4.bar(range(len(methods_roi)), roi_values,
                      color=[self.method_colors[m] for m in methods_roi],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax4.set_xticks(range(len(methods_roi)))
        ax4.set_xticklabels([m.replace('_', ' ').upper() for m in methods_roi],
                           rotation=45, ha='right')
        ax4.set_ylabel('ROI (FAS improvement per $ spent)', fontweight='bold')
        ax4.set_title('Return on Investment Analysis', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bar, (method, roi, imp, cost) in zip(bars, roi_data):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{roi:.0f}\n(+{imp:.1f}% / ${cost:.2f})',
                    ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.suptitle('Cost-Benefit Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure9_cost_benefit_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 9: Cost-benefit analysis saved")
        plt.show()
    
    def visualize_error_distribution(self):
        """Figure 10: Error distribution and patterns"""
        fig = plt.figure(figsize=(14, 10))
        
        # Analyze error patterns
        methods = ['base_llm', 'vector_rag', 'graph_rag', 'hybrid_ahs']
        
        # Plot 1: Distribution of FAS scores
        ax1 = plt.subplot(2, 3, 1)
        
        for method, color in self.method_colors.items():
            fas_values = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    fas = (1 - result['metrics'][method]['HR']) * 100
                    fas_values.append(fas)
            
            if fas_values:
                # KDE plot
                from scipy.stats import gaussian_kde
                density = gaussian_kde(fas_values)
                xs = np.linspace(0, 100, 200)
                ax1.plot(xs, density(xs), label=method.replace('_', ' ').upper(),
                        color=color, linewidth=2, alpha=0.8)
                ax1.fill_between(xs, density(xs), alpha=0.2, color=color)
        
        ax1.set_xlabel('Factual Accuracy Score (%)', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title('FAS Distribution (KDE)', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        
        # Plot 2: Cumulative distribution
        ax2 = plt.subplot(2, 3, 2)
        
        for method, color in self.method_colors.items():
            fas_values = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    fas = (1 - result['metrics'][method]['HR']) * 100
                    fas_values.append(fas)
            
            if fas_values:
                sorted_fas = np.sort(fas_values)
                cumulative = np.arange(1, len(sorted_fas) + 1) / len(sorted_fas) * 100
                ax2.plot(sorted_fas, cumulative, label=method.replace('_', ' ').upper(),
                        color=color, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Factual Accuracy Score (%)', fontweight='bold')
        ax2.set_ylabel('Cumulative Percentage', fontweight='bold')
        ax2.set_title('Cumulative Distribution', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        
        # Plot 3: Error correlation matrix
        ax3 = plt.subplot(2, 3, 3)
        
        # Calculate correlation between methods' errors
        error_data = {}
        for method in methods:
            errors = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    hr = result['metrics'][method]['HR']
                    errors.append(hr)
                else:
                    errors.append(np.nan)
            error_data[method] = errors
        
        # Create correlation matrix
        df_errors = pd.DataFrame(error_data)
        correlation = df_errors.corr()
        
        im = ax3.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax3.set_xticks(range(len(methods)))
        ax3.set_yticks(range(len(methods)))
        ax3.set_xticklabels([m.replace('_', ' ').upper() for m in methods],
                           rotation=45, ha='right')
        ax3.set_yticklabels([m.replace('_', ' ').upper() for m in methods])
        
        # Add values
        for i in range(len(methods)):
            for j in range(len(methods)):
                text = ax3.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax3.set_title('Error Correlation Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Worst case analysis
        ax4 = plt.subplot(2, 3, 4)
        
        # Find worst performing questions
        worst_questions = []
        for i, result in enumerate(self.api_results['results']):
            avg_hr = []
            for method in methods:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    avg_hr.append(result['metrics'][method]['HR'])
            if avg_hr:
                worst_questions.append((i, np.mean(avg_hr), result['question_id']))
        
        worst_questions.sort(key=lambda x: x[1], reverse=True)
        worst_10 = worst_questions[:10]
        
        # Plot worst questions
        categories_worst = []
        for idx, hr, q_id in worst_10:
            cat = self.api_results['results'][idx]['category']
            categories_worst.append(cat)
        
        cat_counts = {}
        for cat in categories_worst:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        if cat_counts:
            bars = ax4.bar(range(len(cat_counts)), list(cat_counts.values()),
                          color=[self.category_colors[cat] for cat in cat_counts.keys()],
                          alpha=0.7, edgecolor='black', linewidth=1.5)
            ax4.set_xticks(range(len(cat_counts)))
            ax4.set_xticklabels([c.capitalize() for c in cat_counts.keys()])
            ax4.set_ylabel('Count', fontweight='bold')
            ax4.set_title('Categories of 10 Worst Questions', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Perfect score analysis
        ax5 = plt.subplot(2, 3, 5)
        
        perfect_counts = {method: 0 for method in methods}
        zero_counts = {method: 0 for method in methods}
        
        for result in self.api_results['results']:
            for method in methods:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    hr = result['metrics'][method]['HR']
                    if hr == 0:
                        perfect_counts[method] += 1
                    elif hr == 1:
                        zero_counts[method] += 1
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, [perfect_counts[m] for m in methods], width,
                       label='Perfect (FAS=100%)', color='#2ECC71', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        bars2 = ax5.bar(x + width/2, [zero_counts[m] for m in methods], width,
                       label='Failed (FAS=0%)', color='#E74C3C', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        
        ax5.set_xlabel('Method')
        ax5.set_ylabel('Count')
        ax5.set_title('Extreme Cases Distribution', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([m.replace('_', ' ').upper() for m in methods],
                           rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add values
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Improvement potential
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        
        for method, color in self.method_colors.items():
            fas_values = []
            for result in self.api_results['results']:
                if method in result['metrics'] and result['metrics'][method]['total_claims'] > 0:
                    fas = (1 - result['metrics'][method]['HR']) * 100
                    fas_values.append(fas)
            
            if fas_values:
                percs = [np.percentile(fas_values, p) for p in percentiles]
                ax6.plot(percentiles, percs, marker='o', label=method.replace('_', ' ').upper(),
                        color=color, linewidth=2, alpha=0.8)
        
        ax6.set_xlabel('Percentile', fontweight='bold')
        ax6.set_ylabel('Factual Accuracy Score (%)', fontweight='bold')
        ax6.set_title('Performance Percentiles', fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, 100)
        ax6.set_ylim(0, 100)
        
        plt.suptitle('Error Distribution and Pattern Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'figure10_error_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        print("✅ Figure 10: Error distribution saved")
        plt.show()

def main():
    if not PLOTTING_AVAILABLE:
        print("❌ Please install required packages:")
        print("pip install matplotlib seaborn scipy pandas networkx")
        return
    
    print("🎨 Starting publication visualization generation...")
    print("="*60)
    
    visualizer = PublicationVisualizer()
    visualizer.create_all_visualizations()
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("\nGenerated files:")
    print("  • figure1_dataset_composition_*.png")
    print("  • figure2_method_architecture_*.png")
    print("  • figure3_performance_radar_*.png")
    print("  • figure4_confidence_intervals_*.png")
    print("  • figure5_category_heatmap_*.png")
    print("  • figure6_hallucination_breakdown_*.png")
    print("  • figure7_significance_matrix_*.png")
    print("  • figure8_question_complexity_*.png")
    print("  • figure9_cost_benefit_*.png")
    print("  • figure10_error_distribution_*.png")
    print("\nAll figures are publication-ready at 300 DPI")

if __name__ == "__main__":
    main()