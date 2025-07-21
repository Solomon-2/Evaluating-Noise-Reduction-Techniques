"""
Evaluation functions for use in Jupyter notebooks.
Provides easy-to-use Python functions for evaluating noise reduction methods.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_all_methods import discover_methods, evaluate_single_method, load_events


def run_consolidated_evaluation(ground_csv, denoised_dir, output_csv=None, overlap_thresh=0.5):
    """
    Run evaluation on all methods and return a pandas DataFrame.
    
    Parameters:
    -----------
    ground_csv : str
        Path to ground truth CSV file
    denoised_dir : str
        Directory containing denoised method subdirectories
    output_csv : str, optional
        Path to save results CSV (if None, results are not saved)
    overlap_thresh : float
        Overlap threshold for matching events (default: 0.5)
    
    Returns:
    --------
    pd.DataFrame : Evaluation results
    """
    print("Running consolidated evaluation...")
    
    # Discover methods
    methods = discover_methods(denoised_dir)
    if not methods:
        print("No valid denoising methods found!")
        return pd.DataFrame()
    
    print(f"Found {len(methods)} methods: {[m[0] for m in methods]}")
    
    # Evaluate each method
    results = []
    for method_name, detected_csv in methods:
        result = evaluate_single_method(ground_csv, detected_csv, method_name, overlap_thresh)
        if result:
            results.append(result)
    
    if not results:
        print("No results generated!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('F1_Score', ascending=False)
    
    # Save if requested
    if output_csv:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
    
    return df


def plot_comparison_metrics(df, title="Noise Reduction Method Comparison"):
    """
    Create comprehensive visualization of evaluation metrics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Evaluation results dataframe
    title : str
        Plot title
    """
    if df.empty:
        print("No data to plot!")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Bar plot of all metrics
    ax1 = axes[0, 0]
    metrics = ['Sensitivity', 'Precision', 'F1_Score']
    x = range(len(df))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        ax1.bar([xi + i*width for xi in x], df[metric], width, 
                label=metric, alpha=0.8)
    
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Score')
    ax1.set_title('All Metrics Comparison')
    ax1.set_xticks([xi + width for xi in x])
    ax1.set_xticklabels(df['Method'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # 2. F1 Score ranking
    ax2 = axes[0, 1]
    bars = ax2.bar(df['Method'], df['F1_Score'], 
                   color=plt.cm.viridis(df['F1_Score']), alpha=0.8)
    ax2.set_xlabel('Methods')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Ranking')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, score in zip(bars, df['F1_Score']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Confusion Matrix Components
    ax3 = axes[1, 0]
    confusion_metrics = ['TP', 'FP', 'FN']
    bottom = [0] * len(df)
    colors = ['green', 'red', 'orange']
    
    for metric, color in zip(confusion_metrics, colors):
        ax3.bar(df['Method'], df[metric], bottom=bottom, 
                label=metric, color=color, alpha=0.7)
        bottom = [b + v for b, v in zip(bottom, df[metric])]
    
    ax3.set_xlabel('Methods')
    ax3.set_ylabel('Count')
    ax3.set_title('True/False Positives and False Negatives')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision vs Recall scatter plot
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['Precision'], df['Sensitivity'], 
                         s=df['F1_Score']*500, c=df['F1_Score'], 
                         cmap='viridis', alpha=0.7, edgecolors='black')
    
    # Add method labels
    for i, method in enumerate(df['Method']):
        ax4.annotate(method, (df['Precision'].iloc[i], df['Sensitivity'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Precision')
    ax4.set_ylabel('Sensitivity (Recall)')
    ax4.set_title('Precision vs Sensitivity\n(Size = F1 Score)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
    cbar.set_label('F1 Score')
    
    plt.tight_layout()
    plt.show()


def print_evaluation_summary(df):
    """
    Print a formatted summary of evaluation results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Evaluation results dataframe
    """
    if df.empty:
        print("No evaluation results to display!")
        return
    
    print("=" * 60)
    print("NOISE REDUCTION METHOD EVALUATION SUMMARY")
    print("=" * 60)
    
    # Overall statistics
    print(f"Number of methods evaluated: {len(df)}")
    print(f"Best F1 Score: {df['F1_Score'].max():.3f}")
    print(f"Average F1 Score: {df['F1_Score'].mean():.3f}")
    print()
    
    # Detailed results table
    print("DETAILED RESULTS:")
    print("-" * 60)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(df.to_string(index=False))
    
    print()
    print("RANKING BY F1 SCORE:")
    print("-" * 30)
    for i, row in df.iterrows():
        print(f"{i+1}. {row['Method']}: {row['F1_Score']:.3f}")
    
    # Best method details
    best_method = df.iloc[0]
    print()
    print("BEST PERFORMING METHOD:")
    print("-" * 30)
    print(f"Method: {best_method['Method']}")
    print(f"F1 Score: {best_method['F1_Score']:.3f}")
    print(f"Sensitivity: {best_method['Sensitivity']:.3f}")
    print(f"Precision: {best_method['Precision']:.3f}")
    print(f"True Positives: {best_method['TP']}")
    print(f"False Positives: {best_method['FP']}")
    print(f"False Negatives: {best_method['FN']}")
    print("=" * 60)


def quick_evaluate(ground_csv="../tests/raw/output.csv", 
                  denoised_dir="../tests/denoised",
                  save_results=True):
    """
    Quick evaluation function with sensible defaults for the notebook.
    
    Parameters:
    -----------
    ground_csv : str
        Path to ground truth CSV
    denoised_dir : str  
        Path to denoised methods directory
    save_results : bool
        Whether to save results to CSV
    
    Returns:
    --------
    pd.DataFrame : Evaluation results
    """
    output_csv = "../results/evaluation_results.csv" if save_results else None
    
    df = run_consolidated_evaluation(ground_csv, denoised_dir, output_csv)
    
    if not df.empty:
        print_evaluation_summary(df)
        plot_comparison_metrics(df)
    
    return df
