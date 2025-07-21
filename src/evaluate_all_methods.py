#!/usr/bin/env python3
"""
Unified evaluation script for comparing all noise reduction methods.
Automatically discovers denoising method directories and computes all metrics.
"""

import os
import csv
import argparse
import pandas as pd
from collections import defaultdict
from normalize_filenames import normalize_filename


def interval_overlap(a_start, a_end, b_start, b_end):
    """Compute overlap between two time intervals."""
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = max(0, earliest_end - latest_start)
    return overlap


def load_events(csv_path):
    """Load events from CSV with normalized filenames."""
    events = defaultdict(list)
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return events
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            full = row['filename']
            fname = normalize_filename(full)
            start = float(row['start_sec'])
            end = float(row['end_sec'])
            events[fname].append((start, end))
    return events


def compute_metrics(gt_events, pred_events, overlap_thresh=0.5):
    """Compute TP, FP, FN for all files, return overall metrics."""
    total_TP = 0
    total_FP = 0
    total_FN = 0
    
    # Get all unique filenames from both ground truth and predictions
    all_files = set(gt_events.keys()) | set(pred_events.keys())
    
    for fname in all_files:
        gt_file_events = gt_events.get(fname, [])
        pred_file_events = pred_events.get(fname, [])
        
        # Match events for this file
        matched_pred = set()
        matched_gt = set()
        file_TP = 0
        
        # Find true positives
        for i, (gt_start, gt_end) in enumerate(gt_file_events):
            gt_duration = gt_end - gt_start
            for j, (pr_start, pr_end) in enumerate(pred_file_events):
                pr_duration = pr_end - pr_start
                overlap = interval_overlap(gt_start, gt_end, pr_start, pr_end)
                
                # Check if overlap meets threshold for both GT and prediction
                gt_overlap_ok = gt_duration > 0 and (overlap / gt_duration) >= overlap_thresh
                pr_overlap_ok = pr_duration > 0 and (overlap / pr_duration) >= overlap_thresh
                
                if (gt_overlap_ok and pr_overlap_ok and 
                    j not in matched_pred and i not in matched_gt):
                    file_TP += 1
                    matched_pred.add(j)
                    matched_gt.add(i)
                    break
        
        file_FN = len(gt_file_events) - file_TP
        file_FP = len(pred_file_events) - file_TP
        
        total_TP += file_TP
        total_FP += file_FP
        total_FN += file_FN
    
    return total_TP, total_FP, total_FN


def evaluate_single_method(ground_csv, detected_csv, method_name, overlap_thresh=0.5):
    """Evaluate a single denoising method and return metrics."""
    print(f"Evaluating {method_name}...")
    
    # Load events
    gt_events = load_events(ground_csv)
    pred_events = load_events(detected_csv)
    
    if not gt_events:
        print(f"  Warning: No ground truth events found")
        return None
    
    if not pred_events:
        print(f"  Warning: No predicted events found for {method_name}")
        # Return zeros if no predictions
        total_gt = sum(len(events) for events in gt_events.values())
        return {
            'Method': method_name,
            'TP': 0,
            'FP': 0,
            'FN': total_gt,
            'Sensitivity': 0.0,
            'Precision': 0.0,
            'F1_Score': 0.0
        }
    
    # Compute metrics
    TP, FP, FN = compute_metrics(gt_events, pred_events, overlap_thresh)
    
    # Calculate derived metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    print(f"  TP={TP}, FP={FP}, FN={FN}")
    print(f"  Sensitivity={sensitivity:.3f}, Precision={precision:.3f}, F1={f1_score:.3f}")
    
    return {
        'Method': method_name,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Precision': precision,
        'F1_Score': f1_score
    }


def discover_methods(denoised_base_dir):
    """Automatically discover denoising method directories."""
    methods = []
    if not os.path.exists(denoised_base_dir):
        print(f"Denoised directory not found: {denoised_base_dir}")
        return methods
    
    for item in os.listdir(denoised_base_dir):
        method_dir = os.path.join(denoised_base_dir, item)
        if os.path.isdir(method_dir):
            output_csv = os.path.join(method_dir, 'output.csv')
            if os.path.exists(output_csv):
                methods.append((item, output_csv))
            else:
                print(f"Warning: No output.csv found for method '{item}'")
    
    return methods


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all noise reduction methods and generate consolidated results"
    )
    parser.add_argument('--ground_csv', required=True, 
                       help='Path to ground truth CSV file')
    parser.add_argument('--denoised_dir', required=True,
                       help='Base directory containing denoised method subdirectories')
    parser.add_argument('--output_csv', default='evaluation_results.csv',
                       help='Output CSV file for consolidated results')
    parser.add_argument('--overlap_thresh', type=float, default=0.5,
                       help='Overlap threshold for matching events (default: 0.5)')
    
    args = parser.parse_args()
    
    print("=== Unified Noise Reduction Method Evaluation ===")
    print(f"Ground truth: {args.ground_csv}")
    print(f"Denoised methods directory: {args.denoised_dir}")
    print(f"Output file: {args.output_csv}")
    print(f"Overlap threshold: {args.overlap_thresh}")
    print()
    
    # Discover methods
    methods = discover_methods(args.denoised_dir)
    if not methods:
        print("No valid denoising methods found!")
        return
    
    print(f"Found {len(methods)} methods: {[m[0] for m in methods]}")
    print()
    
    # Evaluate each method
    results = []
    for method_name, detected_csv in methods:
        result = evaluate_single_method(
            args.ground_csv, detected_csv, method_name, args.overlap_thresh
        )
        if result:
            results.append(result)
        print()
    
    if not results:
        print("No results to save!")
        return
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    # Sort by F1 score (descending)
    df = df.sort_values('F1_Score', ascending=False)
    
    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to: {args.output_csv}")
    
    # Display summary table
    print("\n=== EVALUATION SUMMARY ===")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Find best method
    if len(df) > 0:
        best_method = df.iloc[0]
        print(f"\nBest performing method: {best_method['Method']} (F1 Score: {best_method['F1_Score']:.3f})")


if __name__ == "__main__":
    main()
