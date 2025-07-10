import csv
import argparse
from collections import defaultdict

def interval_overlap(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = max(0, earliest_end - latest_start)
    return overlap

def load_events(csv_path):
    import re
    events = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract .wav filename from the path
            full = row['filename']
            match = re.search(r'[^/\\]*\\.wav$', full)
            if match:
                fname = match.group(0)
            else:
                fname = full  # fallback
            start = float(row['start_sec'])
            end = float(row['end_sec'])
            events[fname].append((start, end))
    return events

def match_events(gt_events, pred_events, overlap_thresh=0.5):
    """Returns TP, FP, FN for a single file's events."""
    matched_pred = set()
    matched_gt = set()
    TP = 0
    for i, (gt_start, gt_end) in enumerate(gt_events):
        gt_duration = gt_end - gt_start
        for j, (pr_start, pr_end) in enumerate(pred_events):
            pr_duration = pr_end - pr_start
            overlap = interval_overlap(gt_start, gt_end, pr_start, pr_end)
            if gt_duration > 0 and pr_duration > 0 and (overlap / gt_duration) >= overlap_thresh and (overlap / pr_duration) >= overlap_thresh and j not in matched_pred and i not in matched_gt:
                TP += 1
                matched_pred.add(j)
                matched_gt.add(i)
                break
    FN = len(gt_events) - TP
    FP = len(pred_events) - TP
    return TP, FP, FN

def main():
    parser = argparse.ArgumentParser(description="Compute F1 score for detected events against ground truth.")
    parser.add_argument('--ground_csv', required=True, help='CSV file with ground truth events')
    parser.add_argument('--detected_csv', required=True, help='CSV file with detected events')
    parser.add_argument('--overlap_thresh', type=float, default=0.5, help='Fractional overlap required to count as match (default: 0.5)')
    args = parser.parse_args()

    gt = load_events(args.ground_csv)
    pred = load_events(args.detected_csv)

    total_TP = 0
    total_FP = 0
    total_FN = 0
    print("filename,TP,FP,FN,Precision,Recall,F1")
    for fname in sorted(set(gt.keys()) | set(pred.keys())):
        gt_events = gt.get(fname, [])
        pred_events = pred.get(fname, [])
        TP, FP, FN = match_events(gt_events, pred_events, args.overlap_thresh)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{fname},{TP},{FP},{FN},{precision:.3f},{recall:.3f},{f1:.3f}")

    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    print(f"\nOverall: TP={total_TP}, FP={total_FP}, FN={total_FN}")
    print(f"Overall Precision: {overall_precision:.3f}")
    print(f"Overall Recall: {overall_recall:.3f}")
    print(f"Overall F1 Score: {overall_f1:.3f}")

if __name__ == "__main__":
    main()
