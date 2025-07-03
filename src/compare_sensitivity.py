import csv
import argparse
from collections import defaultdict

# Helper: compute overlap between two intervals
def interval_overlap(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = max(0, earliest_end - latest_start)
    return overlap

def load_events(csv_path):
    """Returns dict: {filename: [(start, end), ...]}"""
    events = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']
            start = float(row['start_sec'])
            end = float(row['end_sec'])
            events[fname].append((start, end))
    return events

def match_events(gt_events, pred_events, overlap_thresh=0.5):
    """Returns TP, FN for a single file's events."""
    matched_pred = set()
    TP = 0
    for gt_start, gt_end in gt_events:
        gt_duration = gt_end - gt_start
        found_match = False
        for i, (pr_start, pr_end) in enumerate(pred_events):
            overlap = interval_overlap(gt_start, gt_end, pr_start, pr_end)
            if gt_duration > 0 and (overlap / gt_duration) >= overlap_thresh and i not in matched_pred:
                TP += 1
                matched_pred.add(i)
                found_match = True
                break
        # If no match found, it's a FN
    FN = len(gt_events) - TP
    return TP, FN

def main():
    parser = argparse.ArgumentParser(description="Compare ground truth and detected apnea events to compute sensitivity.")
    parser.add_argument('--ground_csv', required=True, help='CSV file with ground truth events')
    parser.add_argument('--detected_csv', required=True, help='CSV file with detected events')
    parser.add_argument('--overlap_thresh', type=float, default=0.5, help='Fractional overlap required to count as match (default: 0.5)')
    args = parser.parse_args()

    gt = load_events(args.ground_csv)
    pred = load_events(args.detected_csv)

    total_TP = 0
    total_FN = 0
    for fname, gt_events in gt.items():
        pred_events = pred.get(fname, [])
        TP, FN = match_events(gt_events, pred_events, args.overlap_thresh)
        total_TP += TP
        total_FN += FN
        print(f"{fname}: TP={TP}, FN={FN}, Sensitivity={TP/(TP+FN) if (TP+FN)>0 else 'NA'}")

    overall_sens = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    print(f"\nOverall Sensitivity: {overall_sens:.3f} (TP={total_TP}, FN={total_FN})")

if __name__ == "__main__":
    main()
