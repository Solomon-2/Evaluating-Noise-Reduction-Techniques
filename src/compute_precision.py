import csv
import argparse
from collections import defaultdict
from normalize_filenames import normalize_filename

def interval_overlap(a_start, a_end, b_start, b_end):
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    overlap = max(0, earliest_end - latest_start)
    return overlap

def load_events(csv_path):
    events = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use normalized filename for matching
            full = row['filename']
            fname = normalize_filename(full)
            start = float(row['start_sec'])
            end = float(row['end_sec'])
            events[fname].append((start, end))
    return events

def match_events_for_precision(gt_events, pred_events, overlap_thresh=0.5):
    """Returns TP, FP for a single file's events (from prediction perspective)."""
    matched_gt = set()
    TP = 0
    for i, (pr_start, pr_end) in enumerate(pred_events):
        pr_duration = pr_end - pr_start
        found_match = False
        for j, (gt_start, gt_end) in enumerate(gt_events):
            overlap = interval_overlap(pr_start, pr_end, gt_start, gt_end)
            if pr_duration > 0 and (overlap / pr_duration) >= overlap_thresh and j not in matched_gt:
                TP += 1
                matched_gt.add(j)
                found_match = True
                break
        # If no match found, it's a FP
    FP = len(pred_events) - TP
    return TP, FP

def main():
    parser = argparse.ArgumentParser(description="Compute precision for detected events against ground truth.")
    parser.add_argument('--ground_csv', required=True, help='CSV file with ground truth events')
    parser.add_argument('--detected_csv', required=True, help='CSV file with detected events')
    parser.add_argument('--overlap_thresh', type=float, default=0.5, help='Fractional overlap required to count as match (default: 0.5)')
    args = parser.parse_args()

    gt = load_events(args.ground_csv)
    pred = load_events(args.detected_csv)

    total_TP = 0
    total_FP = 0
    for fname, pred_events in pred.items():
        gt_events = gt.get(fname, [])
        TP, FP = match_events_for_precision(gt_events, pred_events, args.overlap_thresh)
        total_TP += TP
        total_FP += FP
        print(f"{fname}: TP={TP}, FP={FP}, Precision={TP/(TP+FP) if (TP+FP)>0 else 'NA'}")

    overall_prec = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    print(f"\nOverall Precision: {overall_prec:.3f} (TP={total_TP}, FP={total_FP})")

if __name__ == "__main__":
    main()
