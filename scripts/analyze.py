import os
import json
import glob
import argparse
import math


METRIC_SPECS = [
    ("sr", "success", "Success Rate (SR)", 100.0, "%"),
    ("spl", "spl", "SPL", 100.0, "%"),
    ("ndtw", "ndtw", "nDTW", 100.0, "%"),
    ("sdtw", "sdtw", "SDTW", 100.0, "%"),
    ("ne", "distance_to_goal", "Navigation Error (NE)", 1.0, "m"),
]


def _new_metric_stat():
    return {
        "values": [],
        "missing": 0,
        "nan": 0,
        "inf": 0,
        "invalid": 0,
    }


def _add_metric_value(stat, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        stat["invalid"] += 1
        return

    if math.isnan(value):
        stat["nan"] += 1
    elif math.isinf(value):
        stat["inf"] += 1
    else:
        stat["values"].append(value)


def _format_metric(stat, total, scale, suffix):
    if not stat["values"]:
        value = "N/A"
    else:
        value = f"{(sum(stat['values']) / len(stat['values'])) * scale:.2f}{suffix}"

    notes = []
    if len(stat["values"]) != total:
        notes.append(f"finite {len(stat['values'])}/{total}")
    for key in ("missing", "nan", "inf", "invalid"):
        if stat[key]:
            notes.append(f"{key} {stat[key]}")
    if notes:
        value = f"{value} ({', '.join(notes)})"
    return value

def calculate_metrics(exp_path):
    exp_name = os.path.basename(exp_path)
    # Evaluate the standard validation splits.
    splits = ['val_seen', 'val_unseen']
    
    results = {}
    found_any = False
    
    for split in splits:
        split_dir = os.path.join(exp_path, split)
        if not os.path.exists(split_dir):
            continue

        # Find episode-level JSON result files below the split directory.
        json_files = glob.glob(os.path.join(split_dir, "*/*.json"))
        if not json_files:
            continue

        found_any = True
        metric_count = 0
        metric_stats = {name: _new_metric_stat() for name, _, _, _, _ in METRIC_SPECS}

        for f_path in json_files:
            try:
                with open(f_path, 'r') as f:
                    data = json.load(f)
                    metrics = data.get('metrics', {})
                    if metrics:
                        metric_count += 1
                        for name, key, _, _, _ in METRIC_SPECS:
                            if key in metrics:
                                _add_metric_value(metric_stats[name], metrics[key])
                            else:
                                metric_stats[name]["missing"] += 1
            except Exception:
                continue

        if not metric_count:
            continue
        
        results[split] = {
            "count": metric_count,
            "stats": metric_stats,
        }

    if found_any:
        print(f"{'='*60}")
        print(f" Experiment: {exp_name}")
        print(f" Path: {exp_path}")
        print(f"{'-'*60}")
        for split, metrics in results.items():
            print(f"[ {split.upper()} ]")
            print(f"  Processed Episodes: {metrics['count']}")
            for name, _, label, scale, suffix in METRIC_SPECS:
                value = _format_metric(metrics["stats"][name], metrics["count"], scale, suffix)
                if name == "sr":
                    value = f"\033[92m{value}\033[0m"
                print(f"  {label + ':':<23} {value}")
        print(f"{'='*60}\n")

def find_all_exps(root_dir):
    exps = []
    for root, dirs, files in os.walk(root_dir):
        # Treat a directory containing a validation split as an experiment root.
        if 'val_seen' in dirs or 'val_unseen' in dirs:
            exps.append(root)
    return sorted(exps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None, help="Name of specific experiment folder (optional)")
    parser.add_argument("--result_dir", type=str, default="eval_results", help="Root directory for results")
    args = parser.parse_args()

    if args.exp_name:
        # Search below the result root when an experiment name is provided.
        specific_path = os.path.join(args.result_dir, args.exp_name)
        if os.path.exists(specific_path):
            calculate_metrics(specific_path)
        else:
            # Fall back to a recursive search.
            all_exps = find_all_exps(args.result_dir)
            matched = [e for e in all_exps if os.path.basename(e) == args.exp_name]
            if matched:
                for m in matched: calculate_metrics(m)
            else:
                print(f"Error: Experiment '{args.exp_name}' not found in {args.result_dir}")
    else:
        # Analyze all experiments below the result root.
        all_exps = find_all_exps(args.result_dir)
        if not all_exps:
            print(f"No experiments found in {args.result_dir}")
        else:
            print(f"Found {len(all_exps)} experiments. Analyzing...\n")
            for exp_path in all_exps:
                calculate_metrics(exp_path)
