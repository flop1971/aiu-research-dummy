#!/usr/bin/env python3
"""
POWER/Z migration progress tracker.

Reads WP1/WP2 collector output CSV files and prints a summary of
platform tag adoption and POWER/Z development maturity signals.
"""

import os
import csv
import glob
from datetime import datetime


DATA_DIR_WP1 = os.path.join(os.path.dirname(__file__), "..", "data", "wp1")
DATA_DIR_WP2 = os.path.join(os.path.dirname(__file__), "..", "data", "wp2")


def latest_csv(directory: str, prefix: str) -> str | None:
    pattern = os.path.join(directory, f"{prefix}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarise_platform_tags(wp2_issues_csv: str) -> None:
    rows = read_csv(wp2_issues_csv)
    total = len(rows)
    if total == 0:
        print("  No WP2 issue data found.")
        return

    print(f"\nWP2 Issue Platform Distribution ({total} issues)")
    print("-" * 45)
    # platform column added by collect_wp2.py classify_platform helper
    from collections import Counter
    platforms = Counter(r.get("platform_classification", "unknown") for r in rows)
    for platform, count in platforms.most_common():
        print(f"  {platform:<20} {count:4d}  ({count/total*100:.1f}%)")


if __name__ == "__main__":
    f = latest_csv(DATA_DIR_WP2, "A1_A2_A3_A4_C3_D1_D2_issues")
    if f:
        summarise_platform_tags(f)
    else:
        print("No WP2 issues CSV found. Run collect_wp2.py first.")
