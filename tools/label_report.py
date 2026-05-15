#!/usr/bin/env python3
"""
Generates a coverage report of platform label usage across open issues.

Useful for monitoring adoption of the platform: power-z / platform: spyre-aiu
tagging taxonomy after introduction.
"""

import os
import requests
from collections import Counter

TOKEN = os.getenv("GITHUB_TOKEN", "")
ORG   = os.getenv("GITHUB_ORG", "torch-spyre")
REPO  = os.getenv("GITHUB_REPO", "torch-spyre")

PLATFORM_LABELS = [
    "platform: power-z",
    "platform: spyre-aiu",
    "platform: x86",
    "is_power_z_dev",
]


def fetch_issues(state: str = "open") -> list:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    issues, page = [], 1
    while True:
        r = requests.get(
            f"https://api.github.com/repos/{ORG}/{REPO}/issues",
            headers=headers,
            params={"state": state, "per_page": 100, "page": page},
        )
        batch = r.json()
        if not batch:
            break
        issues.extend([i for i in batch if "pull_request" not in i])
        page += 1
    return issues


def report(issues: list) -> None:
    label_counts = Counter()
    untagged = 0

    for issue in issues:
        labels = [l["name"] for l in issue.get("labels", [])]
        platform_labels = [l for l in labels if l in PLATFORM_LABELS]
        if platform_labels:
            for l in platform_labels:
                label_counts[l] += 1
        else:
            untagged += 1

    total = len(issues)
    print(f"\nPlatform label coverage report ({total} issues)")
    print("-" * 45)
    for label, count in label_counts.most_common():
        pct = count / total * 100
        print(f"  {label:<30} {count:4d}  ({pct:.1f}%)")
    print(f"  {'(no platform label)':<30} {untagged:4d}  ({untagged/total*100:.1f}%)")


if __name__ == "__main__":
    issues = fetch_issues("open")
    report(issues)
