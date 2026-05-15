#!/usr/bin/env python3
"""Validates that all source files have a CODEOWNERS entry."""

import os
import sys


def load_codeowners(path: str = "CODEOWNERS") -> dict:
    owners = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    owners[parts[0]] = parts[1:]
    return owners


def check_coverage(source_root: str, owners: dict) -> list:
    uncovered = []
    for dirpath, _, filenames in os.walk(source_root):
        for fname in filenames:
            if fname.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, fname))
                if not any(rel.startswith(p.lstrip("/")) for p in owners):
                    uncovered.append(rel)
    return uncovered


if __name__ == "__main__":
    owners = load_codeowners()
    uncovered = check_coverage("torch_spyre", owners)
    if uncovered:
        print(f"WARNING: {len(uncovered)} files without CODEOWNERS entry")
        sys.exit(1)
    print("CODEOWNERS coverage: OK")
