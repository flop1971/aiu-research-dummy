#!/usr/bin/env python3
"""
Benchmark runner for Spyre AIU throughput and latency measurements.

Runs a configurable set of microbenchmarks and writes results to JSON.
Used by the nightly CI benchmark workflow (runtests_nightly.yaml).
"""

import json
import time
import argparse
from typing import Dict, List


def run_matmul_benchmark(sizes: List[int], iterations: int = 100) -> Dict:
    """Benchmark matrix multiply throughput across tensor sizes."""
    results = {}
    for size in sizes:
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            # Placeholder: actual AIU kernel call
            time.sleep(0.0001)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0)   # convert to microseconds
        results[f"matmul_{size}x{size}"] = {
            "mean_us": sum(times) / len(times),
            "min_us":  min(times),
            "max_us":  max(times),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="AIU benchmark runner")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    results = run_matmul_benchmark(
        sizes=[128, 256, 512, 1024],
        iterations=args.iterations,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
