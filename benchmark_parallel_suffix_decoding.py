#!/usr/bin/env python3
"""
Benchmark script comparing ParallelSuffixDecodingCache vs SuffixDecodingCache.

Tests performance across different numbers of concurrent requests, measuring:
1. Token extension time
2. Speculation time
3. Total time
4. Speedup

Assumes trees are already created and populated.
"""

import numpy as np
import time
import sys
from typing import List, Tuple
from dataclasses import dataclass

try:
    from arctic_inference.suffix_decoding import (
        SuffixDecodingCache,
        ParallelSuffixDecodingCache
    )
    print("✓ Successfully imported both cache implementations\n")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    num_requests: int
    operation: str  # "extend" or "speculate"
    sequential_time: float
    parallel_time: float
    speedup: float

    def __str__(self):
        return (f"Requests: {self.num_requests:3d} | "
                f"{self.operation:10s} | "
                f"Sequential: {self.sequential_time*1000:7.2f}ms | "
                f"Parallel: {self.parallel_time*1000:7.2f}ms | "
                f"Speedup: {self.speedup:5.2f}x")


class BenchmarkSuite:
    """Benchmark suite for comparing cache implementations."""

    def __init__(self,
                 max_tree_depth: int = 64,
                 num_threads: int = -1,
                 parallel_threshold: int = 2,
                 warmup_iterations: int = 2,
                 benchmark_iterations: int = 5):
        """
        Initialize benchmark suite.

        Args:
            max_tree_depth: Maximum depth for suffix trees
            num_threads: Number of threads for parallel cache (-1 = auto)
            parallel_threshold: Minimum batch size for parallelization
            warmup_iterations: Number of warmup runs before timing
            benchmark_iterations: Number of timed iterations to average
        """
        self.max_tree_depth = max_tree_depth
        self.num_threads = num_threads
        self.parallel_threshold = parallel_threshold
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        # Will be set during benchmark
        self.sequential_cache = None
        self.parallel_cache = None

    def setup_caches(self, num_requests: int) -> Tuple[List[str], List]:
        """
        Setup both caches with identical initial state.

        Args:
            num_requests: Number of requests to create

        Returns:
            Tuple of (req_ids, initial_prompts)
        """
        # Create fresh caches
        self.sequential_cache = SuffixDecodingCache(
            max_tree_depth=self.max_tree_depth,
            max_cached_requests=0,  # Disable global tree for fair comparison
        )

        self.parallel_cache = ParallelSuffixDecodingCache(
            max_tree_depth=self.max_tree_depth,
            num_threads=self.num_threads,
            parallel_threshold=self.parallel_threshold
        )

        # Generate request IDs and prompts
        req_ids = [f"req_{i}" for i in range(num_requests)]
        prompts = []

        # Start all requests with identical data
        for i in range(num_requests):
            # Create varied prompts to make trees more realistic
            prompt_len = 1024 + (i % 5) * 16  # Vary length: 4096, 4112, 4128, 4144, 4160
            prompt = np.array([1, 2, 3] + [i] * (prompt_len - 3), dtype=np.int32)
            prompts.append(prompt)

            # Start request in both caches
            self.sequential_cache.start_request(req_ids[i], prompt)
            self.parallel_cache.start_request(req_ids[i], prompt)

            # Add more initial tokens to build up larger trees
            # Add tokens in multiple stages to create more complex tree structure
            for stage in range(3):
                tokens = np.array([i+stage*100+j for j in range(10, 30)], dtype=np.int32)
                self.sequential_cache.add_active_response(req_ids[i], tokens)
                self.parallel_cache.add_tokens(req_ids[i], tokens)

        return req_ids, prompts

    def benchmark_extend(self, req_ids: List[str]) -> BenchmarkResult:
        """
        Benchmark token extension (adding new tokens).

        Args:
            req_ids: List of request IDs to process

        Returns:
            BenchmarkResult with timing data
        """
        num_requests = len(req_ids)

        # Prepare token batches - add more tokens to make operations measurable
        token_batches = [
            np.array([i+100+j for j in range(15)], dtype=np.int32)
            for i in range(num_requests)
        ]

        # Warmup
        for _ in range(self.warmup_iterations):
            # Sequential warmup
            for req_id, tokens in zip(req_ids, token_batches):
                self.sequential_cache.add_active_response(req_id, tokens)

            # Parallel warmup
            self.parallel_cache.batch_add_tokens(req_ids, token_batches)

        # Benchmark sequential
        sequential_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            for req_id, tokens in zip(req_ids, token_batches):
                self.sequential_cache.add_active_response(req_id, tokens)
            sequential_times.append(time.perf_counter() - start)

        # Benchmark parallel
        parallel_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            self.parallel_cache.batch_add_tokens(req_ids, token_batches)
            parallel_times.append(time.perf_counter() - start)

        # Calculate averages
        seq_time = np.mean(sequential_times)
        par_time = np.mean(parallel_times)
        speedup = seq_time / par_time if par_time > 0 else 0

        return BenchmarkResult(
            num_requests=num_requests,
            operation="extend",
            sequential_time=seq_time,
            parallel_time=par_time,
            speedup=speedup
        )

    def benchmark_speculate(self, req_ids: List[str]) -> BenchmarkResult:
        """
        Benchmark speculation.

        Args:
            req_ids: List of request IDs to process

        Returns:
            BenchmarkResult with timing data
        """
        num_requests = len(req_ids)

        # Prepare contexts - use longer contexts for more realistic speculation
        contexts = [
            np.array([1, 2, 3] + [i+10+j for j in range(20)], dtype=np.int32)
            for i in range(num_requests)
        ]

        # Speculation parameters
        max_spec_tokens = 20
        max_spec_factor = 1.0
        min_token_prob = 0.1

        # Warmup
        for _ in range(self.warmup_iterations):
            # Sequential warmup
            for req_id, ctx in zip(req_ids, contexts):
                self.sequential_cache.speculate(
                    req_id, ctx, max_spec_tokens, max_spec_factor,
                    min_token_prob=min_token_prob
                )

            # Parallel warmup
            self.parallel_cache.batch_speculate(
                req_ids, contexts, max_spec_tokens, max_spec_factor,
                min_token_prob=min_token_prob
            )

        # Benchmark sequential
        sequential_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            for req_id, ctx in zip(req_ids, contexts):
                self.sequential_cache.speculate(
                    req_id, ctx, max_spec_tokens, max_spec_factor,
                    min_token_prob=min_token_prob
                )
            sequential_times.append(time.perf_counter() - start)

        # Benchmark parallel
        parallel_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            self.parallel_cache.batch_speculate(
                req_ids, contexts, max_spec_tokens, max_spec_factor,
                min_token_prob=min_token_prob
            )
            parallel_times.append(time.perf_counter() - start)

        # Calculate averages
        seq_time = np.mean(sequential_times)
        par_time = np.mean(parallel_times)
        speedup = seq_time / par_time if par_time > 0 else 0

        return BenchmarkResult(
            num_requests=num_requests,
            operation="speculate",
            sequential_time=seq_time,
            parallel_time=par_time,
            speedup=speedup
        )

    def run_benchmark(self, num_requests: int) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Run complete benchmark for a given number of requests.

        Args:
            num_requests: Number of concurrent requests

        Returns:
            Tuple of (extend_result, speculate_result)
        """
        print(f"\nBenchmarking with {num_requests} requests...")

        # Setup
        req_ids, _ = self.setup_caches(num_requests)

        # Run benchmarks
        extend_result = self.benchmark_extend(req_ids)
        speculate_result = self.benchmark_speculate(req_ids)

        # Cleanup
        for req_id in req_ids:
            self.sequential_cache.stop_request(req_id)
            self.parallel_cache.stop_request(req_id)

        return extend_result, speculate_result


def print_header():
    """Print benchmark header."""
    print("=" * 90)
    print("Parallel Suffix Decoding Cache Benchmark")
    print("=" * 90)
    print()


def print_config(suite: BenchmarkSuite):
    """Print benchmark configuration."""
    print("Configuration:")
    print(f"  Max Tree Depth:        {suite.max_tree_depth}")
    print(f"  Num Threads:           {suite.num_threads} (auto-detect)" if suite.num_threads == -1
          else f"  Num Threads:           {suite.num_threads}")
    print(f"  Parallel Threshold:    {suite.parallel_threshold}")
    print(f"  Warmup Iterations:     {suite.warmup_iterations}")
    print(f"  Benchmark Iterations:  {suite.benchmark_iterations}")
    print()


def print_results_table(results: List[Tuple[BenchmarkResult, BenchmarkResult]]):
    """Print results in a formatted table."""
    print("\n" + "=" * 90)
    print("EXTEND (Add Tokens) Results")
    print("=" * 90)
    print(f"{'Requests':>10} | {'Sequential':>12} | {'Parallel':>12} | {'Speedup':>10}")
    print("-" * 90)

    for extend_result, _ in results:
        print(f"{extend_result.num_requests:>10} | "
              f"{extend_result.sequential_time*1000:>10.2f} ms | "
              f"{extend_result.parallel_time*1000:>10.2f} ms | "
              f"{extend_result.speedup:>8.2f}x")

    print("\n" + "=" * 90)
    print("SPECULATE Results")
    print("=" * 90)
    print(f"{'Requests':>10} | {'Sequential':>12} | {'Parallel':>12} | {'Speedup':>10}")
    print("-" * 90)

    for _, speculate_result in results:
        print(f"{speculate_result.num_requests:>10} | "
              f"{speculate_result.sequential_time*1000:>10.2f} ms | "
              f"{speculate_result.parallel_time*1000:>10.2f} ms | "
              f"{speculate_result.speedup:>8.2f}x")


def print_summary(results: List[Tuple[BenchmarkResult, BenchmarkResult]]):
    """Print summary statistics."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Find best speedups
    extend_results = [r[0] for r in results]
    speculate_results = [r[1] for r in results]

    best_extend = max(extend_results, key=lambda r: r.speedup)
    best_speculate = max(speculate_results, key=lambda r: r.speedup)

    print(f"\nBest Extend Speedup:     {best_extend.speedup:.2f}x "
          f"(at {best_extend.num_requests} requests)")
    print(f"Best Speculate Speedup:  {best_speculate.speedup:.2f}x "
          f"(at {best_speculate.num_requests} requests)")

    # Average speedup for larger batches (>= 8)
    large_batch_extend = [r for r in extend_results if r.num_requests >= 8]
    large_batch_speculate = [r for r in speculate_results if r.num_requests >= 8]

    if large_batch_extend:
        avg_extend_speedup = np.mean([r.speedup for r in large_batch_extend])
        print(f"\nAverage Extend Speedup (≥8 requests):    {avg_extend_speedup:.2f}x")

    if large_batch_speculate:
        avg_speculate_speedup = np.mean([r.speedup for r in large_batch_speculate])
        print(f"Average Speculate Speedup (≥8 requests): {avg_speculate_speedup:.2f}x")

    # Time savings
    print("\nTime Savings (for 32 requests):")
    r32_extend = next((r for r in extend_results if r.num_requests == 32), None)
    r32_speculate = next((r for r in speculate_results if r.num_requests == 32), None)

    if r32_extend:
        savings = (r32_extend.sequential_time - r32_extend.parallel_time) * 1000
        print(f"  Extend:    {savings:.2f}ms saved per batch")

    if r32_speculate:
        savings = (r32_speculate.sequential_time - r32_speculate.parallel_time) * 1000
        print(f"  Speculate: {savings:.2f}ms saved per batch")


def save_results_csv(results: List[Tuple[BenchmarkResult, BenchmarkResult]],
                     filename: str = "benchmark_results.csv"):
    """Save results to CSV file."""
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'num_requests',
            'extend_sequential_ms', 'extend_parallel_ms', 'extend_speedup',
            'speculate_sequential_ms', 'speculate_parallel_ms', 'speculate_speedup'
        ])

        # Write data
        for extend_result, speculate_result in results:
            writer.writerow([
                extend_result.num_requests,
                extend_result.sequential_time * 1000,
                extend_result.parallel_time * 1000,
                extend_result.speedup,
                speculate_result.sequential_time * 1000,
                speculate_result.parallel_time * 1000,
                speculate_result.speedup
            ])

    print(f"\n✓ Results saved to {filename}")


def main():
    """Main benchmark runner."""
    print_header()

    # Configuration
    suite = BenchmarkSuite(
        max_tree_depth=64,
        num_threads=4,  # Use 4 threads - good balance for most systems
        parallel_threshold=8,  # Lower threshold to see speedup earlier
        warmup_iterations=5,
        benchmark_iterations=20
    )

    print_config(suite)

    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Each configuration will run {suite.benchmark_iterations} iterations")
    print("\nStarting benchmarks...\n")

    # Run benchmarks
    results = []
    for num_requests in batch_sizes:
        try:
            extend_result, speculate_result = suite.run_benchmark(num_requests)
            results.append((extend_result, speculate_result))

            # Print immediate results
            print(f"  Extend:    {extend_result}")
            print(f"  Speculate: {speculate_result}")

        except Exception as e:
            print(f"✗ Benchmark failed for {num_requests} requests: {e}")
            import traceback
            traceback.print_exc()

    # Print results
    print_results_table(results)
    print_summary(results)

    # Save to CSV
    try:
        save_results_csv(results)
    except Exception as e:
        print(f"Warning: Could not save CSV: {e}")

    print("\n" + "=" * 90)
    print("Benchmark Complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
