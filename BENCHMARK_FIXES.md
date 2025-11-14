# How to Fix the Parallel Suffix Decoding Benchmark

## Root Causes of No Speedup

1. **Parallel threshold**: Batch sizes < 8 run sequentially (by design)
2. **Thread overhead**: Operations are too fast (microseconds), thread overhead dominates
3. **Small trees**: Trees with ~1024 tokens are too simple for parallelization to help

## Fixes to Apply

### Fix #1: Use realistic tree sizes

**File**: `benchmark_parallel_suffix_decoding.py:106`

**Current** (creates tiny 1024-token trees):
```python
prompt_len = 1024 + (i % 5) * 16  # Vary length: 4096, 4112, ...
```

**Fix to** (create larger, more realistic trees):
```python
prompt_len = 4096 + (i % 5) * 16  # Actually use 4096 as the comment says!
```

### Fix #2: Lower parallel threshold for testing

**File**: `benchmark_parallel_suffix_decoding.py:406`

**Current**:
```python
parallel_threshold=8,  # Lower threshold to see speedup earlier
```

**Fix to**:
```python
parallel_threshold=2,  # Allow parallelization at batch size 2 for testing
```

⚠️ **Note**: In production, threshold=8 might be better to avoid overhead. For benchmarking, lower it.

### Fix #3: Add more tokens per operation

The benchmark adds only 15-20 tokens per operation, which is too small.

**File**: `benchmark_parallel_suffix_decoding.py:137`

**Current**:
```python
token_batches = [
    np.array([i+100+j for j in range(15)], dtype=np.int32)  # Only 15 tokens
    for i in range(num_requests)
]
```

**Fix to**:
```python
token_batches = [
    np.array([i+100+j for j in range(50)], dtype=np.int32)  # 50 tokens = more work
    for i in range(num_requests)
]
```

**File**: `benchmark_parallel_suffix_decoding.py:192`

**Current**:
```python
contexts = [
    np.array([1, 2, 3] + [i+10+j for j in range(20)], dtype=np.int32)  # 23 tokens
    for i in range(num_requests)
]
```

**Fix to**:
```python
contexts = [
    np.array([1, 2, 3] + [i+10+j for j in range(60)], dtype=np.int32)  # 63 tokens
    for i in range(num_requests)
]
```

### Fix #4: Increase iterations to get stable measurements

**File**: `benchmark_parallel_suffix_decoding.py:408-409`

**Current**:
```python
warmup_iterations=5,
benchmark_iterations=20
```

**Fix to**:
```python
warmup_iterations=10,
benchmark_iterations=50  # More iterations for stable timing
```

### Fix #5: Compare per-request times, not absolute times

When batch size increases, total time increases even with parallelization.
You should compare **time per request**, not total time.

**Add this function** to `benchmark_parallel_suffix_decoding.py`:

```python
def print_per_request_analysis(results: List[Tuple[BenchmarkResult, BenchmarkResult]]):
    """Print per-request time analysis."""
    print("\n" + "=" * 90)
    print("PER-REQUEST TIME ANALYSIS")
    print("=" * 90)
    print(f"{'Requests':>10} | {'Extend ms/req':>14} | {'Speculate ms/req':>18} | {'Parallel?':>10}")
    print("-" * 90)

    for extend_result, speculate_result in results:
        extend_per_req = (extend_result.parallel_time * 1000) / extend_result.num_requests
        spec_per_req = (speculate_result.parallel_time * 1000) / speculate_result.num_requests
        parallel = "✓" if extend_result.num_requests >= 8 else "sequential"

        print(f"{extend_result.num_requests:>10} | {extend_per_req:>14.4f} | "
              f"{spec_per_req:>18.4f} | {parallel:>10}")

    print("\nLook for DECREASING per-request times as batch size increases!")
    print("This indicates successful parallelization.")
```

**Call it** in `main()` after `print_summary(results)`:
```python
print_summary(results)
print_per_request_analysis(results)  # Add this line
```

## Expected Results After Fixes

With these fixes, you should see:

1. **Batch sizes 1-7**: Sequential execution (based on threshold)
2. **Batch sizes 8+**: Parallel execution
3. **Per-request time**: Should decrease as batch size increases (if parallelization helps)
4. **Speedup**: Look for 1.5-3x speedup on larger batches (8+), not 4x (4 threads) due to:
   - Thread synchronization overhead
   - Memory bandwidth limits
   - GIL release/acquire overhead

## Reality Check

**Important**: Suffix tree operations are memory-bound, not CPU-bound. Even with perfect parallelization, speedup is limited by:

- Memory bandwidth (all threads competing for RAM access)
- Cache coherency overhead
- Small amount of actual computation per operation

**Realistic expectations**:
- Batch size 8-16: 1.5-2x speedup
- Batch size 32+: 2-3x speedup
- Never expect 4x speedup with 4 threads for memory-intensive operations

## Quick Test Script

Run this to verify the fixes work:

```bash
# After applying fixes, run:
python benchmark_parallel_suffix_decoding.py

# Look for:
# 1. "sequential" marked for batch sizes < 8
# 2. Per-request times decreasing as batch size increases
# 3. Speedup > 1.0x for batch sizes >= 8
```

## Alternative: Test with Synthetic CPU-Heavy Load

If you want to see more dramatic speedup (for testing), artificially add CPU work:

```python
# In suffix_forest.cc:138 (inside the parallel loop)
// Add artificial work to make speedup more visible
volatile int dummy = 0;
for (int k = 0; k < 10000; ++k) {
    dummy += k;
}
tree.extend(seq_id, token_batches[i]);
```

This will show clear speedup because there's actual CPU work to parallelize.
But it doesn't reflect real performance!
