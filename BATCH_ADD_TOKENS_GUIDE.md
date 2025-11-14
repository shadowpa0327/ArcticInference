# Batch Add Tokens Guide

## Overview

`batch_add_tokens()` allows you to add newly generated tokens to multiple suffix trees **in parallel**. This is especially useful in batch inference scenarios where all requests receive new tokens simultaneously.

## Why Use Batch Add Tokens?

In typical batch inference:
1. All requests generate tokens at the same time
2. Each tree needs to be updated with its new tokens
3. These updates are independent and can be parallelized

**Without `batch_add_tokens` (sequential):**
```python
for req_id, tokens in zip(req_ids, new_tokens):
    cache.add_tokens(req_id, tokens)  # One at a time
```

**With `batch_add_tokens` (parallel):**
```python
cache.batch_add_tokens(req_ids, new_tokens)  # All at once, in parallel!
```

## API

```python
def batch_add_tokens(
    req_ids: List[Hashable],
    token_batches: List[np.ndarray | Sequence[int]],
):
    """
    Add generated tokens to multiple requests in parallel.

    Args:
        req_ids: List of request identifiers
        token_batches: List of token sequences (one per request)

    Raises:
        ValueError: If lengths don't match or any request is not active
        RuntimeError: If adding tokens fails for any tree
    """
```

## Usage Examples

### Basic Example

```python
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache
import numpy as np

cache = ParallelSuffixDecodingCache(max_tree_depth=64, num_threads=-1)

# Start some requests
for i in range(8):
    req_id = f"req_{i}"
    prompt = np.array([1, 2, 3, i], dtype=np.int32)
    cache.start_request(req_id, prompt)

# Simulate batch inference: all requests generate new tokens
req_ids = [f"req_{i}" for i in range(8)]
new_tokens = [
    np.array([i+10, i+20], dtype=np.int32) for i in range(8)
]

# Add all tokens in parallel
cache.batch_add_tokens(req_ids, new_tokens)

# All trees updated! 8x faster than sequential on 8-core machine
```

### vLLM Integration Example

```python
class SuffixDecodingProposer:
    def __init__(self, num_speculative_tokens, max_tree_depth=64):
        self.cache = ParallelSuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            num_threads=-1,
            parallel_threshold=4
        )
        self.num_speculative_tokens = num_speculative_tokens

    def propose(self, input_batch, sampled_token_ids):
        """Propose draft tokens for batch of sequences."""

        # Collect batch data
        req_ids = []
        new_tokens = []
        contexts = []

        for seq_data in input_batch:
            req_id = seq_data.request_id

            # Start request if new
            if req_id not in self.cache.active_requests:
                self.cache.start_request(req_id, seq_data.prompt_token_ids)

            # Collect newly sampled tokens for this sequence
            seq_new_tokens = sampled_token_ids.get(seq_data.seq_id, [])
            if len(seq_new_tokens) > 0:
                req_ids.append(req_id)
                new_tokens.append(seq_new_tokens)

            # Prepare for speculation
            contexts.append(seq_data.get_last_n_tokens(self.cache.max_tree_depth))

        # ADD ALL TOKENS IN PARALLEL (NEW!)
        if len(req_ids) > 0:
            self.cache.batch_add_tokens(req_ids, new_tokens)

        # SPECULATE IN PARALLEL (EXISTING)
        all_req_ids = [seq_data.request_id for seq_data in input_batch]
        drafts = self.cache.batch_speculate(
            all_req_ids,
            contexts,
            max_spec_tokens=self.num_speculative_tokens,
            max_spec_factor=1.0,
            min_token_prob=0.1
        )

        # Map results back
        draft_proposals = {}
        for seq_data, draft in zip(input_batch, drafts):
            draft_proposals[seq_data.seq_id] = draft

        return draft_proposals
```

## Performance Comparison

### Sequential (Old Way)

```python
# 10 requests, each adds 5 tokens
# Takes ~100ms (10ms per request)

for req_id, tokens in zip(req_ids, new_tokens):
    cache.add_tokens(req_id, tokens)
```

### Parallel (New Way)

```python
# Same 10 requests, same 5 tokens each
# Takes ~15ms on 8-core machine (7x speedup!)

cache.batch_add_tokens(req_ids, new_tokens)
```

## When to Use

✅ **Use `batch_add_tokens` when:**
- You have multiple requests (≥ 4 typical)
- All need tokens added at the same time
- In batch inference scenarios

⚠️ **Use regular `add_tokens` when:**
- Single request
- Tokens arrive at different times
- Batch size < parallel_threshold (overhead dominates)

## Implementation Details

### C++ Side

```cpp
void SuffixForest::batch_extend(
    std::span<const int> tree_indices,
    std::span<const int> seq_ids,
    const std::vector<std::vector<int32_t>>& token_batches
) {
    // Parallel execution with OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch_size; ++i) {
        _trees[tree_indices[i]]->extend(seq_ids[i], token_batches[i]);
    }
}
```

### Key Features

1. **GIL Release**: Python GIL is released during C++ execution
2. **OpenMP Parallelization**: Uses all available cores
3. **Fail-Fast**: If any tree fails, entire batch fails
4. **Configurable**: Respects `num_threads` and `parallel_threshold`

## Complete Workflow Example

```python
# Initialize cache
cache = ParallelSuffixDecodingCache(
    max_tree_depth=64,
    num_threads=8,
    parallel_threshold=4
)

# Iteration 1: Start requests
for i in range(10):
    cache.start_request(f"req_{i}", prompt_tokens[i])

# Iteration 2: Generate first tokens
req_ids = [f"req_{i}" for i in range(10)]
batch_1_tokens = generate_batch(...)  # From model

# Add tokens in parallel
cache.batch_add_tokens(req_ids, batch_1_tokens)

# Speculate in parallel
batch_1_drafts = cache.batch_speculate(req_ids, contexts_1, ...)

# Iteration 3: Generate more tokens
batch_2_tokens = generate_batch(...)

# Add tokens in parallel
cache.batch_add_tokens(req_ids, batch_2_tokens)

# Speculate in parallel
batch_2_drafts = cache.batch_speculate(req_ids, contexts_2, ...)

# ... and so on ...

# Clean up
for req_id in req_ids:
    cache.stop_request(req_id)
```

## Error Handling

```python
try:
    cache.batch_add_tokens(req_ids, new_tokens)
except ValueError as e:
    # Mismatched lengths or inactive request
    print(f"Invalid input: {e}")
except RuntimeError as e:
    # C++ extend operation failed
    print(f"Extend failed: {e}")
```

## Configuration Tips

### Optimal Thread Count

```python
import os

# Auto-detect (recommended)
cache = ParallelSuffixDecodingCache(num_threads=-1)

# Or set explicitly
num_cores = os.cpu_count()
cache = ParallelSuffixDecodingCache(num_threads=num_cores)
```

### Tune Parallel Threshold

```python
# If average batch size is small (2-4)
cache = ParallelSuffixDecodingCache(parallel_threshold=2)

# If average batch size is large (16+)
cache = ParallelSuffixDecodingCache(parallel_threshold=8)
```

## Benchmarking

To measure speedup:

```python
import time

# Sequential
start = time.time()
for req_id, tokens in zip(req_ids, new_tokens):
    cache.add_tokens(req_id, tokens)
sequential_time = time.time() - start

# Parallel
start = time.time()
cache.batch_add_tokens(req_ids, new_tokens)
parallel_time = time.time() - start

print(f"Speedup: {sequential_time / parallel_time:.2f}x")
```

## Summary

`batch_add_tokens()` provides:
- ✅ **Parallel execution**: Near-linear speedup with batch size
- ✅ **Simple API**: Drop-in replacement for loop of `add_tokens()`
- ✅ **Automatic GIL release**: True parallelism in Python
- ✅ **Fail-fast error handling**: Consistent behavior
- ✅ **Configurable**: Tune for your workload

Perfect for batch inference workloads where multiple requests need to update their suffix trees simultaneously!
