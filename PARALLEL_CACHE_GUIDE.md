@# ParallelSuffixDecodingCache Guide

## Overview

`ParallelSuffixDecodingCache` is a new, clean implementation of suffix decoding cache that uses C++ `SuffixForest` for parallel batched speculation. It provides the same functionality as `SuffixDecodingCache` but with better performance for batched workloads.

## Key Differences from SuffixDecodingCache

| Feature | SuffixDecodingCache | ParallelSuffixDecodingCache |
|---------|---------------------|------------------------------|
| **Global Tree** | Had global tree (now deprecated) | No global tree |
| **Parallelization** | Sequential | OpenMP parallel batch speculation |
| **Implementation** | Python dict of trees | C++ SuffixForest |
| **API Complexity** | Legacy parameters, backward compat | Clean, focused API |
| **Primary Use Case** | Single request speculation | Batch speculation |
| **Code Clarity** | Mixed old/new code | Clean, separate implementation |

---

## Quick Start

### Basic Usage

```python
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache
import numpy as np

# Create cache
cache = ParallelSuffixDecodingCache(
    max_tree_depth=64,
    num_threads=-1,  # Auto-detect
    parallel_threshold=4
)

# Start a request
req_id = "request_1"
prompt = np.array([1, 2, 3, 4, 5], dtype=np.int32)
cache.start_request(req_id, prompt)

# Add generated tokens
generated = np.array([6, 7, 8], dtype=np.int32)
cache.add_tokens(req_id, generated)

# Speculate
context = np.array([3, 4, 5], dtype=np.int32)
draft = cache.speculate(req_id, context, max_spec_tokens=10)

print(f"Speculated tokens: {draft.token_ids}")
print(f"Score: {draft.score}")

# Stop request
cache.stop_request(req_id)
```

### Batch Speculation (Recommended)

```python
# Start multiple requests
for i in range(10):
    req_id = f"req_{i}"
    prompt = np.array([1, 2, 3, i], dtype=np.int32)
    cache.start_request(req_id, prompt)

    # Add some generated tokens
    generated = np.array([i+10, i+20], dtype=np.int32)
    cache.add_tokens(req_id, generated)

# Batch speculate (parallelized!)
req_ids = [f"req_{i}" for i in range(10)]
contexts = [np.array([1, 2, 3], dtype=np.int32) for _ in range(10)]

drafts = cache.batch_speculate(
    req_ids,
    contexts,
    max_spec_tokens=5,
    max_spec_factor=1.0,
    min_token_prob=0.1
)

# Process results
for req_id, draft in zip(req_ids, drafts):
    print(f"{req_id}: {len(draft.token_ids)} tokens, score={draft.score:.3f}")
```

---

## API Reference

### Constructor

```python
ParallelSuffixDecodingCache(
    max_tree_depth: int = 64,
    num_threads: int = -1,
    parallel_threshold: int = 4
)
```

**Parameters:**
- `max_tree_depth`: Maximum depth of suffix trees (context window size)
- `num_threads`:
  - `-1` = auto-detect from CPU count (recommended)
  - `0` = sequential execution (no parallelization)
  - `>0` = use specified number of threads
- `parallel_threshold`: Minimum batch size to trigger parallelization

### Properties

```python
cache.max_tree_depth -> int          # Maximum tree depth
cache.num_threads -> int              # Actual thread count being used
cache.parallel_threshold -> int       # Parallelization threshold
cache.active_requests -> KeysView     # View of active request IDs
cache.num_active_requests -> int      # Number of active requests
```

### Methods

#### `start_request(req_id, prompt_token_ids)`

Start processing a new request.

```python
cache.start_request(
    req_id="request_123",
    prompt_token_ids=np.array([1, 2, 3], dtype=np.int32)
)
```

**Raises:**
- `ValueError`: If request with same ID already active

---

#### `stop_request(req_id)`

Stop processing a request and free its resources.

```python
cache.stop_request("request_123")
```

**Raises:**
- `ValueError`: If request is not active

---

#### `add_tokens(req_id, token_ids)`

Add generated tokens to a request's suffix tree.

```python
cache.add_tokens(
    req_id="request_123",
    token_ids=np.array([10, 20, 30], dtype=np.int32)
)
```

**Raises:**
- `ValueError`: If request is not active

---

#### `speculate(req_id, context, ...)`

Speculate for a single request.

```python
draft = cache.speculate(
    req_id="request_123",
    context=np.array([1, 2, 3], dtype=np.int32),
    max_spec_tokens=10,
    max_spec_factor=1.0,
    max_spec_offset=0.0,
    min_token_prob=0.1,
    use_tree_spec=False
)
```

**Returns:** `SuffixDecodingDraft`

**Note:** For multiple requests, use `batch_speculate()` instead for better performance.

---

#### `batch_speculate(req_ids, contexts, ...)`

**Primary method for parallel speculation.**

```python
drafts = cache.batch_speculate(
    req_ids=["req_1", "req_2", "req_3"],
    contexts=[
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32),
        np.array([7, 8, 9], dtype=np.int32)
    ],
    max_spec_tokens=10,
    max_spec_factor=1.0,
    max_spec_offset=0.0,
    min_token_prob=0.1,
    use_tree_spec=False
)
```

**Parameters:**
- `req_ids`: List of request identifiers
- `contexts`: List of context sequences (one per request)
- `max_spec_tokens`: Maximum tokens to speculate per request
- `max_spec_factor`: Limits speculation: `num_tokens ≤ max_spec_factor * match_len + max_spec_offset`
- `max_spec_offset`: Offset for speculation limit
- `min_token_prob`: Minimum probability threshold
- `use_tree_spec`: If True, uses tree-based (beam search-like) speculation; if False, uses greedy path-based

**Returns:** `List[SuffixDecodingDraft]`

**Raises:**
- `ValueError`: If `req_ids` and `contexts` have different lengths, or if any request is not active
- `RuntimeError`: If speculation fails for any tree

---

#### `get_stats()`

Get cache statistics.

```python
stats = cache.get_stats()
# {
#     'num_active_requests': 10,
#     'max_tree_depth': 64,
#     'num_threads': 8,
#     'parallel_threshold': 4,
#     'num_trees_in_forest': 10
# }
```

---

## Performance Optimization

### Thread Configuration

**Auto-detect (Recommended):**
```python
cache = ParallelSuffixDecodingCache(num_threads=-1)
```

**Manual tuning:**
```python
import os
num_cores = os.cpu_count()
cache = ParallelSuffixDecodingCache(num_threads=num_cores)
```

**Disable parallelization (debugging):**
```python
cache = ParallelSuffixDecodingCache(num_threads=0)
```

### Batch Size Tuning

**Small batches (< 4):** Sequential execution (overhead would dominate)

**Medium batches (4-16):** Optimal parallelization (default threshold: 4)

**Large batches (16+):** Maximum benefit

**Custom threshold:**
```python
# Parallelize only for batches ≥ 8
cache = ParallelSuffixDecodingCache(parallel_threshold=8)
```

### Memory Optimization

**Tree depth:** Controls memory usage per tree

```python
# Lower depth = less memory, less context
cache = ParallelSuffixDecodingCache(max_tree_depth=32)

# Higher depth = more memory, more context
cache = ParallelSuffixDecodingCache(max_tree_depth=128)
```

---

## Migration from SuffixDecodingCache

### API Mapping

| SuffixDecodingCache | ParallelSuffixDecodingCache |
|---------------------|------------------------------|
| `start_request(req_id, prompt)` | `start_request(req_id, prompt)` ✓ Same |
| `stop_request(req_id)` | `stop_request(req_id)` ✓ Same |
| `add_active_response(req_id, tokens)` | `add_tokens(req_id, tokens)` ⚠️ Renamed |
| `speculate(req_id, context, ...)` | `speculate(req_id, context, ...)` ✓ Same |
| N/A | `batch_speculate(req_ids, contexts, ...)` ✨ New |
| `evict_cached_response(req_id)` | N/A (global tree removed) |

### Step-by-Step Migration

**Old code:**
```python
from arctic_inference.suffix_decoding import SuffixDecodingCache

cache = SuffixDecodingCache(max_tree_depth=64)

# ... in your inference loop ...
for seq_data in batch:
    req_id = seq_data.request_id
    if not cache.is_active(req_id):
        cache.start_request(req_id, seq_data.prompt)

    cache.add_active_response(req_id, seq_data.new_tokens)

    context = seq_data.get_context()
    draft = cache.speculate(req_id, context)
    # ... use draft ...
```

**New code:**
```python
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

cache = ParallelSuffixDecodingCache(
    max_tree_depth=64,
    num_threads=-1,
    parallel_threshold=4
)

# ... in your inference loop ...

# Collect all requests in batch
req_ids = []
contexts = []

for seq_data in batch:
    req_id = seq_data.request_id
    if req_id not in cache.active_requests:
        cache.start_request(req_id, seq_data.prompt)

    cache.add_tokens(req_id, seq_data.new_tokens)  # ⚠️ Renamed

    req_ids.append(req_id)
    contexts.append(seq_data.get_context())

# Single batched call (parallelized!)
drafts = cache.batch_speculate(req_ids, contexts)

# Map results back
for seq_data, draft in zip(batch, drafts):
    # ... use draft ...
```

---

## Testing & Validation

### Run Tests

```bash
# Test basic functionality
python test_parallel_cache.py

# Test C++ bindings directly
python test_suffix_forest.py
```

### Correctness Validation

The test suite compares `ParallelSuffixDecodingCache` against `SuffixDecodingCache` to ensure identical results:

```python
# Both should produce identical speculation results
original = SuffixDecodingCache(max_tree_depth=64)
parallel = ParallelSuffixDecodingCache(max_tree_depth=64)

# ... add same data to both ...

draft_original = original.speculate(req_id, context)
draft_parallel = parallel.speculate(req_id, context)

assert draft_original.token_ids == draft_parallel.token_ids
assert abs(draft_original.score - draft_parallel.score) < 0.001
```

---

## Troubleshooting

### ImportError: cannot import ParallelSuffixDecodingCache

**Cause:** C++ extension not built or SuffixForest not available

**Solution:**
```bash
# Rebuild C++ extension
uv pip install -e .
```

### Performance not improving with batching

**Possible causes:**

1. **Batch size too small:**
   ```python
   # Check your parallel threshold
   print(cache.parallel_threshold)  # Should be ≤ your batch size
   ```

2. **Only 1 thread:**
   ```python
   print(cache.num_threads)  # Should be > 1
   # If 1, OpenMP may not be available
   ```

3. **GIL not released:**
   - This should be automatic if C++ extension built correctly
   - Check build logs for OpenMP detection

### Different results from original cache

**This is expected** if the original cache used global tree. `ParallelSuffixDecodingCache` has no global tree, so cross-request patterns aren't captured.

**To get identical results:** Original cache must have global tree disabled:
```python
original = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=0)
```

---

## Best Practices

1. **Always use batch_speculate for multiple requests**
   ```python
   # ✓ Good
   drafts = cache.batch_speculate(req_ids, contexts)

   # ✗ Bad (loses parallelization)
   drafts = [cache.speculate(rid, ctx) for rid, ctx in zip(req_ids, contexts)]
   ```

2. **Set appropriate thread count**
   ```python
   # ✓ Good: Auto-detect
   cache = ParallelSuffixDecodingCache(num_threads=-1)

   # ⚠️ Careful: May oversubscribe if already parallel at higher level
   cache = ParallelSuffixDecodingCache(num_threads=os.cpu_count())
   ```

3. **Tune parallel threshold for your workload**
   ```python
   # Small average batch size (2-4)
   cache = ParallelSuffixDecodingCache(parallel_threshold=2)

   # Large average batch size (16+)
   cache = ParallelSuffixDecodingCache(parallel_threshold=8)
   ```

4. **Clean up requests**
   ```python
   # Always call stop_request when done
   try:
       # ... inference ...
   finally:
       cache.stop_request(req_id)
   ```

---

## Example: vLLM Integration

```python
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

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
        contexts = []

        for seq_data in input_batch:
            req_id = seq_data.request_id

            # Start request if new
            if req_id not in self.cache.active_requests:
                self.cache.start_request(req_id, seq_data.prompt_token_ids)

            # Add newly sampled tokens
            if len(sampled_token_ids[seq_data.seq_id]) > 0:
                self.cache.add_tokens(req_id, sampled_token_ids[seq_data.seq_id])

            # Prepare for speculation
            req_ids.append(req_id)
            contexts.append(seq_data.get_last_n_tokens(self.cache.max_tree_depth))

        # Single parallel batch speculation
        drafts = self.cache.batch_speculate(
            req_ids,
            contexts,
            max_spec_tokens=self.num_speculative_tokens,
            max_spec_factor=1.0,
            min_token_prob=0.1
        )

        # Map results back to sequences
        draft_proposals = {}
        for seq_data, draft in zip(input_batch, drafts):
            draft_proposals[seq_data.seq_id] = draft

        return draft_proposals
```

---

## FAQ

**Q: Should I migrate to ParallelSuffixDecodingCache?**

A: If you process multiple requests in batches, yes! You'll get near-linear speedup. For single-request processing, the original `SuffixDecodingCache` is fine.

**Q: What speedup can I expect?**

A: Near-linear with batch size up to available cores. E.g., batch of 8 on 8-core machine ≈ 7-8x faster.

**Q: Does it work without OpenMP?**

A: Yes, but falls back to sequential execution. Build will show warning if OpenMP not detected.

**Q: Can I use both caches simultaneously?**

A: Yes, they're completely independent. Useful for gradual migration or A/B testing.

**Q: Memory usage vs original?**

A: Approximately the same. Each active request has one suffix tree in both implementations.
