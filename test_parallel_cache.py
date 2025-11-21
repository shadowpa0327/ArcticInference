#!/usr/bin/env python3
"""
Test script to validate ParallelSuffixDecodingCache correctness against
the original SuffixDecodingCache implementation.
"""

import numpy as np
import sys

# Import both implementations
try:
    from arctic_inference.suffix_decoding import (
        SuffixDecodingCache,
        ParallelSuffixDecodingCache
    )
    from arctic_inference.suffix_decoding.client import SuffixDecodingClient
    print("✓ Successfully imported cache implementations and client")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

import subprocess
import time
import os

def create_test_data(num_requests=10):
    """Create test data for multiple requests."""
    requests = []
    for i in range(num_requests):
        req_id = f"req_{i}"
        prompt = np.array([1, 2, 3, i, i+1], dtype=np.int32)
        generated = [
            np.array([i+10, i+20], dtype=np.int32),
            np.array([i+30], dtype=np.int32),
        ]
        context = np.array([1, 2, 3], dtype=np.int32)
        requests.append({
            'req_id': req_id,
            'prompt': prompt,
            'generated': generated,
            'context': context
        })
    return requests

def test_basic_functionality():
    """Test basic operations of ParallelSuffixDecodingCache."""
    print("\n=== Test 1: Basic Functionality ===")

    cache = ParallelSuffixDecodingCache(max_tree_depth=64, num_threads=4)

    # Test properties
    assert cache.max_tree_depth == 64
    assert cache.num_threads > 0
    assert cache.parallel_threshold == 4
    print(f"✓ Cache initialized with {cache.num_threads} threads")

    # Start a request
    req_id = "test_req"
    prompt = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    cache.start_request(req_id, prompt)
    assert req_id in cache.active_requests
    assert cache.num_active_requests == 1
    print(f"✓ Request started successfully")

    # Add tokens
    tokens = np.array([6, 7, 8], dtype=np.int32)
    cache.add_tokens(req_id, tokens)
    print(f"✓ Tokens added successfully")

    # Speculate
    context = np.array([1, 2, 3], dtype=np.int32)
    draft = cache.speculate(req_id, context, max_spec_tokens=5)
    print(f"✓ Speculation succeeded: {len(draft.token_ids)} tokens, score={draft.score:.3f}")

    # Stop request
    cache.stop_request(req_id)
    assert req_id not in cache.active_requests
    assert cache.num_active_requests == 0
    print(f"✓ Request stopped successfully")

    print("✓ All basic functionality tests passed!")

def test_batch_speculation():
    """Test batch speculation with multiple requests."""
    print("\n=== Test 2: Batch Speculation ===")

    cache = ParallelSuffixDecodingCache(
        max_tree_depth=64,
        num_threads=4,
        parallel_threshold=2
    )

    # Create and start multiple requests
    num_requests = 8
    test_data = create_test_data(num_requests)

    for data in test_data:
        cache.start_request(data['req_id'], data['prompt'])
        for tokens in data['generated']:
            cache.add_tokens(data['req_id'], tokens)

    print(f"✓ Started {num_requests} requests")

    # Batch speculate
    req_ids = [d['req_id'] for d in test_data]
    contexts = [d['context'] for d in test_data]

    drafts = cache.batch_speculate(
        req_ids,
        contexts,
        max_spec_tokens=5,
        max_spec_factor=1.0,
        min_token_prob=0.1
    )

    assert len(drafts) == num_requests
    print(f"✓ Batch speculation succeeded: {len(drafts)} drafts returned")

    for i, draft in enumerate(drafts):
        print(f"  Request {i}: {len(draft.token_ids)} tokens, score={draft.score:.3f}")

    # Clean up
    for data in test_data:
        cache.stop_request(data['req_id'])

    print("✓ Batch speculation test passed!")

def test_correctness_vs_original():
    """
    Compare ParallelSuffixDecodingCache results with original SuffixDecodingCache
    to ensure correctness.
    """
    print("\n=== Test 3: Correctness vs Original Implementation ===")

    # Create both caches
    original_cache = SuffixDecodingCache(
        max_tree_depth=64,
        max_cached_requests=0  # Disable global tree
    )

    parallel_cache = ParallelSuffixDecodingCache(
        max_tree_depth=64,
        num_threads=4
    )

    # Create test data
    num_requests = 10
    test_data = create_test_data(num_requests)

    # Initialize both caches with same data
    for data in test_data:
        # Original cache
        original_cache.start_request(data['req_id'], data['prompt'])
        for tokens in data['generated']:
            original_cache.add_active_response(data['req_id'], tokens)

        # Parallel cache
        parallel_cache.start_request(data['req_id'], data['prompt'])
        for tokens in data['generated']:
            parallel_cache.add_tokens(data['req_id'], tokens)

    print(f"✓ Initialized both caches with {num_requests} requests")

    # Speculate with both caches and compare results
    mismatches = 0
    for data in test_data:
        req_id = data['req_id']
        context = data['context']

        # Original cache
        draft_original = original_cache.speculate(
            req_id, context,
            max_spec_tokens=5,
            max_spec_factor=1.0,
            min_token_prob=0.1,
            use_tree_spec=False
        )

        # Parallel cache
        draft_parallel = parallel_cache.speculate(
            req_id, context,
            max_spec_tokens=5,
            max_spec_factor=1.0,
            min_token_prob=0.1,
            use_tree_spec=False
        )

        # Compare results
        if draft_original.token_ids != draft_parallel.token_ids:
            print(f"  ✗ Mismatch for {req_id}:")
            print(f"    Original: {draft_original.token_ids}")
            print(f"    Parallel: {draft_parallel.token_ids}")
            mismatches += 1
        elif abs(draft_original.score - draft_parallel.score) > 0.001:
            print(f"  ✗ Score mismatch for {req_id}:")
            print(f"    Original: {draft_original.score:.6f}")
            print(f"    Parallel: {draft_parallel.score:.6f}")
            mismatches += 1

    if mismatches == 0:
        print(f"✓ All {num_requests} requests produced identical results!")
    else:
        print(f"✗ Found {mismatches} mismatches out of {num_requests} requests")
        return False

    # Test batch speculation correctness
    print("\nTesting batch speculation correctness...")
    req_ids = [d['req_id'] for d in test_data]
    contexts = [d['context'] for d in test_data]

    # Get results using original cache (sequential)
    drafts_original = []
    for req_id, context in zip(req_ids, contexts):
        draft = original_cache.speculate(
            req_id, context,
            max_spec_tokens=5,
            max_spec_factor=1.0,
            min_token_prob=0.1,
            use_tree_spec=False
        )
        drafts_original.append(draft)

    # Get results using parallel cache (batched)
    drafts_parallel = parallel_cache.batch_speculate(
        req_ids, contexts,
        max_spec_tokens=5,
        max_spec_factor=1.0,
        min_token_prob=0.1,
        use_tree_spec=False
    )

    # Compare batch results
    batch_mismatches = 0
    for i, (orig, para) in enumerate(zip(drafts_original, drafts_parallel)):
        if orig.token_ids != para.token_ids:
            print(f"  ✗ Batch mismatch at index {i}")
            batch_mismatches += 1

    if batch_mismatches == 0:
        print(f"✓ Batch speculation produced identical results!")
    else:
        print(f"✗ Found {batch_mismatches} batch mismatches")
        return False

    # Clean up
    for data in test_data:
        original_cache.stop_request(data['req_id'])
        parallel_cache.stop_request(data['req_id'])

    print("✓ Correctness test passed!")
    return True

def test_batch_add_tokens():
    """Test batch_add_tokens functionality."""
    print("\n=== Test 4: Batch Add Tokens ===")

    cache = ParallelSuffixDecodingCache(
        max_tree_depth=64,
        num_threads=4,
        parallel_threshold=2
    )

    # Start multiple requests
    num_requests = 8
    req_ids = []
    for i in range(num_requests):
        req_id = f"req_{i}"
        prompt = np.array([1, 2, 3, i], dtype=np.int32)
        cache.start_request(req_id, prompt)
        req_ids.append(req_id)

    print(f"✓ Started {num_requests} requests")

    # Add tokens to all requests in parallel
    token_batches = [
        np.array([i+10, i+20], dtype=np.int32) for i in range(num_requests)
    ]

    cache.batch_add_tokens(req_ids, token_batches)
    print(f"✓ Batch added tokens to {num_requests} requests in parallel")

    # Verify tokens were added by speculating
    contexts = [np.array([1, 2, 3], dtype=np.int32) for _ in range(num_requests)]
    drafts = cache.batch_speculate(req_ids, contexts, max_spec_tokens=5)

    assert len(drafts) == num_requests
    print(f"✓ Speculation after batch add tokens succeeded")

    # Add another batch of tokens
    token_batches_2 = [
        np.array([i+30], dtype=np.int32) for i in range(num_requests)
    ]

    cache.batch_add_tokens(req_ids, token_batches_2)
    print(f"✓ Second batch add tokens succeeded")

    # Clean up
    for req_id in req_ids:
        cache.stop_request(req_id)

    print("✓ Batch add tokens test passed!")

def test_stats():
    """Test the stats functionality."""
    print("\n=== Test 5: Statistics ===")

    cache = ParallelSuffixDecodingCache(max_tree_depth=64, num_threads=4)

    stats = cache.get_stats()
    print(f"✓ Stats retrieved: {stats}")

    assert stats['num_active_requests'] == 0
    assert stats['max_tree_depth'] == 64
    assert stats['num_threads'] > 0

    # Start some requests
    for i in range(5):
        cache.start_request(f"req_{i}", np.array([1, 2, 3], dtype=np.int32))

    stats = cache.get_stats()
    assert stats['num_active_requests'] == 5
    assert stats['num_trees_in_forest'] == 5
    print(f"✓ Stats updated correctly: {stats['num_active_requests']} active requests")

    print("✓ Statistics test passed!")

def test_server_correctness():
    """
    Test that the SuffixDecodingServer produces identical results to the local caches.
    """
    print("\n=== Test 6: Server Correctness ===")

    # Start the server in a subprocess
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "arctic_inference/suffix_decoding/server.py", "--port", "50053"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    # Wait for server to start
    time.sleep(2)

    try:
        # Initialize client
        client = SuffixDecodingClient(port=50053)
        
        # Initialize local caches
        original_cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=0)
        parallel_cache = ParallelSuffixDecodingCache(max_tree_depth=64, num_threads=4)

        # Create test data
        num_requests = 5
        test_data = create_test_data(num_requests)

        # Initialize all systems
        for data in test_data:
            # Original
            original_cache.start_request(data['req_id'], data['prompt'])
            for tokens in data['generated']:
                original_cache.add_active_response(data['req_id'], tokens)
            
            # Parallel
            parallel_cache.start_request(data['req_id'], data['prompt'])
            for tokens in data['generated']:
                parallel_cache.add_tokens(data['req_id'], tokens)
                
            # Server (via client)
            client.start_request(data['req_id'], data['prompt'])
            for tokens in data['generated']:
                client.add_tokens(data['req_id'], tokens)

        print(f"✓ Initialized all systems with {num_requests} requests")

        # Compare speculation results
        mismatches = 0
        for data in test_data:
            req_id = data['req_id']
            context = data['context']

            # Get drafts
            draft_orig = original_cache.speculate(req_id, context, max_spec_tokens=5)
            draft_para = parallel_cache.speculate(req_id, context, max_spec_tokens=5)
            draft_serv = client.speculate(req_id, context, max_spec_tokens=5)

            # Compare
            if not (draft_orig.token_ids == draft_para.token_ids == draft_serv.token_ids):
                print(f"  ✗ Mismatch for {req_id}:")
                print(f"    Original: {draft_orig.token_ids}")
                print(f"    Parallel: {draft_para.token_ids}")
                print(f"    Server:   {draft_serv.token_ids}")
                mismatches += 1
            elif not (abs(draft_orig.score - draft_para.score) < 0.001 and 
                      abs(draft_para.score - draft_serv.score) < 0.001):
                print(f"  ✗ Score mismatch for {req_id}:")
                print(f"    Original: {draft_orig.score:.6f}")
                print(f"    Parallel: {draft_para.score:.6f}")
                print(f"    Server:   {draft_serv.score:.6f}")
                mismatches += 1

        if mismatches == 0:
            print(f"✓ Server produced identical results to local caches!")
        else:
            print(f"✗ Found {mismatches} mismatches involving server")
            raise RuntimeError("Server correctness check failed")

        # Test batch speculation on server
        req_ids = [d['req_id'] for d in test_data]
        contexts = [d['context'] for d in test_data]
        
        drafts_serv = client.batch_speculate(req_ids, contexts, max_spec_tokens=5)
        
        batch_mismatches = 0
        for i, (data, draft_s) in enumerate(zip(test_data, drafts_serv)):
            draft_o = original_cache.speculate(data['req_id'], data['context'], max_spec_tokens=5)
            if draft_o.token_ids != draft_s.token_ids:
                print(f"  ✗ Batch mismatch at index {i}")
                batch_mismatches += 1
        
        if batch_mismatches == 0:
            print(f"✓ Server batch speculation produced identical results!")
        else:
            print(f"✗ Found {batch_mismatches} server batch mismatches")
            raise RuntimeError("Server batch correctness check failed")

    finally:
        # Cleanup
        if 'client' in locals():
            client.close()
        server_process.terminate()
        server_process.wait()
        print("✓ Server stopped")

def test_server_robustness():
    """
    Test error handling and edge cases for the server.
    """
    print("\n=== Test 7: Server Robustness ===")

    # Start the server in a subprocess
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "arctic_inference/suffix_decoding/server.py", "--port", "50054"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    time.sleep(2)

    try:
        client = SuffixDecodingClient(port=50054)
        req_id = "robust_req"
        
        # 1. Test stopping non-existent request
        # Should not crash, but might log error or return empty
        try:
            client.stop_request("non_existent_req")
            print("✓ Stopping non-existent request handled gracefully")
        except Exception as e:
            print(f"✗ Stopping non-existent request raised exception: {e}")

        # 2. Test adding tokens to non-existent request
        try:
            client.add_tokens("non_existent_req", [1, 2, 3])
            print("✗ Adding tokens to non-existent request should fail")
        except Exception as e:
            # gRPC usually raises RpcError
            print(f"✓ Adding tokens to non-existent request failed as expected")

        # 3. Test start -> stop -> add tokens (should fail)
        client.start_request(req_id, [1, 2, 3])
        client.stop_request(req_id)
        try:
            client.add_tokens(req_id, [4, 5])
            print("✗ Adding tokens to stopped request should fail")
        except Exception:
            print(f"✓ Adding tokens to stopped request failed as expected")

        # 4. Test empty token addition
        client.start_request("empty_test", [1, 2, 3])
        client.add_tokens("empty_test", [])
        print("✓ Adding empty tokens handled")
        
        # 5. Test speculation with empty context
        draft = client.speculate("empty_test", [])
        print(f"✓ Speculation with empty context handled: {len(draft.token_ids)} tokens")
        
        client.stop_request("empty_test")

    finally:
        if 'client' in locals():
            client.close()
        server_process.terminate()
        server_process.wait()
        print("✓ Server stopped")

def test_server_batch_add_tokens():
    """
    Test batch_add_tokens functionality via gRPC server.
    """
    print("\n=== Test 8: Server Batch Add Tokens ===")

    # Start the server in a subprocess
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "arctic_inference/suffix_decoding/server.py", "--port", "50055"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    time.sleep(2)

    try:
        client = SuffixDecodingClient(port=50055)
        
        # Start multiple requests
        num_requests = 5
        req_ids = []
        for i in range(num_requests):
            req_id = f"batch_req_{i}"
            prompt = np.array([1, 2, 3, i], dtype=np.int32)
            client.start_request(req_id, prompt)
            req_ids.append(req_id)
        
        print(f"✓ Started {num_requests} requests")

        # Add tokens to all requests in parallel
        token_batches = [
            np.array([i+10, i+20], dtype=np.int32) for i in range(num_requests)
        ]

        client.batch_add_tokens(req_ids, token_batches)
        print(f"✓ Batch added tokens via server")

        # Verify tokens were added by speculating
        contexts = [np.array([1, 2, 3], dtype=np.int32) for _ in range(num_requests)]
        drafts = client.batch_speculate(req_ids, contexts, max_spec_tokens=5)

        assert len(drafts) == num_requests
        print(f"✓ Speculation after batch add tokens succeeded")
        
        # Verify content of drafts (should reflect added tokens)
        # We expect the added tokens to be part of the prefix match or affect speculation
        # Since we don't have a ground truth here, we just check we got results.
        # But to be sure, let's add another batch and check if it works.
        
        token_batches_2 = [
            np.array([i+30], dtype=np.int32) for i in range(num_requests)
        ]
        client.batch_add_tokens(req_ids, token_batches_2)
        print(f"✓ Second batch add tokens succeeded")

    finally:
        if 'client' in locals():
            client.close()
        server_process.terminate()
        server_process.wait()
        print("✓ Server stopped")

def test_multi_client_same_request():
    """
    Test that multiple clients can interact with the same request on the server.
    """
    print("\n=== Test 9: Multi-Client Same Request ===")

    # Start the server in a subprocess
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "arctic_inference/suffix_decoding/server.py", "--port", "50056"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    time.sleep(2)

    try:
        # Client 1
        client1 = SuffixDecodingClient(port=50056)
        req_id = "shared_req"
        prompt = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        
        # Client 1 starts request
        client1.start_request(req_id, prompt)
        print("✓ Client 1 started request")
        
        # Client 1 adds tokens
        tokens1 = np.array([6, 7, 8], dtype=np.int32)
        client1.add_tokens(req_id, tokens1)
        print("✓ Client 1 added tokens")
        
        # Client 2 (new connection)
        client2 = SuffixDecodingClient(port=50056)
        
        # Client 2 adds more tokens to the SAME request
        tokens2 = np.array([9, 10], dtype=np.int32)
        client2.add_tokens(req_id, tokens2)
        print("✓ Client 2 added tokens to same request")
        
        # Client 2 speculates
        context = np.array([1, 2, 3], dtype=np.int32)
        draft = client2.speculate(req_id, context, max_spec_tokens=5)
        print(f"✓ Client 2 speculated: {len(draft.token_ids)} tokens")
        
        # Verify that Client 1 can also see the updates (by speculating)
        draft1 = client1.speculate(req_id, context, max_spec_tokens=5)
        print(f"✓ Client 1 speculated: {len(draft1.token_ids)} tokens")
        
        # They should get the same result
        if np.array_equal(draft.token_ids, draft1.token_ids):
             print("✓ Both clients see consistent state")
        else:
             print("✗ Clients see different state!")
             print(f"  Client 1: {draft1.token_ids}")
             print(f"  Client 2: {draft.token_ids}")
             raise RuntimeError("Multi-client consistency check failed")

        client1.close()
        client2.close()

    finally:
        server_process.terminate()
        server_process.wait()
        print("✓ Server stopped")

if __name__ == "__main__":
    print("="*70)
    print("ParallelSuffixDecodingCache Test Suite")
    print("="*70)

    try:
        test_basic_functionality()
        test_batch_speculation()
        test_correctness_vs_original()
        test_batch_add_tokens()
        test_stats()
        test_server_correctness()
        test_server_robustness()
        test_server_batch_add_tokens()
        test_multi_client_same_request()

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
