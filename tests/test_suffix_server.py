import unittest
import time
import subprocess
import sys
import os
import numpy as np
import grpc
from arctic_inference.suffix_decoding.client import SuffixDecodingClient

class TestSuffixDecodingServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start the server in a subprocess
        cls.server_process = subprocess.Popen(
            [sys.executable, "arctic_inference/suffix_decoding/server.py", "--port", "50052"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        # Wait for server to start
        time.sleep(2)
        
        # Initialize client
        cls.client = SuffixDecodingClient(port=50052)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server_process.terminate()
        cls.server_process.wait()

    def test_basic_flow(self):
        req_id = "test_req_1"
        prompt = [1, 2, 3, 4, 5]
        
        # Start request
        self.client.start_request(req_id, prompt)
        
        # Check stats
        stats = self.client.get_stats()
        self.assertEqual(stats['num_active_requests'], 1)
        
        # Add tokens
        self.client.add_tokens(req_id, [6, 7, 8])
        
        # Speculate
        context = [1, 2, 3]
        draft = self.client.speculate(req_id, context, max_spec_tokens=5)
        self.assertIsNotNone(draft)
        self.assertIsInstance(draft.token_ids, list)
        
        # Stop request
        self.client.stop_request(req_id)
        
        # Check stats
        stats = self.client.get_stats()
        self.assertEqual(stats['num_active_requests'], 0)

    def test_batch_flow(self):
        req_ids = ["batch_req_1", "batch_req_2"]
        prompts = [[1, 2, 3], [4, 5, 6]]
        
        for req_id, prompt in zip(req_ids, prompts):
            self.client.start_request(req_id, prompt)
            
        # Batch add tokens
        token_batches = [[10, 11], [20, 21]]
        self.client.batch_add_tokens(req_ids, token_batches)
        
        # Batch speculate
        contexts = [[1, 2], [4, 5]]
        drafts = self.client.batch_speculate(req_ids, contexts, max_spec_tokens=5)
        
        self.assertEqual(len(drafts), 2)
        for draft in drafts:
            self.assertIsInstance(draft.token_ids, list)
            
        # Cleanup
        for req_id in req_ids:
            self.client.stop_request(req_id)

if __name__ == '__main__':
    unittest.main()
