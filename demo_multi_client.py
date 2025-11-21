import time
import threading
import random
import sys
import os
import subprocess
from arctic_inference.suffix_decoding.client import SuffixDecodingClient

def run_client(client_id, num_requests, port):
    """Simulates a client sending multiple requests."""
    client = SuffixDecodingClient(host='localhost', port=port)
    print(f"[Client {client_id}] Connected")

    for i in range(num_requests):
        # Ensure unique req_id across clients
        req_id = f"client_{client_id}_req_{i}"
        prompt = [1, 2, 3, client_id, i]
        
        # Track the full sequence for verification
        full_sequence = list(prompt)
        
        try:
            # 1. Start
            client.start_request(req_id, prompt)
            
            # 2. Add tokens (simulate generation)
            # Use deterministic tokens for verification
            for step in range(3):
                time.sleep(random.uniform(0.01, 0.05)) # Simulate work
                # Deterministic token: client_id * 1000 + i * 100 + step
                new_token = client_id * 1000 + i * 100 + step
                client.add_tokens(req_id, [new_token])
                full_sequence.append(new_token)
            
            # 3. Speculate
            # Use the prompt as context, should return all added tokens
            draft = client.speculate(req_id, prompt)
            
            # Verification
            expected_tokens = full_sequence[len(prompt):]
            if draft.token_ids != expected_tokens:
                print(f"[Client {client_id}] VERIFICATION FAILED for {req_id}")
                print(f"  Expected: {expected_tokens}")
                print(f"  Got:      {draft.token_ids}")
            else:
                # print(f"[Client {client_id}] Verified {req_id}")
                pass

            # 4. Stop
            client.stop_request(req_id)
            
        except Exception as e:
            print(f"[Client {client_id}] Error: {e}")
            
    print(f"[Client {client_id}] Finished {num_requests} requests")
    client.close()

def main():
    port = 50054
    # Start Server
    print("Starting Server...")
    server_process = subprocess.Popen(
        [sys.executable, "arctic_inference/suffix_decoding/server.py", "--port", str(port), "--max-workers", "10"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    time.sleep(2) # Wait for startup

    try:
        # Start multiple clients in threads
        num_clients = 5
        requests_per_client = 10
        threads = []
        
        print(f"Starting {num_clients} clients, each sending {requests_per_client} requests...")
        start_time = time.time()
        
        for i in range(num_clients):
            t = threading.Thread(target=run_client, args=(i, requests_per_client, port))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        duration = time.time() - start_time
        print(f"All clients finished in {duration:.2f}s")
        
        # Verify stats
        admin_client = SuffixDecodingClient(port=port)
        stats = admin_client.get_stats()
        print(f"Final Server Stats: {stats}")
        # Should be 0 active requests if all stopped correctly
        if stats['num_active_requests'] == 0:
            print("SUCCESS: Server state clean.")
        else:
            print(f"WARNING: {stats['num_active_requests']} active requests remaining.")
        admin_client.close()

    finally:
        server_process.terminate()
        server_process.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()
