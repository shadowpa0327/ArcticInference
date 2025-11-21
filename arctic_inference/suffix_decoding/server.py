import logging
import asyncio
import grpc
from concurrent import futures
from typing import Optional
import numpy as np

from arctic_inference.suffix_decoding.parallel_cache import ParallelSuffixDecodingCache
from arctic_inference.suffix_decoding.proto import suffix_decoding_pb2
from arctic_inference.suffix_decoding.proto import suffix_decoding_pb2_grpc

logger = logging.getLogger(__name__)

class SuffixDecodingServicer(suffix_decoding_pb2_grpc.SuffixDecodingServiceServicer):
    def __init__(self, max_tree_depth: int = 64, num_threads: int = -1, parallel_threshold: int = 4):
        self.cache = ParallelSuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            num_threads=num_threads,
            parallel_threshold=parallel_threshold
        )
        logger.info(f"Initialized SuffixDecodingServicer with {self.cache.num_threads} threads")

    def StartRequest(self, request, context):
        try:
            self.cache.start_request(request.req_id, list(request.prompt_token_ids))
            return suffix_decoding_pb2.Empty()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.Empty()

    def StopRequest(self, request, context):
        try:
            self.cache.stop_request(request.req_id)
            return suffix_decoding_pb2.Empty()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.Empty()

    def AddTokens(self, request, context):
        try:
            self.cache.add_tokens(request.req_id, list(request.token_ids))
            return suffix_decoding_pb2.Empty()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.Empty()

    def BatchAddTokens(self, request, context):
        try:
            req_ids = list(request.req_ids)
            token_batches = [list(ts.tokens) for ts in request.token_batches]
            self.cache.batch_add_tokens(req_ids, token_batches)
            return suffix_decoding_pb2.Empty()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.Empty()

    def Speculate(self, request, context):
        try:
            draft = self.cache.speculate(
                request.req_id,
                list(request.context),
                max_spec_tokens=request.max_spec_tokens if request.HasField('max_spec_tokens') else None,
                max_spec_factor=request.max_spec_factor,
                max_spec_offset=request.max_spec_offset,
                min_token_prob=request.min_token_prob,
                use_tree_spec=request.use_tree_spec
            )
            return suffix_decoding_pb2.SpeculateResponse(
                token_ids=draft.token_ids,
                parents=draft.parents,
                probs=draft.probs,
                score=draft.score,
                match_len=draft.match_len
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.SpeculateResponse()

    def BatchSpeculate(self, request, context):
        try:
            req_ids = list(request.req_ids)
            contexts = [list(ts.tokens) for ts in request.contexts]
            drafts = self.cache.batch_speculate(
                req_ids,
                contexts,
                max_spec_tokens=request.max_spec_tokens if request.HasField('max_spec_tokens') else None,
                max_spec_factor=request.max_spec_factor,
                max_spec_offset=request.max_spec_offset,
                min_token_prob=request.min_token_prob,
                use_tree_spec=request.use_tree_spec
            )
            
            response_drafts = []
            for draft in drafts:
                response_drafts.append(suffix_decoding_pb2.SpeculateResponse(
                    token_ids=draft.token_ids,
                    parents=draft.parents,
                    probs=draft.probs,
                    score=draft.score,
                    match_len=draft.match_len
                ))
            
            return suffix_decoding_pb2.BatchSpeculateResponse(drafts=response_drafts)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.BatchSpeculateResponse()

    def GetStats(self, request, context):
        try:
            stats = self.cache.get_stats()
            return suffix_decoding_pb2.StatsResponse(
                num_active_requests=stats['num_active_requests'],
                max_tree_depth=stats['max_tree_depth'],
                num_threads=stats['num_threads'],
                parallel_threshold=stats['parallel_threshold'],
                num_trees_in_forest=stats['num_trees_in_forest']
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.StatsResponse()

def serve(port: int = 50051, max_workers: int = 10, max_tree_depth: int = 64, num_threads: int = -1, parallel_threshold: int = 4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    suffix_decoding_pb2_grpc.add_SuffixDecodingServiceServicer_to_server(
        SuffixDecodingServicer(max_tree_depth, num_threads, parallel_threshold), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f"Server started on port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--max-workers', type=int, default=10)
    parser.add_argument('--max-tree-depth', type=int, default=64)
    parser.add_argument('--num-threads', type=int, default=-1)
    parser.add_argument('--parallel-threshold', type=int, default=4)
    args = parser.parse_args()
    
    serve(args.port, args.max_workers, args.max_tree_depth, args.num_threads, args.parallel_threshold)
