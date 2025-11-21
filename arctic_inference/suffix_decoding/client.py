import grpc
import numpy as np
from typing import Hashable, List, Optional, Sequence, Union

from arctic_inference.suffix_decoding.cache import SuffixDecodingDraft
from arctic_inference.suffix_decoding.proto import suffix_decoding_pb2
from arctic_inference.suffix_decoding.proto import suffix_decoding_pb2_grpc

class SuffixDecodingClient:
    """
    Client for SuffixDecodingService.
    Exposes an API similar to ParallelSuffixDecodingCache.
    """
    def __init__(self, host: str = 'localhost', port: int = 50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = suffix_decoding_pb2_grpc.SuffixDecodingServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def start_request(
        self,
        req_id: Hashable,
        prompt_token_ids: Union[np.ndarray, Sequence[int]],
    ):
        if isinstance(prompt_token_ids, np.ndarray):
            prompt_token_ids = prompt_token_ids.tolist()
        
        request = suffix_decoding_pb2.StartRequestMsg(
            req_id=str(req_id),
            prompt_token_ids=prompt_token_ids
        )
        self.stub.StartRequest(request)

    def stop_request(self, req_id: Hashable):
        request = suffix_decoding_pb2.StopRequestMsg(req_id=str(req_id))
        self.stub.StopRequest(request)

    def add_tokens(
        self,
        req_id: Hashable,
        token_ids: Union[np.ndarray, Sequence[int]],
    ):
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
            
        request = suffix_decoding_pb2.AddTokensMsg(
            req_id=str(req_id),
            token_ids=token_ids
        )
        self.stub.AddTokens(request)

    def batch_add_tokens(
        self,
        req_ids: List[Hashable],
        token_batches: List[Union[np.ndarray, Sequence[int]]],
    ):
        proto_token_batches = []
        for tokens in token_batches:
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            proto_token_batches.append(suffix_decoding_pb2.TokenSequence(tokens=tokens))
            
        request = suffix_decoding_pb2.BatchAddTokensMsg(
            req_ids=[str(rid) for rid in req_ids],
            token_batches=proto_token_batches
        )
        self.stub.BatchAddTokens(request)

    def speculate(
        self,
        req_id: Hashable,
        context: Union[np.ndarray, Sequence[int]],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> SuffixDecodingDraft:
        if isinstance(context, np.ndarray):
            context = context.tolist()
            
        request = suffix_decoding_pb2.SpeculateMsg(
            req_id=str(req_id),
            context=context,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec
        )
        
        response = self.stub.Speculate(request)
        
        return SuffixDecodingDraft(
            token_ids=list(response.token_ids),
            parents=list(response.parents),
            probs=list(response.probs),
            score=response.score,
            match_len=response.match_len
        )

    def batch_speculate(
        self,
        req_ids: List[Hashable],
        contexts: List[Union[np.ndarray, Sequence[int]]],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> List[SuffixDecodingDraft]:
        proto_contexts = []
        for ctx in contexts:
            if isinstance(ctx, np.ndarray):
                ctx = ctx.tolist()
            proto_contexts.append(suffix_decoding_pb2.TokenSequence(tokens=ctx))
            
        request = suffix_decoding_pb2.BatchSpeculateMsg(
            req_ids=[str(rid) for rid in req_ids],
            contexts=proto_contexts,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec
        )
        
        response = self.stub.BatchSpeculate(request)
        
        drafts = []
        for draft_proto in response.drafts:
            drafts.append(SuffixDecodingDraft(
                token_ids=list(draft_proto.token_ids),
                parents=list(draft_proto.parents),
                probs=list(draft_proto.probs),
                score=draft_proto.score,
                match_len=draft_proto.match_len
            ))
        return drafts

    def get_stats(self) -> dict:
        response = self.stub.GetStats(suffix_decoding_pb2.Empty())
        return {
            "num_active_requests": response.num_active_requests,
            "max_tree_depth": response.max_tree_depth,
            "num_threads": response.num_threads,
            "parallel_threshold": response.parallel_threshold,
            "num_trees_in_forest": response.num_trees_in_forest,
        }
