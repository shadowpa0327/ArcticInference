# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Hashable, KeysView, List, Optional, Sequence

import numpy as np

from arctic_inference.suffix_decoding._C import SuffixTree, SuffixForest, Draft
from arctic_inference.suffix_decoding.cache import SuffixDecodingDraft


class ParallelSuffixDecodingCache:
    """
    Parallel implementation of SuffixDecodingCache using SuffixForest for
    batched parallel speculation across multiple requests.

    This class is designed for high-throughput scenarios where multiple requests
    are processed simultaneously. It uses C++ SuffixForest with OpenMP for
    efficient parallel speculation.

    Key differences from SuffixDecodingCache:
    - No global tree (removed as per design plan)
    - Batch speculation with parallel execution
    - Configurable parallelization parameters
    - Simpler API focused on batched operations
    """

    def __init__(self,
                 max_tree_depth: int = 64,
                 num_threads: int = -1,
                 parallel_threshold: int = 4):
        """
        Initialize the ParallelSuffixDecodingCache.

        Args:
            max_tree_depth (int): The maximum depth of the suffix trees.
                Determines how much context history is retained.
            num_threads (int): Number of threads for parallel speculation.
                -1 = auto-detect from CPU count (default)
                0 = sequential execution (no parallelization)
                >0 = use specified number of threads
            parallel_threshold (int): Minimum batch size to trigger parallelization.
                Batches smaller than this will run sequentially to avoid overhead.
                Default: 4.
        """
        self._max_tree_depth = max_tree_depth
        self._num_threads = num_threads
        self._parallel_threshold = parallel_threshold

        # C++ SuffixForest for parallel batched speculation
        self._forest = SuffixForest(max_tree_depth, num_threads, parallel_threshold)

        # Maps request ID to tree index in the forest
        self._req_to_tree_idx = {}

    @property
    def max_tree_depth(self) -> int:
        """Maximum depth of suffix trees."""
        return self._max_tree_depth

    @property
    def num_threads(self) -> int:
        """Number of threads configured for parallel execution."""
        return self._forest.get_num_threads()

    @property
    def parallel_threshold(self) -> int:
        """Minimum batch size to trigger parallelization."""
        return self._forest.get_parallel_threshold()

    @property
    def active_requests(self) -> KeysView:
        """
        Returns a view of the currently active request IDs.

        Active requests are those that have been started via `start_request`
        and not yet stopped via `stop_request`.
        """
        return self._req_to_tree_idx.keys()

    @property
    def num_active_requests(self) -> int:
        """Number of currently active requests."""
        return len(self._req_to_tree_idx)

    def start_request(
        self,
        req_id: Hashable,
        prompt_token_ids: np.ndarray | Sequence[int],
    ):
        """
        Start processing a new request by creating its suffix tree and
        inserting the prompt tokens.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (np.ndarray | Sequence[int]): A sequence of token
                IDs representing the prompt of the request.

        Raises:
            ValueError: If a request with the same `req_id` is already active.
        """
        if req_id in self._req_to_tree_idx:
            raise ValueError(f"Request '{req_id}' is already active")

        # Determine which extend function to use
        if isinstance(prompt_token_ids, np.ndarray):
            self._validate_ndarray(prompt_token_ids)
            extend_func = SuffixTree.extend_ndarray
        else:
            extend_func = SuffixTree.extend

        # Create a new tree in the forest and add the prompt
        tree_idx = self._forest.create_tree()
        self._req_to_tree_idx[req_id] = tree_idx
        tree = self._forest.get_tree(tree_idx)
        extend_func(tree, 0, prompt_token_ids)

    def stop_request(self, req_id: Hashable):
        """
        Stop processing a request and free its suffix tree.

        Args:
            req_id (Hashable): The request identifier.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._req_to_tree_idx:
            raise ValueError(f"Request '{req_id}' is not active")

        tree_idx = self._req_to_tree_idx.pop(req_id)
        self._forest.remove_tree(tree_idx)

    def add_tokens(
        self,
        req_id: Hashable,
        token_ids: np.ndarray | Sequence[int],
    ):
        """
        Add generated tokens to a request's suffix tree.

        This updates the tree with newly generated tokens, allowing them to be
        used for future speculations.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (np.ndarray | Sequence[int]): A sequence of token IDs to
                be appended to the tree.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._req_to_tree_idx:
            raise ValueError(f"Request '{req_id}' is not active")

        # Determine which extend function to use
        if isinstance(token_ids, np.ndarray):
            self._validate_ndarray(token_ids)
            extend_func = SuffixTree.extend_ndarray
        else:
            extend_func = SuffixTree.extend

        tree_idx = self._req_to_tree_idx[req_id]
        tree = self._forest.get_tree(tree_idx)
        extend_func(tree, 0, token_ids)

    def batch_add_tokens(
        self,
        req_ids: List[Hashable],
        token_batches: List[np.ndarray | Sequence[int]],
    ):
        """
        Add generated tokens to multiple requests in parallel.

        This is more efficient than calling add_tokens() in a loop when you
        have multiple requests with new tokens. Perfect for batch inference
        where all requests get new tokens simultaneously.

        Args:
            req_ids (List[Hashable]): List of request identifiers.
            token_batches (List[np.ndarray | Sequence[int]]): List of token
                sequences, one per request.

        Raises:
            ValueError: If req_ids and token_batches have different lengths,
                or if any request is not active.
            RuntimeError: If adding tokens fails for any tree.
        """
        if len(req_ids) != len(token_batches):
            raise ValueError(
                f"req_ids and token_batches must have same length. "
                f"Got {len(req_ids)} and {len(token_batches)}"
            )

        if len(req_ids) == 0:
            return

        # Validate all requests are active and get tree indices
        tree_indices = []
        for req_id in req_ids:
            if req_id not in self._req_to_tree_idx:
                raise ValueError(f"Request '{req_id}' is not active")
            tree_indices.append(self._req_to_tree_idx[req_id])

        # seq_id is always 0 for single sequence per tree
        seq_ids = [0] * len(req_ids)

        # Prepare token batches for C++
        tokens_for_cpp = []
        use_ndarray = False

        for tokens in token_batches:
            if isinstance(tokens, np.ndarray):
                self._validate_ndarray(tokens)
                tokens_for_cpp.append(tokens)
                use_ndarray = True
            else:
                # Convert to list for C++
                tokens_for_cpp.append(
                    list(tokens) if not isinstance(tokens, list) else tokens
                )

        # Call C++ batch extend (releases GIL internally)
        if use_ndarray:
            # Use ndarray-optimized version
            tree_indices_np = np.array(tree_indices, dtype=np.int32)
            seq_ids_np = np.array(seq_ids, dtype=np.int32)
            self._forest.batch_extend_ndarray(
                tree_indices_np,
                seq_ids_np,
                tokens_for_cpp
            )
        else:
            # Use standard vector version
            self._forest.batch_extend(
                tree_indices,
                seq_ids,
                tokens_for_cpp
            )

    def batch_speculate(
        self,
        req_ids: List[Hashable],
        contexts: List[np.ndarray | Sequence[int]],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> List[SuffixDecodingDraft]:
        """
        Perform batched parallel speculation across multiple requests.

        This is the main method for speculation. It processes all requests in
        parallel (if batch size ≥ parallel_threshold) using OpenMP.

        Args:
            req_ids (List[Hashable]): List of request identifiers.
            contexts (List[np.ndarray | Sequence[int]]): List of context sequences,
                one per request. Each context is the most recent tokens to match
                against the suffix tree.
            max_spec_tokens (int, optional): Maximum number of tokens to speculate
                per request. If None, uses max_tree_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched context length. The number of speculated tokens is
                limited by `max_spec_factor * match_length + max_spec_offset`.
            max_spec_offset (float): Offset for speculation limit.
            min_token_prob (float): Minimum estimated probability threshold for
                draft tokens. Tokens with lower probability are not included.
            use_tree_spec (bool): If True, uses tree-based speculation (explores
                multiple paths). If False, uses path-based speculation (greedy).

        Returns:
            List of SuffixDecodingDraft objects, one per request.

        Raises:
            ValueError: If req_ids and contexts have different lengths, or if
                any request is not active.
            RuntimeError: If speculation fails for any tree in the batch.
        """
        if len(req_ids) != len(contexts):
            raise ValueError(
                f"req_ids and contexts must have same length. "
                f"Got {len(req_ids)} and {len(contexts)}"
            )

        if max_spec_tokens is None:
            max_spec_tokens = self._max_tree_depth

        # Validate all requests are active and get tree indices
        tree_indices = []
        for req_id in req_ids:
            if req_id not in self._req_to_tree_idx:
                raise ValueError(f"Request '{req_id}' is not active")
            tree_indices.append(self._req_to_tree_idx[req_id])

        # Truncate contexts to max_tree_depth and prepare for C++
        truncated_contexts = []
        use_ndarray = False

        for ctx in contexts:
            # Truncate if necessary
            if len(ctx) > self._max_tree_depth:
                ctx = ctx[-self._max_tree_depth:]

            # Validate and add to list
            if isinstance(ctx, np.ndarray):
                self._validate_ndarray(ctx)
                truncated_contexts.append(ctx)
                use_ndarray = True
            else:
                # Convert to list for C++
                truncated_contexts.append(
                    list(ctx) if not isinstance(ctx, list) else ctx
                )

        # Call C++ batch speculation (releases GIL internally)
        if use_ndarray:
            # Use ndarray-optimized version
            tree_indices_np = np.array(tree_indices, dtype=np.int32)
            drafts = self._forest.batch_speculate_ndarray(
                tree_indices_np,
                truncated_contexts,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec
            )
        else:
            # Use standard vector version
            drafts = self._forest.batch_speculate(
                tree_indices,
                truncated_contexts,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec
            )

        # Convert C++ Draft objects to Python SuffixDecodingDraft objects
        return [SuffixDecodingDraft.from_native(d) for d in drafts]

    def speculate(
        self,
        req_id: Hashable,
        context: np.ndarray | Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> SuffixDecodingDraft:
        """
        Speculate for a single request.

        This is a convenience method that delegates to batch_speculate with a
        single request. For better performance with multiple requests, use
        batch_speculate directly.

        Args:
            req_id (Hashable): The unique identifier for the request.
            context (np.ndarray | Sequence[int]): Context sequence to match.
            max_spec_tokens (int, optional): Maximum tokens to speculate.
            max_spec_factor (float): Speculation length factor.
            max_spec_offset (float): Speculation length offset.
            min_token_prob (float): Minimum probability threshold.
            use_tree_spec (bool): Use tree-based (vs path-based) speculation.

        Returns:
            SuffixDecodingDraft: The speculation result.

        Raises:
            ValueError: If the request is not active.
        """
        results = self.batch_speculate(
            req_ids=[req_id],
            contexts=[context],
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec
        )
        return results[0]

    def get_stats(self) -> dict:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics including number of active requests,
            configured threads, and parallel threshold.
        """
        return {
            "num_active_requests": self.num_active_requests,
            "max_tree_depth": self._max_tree_depth,
            "num_threads": self.num_threads,
            "parallel_threshold": self.parallel_threshold,
            "num_trees_in_forest": self._forest.num_trees(),
        }

    def _validate_ndarray(self, arr: np.ndarray):
        """Validate that a numpy array has the correct shape and dtype."""
        if arr.ndim != 1:
            raise ValueError(
                f"ndarray input must have ndim=1, got ndim={arr.ndim}"
            )
        if arr.dtype != np.int32:
            raise ValueError(
                f"ndarray input must have dtype=int32, got dtype={arr.dtype.name}"
            )
        if not arr.flags["CONTIGUOUS"]:
            raise ValueError("ndarray input must be contiguous")
