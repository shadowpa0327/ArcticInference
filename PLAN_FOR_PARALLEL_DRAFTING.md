## Context

Currently at `arctic_inference/suffix_decoding/cache.py` it have a implementation of `SuffixDecodingCache`, which maintain a sets of SuffixTree for storing the information of the generated tokens. Inside the cache, it have
+ `self._global_tree`: shared for all requests
+ `self._local_trees`: one tree for each prompt(request) specifically. 

You can refer to `suffix_proposer.py` on how the vLLM leverage this `SuffixDecodingCache` in getting draft tokens for each requests.

## Plan
1. No need global tree anymore
2. Based on 1. I am considering implementing the `SuffixForest` (i.e., self.local_trees) in C++ and support parallel batched speculation. We can assuming that the `max_spec_tokens` and `max_spec_factor` this kind of hypermeter universally align for all requests. 

At `csrc/suffix_decoding` we have the implementation for suffixtree already. So I think what we need is actually only another wrapper to form a forest and support parallelism drafting as every suffix trees are data-independent and is nativelly parallizable. 
