// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "suffix_forest.h"

#include <thread>
#include <stdexcept>
#include <sstream>
#include <iostream>

// OpenMP support (optional, will be enabled via CMake if available)
#ifdef _OPENMP
#include <omp.h>
#endif

SuffixForest::SuffixForest(int max_depth, int num_threads, int parallel_threshold)
    : _max_depth(max_depth),
      _num_threads(_resolve_num_threads(num_threads)),
      _parallel_threshold(parallel_threshold) {

    if (max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    if (parallel_threshold < 0) {
        throw std::invalid_argument("parallel_threshold must be non-negative");
    }
}

int SuffixForest::_resolve_num_threads(int num_threads) {
    if (num_threads < -1) {
        throw std::invalid_argument("num_threads must be >= -1");
    }

    if (num_threads == -1) {
        // Auto-detect from hardware concurrency
        unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) {
            // Fallback if hardware_concurrency fails
            return 4;
        }
        return static_cast<int>(hw_threads);
    }

    return num_threads;
}

int SuffixForest::create_tree() {
    int tree_index = _next_tree_id++;
    _trees[tree_index] = std::make_unique<SuffixTree>(_max_depth);
    return tree_index;
}

void SuffixForest::remove_tree(int tree_index) {
    _validate_tree_index(tree_index);
    _trees.erase(tree_index);
}

SuffixTree& SuffixForest::get_tree(int tree_index) {
    _validate_tree_index(tree_index);
    return *_trees[tree_index];
}

const SuffixTree& SuffixForest::get_tree(int tree_index) const {
    _validate_tree_index(tree_index);
    auto it = _trees.find(tree_index);
    return *it->second;
}

bool SuffixForest::has_tree(int tree_index) const {
    return _trees.find(tree_index) != _trees.end();
}

void SuffixForest::_validate_tree_index(int tree_index) const {
    if (!has_tree(tree_index)) {
        std::ostringstream oss;
        oss << "Tree index " << tree_index << " does not exist in forest";
        throw std::invalid_argument(oss.str());
    }
}

void SuffixForest::batch_extend(
    std::span<const int> tree_indices,
    std::span<const int> seq_ids,
    const std::vector<std::vector<int32_t>>& token_batches) {

    // Validate inputs
    if (tree_indices.size() != seq_ids.size() || tree_indices.size() != token_batches.size()) {
        std::ostringstream oss;
        oss << "tree_indices, seq_ids, and token_batches must have the same size. Got "
            << tree_indices.size() << ", " << seq_ids.size() << ", and " << token_batches.size();
        throw std::invalid_argument(oss.str());
    }

    const size_t batch_size = tree_indices.size();

    // Early return for empty batch
    if (batch_size == 0) {
        return;
    }

    // Validate all tree indices exist before starting
    for (size_t i = 0; i < batch_size; ++i) {
        _validate_tree_index(tree_indices[i]);
    }

    // Determine whether to use parallel or sequential execution
    const bool use_parallel = (batch_size >= static_cast<size_t>(_parallel_threshold)) &&
                             (_num_threads > 0);

    if (use_parallel) {
#ifdef _OPENMP
        // OpenMP parallel execution
        // Flag to track if any thread encountered an error
        bool error_occurred = false;
        std::string error_message;

        // Use num_threads clause instead of omp_set_num_threads() to avoid overhead
        // Use static scheduling for better performance with uniform work
        #pragma omp parallel for schedule(static) num_threads(_num_threads)
        for (size_t i = 0; i < batch_size; ++i) {
            try {
                int tree_idx = tree_indices[i];
                int seq_id = seq_ids[i];
                SuffixTree& tree = *_trees[tree_idx];
                tree.extend(seq_id, token_batches[i]);
            } catch (const std::exception& e) {
                // Fail-fast: mark error and save message
                #pragma omp critical
                {
                    if (!error_occurred) {
                        error_occurred = true;
                        std::ostringstream oss;
                        oss << "Extend failed for tree " << tree_indices[i]
                            << " at batch index " << i << ": " << e.what();
                        error_message = oss.str();
                    }
                }
            }
        }

        // Check if any errors occurred
        if (error_occurred) {
            throw std::runtime_error(error_message);
        }
#else
        // OpenMP not available, fall back to sequential
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            int seq_id = seq_ids[i];
            SuffixTree& tree = *_trees[tree_idx];

            try {
                tree.extend(seq_id, token_batches[i]);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Extend failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
#endif
    } else {
        // Sequential execution (below threshold or num_threads == 0)
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            int seq_id = seq_ids[i];
            SuffixTree& tree = *_trees[tree_idx];

            try {
                tree.extend(seq_id, token_batches[i]);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Extend failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }
}

std::vector<Draft> SuffixForest::batch_speculate(
    std::span<const int> tree_indices,
    const std::vector<std::vector<int32_t>>& contexts,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Validate inputs
    if (tree_indices.size() != contexts.size()) {
        std::ostringstream oss;
        oss << "tree_indices and contexts must have the same size. Got "
            << tree_indices.size() << " and " << contexts.size();
        throw std::invalid_argument(oss.str());
    }

    const size_t batch_size = tree_indices.size();

    // Early return for empty batch
    if (batch_size == 0) {
        return std::vector<Draft>();
    }

    // Validate all tree indices exist before starting speculation
    for (size_t i = 0; i < batch_size; ++i) {
        _validate_tree_index(tree_indices[i]);
    }

    // Allocate results vector
    std::vector<Draft> results(batch_size);

    // Determine whether to use parallel or sequential execution
    const bool use_parallel = (batch_size >= static_cast<size_t>(_parallel_threshold)) &&
                             (_num_threads > 0);

    if (use_parallel) {
#ifdef _OPENMP
        // OpenMP parallel execution
        // Flag to track if any thread encountered an error
        bool error_occurred = false;
        std::string error_message;

        // Use num_threads clause instead of omp_set_num_threads() to avoid overhead
        // Use static scheduling for better performance with uniform work
        #pragma omp parallel for schedule(static) num_threads(_num_threads)
        for (size_t i = 0; i < batch_size; ++i) {
            try {
                int tree_idx = tree_indices[i];
                SuffixTree& tree = *_trees[tree_idx];

                results[i] = tree.speculate(
                    contexts[i],
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                // Fail-fast: mark error and save message
                #pragma omp critical
                {
                    if (!error_occurred) {
                        error_occurred = true;
                        std::ostringstream oss;
                        oss << "Speculation failed for tree " << tree_indices[i]
                            << " at batch index " << i << ": " << e.what();
                        error_message = oss.str();
                    }
                }
            }
        }

        // Check if any errors occurred
        if (error_occurred) {
            throw std::runtime_error(error_message);
        }
#else
        // OpenMP not available, fall back to sequential
        // (This branch should rarely be hit if CMake is configured correctly)
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            SuffixTree& tree = *_trees[tree_idx];

            try {
                results[i] = tree.speculate(
                    contexts[i],
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Speculation failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
#endif
    } else {
        // Sequential execution (below threshold or num_threads == 0)
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            SuffixTree& tree = *_trees[tree_idx];

            try {
                results[i] = tree.speculate(
                    contexts[i],
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Speculation failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }

    return results;
}
