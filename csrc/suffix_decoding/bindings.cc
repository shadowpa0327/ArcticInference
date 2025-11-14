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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "suffix_tree.h"
#include "suffix_forest.h"

namespace nb = nanobind;

using Int32Array1D = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>,
                                 nb::device::cpu, nb::any_contig>;


void extend_ndarray(SuffixTree& tree,
                    int seq_id,
                    const Int32Array1D& tokens) {
    tree.extend(
        seq_id,
        std::span<const int32_t>(tokens.data(), tokens.size()));
}


void extend_vector(SuffixTree& tree,
                   int seq_id,
                   const std::vector<int32_t>& tokens) {
    tree.extend(seq_id, std::span<const int32_t>(tokens));
}


Draft speculate_ndarray(SuffixTree& tree,
                        const Int32Array1D& context,
                        int max_spec_tokens,
                        float max_spec_factor,
                        float max_spec_offset,
                        float min_token_prob,
                        bool use_tree_spec) {
    return tree.speculate(
        std::span<const int32_t>(context.data(), context.size()),
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


Draft speculate_vector(SuffixTree& tree,
                       const std::vector<int32_t>& context,
                       int max_spec_tokens,
                       float max_spec_factor,
                       float max_spec_offset,
                       float min_token_prob,
                       bool use_tree_spec) {
    return tree.speculate(
        std::span<const int32_t>(context),
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


// SuffixForest wrapper functions

void batch_extend_forest(
    SuffixForest& forest,
    const std::vector<int>& tree_indices,
    const std::vector<int>& seq_ids,
    const std::vector<std::vector<int32_t>>& token_batches) {

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    forest.batch_extend(
        std::span<const int>(tree_indices),
        std::span<const int>(seq_ids),
        token_batches);
}

void batch_extend_forest_ndarray(
    SuffixForest& forest,
    const Int32Array1D& tree_indices,
    const Int32Array1D& seq_ids,
    nb::list token_batches_list) {

    // Convert tree_indices ndarray to vector
    std::vector<int> tree_indices_vec(tree_indices.size());
    for (size_t i = 0; i < tree_indices.size(); ++i) {
        tree_indices_vec[i] = tree_indices.data()[i];
    }

    // Convert seq_ids ndarray to vector
    std::vector<int> seq_ids_vec(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        seq_ids_vec[i] = seq_ids.data()[i];
    }

    // Convert token_batches list of ndarrays to vector of vectors
    std::vector<std::vector<int32_t>> token_batches_vec;
    token_batches_vec.reserve(token_batches_list.size());

    for (size_t i = 0; i < token_batches_list.size(); ++i) {
        nb::handle tokens_handle = token_batches_list[i];
        Int32Array1D tokens = nb::cast<Int32Array1D>(tokens_handle);
        token_batches_vec.emplace_back(tokens.data(), tokens.data() + tokens.size());
    }

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    forest.batch_extend(
        std::span<const int>(tree_indices_vec),
        std::span<const int>(seq_ids_vec),
        token_batches_vec);
}

std::vector<Draft> batch_speculate_forest(
    SuffixForest& forest,
    const std::vector<int>& tree_indices,
    const std::vector<std::vector<int32_t>>& contexts,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    return forest.batch_speculate(
        std::span<const int>(tree_indices),
        contexts,
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


std::vector<Draft> batch_speculate_forest_ndarray(
    SuffixForest& forest,
    const Int32Array1D& tree_indices,
    nb::list contexts_list,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Convert tree_indices ndarray to vector
    std::vector<int> tree_indices_vec(tree_indices.size());
    for (size_t i = 0; i < tree_indices.size(); ++i) {
        tree_indices_vec[i] = tree_indices.data()[i];
    }

    // Convert contexts list of ndarrays to vector of vectors
    std::vector<std::vector<int32_t>> contexts_vec;
    contexts_vec.reserve(contexts_list.size());

    for (size_t i = 0; i < contexts_list.size(); ++i) {
        nb::handle ctx_handle = contexts_list[i];
        Int32Array1D ctx = nb::cast<Int32Array1D>(ctx_handle);
        contexts_vec.emplace_back(ctx.data(), ctx.data() + ctx.size());
    }

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    return forest.batch_speculate(
        std::span<const int>(tree_indices_vec),
        contexts_vec,
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


NB_MODULE(_C, m) {
    nb::set_leak_warnings(false);

    nb::class_<Draft>(m, "Draft")
        .def_rw("token_ids", &Draft::token_ids)
        .def_rw("parents", &Draft::parents)
        .def_rw("probs", &Draft::probs)
        .def_rw("score", &Draft::score)
        .def_rw("match_len", &Draft::match_len);

    nb::class_<SuffixTree>(m, "SuffixTree")
        .def(nb::init<int>())
        .def("num_seqs", &SuffixTree::num_seqs)
        .def("remove", &SuffixTree::remove)
        // Overloads for extend method. Use different names to avoid overload
        // resolution overhead at run-time.
        .def("extend", &extend_vector)
        .def("extend_ndarray", &extend_ndarray)
        // Overloads for speculate method.
        .def("speculate", &speculate_vector)
        .def("speculate_ndarray", &speculate_ndarray)
        // Debugging methods, not meant to be used in critical loop.
        .def("check_integrity", &SuffixTree::check_integrity)
        .def("estimate_memory", &SuffixTree::estimate_memory);

    nb::class_<SuffixForest>(m, "SuffixForest")
        .def(nb::init<int, int, int>(),
             nb::arg("max_depth"),
             nb::arg("num_threads") = -1,
             nb::arg("parallel_threshold") = 4,
             "Create a SuffixForest with configurable parallelization settings.\n\n"
             "Args:\n"
             "    max_depth: Maximum depth for all trees in the forest\n"
             "    num_threads: Number of threads (-1=auto, 0=sequential, >0=explicit)\n"
             "    parallel_threshold: Minimum batch size to trigger parallelization")
        .def("create_tree", &SuffixForest::create_tree,
             "Create a new tree in the forest and return its index")
        .def("remove_tree", &SuffixForest::remove_tree,
             nb::arg("tree_index"),
             "Remove a tree from the forest")
        .def("get_tree", nb::overload_cast<int>(&SuffixForest::get_tree),
             nb::arg("tree_index"),
             nb::rv_policy::reference_internal,
             "Get a reference to a tree in the forest")
        .def("has_tree", &SuffixForest::has_tree,
             nb::arg("tree_index"),
             "Check if a tree exists in the forest")
        .def("num_trees", &SuffixForest::num_trees,
             "Get the number of trees currently in the forest")
        .def("get_num_threads", &SuffixForest::get_num_threads,
             "Get the configured number of threads")
        .def("get_parallel_threshold", &SuffixForest::get_parallel_threshold,
             "Get the parallel threshold")
        // Batch extend methods
        .def("batch_extend", &batch_extend_forest,
             nb::arg("tree_indices"),
             nb::arg("seq_ids"),
             nb::arg("token_batches"),
             "Batch extend multiple trees with new tokens in parallel")
        .def("batch_extend_ndarray", &batch_extend_forest_ndarray,
             nb::arg("tree_indices"),
             nb::arg("seq_ids"),
             nb::arg("token_batches"),
             "Batch extend multiple trees with new tokens (ndarray version)")
        // Batch speculation methods
        .def("batch_speculate", &batch_speculate_forest,
             nb::arg("tree_indices"),
             nb::arg("contexts"),
             nb::arg("max_spec_tokens"),
             nb::arg("max_spec_factor"),
             nb::arg("max_spec_offset"),
             nb::arg("min_token_prob"),
             nb::arg("use_tree_spec"),
             "Perform batched parallel speculation across multiple trees")
        .def("batch_speculate_ndarray", &batch_speculate_forest_ndarray,
             nb::arg("tree_indices"),
             nb::arg("contexts"),
             nb::arg("max_spec_tokens"),
             nb::arg("max_spec_factor"),
             nb::arg("max_spec_offset"),
             nb::arg("min_token_prob"),
             nb::arg("use_tree_spec"),
             "Perform batched parallel speculation with ndarray inputs");
}
