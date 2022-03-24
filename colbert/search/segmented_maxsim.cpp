#include <torch/extension.h>

#include <algorithm>
#include <numeric>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>

torch::Tensor segmented_maxsim(const torch::Tensor scores,
                               const torch::Tensor lengths) {
    auto lengths_a = lengths.accessor<int64_t, 1>();
    uint64_t num_docs = lengths.size(0);
    uint64_t num_doc_vectors = scores.size(0);
    uint64_t num_query_vectors = scores.size(1);

    torch::Tensor max_scores =
        torch::zeros({num_docs, num_query_vectors}, scores.options());

    int64_t cumulative_lengths[num_docs + 1];
    cumulative_lengths[0] = 0;
    std::partial_sum(lengths.data_ptr<int64_t>(),
                     lengths.data_ptr<int64_t>() + num_docs,
                     cumulative_lengths + 1);

    at::parallel_for(0, num_docs, 0, [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
            auto max_scores_offset =
                max_scores.data_ptr<float>() + (i * num_query_vectors);
            auto scores_offset = scores.data_ptr<float>() +
                                 (cumulative_lengths[i] * num_query_vectors);
            for (uint64_t j = 0; j < lengths_a[i]; j++) {
                std::transform(max_scores_offset,
                               max_scores_offset + num_query_vectors,
                               scores_offset, max_scores_offset,
                               [](float a, float b) { return std::max(a, b); });
                scores_offset += num_query_vectors;
            }
        }
    });
    return max_scores.sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_maxsim_cpp", &segmented_maxsim, "Segmented MaxSim");
}
