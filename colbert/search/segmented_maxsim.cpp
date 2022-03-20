#include <algorithm>
#include <numeric>

#include <torch/extension.h>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>

torch::Tensor segmented_maxsim(
        const torch::Tensor scores,
        const torch::Tensor lengths) {
    auto lengths_a = lengths.accessor<int64_t, 1>(); 
    uint64_t num_docs = lengths.size(0);
    uint64_t num_query_vectors = scores.size(0);
    uint64_t num_doc_vectors = scores.size(1);
  
    torch::Tensor max_scores = torch::zeros({num_docs, num_query_vectors}, scores.options());
    auto max_scores_a = max_scores.accessor<float, 2>();

    at::parallel_for(0, num_query_vectors, 0, [&](uint64_t start, uint64_t end) {
        for (uint64_t j = start; j < end; j++) {
            float* offset = scores.data_ptr<float>() + (num_doc_vectors * j);
            for (uint64_t i = 0; i < num_docs; i++) {
                max_scores_a[i][j] = *std::max_element(offset, offset + lengths_a[i]);
                offset += lengths_a[i];
            }
         }
    });
    return max_scores.sum(1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_maxsim_cpp", &segmented_maxsim, "Segmented MaxSim");
}

    // auto scores_a = scores.accessor<float, 2>();

            
    /*
    int64_t cumulative_lengths[num_docs + 1];
    cumulative_lengths[0] = 0; 
    std::partial_sum(lengths.data_ptr<int64_t>(), lengths.data_ptr<int64_t>() + num_docs, cumulative_lengths + 1);
    */

            //uint64_t current_pid = 0;
            //uint64_t offset = 0;
            //int64_t current_length = lengths_a[current_pid];


            /*
            for (uint64_t i = 0; i < num_doc_vectors; i++) {
                if (max_scores_a[current_pid][j] < scores.data([offset][j]) {
                    max_scores_a[current_pid][j] = scores_a[offset][j]; 
                }
                offset++;
                current_length -= 1;
                if (current_length == 0) {
                     current_pid++;
                     if (current_pid < num_docs) {
                        current_length = lengths_a[current_pid];
                     }
                }
            }
            */

