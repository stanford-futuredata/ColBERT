#include <torch/extension.h>

#include <algorithm>
#include <numeric>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>

template <typename T>
torch::Tensor segmented_lookup_impl(const torch::Tensor input,
                                    const torch::Tensor pids,
                                    const torch::Tensor lengths,
                                    const torch::Tensor offsets) {
    int64_t num_docs = pids.size(0);
    int64_t num_outputs = std::accumulate(
        lengths.data_ptr<int64_t>(), lengths.data_ptr<int64_t>() + num_docs, 0);

    int64_t dim;
    torch::Tensor output;

    if (input.dim() == 1) {
        dim = 1;
        output = torch::zeros({num_outputs}, input.options());
    } else {
        assert(input.dim() == 2);
        dim = input.size(1);
        output = torch::zeros({num_outputs, dim}, input.options());
    }

    auto lengths_a = lengths.accessor<int64_t, 1>();
    auto offsets_a = offsets.accessor<int64_t, 1>();

    int64_t cumulative_lengths[num_docs + 1];
    cumulative_lengths[0] = 0;
    std::partial_sum(lengths.data_ptr<int64_t>(),
                     lengths.data_ptr<int64_t>() + num_docs,
                     cumulative_lengths + 1);

    at::parallel_for(0, num_docs, 0, [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; i++) {
            std::memcpy(output.data_ptr<T>() + (cumulative_lengths[i] * dim),
                        input.data_ptr<T>() + (offsets_a[i] * dim),
                        lengths_a[i] * dim * sizeof(T));
        }
    });

    return output;
}

torch::Tensor segmented_lookup(const torch::Tensor input,
                               const torch::Tensor pids,
                               const torch::Tensor lengths,
                               const torch::Tensor offsets) {
    if (input.dtype() == torch::kUInt8) {
        return segmented_lookup_impl<uint8_t>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kInt32) {
        return segmented_lookup_impl<int>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kInt64) {
        return segmented_lookup_impl<int64_t>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kFloat32) {
        return segmented_lookup_impl<float>(input, pids, lengths, offsets);
    } else if (input.dtype() == torch::kFloat16) {
        return segmented_lookup_impl<at::Half>(input, pids, lengths, offsets);
    } else {
        assert(false);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("segmented_lookup_cpp", &segmented_lookup, "Segmented lookup");
}
