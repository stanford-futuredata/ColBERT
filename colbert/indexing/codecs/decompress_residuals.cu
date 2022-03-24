#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits>
// binary_residuals, torch::PackedTensorAccessor32<at::Half, 2,
// torch::RestrictPtrTraits> output) {

__global__ void decompress_residuals_kernel(
    const uint8_t* binary_residuals,
    const torch::PackedTensorAccessor32<at::Half, 1, torch::RestrictPtrTraits>
        bucket_weights,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits>
        reversed_bit_map,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits>
        bucket_weight_combinations,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> codes,
    const torch::PackedTensorAccessor32<at::Half, 2, torch::RestrictPtrTraits>
        centroids,
    const int n, const int dim, const int nbits, const int packed_size,
    at::Half* output) {
    const uint8_t two_to_the_nbits = 1 << nbits;
    const int packed_dim = (int)(dim * nbits / packed_size);
    const int i = blockIdx.x;
    const int j = threadIdx.x;

    if (i >= n) return;
    if (j >= dim * nbits / packed_size) return;

    const int code = codes[i];

    uint8_t x = binary_residuals[i * packed_dim + j];
    x = reversed_bit_map[x];
    int output_idx = (int)(j * packed_size / nbits);
    for (int k = 0; k < two_to_the_nbits; k++) {
        assert(output_idx < dim);
        const int bucket_weight_idx = bucket_weight_combinations[x][k];
        output[i * dim + output_idx] = bucket_weights[bucket_weight_idx];
        output[i * dim + output_idx] += centroids[code][output_idx];
        output_idx++;
    }
}

template <typename scalar_t>
__global__ void decompress_residuals_kernel_vectorized(
    const uint8_t* binary_residuals,
    const torch::PackedTensorAccessor32<at::Half, 1, torch::RestrictPtrTraits>
        bucket_weights,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits>
        reversed_bit_map,
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits>
        bucket_weight_combinations,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> codes,
    const torch::PackedTensorAccessor32<at::Half, 2, torch::RestrictPtrTraits>
        centroids,
    const int n, const int dim, const int nbits, const int packed_size,
    at::Half* output) {
    const uint8_t two_to_the_nbits = 1 << nbits;
    const int packed_dim = (int)(dim * nbits / packed_size);
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int remaining_bytes = (n * packed_dim) - (thread_id * sizeof(scalar_t));
    if (remaining_bytes <= 0) return;

    const scalar_t x_wide = *(reinterpret_cast<const scalar_t*>(
        &binary_residuals[thread_id * sizeof(scalar_t)]));
    const uint8_t* x_byte = reinterpret_cast<const uint8_t*>(&x_wide);

    // bytes per thread * 4 values per byte
    at::Half output_vector[sizeof(scalar_t) * 4];

    int output_idx = thread_id * sizeof(scalar_t) * packed_size / nbits;
    int count = 0;
    int i = output_idx / dim;
    int code = codes[i];
    for (int k = 0; k < min((int)sizeof(scalar_t), remaining_bytes); k++) {
        uint8_t x = x_byte[k];
        x = reversed_bit_map[x];
        for (int l = 0; l < two_to_the_nbits; l++) {
            assert(output_idx < n * dim);
            int j = output_idx % dim;
            const int bucket_weight_idx = bucket_weight_combinations[x][l];
            output_vector[count] =
                bucket_weights[bucket_weight_idx] + centroids[code][j];
            count++;
            output_idx++;
            if (j == dim - 1) {
                i = output_idx / dim;
                code = codes[i];
            }
        }
    }

    // n threads * 16 bytes per thread / 2 bytes per value
    output_idx = thread_id * sizeof(scalar_t) * packed_size / nbits;
    assert(output_idx < n * dim);
    int4* vectorized_output = reinterpret_cast<int4*>(&output[output_idx]);
    int4* vectorized_output_vector = reinterpret_cast<int4*>(output_vector);

    // Wrote <count> values * 2 bytes per value / 16 bytes per vectorized output
    assert((count * 2) % sizeof(int4) == 0);
    int num_vectorized_outputs = (int)(count * 2 / sizeof(int4));
    for (int k = 0; k < num_vectorized_outputs; k++) {
        vectorized_output[k] = vectorized_output_vector[k];
        count -= sizeof(int4) / 2;
    }
    assert(count == 0);
}

torch::Tensor decompress_residuals_cuda(
    const torch::Tensor binary_residuals, const torch::Tensor bucket_weights,
    const torch::Tensor reversed_bit_map,
    const torch::Tensor bucket_weight_combinations, const torch::Tensor codes,
    const torch::Tensor centroids, const int dim, const int nbits) {
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::kCUDA, 0)
                       .requires_grad(false);
    torch::Tensor output =
        torch::zeros({(int)binary_residuals.size(0), (int)dim}, options);

    // TODO: Set this automatically?
    const int packed_size = 8;

    /*
    // NOTE: Use this code for vectorized implementation
    const int threads = 32;
    const int blocks = (int) (ceil(((float) binary_residuals.size(0) *
    binary_residuals.size(1)) / (threads * sizeof(int4))));

    decompress_residuals_kernel_vectorized<int4><<<blocks, threads>>>(
        binary_residuals.data<uint8_t>(),
        bucket_weights.packed_accessor32<at::Half, 1, torch::RestrictPtrTraits>(),
        reversed_bit_map.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        bucket_weight_combinations.packed_accessor32<uint8_t, 2,
    torch::RestrictPtrTraits>(), codes.packed_accessor32<int, 1,
    torch::RestrictPtrTraits>(), centroids.packed_accessor32<at::Half, 2,
    torch::RestrictPtrTraits>(), binary_residuals.size(0), dim, nbits, packed_size,
        output.data<at::Half>()
    );
    */

    const int threads = 32;
    const int blocks =
        (binary_residuals.size(0) * binary_residuals.size(1)) / threads;

    decompress_residuals_kernel<<<blocks, threads>>>(
        binary_residuals.data<uint8_t>(),
        bucket_weights
            .packed_accessor32<at::Half, 1, torch::RestrictPtrTraits>(),
        reversed_bit_map
            .packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        bucket_weight_combinations
            .packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        codes.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        centroids.packed_accessor32<at::Half, 2, torch::RestrictPtrTraits>(),
        binary_residuals.size(0), dim, nbits, packed_size,
        output.data<at::Half>());

    return output;
}
