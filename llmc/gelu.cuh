/*
(Approximate) GeLU non-linearity layer
*/
#include <assert.h>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp, int N_vec) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N_vec) { return; }

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

__global__ void gelu_forward_tail_kernel(floatX* out, const floatX* inp, int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }
    float xi = (float)inp[idx];
    float cube = 0.044715f * xi * xi * xi;
    out[idx] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
}

__global__ void gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp, int N_vec) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N_vec) { return; }

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}

__global__ void gelu_backward_inplace_tail_kernel(floatX* d_in_out, const floatX* inp, int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }
    float x = (float)inp[idx];
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
    d_in_out[idx] = (floatX)(local_grad * (float)d_in_out[idx]);
}

// ----------------------------------------------------------------------------
// kernel launchers

void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (N <= 0) { return; }

    const int vec_block_size = 512;
    const int tail_block_size = 256;
    const int N_vec = (N / (int)x128::size) * (int)x128::size;

    if (N_vec > 0) {
        const int vec_grid_size = CEIL_DIV(N_vec, vec_block_size * (int)x128::size);
        gelu_forward_kernel2<<<vec_grid_size, vec_block_size, 0, stream>>>(out, inp, N_vec);
    }
    if (N_vec < N) {
        const int tail_grid_size = CEIL_DIV(N - N_vec, tail_block_size);
        gelu_forward_tail_kernel<<<tail_grid_size, tail_block_size, 0, stream>>>(out, inp, N_vec, N);
    }

    cudaCheck(cudaGetLastError());
}

void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    if (N <= 0) { return; }

    const int vec_block_size = 128;
    const int tail_block_size = 256;
    const int N_vec = (N / (int)x128::size) * (int)x128::size;

    if (N_vec > 0) {
        const int vec_grid_size = CEIL_DIV(N_vec, vec_block_size * (int)x128::size);
        gelu_backward_inplace_kernel<<<vec_grid_size, vec_block_size, 0, stream>>>(d_in_out, inp, N_vec);
    }
    if (N_vec < N) {
        const int tail_grid_size = CEIL_DIV(N - N_vec, tail_block_size);
        gelu_backward_inplace_tail_kernel<<<tail_grid_size, tail_block_size, 0, stream>>>(d_in_out, inp, N_vec, N);
    }

    cudaCheck(cudaGetLastError());
}
