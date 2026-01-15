#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "neighbor_map.h"
#include "../hash/api.h"
#include "../hash/hash.cuh"

/**
 * Lookup sparse submanifold 2D convolution with hashmap.
 * @param N                 number of elements in the hashmap.
 * @param M                 number of 3D coordinates.
 * @param W                 width dimension.
 * @param H                 height dimension.
 * @param D                 depth dimension.
 * @param V                 volume of the kernel.
 * @param Kw                kernel width dimension.
 * @param Kh                kernel height dimension.
 * @param Kd                kernel depth dimension.
 * @param Dw                dilation of width.
 * @param Dh                dilation of height.
 * @param Dd                dilation of depth.
 * @param hashmap_keys      [N] uint32/uint64 tensor containing the hashmap keys.
 * @param hashmap_values    [N] uint32 tensor containing the hashmap values.
 * @param coords            [M, 4] int32 tensor containing the keys to be looked up.
 * @param neighbor          [M, V] uint32 tensor containing the submanifold neighbor map.
 */
template <typename K>
static __global__ void hashmap_lookup_submanifold_conv_neighbor_map_cuda_naive(
    const size_t N,
    const size_t M,
    int W, int H, int D,
    int V,
    int Kw, int Kh, int Kd,
    int Dw, int Dh, int Dd,
    const K* __restrict__ hashmap_keys,
    const uint32_t* __restrict__ hashmap_values,
    const int32_t* coords,
    uint32_t* neighbor
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // int A_half = A / 2 + 1;
    size_t idx = thread_id / V;
    if (idx < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[idx];
        int b = coord.x;
        int x = coord.y - Kw / 2 * Dw;   // Center the kernel
        int y = coord.z - Kh / 2 * Dh;
        int z = coord.w - Kd / 2 * Dd;
        int KhKd = Kh * Kd;

        int v = thread_id % V;      // offset within the kernel

        uint32_t value = std::numeric_limits<uint32_t>::max();
        if (v == V / 2) {
            value = idx;
        } else {
            int kx = x + v / KhKd * Dw;
            int ky = y + v / Kd % Kh * Dh;
            int kz = z + v % Kd * Dd;
            
            if (kx >= 0 && kx < W && ky >= 0 && ky < H && kz >= 0 && kz < D) {
                size_t flat_idx = (size_t)b * W * H * D + (size_t)kx * H * D + (size_t)ky * D + kz;
                K key = static_cast<K>(flat_idx);
                value = linear_probing_lookup(hashmap_keys, hashmap_values, key, N);
            }
        }

        if (value != std::numeric_limits<uint32_t>::max()) {
            neighbor[idx * V + v] = value;
        }
    }
}

torch::Tensor hashmap_build_submanifold_conv_neighbor_map_cuda_naive(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W, int H, int D,
    int Kw, int Kh, int Kd,
    int Dw, int Dh, int Dd
) {
    int V = Kw * Kh * Kd;

    hashmap_insert_3d_idx_as_val_cuda(
        hashmap_keys,
        hashmap_values,
        coords,
        W, H, D
    );

    auto neighbor = torch::full(
        {coords.size(0), V}, 
        std::numeric_limits<uint32_t>::max(), 
        torch::dtype(torch::kUInt32).device(hashmap_keys.device())
    );

    if (hashmap_keys.dtype() == torch::kUInt32) {
        hashmap_lookup_submanifold_conv_neighbor_map_cuda_naive<<<
            (coords.size(0) * V + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            coords.size(0),
            W, H, D,
            V,
            Kw, Kh, Kd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint32_t>(),
            hashmap_values.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>()
        );
    } else if (hashmap_keys.dtype() == torch::kUInt64) {
        hashmap_lookup_submanifold_conv_neighbor_map_cuda_naive<<<
            (coords.size(0) * V + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap_keys.size(0),
            coords.size(0),
            W, H, D,
            V,
            Kw, Kh, Kd,
            Dw, Dh, Dd,
            hashmap_keys.data_ptr<uint64_t>(),
            hashmap_values.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            neighbor.data_ptr<uint32_t>()
        );
    }

    return neighbor;
}