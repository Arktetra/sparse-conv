#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "api.h"
#include "hash.cuh"

template<typename K, typename V>
static __global__ void hashmap_insert_cuda_kernel(
    const size_t N,
    const size_t M,
    K* __restrict__ hashmap_keys,
    V* __restrict__ hashmap_values,
    const K* __restrict__ keys,
    const V* __restrict__ values
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < M) {
        K key = keys[thread_id];
        V value = values[thread_id];
        linear_probing_insert(hashmap_keys, hashmap_values, key, value, N);
    }
}

template<typename K, typename V>
static void dispatch_hashmap_insert_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    const torch::Tensor& values
) {
    hashmap_insert_cuda_kernel<<<
        (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        hashmap_keys.size(0),
        keys.size(0),
        hashmap_keys.data_ptr<K>(),
        hashmap_values.data_ptr<V>(),
        keys.data_ptr<K>(),
        values.data_ptr<V>()
    );
}

void hashmap_insert_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    const torch::Tensor& values
) {
    TORCH_CHECK(hashmap_keys.is_cuda(), "Hashmap keys must be on GPU.");
    TORCH_CHECK(hashmap_values.is_cuda(), "Hashmap values must be on GPU.");
    TORCH_CHECK(keys.is_cuda(), "Keys must be on GPU.");
    TORCH_CHECK(values.is_cuda(), "Values must be on GPU.");

    if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt32) {
        TORCH_CHECK(keys.dtype() == torch::kUInt32, "keys must be uint32");
        TORCH_CHECK(values.dtype() == torch::kUInt32, "valuess must be uint32");
        dispatch_hashmap_insert_cuda<uint32_t, uint32_t>(hashmap_keys, hashmap_values, keys, values);
    } else if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt64) {
        TORCH_CHECK(keys.dtype() == torch::kUInt32, "keys must be uint32");
        TORCH_CHECK(values.dtype() == torch::kUInt64, "valuess must be uint64");
        dispatch_hashmap_insert_cuda<uint32_t, uint64_t>(hashmap_keys, hashmap_values, keys, values);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt32) {
        TORCH_CHECK(keys.dtype() == torch::kUInt64, "keys must be uint64");
        TORCH_CHECK(values.dtype() == torch::kUInt32, "valuess must be uint32");
        dispatch_hashmap_insert_cuda<uint64_t, uint32_t>(hashmap_keys, hashmap_values, keys, values);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt64) {
        TORCH_CHECK(keys.dtype() == torch::kUInt64, "keys must be uint64");
        TORCH_CHECK(values.dtype() == torch::kUInt64, "valuess must be uint64");
        dispatch_hashmap_insert_cuda<uint64_t, uint64_t>(hashmap_keys, hashmap_values, keys, values);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}

template<typename K, typename V>
static __global__ void hashmap_lookup_cuda_kernel(
    const size_t N,
    const size_t M,
    const K* __restrict__ hashmap_keys,
    const V* __restrict__ hashmap_values,
    const K* __restrict__ keys,
    V* __restrict__ values
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        K key = keys[thread_id];
        values[thread_id] = linear_probing_lookup(hashmap_keys, hashmap_values, key, N);
    }
}

template<typename K, typename V>
static void dispatch_hashmap_lookup_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    torch::Tensor& values
) {
    hashmap_lookup_cuda_kernel<<<
        (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        hashmap_keys.size(0),
        keys.size(0),
        hashmap_keys.data_ptr<K>(),
        hashmap_values.data_ptr<V>(),
        keys.data_ptr<K>(),
        values.data_ptr<V>()
    );
}

torch::Tensor hashmap_lookup_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys
) {
    auto output = torch::empty({keys.size(0)}, torch::dtype(hashmap_values.dtype()).device(hashmap_values.device()));

    TORCH_CHECK(hashmap_keys.is_cuda(), "Hashmap Keys must be on GPU.");
    TORCH_CHECK(hashmap_values.is_cuda(), "Hashmap Values must be on GPU.");
    TORCH_CHECK(keys.is_cuda(), "Keys must be on GPU.");

    if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt32) {
        TORCH_CHECK(keys.dtype() == torch::kUInt32, "Keys must be uint32");
        TORCH_CHECK(output.dtype() == torch::kUInt32, "Keys must be uint32");
        dispatch_hashmap_lookup_cuda<uint32_t, uint32_t>(hashmap_keys, hashmap_values, keys, output);
    } else if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt64) {
        TORCH_CHECK(keys.dtype() == torch::kUInt32, "Keys must be uint32");
        TORCH_CHECK(output.dtype() == torch::kUInt64, "Keys must be uint64");
        dispatch_hashmap_lookup_cuda<uint32_t, uint64_t>(hashmap_keys, hashmap_values, keys, output);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt32) {
        TORCH_CHECK(keys.dtype() == torch::kUInt64, "Keys must be uint64");
        TORCH_CHECK(output.dtype() == torch::kUInt32, "Keys must be uint32");
        dispatch_hashmap_lookup_cuda<uint64_t, uint32_t>(hashmap_keys, hashmap_values, keys, output);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt64) {
        TORCH_CHECK(keys.dtype() == torch::kUInt64, "Keys must be uint32");
        TORCH_CHECK(output.dtype() == torch::kUInt64, "Keys must be uint32");
        dispatch_hashmap_lookup_cuda<uint64_t, uint64_t>(hashmap_keys, hashmap_values, keys, output);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return output;
}

template<typename K, typename V>
static __global__ void hashmap_insert_3d_cuda_kernel(
    const size_t N,
    const size_t M,
    const int W,
    const int H,
    const int D,
    K* __restrict__ hashmap_keys,
    V* __restrict__ hashmap_values,
    const int32_t* __restrict__ coords,
    const V* __restrict__ values
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + z;
        K key = static_cast<K>(flat_idx);
        V value = values[thread_id];
        linear_probing_insert(hashmap_keys, hashmap_values, key, value, N);
    }
}

template<typename K, typename V>
static void dispatch_hashmap_insert_3d_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    int W, int H, int D
) {
    hashmap_insert_3d_cuda_kernel<<<
        (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        hashmap_keys.size(0),
        keys.size(0),
        W, H, D,
        hashmap_keys.data_ptr<K>(),
        hashmap_values.data_ptr<V>(),
        keys.data_ptr<int32_t>(),
        values.data_ptr<V>()
    );
}

void hashmap_insert_3d_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    int W, int H, int D
) {
    TORCH_CHECK(hashmap_keys.is_cuda(), "Hashmap keys must be on GPU.");
    TORCH_CHECK(hashmap_values.is_cuda(), "Hashmap values must be on GPU.");
    TORCH_CHECK(keys.is_cuda(), "Keys must be on GPU.");
    TORCH_CHECK(values.is_cuda(), "Values must be on GPU.");

    if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt32) {
        TORCH_CHECK(values.dtype() == torch::kUInt32, "valuess must be uint32");
        dispatch_hashmap_insert_3d_cuda<uint32_t, uint32_t>(hashmap_keys, hashmap_values, keys, values, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt64) {
        TORCH_CHECK(values.dtype() == torch::kUInt64, "valuess must be uint64");
        dispatch_hashmap_insert_3d_cuda<uint32_t, uint64_t>(hashmap_keys, hashmap_values, keys, values, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt32) {
        TORCH_CHECK(values.dtype() == torch::kUInt32, "valuess must be uint32");
        dispatch_hashmap_insert_3d_cuda<uint64_t, uint32_t>(hashmap_keys, hashmap_values, keys, values, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt64) {
        TORCH_CHECK(values.dtype() == torch::kUInt64, "valuess must be uint64");
        dispatch_hashmap_insert_3d_cuda<uint64_t, uint64_t>(hashmap_keys, hashmap_values, keys, values, W, H, D);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}

template<typename K, typename V>
static __global__ void hashmap_lookup_3d_cuda_kernel(
    const size_t N,
    const size_t M,
    int W, int H, int D,
    const K* __restrict__ hashmap_keys,
    const V* __restrict__ hashmap_values,
    const int32_t* __restrict__ keys,
    V* __restrict__ values
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(keys)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;

        if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
            values[thread_id] = std::numeric_limits<V>::max();
            return;
        }
        
        size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + z;
        K key = static_cast<K>(flat_idx);
        values[thread_id] = linear_probing_lookup(hashmap_keys, hashmap_values, key, N);
    }
}

template<typename K, typename V>
void dispatch_hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    torch::Tensor& values,
    int W, int H, int D
) {
    hashmap_lookup_3d_cuda_kernel<<<
        (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        hashmap_keys.size(0),
        keys.size(0),
        W, H, D,
        hashmap_keys.data_ptr<K>(),
        hashmap_values.data_ptr<V>(),
        keys.data_ptr<int32_t>(),
        values.data_ptr<V>()
    );
}

torch::Tensor hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    int W, int H, int D
) {
    auto output = torch::empty({keys.size(0)}, torch::dtype(hashmap_values.dtype()).device(hashmap_values.device()));

    TORCH_CHECK(hashmap_keys.is_cuda(), "Hashmap keys must be on GPU.");
    TORCH_CHECK(hashmap_values.is_cuda(), "Hashmap values must be on GPU.");
    TORCH_CHECK(keys.is_cuda(), "Keys must be on GPU.");

    if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt32) {
        dispatch_hashmap_lookup_3d_cuda<uint32_t, uint32_t>(hashmap_keys, hashmap_values, keys, output, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt64) {
        dispatch_hashmap_lookup_3d_cuda<uint32_t, uint64_t>(hashmap_keys, hashmap_values, keys, output, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt32) {
        dispatch_hashmap_lookup_3d_cuda<uint64_t, uint32_t>(hashmap_keys, hashmap_values, keys, output, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt64) {
        dispatch_hashmap_lookup_3d_cuda<uint64_t, uint64_t>(hashmap_keys, hashmap_values, keys, output, W, H, D);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return output;
}

template<typename K, typename V>
static __global__ void hashmap_insert_2d_idx_as_val_cuda_kernel(
    const size_t N,
    const size_t M,
    int W, int H,
    K* __restrict__ hashmap_keys,
    V* __restrict__ hashmap_values,
    const int32_t* __restrict__ keys
) {
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < M) {
        int3 coord = reinterpret_cast<const int3*>(keys)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;

        size_t flat_idx = (size_t)b * W * H + (size_t)x * H + y;
        K key = static_cast<K>(flat_idx);
        V value = static_cast<V>(thread_id)
        linear_probing_insert(hashmap_keys, hashmap_values, key, value, N);
    }
}

template<typename K, typename V>
void dispatch_hashmap_insert_2d_idx_as_val_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    int W, int H
) {
    hashmap_insert_2d_idx_as_val_cuda_kernel<<<
        (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        hashmap_keys.size(0),
        keys.size(0),
        W, H,
        hashmap_keys.data_ptr<K>(),
        hashmap_values.data_ptr<V>(),
        keys.data_ptr<int32_t>()
    );
}

void hashmap_insert_2d_idx_as_val_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    int W, int H
) {
    if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt32) {
        dispatch_hashmap_insert_2d_idx_as_val_cuda<uint32_t, uint32_t>(hashmap_keys, hashmap_values, keys, W, H);
    } else if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt64) {
        dispatch_hashmap_insert_2d_idx_as_val_cuda<uint32_t, uint64_t>(hashmap_keys, hashmap_values, keys, W, H);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt32) {
        dispatch_hashmap_insert_2d_idx_as_val_cuda<uint64_t, uint32_t>(hashmap_keys, hashmap_values, keys, W, H);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt64) {
        dispatch_hashmap_insert_2d_idx_as_val_cuda<uint64_t, uint64_t>(hashmap_keys, hashmap_values, keys, W, H);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}

template<typename K, typename V>
static __global__ void hashmap_insert_3d_idx_as_val_cuda_kernel(
    const size_t N,
    const size_t M,
    int W, int H, int D,
    K* __restrict__ hashmap_keys,
    V* __restrict__ hashmap_values,
    const int32_t* __restrict__ keys
) {
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(keys)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        size_t flat_idx = (size_t)b * W * H * D + (size_t)x * H * D + (size_t)y * D + z;
        K key = static_cast<K>(flat_idx);
        V value = static_cast<V>(thread_id);
        linear_probing_insert(hashmap_keys, hashmap_values, key, value, N);
    }
}

template<typename K, typename V>
void dispatch_hashmap_insert_3d_idx_as_val_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    int W, int H, int D
) {
    hashmap_insert_3d_idx_as_val_cuda_kernel<<<
        (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
        BLOCK_SIZE
    >>>(
        hashmap_keys.size(0),
        keys.size(0),
        W, H, D,
        hashmap_keys.data_ptr<K>(),
        hashmap_values.data_ptr<V>(),
        keys.data_ptr<int32_t>()
    );
}

void hashmap_insert_3d_idx_as_val_cuda(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& keys,
    int W, int H, int D
) {
    TORCH_CHECK(hashmap_keys.is_cuda(), "Hashmap keys must be on GPU.");
    TORCH_CHECK(hashmap_values.is_cuda(), "Hashmap values must be on GPU.");
    TORCH_CHECK(keys.is_cuda(), "Keys must be on GPU.");

    if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt32) {
        dispatch_hashmap_insert_3d_idx_as_val_cuda<uint32_t, uint32_t>(hashmap_keys, hashmap_values, keys, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt32 && hashmap_values.dtype() == torch::kUInt64) {
        dispatch_hashmap_insert_3d_idx_as_val_cuda<uint32_t, uint64_t>(hashmap_keys, hashmap_values, keys, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt32) {
        dispatch_hashmap_insert_3d_idx_as_val_cuda<uint64_t, uint32_t>(hashmap_keys, hashmap_values, keys, W, H, D);
    } else if (hashmap_keys.dtype() == torch::kUInt64 && hashmap_values.dtype() == torch::kUInt64) {
        dispatch_hashmap_insert_3d_idx_as_val_cuda<uint64_t, uint64_t>(hashmap_keys, hashmap_values, keys, W, H, D);
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}