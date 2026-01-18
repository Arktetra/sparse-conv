#pragma once
#include <torch/extension.h>

torch::Tensor hashmap_build_submanifold_conv2d_neighbor_map_cuda_naive(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W,
    int H,
    int Kw,
    int Kh,
    int Dw,
    int Dh
);

torch::Tensor hashmap_build_submanifold_conv_neighbor_map_cuda_naive(
    torch::Tensor& hashmap_keys,
    torch::Tensor& hashmap_values,
    const torch::Tensor& coords,
    int W,
    int H,
    int D,
    int Kw,
    int Kh,
    int Kd,
    int Dw,
    int Dh,
    int Dd
);