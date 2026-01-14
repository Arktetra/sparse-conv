#pragma once
#include <torch/extension.h>

torch::Tensor hashmap_lookup_submanifold_conv_neighbor_map_cuda(
    const torch::Tensor& hashmap_keys,
    const torch::Tensor& hashmap_values,
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