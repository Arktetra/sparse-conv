#include <torch/extension.h>
#include "hash/api.h"
#include "neighbor_map/api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hashmap_insert_cuda", &hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &hashmap_lookup_cuda);
    m.def("hashmap_build_submanifold_conv_neighbor_map_cuda_naive", &hashmap_build_submanifold_conv_neighbor_map_cuda_naive);
}