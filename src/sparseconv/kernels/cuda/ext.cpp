#include <torch/extension.h>
#include "hash/api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hashmap_insert_cuda", &hashmap_insert_cuda);
    m.def("hashmap_lookup_cuda", &hashmap_lookup_cuda);
}