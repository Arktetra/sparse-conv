# sparse-conv

Sparse convolution library.

## Installation

```bash
git clone https://github.com/Arktetra/sparse-conv.git
cd sparse-conv
pip install . --no-build-isolation
```

## Example

```python
import torch
import torch.nn as nn

from sparseconv import SparseConvTensor
from sparseconv.ops.submanifold_conv3d import Algorithm, sparse_submanifold_conv3d

B, Cin, W, H, D = 3, 3, 5, 5, 5
Kw, Kh, Kd = 3, 3, 3
Cout = 3

dense_features = torch.zeros((B, W, H, D, Cin), device="cuda")

for i in range(W):
    dense_features[:, i, i, i, :] = torch.randn((Cin, ))

sparse_features = SparseConvTensor.from_dense(dense_features)
feats = sparse_features.features
coords = sparse_features.indices
shape = torch.Size((sparse_features.batch_size, feats.shape[-1], *sparse_features.spatial_shape))
weight = torch.randn((Cout, Kw, Kh, Kd, Cin), device="cuda")

subm_conv_out_feats, _ = sparse_submanifold_conv3d(
    feats, coords, shape, weight, Algorithm.IMPLICIT_GEMM
)
```