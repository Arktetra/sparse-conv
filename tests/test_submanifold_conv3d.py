import torch
import torch.nn as nn

from sparseconv.ops.submanifold_conv3d import Algorithm, sparse_submanifold_conv3d
from sparseconv.core import SparseConvTensor

class TestSubMConv3d:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, Cin, W, H, D = 3, 3, 5, 5, 5
    Kw, Kh, Kd = 3, 3, 3
    Cout = 3
    dense_features = torch.zeros((B, W, H, D, Cin)).to(device)

    for i in range(W):
        dense_features[:, i, i, i, :] = torch.randn((Cin, ))

    sparse_features = SparseConvTensor.from_dense(dense_features)
    feats = sparse_features.features
    coords = sparse_features.indices
    shape = torch.Size((sparse_features.batch_size, feats.shape[-1], *sparse_features.spatial_shape))
    weight = torch.randn((Cout, Kw, Kh, Kd, Cin)).to(device)
    mask = dense_features != 0

    conv3d = nn.Conv3d(Cin, Cout, (Kw, Kh, Kd), padding=1, bias=False).to(device)
    conv3d.weight = nn.Parameter(weight.permute(0, -1, 1, 2, 3))

    def test_explicit_gemm_no_bias(self):
        conv_out_feats = self.conv3d(self.dense_features.permute(0, -1, 1, 2, 3))
        subm_conv_out_feats, _ = sparse_submanifold_conv3d(
            self.feats, self.coords, self.shape, self.weight
        )

        assert torch.allclose(conv_out_feats.permute(0, 2, 3, 4, 1)[self.mask], subm_conv_out_feats.view(-1))

    def test_implicit_gemm_no_bias(self):
        conv_out_feats = self.conv3d(self.dense_features.permute(0, -1, 1, 2, 3))
        subm_conv_out_feats, _ = sparse_submanifold_conv3d(
            self.feats, self.coords, self.shape, self.weight, algorithm=Algorithm.IMPLICIT_GEMM
        )
        assert torch.allclose(
            conv_out_feats.permute(0, 2, 3, 4, 1)[self.mask], subm_conv_out_feats.view(-1),
            rtol=1e-2
        )


