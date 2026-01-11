import torch

from sparseconv.core import SparseConvTensor


class TestSparseConvTensor:
    def test_conversion_one(self):
        x = torch.zeros((1, 255, 255, 3))
        x_sp = SparseConvTensor.from_dense(x)
        x_d = x_sp.dense()
        assert (x_d.permute(0, 3, 2, 1) == x).all() == True

    def test_conversion_two(self):
        x = torch.zeros((1, 255, 255, 3))
        x_sp = SparseConvTensor.from_dense(x)
        x_d = x_sp.dense(channels_first=False)
        assert (x_d == x).all() == True
