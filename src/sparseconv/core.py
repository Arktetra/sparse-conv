import numpy as np
import torch

from torch.fx import ProxyableClassMeta
from typing import Union, List

SparseConvTensorMeta = ProxyableClassMeta

def scatter_nd(indices, src, shape):
    res = torch.zeros(*shape, dtype=src.dtype, device=src.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    res[tuple(slices)] = src.view(*output_shape)
    return res

class SparseConvTensor(metaclass=SparseConvTensorMeta):
    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: Union[List[int], np.ndarray],
        batch_size: int,
    ):
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(v) for v in spatial_shape]
        self.batch_size = batch_size

    def __repr__(self):
        return f"SparseConvTensor[shape={self.features.shape}]"
    
    def replace_feature(self, feature: torch.Tensor):
        new_spt = SparseConvTensor(
            feature,
            self.indices,
            self.spatial_shape,
            self.batch_size
        )
        return new_spt
    
    @property
    def features(self):
        return self._features
    
    @features.setter
    def features(self, val):
        raise ValueError(
            "Due to issues related to fx, you cannot set features directly."
            f"Use 'x = x.replace_feature(new_feature)'"
            "to generate new SparseConvTensor instead."
        )
    
    @classmethod
    def from_dense(cls, x: torch.Tensor):
        x_sp = x.to_sparse(x.ndim - 1)
        spatial_shape = x_sp.shape[1:-1]
        batch_size = x_sp.shape[0]
        indices = x_sp.indices().permute(1, 0).contiguous().int()
        features = x_sp.values()
        return cls(features, indices, spatial_shape, batch_size)
    
    def dense(self, channels_first: bool = True):
        output_shape = [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(
            self.indices.to(self.features.device).long(),
            self.features,
            output_shape
        )
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()