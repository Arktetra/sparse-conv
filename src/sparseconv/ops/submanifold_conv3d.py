from typing import Optional, Tuple
import torch

from torch.autograd import Function

from sparseconv.kernels import cuda
from sparseconv.ops.utils import init_hashmap

class Algorithm:
    EXPLICIT_GEMM = "explicit_gemm"
    IMPLICIT_GEMM = "implicit_gemm"
    IMPLICIT_GEMM_SPLITK = "implicit_gemm_splitk"
    MASKED_IMPLICIT_GEMM = "masked_implicit_gemm"
    MASKED_IMPLICIT_GEMM_SPLITK = "masked_implicit_gemm_splitk"

class SubMConv3dNeighborCache:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

class SubMConv3dFunction(Function):
    @staticmethod
    def _compute_neighbor_cache(
        coords: torch.Tensor,
        shape: torch.Size,
        kernel_size: Tuple[int, int, int],
        dilation: Tuple[int, int, int],
        algorithm: Algorithm = Algorithm.EXPLICIT_GEMM,
        hashmap_ratio: float = 2.0
    ) -> SubMConv3dNeighborCache:
        assert coords.is_contiguous(), "Coords should be contiguous."
        assert coords.dtype == torch.int32, "Unsupported coords dtype. Expected int32."

        N, C, W, H, D = shape

        hashmap_keys, hashmap_values = init_hashmap(
            shape, int(hashmap_ratio * coords.shape[0]), coords.device
        )

        if algorithm in [Algorithm.EXPLICIT_GEMM, Algorithm.IMPLICIT_GEMM, Algorithm.IMPLICIT_GEMM_SPLITK]:
            if coords.is_cuda:
                neighbor_map = cuda.hashmap_build_submanifold_conv_neighbor_map_cuda_naive(
                    hashmap_keys, hashmap_values, coords,
                    W, H, D,
                    kernel_size[0], kernel_size[1], kernel_size[2],
                    dilation[0], dilation[1], dilation[2]
                )
            else:
                raise NotImplementedError("CPU version of hashmap is not implemented.")
            return SubMConv3dNeighborCache(**{
                "neighbor_map": neighbor_map,
            })
        else:
            raise ValueError(f"Unsupported Algorithm {algorithm}.")
        
    @staticmethod
    def _sparse_submanifold_conv_forward(
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        algorithm: Algorithm = Algorithm.EXPLICIT_GEMM,
    ) -> torch.Tensor:
        assert feats.is_contiguous(), "Input features should be contiguous."
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kw * Kh * Kd

        if algorithm == Algorithm.EXPLICIT_GEMM:
            im2col = torch.zeros((N * V, Ci), dtype=feats.dtype, device=feats.device)
            mask = neighbor_cache.neighbor_map.view(-1) != 0xffffffff
            im2col[mask] = feats[neighbor_cache.neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Ci)

            weights = weight.view(Co, -1).transpose(1, 0)
            if bias is not None:
                output = torch.addmm(bias, im2col, weights)
            else:
                output = torch.mm(im2col, weights)
        elif algorithm == Algorithm.IMPLICIT_GEMM:
            pass
        elif algorithm == Algorithm.IMPLICIT_GEMM_SPLITK:
            pass
        elif algorithm == Algorithm.MASKED_IMPLICIT_GEMM:
            pass
        elif algorithm == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            pass
        else:
            raise ValueError(f"Unsupported Algorithm {algorithm}.")
        
        return output
    
    @staticmethod
    def _sparse_submanifold_conv_backward(
        grad_output: torch.Tensor,
        feats: torch.Tensor,
        neighbor_cache: SubMConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        algorithm: Algorithm = Algorithm.EXPLICIT_GEMM
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        N = feats.shape[0]
        Co, Kw, Kh, Kd, Ci = weight.shape
        V = Kw * Kh * Kd

        if algorithm == Algorithm.EXPLICIT_GEMM:
            neighbor_map = neighbor_cache["neighbor_map"]

            if feats.requires_grad:
                im2col = torch.zeros((N * V, Co), dtype=feats.dtype, device=feats.device)
                inv_neighbor_map = torch.flip(neighbor_map, [1])
                mask = inv_neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Co)

                grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
            else:
                grad_input = None
            
            if weight.requires_grad:
                im2col = torch.zeros((N * V, Ci), dtype=weight.dtype, device=weight.device)
                mask = neighbor_map.view(-1) != 0xffffffff
                im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
                im2col = im2col.view(N, V * Ci)

                grad_weight = torch.mm(im2col.t(), grad_output.view(N, -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous().view(Co, Kw, Kh, Kd, Ci)
            else:
                grad_weight = None

            if bias is not None and bias.requires_grad:
                grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None

        elif algorithm == Algorithm.IMPLICIT_GEMM:
            pass
        elif algorithm == Algorithm.IMPLICIT_GEMM_SPLITK:
            pass
        elif algorithm == Algorithm.MASKED_IMPLICIT_GEMM:
            pass
        elif algorithm == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
            pass
        else:
            raise ValueError(f"Unsupported algorithm {algorithm}.")
        
        return grad_input, grad_weight, grad_bias
    
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size,
        neighbor_cache: Optional[SubMConv3dNeighborCache],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        dilation: Tuple[int, int, int] = (1, 1, 1),
        algorithm: Algorithm = Algorithm.EXPLICIT_GEMM
    ) -> Tuple[torch.Tensor, SubMConv3dNeighborCache]:
        Co, Kw, Kh, Kd, Ci = weight.shape
        assert feats.shape[-1] == Ci, \
            f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"
        
        if neighbor_cache is None:
            neighbor_cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, (Kw, Kh, Kd), dilation)

        output = SubMConv3dFunction._sparse_submanifold_conv_forward(feats, neighbor_cache, weight, bias, algorithm)

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache

        return output, neighbor_cache
    
    @staticmethod
    def backward(
        ctx, 
        grad_output: torch.Tensor,
        algorithm: Algorithm = Algorithm.EXPLICIT_GEMM
    ):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        grad_input, grad_weight, grad_bias = SubMConv3dFunction._sparse_submanifold_conv_backward(
            grad_output, feats, neighbor_cache, weight, bias, algorithm
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None

        return grad_input, None, None, None, grad_weight, grad_bias, None

def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    neighbor_cache: Optional[SubMConv3dNeighborCache] = None,
    dilation: Tuple[int, int, int] = (1, 1, 1),
    algorithm: Algorithm = Algorithm.EXPLICIT_GEMM
) -> Tuple[torch.Tensor, SubMConv3dNeighborCache]:
    """
    Sparse submanifold convolution for 3D input.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, 4] tensor of input coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (Optional[torch.Tensor], optional): [Co] tensor of biases. Defaults to None.
        neighbor_cache (Optional[SubMConv3dNeighborCache], optional): neighbor cache for forward. Defaults to None.
        dilation (Tuple[int, int, int], optional): dilation rate. Defaults to (1, 1, 1).
        algorithm (Algorithm, optional): algorithm for performing submanifold convolution. Defaults to Algorithm.EXPLICIT_GEMM.

    Returns:
        Tuple[torch.Tensor, SubMConv3dNeighborCache]:
            - output (torch.Tensor): [N, Co] tensor of output features.
            - neighbor_cache (SubMConv3dNeighborCache): neighbor cache for backward.
    """
    return SubMConv3dFunction.apply(feats, coords, shape, neighbor_cache, weight, bias, dilation, algorithm)