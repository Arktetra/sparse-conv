import torch
import triton
import triton.language as tl

def sparse_submanifold_conv_fwd_implicit_gemm_kernel(
    input_ptr,  # [M, Ci]
    weight_ptr, # [Co, V, Ci]
    bias_ptr,   # [Co]
    neighbor_ptr,   # [M, V], where V = Kw * Kh * Kd
    output_ptr,     # [M, Co]
    M, Ci, Co, V,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_Co: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, # Block size for K dimension (V * Ci),
):
    pid = tl.program_id(axis=0)
    num_pid_Co = tl.cdiv(Co, BLOCK_SIZE_Co)
    num_pid_K = tl.cdiv(V * Ci, BLOCK_SIZE_K)
    pid_Co = pid % num_pid_Co
    pid_M = pid // num_pid_Co

    offset_M = (pid_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_Co = (pid_Co * BLOCK_SIZE_Co  + tl.arange(0, BLOCK_SIZE_Co)) % Co
    offset_K = tl.arange(0, BLOCK_SIZE_K)
    weight_ptrs = weight_ptr + (offset_Co[None, :] * V * Ci + offset_K[:, None])      # [BLOCK_SIZE_K, BLOCK_SIZE_Co]
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_Co), dtype=tl.float32)

    for k in range(num_pid_K):
        # v = k // num_pid_K
        # bk = k % num_pid_k

        offset_neighbor_M = tl.load(neighbor_ptr + offset_M * V)
        input_ptrs = input_ptr + (offset_neighbor_M[:, None].to(tl.int64) * Ci + offset_K[None, :])
        neighbor_mask = offset_neighbor_M != 0xffffffff
        k_mask = offset_K < V * Ci - k * BLOCK_SIZE_K
        input_block = tl.load(input_ptrs, mask=neighbor_mask[:, None] & k_mask[None, :], other=0.0)
        weight_block = tl.load(weight_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator = tl.dot(input_block, weight_block, accumulator)

        weight_ptrs += BLOCK_SIZE_K
        input_ptrs += BLOCK_SIZE_K

    result = accumulator.to(input_ptr.type.element_ty)

    if bias is not None:
        bias_block = tl.load(bias_ptr + offset_Co)
        result += bias_block[None, :]

    output_offset_M = pid_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    output_offset_Co = pid_Co * BLOCK_SIZE_Co + tl.arange(0, BLOCK_SIZE_Co)
    output_ptrs = output_ptr + (output_offset_M[:, None] * Co + output_offset_Co[None, :])
    output_mask = (output_offset_M[:, None] < N) & (output_offset_Co[None, :] < Co)
    tl.store(output_ptrs, result, mask=output_mask)

def sparse_submanifold_conv_fwd_implicit_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    neighbor: torch.Tensor,
) -> torch.Tensor:
    assert input.shape[1] == weight.shape[2], "Incompatible dimensions for matrix multiplication."
    assert input.is_contiguous(), "input must be contiguous."
    assert weight.is_contiguous(), "weight must be contiguous."
    assert neighbor.is_contiguous(), "neighbor must be contiguous."
    M, Ci, Co, V = neighbor.shape[0], input.shape[-1], weight.shape[0]. weight.shape[1]

    output = torch.empty((M, Co), device=input.device, dtype=input.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_NEIGHBOR_M"])
        * triton.cdiv(Co, META["BLOCK_SIZE_OUTPUT_Co"])
    )

    sparse_submanifold_conv_fwd_implicit_gemm_kernel[grid](
        input, weight, bias, neighbor, output,
        M, Ci, Co, V,
    )

    return output
