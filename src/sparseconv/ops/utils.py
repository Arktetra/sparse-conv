import torch

def init_hashmap(spatial_size, hashmap_size, device):
    N, C, W, H, D = spatial_size
    VOL = N * W * H * D

    if VOL < 2 ** 32:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2 ** 64:
        hashmap_keys = torch.full((hashmap_size,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Volumn {VOL} > 2^64.")
    
    hashmap_values = torch.empty((hashmap_size,), dtype=torch.uint32, device=device)

    return hashmap_keys, hashmap_values