import torch

def lens2mask(lens):
    """Calculates masks for lengths
    Args:
        lens (list of int): bsize
    Returns:
        Tensor: masks of length, bsize x max_len
    """
    bsize = lens.numel()
    max_len = lens.max()
    masks = torch.arange(0, max_len) \
        .type_as(lens) \
        .to(lens.device) \
        .repeat(bsize, 1) \
        .lt(lens.unsqueeze(1))
    masks.requires_grad = False

    return masks

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    E.g. [1, 2, 3], count=2 ==> [1, 1, 2, 2, 3, 3]
        [[1, 2], [3, 4]], count=3, dim=1 ==> [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
    Different from torch.repeat
    """
    if x is None:
        return x
    elif type(x) in [list, tuple]:
        return type(x)([tile(each, count, dim) for each in x])
    else:
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.contiguous().view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x