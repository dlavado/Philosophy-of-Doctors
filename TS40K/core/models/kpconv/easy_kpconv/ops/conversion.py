from typing import Optional, Tuple

import torch
from torch import Tensor


def _get_indices_from_lengths(lengths: Tensor, num_items: int) -> Tensor:
    """Compute the indices in flattened batch tensor from the lengths in pack mode."""
    length_list = lengths.detach().cpu().numpy().tolist()
    chunks = [(i * num_items, i * num_items + length) for i, length in enumerate(length_list)]
    indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
    return indices

def batch_to_pack(batch_tensor: Tensor, masks: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Convert Tensor from batch mode to stack mode with masks.

    Args:
        batch_tensor (Tensor): the input tensor in batch mode (B, N, C) or (B, N).
        masks (BoolTensor): the masks of items of each sample in the batch (B, N).

    Returns:
        A Tensor in pack mode in the shape of (M, C) or (M).
        A LongTensor of the length of each sample in the batch in the shape of (B).
    """
    if masks is None:
        return batch_tensor.view(-1, batch_tensor.size(-1)), torch.full((batch_tensor.size(0),), batch_tensor.size(1), dtype=torch.long).to(batch_tensor.device)

    packed_tensor = batch_tensor[masks]
    lengths = masks.sum(dim=1).long()
    
    return packed_tensor, lengths


def pack_to_batch(pack_tensor: Tensor, lengths: Tensor, max_length=None, fill_value=0.0) -> Tuple[Tensor, Tensor]:
    """Convert Tensor from pack mode to batch mode.

    Args:
        pack_tensor (Tensor): The input tensors in pack mode (M, C).
        lengths (LongTensor): The number of items of each sample in the batch (B)
        max_length (int, optional): The maximal length of each sample in the batch.
        fill_value (float or int or bool): The default value in the empty regions. Default: 0.

    Returns:
        A Tensor in stack mode in the shape of (B, N, C), where N is max(lengths).
        A BoolTensor of the masks of each sample in the batch in the shape of (B, N).
    """
    batch_size = lengths.shape[0]
    if max_length is None:
        max_length = lengths.max().item()
    tgt_indices = _get_indices_from_lengths(lengths, max_length)

    num_channels = pack_tensor.shape[1]
    batch_tensor = pack_tensor.new_full(size=(batch_size * max_length, num_channels), fill_value=fill_value)
    batch_tensor[tgt_indices] = pack_tensor
    batch_tensor = batch_tensor.view(batch_size, max_length, num_channels)

    masks = torch.zeros(size=(batch_size * max_length,), dtype=torch.bool).cuda()
    masks[tgt_indices] = True
    masks = masks.view(batch_size, max_length)

    return batch_tensor, masks
