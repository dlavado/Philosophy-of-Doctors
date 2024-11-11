from typing import Optional, Tuple

import torch
from torch import Tensor


def _get_indices_from_lengths(lengths: Tensor, num_items: int) -> Tensor:
    """Compute the indices in flattened batch tensor from the lengths in pack mode."""
    length_list = lengths.detach().cpu().numpy().tolist()
    chunks = [(i * num_items, i * num_items + length) for i, length in enumerate(length_list)]
    indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
    return indices


# def batch_to_pack(batch_tensor: Tensor, masks: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
#     """Convert Tensor from batch mode to stack mode with masks.

#     Args:
#         batch_tensor (Tensor): the input tensor in batch mode (B, N, C) or (B, N).
#         masks (BoolTensor): the masks of items of each sample in the batch (B, N).

#     Returns:
#         A Tensor in pack mode in the shape of (M, C) or (M).
#         A LongTensor of the length of each sample in the batch in the shape of (B).
#     """
#     if masks is not None:
#         pack_tensor = batch_tensor[masks]
#         lengths = masks.sum(dim=1)
#     else:
#         lengths = torch.full(size=(batch_tensor.shape[0],), fill_value=batch_tensor.shape[1], dtype=torch.long).cuda()
#         pack_tensor = batch_tensor
#     return pack_tensor, lengths

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
        packed_tensor = batch_tensor.view(-1, batch_tensor.size(-1))
        lengths = torch.full(size=(batch_tensor.shape[0],), fill_value=batch_tensor.shape[1], dtype=torch.long).cuda()
        return packed_tensor, lengths

    packed_tensor = batch_tensor[masks]
    lengths = masks.sum(dim=1).long()
    
    return packed_tensor, lengths


def lengths_to_batchvector(lengths: Tensor) -> Tensor:
    """
        Convert LongTensor of lengths to a batch vector.

    Parameters:
    -----------
    lengths - LongTensor:
        shape (B,) with the number of items of each sample in the batch. 
    
    Returns:
    --------
    batch_vector - LongTensor:
        shape (B*N,) with the indices of the items in the batch.
    """
    if torch.unique(lengths).shape[0] == 1:
        batch_vector = torch.arange(lengths.shape[0]).unsqueeze(1).repeat(1, lengths.max()).view(-1)
    else:
        batch_vector = torch.cat([torch.full((lengths[i],), i, dtype=torch.long) for i in range(lengths.shape[0])])

    return batch_vector.to(lengths.device)


def pack_to_batch(pack_tensor: Tensor, lengths: Tensor, max_length=None, fill_value=-1.0) -> Tuple[Tensor, Tensor]:
    """Convert Tensor from pack mode to batch mode.

    Args:
        pack_tensor (Tensor): The input tensors in pack mode (M, C).
        lengths (LongTensor): The number of items of each sample in the batch (B)
        max_length (int, optional): The maximal length of each sample in the batch.
        fill_value (float or int or bool): The default value in the empty regions. Default: -1.0.

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



def _list_to_pack_tensor(list_tensor):
    """
    Parameters
    ----------
    
    list_tensor : list[torch.Tensor]
        list of tensors to pack of shape (N, ...)
        
    Returns
    -------
    packed_tensor : torch.Tensor
        tensor with the packed tensors. shape (B*N, ...)
        
    lengths : torch.Tensor
        tensor with the lengths of each sample in the batch. shape (B)
    """
    packed_tensor = torch.cat(list_tensor, dim=0)
    
    lengths = torch.zeros(len(list_tensor), dtype=torch.long)
    for i, t in enumerate(list_tensor):
        lengths[i] = t.shape[0]
    
    return packed_tensor, lengths

def _pack_to_list_tensor(packed_tensor, lengths):
    """
    Parameters
    ----------
    
    packed_tensor : torch.Tensor
        tensor with the packed tensors. shape (B*N, ...)
        
    lengths : torch.Tensor
        tensor with the lengths of each sample in the batch. shape (B)
        
    Returns
    -------
    list_tensor : list[torch.Tensor]
        list of tensors to pack of shape (N, ...)
    """
    list_tensor = []
    start = 0
    for l in lengths:
        list_tensor.append(packed_tensor[start:start+l])
        start += l
        
    return list_tensor

def _pack_tensor_to_batch(packed_tensor:torch.Tensor, lengths:torch.Tensor, pad_value=-1.0):
    """
    Parameters
    ----------
    
    packed_tensor : torch.Tensor
        tensor with the packed tensors. shape (B*N, ...), with a variable N for each sample
        
    lengths : torch.Tensor
        tensor with the lengths of each sample in the batch. shape (B)
        
    pad_value : float
        value to pad the tensors with
        
    Returns
    -------
    batch_tensor : torch.Tensor
        tensor with the batched tensors. shape (B, N, ...)
    """
    batch_tensor = []
    start = 0
    max_length = max(lengths)
    for l in lengths:
        sample = packed_tensor[start:start+l]
        if l < max_length:
            sample = torch.cat([sample, torch.full((max_length-l, *sample.shape[1:]), pad_value, device=packed_tensor.device, dtype=packed_tensor.dtype)], dim=0)
        batch_tensor.append(sample)
        start += l
        
    return torch.stack(batch_tensor, dim=0)
  