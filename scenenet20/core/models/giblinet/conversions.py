from typing import Optional, Tuple

import torch
from torch import Tensor


def _naive_get_indices_from_lengths(lengths: Tensor, num_items: int) -> Tensor:
    """Compute the indices in flattened batch tensor from the lengths in pack mode."""
    length_list = lengths.detach().cpu().numpy().tolist()
    chunks = [(i * num_items, i * num_items + length) for i, length in enumerate(length_list)]
    indices = torch.cat([torch.arange(x, y) for x, y in chunks], dim=0).cuda()
    return indices


@torch.jit.script
def _get_indices_from_lengths(lengths: torch.Tensor, num_items: int) -> torch.Tensor:
    """
    Compute the indices in a flattened batch tensor from the lengths in pack mode.

    Parameters
    ----------
    `lengths` - torch.Tensor:
        1D tensor containing the lengths for each batch element.

    `num_items` - int:
        The maximum number of items in the batch.

    Returns
    -------
    `indices` - torch.Tensor:
        1D tensor containing the indices in the flattened representation.
    """
    batch_size = lengths.size(0)
    max_len = torch.max(lengths)

    # Create a 2D index grid
    range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # Shape: (1, max_len)
    range_col = torch.arange(batch_size, device=lengths.device).unsqueeze(1)  # Shape: (batch_size, 1)

    # Expand lengths and compare to the index grid
    valid_mask = range_row < lengths.unsqueeze(1)  # Shape: (batch_size, max_len)

    # Flatten and filter indices
    flattened_indices = range_row.repeat(batch_size, 1)[valid_mask] + range_col.repeat(1, max_len)[valid_mask] * num_items

    return flattened_indices



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

@torch.jit.script
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
        lengths = torch.full(size=(batch_tensor.shape[0],), fill_value=batch_tensor.shape[1], dtype=torch.long, device=batch_tensor.device)
        return packed_tensor, lengths

    packed_tensor = batch_tensor[masks]
    lengths = masks.sum(dim=1).long()
    
    return packed_tensor, lengths

# def lengths_to_batchvector(lengths: Tensor) -> Tensor:
#     """
#         Convert LongTensor of lengths to a batch vector.

#     Parameters:
#     -----------
#     lengths - LongTensor:
#         shape (B,) with the number of items of each sample in the batch. 
    
#     Returns:
#     --------
#     batch_vector - LongTensor:
#         shape (B*N,) with the indices of the items in the batch.
#     """
#     if torch.unique(lengths).shape[0] == 1:
#         batch_vector = torch.arange(lengths.shape[0]).unsqueeze(1).repeat(1, lengths.max()).view(-1)
#     else:
#         batch_vector = torch.cat([torch.full((lengths[i],), i, dtype=torch.long) for i in range(lengths.shape[0])])

#     return batch_vector.to(lengths.device)


@torch.jit.script
def lengths_to_batchvector(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert LongTensor of lengths to a batch vector.

    Parameters:
    -----------
    lengths - torch.Tensor:
        shape (B,) with the number of items of each sample in the batch. 
    
    Returns:
    --------
    batch_vector - torch.Tensor:
        shape (sum(lengths),) with the batch indices corresponding to each item.
    """
    batch_size = lengths.size(0)
    max_length = lengths.max()

    # Create a grid of batch indices
    batch_indices = torch.arange(batch_size, device=lengths.device).unsqueeze(1).expand(-1, max_length)

    # Mask invalid entries beyond the length of each batch
    valid_mask = torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    # Apply the mask and flatten
    batch_vector = batch_indices[valid_mask]

    return batch_vector



def naive_pack_to_batch(pack_tensor: Tensor, lengths: Tensor, max_length=None, fill_value=-1.0) -> Tuple[Tensor, Tensor]:
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


@torch.jit.script
def pack_to_batch(
    pack_tensor: torch.Tensor, 
    lengths: torch.Tensor, 
    max_length: int = -1, 
    fill_value: float = -1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Tensor from pack mode to batch mode.

    Args:
        pack_tensor (Tensor): The input tensors in pack mode (M, C).
        lengths (Tensor): The number of items of each sample in the batch (B).
        max_length (int, optional): The maximal length of each sample in the batch. Default: -1 (calculated from lengths).
        fill_value (float): The default value in the empty regions. Default: -1.0.

    Returns:
        A Tensor in batch mode in the shape of (B, N, C), where N is max(lengths).
        A BoolTensor of the masks of each sample in the batch in the shape of (B, N).
    """
    batch_size = lengths.shape[0]
    if max_length < 0:
        max_length = torch.max(lengths).item()

    num_channels = pack_tensor.size(1)
    tgt_indices = _get_indices_from_lengths(lengths, max_length)

    batch_tensor = torch.full(
        (batch_size * max_length, num_channels),
        fill_value=fill_value,
        dtype=pack_tensor.dtype,
        device=pack_tensor.device,
    )
    batch_tensor[tgt_indices] = pack_tensor
    batch_tensor = batch_tensor.view(batch_size, max_length, num_channels).contiguous()

    # Create masks
    masks = torch.zeros(
        (batch_size * max_length,),
        dtype=torch.bool,
        device=pack_tensor.device,
    )
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


@torch.jit.script
def list_tensor_to_batch(list_tensor: list[torch.Tensor], pad_value: float = -1.0) -> torch.Tensor:
    """
    Convert a list of tensors to a single batch tensor with padding.

    Parameters
    ----------
    list_tensor : List[torch.Tensor]
        List of tensors to pack of shape (N, ...).
        
    pad_value : float
        Value to pad the tensors with.
        
    Returns
    -------
    batch_tensor : torch.Tensor
        Tensor with the batched tensors. Shape (B, N, ...).
    """
    max_length = max(t.size(0) for t in list_tensor)
    batch_size = len(list_tensor)
    extra_dims = list_tensor[0].size()[1:]  # Additional dimensions as a tuple

    batch_shape = (batch_size, max_length) + extra_dims

    batch_tensor = torch.full(
        batch_shape,
        pad_value,
        device=list_tensor[0].device,
        dtype=list_tensor[0].dtype,
    )

    for i, t in enumerate(list_tensor):
        batch_tensor[i, :t.size(0)] = t

    return batch_tensor


@torch.jit.script
def compute_centered_support_points(points: torch.Tensor, q_points: torch.Tensor, support_idxs: torch.Tensor):
    """
    Compute the centered support points for each query point.

    Parameters
    ----------
    `points` - torch.Tensor:
        Tensor of shape (B, N, 3) representing the point cloud.

    `q_points` - torch.Tensor:
        Tensor of shape (B, M, 3) representing the query points.

    `support_idxs` - torch.Tensor:
        Tensor of shape (B, M, K) representing the indices of the support points for each query point.

    Returns
    -------
    `s_centered` - torch.Tensor:
        Tensor of shape (B, M, K, 3) representing the centered support points for each query point.

    `valid_mask` - torch.Tensor:
        Tensor of shape (B, M, K) representing a mask where valid points are marked as `True` and invalid (`-1`) points as `False`.

    """
    if points.dim() == 2:
        # If unbatched, add a batch dimension
        points = points.unsqueeze(0)
        q_points = q_points.unsqueeze(0)
        support_idxs = support_idxs.unsqueeze(0)
        batched = False
    else:
        batched = True

    B, M, K = support_idxs.shape
    F = points.shape[-1]
    
    # print(f'points: {points.shape}, q_points: {q_points.shape}, support_idxs: {support_idxs.shape}')
    # print(f'points: {points.dtype}, q_points: {q_points.dtype}, support_idxs: {support_idxs.dtype}')

    valid_mask = support_idxs != -1
    # Flatten support indices to gather them in one step
    #flat_supports_idxs = support_idxs.reshape(B, -1).to(torch.long)  # (B, M*K)
    flat_supports_idxs = torch.where(valid_mask, support_idxs, 0).reshape(B, -1).to(torch.long)  # (B, M*K)
    # flat_supports_idxs = torch.where(flat_supports_idxs == -1, torch.zeros_like(flat_supports_idxs), flat_supports_idxs)

    # Gather the support points (B, M*K, F)
    support_points = torch.gather(points, 1, flat_supports_idxs.unsqueeze(-1).expand(-1, -1, F)).to(points.dtype)
    # Reshape back to (B, M, K, 3)
    support_points = support_points.reshape(B, M, K, F).contiguous()

    # Center the support points: (B, M, K, 3) - (B, M, 1, 3)
    s_centered = support_points - q_points.unsqueeze(2)  # (B, M, K, 3)

    return s_centered.contiguous(), valid_mask, batched





############### Point Transformer Utils ####################
"""
Usually, a point cloud is represented as a tensor of shape (B, N, F), 
where B is the batch size, N is the number of points, and F is the number of features per point.

In this setting, point clouds are packed into a single tensor of shape (B*N, F);
this way, point clouds of different sizes can be processed in parallel and ops are more efficient.

To support this, auxiliary tensors are used to keep track of the offsets of each point cloud in the packed tensor.
For example, if the lengths of the point clouds are `[3, 4, 2]`, the `offset_vector` would be `[3, 7, 9]`.
In  turn, the `batch_vector` would be `[0, 0, 0, 1, 1, 1, 1, 2, 2]`.

The following functions provide utilities to convert between these representations.
"""


def get_batch_vector(batched_tensor:torch.Tensor):
    """
    Parameters
    ----------
    
    `batched_tensor` : torch.Tensor
        tensor in batch mode. shape (B, N, ...)
        
    Returns
    -------
    `batch_vector` : torch.Tensor
        tensor with the batch indices corresponding to each item. shape (B*N)
    """
    return torch.arange(batched_tensor.size(0)).unsqueeze(1).repeat(1, batched_tensor.size(1)).view(-1).to(batched_tensor.device)


def get_offset_vector(batched_tensor:torch.Tensor):
    """
    Get packed tensor offsets from batched tensor.
    
    Parameters
    ----------
    
    `batched_tensor` : torch.Tensor
        tensor in batch mode. shape (B, N, ...)
        
    Returns
    -------
    `offset_vector` : torch.Tensor
        tensor with the offset indices corresponding to each item. shape (B)
        For example, if the lengths of the point clouds are `[3, 4, 2]`, the `offset_vector` would be `[3, 7, 9]`.
    """
    valid_mask = (batched_tensor != 0).any(dim=-1)  # Shape: (B, N)
    lengths = valid_mask.sum(dim=-1)  # Shape: (B)
    offset_vector = torch.cumsum(lengths, dim=0)
    return offset_vector
    
import torch

def build_batch_tensor(packed_tensor: torch.Tensor, offset_vector: torch.Tensor, offset_indices=None, pad_value=-1):
    """
    Build a batched tensor from a packed tensor and its offset vector.
    
    Parameters
    ----------
    `packed_tensor` : torch.Tensor
        Tensor in packed mode. Shape (B*N, ...).
        
    `offset_vector` : torch.Tensor
        Tensor with the offset indices corresponding to each batch. Shape (B).
        
    `offset_indices` : torch.Tensor
        Tensor with the offset of each batch in the packed tensor. Shape (B).
        
    `pad_value` : int or float
        The value used to pad the tensor for batches with fewer elements than the maximum.
        
    Returns
    -------
    `batch_tensor` : torch.Tensor
        Tensor in batch mode. Shape (B, N, ...).
    """
    
    if offset_indices is not None:
        # if the packed tensor is made up of indices that need to converted to their 'un-offset' values; e.g. knn indices of a packed tensor
        batch_offsets = torch.cat([torch.tensor([0], device=offset_indices.device), offset_indices[:-1]])
        batch_offsets = batch_offsets.repeat_interleave(torch.diff(torch.cat([torch.tensor([0], device=offset_vector.device), offset_vector])))
        packed_tensor -= batch_offsets.unsqueeze(1)
        
    
    lengths = torch.diff(torch.cat([torch.tensor([0], device=offset_vector.device), offset_vector]))  # Shape (B,)
    max_length = lengths.max().item()
    
    batch_tensor = torch.full((len(lengths), max_length, *packed_tensor.shape[1:]), \
                            pad_value, dtype=packed_tensor.dtype, device=packed_tensor.device)
    
    # Generate index ranges for each batch
    indices = torch.arange(max_length, device=offset_vector.device).unsqueeze(0)  # Shape (1, max_length)
    mask = indices < lengths.unsqueeze(1)  # Shape (B, max_length)
    
    # Flatten packed_tensor into the batch tensor
    start_indices = torch.cat([torch.tensor([0], device=offset_vector.device), offset_vector[:-1]])
    
    flat_indices = torch.cat([torch.arange(start, end, device=packed_tensor.device) 
                              for start, end in zip(start_indices, offset_vector)])
    
    batch_tensor[mask] = packed_tensor[flat_indices]
    

    return batch_tensor


def build_batch_tensor_autograd(packed_tensor: torch.Tensor, offset_vector: torch.Tensor, offset_indices=None, pad_value=-1):
    """
    Build a batched tensor from a packed tensor and its offset vector, ensuring compatibility with autograd.
    
    Parameters
    ----------
    `packed_tensor` : torch.Tensor
        Tensor in packed mode. Shape (B*N, ...).
        
    `offset_vector` : torch.Tensor
        Tensor with the offset indices corresponding to each batch. Shape (B).
        
    `offset_indices` : torch.Tensor, optional
        Tensor with the offset of each batch in the packed tensor. Shape (B).
        
    `pad_value` : int or float
        The value used to pad the tensor for batches with fewer elements than the maximum.
        
    Returns
    -------
    `batch_tensor` : torch.Tensor
        Tensor in batch mode. Shape (B, N, ...).
    """
    # Handle offsets if `offset_indices` is provided
    if offset_indices is not None:
        batch_offsets = torch.cat([torch.zeros(1, device=offset_indices.device), offset_indices[:-1]])
        batch_offsets = batch_offsets.repeat_interleave(torch.diff(torch.cat([torch.zeros(1, device=offset_vector.device), offset_vector])))
        packed_tensor = packed_tensor - batch_offsets.unsqueeze(1)
    
    lengths = torch.diff(torch.cat([torch.zeros(1, device=offset_vector.device), offset_vector]))  # Shape (B,)
    max_length = lengths.max().item()
    
    # Initialize the batch tensor with padding values
    batch_tensor = torch.full(
        (len(lengths), int(max_length), *packed_tensor.shape[1:]),
        pad_value,
        dtype=packed_tensor.dtype,
        device=packed_tensor.device,
    )
    
    # Generate masks and index ranges
    indices = torch.arange(max_length, device=offset_vector.device).unsqueeze(0)  # Shape (1, max_length)
    mask = indices < lengths.unsqueeze(1)  # Shape (B, max_length)
    
    # Compute flat indices for packed_tensor
    start_indices = torch.cat([torch.zeros(1, device=offset_vector.device), offset_vector[:-1]])
    flat_indices = torch.cat([torch.arange(start, end, device=packed_tensor.device) 
                              for start, end in zip(start_indices, offset_vector)])
    
    # Use the mask to scatter packed_tensor into batch_tensor
    batch_tensor = batch_tensor.scatter(1, mask.nonzero(as_tuple=True)[1].unsqueeze(-1), packed_tensor[flat_indices.to(torch.long)].unsqueeze(-1))
    
    return batch_tensor
    
    
    