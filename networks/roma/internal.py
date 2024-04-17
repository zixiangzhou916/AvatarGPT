# RoMa
# Copyright (c) 2020 NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use.
"""
Set of functions for internal module use.
"""

import torch

try:
    import torch_batch_svd
    _fast_gpu_svd = torch_batch_svd.svd
    _IS_TORCH_BATCH_SVD_AVAILABLE = True
except ModuleNotFoundError:
    # torch_batch_svd (https://github.com/KinglittleQ/torch-batch-svd) is not installed
    # and is required for maximum efficiency of special_procrustes using GPUs.
    # Using torch.svd as a fallback.
    _IS_TORCH_BATCH_SVD_AVAILABLE = False
    _fast_gpu_svd = torch.svd

def flatten_batch_dims(tensor, end_dim):
    """
    :meta private:
    Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.
    """
    batch_shape = tensor.shape[:end_dim+1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape

def unflatten_batch_dims(tensor, batch_shape):
    """
    :meta private:
    Revert flattening of a tensor.
    """
    # Note: alternative to tensor.unflatten(dim=0, sizes=batch_shape) that was not supported by PyTorch 1.6.0.
    return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.squeeze(0)

def _pseudo_inverse(x, eps):
    """
    :meta private:
    Element-wise pseudo inverse.
    """
    inv = 1.0/x
    inv[torch.abs(x) < eps] = 0.0
    return inv    

def svd(M):
    """
    Singular Value Decomposition wrapper, using efficient batch decomposition on GPU when available.

    Args:
        M (BxMxN tensor): batch of real matrices.
    Returns:
        (U,D,V) decomposition, such as :math:`M = U @ diag(D) @ V^T`.
    """
    if M.is_cuda and M.shape[1] < 32 and M.shape[2] < 32:
        return _fast_gpu_svd(M)
    else:
        return torch.svd(M)

# Batched eigenvalue decomposition.
# Recent version of PyTorch deprecated the use of torch.symeig.
try:
    torch.linalg.eigh
    def symeig_lower(A):
        """
        Batched eigenvalue decomposition. Only the lower part of the matrix is considered.
        """
        return torch.linalg.eigh(A, UPLO='L')
except (NameError, AttributeError):
    # Older PyTorch version
    def symeig_lower(A):
        """
        Batched eigenvalue decomposition. Only the lower part of the matrix is considered.
        """
        return torch.symeig(A, upper=False, eigenvectors=True)