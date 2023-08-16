





import torch
import torch.nn as nn


def PositionalEncoding(max_len, d_model, device):
    """
    compute sinusoid encoding.
    """
    """
    constructor of sinusoid encoding class

    :param d_model: dimension of model
    :param max_len: max sequence length
    :param device: hardware device setting
    """

    # same size with input matrix (for adding with input matrix)
    encoding = torch.zeros(max_len, d_model, device=device)
    encoding.requires_grad = False  # we don't need to compute gradient

    pos = torch.arange(0, max_len, device=device)
    pos = pos.float().unsqueeze(dim=1)
    # 1D => 2D unsqueeze to represent word's position

    _2i = torch.arange(0, d_model, step=2, device=device).float()
    # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
    # "step=2" means 'i' multiplied with two (same with 2 * i)

    encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
    encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))


    return  encoding