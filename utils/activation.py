import torch


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "gelu":
        return torch.nn.GELU()
    if activation == "glu":
        return torch.nn.GLU()
    if activation == "identity":
        return torch.nn.Identity()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
