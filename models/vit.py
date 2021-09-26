"""
Copyright AriadNEXT, Inc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""
from utils.mlp import MLP
from typing import Tuple, Union
import torch.nn as nn
import torch


def to_tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return x, x
    return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 mlp_size: int,
                 activation: str = 'gelu',
                 dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.post_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = MLP(input_dim=hidden_dim,
                       hidden_dim=mlp_size,
                       output_dim=hidden_dim,
                       num_layers=2,
                       activation=activation,
                       dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_norm(x)
        x_attn = self.attention(x_norm, x_norm, x_norm)[0] + x
        x = self.mlp(self.post_norm(x_attn)) + x_attn
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 depth: int,
                 hidden_dim: int,
                 num_heads: int,
                 mlp_size: int,
                 activation: str = 'gelu',
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim=hidden_dim,
                         num_heads=num_heads,
                         mlp_size=mlp_size,
                         activation=activation,
                         dropout=dropout) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 num_classes: int,
                 input_size: Union[int, Tuple[int, int]] = 224,
                 in_channels: int = 3,
                 patch_size: Union[int, Tuple[int, int]] = 224,
                 hidden_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 depth: int = 12):
        """ Vanilla Visual-Transformer model from:

        https://openreview.net/pdf?id=YicbFdNTTy

        All default arguments are based on the ViT-Base configuration.

        :param input_size:  Tuple or int representing input size in pixel in format (H,W).
        :param in_channels: Number of input channels.
        :param patch_size:  Tuple or int representing patch sizes in pixel in format (Ph, Pw).
        :param hidden_dim:  Hidden dimension of the model.
        :param mlp_size:    Size of the hidden layers of the MLP blocks.
        :param num_heads:   Number of heads of the multi-head self attention block.
        :param depth:       Number of transformer encoder blocks.
        """
        super(VisionTransformer, self).__init__()
        self.input_size = to_tuple(input_size)
        if patch_size is None:
            self.patch_size = (self.input_size[0] / 16, self.input_size[1] / 16)
        else:
            self.patch_size = to_tuple(patch_size)
        assert (self.input_size[0] / self.patch_size[0]).is_integer(), 'height should be divisible by patch size 0'
        assert (self.input_size[1] / self.patch_size[1]).is_integer(), 'width should be divisible by patch size 1'
        self.in_channels = in_channels
        self.patch_dim = int(self.in_channels * self.patch_size[0] * self.patch_size[1])
        self.patch_count = int(self.input_size[0] // self.patch_size[0]) * int(self.input_size[1] // self.patch_size[1])
        # Linear projection of flattened patches
        self.input_projection = nn.Linear(in_features=self.patch_dim, out_features=hidden_dim)
        self.pos_embedding = nn.Parameter(torch.rand((1, self.patch_count + 1, hidden_dim)))
        self.cls_token = nn.Parameter(torch.rand((1, 1, hidden_dim)))
        # Transformer encoder
        self.transformer = TransformerEncoder(depth=depth,
                                              num_heads=num_heads,
                                              hidden_dim=hidden_dim,
                                              mlp_size=mlp_size)
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches: BxCxHxW -> NxHpxWpxPhxPwxC with (Ph,Pw) the patch sizes
        x = x.unfold(2, self.patch_size[0], self.patch_size[0])
        x = x.unfold(3, self.patch_size[1], self.patch_size[1])
        x = x.permute((0, 2, 3, 4, 5, 1))
        # Flatten to BxNxL with L = Ph*Pw*C and N = HpxWp
        x = x.flatten(start_dim=3).flatten(start_dim=1, end_dim=2)
        # Linear projection
        x = self.input_projection(x)
        # Add the class token + embedding
        x = torch.cat([self.cls_token, x], dim=1) + self.pos_embedding
        # Go through the transformer
        x = self.transformer(x)
        # Classification head
        y = self.mlp_head(x[:, 0, :])
        return y


def main():
    x = torch.rand((1, 3, 224, 224))
    vit = VisionTransformer(num_classes=1000,
                            input_size=(224, 224),
                            patch_size=(16, 16),
                            depth=12,
                            hidden_dim=768,
                            num_heads=12)
    y = vit(x)
    print(y.shape)


if __name__ == '__main__':
    main()
