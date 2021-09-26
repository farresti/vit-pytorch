"""
Copyright AriadNEXT, Inc - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""
from utils.activation import get_activation_fn
from typing import List
import torch.nn as nn
import torch



class MLP(nn.Module):
    """
     Very simple multi-layer perceptron (also called FFN)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 activation: str = 'relu',
                 dropout: float = 0.):
        """
            structure is: Input_layer -> hidder_layer_0 -> ... -> hidden_layer_$(num_layers - 2) -> Output_layer

        :param input_dim:   Input dimension of the first layer.
        :param hidden_dim:  Dimension of the hidden layers.
        :param output_dim:  Dimension of the output layers.
        :param num_layers:  Number of layers (including input and output ones).
        :param activation:  Activation function between hidden layers.
        :param dropout:     Dropout to be applied.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        layers: List[nn.Module] = [nn.Sequential(nn.Linear(n, k), nn.ReLU()) for n, k in zip([input_dim] + h[:-1], h)]
        layers.append(nn.Linear(h[-1], output_dim))
        self.layers = nn.ModuleList(layers)
        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers[:-1]:
            x = l(x)
            x = self.dropout(self.activation(x))
        x = self.dropout(self.layers[-1](x))
        return x
