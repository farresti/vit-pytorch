from utils.activation import get_activation_fn
from typing import List, Tuple
import torch.nn as nn
import torch


class Conv2dNorm(nn.Module):
    """
        2D Convolution followed by an activation layer and a 2D BatchNorm layer.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int],
                 activation: str = "gelu",
                 **kwargs):
        """

        :param in_channels:   Number of input channels.
        :param out_channels:  Number of output channels.
        :param kernel_size:   Kernel size of the layer.
        :param activation:    Activation function to be used. (default: nn.GELU)
        :param kwargs:        Extra parameters to be passed to nn.Conv2d. For more details:
                              https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
        """
        super(Conv2dNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              **kwargs)
        self.activation = get_activation_fn(activation)
        self.norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class CMTStem(nn.Module):
    """
        CMT Stem module as described in:
        CMT: Convolutional Neural Networks Meet Vision Transformers
        https://arxiv.org/pdf/2107.06263.pdf
    """

    def __init__(self, in_channels: int, out_channels: int, activation: str = "gelu"):
        """
        :param in_channels:   Number of input channels.
        :param out_channels:  Number of output channels.
        :param activation:    Activation function to be used. (default: nn.GELU)
        """
        super(CMTStem, self).__init__()
        self.conv1 = Conv2dNorm(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                activation=activation,
                                stride=(2, 2),
                                padding=(1, 1))
        self.conv2 = Conv2dNorm(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                activation=activation)
        self.conv3 = Conv2dNorm(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LocalPerceptionUnit(nn.Module):
    """
        Local Perception Unit module as described in:
        CMT: Convolutional Neural Networks Meet Vision Transformers
        https://arxiv.org/pdf/2107.06263.pdf
    """

    def __init__(self, channels: int):
        """
        :param channels: Number of channels.
        """
        super(LocalPerceptionUnit, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels=channels,
                                 out_channels=channels,
                                 groups=channels,
                                 kernel_size=(3, 3),
                                 padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dw_conv(x)
        return x


class LightWeightMultiHeadSelfAttention(nn.Module):
    """
        Lightweight Multi-head Self-attention module as described in:
        CMT: Convolutional Neural Networks Meet Vision Transformers
        https://arxiv.org/pdf/2107.06263.pdf

        Inputs:
            Q: [N, C, H, W]
            K: [N, C, H / reduction_rate, W / reduction_rate]
            V: [N, C, H / reduction_rate, W / reduction_rate]
        Outputs:
            X: [N, C, H, W]
    """

    def __init__(self, input_size: int, channels: int, heads_count: int, reduction_rate: int):
        """
        :param input_size:      2D input size (H == W).
        :param channels:        Number of input channels.
        :param heads_count:     Number of heads.
        :param reduction_rate:  Reduction rate to be applied on Key/Value.
        """
        super(LightWeightMultiHeadSelfAttention, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.d_head = channels // heads_count
        self.heads_count = heads_count
        self.flat_size = self.input_size * self.input_size
        self.scaled_factor = self.d_head ** -0.5
        self.k_dw_conv = nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels,
                                   kernel_size=(reduction_rate, reduction_rate),
                                   stride=(reduction_rate, reduction_rate), padding=1, bias=True)
        self.v_dw_conv = nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels,
                                   kernel_size=(reduction_rate, reduction_rate),
                                   stride=(reduction_rate, reduction_rate), padding=1, bias=True)
        self.fc_q = nn.Linear(channels, self.d_head * self.heads_count)
        self.fc_k = nn.Linear(channels, self.d_head * self.heads_count)
        self.fc_v = nn.Linear(channels, self.d_head * self.heads_count)
        self.fc_o = nn.Linear(self.d_head * self.heads_count, channels)

        size = int((input_size - reduction_rate + 2) / reduction_rate + 1)
        self.bias = nn.Parameter(
            torch.Tensor(1, self.heads_count, input_size ** 2, size ** 2), requires_grad=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshape = x.view(batch_size, self.channels, self.flat_size).permute(0, 2, 1)
        # queries
        q = self.fc_q(x_reshape)
        q = q.view(batch_size, self.flat_size, self.heads_count, self.d_head).permute(0, 2, 1, 3).contiguous()
        # keys
        k = self.k_dw_conv(x)
        k = k.view(batch_size, self.channels, -1).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(batch_size, -1, self.heads_count, self.d_head).permute(0, 2, 1, 3).contiguous()
        # values
        v = self.v_dw_conv(x)
        v = v.view(batch_size, self.channels, -1).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(batch_size, -1, self.heads_count, self.d_head).permute(0, 2, 1, 3).contiguous()
        # Lightweight multi head self attention
        attention = torch.matmul(q, k.transpose(2, 3))
        attention = attention * self.scaled_factor + self.bias
        attention = torch.matmul(torch.softmax(attention, dim=-1), v).permute(0, 2, 1, 3)
        attention = attention.contiguous().view(batch_size, self.flat_size, self.heads_count * self.d_head)
        attention = self.fc_o(attention).view(batch_size, self.channels, self.input_size, self.input_size)
        return attention


class InvertedResidualFFN(nn.Module):
    """
       Inverted Residual Feed-forward Network module as described in:
        CMT: Convolutional Neural Networks Meet Vision Transformers
        https://arxiv.org/pdf/2107.06263.pdf
    """

    def __init__(self, channels: int, expansion_ratio: float, activation: str = "gelu"):
        """
        :param channels:        Number of input channels.
        :param expansion_ratio: Expansion ratio of the layer.
        :param activation:      Activation function to be used. (default: nn.GELU)
        """
        super(InvertedResidualFFN, self).__init__()
        # Expansion layer
        exp_channels = int(expansion_ratio * channels)
        self.conv1 = Conv2dNorm(in_channels=channels,
                                out_channels=exp_channels,
                                kernel_size=(1, 1),
                                activation=activation)
        # Depth wise convolution
        self.dw_conv = nn.Conv2d(in_channels=exp_channels,
                                 out_channels=exp_channels,
                                 kernel_size=(3, 3),
                                 groups=exp_channels,
                                 padding=1)
        # Residual connection
        self.activation = get_activation_fn(activation)
        self.bn_res = nn.BatchNorm2d(num_features=exp_channels)
        # projection layer
        self.conv2 = Conv2dNorm(in_channels=exp_channels,
                                out_channels=channels,
                                kernel_size=(1, 1),
                                activation="identity")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expansion layer
        x = self.conv1(x)
        # depth wise
        y = self.dw_conv(x)
        y = self.activation(y + x)
        y = self.bn_res(y)
        # projection layer
        x = self.conv2(y)
        return x


class CMTBlock(nn.Module):
    """
        CMT Block module as described in:
        CMT: Convolutional Neural Networks Meet Vision Transformers
        https://arxiv.org/pdf/2107.06263.pdf
    """

    def __init__(self,
                 input_shape: List[int],
                 expansion_ratio: float,
                 heads_count: int,
                 reduction_rate: int,
                 activation: str = "gelu"):
        """
        :param input_shape:      Shape of inputs in format [C, H, W]
        :param expansion_ratio:  Expansion ratio of the block (see paper for more details).
        :param heads_count:      Number of heads in the Multi-Head Self Attention block.
        :param reduction_rate:   Reduction rate in the Multi-Head Self Attention block.
        :param activation:       Activation function to be used. (default: nn.GELU)
        """
        super(CMTBlock, self).__init__()
        in_channels = input_shape[0]
        self.input_shape = input_shape
        self.lpu = LocalPerceptionUnit(channels=in_channels)
        self.ln1 = nn.LayerNorm(input_shape[1:], elementwise_affine=True)
        self.lmhsa = LightWeightMultiHeadSelfAttention(input_size=input_shape[1],
                                                       channels=in_channels,
                                                       heads_count=heads_count,
                                                       reduction_rate=reduction_rate)
        self.ln2 = nn.LayerNorm(input_shape[1:], elementwise_affine=True)
        self.irffn = InvertedResidualFFN(channels=in_channels,
                                         expansion_ratio=expansion_ratio,
                                         activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local Perception Unit
        x_lpu = self.lpu(x)
        # Layer Norm
        x = self.ln1(x_lpu)
        # Lightweight Multi Head Self Attention
        x = self.lmhsa(x)
        x_mhsa = x + x_lpu
        # Layer Norm
        x = self.ln2(x_mhsa)
        # Inverted Residual FFN
        x = self.irffn(x)
        x = x + x_mhsa
        return x


class PatchAggregate(nn.Module):
    """
        Patch aggregation layer module as described in:
        CMT: Convolutional Neural Networks Meet Vision Transformers
        https://arxiv.org/pdf/2107.06263.pdf
    """

    def __init__(self, input_shape: List[int], in_channels: int, out_channels: int):
        """
        :param input_shape:  Shape of inputs in format [C, H, W]
        :param in_channels:  Number of input channels.
        :param out_channels: Number of output channels.
        """
        super(PatchAggregate, self).__init__()
        self.input_shape = input_shape
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.ln = nn.LayerNorm(normalized_shape=input_shape[1:], elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.ln(x)
        return x


class CMT(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 stem_channels: int,
                 expansion_ratio: float,
                 stage_channels: List[int],
                 stage_blocks: List[int],
                 in_channels: int = 3,
                 activation: str = "gelu"):
        super(CMT, self).__init__()
        self.sizes = [input_size // 4, input_size // 8, input_size // 16, input_size // 32]
        self.stem = CMTStem(in_channels=in_channels,
                            out_channels=stem_channels,
                            activation=activation)
        input_shape = [stage_channels[0], self.sizes[0], self.sizes[0]]
        # patch aggregation channels = stage_channels[0]
        self.patch1 = PatchAggregate(input_shape=input_shape,
                                     in_channels=stem_channels,
                                     out_channels=stage_channels[0])
        # stage 1 channels = stage_channels[0]
        self.stage1 = nn.Sequential(*[
            CMTBlock(input_shape=input_shape,
                     expansion_ratio=expansion_ratio,
                     heads_count=1,
                     reduction_rate=8,
                     activation=activation) for _ in range(stage_blocks[0])
        ])
        input_shape = [stage_channels[1], self.sizes[1], self.sizes[1]]
        # patch aggregation channels = stage_channels[1]
        self.patch2 = PatchAggregate(input_shape=input_shape,
                                     in_channels=stage_channels[0],
                                     out_channels=stage_channels[1])
        # stage 2 channels = stage_channels[1]
        self.stage2 = nn.Sequential(*[
            CMTBlock(input_shape=input_shape,
                     expansion_ratio=expansion_ratio,
                     heads_count=2,
                     reduction_rate=4,
                     activation=activation) for _ in range(stage_blocks[1])
        ])
        input_shape = [stage_channels[2], self.sizes[2], self.sizes[2]]
        # patch aggregation channels = stage_channels[2]
        self.patch3 = PatchAggregate(input_shape=input_shape,
                                     in_channels=stage_channels[1],
                                     out_channels=stage_channels[2])
        # stage 3 channels = stage_channels[2]
        self.stage3 = nn.Sequential(*[
            CMTBlock(input_shape=input_shape,
                     expansion_ratio=expansion_ratio,
                     heads_count=4,
                     reduction_rate=2,
                     activation=activation) for _ in range(stage_blocks[2])
        ])
        input_shape = [stage_channels[3], self.sizes[3], self.sizes[3]]
        # patch aggregation channels = stage_channels[3]
        self.patch4 = PatchAggregate(input_shape=input_shape,
                                     in_channels=stage_channels[2],
                                     out_channels=stage_channels[3])
        # stage 4 channels = stage_channels[3]
        self.stage4 = nn.Sequential(*[
            CMTBlock(input_shape=input_shape,
                     expansion_ratio=expansion_ratio,
                     heads_count=8,
                     reduction_rate=1,
                     activation=activation) for _ in range(stage_blocks[3])
        ])
        # avg pool
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1))
        # features + classification
        self.fc = nn.Linear(in_features=stage_channels[3], out_features=1280)
        self.classifier = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        x = self.embeddings(x)
        # classifier
        x = self.classifier(x)
        return x

    def endpoints(self, x: torch.Tensor) -> List[torch.Tensor]:
        stages = []
        # CMT Stem
        x = self.stem(x)
        # Conv 2x2 stride=2
        x = self.patch1(x)
        # Stage 1
        x = self.stage1(x)
        stages.append(x.clone())
        # Conv 2x2 stride=2
        x = self.patch2(x)
        # Stage 2
        x = self.stage2(x)
        stages.append(x.clone())
        # Conv 2x2 stride=2
        x = self.patch3(x)
        # Stage 3
        x = self.stage3(x)
        stages.append(x.clone())
        # Conv 2x2 stride=2
        x = self.patch4(x)
        # Stage 4
        x = self.stage4(x)
        stages.append(x)
        return stages

    def embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # CMT Stem
        x = self.stem(x)
        # Conv 2x2 stride=2
        x = self.patch1(x)
        # Stage 1
        x = self.stage1(x)
        # Conv 2x2 stride=2
        x = self.patch2(x)
        # Stage 2
        x = self.stage2(x)
        # Conv 2x2 stride=2
        x = self.patch3(x)
        # Stage 3
        x = self.stage3(x)
        # Conv 2x2 stride=2
        x = self.patch4(x)
        # Stage 4
        x = self.stage4(x)
        # AVG pool
        x = self.avg_pool(x)
        x = self.fc(x)
        return x


def cmt_configs(type: str):
    if type == 'CMT-Ti':
        return {
            'stem_channels': 16,
            'expansion_ratio': 3.6,
            'stage_channels': [46, 92, 184, 368],
            'stage_blocks': [2, 2, 10, 2],
            'in_channels': 3,
        }
    elif type == 'CMT-XS':
        return {
            'stem_channels': 16,
            'expansion_ratio': 3.8,
            'stage_channels': [52, 104, 208, 416],
            'stage_blocks': [3, 3, 12, 3],
            'in_channels': 3,
        }
    elif type == 'CMT-S':
        return {
            'stem_channels': 32,
            'expansion_ratio': 4,
            'stage_channels': [64, 128, 256, 512],
            'stage_blocks': [3, 3, 16, 3],
            'in_channels': 3,
        }
    elif type == 'CMT-B':
        return {
            'stem_channels': 38,
            'expansion_ratio': 4,
            'stage_channels': [76, 152, 304, 608],
            'stage_blocks': [4, 4, 20, 4],
            'in_channels': 3,
        }
    else:
        raise ValueError(f'{type} is not a valid configuration.')


def main():
    x = torch.rand((1, 3, 224, 224))
    cmt = CMT(num_classes=1000,
              input_size=224,
              **cmt_configs('CMT-Ti'))
    y = cmt(x)
    print(y.shape)


if __name__ == '__main__':
    main()
