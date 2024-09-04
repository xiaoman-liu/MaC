"""
Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
import numpy as np


@dataclass
class ModelArgs:
    d_model: int # 隐藏层维度大小
    n_layer: int #
    vocab_size: int
    d_state: int = 16 # 状态维度，对应论文中的N
    expand: int = 2 #扩张系数，对应论文中的E
    dt_rank: Union[int, str] = 'auto' # Δ的秩，对应论文中的∆
    d_conv: int = 4 #
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    output_dim: int = 1
    input_dim: int = 1
    embedding_dim: int = 1

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model) # 内部维度，对应论文中的D

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16) # 对应论文中的∆

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba1(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args

        self.embedding = CharEmbedding(args)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        # self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
        # See "Weight Tying" paper
        # self.lm_head = nn.Linear(args.d_model, args.output_dim, bias=False)
        self.input_layer = nn.Linear(1, args.d_model)
        self.output_layer = nn.Linear(args.input_dim * args.d_model, args.output_dim)
        self.d_model = args.d_model
        self.conv_block = ConvBlock(args.d_model)

        ## new



    def forward(self, input_ids, char_embed=None):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        ## cnn feature extractor

        ## mamba
        # x = self.embedding(input_ids)
        ## todo

        if char_embed is not None:
            char_embed = self.embedding(char_embed)
            input_ids = torch.cat([input_ids, char_embed], dim=1)

        x = self.input_layer(input_ids)

        x = self.conv_block(x)


        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)


        # logits = self.lm_head(x)
        x = x.view(-1, self.args.input_dim * self.d_model)
        logits = self.output_layer(x)

        ## output layer


        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba1(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) + x

        return output




class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n],
                                    dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNormold(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x / rms * self.weight + self.bias

        return output





class ConvBlock(nn.Module):
    def __init__(self, feature_maps):
        super(ConvBlock, self).__init__()
        # self.upsample = nn.Upsample(size=64, mode='linear', align_corners=False)

        self.conv1 = nn.Conv1d(in_channels=feature_maps, out_channels=feature_maps, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(feature_maps)

        self.conv2 = nn.Conv1d(in_channels=feature_maps, out_channels=feature_maps, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(feature_maps)

        self.conv3 = nn.Conv1d(in_channels=feature_maps, out_channels=feature_maps, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(feature_maps)

        self.shortcut1 = nn.Conv1d(in_channels=feature_maps, out_channels=feature_maps, kernel_size=1, padding=0)
        self.bn_shortcut1 = nn.BatchNorm1d(feature_maps)

        self.conv4 = nn.Conv1d(in_channels=feature_maps, out_channels=feature_maps * 2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(feature_maps * 2)

        self.conv5 = nn.Conv1d(in_channels=feature_maps * 2, out_channels=feature_maps * 2, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(feature_maps * 2)

        self.conv6 = nn.Conv1d(in_channels=feature_maps * 2, out_channels=feature_maps * 2, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(feature_maps * 2)

        self.shortcut2 = nn.Conv1d(in_channels=feature_maps, out_channels=feature_maps * 2, kernel_size=1, padding=0)
        self.bn_shortcut2 = nn.BatchNorm1d(feature_maps * 2)

        self.conv7 = nn.Conv1d(in_channels=feature_maps * 2, out_channels=feature_maps * 2, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(feature_maps * 2)

        self.conv8 = nn.Conv1d(in_channels=feature_maps * 2, out_channels=feature_maps * 2, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(feature_maps * 2)

        self.conv9 = nn.Conv1d(in_channels=feature_maps * 2, out_channels=feature_maps * 2, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(feature_maps * 2)

        self.shortcut3 = nn.Conv1d(in_channels=feature_maps * 2, out_channels=feature_maps * 2, kernel_size=1, padding=0)
        self.bn_shortcut3 = nn.BatchNorm1d(feature_maps * 2)

        self.conv10 = nn.Conv1d(in_channels=feature_maps * 2, out_channels=feature_maps, kernel_size=1, padding=0)
        self.bn10 = nn.BatchNorm1d(feature_maps)
        self.apply(self.nn_init)

    def nn_init(self, layer):
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Upsample the input to length 64

        # x = self.upsample(x)

        # Block 1
        x = x.permute(0, 2, 1)
        conv_x = F.relu(self.bn1(self.conv1(x)))
        conv_y = F.relu(self.bn2(self.conv2(conv_x)))
        conv_z = self.bn3(self.conv3(conv_y))

        shortcut_y = self.bn_shortcut1(self.shortcut1(x))
        output_block_1 = F.relu(conv_z + shortcut_y)

        # Block 2
        conv_x = F.relu(self.bn4(self.conv4(output_block_1)))
        conv_y = F.relu(self.bn5(self.conv5(conv_x)))
        conv_z = self.bn6(self.conv6(conv_y))

        shortcut_y = self.bn_shortcut2(self.shortcut2(output_block_1))
        output_block_2 = F.relu(conv_z + shortcut_y)

        # Block 3
        conv_x = F.relu(self.bn7(self.conv7(output_block_2)))
        conv_y = F.relu(self.bn8(self.conv8(conv_x)))
        conv_z = self.bn9(self.conv9(conv_y))

        shortcut_y = self.bn_shortcut3(output_block_2)
        output_block_3 = F.relu(conv_z + shortcut_y)

        output_block_4 = F.relu(self.bn10(self.conv10(output_block_3)))

        output_block_4 = output_block_4.permute(0, 2, 1)


        return output_block_4



class CharEmbedding(nn.Module):
    def __init__(self, args: ModelArgs, char_input=None):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=args.embedding_dim)
        self.mask_value = 0
        self.linear = nn.Linear(args.embedding_dim, 1)

        # char_input shape: (batch_size, features, 1)
        if char_input is not None:

            # Squeeze to remove the last dimension
            if isinstance(char_input, np.ndarray):
                char_input = torch.tensor(char_input, dtype=torch.long)
            char_input = char_input.squeeze(-1)  # Now shape: (batch_size, features)

            # Ensure the input tensor is of type torch.long
            char_input = char_input.long()

            # Masking: Replace mask_value with zeros
            mask = (char_input != self.mask_value).float()
            char_input = char_input * mask.long()

            # Embedding
            char_embedding = self.embedding(char_input)  # Now shape: (batch_size, features, embedding_dim)
            char_embedding_reduced = self.linear(char_embedding)  # Now shape: (batch_size, features, 1)

            return char_embedding_reduced

    def forward(self, char_input):
        # char_input shape: (batch_size, features, 1)
        if isinstance(char_input, np.ndarray):
            char_input = torch.tensor(char_input, dtype=torch.long)
        char_input = char_input.squeeze(-1)  # Now shape: (batch_size, features)

        # Ensure the input tensor is of type torch.long
        char_input = char_input.long()

        # Masking: Replace mask_value with zeros
        mask = (char_input != self.mask_value).float()
        char_input = char_input * mask.long()

        # Embedding
        char_embedding = self.embedding(char_input)  # Now shape: (batch_size, features, embedding_dim)
        char_embedding_reduced = self.linear(char_embedding)  # Now shape: (batch_size, features, 1)

        return char_embedding_reduced