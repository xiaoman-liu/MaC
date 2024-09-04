

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
# try:
#     from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
# except ImportError:
#     causal_conv1d_fn, causal_conv1d_update = None, None
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
# try:
#     from mamba_ssm.ops.triton.selective_state_update import selective_state_update
# except ImportError:
#     selective_state_update = None
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
# try:
#     from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from dataclasses import dataclass
from typing import Union

import sys
import os


current_directory = os.path.dirname(os.path.abspath(__file__))


sys.path.append(current_directory)

@dataclass
class ModelArgs:
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: Union[int, str] = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    layer_idx: Optional[int] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    output_dim: Optional[int] = None
    input_dim: Optional[int] = None
    n_layer: int = 2
    embedding_dim: int= 4
    shapes: Optional[dict] = None
    char_token_order: Optional[str] = None




class Mamba1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


        self.embedding = CharEmbedding(args)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.input_layer = nn.Linear(1, args.d_model)
        self.output_layer = nn.Linear(args.input_dim * args.d_model, args.output_dim)
        self.d_model = args.d_model
        self.conv_block = ConvBlock(args.d_model)
        self.group_attention = GroupAttention(args, n_heads=4)



        ## new

    def forward(self, input_ids, char_embed=None):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)
        """
        x = self.input_layer(input_ids)

        if char_embed is not None:
            char_embed = self.embedding(char_embed)
            x = torch.cat([x, char_embed], dim=1)
        char_shape = char_embed.shape[1]



        # x = self.conv_block(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        # logits = self.lm_head(x)
        x = self.group_attention(x, char_shape)
        # print("x", x.shape)
        x = x.view(-1, self.args.input_dim * self.d_model)
        logits = self.output_layer(x)

        ## output layer

        return logits

class AttentionBlock(nn.Module):
    def __init__(self, output_dim, nheads, name=None):
        super(AttentionBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.multihead_attn = MultiHeadAtten(output_dim=output_dim, nheads=nheads)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.dense1 = nn.Linear(output_dim, output_dim)
        self.dense2 = nn.Linear(output_dim, output_dim)


    def forward(self, input):
        # Layer normalization
        # print("input", input.shape)
        # print("self.layer_norm1", self.layer_norm1)
        output_block = self.layer_norm1(input)

        # Multi-head attention
        output_block = self.multihead_attn(output_block)

        # Residual connection
        output_block = output_block + input

        # Layer normalization
        FFN_6 = self.layer_norm2(output_block)

        # Feedforward network
        FFN_6 = F.relu(self.dense1(FFN_6))
        FFN_6 = self.dense2(FFN_6)

        # Residual connection
        FFN_6 = FFN_6 + output_block
        return FFN_6


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        # self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x / rms * self.weight

        return output

class ResidualBlock(nn.Module):
    def __init__(self, args):
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
    def __init__(
        self, args
    ):
        factory_kwargs = {"device": args.device, "dtype": args.dtype}
        super().__init__()

        self.d_model = args.d_model
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if args.dt_rank == "auto" else args.dt_rank
        self.use_fast_path = args.use_fast_path
        self.layer_idx = args.layer_idx
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=args.bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=self.d_inner,
            padding=args.d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * args.dt_scale
        if args.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif args.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        ).clamp(min=args.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=args.device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=args.device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """

        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        # inference_params  ={"key_value_memory_dict":{}}
        # if inference_params is not None:
        #     conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         # The states are updated inplace
        #         out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        #         return out

        # We do matmul and transpose BLH -> HBL at the same time
        ## todo ssm_state
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)


        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        # print(inference_params)
        # print("type of inference_params.key_value_memory_dict", type(inference_params))
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class CharEmbedding(nn.Module):
    def __init__(self, args, char_input=None):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=args.embedding_dim)
        self.mask_value = 0
        self.char_token_order = args.char_token_order
        self.shapes = args.shapes
        self.linear = nn.Linear(args.embedding_dim, args.d_model)



        # char_input shape: (batch_size, features, 1)

    def forward(self, char_input):
        # char_input shape: (batch_size, features, 1)
        # print("char_input", char_input.shape)
        if isinstance(char_input, np.ndarray):
            char_input = torch.tensor(char_input, dtype=torch.long)
        char_input = char_input.squeeze(-1)  # Now shape: (batch_size, features)

        # Ensure the input tensor is of type torch.long
        char_input = char_input.long()

        # Masking: Replace mask_value with zeros
        mask = (char_input != self.mask_value).float()
        char_input = char_input * mask.long()

        # Embedding
        char_embedding = self.embedding(char_input)
        char_embedding = self.linear(char_embedding)
        sys_char = char_embedding[:, -self.shapes[4][0]:, :]
        work_char = char_embedding[:, :self.shapes[6][0], :]

        sys_pooled_feature = self.pooled_feature(sys_char, "System_info")
        work_char_pooled_feature = self.pooled_feature(work_char, "Workload_info")
        output = torch.cat([work_char_pooled_feature, sys_pooled_feature], dim=1)
        # Now shape: (batch_size, features, embedding_dim)  # Now shape: (batch_size, features, 1)

        return output

    def pooled_feature(self, x, name_suffix):
        pooled_features = []
        left = 0
        for i in range(len(self.char_token_order[name_suffix])):
            dim = self.char_token_order[name_suffix][i][1]
            right = dim + left
            feature_output = x[:, left:right, :]

            # Use PyTorch's average pooling
            # We need to permute dimensions because PyTorch expects (N, C, L) for 1D pooling
            feature_output = feature_output.permute(0, 2, 1)  # Change to (batch_size, channels, length)
            pooled_output = F.avg_pool1d(feature_output, kernel_size=dim, stride=dim)
            pooled_output = pooled_output.permute(0, 2, 1)  # Change back to (batch_size, length, channels)

            pooled_features.append(pooled_output)
            left = right


        # Concatenate along the second dimension
        concatenated_features = torch.cat(pooled_features, dim=1)
        return concatenated_features


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

class GroupAttention(nn.Module):
    def __init__(self, args, n_heads=2):
        super(GroupAttention, self).__init__()
        self.n_feature_maps = args.d_model
        self.mem_numer_x_train_shape = args.shapes[1]
        self.cpu_numer_x_train_shape = args.shapes[3]
        self.system_numer_x_train_shape = args.shapes[5]
        self.multi_heads = n_heads
        self.attention_block1 = AttentionBlock(output_dim=args.d_model, nheads=n_heads, name="mem_num")
        self.attention_block2 = AttentionBlock(output_dim=args.d_model, nheads=n_heads, name="cpu_num")
        self.attention_block3 = AttentionBlock(output_dim=args.d_model, nheads=n_heads, name="system_num")
        self.attention_blockc = AttentionBlock(output_dim=args.d_model, nheads=n_heads,name="char")
        self.attention_block = AttentionBlock(output_dim=args.d_model, nheads=n_heads, name="group")



    def forward(self, x, char_shape):
        num_feature_resnet_block = x


        # Split the tensor into parts
        part1 = num_feature_resnet_block[:, :self.mem_numer_x_train_shape[0], :]
        # print("self.mem_numer_x_train_shape[0]", self.mem_numer_x_train_shape[0])
        part2 = num_feature_resnet_block[:,
                self.mem_numer_x_train_shape[0]:self.mem_numer_x_train_shape[0] + self.cpu_numer_x_train_shape[0], :]
        part3 = num_feature_resnet_block[:,
                self.mem_numer_x_train_shape[0] + self.cpu_numer_x_train_shape[0]:self.mem_numer_x_train_shape[0] +
                                                                                  self.cpu_numer_x_train_shape[0] +
                                                                                  self.system_numer_x_train_shape[0], :]
        char_feature_resnet_block = num_feature_resnet_block[:,-char_shape:, :]

        # Apply attention blocks
        # print("self.attention_blockc", self.attention_blockc)
        # for layer1 in self.self.attention_blockc:
        #     print("layer1", layer1)
        #     char_feature_resnet_block = layer1(char_feature_resnet_block)
        # print("shape of num_feature_resnet_block", num_feature_resnet_block.shape)
        # for layer2 in self.attention_block1:
        #     part1 = layer2(part1)    # (batch_size, num_features, d_model)
        # print("shape of part1", part1.shape)
        # for layer3 in self.attention_block2:
        #     part2 = layer3(part2)
        # for layer4 in self.attention_block3:
        #     part3 = layer4(part3)
        char_feature_resnet_block = self.attention_blockc(char_feature_resnet_block)
        part1 = self.attention_block1(part1)
        part2 = self.attention_block2(part2)
        part3 = self.attention_block3(part3)



        # Apply average pooling
        # part1_attention_block = F.adaptive_avg_pool1d(part1.permute(0, 2, 1), output_size=1).squeeze(-1)
        # part2_attention_block = F.adaptive_avg_pool1d(part2.permute(0, 2, 1), output_size=1).squeeze(-1)
        # part3_attention_block = F.adaptive_avg_pool1d(part3.permute(0, 2, 1), output_size=1).squeeze(-1)
        # char_features_attention_block = F.adaptive_avg_pool1d(char_feature_resnet_block.permute(0, 2, 1),
        #                                                       output_size=1).squeeze(-1)

        # Concatenate along the feature dimension
        concatenated_inputs = torch.cat(
            [part1, part2, part3, char_feature_resnet_block], dim=1)

        # Apply group attention block
        # print("concatenated_inputs", concatenated_inputs.shape)

        group_attention_block = self.attention_block(concatenated_inputs)


        return group_attention_block



class MultiHeadAtten(nn.Module):
    """
    Attention layer for RNN models
    """

    def __init__(self, output_dim, nheads, return_attention=False, trainable=True):
        super(MultiHeadAtten, self).__init__()
        self.return_attention = return_attention
        self.output_dim = output_dim
        self.hidden_size = output_dim // nheads
        self._n_heads = nheads
        self.trainable = trainable

        # Define linear layers for Q, K, V
        self.wq = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.wk = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.wv = nn.Linear(self.output_dim, self.output_dim, bias=False)
        self.wo = nn.Linear(self.output_dim, self.output_dim, bias=False)  # Output linear layer

    def split_heads(self, x):
        batch_size, seq_len, feature_dim = x.size()
        head_dim = feature_dim // self._n_heads
        output = x.view(batch_size, seq_len, self._n_heads, head_dim).transpose(1, 2)
        return output  #  (batch_size, num_heads, seq_len, feature_dim)


    def forward(self, x, mask=None):


        # print("before attention", x.shape)
        batch_size, seq_len, d_model = x.size()
        # print("x.size()", x.size())
        Qi = self.split_heads(self.wq(x))
        Ki = self.split_heads(self.wk(x))
        Vi = self.split_heads(self.wv(x))
        # print("Qi", Qi.shape)

        matmul_qk = torch.matmul(Qi, Ki.transpose(-2, -1))  # (..., seq_length_q, seq_length_k)
        dk = Ki.size()[-1]
        scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))



        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        out = torch.matmul(attention_weights, Vi)
        # print("out", out.shape)
        # Concatenate heads
        attention_output = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear layer
        output = self.wo(attention_output)
        # print("output", output.shape)

        if self.return_attention:
            return output, attention_weights
        return output

