# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Diffusion Transformer model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input with the shift and scale."""
    return x * (1.0 + scale) + shift


def get_multidimensional_position_embeddings(position_embeddings: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Extracts position embeddings for a multidimensional sequence given by coordinates.

    Position embeddings are shape (D, T, F). Coords are shape (B, S, D). Position embeddings should be
    interpreted as D dimensional embeddings with F features each for a maximum of T timesteps.
    Coordinates or `coords` is a batch of size B of sequences of length S with D dimensional integer
    coordinates. For example, if D=2, then each of the B, S elements of the sequence would have a 2D
    X,Y coordinate.

    Args:
        position_embeddings (torch.Tensor): Position embeddings of shape (D, T, F).
        coords (torch.Tensor): Coordinates of shape (B, S, D).

    Returns:
        torch.Tensor: Sequenced embeddings of shape (B, S, F, D)
    """
    B = coords.shape[0]
    F = position_embeddings.shape[-1]  # Position embedding dimensions
    # Transpose coords to shape (D, B, S)
    coords = coords.permute(2, 0, 1)  # (D, B, S)
    # Expand position_embeddings to match the batch size
    position_embeddings = position_embeddings.unsqueeze(1).expand(-1, B, -1, -1)  # (D, B, T, F)
    # Prepare indices for torch.gather
    coords = coords.unsqueeze(-1).expand(-1, -1, -1, F)  # (D, B, S, F)
    # Use torch.gather to collect embeddings
    embeddings = torch.gather(position_embeddings, 2, coords)  # (D, B, S, F)
    # Rearrange embeddings to the desired output shape
    embeddings = embeddings.permute(1, 2, 3, 0)  # (B, S, F, D)
    return embeddings


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm.

    Scales and shifts the output of a LayerNorm using an MLP conditioned on a scalar.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        # MLP for computing modulations.
        # Initialized to zero so modulation acts as identity at initialization.
        self.adaLN_mlp_linear_shift = nn.Linear(self.num_features, self.num_features, bias=True)
        self.adaLN_mlp_linear_scale = nn.Linear(self.num_features, self.num_features, bias=True)
        nn.init.zeros_(self.adaLN_mlp_linear_shift.weight)
        nn.init.zeros_(self.adaLN_mlp_linear_scale.weight)
        nn.init.zeros_(self.adaLN_mlp_linear_shift.bias)
        nn.init.zeros_(self.adaLN_mlp_linear_scale.bias)
        self.adaLN_mlp_shift = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear_shift)
        self.adaLN_mlp_scale = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear_scale)
        # LayerNorm
        self.layernorm = nn.LayerNorm(self.num_features, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Calculate the modulations
        shift = self.adaLN_mlp_linear_shift(t)
        scale = self.adaLN_mlp_linear_scale(t)
        # Apply the modulations
        return modulate(self.layernorm(x), shift, scale)


class ModulationLayer(nn.Module):
    """Modulation layer.

    Scales the input by a factor determined by a scalar input.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        # MLP for computing modulation.
        # Initialized to zero so modulation starts off at zero.
        self.adaLN_mlp_linear = nn.Linear(self.num_features, self.num_features, bias=True)
        nn.init.zeros_(self.adaLN_mlp_linear.weight)
        nn.init.zeros_(self.adaLN_mlp_linear.bias)
        self.adaLN_mlp = nn.Sequential(nn.SiLU(), self.adaLN_mlp_linear)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Calculate the modulations
        mods = self.adaLN_mlp(t)
        return x * mods


class ScalarEmbedding(nn.Module):
    """Embedding block for scalars.

    Embeds a scalar into a vector of size `num_features` using a sinusoidal embedding followed by an MLP.

    Args:
        num_features (int): The size of the output vector.
        sinusoidal_embedding_dim (int): The size of the intermediate sinusoidal embedding. Default: `256`.
        max_period (int): The maximum period of the sinusoidal embedding. Default: `10000`.

    Returns:
        torch.Tensor: The embedded scalar
    """

    def __init__(self, num_features: int, sinusoidal_embedding_dim: int = 256, max_period: int = 10000):
        super().__init__()
        self.num_features = num_features
        self.sinusoidal_embedding_dim = sinusoidal_embedding_dim
        self.max_period = max_period
        self.linear_1 = nn.Linear(self.sinusoidal_embedding_dim, self.num_features)
        self.linear_2 = nn.Linear(self.num_features, self.num_features)
        self.mlp = nn.Sequential(self.linear_1, nn.SiLU(), self.linear_2)
        # Make the freqs
        half_dim = self.sinusoidal_embedding_dim // 2
        self.freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) /
                               half_dim)

    def _apply(self, fn):
        super(ScalarEmbedding, self)._apply(fn)
        self.freqs = fn(self.freqs)
        return self

    def timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            timesteps (torch.Tensor): The timesteps to embed.
        """
        args = timesteps[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sinusoidal_embedding = self.timestep_embedding(x)
        # Ensure embedding is the correct dtype
        sinusoidal_embedding = sinusoidal_embedding.to(next(self.parameters()).dtype)
        return self.mlp(sinusoidal_embedding)


class VectorEmbedding(nn.Module):
    """Embedding block for vectors.

    Embeds vectors via an MLP into a vector of size `num_features`.

    Args:
        input_features (int): The size of the input vector.
        num_features (int): The size of the output vector.
    """

    def __init__(self, input_features: int, num_features: int):
        super().__init__()
        self.input_features = input_features
        self.num_features = num_features
        self.linear_1 = nn.Linear(self.input_features, self.num_features)
        self.linear_2 = nn.Linear(self.num_features, self.num_features)
        self.mlp = nn.Sequential(self.linear_1, nn.SiLU(), self.linear_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PreAttentionBlock(nn.Module):
    """Block to compute QKV before attention.

    Includes QK layernorms and an adaptive layernorms.

    Args:
        num_features (int): Number of input features.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        # Adaptive layernorm
        self.adaptive_layernorm = AdaptiveLayerNorm(self.num_features)
        # Linear layer to get q, k, and v
        self.q_proj = nn.Linear(self.num_features, self.num_features)
        self.k_proj = nn.Linear(self.num_features, self.num_features)
        self.v_proj = nn.Linear(self.num_features, self.num_features)
        # QK layernorms. Original MMDiT used RMSNorm here.
        self.q_norm = nn.LayerNorm(self.num_features, elementwise_affine=True, eps=1e-6)
        self.k_norm = nn.LayerNorm(self.num_features, elementwise_affine=True, eps=1e-6)
        for l in [self.q_proj, self.k_proj, self.v_proj]:
            # Initialize all biases to zero
            nn.init.zeros_(l.bias)
            # Init the standard deviation of the weights to 0.02 as is tradition
            nn.init.normal_(l.weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.adaptive_layernorm(x, t)
        # Calculate the query, key, and values all in one go
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        return q, k, v


class SelfAttention(nn.Module):
    """Standard multihead self attention layer that supports masking.

    Args:
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, num_features: int, num_heads: int):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_dim = num_features // num_heads

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = v.size()
        H = self.num_heads
        D = self.head_dim

        # Reshape and permute to (B, H, T, D) in one step
        def reshape_to_heads(x):
            return x.reshape(B, T, H, D).permute(0, 2, 1, 3)

        q = reshape_to_heads(q)  # Shape: (B, H, T, D)
        k = reshape_to_heads(k)
        v = reshape_to_heads(v)
        # Apply scaled dot-product attention
        attention_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # attention_out is (B, H, T, D)
        # Permute back and reshape to (B, T, C)
        attention_out = attention_out.permute(0, 2, 1, 3).reshape(B, T, C)
        return attention_out


class PostAttentionBlock(nn.Module):
    """Block to postprocess v after attention.

    Includes adaptive layernorms.

    Args:
        num_features (int): Number of input features.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
    """

    def __init__(self, num_features: int, expansion_factor: int = 4):
        super().__init__()
        self.num_features = num_features
        self.expansion_factor = expansion_factor
        # Input modulation
        self.modulate_v = ModulationLayer(self.num_features)
        # Linear layer to process v
        self.linear_v = nn.Linear(self.num_features, self.num_features)
        # Layernorm for the output
        self.output_norm = AdaptiveLayerNorm(self.num_features)
        # Transformer style MLP layers
        self.linear_1 = nn.Linear(self.num_features, self.expansion_factor * self.num_features)
        self.nonlinearity = nn.GELU(approximate='tanh')
        self.linear_2 = nn.Linear(self.expansion_factor * self.num_features, self.num_features)
        # Initialize all biases to zero
        nn.init.zeros_(self.linear_1.bias)
        nn.init.zeros_(self.linear_2.bias)
        # Output MLP
        self.output_mlp = nn.Sequential(self.linear_1, self.nonlinearity, self.linear_2)
        # Output modulation
        self.modulate_output = ModulationLayer(self.num_features)

    def forward(self, v: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward takes v from self attention and the original sequence x with scalar conditioning t."""
        # Postprocess v with linear + gating modulation
        y = self.modulate_v(self.linear_v(v), t)
        y = x + y
        # Adaptive layernorm
        y = self.output_norm(y, t)
        # Output MLP
        y = self.output_mlp(y)
        # Gating modulation for the output
        y = self.modulate_output(y, t)
        y = x + y
        return y


class MMDiTBlock(nn.Module):
    """Transformer block that supports masking, multimodal attention, and adaptive norms.

    Can optionally be the last block in the network, in which case it does not apply post-attention layers to the
    conditioning sequence, as those params may not be used.

    Args:
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
        is_last (bool): Whether this is the last block in the network. Default: `False`.
    """

    def __init__(self, num_features: int, num_heads: int, expansion_factor: int = 4, is_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.is_last = is_last
        # Pre-attention blocks for two modalities
        self.pre_attention_block_1 = PreAttentionBlock(self.num_features)
        self.pre_attention_block_2 = PreAttentionBlock(self.num_features)
        # Self-attention
        self.attention = SelfAttention(self.num_features, self.num_heads)
        # Post-attention blocks for two modalities
        self.post_attention_block_1 = PostAttentionBlock(self.num_features, self.expansion_factor)
        if not self.is_last:
            self.post_attention_block_2 = PostAttentionBlock(self.num_features, self.expansion_factor)

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention for the two modalities
        with record_function('Pre-attention'):
            q1, k1, v1 = self.pre_attention_block_1(x1, t)
            q2, k2, v2 = self.pre_attention_block_2(x2, t)
        # Concat q, k, v along the sequence dimension
        q = torch.cat([q1, q2], dim=1)
        k = torch.cat([k1, k2], dim=1)
        v = torch.cat([v1, v2], dim=1)
        # Self-attention
        with record_function('Attention'):
            v = self.attention(q, k, v, mask=mask)
        # Split the attention output back into the two modalities
        seq_len_1, seq_len_2 = x1.size(1), x2.size(1)
        y1, y2 = v[:, :seq_len_1], v[:, seq_len_1:seq_len_1 + seq_len_2]
        # Post-attention for the two modalities
        with record_function('Post attention'):
            y1 = self.post_attention_block_1(y1, x1, t)
            if not self.is_last:
                y2 = self.post_attention_block_2(y2, x2, t)
        return y1, y2


class MMDiTGroup(nn.Module):
    """A group of MMDiT blocks, for convenience.

    Args:
        num_blocks (int): Number of blocks in the group
        num_features (int): Number of input features.
        num_heads (int): Number of attention heads.
        expansion_factor (int): Expansion factor for the MLP. Default: `4`.
        is_last (bool): Whether this is the last block in the network. Default: `False`.
    """

    def __init__(self, num_blocks, num_features, num_heads, expansion_factor, is_last=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            MMDiTBlock(num_features, num_heads, expansion_factor, is_last=(is_last and i == num_blocks - 1))
            for i in range(num_blocks)
        ])

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                t: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x1, x2 = block(x1, x2, t, mask=mask)
        return x1, x2


class DiffusionTransformer(nn.Module):
    """Transformer model for generic diffusion.

    Supports input and conditioning sequences with different lengths and dimensions.

    Args:
        num_features (int): Number of hidden features.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        input_features (int): Number of features in the input sequence. Default: `192`.
        input_max_sequence_length (int): Maximum sequence length for the input sequence. Default: `1024`.
        input_dimension (int): Dimension of the input sequence. Default: `2`.
        conditioning_features (int): Number of features in the conditioning sequence. Default: `1024`.
        conditioning_max_sequence_length (int): Maximum sequence length for the conditioning sequence. Default: `77`.
        conditioning_dimension (int): Dimension of the conditioning sequence. Default: `1`.
        expansion_factor (int): Expansion factor for the MLPs. Default: `4`.
        num_register_tokens (int): Number of register tokens to use. Default: `0`.
        block_group_size (int): Size of transformer block groups. Default: `4`.
    """

    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 num_layers: int,
                 input_features: int = 192,
                 input_max_sequence_length: int = 1024,
                 input_dimension: int = 2,
                 conditioning_features: int = 1024,
                 conditioning_max_sequence_length: int = 77,
                 conditioning_dimension: int = 1,
                 expansion_factor: int = 4,
                 num_register_tokens: int = 0,
                 block_group_size: int = 4):
        super().__init__()
        # Params for the network architecture
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        self.block_group_size = block_group_size
        # Params for input embeddings
        self.input_features = input_features
        self.input_dimension = input_dimension
        self.input_max_sequence_length = input_max_sequence_length
        # Params for conditioning embeddings
        self.conditioning_features = conditioning_features
        self.conditioning_dimension = conditioning_dimension
        self.conditioning_max_sequence_length = conditioning_max_sequence_length
        self.num_register_tokens = num_register_tokens
        # Embedding block for the timestep
        self.timestep_embedding = ScalarEmbedding(self.num_features)
        # Projection layer for the input sequence
        self.input_embedding = nn.Linear(self.input_features, self.num_features)
        # Embedding layer for the input sequence
        input_position_embedding = torch.randn(self.input_dimension, self.input_max_sequence_length, self.num_features)
        input_position_embedding /= math.sqrt(self.num_features)
        self.input_position_embedding = torch.nn.Parameter(input_position_embedding, requires_grad=True)
        # Projection layer for the conditioning sequence
        self.conditioning_embedding = nn.Linear(self.conditioning_features, self.num_features)
        # Embedding layer for the conditioning sequence
        conditioning_position_embedding = torch.randn(self.conditioning_dimension,
                                                      self.conditioning_max_sequence_length, self.num_features)
        conditioning_position_embedding /= math.sqrt(self.num_features)
        self.conditioning_position_embedding = torch.nn.Parameter(conditioning_position_embedding, requires_grad=True)
        # Optional register tokens
        if self.num_register_tokens > 0:
            register_tokens = torch.randn(1, self.num_register_tokens, self.num_features) / math.sqrt(self.num_features)
            self.register_tokens = torch.nn.Parameter(register_tokens, requires_grad=True)

        # Transformer blocks
        assert self.num_layers % self.block_group_size == 0, 'num_layers must be divisible by block_group_size.'
        num_groups = self.num_layers // self.block_group_size
        # Transformer groups
        self.transformer_groups = nn.ModuleList([
            MMDiTGroup(self.block_group_size, self.num_features, self.num_heads, expansion_factor=self.expansion_factor)
            for _ in range(num_groups - 1)
        ])
        # Turn off post attn layers for conditioning sequence in final block
        self.transformer_groups.append(
            MMDiTGroup(self.block_group_size,
                       self.num_features,
                       self.num_heads,
                       expansion_factor=self.expansion_factor,
                       is_last=True))
        # Output projection layer
        self.final_norm = AdaptiveLayerNorm(self.num_features)
        self.final_linear = nn.Linear(self.num_features, self.input_features)
        # Init the output layer to zero
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        if isinstance(module, MMDiTGroup):
            return True
        return False

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        if isinstance(module, MMDiTGroup):
            return True
        return False

    def forward(self,
                x: torch.Tensor,
                input_coords: torch.Tensor,
                t: torch.Tensor,
                conditioning: torch.Tensor,
                conditioning_coords: torch.Tensor,
                input_mask: Optional[torch.Tensor] = None,
                conditioning_mask: Optional[torch.Tensor] = None,
                constant_conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the diffusion transformer.

        Args:
            x (torch.Tensor): The input sequence of shape (B, T1, C1).
            input_coords (torch.Tensor): The coordinates of the D dimensional input sequence of shape (B, T1, D).
            t (torch.Tensor): The scalar timesteps of shape (B, 1).
            conditioning (torch.Tensor): The conditioning sequence of shape (B, T2, C2).
            conditioning_coords (torch.Tensor): The coordinates of the D dimensional conditioning sequence of shape (B, T2, D).
            input_mask (Optional[torch.Tensor]): The mask for the input sequence of shape (B, T1).
            conditioning_mask (Optional[torch.Tensor]): The mask for the conditioning sequence of shape (B, T2).
            constant_conditioning (Optional[torch.Tensor]): Optional additional constant conditioning (B, num_features).

        Returns:
            torch.Tensor: The output sequence of shape (B, T1, C1).
        """
        # Embed the timestep
        with record_function('Preprocessing'):
            with record_function('Timestep embedding'):
                t = self.timestep_embedding(t)
                # Optionally add constant conditioning. This assumes it has been embedded already.
                if constant_conditioning is not None:
                    t = t + constant_conditioning
                t = t.unsqueeze(1)
            with record_function('Sequence embedding'):
                # Embed the input
                y = self.input_embedding(x)  # (B, T1, C)
                # Get the input position embeddings and add them to the input
                y_position_embeddings = get_multidimensional_position_embeddings(self.input_position_embedding,
                                                                                 input_coords)
                y_position_embeddings = y_position_embeddings.sum(dim=-1)  # (B, T1, C)
                y = y + y_position_embeddings  # (B, T1, C)
                # Embed the conditioning
                c = self.conditioning_embedding(conditioning)  # (B, T2, C)
                # Get the conditioning position embeddings and add them to the conditioning
                c_position_embeddings = get_multidimensional_position_embeddings(self.conditioning_position_embedding,
                                                                                 conditioning_coords)
                c_position_embeddings = c_position_embeddings.sum(dim=-1)  # (B, T2, C)
                c = c + c_position_embeddings  # (B, T2, C)
            # Optionally add the register tokens
            if self.num_register_tokens > 0:
                repeated_register = self.register_tokens.repeat(c.shape[0], 1, 1)
                c = torch.cat([c, repeated_register], dim=1)

        # Pass through the transformer blocks
        with record_function('Blocks forward'):
            for block in self.transformer_groups:
                y, c = block(y, c, t, mask=None)
        # Pass through the output layers to get the right number of elements
        y = self.final_norm(y, t)
        y = self.final_linear(y)
        return y
