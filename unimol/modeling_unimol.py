from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_unimol import UniMolConfig


@torch.jit.script
def gaussian_rbf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Gaussian radial basis function.

    Args:
        x (torch.Tensor): Input tensor.
        mean (torch.Tensor): Mean of the Gaussian.
        std (torch.Tensor): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Output tensor after applying Gaussian RBF.
    """
    pi = 3.14159265359
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """
    Gaussian basis functions for encoding pairwise distances.
    """

    def __init__(self, num_kernels: int = 128, num_edge_types: int = 961):
        super().__init__()
        self.K = num_kernels
        self.means = nn.Embedding(1, num_kernels)
        self.stds = nn.Embedding(1, num_kernels)
        self.mul = nn.Embedding(num_edge_types, 1)
        self.bias = nn.Embedding(num_edge_types, 1)

        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, distances: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        mul = self.mul(edge_types).type_as(distances)
        bias = self.bias(edge_types).type_as(distances)
        x = mul * distances.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)

        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5

        return gaussian_rbf(x.float(), mean, std).type_as(self.means.weight)


class NonLinearHead(nn.Module):
    """
    Two-layer MLP projection head.
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "relu", hidden: int = None):
        super().__init__()
        hidden = hidden or in_features
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, out_features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "tanh":
            x = torch.tanh(x)
        return self.linear2(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with fused in_proj (matches original checkpoint format).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # fused Q/K/V projection (matches checkpoint: in_proj.weight [1536, 512])
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.size()

        # fused QKV projection
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q * self.scaling
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        # add attention bias from GBF
        if attn_bias is not None:
            attn_bias_reshaped = attn_bias.view(bsz, self.num_heads, seq_len, seq_len)
            attn_weights = attn_weights + attn_bias_reshaped

        # padding mask
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        out = self.out_proj(out)

        attn_out = attn_weights.view(bsz * self.num_heads, seq_len, seq_len)
        return out, attn_out


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer matching original implementation.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_dim: int = 2048,
        num_heads: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # pre-norm + self attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn_out = self.self_attn(x, attn_bias, padding_mask)
        x = self.dropout(x)
        x = residual + x

        # pre-norm + FFN
        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.activation_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x

        return x, attn_out


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers.
    """

    def __init__(self, config: UniMolConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.emb_layer_norm = nn.LayerNorm(config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=config.hidden_size,
                ffn_dim=config.intermediate_size,
                num_heads=config.num_attention_heads,
                dropout=config.hidden_dropout_prob,
                attention_dropout=config.attention_probs_dropout_prob,
                activation_dropout=config.activation_dropout,
            )
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()

        x = self.emb_layer_norm(x)
        x = self.dropout(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).float())
            attn_mask = attn_mask.view(bsz, self.num_heads, seq_len, seq_len)
            attn_mask = attn_mask.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_mask = attn_mask.view(bsz * self.num_heads, seq_len, seq_len)

        for layer in self.layers:
            x, attn_mask = layer(x, attn_mask, padding_mask=None)

        x = self.final_layer_norm(x)
        return x


class UniMolEncoder(nn.Module):
    """
    UniMol encoder for molecules or pockets.
    """

    def __init__(self, config: UniMolConfig):
        super().__init__()
        self.config = config
        self.padding_idx = 0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.encoder = TransformerEncoder(config)

        num_edge_types = config.vocab_size * config.vocab_size
        self.gbf = GaussianLayer(config.gbf_kernels, num_edge_types)
        self.gbf_proj = NonLinearHead(config.gbf_kernels, config.num_attention_heads, "gelu")

    def forward(
        self,
        tokens: torch.Tensor,
        distances: torch.Tensor,
        edge_types: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = tokens.eq(self.padding_idx)
        x = self.embed_tokens(tokens)

        # gBF distance features -> attention bias
        n_node = distances.size(-1)
        gbf_feat = self.gbf(distances, edge_types)
        attn_bias = self.gbf_proj(gbf_feat)
        attn_bias = attn_bias.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

        x = self.encoder(x, attn_bias, padding_mask)
        return x