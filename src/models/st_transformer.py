# Space-Time Transformer — https://arxiv.org/pdf/2001.02908
# SwiGLU               — https://arxiv.org/pdf/2002.05202
import math
import mlx.core as mx
import mlx.nn as nn

from .norms import AdaptiveNormalizer
from .positional_encoding import sincos_time
from einops import rearrange

class SpatialAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, conditioning_dim: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.norm = AdaptiveNormalizer(self.embed_dim, conditioning_dim)

    def __call__(self, x, conditioning=None):
        # x: [B, T, P, E], attends over P
        # D = E / H
        B, T = x.shape[0], x.shape[1]

        xr = rearrange(x, "b t p e -> (b t) p e")
        q = self.q_proj(xr)
        k = self.k_proj(xr)
        v = self.v_proj(xr)
        q = rearrange(q, "bt p (h d) -> bt h p d", h=self.num_heads)
        k = rearrange(k, "bt p (h d) -> bt h p d", h=self.num_heads)
        v = rearrange(v, "bt p (h d) -> bt h p d", h=self.num_heads)

        s = q @ rearrange(k, "bt h p d -> bt h d p") * mx.rsqrt(self.head_dim)
        attn = mx.softmax(s, axis=-1) @ v

        out = rearrange(attn, "bt h p d -> bt p (h d)")
        out = self.out_proj(out)
        out = rearrange(out, "(b t) p e -> b t p e", b=B, t=T)

        return self.norm(x + out, conditioning)

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, causal: bool = True, conditioning_dim: int | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.causal = causal

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.norm = AdaptiveNormalizer(self.embed_dim, conditioning_dim)
        
    def __call__(self, x, conditioning=None):
        # x: [B, T, P, E], attends over T. Causal mask over T when self.causal.
        # D = E / H
        B, T, P = x.shape[0], x.shape[1], x.shape[2]

        xr = rearrange(x, "b t p e -> (b p) t e")
        q = self.q_proj(xr)
        k = self.k_proj(xr)
        v = self.v_proj(xr)
        q = rearrange(q, "bp t (h d) -> bp h t d", h=self.num_heads)
        k = rearrange(k, "bp t (h d) -> bp h t d", h=self.num_heads)
        v = rearrange(v, "bp t (h d) -> bp h t d", h=self.num_heads)

        s = q @ rearrange(k, "bt h d t -> bt h t d") * mx.rsqrt(self.num_heads)

        if self.causal:
            mask = mx.triu(mx.full((T, T), -mx.inf), k=1)
            s = s + mask

        attn = mx.softmax(s, axis=-1) @ v

        out = rearrange(attn, "bp h t d -> bp t (h d)")
        out = self.out_proj(out)
        out = rearrange(out, "(b p) t e -> b t p e", b=B, p=P)

        return self.norm(x + out, conditioning)

class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, conditioning_dim: int | None = None):
        super().__init__()
        h = math.floor(2 * hidden_dim / 3)
        self.w_v = nn.Linear(embed_dim, h)
        self.w_g = nn.Linear(embed_dim, h)
        self.w_o = nn.Linear(h, embed_dim)

        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def __call__(self, x, conditioning=None):
        o = self.w_o(nn.silu(self.w_v(x)) * self.w_g(x))
        return self.norm(x + o, conditioning)


class STTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, causal: bool = True,
                 conditioning_dim: int | None = None):
        super().__init__()

        self.spatial_attn = SpatialAttention(embed_dim, num_heads, conditioning_dim)
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, causal, conditioning_dim)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim, conditioning_dim)

        self.norm = AdaptiveNormalizer(embed_dim, conditioning_dim)

    def __call__(self, x, conditioning=None):
        x = self.spatial_attn(x, conditioning)
        x = self.temporal_attn(x, conditioning)
        x = self.ffn(x, conditioning)

        return self.norm(x, conditioning)

class STTransformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, num_blocks: int,
                 causal: bool = True, conditioning_dim: int | None = None):
        super().__init__()
        self.temporal_dim = (embed_dim // 3) & ~1
        self.spatial_dims = embed_dim - self.temporal_dim

        self.blocks = [STTransformerBlock(embed_dim, num_heads, hidden_dim, causal, conditioning_dim) for _ in range(num_blocks)]

    def __call__(self, x, conditioning=None):
        # x: [B, T, P, E]
        pe = sincos_time(x.shape[-3], self.temporal_dim)
        pe = mx.pad(pe, ((0, 0), (0, self.spatial_dims)))
        pe = rearrange(pe, "t e -> 1 t 1 e")

        x = x + pe

        for block in self.blocks:
            x = block(x, conditioning)

        return x
