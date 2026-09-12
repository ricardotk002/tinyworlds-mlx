# See book/02-space-time-transformer.md, Task 1.
import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

from .positional_encoding import build_spatial_only_pe


class PatchEmbedding(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size: int = 8, embed_dim: int = 128):
        super().__init__()
        H, W = frame_size
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp

        self._pos_spatial = build_spatial_only_pe(self.frame_size, self.patch_size, self.embed_dim)

        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)

    def __call__(self, frames):
        # frames: [B, T, C, H, W] -> [B, T, P, E]
        x = rearrange(frames, "b t c (h1 p1) (w1 p2) -> b t (h1 w1) (c p1 p2)", p1=self.patch_size, p2=self.patch_size)
        x = self.proj(x)
        x = x + self._pos_spatial
        return x
