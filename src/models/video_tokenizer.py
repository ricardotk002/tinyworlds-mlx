# Video Tokenizer (FSQ-VAE #1). See book/03-video-tokenizer.md.
import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

from .st_transformer import STTransformer
from .fsq import FiniteScalarQuantizer
from .patch_embed import PatchEmbedding
from .positional_encoding import build_spatial_only_pe


class VideoTokenizerEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size: int = 8, embed_dim: int = 128,
                 num_heads: int = 8, hidden_dim: int = 256, num_blocks: int = 4, latent_dim: int = 5):
        super().__init__()

        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, True)
        self.ln = nn.LayerNorm(embed_dim)
        self.to_latent = nn.Linear(embed_dim, latent_dim)
        self.latent_dim = latent_dim

    def __call__(self, frames):
        # frames: [B, T, C, H, W] -> [B, T, P, latent_dim]
        x = self.patch_embed(frames)
        x = self.transformer(x)

        x = self.ln(x)
        return self.to_latent(x)


class PixelShuffleFrameHead(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int = 8, channels: int = 3, H: int = 128, W: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.Hp, self.Wp = H // patch_size, W // patch_size

        self.to_pixels = nn.Linear(embed_dim, channels * patch_size ** 2)

    def __call__(self, tokens):
        # tokens: [B, T, P, E] -> [B, T, C, H, W]
        x = self.to_pixels(tokens)
        P = x.shape[-2]
        hp = wp = int(P ** 0.5)
        return rearrange(x, "b t (hp wp) (c p1 p2) -> b t c (hp p1) (wp p2)", c=self.channels, wp=wp, hp=hp, p1=self.patch_size, p2=self.patch_size)


class VideoTokenizerDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size: int = 8, embed_dim: int = 128,
                 num_heads: int = 8, hidden_dim: int = 256, num_blocks: int = 4, latent_dim: int = 5):
        super().__init__()
        H, W = frame_size
        self.patch_size = patch_size
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp

        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, True)
        self.frame_head = PixelShuffleFrameHead(embed_dim, patch_size, channels=3, H=H, W=W)
        self._pos_spatial_dec = build_spatial_only_pe(frame_size, patch_size, embed_dim)

    def __call__(self, latents):
        # latents: [B, T, P, latent_dim] -> [B, T, C, H, W]
        x = self.latent_embed(latents) + self._pos_spatial_dec
        x = self.transformer(x)
        return self.frame_head(x)

class VideoTokenizer(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size: int = 8, embed_dim: int = 128,
                 num_heads: int = 8, hidden_dim: int = 256, num_blocks: int = 4,
                 latent_dim: int = 3, num_bins: int = 4):
        super().__init__()

        self.encoder = VideoTokenizerEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.decoder = VideoTokenizerDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, latent_dim)
        self.quantizer = FiniteScalarQuantizer(latent_dim, num_bins)

        self.codebook_size = num_bins ** latent_dim

    def __call__(self, frames):
        # frames: [B, T, C, H, W] -> (recon_loss, x_hat)
        z = self.encoder(frames)
        z_q = self.quantizer(z)
        x_hat = self.decoder(z_q)

        loss = nn.losses.smooth_l1_loss(x_hat, frames)

        return loss, x_hat

    def tokenize(self, frames):
        # frames: [B, T, C, H, W] -> indices [B, T, P] (int)
        z = self.encoder(frames)
        z_q = self.quantizer(z)
        return self.quantizer.get_indices_from_latents(z_q)

    def detokenize(self, quantized_z):
        # quantized_z: [B, T, P, latent_dim] -> frames [B, T, C, H, W]
        return self.decoder(quantized_z)
