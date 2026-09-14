# Latent Action Model (FSQ-VAE #2, self-supervised action inference).
import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses

from .patch_embed import PatchEmbedding
from .st_transformer import STTransformer
from .fsq import FiniteScalarQuantizer
from einops import rearrange

NUM_LATENT_ACTIONS_BINS = 2

class LatentActionsEncoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size: int = 8, embed_dim: int = 128,
                 num_heads: int = 8, hidden_dim: int = 256, num_blocks: int = 4, action_dim: int = 3):
        super().__init__()
        self.action_dim = action_dim

        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, causal=True)

        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 4 * action_dim),
            nn.GELU(),
            nn.Linear(4 * action_dim, action_dim)
        )

    def __call__(self, frames):
        # frames: [B, T, C, H, W] -> actions: [B, T-1, action_dim]
        B = frames.shape[0]
        T = frames.shape[1]
        actions = mx.zeros((B, T-1, self.action_dim))
        x = self.patch_embed(frames)

        x = self.transformer(x)
        x = mx.mean(x, axis=-2) # pooling

        for t in range(T-1):
            pair = mx.concat([x[:,t], x[:,t+1]], axis=-1)
            actions[:, t] = self.action_head(pair)

        return actions

class LatentActionsDecoder(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size: int = 8, embed_dim: int = 128,
                 num_heads: int = 8, hidden_dim: int = 256, num_blocks: int = 4, conditioning_dim: int = 3):
        super().__init__()
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.Hp, self.Wp = frame_size[0] // patch_size, frame_size[1] // patch_size
        self.num_patches = self.Hp * self.Wp

        self.patch_embed = PatchEmbedding(frame_size, patch_size, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, True, conditioning_dim)
        self.frame_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3 * patch_size * patch_size),
            nn.Tanh()
        )

        self.mask_token = mx.zeros((1, 1, 1, embed_dim))

    def __call__(self, frames, actions, keep_rate: float = 0.0, training: bool = True):
        # frames: [B, T, C, H, W], actions: [B, T-1, A] -> pred_frames: [B, T-1, C, H, W]

        B, T, C, H, W = frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[-2], frames.shape[-1]
        frames = frames[:, :-1]
        x = self.patch_embed(frames)
        P = x.shape[-2]

        if training and self.training:
            keep = mx.random.uniform(shape=(B, T-1, P, 1)) < keep_rate
            keep[:, 0] = True
            x = mx.where(keep, x, self.mask_token)

        x = self.transformer(x, actions)
        x = self.frame_head(x)

        return rearrange(x, "b t (hp wp) (c p1 p2) -> b t c (hp p1) (wp p2)", hp=self.Hp, wp=self.Wp, p1=self.patch_size, p2=self.patch_size)

class LatentActionModel(nn.Module):
    def __init__(self, frame_size=(128, 128), n_actions: int = 8, patch_size: int = 8,
                 embed_dim: int = 128, num_heads: int = 8, hidden_dim: int = 256, num_blocks: int = 4):
        super().__init__()
        assert NUM_LATENT_ACTIONS_BINS ** int(round(math.log(n_actions, NUM_LATENT_ACTIONS_BINS))) == n_actions

        self.action_dim = int(math.log(n_actions, 2))
        self.encoder = LatentActionsEncoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, self.action_dim)
        self.quantizer = FiniteScalarQuantizer(self.action_dim, NUM_LATENT_ACTIONS_BINS)
        self.decoder = LatentActionsDecoder(frame_size, patch_size, embed_dim, num_heads, hidden_dim, num_blocks, self.action_dim)
        self.var_target = 0.01
        self.var_lambda = 100.0

    def __call__(self, frames, keep_rate: float = 0.0):
        # frames: [B, T, C, H, W] -> (total_loss: scalar, pred_frames: [B, T-1, C, H, W])
        action_latents = self.encoder(frames)
        al_quantized = self.quantizer(action_latents)
        pred_frames = self.decoder(frames, al_quantized, keep_rate, True)
        recon_loss = losses.smooth_l1_loss(pred_frames, frames[:, 1:])
        z_var = mx.mean(mx.var(action_latents, axis=0), axis=-1)
        var_penalty = nn.relu(self.var_target - z_var)

        total_loss = recon_loss + self.var_lambda * var_penalty
        return total_loss[0], pred_frames

    def encode(self, frames):
        # frames: [B, T, C, H, W] -> quantized actions: [B, T-1, action_dim]
        return self.quantizer(self.encoder(frames))
