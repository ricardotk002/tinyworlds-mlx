# MaskGIT — https://arxiv.org/pdf/2202.04200
# BERT    — https://arxiv.org/pdf/1810.04805
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .st_transformer import STTransformer
from .positional_encoding import build_spatial_only_pe


def exp_schedule(t: int, T: int, P_total: int, k: float) -> float:
    # Number of prediction-horizon tokens that should be unmasked by step t (of T total
    # steps), following an exponential ramp
    return P_total if t == T - 1 else P_total * mx.expm1(k * t/T) / mx.expm1(k) 


class DynamicsModel(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size: int = 4, embed_dim: int = 128,
                 num_heads: int = 8, hidden_dim: int = 128, num_blocks: int = 4,
                 num_bins: int = 4, latent_dim: int = 5, conditioning_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = num_bins ** latent_dim

        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.transformer = STTransformer(embed_dim, num_heads, hidden_dim, num_blocks, True, conditioning_dim)
        self.output_mlp = nn.Linear(embed_dim, self.codebook_size)

        self._pos_spatial = build_spatial_only_pe(frame_size, patch_size, embed_dim) # non trainable
        self.mask_token = mx.random.normal((1, 1, 1, latent_dim)) * 0.02

    def __call__(self, discrete_latents, training: bool = True, conditioning=None, targets=None):
        # discrete_latents: [B, T, P, L] (continuous FSQ latents, possibly mask-replaced)
        # targets: [B, T, P] int codebook indices (required when training)
        # conditioning: [B, T, C] or None
        # Returns (predicted_logits [B,T,P,codebook_size], mask_positions [B,T,P] or None,
        #          loss (scalar) or None)

        mask = None
        loss = None
        logits = None
        if training and self.training:
            B, T, P = discrete_latents.shape[0], discrete_latents.shape[1], discrete_latents.shape[2]
            mask_ratio = mx.random.uniform(0.5, 1.0)
            mask = mx.random.bernoulli(mask_ratio, shape=[B, T, P])

            anchor_t = mx.random.randint(0, T, shape=[B, P])
            mask = mx.put_along_axis(mask, anchor_t[:, None, :], mx.array(False), axis=1)

            discrete_latents = mx.where(mask[..., None], self.mask_token, discrete_latents)

        x = self.latent_embed(discrete_latents) + self._pos_spatial
        x = self.transformer(x, conditioning)
        logits = self.output_mlp(x)

        if training and self.training:
            loss_per_pos = nn.losses.cross_entropy(logits, targets, reduction="none")
            loss = mx.sum(loss_per_pos * mask) / mx.sum(mask)

        return logits, mask, loss

    def forward_inference(self, context_latents, prediction_horizon: int, num_steps: int,
                           index_to_latents_fn, conditioning=None, schedule_k: float = 5.0,
                           temperature: float = 0.0):
        # context_latents: [B, T_ctx, P, L]. Returns [B, T_ctx + prediction_horizon, P, L].
        B, T_ctx, P, L = context_latents.shape
        T = T_ctx + prediction_horizon
        P_total = prediction_horizon * P

        masked_frames = mx.broadcast_to(self.mask_token, (B, prediction_horizon, P, L))
        buffer_np = np.array(mx.concat([context_latents, masked_frames], axis=1))  # [B, T, P, L]

        mask_np = np.zeros((B, T, P), dtype=bool)
        mask_np[:, T_ctx:, :] = True

        unmasked_count = 0
        b_idx = np.arange(B)[:, None]

        for step in range(num_steps):
            logits, _, _ = self(mx.array(buffer_np), training=False, conditioning=conditioning)
            logits_np = np.array(logits)
            confidence = np.array(mx.max(mx.softmax(logits, axis=-1), axis=-1))  # [B, T, P]

            remaining_masked = P_total - unmasked_count
            target = float(exp_schedule(step, num_steps, P_total, schedule_k))
            n_new = int(np.clip(target - unmasked_count, P_total // 16, remaining_masked))

            if n_new > 0:
                masked_conf = np.where(mask_np, confidence, -np.inf).reshape(B, -1)
                chosen = np.argpartition(-masked_conf, n_new - 1, axis=-1)[:, :n_new]  # [B, n_new]
                t_idx, p_idx = np.unravel_index(chosen, (T, P))

                chosen_logits = mx.array(logits_np[b_idx, t_idx, p_idx])  # [B, n_new, codebook_size]
                if temperature == 0.0:
                    chosen_ids = mx.argmax(chosen_logits, axis=-1)
                else:
                    chosen_ids = mx.random.categorical(chosen_logits / temperature)

                buffer_np[b_idx, t_idx, p_idx] = np.array(index_to_latents_fn(chosen_ids))
                mask_np[b_idx, t_idx, p_idx] = False
                unmasked_count += n_new

            if not mask_np.any():
                break

        if mask_np.any():
            logits, _, _ = self(mx.array(buffer_np), training=False, conditioning=conditioning)
            logits_np = np.array(logits)
            b_r, t_r, p_r = np.nonzero(mask_np)
            final_ids = mx.argmax(mx.array(logits_np[b_r, t_r, p_r]), axis=-1)
            buffer_np[b_r, t_r, p_r] = np.array(index_to_latents_fn(final_ids))

        return mx.array(buffer_np)

