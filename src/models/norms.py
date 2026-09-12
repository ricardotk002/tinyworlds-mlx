# RMSNorm — https://arxiv.org/abs/1910.07467
# FiLM     — https://arxiv.org/abs/1709.07871
import mlx.core as mx
import mlx.nn as nn
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(embed_dim)

    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps) * self.weight


class SimpleLayerNorm(nn.Module):
    # Zero-mean, unit-variance. No learned affine — FiLM supplies that externally.
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x):
        return (x - mx.mean(x, axis=-1, keepdims=True)) * mx.rsqrt(mx.var(x, axis=-1, keepdims=True) + self.eps)


class AdaptiveNormalizer(nn.Module):
    # Unconditioned -> RMSNorm. Conditioned -> FiLM (SimpleLayerNorm + predicted gamma/beta).
    def __init__(self, embed_dim: int, conditioning_dim: int | None = None):
        super().__init__()
        self.rms = None
        self.ln = None
        self.to_gamma_beta = None

        if conditioning_dim is None:
            self.rms = RMSNorm(embed_dim)
        else:
            self.ln = SimpleLayerNorm(embed_dim)
            linear = nn.Linear(conditioning_dim, 2 * embed_dim)
            linear.weight = nn.init.normal(std=1e-3)(linear.weight)
            linear.bias = nn.init.constant(0.0)(linear.bias)
            self.to_gamma_beta = nn.Sequential(nn.SiLU(), linear)
            

    def __call__(self, x, conditioning=None):
        # x: [B, T, P, E]
        # conditioning: [B, T, C] or [B, T-1, C]
        if conditioning is None:
            return self.rms(x)
        else:
            x = self.ln(x)

            if x.shape[-3] - 1 == conditioning.shape[-2]:
                conditioning = mx.pad(conditioning, ((0, 0), (1, 0), (0, 0)))

            gamma, beta = mx.split(self.to_gamma_beta(conditioning), 2, axis=-1)
            gamma = mx.expand_dims(gamma, axis=-2)
            beta = mx.expand_dims(beta, axis=-2)

            return x * (1 + gamma) + beta
