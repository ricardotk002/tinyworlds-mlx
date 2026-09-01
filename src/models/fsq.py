# Finite Scalar Quantization — https://arxiv.org/abs/2309.15505
import mlx.core as mx
import mlx.nn as nn


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, latent_dim: int = 5, num_bins: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.codebook_size = num_bins ** latent_dim

        self._basis = self.num_bins ** mx.arange(latent_dim)

    def scale_and_shift(self, z):
        # [-1, 1] -> [0, num_bins - 1]
        return 0.5 * (z + 1) * (self.num_bins - 1)

    def unscale_and_unshift(self, z):
        # [0, num_bins - 1] -> [-1, 1]
        return 2 * z / (self.num_bins - 1) - 1

    def __call__(self, z):
        # tanh -> scale_and_shift -> round -> straight-through -> unscale_and_unshift
        bounded = self.scale_and_shift(mx.tanh(z))
        rounded = mx.round(bounded)
        quantized = bounded + mx.stop_gradient(rounded - bounded)
        return self.unscale_and_unshift(quantized)

    def get_indices_from_latents(self, latents, axis=-1):
        # latents: [..., latent_dim] (values in [-1, 1]) -> indices: [...] (int, in [0, codebook_size))
        unscaled = self.unscale_and_unshift(latents)
        return mx.sum(unscaled * self._basis, axis=axis)

    def get_latents_from_indices(self, indices, axis=-1):
        # indices: [...] (int) -> latents: [..., latent_dim] (values in [-1, 1])
        expanded = (mx.expand_dims(indices, -1) // self._basis) % self.num_bins
        return self.scale_and_shift(expanded)
