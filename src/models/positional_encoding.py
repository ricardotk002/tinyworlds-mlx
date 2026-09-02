# Sinusoidal positional encoding — https://arxiv.org/abs/1706.03762
import mlx.core as mx

def sincos_1d(L: int, D: int) -> mx.array:
    # PE[pos, 2i]   = sin(pos / 10000^(2i/D))
    # PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
    assert D % 2 == 0, "Encoding dimension must be even"

    l = mx.arange(L)[:,None]
    d = mx.arange(D)[None,:]

    return mx.where(
        d % 2 == 0,
        mx.sin(l / (10000 ** ((2 * d) / D))),
        mx.cos(l / (10000 ** ((2 * d) / D)))
    )

def sincos_time(T: int, D: int) -> mx.array:
    return sincos_1d(T, D)

def build_spatial_only_pe(frame_size: tuple[int, int], patch_size: int, embed_dim: int) -> mx.array:
    H, W = frame_size
    Hp, Wp = H // patch_size, W // patch_size

    temporal_dim = (embed_dim // 3) & ~1
    spatial_dims = embed_dim - temporal_dim
    spatial_x_dim = (spatial_dims // 2) & ~1
    spatial_y_dim = spatial_dims - spatial_x_dim

    pe_x = sincos_1d(Wp, spatial_x_dim) # Wp, d
    pe_y = sincos_1d(Hp, spatial_y_dim) # Hp, d

    pe_x = pe_x[None, :, :] # 1, Wp, d
    pe_x = mx.broadcast_to(pe_x, (Hp, Wp, spatial_x_dim))
    pe_y = pe_y[:, None, :] # Hp, 1, d
    pe_y = mx.broadcast_to(pe_y, (Hp, Wp, spatial_y_dim))

    return mx.concat([pe_x, pe_y, mx.zeros((Hp, Wp, temporal_dim))], axis=-1).reshape(1, Hp * Wp, -1)
