import h5py
import numpy as np
import mlx.core as mx
from tqdm import tqdm


class H5VideoDataset:
    def __init__(self, path: str, num_frames: int, stride: int = 1, val_fraction: float = 0.02,
                 key: str = "frames", chunk_size: int = 5000):
        self.num_frames = num_frames
        self.stride = stride

        with h5py.File(path, "r") as f:
            dset = f[key]
            n = len(dset)
            self.data = np.empty(dset.shape, dtype=dset.dtype)
            for i in tqdm(range(0, n, chunk_size), desc=f"Loading {path}"):
                j = min(i + chunk_size, n)
                self.data[i:j] = dset[i:j]

        split = int(n * (1 - val_fraction))
        self.ranges = {"train": (0, split), "val": (split, n)}

    def _sample_starts(self, split: str, batch_size: int):
        lo, hi = self.ranges[split]
        max_start = hi - self.num_frames * self.stride
        if max_start <= lo:
            raise ValueError(
                f"Not enough frames in '{split}' split ({hi - lo}) for "
                f"num_frames={self.num_frames} at stride={self.stride}"
            )
        return np.random.randint(lo, max_start, size=batch_size)

    def sample_batch(self, batch_size: int, split: str = "train") -> mx.array:
        starts = self._sample_starts(split, batch_size)
        window = self.num_frames * self.stride
        batch = np.stack([
            self.data[s:s + window:self.stride] for s in starts
        ])  # [B, T, H, W, C]
        batch = batch.astype(np.float32) / 255.0
        batch = (batch - 0.5) / 0.5  # -> [-1, 1]
        batch = np.transpose(batch, (0, 1, 4, 2, 3))  # -> [B, T, C, H, W]
        return mx.array(batch)

    def close(self):
        pass
