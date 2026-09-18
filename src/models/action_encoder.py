import mlx.core as mx
import mlx.nn as nn


class ActionEncoder(nn.Module):
    def __init__(self, num_buttons: int = 12, hidden_dim: int = 32, conditioning_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(num_buttons, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, conditioning_dim)
        self.conditioning_dim = conditioning_dim

    def __call__(self, actions):
        # actions: [..., num_buttons] (0/1) -> [..., conditioning_dim]
        x = nn.silu(self.fc1(actions.astype(mx.float32)))
        return self.fc2(x)
