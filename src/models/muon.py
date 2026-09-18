import mlx.core as mx
import mlx.optimizers as optim

def zeropower_via_newtonschulz5(G: mx.array, steps: int = 5, eps: float = 1e-7) -> mx.array:
    # G: [rows, cols], 2D. Returns a matrix of the same shape whose singular values have
    # been pushed towards 1 (i.e. close to a semi-orthogonal matrix), via 5 (default) steps
    # of the quintic Newton-Schulz iteration: X <- a*X + (b*A + c*A^2) @ X, A = X @ X^T,
    # with a, b, c = 3.4445, -4.7750, 2.0315.
    assert G.ndim == 2

    orig_type = G.dtype
    G = G.astype(mx.bfloat16)
    transposed = False

    if G.shape[0] > G.shape[1]:
        transposed = True
        G = G.T

    G = G / mx.sqrt(mx.sum(mx.square(G))) + eps
    a, b, c = 3.4445, -4.7750, 2.0315  

    for _ in range(steps):
        A = G @ G.T
        G = a * G + (b * A + c * (A @ A)) @ G

    G = G.astype(orig_type)

    return G.T if transposed else G

class Muon(optim.Optimizer):
    def __init__(self, learning_rate, momentum: float = 0.95, weight_decay: float = 0.01,
                 backend_steps: int = 5, nesterov: bool = True):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.backend_steps = backend_steps
        self.nesterov = nesterov

    def init_single(self, parameter, state):
        state["m"] = mx.zeros_like(parameter)

    def apply_single(self, gradient, parameter, state):
        lr = self.learning_rate.astype(gradient.dtype)
        buf = state["m"]
        buf = buf * self.momentum + gradient
        state["m"] = buf

        g = (gradient + self.momentum * buf) if self.nesterov else buf

        if parameter.ndim >= 2:
            g = zeropower_via_newtonschulz5(g)
            g = g * max(1, g.shape[-2] / g.shape[-1]) ** 0.5
        else:
            g = buf

        return parameter * (1 - lr * self.weight_decay) - lr * g
