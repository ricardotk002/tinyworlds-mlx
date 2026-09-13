import mlx.core as mx
import mlx.optimizers as optim


def train_step(model, optimizer, loss_fn, batch, max_grad_norm: float = 1.0):
    def wrapped(params):
        model.update(params)
        return loss_fn(model, batch)

    loss, grads = mx.value_and_grad(wrapped)(model.trainable_parameters())
    grads, grad_norm = optim.clip_grad_norm(grads, max_grad_norm)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss, grad_norm
