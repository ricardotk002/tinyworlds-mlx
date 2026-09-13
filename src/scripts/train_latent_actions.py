import argparse
import os
import time

import mlx.core as mx
import mlx.optimizers as optim

from models.latent_actions import LatentActionModel
from utils.data import H5VideoDataset
from utils.scheduler import cosine_with_warmup
from utils.training import train_step


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/topgear.h5")
    p.add_argument("--out", default="checkpoints/latent_actions.safetensors")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--context-length", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-updates", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--frame-height", type=int, default=64)
    p.add_argument("--frame-width", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--n-actions", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    warmup_steps = args.warmup_steps or max(1, args.n_updates // 20)
    frame_size = (args.frame_height, args.frame_width)
    for dim, name in [(args.frame_height, "--frame-height"), (args.frame_width, "--frame-width")]:
        if dim % args.patch_size != 0:
            raise ValueError(f"{name}={dim} must be divisible by --patch-size={args.patch_size}")

    dataset = H5VideoDataset(args.data, num_frames=args.context_length, stride=args.stride)

    model = LatentActionModel(
        frame_size=frame_size,
        n_actions=args.n_actions,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    )
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    def loss_fn(m, batch):
        loss, _ = m(batch, keep_rate=0.0)
        return loss

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start = time.time()
    for step in range(args.n_updates):
        batch = dataset.sample_batch(args.batch_size, split="train")
        lr_frac = cosine_with_warmup(step, warmup_steps=warmup_steps, total_steps=args.n_updates)
        optimizer.learning_rate = lr_frac * args.learning_rate

        model.train()
        loss, grad_norm = train_step(model, optimizer, loss_fn, batch)

        if step % args.log_interval == 0 or step == args.n_updates - 1:
            elapsed = time.time() - start
            print(f"step {step:>7}/{args.n_updates}  loss {loss.item():.4f}  "
                  f"grad_norm {grad_norm.item():.3f}  lr {optimizer.learning_rate:.2e}  "
                  f"elapsed {elapsed:.0f}s")

        if step > 0 and (step % args.checkpoint_interval == 0 or step == args.n_updates - 1):
            model.save_weights(args.out)
            print(f"saved checkpoint to {args.out}")

    dataset.close()


if __name__ == "__main__":
    main()
