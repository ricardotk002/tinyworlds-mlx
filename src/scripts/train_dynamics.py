import argparse
import math
import os
import time
import wandb

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from models.video_tokenizer import VideoTokenizer
from models.latent_actions import LatentActionModel
from models.dynamics import DynamicsModel
from utils.data import H5VideoDataset
from utils.scheduler import cosine_with_warmup
from utils.training import train_step


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/topgear.h5")
    p.add_argument("--video-tokenizer-checkpoint", default="checkpoints/video_tokenizer.safetensors")
    p.add_argument("--latent-actions-checkpoint", default="checkpoints/latent_actions.safetensors")
    p.add_argument("--out", default="checkpoints/dynamics.safetensors")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--context-length", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-updates", type=int, default=300000)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--checkpoint-interval", type=int, default=2000)

    p.add_argument("--frame-height", type=int, default=64)
    p.add_argument("--frame-width", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--tokenizer-num-blocks", type=int, default=4)
    p.add_argument("--latent-actions-num-blocks", type=int, default=2)
    p.add_argument("--dynamics-num-blocks", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=5)
    p.add_argument("--num-bins", type=int, default=4)
    p.add_argument("--n-actions", type=int, default=4)
    p.add_argument("--wandb-project", default="tinyworlds-dynamics")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    warmup_steps = args.warmup_steps or max(1, args.n_updates // 20)
    frame_size = (args.frame_height, args.frame_width)
    for dim, name in [(args.frame_height, "--frame-height"), (args.frame_width, "--frame-width")]:
        if dim % args.patch_size != 0:
            raise ValueError(f"{name}={dim} must be divisible by --patch-size={args.patch_size}")

    for name, path in [("video tokenizer", args.video_tokenizer_checkpoint),
                        ("latent actions", args.latent_actions_checkpoint)]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} checkpoint not found at {path} -- train it first "
                f"(train_video_tokenizer.py / train_latent_actions.py)"
            )

    video_tokenizer = VideoTokenizer(
        frame_size=frame_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, hidden_dim=args.hidden_dim, num_blocks=args.tokenizer_num_blocks,
        latent_dim=args.latent_dim, num_bins=args.num_bins,
    )
    video_tokenizer.load_weights(args.video_tokenizer_checkpoint)
    video_tokenizer.eval()

    latent_action_model = LatentActionModel(
        frame_size=frame_size, n_actions=args.n_actions, patch_size=args.patch_size,
        embed_dim=args.embed_dim, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
        num_blocks=args.latent_actions_num_blocks,
    )
    latent_action_model.load_weights(args.latent_actions_checkpoint)
    latent_action_model.eval()

    dynamics_model = DynamicsModel(
        frame_size=frame_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, hidden_dim=args.hidden_dim, num_blocks=args.dynamics_num_blocks,
        num_bins=args.num_bins, latent_dim=args.latent_dim,
        conditioning_dim=latent_action_model.action_dim,
    )
    optimizer = optim.AdamW(learning_rate=args.learning_rate)

    dataset = H5VideoDataset(args.data, num_frames=args.context_length, stride=args.stride)

    def loss_fn(m, batch):
        video_latents, targets, conditioning = batch
        _, _, loss = m(video_latents, training=True, conditioning=conditioning, targets=targets)
        return loss

    optim_state_path = os.path.splitext(args.out)[0] + ".optim.safetensors"
    start_step = 0
    wandb_run_id = None

    if os.path.exists(args.out):
        print(f"loaded existing weights from {args.out}")
        dynamics_model.load_weights(args.out)

    if os.path.exists(optim_state_path):
        flat_state, metadata = mx.load(optim_state_path, return_metadata=True)
        optimizer.state = tree_unflatten(list(flat_state.items()))
        start_step = int(metadata["step"])
        wandb_run_id = metadata.get("wandb_run_id") or None
        print(f"resumed optimizer state from {optim_state_path}, continuing from step {start_step}")

    use_wandb = not args.no_wandb
    if use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            config=vars(args),
        )
        wandb_run_id = run.id

    def save_checkpoint(step):
        dynamics_model.save_weights(args.out)
        flat_state = dict(tree_flatten(optimizer.state))
        metadata = {"step": str(step + 1)}
        if wandb_run_id:
            metadata["wandb_run_id"] = wandb_run_id
        mx.save_safetensors(optim_state_path, flat_state, metadata=metadata)
        print(f"saved checkpoint to {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start = time.time()
    for step in range(start_step, args.n_updates):
        frames = dataset.sample_batch(args.batch_size, split="train")

        tokens = mx.stop_gradient(video_tokenizer.tokenize(frames)).astype(mx.int32)  # [B, T, P]
        video_latents = mx.stop_gradient(video_tokenizer.quantizer.get_latents_from_indices(tokens))  # [B, T, P, L]
        conditioning = mx.stop_gradient(latent_action_model.encode(frames))  # [B, T-1, A]

        lr_frac = cosine_with_warmup(step, warmup_steps=warmup_steps, total_steps=args.n_updates)
        optimizer.learning_rate = lr_frac * args.learning_rate

        dynamics_model.train()
        loss, grad_norm = train_step(dynamics_model, optimizer, loss_fn, (video_latents, tokens, conditioning))

        if step % args.log_interval == 0 or step == args.n_updates - 1:
            elapsed = time.time() - start
            loss_val, grad_norm_val, lr_val = loss.item(), grad_norm.item(), optimizer.learning_rate.item()
            print(f"step {step:>7}/{args.n_updates}  loss {loss_val:.4f}  "
                  f"grad_norm {grad_norm_val:.3f}  lr {lr_val:.2e}  "
                  f"elapsed {elapsed:.0f}s")
            if use_wandb:
                wandb.log(
                    {"loss": loss_val, "grad_norm": grad_norm_val, "lr": lr_val, "elapsed": elapsed},
                    step=step,
                )

        if step > 0 and (step % args.checkpoint_interval == 0 or step == args.n_updates - 1):
            save_checkpoint(step)

    dataset.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
