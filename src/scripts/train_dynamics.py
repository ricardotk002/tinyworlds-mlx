import argparse
import os
import time
import wandb

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from models.video_tokenizer import VideoTokenizer
from models.dynamics import DynamicsModel
from models.action_encoder import ActionEncoder
from utils.data import H5ActionVideoDataset
from utils.scheduler import cosine_with_warmup
from utils.training import train_step
from models.muon import Muon

class DynamicsWithActionEncoder(nn.Module):
    def __init__(self, dynamics_model, action_encoder):
        super().__init__()
        self.dynamics_model = dynamics_model
        self.action_encoder = action_encoder

    def __call__(self, video_latents, raw_actions, targets):
        conditioning = self.action_encoder(raw_actions)
        _, _, loss = self.dynamics_model(video_latents, training=True, conditioning=conditioning, targets=targets)
        return loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/topgear_actions_140.h5")
    p.add_argument("--video-tokenizer-checkpoint", default="checkpoints/video_tokenizer.safetensors")
    p.add_argument("--out", default="checkpoints/dynamics.safetensors")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--context-length", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-updates", type=int, default=300000)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--checkpoint-interval", type=int, default=2000)

    p.add_argument("--frame-height", type=int, default=140)
    p.add_argument("--frame-width", type=int, default=160)
    p.add_argument("--patch-size", type=int, default=10)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--tokenizer-num-blocks", type=int, default=4)
    p.add_argument("--dynamics-num-blocks", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=5)
    p.add_argument("--num-bins", type=int, default=4)

    p.add_argument("--num-buttons", type=int, default=12)
    p.add_argument("--action-hidden-dim", type=int, default=32)
    p.add_argument("--conditioning-dim", type=int, default=16)

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

    if not os.path.exists(args.video_tokenizer_checkpoint):
        raise FileNotFoundError(
            f"video tokenizer checkpoint not found at {args.video_tokenizer_checkpoint} -- "
            f"train it first (train_video_tokenizer.py)"
        )

    video_tokenizer = VideoTokenizer(
        frame_size=frame_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, hidden_dim=args.hidden_dim, num_blocks=args.tokenizer_num_blocks,
        latent_dim=args.latent_dim, num_bins=args.num_bins,
    )
    video_tokenizer.load_weights(args.video_tokenizer_checkpoint)
    video_tokenizer.eval()

    dynamics_model = DynamicsModel(
        frame_size=frame_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        num_heads=args.num_heads, hidden_dim=args.hidden_dim, num_blocks=args.dynamics_num_blocks,
        num_bins=args.num_bins, latent_dim=args.latent_dim,
        conditioning_dim=args.conditioning_dim,
    )
    action_encoder = ActionEncoder(
        num_buttons=args.num_buttons, hidden_dim=args.action_hidden_dim,
        conditioning_dim=args.conditioning_dim,
    )
    model = DynamicsWithActionEncoder(dynamics_model, action_encoder)
    # optimizer = optim.AdamW(learning_rate=args.learning_rate)
    optimizer = Muon(learning_rate=args.learning_rate, momentum=0.99, weight_decay=0.1)

    dataset = H5ActionVideoDataset(args.data, num_frames=args.context_length, stride=args.stride)

    def loss_fn(m, batch):
        video_latents, targets, raw_actions = batch
        return m(video_latents, raw_actions, targets)

    optim_state_path = os.path.splitext(args.out)[0] + ".optim.safetensors"
    start_step = 0
    wandb_run_id = None

    if os.path.exists(args.out):
        print(f"loaded existing weights from {args.out}")
        model.load_weights(args.out)

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
        model.save_weights(args.out)
        flat_state = dict(tree_flatten(optimizer.state))
        metadata = {"step": str(step + 1)}
        if wandb_run_id:
            metadata["wandb_run_id"] = wandb_run_id
        mx.save_safetensors(optim_state_path, flat_state, metadata=metadata)
        print(f"saved checkpoint to {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    start = time.time()
    for step in range(start_step, args.n_updates):
        frames, raw_actions = dataset.sample_batch(args.batch_size, split="train")

        tokens = mx.stop_gradient(video_tokenizer.tokenize(frames)).astype(mx.int32)  # [B, T, P]
        video_latents = mx.stop_gradient(video_tokenizer.quantizer.get_latents_from_indices(tokens))  # [B, T, P, L]
        transition_actions = raw_actions[:, 1:]  # [B, T-1, num_buttons] -- action into each frame after the first

        lr_frac = cosine_with_warmup(step, warmup_steps=warmup_steps, total_steps=args.n_updates)
        optimizer.learning_rate = lr_frac * args.learning_rate

        model.train()
        loss, grad_norm = train_step(model, optimizer, loss_fn, (video_latents, tokens, transition_actions))

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
