import math


def cosine_with_warmup(step: int, *, warmup_steps: int, total_steps: int, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        return min_lr + (1 - min_lr) * cosine
