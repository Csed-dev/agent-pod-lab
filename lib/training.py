import math
import time
from dataclasses import dataclass

import torch

from prepare import TIME_BUDGET


@dataclass
class TrainConfig:
    lr: float
    weight_decay: float = 1e-4
    matrices_per_epoch: int = 16
    num_probes: int = 8
    loss_skip_threshold: float = 50.0
    warmup_epochs: int = 20
    min_lr_ratio: float = 0.1
    checkpoint_path: str = "best_model.pt"
    optimizer_class: str = "adam"  # "adam" or "adamw"


@dataclass
class TrainResult:
    best_loss: float
    num_epochs: int
    training_seconds: float
    checkpoint_path: str


def train_loop(model, dataset, loss_fn, save_fn, config: TrainConfig) -> TrainResult:
    opt_cls = torch.optim.AdamW if config.optimizer_class == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    estimated_epochs = int(TIME_BUDGET / 2.0)

    def lr_lambda(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        progress = (epoch - config.warmup_epochs) / max(1, estimated_epochs - config.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    total_training_time = 0.0
    epoch = 0
    data_iter = iter(dataset)
    smooth_loss = 0.0
    skipped_count = 0

    while True:
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        valid_count = 0

        for _ in range(config.matrices_per_epoch):
            data = next(data_iter)
            A = torch.sparse_coo_tensor(
                data.indices, data.values[0], (data.n, data.n)
            ).coalesce().to_sparse_csc()

            loss = loss_fn(model, A, config.num_probes)
            loss_val = loss.item()

            if not math.isfinite(loss_val) or loss_val > config.loss_skip_threshold:
                skipped_count += 1
                continue

            epoch_loss += loss_val
            valid_count += 1

            scaled_loss = loss / config.matrices_per_epoch
            scaled_loss.backward()

        if valid_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        avg_loss = epoch_loss / max(valid_count, 1)

        if avg_loss < best_loss and valid_count > 0:
            best_loss = avg_loss
            save_fn(model, config.checkpoint_path)

        t1 = time.time()
        dt = t1 - t0

        if epoch > 5:
            total_training_time += dt

        ema_beta = 0.95
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * avg_loss
        debiased = smooth_loss / (1 - ema_beta ** (epoch + 1))

        remaining = max(0, TIME_BUDGET - total_training_time)
        current_lr = scheduler.get_last_lr()[0]
        print(f"\repoch {epoch:04d} | loss: {debiased:.4e} | best: {best_loss:.4e} | lr: {current_lr:.1e} | skip: {skipped_count} | dt: {dt*1000:.0f}ms | {remaining:.0f}s    ", end="", flush=True)

        epoch += 1

        if epoch > 5 and total_training_time >= TIME_BUDGET:
            break

    print()
    return TrainResult(
        best_loss=best_loss,
        num_epochs=epoch,
        training_seconds=total_training_time,
        checkpoint_path=config.checkpoint_path,
    )
