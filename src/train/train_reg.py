# Copyright (c) Yonglan Liu
# Licensed under the MIT License.

import os
from pathlib import Path
import numpy as np
import torch
from loss.Losses import loss_regression_plus_selectivity


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Batch helpers (dict batch)
# =========================
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    def _move(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: _move(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_move(v) for v in x)
        if hasattr(x, "to"):
            try:
                return x.to(device)
            except TypeError:
                return x
        return x
    return _move(batch)


def sanitize_labels(y: torch.Tensor):
    # y -> [B,T] float, mask -> [B,T] float {0,1}
    if y.dim() == 1:
        y = y.view(y.size(0), -1)
    elif y.dim() > 2:
        y = y.view(y.size(0), -1)
    y = y.float()
    mask = torch.isfinite(y)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, mask.to(y.dtype)


def enforce_graphormer_batch(batch: dict, device: torch.device) -> dict:
    """
    Expects Graphormer-ready dict keys like:
      x, in_degree, out_degree, spatial_pos, attn_edge_type, edge_input, attn_bias, y
    Moves to device and forces correct dtypes for embedding indices.
    """
    batch = move_batch_to_device(batch, device)

    # Index tensors used in embeddings must be long
    index_keys = ["x", "in_degree", "out_degree", "spatial_pos", "attn_edge_type", "edge_input"]
    for k in index_keys:
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device, non_blocking=True)
            if batch[k].dtype != torch.long:
                batch[k] = batch[k].long()

    # attn_bias should be float; also avoid -inf NaN issues by clamping infs to large finite
    if "attn_bias" in batch and torch.is_tensor(batch["attn_bias"]):
        batch["attn_bias"] = batch["attn_bias"].to(device, non_blocking=True).float()
        batch["attn_bias"] = torch.nan_to_num(batch["attn_bias"], nan=0.0, posinf=1e4, neginf=-1e4)

    return batch



# =========================
# Optimizer: different LR for encoder vs adaptor/head
# =========================
def make_param_groups(
    model: torch.nn.Module,
    *,
    encoder_lr: float,
    adaptor_lr: float,
    weight_decay: float = 0.01,
    adaptor_name_keywords: tuple[str, ...] = ("soft_share", "adaptor", "adapter", "head", "heads"),
    freeze_encoder: bool = False,
):
    """
    Splits parameters into:
      - encoder group (lower lr)
      - adaptor/head group (higher lr)
    Uses parameter names to decide.
    """
    enc, adap = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_adaptor = any(k in name for k in adaptor_name_keywords)
        if is_adaptor:
            adap.append(p)
        else:
            enc.append(p)

    if freeze_encoder:
        for p in enc:
            p.requires_grad_(False)
        enc = []

    param_groups = []
    if enc:
        param_groups.append({"params": enc, "lr": encoder_lr, "weight_decay": weight_decay})
    if adap:
        param_groups.append({"params": adap, "lr": adaptor_lr, "weight_decay": weight_decay})

    if not param_groups:
        raise ValueError("No trainable parameters found. Check freeze flags / requires_grad.")

    # Helpful print
    n_enc = sum(p.numel() for p in enc)
    n_adp = sum(p.numel() for p in adap)
    print(f"Param groups: encoder={n_enc:,} params @ lr={encoder_lr} | adaptor/head={n_adp:,} params @ lr={adaptor_lr}")
    return param_groups


# =========================
# Train/Eval
# =========================
def train_step(
    model,
    batch: dict,
    optimizer,
    *,
    device,
    lambda_aux: float,
    huber_delta: float,
    grad_clip: float | None,
    debug_nan: bool = False,
    min_valid_labels: int = 10,   # skip batches with too few valid labels
):
    """
    One training step for a dict batch.

    Returns:
        (loss_value: float, batch_size: int)
    """
    model.train()

    # Move/cast batch tensors correctly for Graphormer
    batch = enforce_graphormer_batch(batch, device)

    # Labels -> finite mask; replace NaN/Inf labels with 0 so loss is well-defined under mask
    y, mask = sanitize_labels(batch["y"])
    y = y.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    bs = int(y.size(0))

    # --- Skip very sparse-label batches (prevents loss spikes / instability) ---
    valid_ct = int(mask.sum().item())
    if valid_ct < min_valid_labels:
        # No optimizer step; return "0 loss" so outer loop keeps working
        return 0.0, bs

    optimizer.zero_grad(set_to_none=True)

    # Forward
    out = model(batch)
    pred = out[:, 0, :] if out.dim() == 3 else out  # [B,T] or [B,N,T] -> [B,T]

    if debug_nan and (not torch.isfinite(pred).all()):
        raise RuntimeError("pred has NaN/Inf")

    # Loss
    loss = loss_regression_plus_selectivity(
        pred=pred,
        y=y,
        mask=mask,
        lambda_delta=lambda_aux,
        huber_delta=huber_delta,
    )

    if debug_nan and (not torch.isfinite(loss)):
        raise RuntimeError(f"loss is NaN/Inf: {loss.item()}")

    # Backward + step
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item()), bs



@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    device,
    lambda_aux: float,
    huber_delta: float,
    debug_nan: bool = False,
):
    model.eval()
    running, denom = 0.0, 0

    for batch in loader:
        batch = enforce_graphormer_batch(batch, device)
        y, mask = sanitize_labels(batch["y"])
        y = y.to(device)
        mask = mask.to(device)

        out = model(batch)
        pred = out[:, 0, :] if out.dim() == 3 else out

        loss = loss_regression_plus_selectivity(
            pred=pred, y=y, mask=mask, lambda_delta=lambda_aux, huber_delta=huber_delta
        )

        if debug_nan and (not torch.isfinite(loss)):
            raise RuntimeError(f"eval loss is NaN/Inf: {loss.item()}")

        bs = int(y.size(0))
        running += float(loss.item()) * bs
        denom += bs

    return running / max(1, denom)


def train_with_eval_earlystop(
    model,
    train_loader,
    valid_loader,
    *,
    device,
    optimizer,
    scheduler=None,
    epochs: int = 50,
    grad_clip: float | None = 1.0,
    lambda_aux: float = 0.5,
    huber_delta: float = 1.0,
    save_dir: str = "./runs",
    run_name: str = "graphormer_regress",
    save_every_epochs: int = 1,
    patience: int = 10,
    min_delta: float = 0.0,
    debug_nan: bool = False,
):
    save_dir = Path(save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"epoch": [], "train_loss": [], "valid_loss": []}
    best_val = float("inf")
    best_epoch = 0
    global_step = 0

    for epoch in range(1, epochs + 1):
        running, denom = 0.0, 0

        for batch in train_loader:
            loss_item, bs = train_step(
                model, batch, optimizer,
                device=device,
                lambda_aux=lambda_aux,
                huber_delta=huber_delta,
                grad_clip=grad_clip,
                debug_nan=debug_nan,
            )
            running += loss_item * bs
            denom += bs
            global_step += 1

        train_loss = running / max(1, denom)
        valid_loss = evaluate(
            model, valid_loader,
            device=device,
            lambda_aux=lambda_aux,
            huber_delta=huber_delta,
            debug_nan=debug_nan,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        lrs = [g["lr"] for g in optimizer.param_groups]
        print(f"[Epoch {epoch:03d}] train={train_loss:.6f}  valid={valid_loss:.6f}  lrs={lrs}")


        # Step scheduler (choose style you use)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()


        # periodic checkpoint
        if epoch % save_every_epochs == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "history": history,
                },
                save_dir / f"checkpoint_epoch{epoch:03d}.pt",
            )

        # best checkpoint
        improved = (best_val - valid_loss) > min_delta
        if improved:
            best_val = valid_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "history": history,
                },
                save_dir / "checkpoint_best.pt",
            )

        torch.save(history, save_dir / "history.pt")

        # early stop
        if (epoch - best_epoch) >= patience:
            print(
                f"Early stopping: no improvement for {patience} epochs "
                f"(best at epoch {best_epoch}, val={best_val:.6f})."
            )
            break

    return history
