"""模仿學習：Board → Action 的行為克隆訓練。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.reader import DatasetReader, TensorDataset, split_indices
from policies.action_heads import masked_cross_entropy, masked_logits
from policies.factory import build_network, count_parameters
from trainers.common import ensure_dir, load_checkpoint, resolve_device, save_checkpoint, set_seed


@dataclass
class ILConfig:
    """IL 訓練設定（對應 configs/il.yaml）。"""

    data_root: str = "datasets/heuristic-v1"
    checkpoint_dir: str = "checkpoints/il"
    network: str = "resnet"
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-5
    label_smoothing: float = 0.02
    val_fraction: float = 0.1
    early_stopping_patience: int = 5
    topk_soft_targets: int = 3
    topk_weight: float = 0.3
    topk_temperature: float = 0.5
    grad_clip: float = 1.0
    hidden_dim: int = 256
    dropout: float = 0.0
    seed: int = 12345
    num_workers: int = 0
    device: str = "auto"
    limit: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ILConfig:
        data_cfg = data.get("data", {})
        model_cfg = data.get("model", {})
        train_cfg = data.get("train", {})
        ckpt_cfg = data.get("checkpoint", {})
        return cls(
            data_root=str(data_cfg.get("root", cls.data_root)),
            checkpoint_dir=str(ckpt_cfg.get("dir", cls.checkpoint_dir)),
            network=str(model_cfg.get("network", cls.network)),
            batch_size=int(data_cfg.get("batch_size", cls.batch_size)),
            epochs=int(train_cfg.get("epochs", cls.epochs)),
            lr=float(train_cfg.get("lr", cls.lr)),
            weight_decay=float(train_cfg.get("weight_decay", cls.weight_decay)),
            label_smoothing=float(train_cfg.get("label_smoothing", cls.label_smoothing)),
            val_fraction=float(data_cfg.get("val_fraction", cls.val_fraction)),
            early_stopping_patience=int(train_cfg.get("early_stopping_patience", cls.early_stopping_patience)),
            topk_soft_targets=int(train_cfg.get("topk_soft_targets", cls.topk_soft_targets)),
            topk_temperature=float(train_cfg.get("topk_temperature", cls.topk_temperature)),
            hidden_dim=int(model_cfg.get("hidden_dim", cls.hidden_dim)),
            dropout=float(model_cfg.get("dropout", cls.dropout)),
            num_workers=int(data_cfg.get("num_workers", cls.num_workers)),
        )


class ILTrainer:
    """行為克隆訓練器。"""

    def __init__(self, config: ILConfig | dict[str, Any] | None = None, *, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.config = config if isinstance(config, ILConfig) else (
            ILConfig.from_dict(config) if config else ILConfig()
        )
        set_seed(self.config.seed)
        self.device = resolve_device(device or self.config.device)
        self.model = build_network(
            self.config.network,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ 資料
    def build_datasets(self) -> tuple[TensorDataset, TensorDataset]:
        reader = DatasetReader(self.config.data_root, limit=self.config.limit)
        arrays = reader.arrays()
        train_idx, val_idx = split_indices(
            arrays["action"].shape[0],
            val_fraction=self.config.val_fraction,
            seed=self.config.seed,
        )
        return TensorDataset(arrays, train_idx), TensorDataset(arrays, val_idx)

    # ------------------------------------------------------------------ 訓練
    def _loss(self, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
        board = batch["board"].to(self.device)
        vector = batch["vector"].to(self.device)
        mask = batch["mask"].to(self.device)
        target = batch["action"].to(self.device)
        logits, _ = self.model(board, vector, mask)
        # logits 已把不合法動作壓到 -1e9，必須用 masked 版本計算 loss：
        # 若直接呼叫 cross_entropy(label_smoothing>0)，平滑項會把 -1e9 一起平均，
        # 以實際盤面（約 47% 動作不合法）實測會讓 loss 從 ~3.4 爆到 ~1e7。
        loss = masked_cross_entropy(
            logits,
            target,
            mask,
            label_smoothing=self.config.label_smoothing,
        )
        if self.config.topk_soft_targets > 1 and self.config.topk_weight > 0:
            loss = loss + self.config.topk_weight * self._soft_target_kl(batch, logits, mask)
        return loss, logits, target

    def _soft_target_kl(self, batch: dict[str, Any], logits: Any, mask: Any) -> Any:
        """教師 top-k 分佈與學生分佈的 KL 散度（只在合法動作上計算）。

        * 教師分數先做 per-sample min-max 正規化再用溫度 softmax，
          避免 Dellacherie 分數尺度造成極端 one-hot 或數值溢位。
        * 只保留「分數有限」且「在 action mask 內」的候選，其餘權重歸零，
          因此不會出現 -1e9 被乘進 loss 的情況。
        """

        torch = self.torch
        scores = batch["topk_scores"].to(self.device)
        actions = batch["topk_actions"].to(self.device).clamp(0, logits.shape[-1] - 1)
        in_mask = mask.gather(1, actions) > 0
        valid = torch.isfinite(scores) & in_mask
        if not bool(valid.any()):
            return torch.zeros((), device=self.device, dtype=logits.dtype)

        neg_inf = torch.finfo(scores.dtype).min
        scores_valid = torch.where(valid, scores, torch.full_like(scores, neg_inf))
        s_max = scores_valid.amax(dim=1, keepdim=True)
        s_min = torch.where(valid, scores, torch.full_like(scores, float("inf"))).amin(dim=1, keepdim=True)
        span = (s_max - s_min).clamp(min=1e-6)
        normalized = torch.where(valid, (scores_valid - s_min) / span, torch.full_like(scores, neg_inf))
        teacher = torch.softmax(normalized / max(float(self.config.topk_temperature), 1e-3), dim=-1)
        teacher = torch.where(valid, teacher, torch.zeros_like(teacher))
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        student = torch.log_softmax(masked_logits(logits, mask), dim=-1).gather(1, actions)
        kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).sum(dim=-1)
        return kl.mean()

    @staticmethod
    def _accuracy(logits: Any, target: Any, mask: Any, topk: int = 1) -> float:
        torch = __import__("torch")
        masked = logits + (mask <= 0) * -1e9
        top = torch.topk(masked, k=min(topk, masked.shape[-1]), dim=-1).indices
        return float((top == target[:, None]).any(dim=-1).float().mean().item())

    @staticmethod
    def _tolerant_accuracy(
        logits: Any,
        mask: Any,
        topk_actions: Any,
        topk_scores: Any,
        *,
        tolerance: float = 0.5,
    ) -> float:
        """平手容忍準確率。

        教師的最佳落點常常與次佳只差不到 0.5 分（實測 20.8% 的樣本如此），
        這些樣本的 top-1 標籤本質上是雜訊。此指標把「預測落點的教師分數
        與最佳分數差距 ≤ tolerance」也算正確，用來衡量真正的決策品質。
        """

        torch = __import__("torch")
        masked = logits + (mask <= 0) * -1e9
        predicted = masked.argmax(dim=-1)
        best = topk_scores[:, 0]
        predicted_score = torch.full_like(best, float("-inf"))
        for column in range(topk_actions.shape[1]):
            match = topk_actions[:, column] == predicted
            predicted_score = torch.where(match, topk_scores[:, column], predicted_score)
        gaps = best - predicted_score
        finite = torch.isfinite(gaps)
        hits = finite & (gaps <= tolerance)
        return float(hits.float().mean().item())

    def fit(self, *, epochs: int | None = None) -> dict[str, Any]:
        torch = self.torch
        from torch.utils.data import DataLoader

        epochs = int(epochs or self.config.epochs)
        train_ds, val_ds = self.build_datasets()
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        best_top1 = -1.0
        best_path = Path(self.config.checkpoint_dir) / "best.pt"
        ensure_dir(self.config.checkpoint_dir)
        patience = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                loss, _, _ = self._loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                train_loss += float(loss.item())
            train_loss /= max(1, len(train_loader))

            metrics = self.evaluate(val_loader)
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": metrics["val_loss"],
                "top1": metrics["top1"],
                "top3": metrics["top3"],
                "top1_tolerant": metrics["top1_tolerant"],
            }
            self.history.append(record)

            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                patience = 0
                save_checkpoint(
                    best_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=record,
                    extra={"config": self.config.__dict__, "network": self.config.network},
                )
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    break

        return {
            "best_top1": best_top1,
            "history": self.history,
            "checkpoint": str(best_path),
            "parameters": count_parameters(self.model),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        }

    def evaluate(self, loader: Any) -> dict[str, float]:
        torch = self.torch
        self.model.eval()
        total_loss = 0.0
        top1 = 0.0
        top3 = 0.0
        top1_tolerant = 0.0
        batches = 0
        with torch.no_grad():
            for batch in loader:
                loss, logits, target = self._loss(batch)
                mask = batch["mask"].to(self.device)
                total_loss += float(loss.item())
                top1 += self._accuracy(logits, target, mask, topk=1)
                top3 += self._accuracy(logits, target, mask, topk=3)
                top1_tolerant += self._tolerant_accuracy(
                    logits,
                    mask,
                    batch["topk_actions"].to(self.device).clamp(0, logits.shape[-1] - 1),
                    batch["topk_scores"].to(self.device),
                )
                batches += 1
        batches = max(1, batches)
        return {
            "val_loss": total_loss / batches,
            "top1": top1 / batches,
            "top3": top3 / batches,
            "top1_tolerant": top1_tolerant / batches,
        }

    def load(self, path: str | Path) -> dict[str, Any]:
        return load_checkpoint(path, self.model, optimizer=self.optimizer)
