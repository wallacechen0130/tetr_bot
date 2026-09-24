"""IL loss 回歸測試。

背景：模型 forward 會把不合法動作的 logits 壓到 -1e9，
若直接用 ``torch.nn.functional.cross_entropy(label_smoothing>0)``，
平滑項會對「全部 80 個類別」取平均，把 -1e9 一起平均進來，
實測 loss 會從 ~3.4 爆到 ~1e7（實際訓練時看到 9.4e6）。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from envs.gym.obs_encoder import VECTOR_DIM  # noqa: E402
from policies.action_heads import masked_cross_entropy, masked_logits  # noqa: E402
from policies.factory import build_network  # noqa: E402


def _fake_data(batch: int = 4, hidden_rows: int = 20):
    board = torch.zeros(batch, 2, hidden_rows, 10)
    vector = torch.zeros(batch, VECTOR_DIM)
    mask = torch.zeros(batch, 80)
    mask[:, :38] = 1  # 約 47% 動作不合法，與實際盤面比例一致
    return board, vector, mask


def test_masked_cross_entropy_is_bounded():
    network = build_network("small_cnn")
    board, vector, mask = _fake_data()
    target = torch.zeros(board.shape[0], dtype=torch.long)
    logits, _ = network(board, vector, mask)

    loss = masked_cross_entropy(logits, target, mask, label_smoothing=0.02)
    assert torch.isfinite(loss)
    value = float(loss.detach())
    assert value < 20.0, f"loss 應該在正常範圍，得到 {value}"


def test_plain_cross_entropy_on_masked_logits_explodes():
    """記錄這個 bug 的成因，避免有人改回去。"""

    network = build_network("small_cnn")
    board, vector, mask = _fake_data()
    target = torch.zeros(board.shape[0], dtype=torch.long)
    logits, _ = network(board, vector, mask)

    plain = torch.nn.functional.cross_entropy(logits, target, label_smoothing=0.02)
    value = float(plain.detach())
    assert value > 1e6, "這是 bug 的行為：plain CE + label smoothing 會被 -1e9 汙染"


def test_masked_cross_entropy_without_smoothing_matches_torch():
    network = build_network("small_cnn")
    board, vector, mask = _fake_data()
    target = torch.randint(0, 38, (board.shape[0],), dtype=torch.long)
    logits, _ = network(board, vector, mask)

    ours = masked_cross_entropy(logits, target, mask, label_smoothing=0.0)
    reference = torch.nn.functional.cross_entropy(masked_logits(logits, mask), target)
    assert ours.item() == pytest.approx(reference.item(), rel=1e-5)


def test_il_trainer_loss_is_bounded_including_topk_kl():
    from trainers.il_trainer import ILConfig, ILTrainer

    config = ILConfig(
        epochs=1,
        batch_size=3,
        network="small_cnn",
        hidden_dim=64,
        data_root="unused-in-this-test",
        topk_soft_targets=3,
    )
    trainer = ILTrainer(config, device="cpu")

    board, vector, mask = _fake_data(batch=3)
    batch = {
        "board": board,
        "vector": vector,
        "mask": mask,
        "action": torch.tensor([0, 2, 4], dtype=torch.long),
        "topk_actions": torch.tensor([[0, 2, 4, -1, -1]] * 3, dtype=torch.long),
        "topk_scores": torch.tensor([[3.0, 2.0, 1.0, float("-inf"), float("-inf")]] * 3),
    }
    loss, logits, target = trainer._loss(batch)
    assert torch.isfinite(loss)
    assert float(loss) < 20.0, f"IL loss 應該在正常範圍，得到 {float(loss)}"
    assert logits.shape == (3, 80)
    assert target.shape == (3,)


def test_teacher_actions_outside_mask_are_ignored():
    """教師候選若不在 action mask 內，權重必須歸零而不是變成 -1e9。"""

    from trainers.il_trainer import ILConfig, ILTrainer

    config = ILConfig(network="small_cnn", hidden_dim=64, data_root="unused", topk_soft_targets=3)
    trainer = ILTrainer(config, device="cpu")

    board, vector, mask = _fake_data(batch=2)
    mask[:, 60:] = 1  # 讓索引 0/2/4 變成不合法，只有 60 以後合法
    batch = {
        "board": board,
        "vector": vector,
        "mask": mask,
        "action": torch.tensor([60, 61], dtype=torch.long),
        "topk_actions": torch.tensor([[0, 2, 4, 60, -1]] * 2, dtype=torch.long),
        "topk_scores": torch.tensor([[9.0, 8.0, 7.0, 1.0, float("-inf")]] * 2),
    }
    loss, _logits, _target = trainer._loss(batch)
    assert torch.isfinite(loss)
    assert float(loss) < 30.0
