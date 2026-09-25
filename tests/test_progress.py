"""進度條工具與訓練/收集流程的進度整合測試。"""

from __future__ import annotations

from envs.progress import format_seconds, in_notebook, progress_bar, progress_enabled


def test_progress_bar_can_be_disabled():
    bar = progress_bar(total=3, desc="測試", enable=False)
    assert bar.total is None  # NullBar 不追蹤 total，但介面必须完整
    bar.update(1)
    bar.set_postfix(loss=0.5)
    bar.set_description("x")
    bar.close()
    with progress_bar(total=1, enable=False) as inner:
        inner.update(1)


def test_progress_enabled_returns_bool():
    assert progress_enabled(True) is True
    assert progress_enabled(False) is False
    assert isinstance(progress_enabled(None), bool)


def test_notebook_detection_via_colab_env(monkeypatch):
    """Colab 的子行程沒有 ipykernel，但會繼承 COLAB_* 環境變數。"""

    for key in list(__import__("os").environ):
        if key.startswith("COLAB_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-test")
    assert in_notebook() is True
    assert progress_enabled(None) is True


def test_notebook_batch_bar_is_suppressed(monkeypatch):
    """Colab 只留外層 epoch 進度條，避免一堆 widget。"""

    pytest = __import__("pytest")
    pytest.importorskip("torch")
    from trainers.il_trainer import ILConfig, ILTrainer

    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-test")
    trainer = ILTrainer(ILConfig(network="small_cnn", hidden_dim=32, data_root="unused"), device="cpu")
    assert trainer._batch_bar_enabled() is False
    trainer.config.show_progress = False
    assert trainer._batch_bar_enabled() is False


def test_format_seconds():
    assert format_seconds(12.34) == "12.3s"
    assert format_seconds(65) == "1m05s"
    assert format_seconds(3725) == "1h02m"
    assert format_seconds(-5) == "0.0s"


def test_ppo_progress_callback_postfix_reads_logger():
    from trainers.callbacks import TqdmProgressCallback

    class _Logger:
        name_to_value = {"time/fps": 12.5, "rollout/ep_rew_mean": -47.123, "unknown/key": 1.0}

    class _Model:
        logger = _Logger()

    callback = TqdmProgressCallback(enable=False)
    callback.model = _Model()
    postfix = callback._postfix()
    assert postfix["fps"] == "12.5"
    assert postfix["rew"].startswith("-47.1")
    assert "unknown" not in "".join(postfix)


def test_dataset_generator_reports_progress(tmp_path):
    """1 個 worker、1 局、少量方塊 → 驗證 Manager queue 的進度回報不會壞掉。"""

    from datasets.reader import DatasetReader
    from scripts.generate_dataset import generate

    summary = generate(
        out=str(tmp_path / "tiny"),
        episodes=1,
        workers=1,
        depth=1,
        max_pieces=5,
        formats=("npz",),
        name="tiny",
        show_progress=False,
    )
    assert summary["num_samples"] > 0
    reader = DatasetReader(tmp_path / "tiny")
    assert len(reader.arrays()["action"]) == summary["num_samples"]
