"""可選依賴的隔離測試。

IL 只需要 torch；PPO 才需要 stable-baselines3。
這裡用子行程把 stable_baselines3 標記成不存在，確認 IL 路徑仍可 import。
"""

from __future__ import annotations

import subprocess
import sys

BLOCK_SB3 = (
    "import sys; sys.modules['stable_baselines3'] = None; "
    "sys.modules['sb3_contrib'] = None; "
)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCK_SB3 + code],
        capture_output=True,
        text=True,
        cwd=".",
    )


def test_il_imports_without_stable_baselines3():
    result = _run(
        "from trainers.il_trainer import ILConfig, ILTrainer; "
        "from trainers.common import write_json; "
        "import trainers; "
        "print('ok', ILConfig.__name__, ILTrainer.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_scripts_train_il_module_imports_without_sb3():
    result = _run("import scripts.train_il; print('ok')")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_ppo_attr_is_lazy_and_reports_missing_dependency():
    result = _run(
        "import trainers; "
        "print('import ok'); "
        "\ntry:\n"
        "    trainers.PPOTrainer\nexcept ModuleNotFoundError as exc:\n"
        "    print('lazy-error', 'stable_baselines3' in str(exc))\n"
    )
    assert result.returncode == 0, result.stderr
    assert "import ok" in result.stdout
    assert "lazy-error True" in result.stdout


def test_ppo_trainer_import_when_available():
    try:
        import sb3_contrib  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError:  # pragma: no cover - 沒裝 SB3 的環境
        return
    from trainers.ppo_trainer import PPOTrainer

    assert PPOTrainer is not None
