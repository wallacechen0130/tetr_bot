"""神經網路策略 agent：把 observation 丟進模型並在 mask 下取 argmax。"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base import BaseAgent
from envs.gym.obs_encoder import vector_from_observation


class PolicyAgent(BaseAgent):
    """包裝訓練好的 ``TetrisNetwork``（torch 延遲匯入）。

    載入的是 IL 訓練產生的 ``.pt`` state_dict；PPO 的 SB3 模型是 ``.zip``，
    請改用 :class:`PPOAgent`（或 ``scripts/evaluate.py --agent ppo``）。
    """

    name = "policy"

    def __init__(self, model_path: str | None = None, *, network: Any = None, device: str | None = None, seed: int | None = None) -> None:
        super().__init__(seed)
        import torch

        from policies.factory import build_network

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if network is not None:
            self.network = network
        else:
            if model_path and str(model_path).lower().endswith(".zip"):
                raise ValueError(
                    f"{model_path} 是 stable-baselines3 的壓縮模型，PolicyAgent 無法載入。\n"
                    "請改用：--agent ppo（或 PPOAgent）"
                )
            payload = torch.load(model_path, map_location="cpu", weights_only=False) if model_path else None
            state = payload.get("model", payload) if isinstance(payload, dict) else payload
            network_name = self._infer_network(payload) if state is not None else "resnet"
            self.network = build_network(network_name)
            if state is not None:
                target = self.network.state_dict()
                matched = {
                    key: value
                    for key, value in state.items()
                    if key in target and tuple(target[key].shape) == tuple(value.shape)
                }
                self.network.load_state_dict(matched, strict=False)
                ratio = len(matched) / max(1, len(target))
                print(f"[PolicyAgent] 載入 {model_path}：network={network_name}，{len(matched)}/{len(target)} 個張量")
                if ratio < 0.8:
                    print(
                        f"[PolicyAgent] ⚠️ 只匹配到 {ratio:.0%} 的張量，模型可能與 checkpoint 架構不符；"
                        "請確認 checkpoint 的 network 設定。"
                    )
                extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
                config = extra.get("config", {}) if isinstance(extra, dict) else {}
                if config.get("limit") is not None:
                    print(
                        f"[PolicyAgent] ⚠️ 這是在 limit={config['limit']} 的小樣本上訓練的模型"
                        f"（metrics={payload.get('metrics')}），不是正式訓練結果。"
                    )
        self.network.to(self.device)
        self.network.eval()

    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)

    def action_probs(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """回傳 masked softmax 機率（供 controller 重排序）。"""

        torch = self.torch
        board = torch.as_tensor(observation["board"][None], dtype=torch.float32, device=self.device)
        vector = torch.as_tensor(vector_from_observation(observation)[None], dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(observation["action_mask"][None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, _ = self.network(board, vector, mask)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()[0]

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        probs = self.action_probs(observation)
        mask = np.asarray(observation["action_mask"]).reshape(-1) > 0
        if not mask.any():
            return 0
        masked = np.where(mask, probs, -np.inf)
        return int(np.argmax(masked))

    @staticmethod
    def _infer_network(payload: Any) -> str:
        """從 checkpoint 的設定或 state_dict 推斷網路架構。

        以前這裡固定用 ``resnet``，載入 ``small_cnn`` 的 IL 模型時會因為
        ``strict=False`` 而**無聲地什麼都沒載到**（等於隨機權重）。
        """

        if isinstance(payload, dict):
            extra = payload.get("extra", {})
            if isinstance(extra, dict):
                config = extra.get("config", {})
                if isinstance(config, dict) and config.get("network"):
                    return str(config["network"])
                if extra.get("network"):
                    return str(extra["network"])
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        keys = list(state.keys()) if hasattr(state, "keys") else []

        def has(prefix: str) -> bool:
            return any(key.startswith(prefix) for key in keys)

        if has("board_encoder.conv.0"):
            return "small_cnn"
        if has("board_encoder.stem"):
            return "resnet"
        if has("board_encoder.patch_embed"):
            return "transformer"
        if has("board_encoder.se") or has("board_encoder.spatial"):
            return "cnn_attention"
        return "resnet"


class PPOAgent(BaseAgent):
    """載入 stable-baselines3 的 MaskablePPO 模型（``.zip``）做評估或遊玩。"""

    name = "ppo"

    def __init__(
        self,
        model_path: str,
        *,
        deterministic: bool = True,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)
        from sb3_contrib import MaskablePPO

        self.model_path = str(model_path)
        self.deterministic = bool(deterministic)
        self.model = MaskablePPO.load(self.model_path, device=device)

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> int:
        mask = np.asarray(observation["action_mask"]).reshape(-1) > 0
        action, _ = self.model.predict(observation, deterministic=self.deterministic, action_masks=mask)
        return int(action)
