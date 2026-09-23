"""神經網路策略 agent：把 observation 丟進模型並在 mask 下取 argmax。"""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.base import BaseAgent


class PolicyAgent(BaseAgent):
    """包裝訓練好的 ``TetrisNetwork``（torch 延遲匯入）。"""

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
            self.network = build_network("resnet")
            if model_path:
                payload = torch.load(model_path, map_location="cpu")
                state = payload.get("model", payload) if isinstance(payload, dict) else payload
                self.network.load_state_dict(state, strict=False)
        self.network.to(self.device)
        self.network.eval()

    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)

    def action_probs(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """回傳 masked softmax 機率（供 controller 重排序）。"""

        torch = self.torch
        board = torch.as_tensor(observation["board"][None], dtype=torch.float32, device=self.device)
        vector = torch.as_tensor(self._vector(observation)[None], dtype=torch.float32, device=self.device)
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
    def _vector(observation: dict[str, np.ndarray]) -> np.ndarray:
        from envs.gym.obs_encoder import FLAT_KEYS

        parts = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in FLAT_KEYS]
        return np.concatenate(parts).astype(np.float32)
