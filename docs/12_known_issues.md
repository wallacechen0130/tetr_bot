# 12. 修正記錄與已知問題

本檔記錄實測踩到的 bug（含根因與修正後的驗證數字），以及目前尚未解決的限制。

## 已修正

### Bug 1：IL loss 爆到 1e7（`label_smoothing` × masked logits）

**症狀**：訓練時 `train_loss ≈ 9,398,106`、`val_loss ≈ 9,403,019`，top1 只有 0.15。

**根因**：模型 forward 會把不合法動作的 logits 壓到 `-1e9`（`policies/action_heads.py:masked_logits`）。
但 `torch.nn.functional.cross_entropy(label_smoothing>0)` 的平滑項是對**全部 80 個類別**取平均，
於是那些 `-1e9` 被一起平均進來：

```text
實測：CE(no smoothing) = 3.41          ← 正常
      CE(smoothing=0.02) = 10,500,003  ← 爆炸
      mean(log_softmax over 80) = -5.25e8
      0.02 × 5.25e8 ≈ 1.05e7（與觀測值同量級）
```

（非法動作比例約 47%，與實際盤面的 43.6/80 一致。）

**修正**：新增 `policies/action_heads.py:masked_cross_entropy()`，平滑項只在合法動作上平均；
`trainers/il_trainer.py` 改用它，並修正 top-k KL 項（只保留「分數有限且落在 action mask 內」的候選，
教師分數先做 per-sample min-max 正規化再套溫度 softmax）。

**驗證**：`tests/test_il_loss.py`（6 項，其中一項刻意斷言 plain CE 仍會 >1e6，避免有人改回去）。
修正後 loss 回到 2.8–3.7 的正常範圍。

### Bug 2：survival 模式偷偷變成 40L

**症狀**：`TetrisSurvival-v0` 的 episode 在 **40 行**就 `terminated`（`top_out=False`）；
資料集每局只有約 100 顆方塊。

**根因**：`GameConfig.from_dict()` 用
`{k: v for k, v in overrides.items() if v is not None}` 過濾覆寫值，
導致 survival 模式用來「取消行數目標」的 `target_lines=None` 被丟掉，
於是沿用 `configs/default.yaml` 的 `env.target_lines: 40`。

**修正**：改成直接 `env.update(overrides)`，並在註解說明「None 是有效覆寫值（代表取消限制）」。

**驗證**：`tests/test_env_api.py` 新增 2 項測試（各模式限制不互相汙染、survival 不會在 40 行結束）。
修正後 survival 局跑滿 120 秒（240 顆、95 行），單局樣本數從 ~100 提升到 240（2.4 倍）。

### Bug 3：`--no-warm-start` 沒有作用

**症狀**：`python -m scripts.train_ppo --no-warm-start` 仍然載入 IL 權重（log 顯示 `IL warm start：95/95`）。

**根因**：CLI 只把 `il_weights` 設成 `None`，但 `PPOTrainer.run()` 在 `il_weights is None` 時
會回頭讀 `configs/ppo.yaml` 的 `train.il_warm_start`，等於沒有關閉。

**修正**：`scripts/train_ppo.py` 在 `--no-warm-start` 時一併把 config 的 `il_warm_start` 設為 `None`。

## 已知限制

### IL top-1 準確率的先天天花板

教師的最佳落點常常與次佳幾乎同分：

| 指標 | 實測（28,800 樣本） |
|---|---|
| rank1 − rank2 分數差中位數 | 5.48 |
| 差距 < 0.5 的樣本比例 | **20.8%** |
| 差距 < 0.1 的樣本比例 | **13.6%** |

也就是說，約 14–21% 的樣本「標籤本身是雜訊」（兩個落點實質等價，教師只是取排序第一個）。
因此 top-1 精確匹配有結構性上限；本專案另外提供 `top1_tolerant` 指標
（預測落點的教師分數與最佳分數差距 ≤ 0.5 即算正確）來衡量真正的決策品質。

**實測學習曲線**（28,800 樣本、`small_cnn`、batch 512、本機 CPU）：

| epoch | train_loss | val_loss | top1 | top3 | top1_tolerant |
|---|---|---|---|---|---|
| 5 | 3.212 | 3.177 | 0.300 | 0.536 | 0.326 |
| 10 | 2.624 | 2.787 | 0.371 | 0.629 | 0.401 |
| 20 | 1.756 | 2.430 | 0.471 | 0.718 | 0.505 |
| **28（最佳）** | 1.283 | 2.373 | **0.485** | **0.734** | **0.518** |

（修正前同一份程式碼：top1 ≈ 0.15。）

e20–e28 之後 train 明顯低於 val（1.28 vs 2.37）代表**已經開始過擬合**，
要再往上推需要的是**更多資料**而不是更多 epoch：

```powershell
python -m scripts.generate_dataset --episodes 1000 --workers 8 --max-pieces 240   # ~240k 樣本
python -m scripts.train_il --data-root datasets/heuristic-v2 --network resnet --epochs 60
```

建議的階段目標：29k 樣本 → top1 ≥ 0.45；240k 樣本 + `resnet` → top1 ≥ 0.65（尚待實測）。

### 其他既有取捨

| 項目 | 說明 |
|---|---|
| 180° kick 表 | TETR.IO 未完整公開，使用社群近似值（`configs/rules_tetrio.yaml` 標記 `kicks_180_approximate`） |
| tuck 型落點 | 高階動作列舉只涵蓋「出生列 BFS + hard drop」與「落地後再旋轉」，不含先下降再橫移的 tuck |
| `target_apm` | 是**上限**而非目標；實際 APM 需要 PPO 訓練才會追上（見 04 節實測表） |
| TETR.IO 接入 | 僅介面與 stub，未實作螢幕擷取與按鍵注入 |
