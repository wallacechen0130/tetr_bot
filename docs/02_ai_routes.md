# 2. AI 路線評估

## 2.1 九軸比較表

評分 1–5（5 為最高）；「訓練成本」越高越貴、「Colab 適合度」越高越適合免費 T4。

| 路線 | 開發難度 | 訓練成本 | GPU 需求 | Colab 適合度 | 收斂速度 | 最終上限 | 強度控制能力 | 維護難度 | 建議用途 |
|---|---|---|---|---|---|---|---|---|---|
| Heuristic Bot | 2 | 0（無需訓練） | 無 | 5 | 立即 | 3 | 5（權重/深度即強度） | 2 | **MVP 主線、資料教師** |
| Genetic Algorithm | 3 | 中（CPU 大量對局） | 無 | 3 | 慢 | 3 | 3 | 3 | 選配：調啟發式權重 |
| Imitation Learning | 2 | 低（幾百 MB 樣本） | 低 | 5 | 快（1–2 小時） | 4 | 4 | 2 | **MVP 主線：Board → Action** |
| DQN | 4 | 高（樣本效率差） | 中 | 3 | 慢 | 3 | 3 | 4 | 不推薦（見 2.3） |
| Double DQN | 4 | 高 | 中 | 3 | 慢 | 3 | 3 | 4 | 不推薦 |
| PPO | 3 | 中 | 中 | 4 | 中 | 5 | 5 | 3 | **MVP 主線：超越教師** |
| A2C | 3 | 中 | 低 | 4 | 快但不穩 | 3 | 4 | 3 | 僅作 PPO 的 ablation |
| Transformer Policy | 4 | 高 | 高 | 2 | 慢 | 5 | 4 | 4 | Phase 7 研究項（已附實作） |

## 2.2 各路線評語

**Heuristic Bot**：以 Dellacherie 特徵（landing height、eroded piece cells、row/column transitions、holes、well sums）
加權評分，深度 1 即可穩定存活。它同時是**免費且無限的教師**，這是本專案選它當起點的主因。
實測：40L 完成率 100%、平均 51.2 秒、終局洞數 0。

**Genetic Algorithm**：把權重當染色體、以「存活行數」為 fitness 做演化。
可行性高但耗 CPU，且天花板受限於特徵設計；適合在 Phase 2 之後當作權重微調工具（`configs/heuristic.yaml` 可直接被覆寫）。

**Imitation Learning**：以教師產生的 `State → Best Move` 做分類。收斂快、可在 CPU 完成、
輸出可解釋（動作分佈），缺點是天花板等於教師（會一併學到教師的保守風格）。

**DQN / Double DQN**：Tetris 的 reward 稀疏且延遲（現在墊高是為了三顆後的 Tetris），
value-based 方法需要大量互動才能把信用分配傳回去；再加上我們用高階動作空間，
`argmax over 80 actions` 的過度估計問題明顯。Double DQN 只緩解不解決，故兩者都不列入主線。

**PPO**：on-policy、對 reward shaping 容忍度高、支援 action masking（`sb3-contrib`），
最適合「從教師起點往上推」的微調；也是唯一能同時學到「高效率堆疊 → Tetris → 攻擊」的策略路線。

**A2C**：PPO 的簡化版，訓練快但樣本效率與穩定性差，只保留為對照組。

**Transformer Policy**：對棋盤的長距離關係（例如左右兩側的洞與 well）有表達優勢，
但 20×10 的規模用 ResNet 已足夠，Transformer 在 CPU 上慢得不值得；保留實作供 Phase 7 比較。

## 2.3 推薦方案（本專案主線）

```text
Heuristic Teacher（產生資料 + 可立即遊玩）
        ↓ 模仿學習（Board → Action）
IL 模型（達到教師水準）
        ↓ PPO 微調（MaskablePPO + 攻擊導向 reward）
PPO 模型（超越教師：更高 APM、更積極的 Tetris/T-Spin 選擇）
```

理由：

1. **風險最低**：Phase 2 結束就有能玩的 AI，不必等訓練成功。
2. **成本最低**：資料產生與 IL 可在本機 CPU 完成，Colab 只用於 PPO。
3. **上限最高**：IL 保證起點，PPO 突破教師；兩者共用同一份 observation/action 介面。
4. **強度控制最好**：DifficultyController 直接作用在「候選落點排序 + 出手時機」，與模型解耦。
