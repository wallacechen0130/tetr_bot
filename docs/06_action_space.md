# 6. Action Space 設計

## 6.1 兩種動作空間

### 低階動作（Low Level）

```text
Left / Right / Rotate CW / Rotate CCW / Soft Drop / Hard Drop / Hold
```

* 優點：最接近真人操作，失誤模型自然（多按一次、晚按 hard drop）。
* 缺點：credit assignment 極難（一個 Tetris 需要 8–10 個正確按鍵），
  探索空間大、收斂慢數倍；同一「意圖」有多種按鍵序列，等於對模型增加雜訊。

### 高階動作（High Level，本專案主線）

```text
(target column, target rotation, hold decision)
```

* 由 `envs/engine/placement.py:enumerate_placements()` 用 BFS（左右移動 + 旋轉含 kick）
  在出生列尋找所有可達狀態，再直線 hard drop 得到落點。
* 編碼：`action = ((column * 4) + rotation) * 2 + hold`，共 **80** 個離散動作。
  `column` 是落點**最左側佔用格**的欄位（0–9），非法組合由 action mask 遮蔽。
* 優點：收斂快、可解釋（每個動作是一個落點）、可直接讓啟發式教師產生標籤、
  與 PPS/APM 控制天然相容（一顆方塊 = 一個動作）。
* 缺點：無法表達「先軟降再 tuck」的少數落點（見 6.3 已知限制）。

## 6.2 比較表

| 面向 | 低階動作 | 高階動作 |
|---|---|---|
| 動作數 | 8（連續按壓） | 80（單次離散） |
| 訓練難度 | 高（階層式 credit assignment） | 低 |
| 收斂速度 | 慢（需 frame stacking、reward shaping 更重） | 快 |
| 最終強度 | 理論上限相同 | 實務上限高（能專注在「決策品質」而非按鍵） |
| 可解釋性 | 低 | 高（可列出每個落點的分數） |
| 與啟發式教師整合 | 難（教師輸出是落點） | 直接對應 |
| 人類化操作 | 天然 | 需經 `compile_placement_plan()` 編譯成按鍵 |
| 強度控制 | 需控制按鍵延遲 | 直接控制落子節奏（PPS） |

**結論：高階動作為訓練主線，低階動作保留為「執行器」。**

## 6.3 已知限制與對策

| 限制 | 說明 | 對策 |
|---|---|---|
| 無 tuck / 深層 spin | BFS 只在出生列搜尋，落地後才旋轉的候選另外列舉，仍可能漏掉需要先下降再橫移的落點 | 以「落地後再旋轉 + kick」補償；Phase 7 可改成完整 state-space BFS |
| 動作索引碰撞 | 不同落點可能對應同一個 `(column, rotation, hold)` | `build_action_map()` 優先保留非 kick 落點，kick 落點只填補空索引 |
| 180 度 kick 表 | TETR.IO 未完整公開 | 使用社群近似值，數值放在 `configs/rules_tetrio.yaml` 並標記 `kicks_180_approximate: true` |
| 無法表達「放慢」 | 高階動作都是瞬時落子 | 由 DifficultyController 的 `delay_s` 與 TokenBucket 控制節奏 |

## 6.4 低階執行器

`envs/gym/low_level.py:compile_placement_plan()` 會把落點編譯成按鍵序列，例如：

```python
KeyPlan(keys=('hold', 'rotate_cw', 'right', 'right', 'hard_drop'))
```

這個序列有兩個用途：

1. **未來 TETR.IO 接入**：交給 `InputController` 實際發送按鍵。
2. **人類化模擬**：在按鍵之間加入 `soft_drop` 與延遲，模擬真人操作節奏。
