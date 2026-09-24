# 3. 最佳專案路線（Phase 1–6）

每個階段都對應可執行的程式碼與測試，時間估計以「每天 1–2 小時的個人開發」為基準。

## Phase 1：建立 Tetris Engine（已完成）

| 項目 | 內容 |
|---|---|
| 工作內容 | Board / Piece / SRS+ 旋轉含 180 kick / Hold / 7-bag / B2B / Combo / T-Spin 判定 / 垃圾行 / Top Out |
| 技術重點 | 純 numpy 實作、seed 可重現、規則全部放 `configs/rules_tetrio.yaml` |
| 預估時間 | 1–2 週 |
| 難度 | ★★☆☆☆ |
| 預期成果 | `envs/engine/`，`tests/test_engine_*.py` 全綠 |
| 實測 | 24 項 engine 測試通過；單次落點列舉 ~0.2 ms |

關鍵設計：

* 座標系 `(row, col)`，row 由上往下；I/O 用 4×4、其餘用 3×3 的 bounding box，讓 SRS kick 表可直接套用。
* T-Spin 用 3-corner 規則 + 面向側兩角判定 mini；若使用第 5 個 kick 則升為完整 T-Spin。
* 攻擊表、B2B 加成、Combo 表、Perfect Clear 全部可在 YAML 調整，並由測試鎖定數值。

## Phase 2：建立 Heuristic AI（已完成）

| 項目 | 內容 |
|---|---|
| 工作內容 | 特徵評估（Height / Holes / Bumpiness / Well / T-Spin 潛力 / PC 潛力）+ 深度 1–3 搜尋 |
| 技術重點 | Dellacherie 經典權重為骨架，`configs/heuristic.yaml` 可調；落點列舉用 BFS + hard drop |
| 預估時間 | 1 週 |
| 難度 | ★★☆☆☆ |
| 預期成果 | `agents/heuristic_agent.py`，可完成 40L 並穩定存活 |
| 實測 | 40L 完成率 100%、平均 51.2 秒（2 PPS）、46.9 行/分、終局洞數 0、hold 使用率 0.43 |

## Phase 3：自動資料產生（已完成）

| 項目 | 內容 |
|---|---|
| 工作內容 | 以教師平行產生 `State → Best Move`，輸出 npz 分片與 parquet，附 manifest |
| 技術重點 | multiprocessing（每 worker 寫自己的 shard，最後合併 manifest）；board 用 uint8、vector 用 float16 壓縮 |
| 預估時間 | 3–5 天 |
| 難度 | ★★☆☆☆ |
| 預期成果 | `datasets/heuristic-v1/` |
| 實測 | 2 workers、80 樣本 1.5 秒；每樣本約 0.85 KB |

進度顯示：`收集資料: 42%|█████▍ | 5/12 [00:01<00:02, 3.22episode/s, samples=560, rate=112 sample/s]`（tqdm 提供 ETA）。

規模建議：正式訓練用 20–50 萬樣本（約 170–430 MB npz），以 8 workers 約 15–30 分鐘。

## Phase 4：Imitation Learning（已完成）

| 項目 | 內容 |
|---|---|
| 工作內容 | `Board + Vector → Action` 的分類訓練，含 mask、top-k soft target、early stopping |
| 技術重點 | masked cross-entropy + KL 到教師 top-k 分佈；ResNet 抽取器可與 PPO 共用 |
| 預估時間 | 1 週 |
| 難度 | ★★☆☆☆ |
| 預期成果 | `checkpoints/il/best.pt` |
| 驗收門檻 | 29k 樣本：top-1 ≥ 0.45、top-3 ≥ 0.70、top1_tolerant ≥ 0.50（受限於等價標籤，見 [12_known_issues.md](12_known_issues.md)） |
| 實測 | 28.8k 樣本 / 28 epochs：top-1 **0.485**、top-3 **0.734**、平手容忍 **0.518**（修 bug 前僅 0.151） |

進度顯示：外層 epoch 進度條（含 ETA 與 val 指標）＋內層 batch 進度條（即時 train loss）。

## Phase 5：PPO Fine-Tuning（已完成程式碼，訓練待放大）

| 項目 | 內容 |
|---|---|
| 工作內容 | MaskablePPO 微調；課程式 reward（前 20% 偏存活、後段偏攻擊） |
| 技術重點 | 自訂 `TetrisFeaturesExtractor` 載入 IL 權重；`SubprocVecEnv` 8 環境；CPU 8–74 fps |
| 預估時間 | 1–2 週（含 Colab 訓練時間） |
| 難度 | ★★★☆☆ |
| 預期成果 | `checkpoints/ppo/best.zip`，APM 明顯高於教師 |
| 實測 | 2 envs / 2048 步 smoke 通過，IL 權重 95/95 張量載入 |

## Phase 6：接入 TETR.IO（設計完成、未實作）

| 項目 | 內容 |
|---|---|
| 工作內容 | 螢幕擷取 → 棋盤辨識 → 按鍵控制 → 延遲控制 |
| 技術重點 | mss 擷取視窗、OpenCV 色彩遮罩 + 模板比對、pynput/SendInput 注入、TokenBucket 控制節奏 |
| 預估時間 | 2–4 週 |
| 難度 | ★★★★☆ |
| 預期成果 | 能在離線 40L / Blitz 自動遊玩 |
| 目前狀態 | 僅提供 Protocol 介面（`integrations/base.py`）與 stub |

風險與對策：

| 風險 | 對策 |
|---|---|
| 服務條款禁止線上自動化 | 只做離線練習模式；多人對戰不實作 |
| 解析度 / DPI / 視窗縮放 | 以棋盤外框自動校正網格，失敗則要求使用者提供截圖 |
| 辨識錯誤導致亂放 | 加入辨識信心值與「不合理盤面」防呆（與上一幀差異過大時跳過該幀） |
| 反作弊機制 | 不繞過、不注入遊戲記憶體；僅使用畫面與鍵盤輸入 |

## 里程碑驗收（本機 CPU 實測）

| 驗收項目 | 門檻 | 實測 |
|---|---|---|
| `pytest -q` | 全綠 | 100 項測試通過 |
| `python -m scripts.run_smoke` | < 3 分鐘 | 38.5 秒 |
| 啟發式 40L | 完成且 30–60 秒 | 100% / 51.2 秒 |
| Blitz 存活 2 分鐘 | 不 Top Out | 20 秒測試局未 Top Out（可延長） |
| 難度 PPS 保真度 | 誤差 ≤ 5% | 最大 0.8% |
| 難度 APM | 不超過目標（上限治理） | 七級全部符合；實際達成值見 04 |
