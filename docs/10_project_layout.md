# 10. 專案資料夾架構

專案根目錄就是 `C:\Users\Walla\tetr`（不另建一層 `tetrio-ai/`）。

```text
tetr/                          # 專案根目錄
├── configs/                   # 所有可調參數（唯一真實來源）
│   ├── default.yaml           # 全域預設 + 參照其他設定
│   ├── rules_tetrio.yaml      # 棋盤、SRS+、攻擊表、垃圾行規則
│   ├── reward.yaml            # Reward System
│   ├── difficulty.yaml        # 七級難度參數
│   ├── heuristic.yaml         # 啟發式權重與搜尋深度
│   ├── il.yaml                # 模仿學習設定
│   ├── ppo.yaml               # PPO 設定
│   └── logging.yaml           # 日誌與 TensorBoard/W&B 設定
├── envs/                      # 模擬環境
│   ├── config.py              # YAML 載入與合併工具
│   ├── __init__.py            # Gymnasium 環境註冊 + make_env
│   ├── engine/                # 純規則層（不依賴 gym / torch）
│   │   ├── board.py           # 棋盤、消行、形狀特徵
│   │   ├── piece.py           # 方塊幾何與旋轉狀態
│   │   ├── bag.py             # 7-bag 隨機袋
│   │   ├── srs.py             # SRS+ kick 表與 T-Spin 角判定
│   │   ├── rules.py           # 消行分類、攻擊表、B2B / Combo
│   │   ├── garbage.py         # 垃圾行佇列、抵消、落下
│   │   ├── events.py          # 落子事件結構
│   │   ├── placement.py       # 落點列舉（BFS + hard drop）
│   │   └── simulator.py       # 遊戲主邏輯與快照
│   ├── gym/                   # Gymnasium 介面層
│   │   ├── tetris_env.py      # 環境本體（3 種模式）
│   │   ├── obs_encoder.py     # Observation Space
│   │   ├── placement.py       # 高階動作編碼（80 維）
│   │   └── low_level.py       # 高階落點 → 按鍵序列
│   ├── wrappers/              # reward 課程與腳本對手 wrapper
│   └── render/                # ASCII 與 numpy RGB 渲染
├── datasets/                  # 資料集 IO 與資料本體
│   ├── writer.py / reader.py / manifest.py
│   └── heuristic-v1/          # 產生的資料集（不進版控）
├── checkpoints/               # IL / PPO 模型（不進版控）
├── models/                    # 匯出的 ONNX（不進版控）
├── logs/                      # 訓練與評估輸出（不進版控）
├── agents/                    # 決策者
│   ├── base.py                # Agent Protocol / BaseAgent
│   ├── random_agent.py        # 隨機基線
│   ├── heuristic_agent.py     # 啟發式教師（含候選評分）
│   ├── policy_agent.py        # 神經網路 agent
│   └── scripted_agent.py      # 腳本玩家 / 腳本對手模型
├── policies/                  # 神經網路
│   ├── extractors.py          # SmallCNN / ResNet / Attention / Transformer
│   ├── mlp.py                 # MLP 基線
│   ├── action_heads.py        # policy + value head、masked logits
│   ├── factory.py             # TetrisNetwork 與網路工廠
│   └── sb3_extractor.py       # SB3 BaseFeaturesExtractor 介接
├── trainers/                  # 訓練器
│   ├── common.py              # 種子、裝置、checkpoint 工具
│   ├── il_trainer.py          # 模仿學習
│   ├── ppo_trainer.py         # MaskablePPO 微調
│   └── callbacks.py           # checkpoint / 最佳模型 / 課程 / W&B
├── evaluators/                # 評估
│   ├── metrics.py             # 指標定義與彙總
│   ├── benchmark.py           # 對局執行器
│   ├── difficulty_eval.py     # 七級保真度檢查
│   └── report.py              # JSON / Markdown 報告
├── controllers/               # 強度控制
│   ├── difficulty.py          # DifficultyController（主入口）
│   ├── timing.py              # Clock / TokenBucket / ReactionModel
│   ├── misplay.py             # 操作失誤
│   ├── style.py               # 打法風格偏好
│   └── governor.py            # APM 上限治理
├── integrations/              # 外部整合（Phase 6）
│   ├── base.py                # 四個 Protocol 介面
│   ├── mock.py                # 離線假實作
│   └── tetrio/                # TETR.IO stub + 說明
├── notebooks/                 # Colab
│   ├── 00_colab_setup.ipynb
│   ├── 01_il_train_colab.ipynb
│   └── 02_ppo_train_colab.ipynb
├── tests/                     # pytest（100 項）
├── scripts/                   # 可執行入口（python -m scripts.xxx）
│   ├── run_smoke.py           # 端到端 smoke test（38 秒）
│   ├── generate_dataset.py    # 平行產生資料集
│   ├── train_il.py
│   ├── train_ppo.py
│   ├── evaluate.py            # 基準評估 / 難度保真度
│   ├── export_onnx.py         # 匯出 ONNX
│   ├── model_summary.py       # 各架構參數量
│   ├── play_ascii.py          # 終端機觀戰
│   └── sync_drive.py          # Google Drive 同步
└── docs/                      # 本設計文件集
```

## 命名慣例

| 項目 | 慣例 | 範例 |
|---|---|---|
| 模組 / 檔案 | snake_case | `heuristic_agent.py` |
| 類別 | PascalCase | `DifficultyController` |
| 函式 / 變數 | snake_case | `enumerate_placements` |
| 常數 | UPPER_SNAKE_CASE | `N_ACTIONS` |
| 設定鍵 | snake_case | `target_pps` |
| 中文 | 只用於 docstring、註解與文件 | — |

## 不進版控的產物

`.gitignore` 排除 `.venv/`、`logs/`、`checkpoints/*.pt|zip`、`models/*.onnx|pt`、
`datasets/*/shards/`、`datasets/*/data.parquet`、`__pycache__/`、`.pytest_cache/`。
