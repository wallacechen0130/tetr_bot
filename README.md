# TETR.IO 風格 Tetris AI（設計文件 + 可執行 MVP）

一個以「職業玩家風格」為目標的 Tetris AI 專案：自建 TETR.IO 風格規則引擎、
啟發式教師、模仿學習與 PPO 微調，並內建七級強度控制（PPS / APM / 反應時間 / 失誤 / 打法風格）。

> ⚠️ **TETR.IO 服務條款禁止在線上多人對戰使用自動化程式。**
> 本專案目前**完全沒有**接入 TETR.IO：只提供介面（`integrations/`）與設計文件，
> 未來若實作，目標僅限離線練習模式（40L / Blitz）。

## 快速開始（Windows / 本機 CPU）

```powershell
cd C:\Users\Walla\tetr

# 1) 建立虛擬環境並安裝依賴
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 跑測試（約 16 秒）
.\.venv\Scripts\python.exe -m pytest tests -q

# 3) 端到端 smoke test（引擎 → 教師 → 資料集 → IL → PPO → 評估，約 40 秒）
.\.venv\Scripts\python.exe -m scripts.run_smoke
```

## 常用指令

| 目的 | 指令 |
|---|---|
| 看 AI 打 40L（ASCII） | `python -m scripts.play_ascii --env-id Tetris40L-v0 --tier Advanced` |
| 產生訓練資料 | `python -m scripts.generate_dataset --episodes 20 --workers 8` |
| 模仿學習 | `python -m scripts.train_il --data-root datasets/heuristic-v1` |
| PPO 微調 | `python -m scripts.train_ppo --timesteps 200000 --vec subproc` |
| 評估（含七級保真度） | `python -m scripts.evaluate --agent heuristic --env-id Tetris40L-v0 --episodes 5` |
| 難度保真度報表 | `python -m scripts.evaluate --difficulty-suite` |
| 網路參數量 | `python -m scripts.model_summary` |
| 匯出 ONNX | `python -m scripts.export_onnx --checkpoint checkpoints/il/best.pt` |
| 與 Google Drive 同步 | `python -m scripts.sync_drive --direction to_drive` |

### Google Drive 路徑怎麼填

Google Drive 桌面版的「我的雲端硬碟」資料夾**名稱會跟著系統語系**，而且**不叫 `MyDrive`**：

| 環境 | 正確路徑 |
|---|---|
| 繁體中文版 Windows | `G:\我的雲端硬碟\tetrio-ai` |
| 英文版 Windows | `G:\My Drive\tetrio-ai` |
| Colab | `/content/drive/MyDrive/tetrio-ai`（Colab 固定是 `MyDrive`） |

不確定路徑時先讓工具找給你：

```powershell
python -m scripts.sync_drive --list-drives
# 偵測到的 Google Drive 資料夾：
#   G:\我的雲端硬碟
#     建議的專案路徑：--drive "G:\我的雲端硬碟\tetrio-ai"
```

路徑寫錯（例如打成 `G:\MyDrive`）時會直接告訴你正確候選路徑，不會再丟出
`FileNotFoundError: [WinError 2]`。工具也會自動處理重複的反斜線、`/` 混用、
路徑前後的引號，以及 `~/` 展開。

## 進度與 ETA

資料收集、IL 訓練、PPO 訓練與評估都內建 tqdm 進度條，附**預估剩餘時間（ETA）**：

```text
收集資料:  42%|█████▍      | 5/12 [00:01<00:02, 3.22episode/s, samples=560, rate=112 sample/s]
IL 訓練:   50%|█████████▌  | 1/2 [00:03<00:03, 3.42s/epoch, train_loss=4.192, val_loss=3.919, top1=0.153]
PPO 訓練:  51%|██████      | 1036/2048 [00:18<00:57, 17.49step/s, kl=0.0001487]
評估:      67%|███████▎    | 2/3 [00:00<00:00, 12.44episode/s, lines=0.0, rew=-54.6]
```

* 輸出被重導向、在 CI 或 pytest 底下執行時會**自動關閉**，不會汙染日誌。
* 在 Colab / Jupyter 會自動啟用：kernel 內執行時是原生 widget 進度條，子行程則以 `\r` 重畫同一行。
* 需要關掉時可加 `--no-progress`：`python -m scripts.train_il --no-progress`。

## 目前實測結果（本機 i5-13500H / CPU only）

| 項目 | 結果 |
|---|---|
| `pytest tests -q` | 100 項全綠（約 16 秒） |
| `python -m scripts.run_smoke` | 38.5 秒完成全流程 |
| 啟發式教師 40L | 完成率 100%、平均 51.2 秒（2 PPS）、46.9 行/分、終局洞數 0 |
| IL（28.8k 樣本、28 epochs） | top-1 **0.485**、top-3 0.734、平手容忍 0.518（修 bug 前 0.151） |
| PPO（2048 步 smoke） | 可訓練、IL 權重 95/95 張量載入 |
| 難度 PPS 保真度 | 七級最大誤差 0.8% |
| 難度 APM | 七級全部不超過目標上限（上限治理） |

## 技術主線

```text
Heuristic Teacher → Imitation Learning (Board → Action) → PPO Fine-tuning (MaskablePPO)
                     ↘ DifficultyController（PPS / APM / 反應 / 失誤 / 風格）
```

| 元件 | 檔案 | 說明 |
|---|---|---|
| 規則引擎 | `envs/engine/` | SRS+、7-bag、B2B、Combo、T-Spin、垃圾行、Perfect Clear |
| Gym 環境 | `envs/gym/` | 3 種模式（Survival / 40L / Versus）、Dict 觀測、80 維高階動作 |
| 啟發式教師 | `agents/heuristic_agent.py` | Dellacherie 權重 + 深度 1–3 搜尋 |
| 強度控制 | `controllers/` | 七級難度、TokenBucket PPS、APM 治理、失誤模型、風格偏好 |
| 資料集 | `datasets/` | npz 分片 + parquet + manifest（設定雜湊） |
| 訓練 | `trainers/` | IL（masked CE + top-k KL）、PPO（MaskablePPO + IL 熱啟動） |
| 評估 | `evaluators/` | 對局指標、難度保真度、JSON/Markdown 報告 |
| 整合介面 | `integrations/` | ScreenCapture / BoardRecognizer / GameStateReader / InputController |

## 文件

想在 Colab 跑訓練的話，直接把 [`notebooks/tetrio_ai_all_in_one.ipynb`](notebooks/tetrio_ai_all_in_one.ipynb)
上傳到 Colab 即可（單一檔案包含：環境設定 → 資料集檢查 → IL → PPO → TensorBoard → 評估；
程式碼會自動從 GitHub clone）。分章版 notebook 在 [`notebooks/`](notebooks/README.md)。

完整設計文件在 [`docs/`](docs/00_index.md)：

| 章節 | 連結 |
|---|---|
| 1 系統架構（3 張 Mermaid Pipeline） | [docs/01_architecture.md](docs/01_architecture.md) |
| 2 AI 路線評估 | [docs/02_ai_routes.md](docs/02_ai_routes.md) |
| 3 開發路線 Phase 1–6 | [docs/03_roadmap.md](docs/03_roadmap.md) |
| 4 強度控制系統 | [docs/04_difficulty_control.md](docs/04_difficulty_control.md) |
| 5 Observation Space | [docs/05_observation.md](docs/05_observation.md) |
| 6 Action Space | [docs/06_action_space.md](docs/06_action_space.md) |
| 7 Reward Function | [docs/07_reward.md](docs/07_reward.md) |
| 8 神經網路架構 | [docs/08_networks.md](docs/08_networks.md) |
| 9 Colab 訓練方案 | [docs/09_colab_training.md](docs/09_colab_training.md) |
| 10 專案資料夾架構 | [docs/10_project_layout.md](docs/10_project_layout.md) |
| 11 生產級程式設計 | [docs/11_engineering.md](docs/11_engineering.md) |

## 環境需求

* Python ≥ 3.11（本機以 3.13 驗證）
* 本機：`numpy`、`torch`（CPU）、`gymnasium`、`stable-baselines3`、`sb3-contrib`、`PyYAML`、`pandas`、`pyarrow`、`tqdm`、`tensorboard`、`onnx`、`pytest`
* Colab：`requirements-colab.txt`
* TETR.IO 接入（Phase 6，選用）：`requirements-tetrio.txt`（opencv-python / mss / pynput）

## 已知限制

1. 高階動作空間不包含需要「先下降再橫移」的 tuck 型落點（見 `docs/06_action_space.md`）。
2. 180° kick 表使用社群近似值，之後可替換 `configs/rules_tetrio.yaml` 的數值。
3. 難度等級的 `target_apm` 是**上限**：要讓實際 APM 追上 50–150，需要完成 Phase 5 的 PPO 訓練。
4. TETR.IO 接入尚未實作（僅介面與設計）。
5. 教師標籤有約 20.8% 是「前兩名分數差 < 0.5」的等價落點，top-1 精確匹配因此有結構性上限；
   詳見 [docs/12_known_issues.md](docs/12_known_issues.md)（含 2026-09-25 修正的兩個 bug 記錄）。
