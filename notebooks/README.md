# Colab Notebooks

## 一頁式（推薦，只需要上傳一個檔案）

| Notebook | 用途 |
|---|---|
| `tetrio_ai_all_in_one.ipynb` | 環境 bootstrap → 資料集檢查 → IL（小樣本＋全量）→ PPO → TensorBoard → 評估，全部在一個檔案 |

直接把它拖進 Colab 就能跑；程式碼會自動從 GitHub clone，不需要額外上傳專案。

## 分章版（想單獨重跑某一段時用）

| Notebook | 用途 |
|---|---|
| `00_colab_setup.ipynb` | 掛載 Google Drive、安裝依賴、取得程式碼、檢查環境 |
| `01_il_train_colab.ipynb` | 模仿學習訓練（可選先產生資料集） |
| `02_ppo_train_colab.ipynb` | PPO 微調（IL 熱啟動、resume、TensorBoard / W&B） |

## 程式碼從哪裡來

`00` 會依序嘗試三種方式，**預設直接從 GitHub 抓，不需要上傳 zip**：

1. 已有 `/content/tetrio-ai/.git` → `git pull` 更新
2. 已有 `/content/tetrio-ai/requirements.txt` → 直接沿用
3. 否則 `git clone --depth 1 https://github.com/wallacechen0130/tetr_bot.git /content/tetrio-ai`
   （若 clone 失敗才退回讀 `MyDrive/tetrio-ai/code/tetrio-ai.zip`）

`01` 與 `02` 的第一個 cell 會自己檢查專案是否存在、必要時重新 clone，並重新掛載 Drive，
所以 **Colab 重啟後可以直接從 `01` 或 `02` 開始跑**，不必回頭執行 `00`。
同一個 cell 也會檢查 `numpy / pandas / pyarrow / torch / stable_baselines3 / sb3_contrib`，
缺少時才自動 `pip install -r requirements-colab.txt`（已裝過就跳過，重跑很快）。

> IL 只需要 torch，PPO 才需要 stable-baselines3；程式碼已做延遲載入，
> 所以只跑 `01` 的環境不會因為 SB3 缺失而爆掉。

## 使用流程

1. 在本機（CPU 較快）產生資料集並上傳到 Drive：

   ```powershell
   python -m scripts.generate_dataset --episodes 200 --workers 8
   python -m scripts.sync_drive --list-drives          # 先確認路徑（中文版是 G:\我的雲端硬碟）
   $env:TETRIO_AI_DRIVE = "G:\我的雲端硬碟\tetrio-ai"   # 英文版：G:\My Drive\tetrio-ai
   python -m scripts.sync_drive --direction to_drive
   ```

2. 在 Colab 依序執行 `00 → 01 → 02`（重啟後也可直接從 `01` 開始）。
   `01` 會自動挑選 Drive 上最新的資料集（沒有 `heuristic-v1` 時會用 `verify-v2` 之類的）。
3. 訓練完把 checkpoint 取回本機：

   ```powershell
   python -m scripts.sync_drive --direction from_drive
   python -m scripts.evaluate --agent policy --model checkpoints/ppo/best.zip
   ```

> 註：`PolicyAgent` 讀取的是 IL 的 `.pt`（`TetrisNetwork` state_dict）。
> PPO 產出的 `.zip` 是 SB3 格式，評估請用 `trainers/ppo_trainer.py` 的 `evaluate()`。
