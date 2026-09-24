# 9. Colab 訓練方案

本機（Windows / CPU）負責：環境開發、資料產生、IL 訓練、評估。
Colab（免費 T4）負責：PPO 大量 timesteps 訓練。

## 9.1 Google Drive 佈局

```text
MyDrive/tetrio-ai/
├── datasets/            # npz shards + parquet + manifest.json
├── checkpoints/
│   ├── il/best.pt
│   └── ppo/{ppo_*.zip, best.zip, final.zip}
├── runs/                # Colab 上的 tensorboard 輸出
└── exports/             # ONNX
```

同步指令（本機 ⇄ Drive，Colab 上 Drive 已掛載成檔案系統）：

```powershell
$env:TETRIO_AI_DRIVE = "G:\我的雲端硬碟\tetrio-ai"   # 本機：中文版 Google Drive 桌面版
# 英文版是 "G:\My Drive\tetrio-ai"；Colab 固定是 /content/drive/MyDrive/tetrio-ai
python -m scripts.sync_drive --direction to_drive   # 上傳
python -m scripts.sync_drive --direction from_drive # 取回
python -m scripts.sync_drive --list-drives          # 不確定路徑時先列出偵測結果
```

> ⚠️ Google Drive 桌面版的資料夾叫「我的雲端硬碟」（中文）或「My Drive」（英文），**不是 `MyDrive`**。
> 寫錯時 `mkdir` 會回報 `FileNotFoundError: [WinError 2]`；本工具會攔截並提示正確路徑。

`scripts/sync_drive.py` 會比較檔案大小，相同者略過，並輸出 `logs/sync_report.json`。
`__pycache__`、`*.pyc`、`.venv` 等本機產物會自動排除。

## 9.2 資料集結構

```text
datasets/heuristic-v1/
├── manifest.json     # schema 版本、樣本數、shard 清單、設定雜湊、產生環境
├── data.parquet      # 交換格式（pyarrow + zstd）
└── shards/
    ├── worker00_00000.npz
    └── ...
```

每個 shard 內含 `board (N,2,20,10) uint8`、`vector (N,160) float16`、`mask (N,80) uint8`、
`action (N,) int16`、`topk_actions (N,5) int16`、`topk_scores (N,5) float32`、`context (N,4) int32`。

產生：

```powershell
python -m scripts.generate_dataset --episodes 200 --workers 8 --depth 1 --max-pieces 200
```

## 9.3 Colab Workflow（`notebooks/*.ipynb`）

| 步驟 | notebook | 內容 |
|---|---|---|
| 1 | `00_colab_setup.ipynb` | 掛載 Drive、安裝 pinned 依賴、取得程式碼、檢查 GPU/版本 |
| 2 | `01_il_train_colab.ipynb` | IL 訓練（若已在 Colab 產生資料集則直接訓練） |
| 3 | `02_ppo_train_colab.ipynb` | PPO 微調 + resume + TensorBoard / W&B |

典型流程：

```python
from google.colab import drive; drive.mount("/content/drive")
%cd /content/tetrio-ai
!pip -q install -r requirements-colab.txt
!python -m scripts.train_il --data-root /content/drive/MyDrive/tetrio-ai/datasets/heuristic-v1
!python -m scripts.train_ppo --timesteps 2000000 --n-envs 8 --vec subproc --wandb
```

## 9.4 Checkpoint 與 Resume

* IL：`checkpoints/il/best.pt` 存 `model` / `optimizer` / `epoch` / `metrics` / `config`。
* PPO：SB3 的 `.zip` 由 `CheckpointCallback` 每 `save_freq` 步存一次，最佳模型另存 `best.zip`。
* Resume：

```powershell
python -m scripts.train_ppo --resume checkpoints/ppo/ppo_500000_steps.zip --timesteps 1000000
```

* IL → PPO 熱啟動：`configs/ppo.yaml` 的 `train.il_warm_start` 指向 `checkpoints/il/best.pt`；
  架構名稱（`model.network`）需與 IL 相同才能全部載入（實測 95/95 張量）。

## 9.5 監控：TensorBoard / W&B

```powershell
tensorboard --logdir logs/tensorboard        # 本機（需 pip install tensorboard）
python -m scripts.train_ppo --wandb          # W&B（離線模式預設，config 可改 online）
```

* TensorBoard：SB3 內建（rollout/ep_rew_mean、train/approx_kl、time/fps …）。
* W&B：`trainers/callbacks.py:make_wandb_callback()`，未安裝 wandb 時自動略過。
* 若環境沒有 tensorboard，`PPOTrainer` 會自動退回 stdout 記錄並提示安裝指令（不會讓訓練失敗）。

## 9.6 混合精度

* `configs/ppo.yaml:train.mixed_precision` 支援 `auto | off | bf16`。
* T4 建議 `bf16`（`torch.amp.autocast('cuda', dtype=torch.bfloat16)`），CPU 會自動停用。
* Batch size 建議：`n_steps=512 × n_envs=8 = 4096` rollout，`batch_size=1024`、`n_epochs=4`。

## 9.7 Colab 省時技巧

| 技巧 | 說明 |
|---|---|
| 先在本機產生資料集 | 資料產生是 CPU 密集，本機 8 workers 比 Colab 快 |
| 用 `--limit` 做小樣本驗證 | 先跑 10 k 樣本確認 loss 下降，再放大 |
| Drive 掛載後直接讀 | 避免每次上傳數百 MB |
| 斷線保護 | `CheckpointCallback` 每 50 k 步存檔；`--resume` 接續 |
| 先 smoke 再長跑 | `python -m scripts.run_smoke` 可在 40 秒內驗證整條管線 |
