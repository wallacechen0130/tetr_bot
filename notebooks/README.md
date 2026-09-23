# Colab Notebooks

| Notebook | 用途 |
|---|---|
| `00_colab_setup.ipynb` | 掛載 Google Drive、安裝依賴、取得程式碼、檢查環境 |
| `01_il_train_colab.ipynb` | 模仿學習訓練（可選先產生資料集） |
| `02_ppo_train_colab.ipynb` | PPO 微調（IL 熱啟動、resume、TensorBoard / W&B） |

## 使用流程

1. 在本機（CPU 較快）產生資料集並上傳到 Drive：

   ```powershell
   python -m scripts.generate_dataset --episodes 200 --workers 8
   $env:TETRIO_AI_DRIVE = "G:/MyDrive/tetrio-ai"
   python -m scripts.sync_drive --direction to_drive
   ```

2. 依序執行 `00 → 01 → 02`。
3. 訓練完把 checkpoint 取回本機：

   ```powershell
   python -m scripts.sync_drive --direction from_drive
   python -m scripts.evaluate --agent policy --model checkpoints/ppo/best.zip
   ```

> 註：`PolicyAgent` 讀取的是 IL 的 `.pt`（`TetrisNetwork` state_dict）。
> PPO 產出的 `.zip` 是 SB3 格式，評估請用 `trainers/ppo_trainer.py` 的 `evaluate()`。
