# TETR.IO 風格 Tetris AI — 設計文件索引

本文件集描述一個「職業玩家風格」Tetris AI 的完整設計與可執行 MVP 實作。
程式碼位於專案根目錄（`C:\Users\Walla\tetr`），文件與程式碼一一對應。

## 閱讀順序

| 順序 | 文件 | 對應章節 | 內容 |
|---|---|---|---|
| 1 | [01_architecture.md](01_architecture.md) | 1 | 系統架構與三條 Pipeline（Mermaid） |
| 2 | [02_ai_routes.md](02_ai_routes.md) | 2 | 八種 AI 路線的九軸比較與推薦 |
| 3 | [03_roadmap.md](03_roadmap.md) | 3 | Phase 1–6 開發路線 |
| 4 | [04_difficulty_control.md](04_difficulty_control.md) | 4 | DifficultyController 七級強度控制 |
| 5 | [05_observation.md](05_observation.md) | 5 | Observation Space 設計 |
| 6 | [06_action_space.md](06_action_space.md) | 6 | 動作空間設計（高階 vs 低階） |
| 7 | [07_reward.md](07_reward.md) | 7 | Reward System 完整設計 |
| 8 | [08_networks.md](08_networks.md) | 8 | 四種神經網路架構與參數量 |
| 9 | [09_colab_training.md](09_colab_training.md) | 9 | Google Colab 訓練方案 |
| 10 | [10_project_layout.md](10_project_layout.md) | 10 | 專案資料夾架構 |
| 11 | [11_engineering.md](11_engineering.md) | 11 | 類別圖、模組依賴、API 與流程圖 |
| 12 | [../README.md](../README.md) | 12 | 可執行 MVP 的使用說明 |
| 13 | [12_known_issues.md](12_known_issues.md) | — | 修正記錄、IL 準確率天花板與已知限制 |

## 名詞表

| 名詞 | 說明 |
|---|---|
| Placement | 一次落子結果：方塊種類 + 旋轉 + 最終佔用格 |
| High-level action | 高階動作 `(column, rotation, hold)`，共 80 個離散動作 |
| Low-level action | 低階按鍵：Left / Right / Rotate CW / Rotate CCW / Soft Drop / Hard Drop / Hold |
| PPS | Pieces Per Second，每秒落子數 |
| APM | Attack Per Minute，每分鐘送出的垃圾行數 |
| T-Spin | 利用旋轉踢牆把 T 方塊塞入凹槽的進階技巧 |
| B2B | Back-to-Back，連續困難消行（Tetris / T-Spin 消行） |
| Perfect Clear | 一次消行後棋盤全空 |
| Cheese | 洞位置固定、用來消耗對手的一次性垃圾行 |
| 教師（teacher） | 產生訓練資料的啟發式 AI |

## 目前進度（本機實測）

| 項目 | 狀態 | 實測數字 |
|---|---|---|
| Tetris Engine（SRS+ / B2B / Combo / 垃圾行） | 完成 | 單元測試全綠 |
| Gymnasium 環境（3 種模式） | 完成 | `gymnasium.utils.env_checker` 通過 |
| 啟發式教師 | 完成 | 40L 完成率 100%，平均 51.2 秒（2 PPS） |
| 資料集產生（npz + parquet） | 完成 | 80 樣本 / 1.5 秒（2 workers） |
| 模仿學習 | 完成 | 28.8k 樣本 / 28 epochs：top1 **0.485**、top3 0.734（修正前 0.151） |
| PPO 微調（MaskablePPO） | 完成 | 2048 步 smoke test 通過，IL 權重 95/95 張量載入 |
| 評估報告 | 完成 | JSON + Markdown，7 級 PPS 保真度誤差 < 0.8% |
| TETR.IO 接入 | 介面完成、實作未做 | Phase 6（見 03_roadmap.md） |

> 2026-09-25：修掉兩個實測 bug（IL loss 1e7、survival 模式誤用 40L 上限），
> 詳見 [12_known_issues.md](12_known_issues.md)。

測試指令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m scripts.run_smoke
```
