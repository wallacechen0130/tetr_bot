# 5. 模型輸入設計（Observation Space）

實作：`envs/gym/obs_encoder.py:ObservationEncoder`；契約由 `tests/test_obs_encoder.py` 鎖定。

## 5.1 完整規格

| 鍵 | 形狀 | dtype | 範圍 | 說明 |
|---|---|---|---|---|
| `board` | (2, 20, 10) | float32 | 0/1 | ch0 = 佔用格、ch1 = 垃圾行 |
| `piece_onehot` | (7,) | float32 | 0/1 | 目前方塊 I,J,L,O,S,T,Z |
| `hold_onehot` | (8,) | float32 | 0/1 | 7 種方塊 + 「空」 |
| `hold_available` | (1,) | float32 | 0/1 | 本顆是否還能 hold |
| `next_queue` | (7, 7) | float32 | 0/1 | 7 顆預覽（需求為 5–7 顆，取 7） |
| `combo_b2b` | (4,) | float32 | 0–1 | combo/20、b2b_active、b2b_chain/20、pieces/1000 |
| `garbage` | (3,) | float32 | 0–1 | pending/20、attack_sent/200、garbage_received/200 |
| `stats` | (8,) | float32 | 0–1 | 已消行、40L 剩餘、已用時間、PPS、APM、最大高度、洞數、bumpiness |
| `action_mask` | (80,) | int8 | 0/1 | 合法高階動作 |

* **Dense 特徵向量**：`VECTOR_KEYS` = 上表扣掉 `board` 的 160 維（含 action mask），
  順序固定於 `envs/gym/obs_encoder.py`，IL / PPO / ONNX 共用。
* **扁平基線**：`ObservationEncoder.flatten()` 產生 480 + 80 = 560 維，供 MLP 基線使用。
* 正規化策略：所有數值都裁切（clip）到 0–1，避免 scale 問題；棋盤不做灰階壓縮以保留垃圾行資訊。

關於「Garbage / Attack Information」：`garbage` 三個數字讓模型知道「我還有多少垃圾要落地」、
「我已經送出多少」、「我被打了多少」，是 versus 模式必要的資訊；`stats` 的洞數與高度則是存活品質。

## 5.2 四種輸入編碼的比較

| 方案 | 表達力 | 參數量（實測） | CPU 速度 | 優點 | 缺點 | 適用情境 |
|---|---|---|---|---|---|---|
| Dense（MLP） | 低 | 577 K（含 MLP 頭） | 最快 | 最簡單、除錯容易 | 丟失棋盤空間結構、無法學 tuck/spin | 快速驗證、CPU 極限環境 |
| CNN（SmallCNN） | 中 | 577 K | 快 | 局部形狀偵測足夠、好訓練 | 感受野需堆疊才夠大 | IL 基線、本機 smoke |
| **ResNet（主線）** | 高 | 739 K | 中 | 深層特徵 + 殘差好訓練；對稱與長距離關係都能學 | 比 CNN 慢一些 | **本專案預設** |
| CNN + Attention | 高 | 1.19 M | 慢 | 可強調關鍵欄位（well / 洞） | 參數多、CPU 上偏慢 | 上 GPU 後的進階實驗 |
| Transformer | 高 | 913 K | 最慢 | 全域關係建模最好 | 需要更多資料、CPU 不可行 | Phase 7 研究 |

## 5.3 推薦

**ResNet 為預設**，原因：

1. 20×10 的棋盤不需要 Transformer 的全域注意力就足以表達「洞、well、高度差」。
2. 739 K 參數在 T4 上可高速訓練，在本機 CPU 也能跑 smoke。
3. 與 IL 教師的學習訊號契合（局部 3×3 感受野即可判斷落點好壞）。

實務建議：

* 本機 CPU 迭代 → 用 `small_cnn`（毫秒級 forward）。
* Colab 正式訓練 → 用 `resnet`。
* 想比較架構時用 `python -m scripts.model_summary` 取得實際參數量，不要凭印象選模型。
