# 8. 神經網路架構

參數量由 `python -m scripts.model_summary` 以本機環境**實測**（torch 2.14.0+cpu、vector_dim=160、hidden_dim=256）。

## 8.1 實測參數量

| 網路 | 棋盤編碼器參數 | 總參數 | CPU forward | 建議情境 |
|---|---|---|---|---|
| `small_cnn` / `cnn` | 219,120 | 577,601 | 最快 | 本機 smoke、快速實驗 |
| `resnet`（預設） | 380,224 | 738,705 | 快 | **正式訓練主線** |
| `cnn_attention` / `attention` | 831,045 | 1,189,526 | 中 | GPU 上的進階比較 |
| `transformer` | 554,240 | 912,721 | 慢 | Phase 7 研究 |

（完整 JSON 見執行後的 `logs/model_summary.json`。）

## 8.2 架構細節

### CNN Policy（SmallCNN）

```text
board (B,2,20,10)
  → Conv3x3(2→16) + ReLU
  → Conv3x3(16→32) + ReLU
  → MaxPool2
  → Conv3x3(32→32) + ReLU
  → Flatten(32×10×5=1600) → Linear(1600→128) + ReLU      # board embedding 128
vector (B,160) → MLP(160→256)                            # vector embedding 256
concat(128+256=384) → Linear(384→256) + ReLU + LayerNorm # fusion 256
  → policy head: Linear(256→256) + ReLU + Linear(256→80)
  → value  head: Linear(256→256) + ReLU + Linear(256→1)
```

### ResNet Policy（主線）

```text
board → Conv3x3(2→64)+BN+ReLU (stem)
      → 5 × ResidualBlock(Conv-BN-ReLU-Conv-BN + skip, 64ch)
      → AdaptiveAvgPool2d(1) → Linear(64→128) + ReLU      # board embedding 128
vector → MLP(160→256)
concat → Linear(384→256)+ReLU+LayerNorm → PolicyValueHead(80 / 1)
```

### CNN + Attention

`Conv3x3(2→32)` ×2 → **SE 通道注意力** → **空間注意力**（7×7 conv + sigmoid）→ Flatten → Linear(6400→128)。
用於強調「well 與洞所在的欄位」；參數較多但對關鍵欄位更敏感。

### Transformer Policy

`Conv2d(2→128, kernel=2, stride=2)` 把 20×10 切成 **50 個 2×2 patch token** →
`cls` token + 位置編碼 → 4 層 TransformerEncoder（d=128、4 heads、FFN 256）→ cls 輸出 → Linear(128→128)。

## 8.3 優缺點

| 網路 | 優點 | 缺點 |
|---|---|---|
| CNN | 快、好訓練、CPU 可行 | 感受野小，深層形狀需更多層 |
| ResNet | 深層仍穩定、表達力足夠、IL/PPO 共用 | 比 CNN 慢約 2–3 倍 |
| CNN + Attention | 可解釋性較高（注意力圖）、對關鍵欄位敏感 | 參數多、CPU 慢、容易過擬合小資料集 |
| Transformer | 全域建模、可與 next queue 一起做序列建模 | 需要大量資料、推論延遲高、CPU 不可行 |

## 8.4 最終選擇

**ResNet Policy 為預設主線**，`small_cnn` 作為本機快跑基線，`cnn_attention` 與 `transformer`
保留為可比實驗（同一份 `policies/factory.py:build_network()` 介面即可切換）。

與 SB3 的整合方式：`policies/sb3_extractor.py:TetrisFeaturesExtractor`
把 `TetrisNetwork` 包成 `BaseFeaturesExtractor`，輸出 256 維 embedding，
再由 SB3 的 `net_arch=[256,256]` 接上 actor/critic 頭。
好處是 **IL 訓練的 encoder 權重可直接載入 PPO**（實測 95/95 張量轉移）。

```powershell
# 切換架構（IL 與 PPO 用同一個名稱才吃得到權重轉移）
python -m scripts.train_il --network resnet
python -m scripts.train_ppo
```
