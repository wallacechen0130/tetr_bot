# datasets/

資料集根目錄，每個資料集一個子資料夾：

```text
datasets/heuristic-v1/
├── manifest.json      # 版本、參數量、shard 清單、設定雜湊
├── data.parquet       # 交換格式（pyarrow + zstd 壓縮）
└── shards/
    ├── shard_00000.npz
    └── ...
```

每個樣本欄位（`datasets/writer.py:SAMPLE_KEYS`）：

| 欄位 | 形狀 | dtype | 說明 |
|---|---|---|---|
| `board` | (2, 20, 10) | uint8 | 佔用格與垃圾格 |
| `vector` | (160,) | float16 | 方塊 / 佇列 / 統計 / action mask |
| `mask` | (80,) | uint8 | 合法動作 |
| `action` | () | int16 | 教師最佳落點的動作索引 |
| `topk_actions` | (5,) | int16 | 前 5 名落點 |
| `topk_scores` | (5,) | float32 | 對應分數 |
| `context` | (4,) | int32 | [局數, 第幾顆, 已消行, combo] |

產生方式：`python -m scripts.generate_dataset --episodes 20 --workers 8`
