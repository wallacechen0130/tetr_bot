# 7. Reward Function 設計

## 7.1 完整 reward_config

與 `configs/reward.yaml` 完全同步（程式預設值在 `envs/reward.py:DEFAULT_REWARD`）。

```python
reward_config = {
    # ---- 存活與消行 ----
    "survival_per_step": 0.01,     # 每落一顆方塊的存活訊號
    "single": 1.0,
    "double": 3.0,
    "triple": 6.0,
    "tetris": 12.0,

    # ---- T-Spin（消行難度越高、獎勵越大）----
    "tspin_mini": 4.0,
    "tspin_single": 8.0,
    "tspin_double": 14.0,
    "tspin_triple": 20.0,

    # ---- 連續技 ----
    "combo_bonus_per_level": 0.5,
    "combo_bonus_cap": 10,
    "b2b_multiplier": 1.5,         # B2B 時消行/T-Spin 獎勵加倍率
    "perfect_clear": 30.0,

    # ---- 生死 ----
    "top_out": -20.0,

    # ---- 垃圾行（多人模式）----
    "garbage_sent_per_line": 0.4,
    "garbage_received_per_line": -0.3,
    "garbage_efficiency": 0.2,     # 每消一行所換得的攻擊行數

    # ---- 形狀塑形（dense shaping）----
    "new_hole": -0.5,              # 每個新增的洞
    "hole_state": -0.05,           # 洞數變化
    "height_state": -0.02,         # 超過高度的每格
    "height_threshold": 12.0,
    "bumpiness_state": -0.05,

    # ---- 課程序列 ----
    "schedule": {
        "survival_phase_fraction": 0.2,   # 前 20% 訓練偏存活
        "survival_multiplier": 1.5,
        "attack_multiplier_late": 1.5,    # 後段攻擊獎勵放大
    },
}
```

## 7.2 設計理由

| 項目 | 為什麼這樣設計 |
|---|---|
| Survival 只給 0.01 | 提供「活著就好」的基礎訊號，但不可主導學習；否則 AI 會學會龜縮不消行 |
| 消行獎勵非線性（1/3/6/12） | 四行一次消的「每行效率」是單行的 3 倍，引導 AI 學會堆疊 |
| T-Spin 高於同消行數 | TSD(14) > Double(3) > 0，因為 T-Spin 同時是攻擊與形狀管理技巧 |
| B2B ×1.5 而非加固定值 | 讓連續困難消行的獎勵隨規模成長，鼓勵維持 B2B 鏈 |
| Combo 每級 +0.5、上限 10 級 | 引導保留低盤面做連消，但避免無限延長單一 combo 的投機行為 |
| Perfect Clear +30 | 一次性、極難、且能瞬間清空盤面，給高額稀疏獎勵合理 |
| Top Out −20 | 需大於「一次 Tetris (+12)」，否則 AI 願意用一條命換一次攻擊 |
| 垃圾行 +0.4 / −0.3 | 送出行為正獎勵、被攻擊負獎勵，讓 versus 模式有意義的攻防取捨 |
| 垃圾效率 +0.2×攻擊/消行 | 防止「只消單行就送攻擊」的低效打法（單行攻擊為 0，效率值自然低） |
| 新增洞 −0.5、洞數變化 −0.05 | 前者是「行為懲罰」，後者是「狀態懲罰」，兩者互補：懲罰製造洞，也懲罰長期留洞 |
| 高度 −0.02 只算超過 12 列 | 允許中低盤面自由堆疊，只在接近危險時施壓 |
| bumpiness −0.05 | 鼓勵平整，減少未來落點限制 |

## 7.3 課程式 reward schedule

```text
訓練進度 0–20%  → survival_multiplier = 1.5，attack 維持 1.0（先學會不死的活法）
訓練進度 20–100% → survival = 1.0，attack_multiplier_late = 1.5（開始追求攻擊）
```

實作於 `envs/reward.py:RewardCalculator.set_progress()`，由環境（`TetrisEnv._progress()`）
與 `trainers/callbacks.py:CurriculumCallback` 驅動。

## 7.4 檢查方式

`tests/test_reward.py` 對下列事件逐項驗證：

* Tetris 的 `line_clear == 12.0`、送出 2 行的 `garbage_sent ≈ 0.8`
* B2B + Combo 3 + Perfect Clear 的加成量
* Top Out −20、新增 2 個洞 −1.0、被攻擊 3 行 −0.9
* 課程前後段權重縮放方向正確
