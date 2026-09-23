# TETR.IO 接入（Phase 6）

> ⚠️ **TETR.IO 服務條款禁止在線上多人對戰使用自動化程式。**
> 本目錄的實作目標僅限**離線練習模式**（40L / Blitz / 練習場）。

## 為什麼 MVP 不實作

V1 的重點是「決策品質」：自建引擎、啟發式教師、IL 與 PPO。
螢幕擷取與鍵盤注入會引入作業系統相依性（DPI、視窗焦點、反作弊），
且無法用單元測試驗證，因此排在 Phase 6。

## 介面契約

`integrations/base.py` 定義四個 Protocol，Phase 6 只需要提供實作：

| Protocol | 職責 | 建議實作 |
|---|---|---|
| `ScreenCapture` | 擷取遊戲畫面 | `mss` + 視窗標題定位，≥60 fps |
| `BoardRecognizer` | 畫面 → 20×10 棋盤 | OpenCV 色彩遮罩 + 網格校正 |
| `GameStateReader` | hold / next / garbage | 模板比對 + OCR（可選） |
| `InputController` | 送出按鍵 | `pynput` 或 `SendInput`，含人類抖動 |

`integrations/mock.py` 提供不需要螢幕的假實作，讓整條管線可以離線測試。

## 接入管線

```text
ScreenCapture → BoardRecognizer → GameStateReader → Decision Maker(模型+DifficultyController)
      ↑                                                                  ↓
      └──────────────── 迴圈（PPS 節奏） ──────── InputController ← 低階按鍵計畫
```

節奏控制由 `controllers/timing.py` 的 `TokenBucket` 與 `ReactionModel` 決定，
`envs/gym/low_level.py` 的 `compile_placement_plan()` 負責把高階落點轉成按鍵序列。
