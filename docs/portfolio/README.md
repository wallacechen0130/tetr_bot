# 備審資料素材（特殊選才）

這個資料夾放「特殊選才」備審資料用的素材。

| 檔案 | 用途 |
|---|---|
| [特殊選才_專題成果報告.md](特殊選才_專題成果報告.md) | 專題成果主文：可直接擷取章節放進各校備審格式，同時也是給 AI 協作者的專案說明（附錄 A） |

## 提交前檢查清單

1. **替換個人資料佔位符**：`【姓名】`、`【就讀學校】`、`【年級】`、`【聯絡方式】`、`【GitHub 連結】`。
2. **刪除給 AI 的段落**：附錄 A 是專案交接說明，提交時整節刪除（或改成「作品連結」一節）。
3. **依各校格式調整**：不同學校要的欄位不一樣，用文末的「附錄 B：備審資料對應表」擷取對應章節。
4. **附上可驗證的連結**：GitHub repo、設計文件資料夾、Colab notebook。
5. **確認數字是最新的**：文件內的數據取自 2026-10 的實測。若已重跑訓練，
   請更新「第四章 成果總覽」的表格（測試數、資料集樣本數、模型準確率）。

## 隱私提醒

本專案的 GitHub repo 是**公開**的。若你不希望個資公開：

* 個人資料只寫在另外的備審檔案，不要 commit 進這個 repo。
* 或者把 repo 改成 private（但這樣審查者就看不到程式碼，建議改成「提供壓縮檔」或「邀請審查者」）。

## 建議加值素材

評審通常喜歡「能看到成果」的證據，建議另外準備：

| 素材 | 怎麼做 |
|---|---|
| 訓練曲線圖 | `tensorboard --logdir logs/tensorboard` 截圖，或從 `logs/*.json` 畫 top-1 隨 epoch 的變化 |
| 遊戲畫面 | `python -m scripts.play_ascii --env-id Tetris40L-v0 --tier Advanced` 的終端機截圖 |
| 短影片 | 錄製 AI 完成 40L 的過程（可用 ASCII 模式錄終端機，或之後接上 TETR.IO 實作再錄） |
| 對局數據 | `python -m scripts.evaluate --agent heuristic --env-id Tetris40L-v0 --episodes 3` 產生的 `logs/heuristic_40l.md` |
| 難度保真度報告 | `python -m scripts.evaluate --difficulty-suite` 產生的 `logs/difficulty.md` |

## 文件數字來源（方便日後更新）

| 數字 | 來源 |
|---|---|
| 141 項測試 | `python -m pytest tests -q` |
| 7,583 行程式 / 95 檔 | `Get-ChildItem <套件> -Recurse -Filter *.py` 統計 |
| 資料集 2,063,231 筆 | `datasets/heuristic-v1/manifest.json` |
| 40L 51.2 秒 / 完成率 100% | `logs/heuristic_40l.json` |
| 難度 PPS 誤差 0.8% | `logs/difficulty.json`（`max_pps_error`） |
| IL top-1 0.485 | `docs/12_known_issues.md` 的學習曲線表 |
