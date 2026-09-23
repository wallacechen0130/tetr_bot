# 11. 生產級程式設計

## 11.1 類別圖

```mermaid
classDiagram
    class Board {
        +int visible_rows
        +int cols
        +ndarray cells
        +collides(cells) bool
        +lock(cells, value)
        +clear_lines() list
        +holes() int
        +column_heights() ndarray
        +feature_vector() ndarray
    }
    class Piece {
        +str kind
        +int rotation
        +int row
        +int col
        +cells() tuple
        +moved(dr, dc) Piece
        +rotated(rotation, row, col) Piece
    }
    class Placement {
        +str kind
        +int rotation
        +tuple cells
        +bool hold_used
        +int kick_index
        +left_column int
        +landing_height(total_rows) int
    }
    class GarbageManager {
        +receive(amount) int
        +offset(attack) tuple
        +advance(pieces)
        +apply(board) int
    }
    class TetrisSimulator {
        +reset(seed)
        +legal_placements() list
        +step_placement(placement) StepResult
        +snapshot() GameSnapshot
    }
    class GameSnapshot {
        +ndarray board
        +str current
        +tuple next_queue
        +int combo
        +int b2b_chain
        +int garbage_pending
    }
    class TetrisEnv {
        +reset(seed)
        +step(action)
        +action_masks() ndarray
        +set_piece_time(dt)
        +current_placements dict
    }
    class ObservationEncoder {
        +observation_space() Dict
        +encode(snapshot, mask) dict
        +flatten(observation) ndarray
    }
    class HeuristicAgent {
        +dict weights
        +int depth
        +rank(snapshot, placements) list
        +evaluate(board, snapshot, placement) Candidate
    }
    class DifficultyController {
        +DifficultyProfile profile
        +set_tier(tier)
        +decide(candidates, probs, action_mask) Decision
        +note_action(attack, lines)
        +stats() dict
    }
    class TetrisNetwork {
        +features(board, vector) Tensor
        +forward(board, vector, mask) tuple
    }
    class ILTrainer {
        +fit(epochs) dict
        +evaluate(loader) dict
    }
    class PPOTrainer {
        +build_vec_env() VecEnv
        +warm_start_from_il(path) TransferReport
        +run(total_timesteps) dict
        +evaluate(episodes) dict
    }
    class BenchmarkRunner {
        +run_episode(seed) EpisodeMetrics
        +run(episodes) list
    }

    TetrisSimulator --> Board
    TetrisSimulator --> GarbageManager
    TetrisSimulator --> Placement
    TetrisSimulator --> GameSnapshot
    TetrisEnv --> TetrisSimulator
    TetrisEnv --> ObservationEncoder
    TetrisEnv --> RewardCalculator
    HeuristicAgent --> GameSnapshot
    HeuristicAgent --> Placement
    DifficultyController --> HeuristicAgent
    DifficultyController --> Decision
    TetrisNetwork --> Placement
    PPOTrainer --> TetrisNetwork
    PPOTrainer --> TetrisEnv
    ILTrainer --> TetrisNetwork
    BenchmarkRunner --> TetrisEnv
    BenchmarkRunner --> DifficultyController
```

## 11.2 模組依賴圖

```mermaid
flowchart TD
    cfg[configs/*.yaml] --> config[envs/config.py]
    config --> engine[envs/engine]
    engine --> gymenv[envs/gym]
    gymenv --> wrappers[envs/wrappers]
    gymenv --> render[envs/render]
    engine --> agents[agents]
    gymenv --> agents
    agents --> controllers[controllers]
    gymenv --> datasets[datasets]
    datasets --> trainers[trainers]
    policies[policies] --> trainers
    agents --> trainers
    gymenv --> evaluators[evaluators]
    controllers --> evaluators
    trainers --> evaluators
    engine --> integrations[integrations]
    gymenv --> scripts[scripts]
    trainers --> scripts
    evaluators --> scripts
```

**依賴方向單向**：`engine → gym → agents / controllers → trainers / evaluators → scripts`。
反向依賴一律禁止，這讓規則層可獨立測試，也讓 Phase 6 只需實作 `integrations/base.py` 的介面。

## 11.3 公開 API 設計

| 層 | API | 簽章 | 說明 |
|---|---|---|---|
| 環境 | 建立 | `envs.make_env(env_id, **kwargs)` | 已註冊的 Gymnasium 環境 |
| 環境 | 觀測 | `TetrisEnv.reset/step` | 標準 5-tuple，`info` 含 placements / snapshot / events |
| 環境 | 遮罩 | `TetrisEnv.action_masks() -> np.ndarray[bool]` | 供 MaskablePPO 使用 |
| 環境 | 節奏 | `TetrisEnv.set_piece_time(dt)` | 模擬目標 PPS |
| 引擎 | 落點 | `TetrisSimulator.legal_placements()` | 含 hold 變體 |
| 引擎 | 落子 | `TetrisSimulator.step_placement(p, time_cost=None)` | 產生 `StepEvents` |
| Agent | 決策 | `Agent.act(observation, info) -> int` | 所有 agent 共用 |
| 教師 | 排序 | `HeuristicAgent.rank(snapshot, placements, depth=None)` | 供 controller 重排 |
| 控制 | 決策 | `DifficultyController.decide(candidates, probs=None, action_mask=None)` | 回傳含延遲的 `Decision` |
| 控制 | 等級 | `set_tier("Godlike")` / `with_profile(target_apm=45)` | 執行期調整 |
| 資料 | 產生 | `scripts.generate_dataset.generate(out=..., episodes=..., workers=...)` | 平行產生 |
| 訓練 | IL | `ILTrainer(config).fit(epochs=...)` | 回傳 best_top1 等指標 |
| 訓練 | PPO | `PPOTrainer(config).run(total_timesteps=..., resume_from=...)` | 支援 resume 與 IL 熱啟動 |
| 評估 | 基準 | `BenchmarkRunner(env_factory, agent, controller=...).run(episodes)` | 回傳 `EpisodeMetrics` |
| 評估 | 難度 | `run_difficulty_suite(env_factory)` | 七級保真度報表 |

## 11.4 訓練流程圖

```mermaid
sequenceDiagram
    participant CLI as scripts/generate_dataset
    participant Worker as Worker process
    participant Env as TetrisEnv
    participant Teacher as HeuristicAgent
    participant DS as DatasetWriter
    participant IL as ILTrainer
    participant PPO as PPOTrainer

    CLI->>Worker: spawn N workers
    Worker->>Env: make(mode=survival) + reset(seed)
    loop 每個 episode
        Env->>Teacher: snapshot + placements
        Teacher-->>Env: best action + top-k
        Worker->>DS: build_sample(obs, action, topk)
        Env->>Env: step(action)
    end
    DS-->>CLI: shards + manifest.json
    CLI->>IL: fit(dataset)
    IL-->>CLI: checkpoints/il/best.pt
    CLI->>PPO: run(il_warm_start=best.pt)
    PPO-->>CLI: checkpoints/ppo/final.zip
```

## 11.5 推論流程圖

```mermaid
sequenceDiagram
    participant Runner as BenchmarkRunner or TETR.IO loop
    participant Env as TetrisEnv
    participant Enc as ObservationEncoder
    participant Ctrl as DifficultyController
    participant Agent as PolicyAgent or HeuristicAgent

    Env->>Enc: snapshot + action_mask
    Enc-->>Agent: Dict observation
    Agent-->>Ctrl: candidates or probs
    Ctrl-->>Runner: Decision(action, delay_s)
    Runner->>Runner: clock.advance(delay_s)
    Runner->>Env: set_piece_time(delay_s) + step(action)
    Env-->>Runner: reward, terminated, info(events)
```

## 11.6 SOLID 對應

| 原則 | 落實方式 |
|---|---|
| S 單一職責 | 規則 / 環境 / 決策 / 控制 / 訓練 / 評估各自獨立套件；`TetrisSimulator` 不管時間，時間交給 controller |
| O 開放封閉 | 新增難度只要加 YAML；新增網路只要在 `BOARD_ENCODERS` 註冊；新增模式只要加 `ENV_IDS` |
| L 里氏替換 | 所有 agent 符合 `Agent` Protocol，`RandomAgent` 與 `PolicyAgent` 可互換 |
| I 介面隔離 | `integrations/base.py` 用四個小型 Protocol，而非巨型遊戲介面 |
| D 依賴反轉 | engine 不知道 gym，gym 不知道 torch；`DifficultyController` 依賴「候選分數」介面而非具體模型 |

## 11.7 可測試性

| 測試檔 | 覆蓋範圍 |
|---|---|
| `test_engine_*.py` | 棋盤、旋轉與 kick、攻擊表、模擬器事件、垃圾行 |
| `test_env_api.py` / `test_env_mask.py` | Gymnasium 契約、mask 正確性、可重現性 |
| `test_obs_encoder.py` / `test_reward.py` | 觀測契約、reward 分量 |
| `test_difficulty.py` | PPS 保真度（±5%）、反應時間、失誤分佈、風格重排 |
| `test_heuristic_agent.py` | 40L 完成、排序、搜尋深度 |
| `test_dataset_io.py` | npz / parquet 往返、manifest |
| `test_evaluators.py` | 指標彙總、報告輸出 |
| `test_integrations_stub.py` | Protocol 結構、mock 行為、stub 明確失敗 |
| `test_smoke.py` | 資料集 → IL → PPO 的端到端可訓練性 |

執行：`python -m pytest tests -q`（實測 100 項通過）。
