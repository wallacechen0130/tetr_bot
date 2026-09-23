"""報告輸出：JSON + Markdown。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluators.metrics import aggregate_metrics
from trainers.common import ensure_dir


def write_report(
    episodes_or_summary: Any,
    *,
    json_path: str | Path = "logs/report.json",
    markdown_path: str | Path | None = "logs/report.md",
    title: str = "Tetris AI 評估報告",
) -> dict[str, Any]:
    """把評估結果寫成 JSON 與 Markdown。"""

    if isinstance(episodes_or_summary, list):
        summary = aggregate_metrics(episodes_or_summary)
        episodes = [ep.to_dict() for ep in episodes_or_summary]
    else:
        summary = dict(episodes_or_summary)
        episodes = []

    payload = {"title": title, "summary": summary, "episodes": episodes}
    target = Path(json_path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if markdown_path:
        lines = [f"# {title}", "", "## 彙總", "", "| 指標 | 值 |", "|---|---|"]
        for key, value in summary.items():
            lines.append(f"| {key} | {_fmt(value)} |")
        if episodes:
            lines += ["", "## 逐局結果", "", "| # | tier | lines | pieces | elapsed | pps | apm | top_out |", "|---|---|---|---|---|---|---|---|"]
            for index, episode in enumerate(episodes, start=1):
                lines.append(
                    f"| {index} | {episode['tier']} | {episode['lines']} | {episode['pieces']} | "
                    f"{episode['elapsed']:.1f} | {episode['pps']:.2f} | {episode['apm']:.1f} | {episode['top_out']} |"
                )
        md_target = Path(markdown_path)
        ensure_dir(md_target.parent)
        md_target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return payload


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
