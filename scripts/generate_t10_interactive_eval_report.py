#!/usr/bin/env python3
"""Generate a self-contained interactive HTML report for a T10 eval run."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIFFICULTY_LABELS = {
    "simple": "easy",
    "moderate": "moderate",
    "challenging": "hard",
}

DIFFICULTY_ORDER = ["easy", "moderate", "hard"]


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def bucket_error(example: dict[str, Any]) -> str:
    if example.get("correct"):
        return "correct"
    if example.get("exec_failed"):
        return example.get("error_category") or "exec_failed"
    if example.get("wrong_result"):
        return "wrong_result"
    return example.get("error_category") or "other_error"


def build_analysis(
    eval_report_path: Path,
    per_example_path: Path,
    dataset_jsonl_path: Path,
    bird_dev_json_path: Path,
) -> dict[str, Any]:
    eval_report = load_json(eval_report_path)
    per_example = load_jsonl(per_example_path)
    dataset_rows = load_jsonl(dataset_jsonl_path)
    bird_dev_rows = load_json(bird_dev_json_path)

    question_lookup = {
        row["question_id"]: {
            "question": row.get("question"),
            "evidence": row.get("evidence"),
            "gold_sql": row.get("gold_sql"),
            "db_id": row.get("db_id"),
            "difficulty_source": row.get("difficulty"),
        }
        for row in dataset_rows
    }

    dataset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in bird_dev_rows:
        db_id = row["db_id"]
        difficulty = DIFFICULTY_LABELS.get(row["difficulty"], row["difficulty"])
        dataset_counts[db_id][difficulty] += 1
        dataset_counts[db_id]["total"] += 1

    db_stats: dict[str, Counter[str]] = defaultdict(Counter)
    db_difficulty_stats: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    error_by_db: dict[str, Counter[str]] = defaultdict(Counter)
    error_by_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    error_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    example_rows: list[dict[str, Any]] = []

    for example in per_example:
        question_meta = question_lookup.get(example["question_id"], {})
        db_id = example["db_id"]
        difficulty_source = example.get("difficulty")
        difficulty = DIFFICULTY_LABELS.get(difficulty_source, difficulty_source)
        primary_error = bucket_error(example)
        correct = bool(example.get("correct"))
        exec_failed = bool(example.get("exec_failed"))
        wrong_result = bool(example.get("wrong_result"))

        db_stats[db_id]["total"] += 1
        db_stats[db_id]["correct"] += int(correct)
        db_stats[db_id]["exec_failed"] += int(exec_failed)
        db_stats[db_id]["wrong_result"] += int(wrong_result)

        bucket = db_difficulty_stats[db_id][difficulty]
        bucket["total"] += 1
        bucket["correct"] += int(correct)
        bucket["exec_failed"] += int(exec_failed)
        bucket["wrong_result"] += int(wrong_result)

        error_by_db[db_id][primary_error] += 1
        error_by_difficulty[difficulty][primary_error] += 1

        row = {
            "question_id": example["question_id"],
            "db_id": db_id,
            "difficulty": difficulty,
            "difficulty_source": difficulty_source,
            "question": question_meta.get("question"),
            "evidence": question_meta.get("evidence"),
            "correct": correct,
            "exact_match": bool(example.get("exact_match")),
            "exec_failed": exec_failed,
            "wrong_result": wrong_result,
            "error_category": example.get("error_category"),
            "primary_error": primary_error,
            "predicted_sql": example.get("predicted_sql"),
            "gold_sql": example.get("gold_sql") or question_meta.get("gold_sql"),
            "pred_error": example.get("pred_error"),
            "gold_error": example.get("gold_error"),
        }
        example_rows.append(row)

        if not correct and len(error_examples[primary_error]) < 8:
            error_examples[primary_error].append(
                {
                    "question_id": row["question_id"],
                    "db_id": db_id,
                    "difficulty": difficulty,
                    "question": row["question"],
                    "predicted_sql": row["predicted_sql"],
                    "gold_sql": row["gold_sql"],
                }
            )

    databases = sorted(db_stats)
    db_summary = []
    for db_id in databases:
        stats = db_stats[db_id]
        difficulty_split = {}
        for difficulty in DIFFICULTY_ORDER:
            split = db_difficulty_stats[db_id].get(difficulty, Counter())
            total = split.get("total", 0)
            difficulty_split[difficulty] = {
                "total": total,
                "correct": split.get("correct", 0),
                "accuracy": pct(split.get("correct", 0), total),
                "exec_failed": split.get("exec_failed", 0),
                "wrong_result": split.get("wrong_result", 0),
            }

        db_summary.append(
            {
                "db_id": db_id,
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": pct(stats["correct"], stats["total"]),
                "exec_failed": stats["exec_failed"],
                "wrong_result": stats["wrong_result"],
                "dataset_total": dataset_counts[db_id]["total"],
                "dataset_easy": dataset_counts[db_id]["easy"],
                "dataset_moderate": dataset_counts[db_id]["moderate"],
                "dataset_hard": dataset_counts[db_id]["hard"],
                "difficulty_split": difficulty_split,
                "error_breakdown": dict(sorted(error_by_db[db_id].items())),
            }
        )

    difficulty_summary = []
    for difficulty in DIFFICULTY_ORDER:
        matching = [row for row in example_rows if row["difficulty"] == difficulty]
        total = len(matching)
        correct = sum(int(row["correct"]) for row in matching)
        exec_failed = sum(int(row["exec_failed"]) for row in matching)
        wrong_result = sum(int(row["wrong_result"]) for row in matching)
        difficulty_summary.append(
            {
                "difficulty": difficulty,
                "total": total,
                "correct": correct,
                "accuracy": pct(correct, total),
                "exec_failed": exec_failed,
                "wrong_result": wrong_result,
                "error_breakdown": dict(sorted(error_by_difficulty[difficulty].items())),
            }
        )

    top_databases = sorted(db_summary, key=lambda row: (-row["accuracy"], -row["total"]))[:5]
    bottom_databases = sorted(db_summary, key=lambda row: (row["accuracy"], -row["total"]))[:5]

    overall_error_counts = Counter(row["primary_error"] for row in example_rows)
    overall_summary = {
        "total_examples": len(example_rows),
        "correct": sum(int(row["correct"]) for row in example_rows),
        "execution_accuracy": pct(
            sum(int(row["correct"]) for row in example_rows),
            len(example_rows),
        ),
        "exact_match_count": sum(int(row["exact_match"]) for row in example_rows),
        "exact_match_accuracy": pct(
            sum(int(row["exact_match"]) for row in example_rows),
            len(example_rows),
        ),
        "exec_fail_count": sum(int(row["exec_failed"]) for row in example_rows),
        "wrong_result_count": sum(int(row["wrong_result"]) for row in example_rows),
        "database_count": len(databases),
        "top_database": top_databases[0]["db_id"] if top_databases else None,
        "bottom_database": bottom_databases[0]["db_id"] if bottom_databases else None,
        "source_summary": eval_report.get("summary", {}),
    }

    return {
        "metadata": {
            "eval_report_path": str(eval_report_path),
            "per_example_path": str(per_example_path),
            "dataset_jsonl_path": str(dataset_jsonl_path),
            "bird_dev_json_path": str(bird_dev_json_path),
        },
        "overall_summary": overall_summary,
        "difficulty_summary": difficulty_summary,
        "database_summary": db_summary,
        "top_databases": top_databases,
        "bottom_databases": bottom_databases,
        "overall_error_breakdown": dict(sorted(overall_error_counts.items())),
        "error_examples": error_examples,
        "examples": example_rows,
    }


def render_html(analysis: dict[str, Any], title: str) -> str:
    payload = json.dumps(analysis, ensure_ascii=False)
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: #fffdf9;
      --panel-alt: #f2ede2;
      --text: #1f1d18;
      --muted: #6f6759;
      --line: #d8cfbe;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
      --warn: #b45309;
      --bad: #b42318;
      --good: #166534;
      --shadow: 0 10px 30px rgba(31, 29, 24, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.09), transparent 28%),
        radial-gradient(circle at top right, rgba(180,83,9,0.08), transparent 24%),
        var(--bg);
      color: var(--text);
    }}
    .wrap {{
      width: min(1480px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 18px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: 32px;
      letter-spacing: -0.03em;
      margin-bottom: 8px;
    }}
    .sub {{
      color: var(--muted);
      line-height: 1.5;
      max-width: 980px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      background: var(--panel-alt);
      color: var(--muted);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(12, minmax(0, 1fr));
    }}
    .span-12 {{ grid-column: span 12; }}
    .span-8 {{ grid-column: span 8; }}
    .span-7 {{ grid-column: span 7; }}
    .span-6 {{ grid-column: span 6; }}
    .span-5 {{ grid-column: span 5; }}
    .span-4 {{ grid-column: span 4; }}
    .span-3 {{ grid-column: span 3; }}
    .panel {{
      padding: 18px;
      overflow: hidden;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .section-head p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: linear-gradient(180deg, #fff, #fbf7ef);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .card .value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 8px;
      letter-spacing: -0.03em;
    }}
    .card .detail {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }}
    select, input {{
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      min-width: 180px;
    }}
    input.search {{
      flex: 1 1 320px;
      min-width: 280px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    tbody tr:hover {{
      background: rgba(15, 118, 110, 0.04);
    }}
    .scroll {{
      overflow: auto;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .bar-list {{
      display: grid;
      gap: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 170px 1fr 72px;
      align-items: center;
      gap: 10px;
    }}
    .bar-track {{
      position: relative;
      height: 12px;
      border-radius: 999px;
      background: #ece5d6;
      overflow: hidden;
    }}
    .bar-fill {{
      position: absolute;
      inset: 0 auto 0 0;
      background: linear-gradient(90deg, var(--accent), #14b8a6);
      border-radius: 999px;
    }}
    .bar-row.bad .bar-fill {{
      background: linear-gradient(90deg, #ef4444, #f59e0b);
    }}
    .db-name {{
      font-weight: 600;
    }}
    .small {{
      color: var(--muted);
      font-size: 12px;
    }}
    .tag {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 8px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 12px;
    }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    .split-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .split-card {{
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-alt);
    }}
    .split-card .big {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 6px;
    }}
    .empty {{
      color: var(--muted);
      padding: 10px 0;
      font-size: 14px;
    }}
    @media (max-width: 1100px) {{
      .span-8, .span-7, .span-6, .span-5, .span-4, .span-3 {{ grid-column: span 12; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .bar-row {{ grid-template-columns: 130px 1fr 64px; }}
    }}
    @media (max-width: 720px) {{
      .wrap {{ width: min(100vw - 20px, 1480px); margin: 10px auto 30px; }}
      .hero, .panel {{ border-radius: 16px; }}
      .hero {{ padding: 20px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .split-grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{safe_title}</h1>
      <div class="sub">
        Interactive analysis for the selected eval run. Difficulty labels in the source data are
        <span class="tag">simple</span>, <span class="tag">moderate</span>, and <span class="tag">challenging</span>;
        this report shows them as <span class="tag">easy</span>, <span class="tag">moderate</span>, and <span class="tag">hard</span>.
      </div>
      <div class="meta">
        <div class="pill">Eval report: <span class="mono" id="meta-eval-path"></span></div>
        <div class="pill">Examples: <span class="mono" id="meta-example-path"></span></div>
        <div class="pill">Dataset: <span class="mono" id="meta-dataset-path"></span></div>
      </div>
    </section>

    <div class="grid">
      <section class="panel span-12">
        <div class="section-head">
          <div>
            <h2>Overview</h2>
            <p>Main run metrics plus best and worst database snapshots.</p>
          </div>
        </div>
        <div class="cards" id="overview-cards"></div>
      </section>

      <section class="panel span-7">
        <div class="section-head">
          <div>
            <h2>Database Performance</h2>
            <p>How each database performs overall and by easy, moderate, and hard questions.</p>
          </div>
        </div>
        <div class="controls">
          <input id="db-search" class="search" type="search" placeholder="Filter databases by name">
          <select id="db-sort">
            <option value="accuracy_desc">Sort by accuracy ↓</option>
            <option value="accuracy_asc">Sort by accuracy ↑</option>
            <option value="total_desc">Sort by total ↓</option>
            <option value="name_asc">Sort by name A-Z</option>
          </select>
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>Database</th>
                <th>Total</th>
                <th>Acc</th>
                <th>Easy</th>
                <th>Moderate</th>
                <th>Hard</th>
                <th>Exec Fail</th>
                <th>Wrong Result</th>
              </tr>
            </thead>
            <tbody id="database-table"></tbody>
          </table>
        </div>
      </section>

      <section class="panel span-5">
        <div class="section-head">
          <div>
            <h2>Top And Bottom Databases</h2>
            <p>Quick ranking by execution accuracy.</p>
          </div>
        </div>
        <div class="bar-list" id="ranking-bars"></div>
      </section>

      <section class="panel span-4">
        <div class="section-head">
          <div>
            <h2>Overall Difficulty Split</h2>
            <p>Performance trend across easy, moderate, and hard prompts.</p>
          </div>
        </div>
        <div class="split-grid" id="difficulty-cards"></div>
      </section>

      <section class="panel span-4">
        <div class="section-head">
          <div>
            <h2>Overall Error Types</h2>
            <p>Primary failure mode counts across the whole run.</p>
          </div>
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr><th>Error Type</th><th>Count</th><th>Rate</th></tr>
            </thead>
            <tbody id="overall-error-table"></tbody>
          </table>
        </div>
      </section>

      <section class="panel span-4">
        <div class="section-head">
          <div>
            <h2>Selected Database</h2>
            <p>Inspect one database in detail.</p>
          </div>
        </div>
        <div class="controls">
          <select id="database-focus"></select>
        </div>
        <div id="selected-db-summary"></div>
      </section>

      <section class="panel span-6">
        <div class="section-head">
          <div>
            <h2>Selected Database Difficulty Split</h2>
            <p>Per-difficulty execution accuracy for the chosen database.</p>
          </div>
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>Difficulty</th>
                <th>Total</th>
                <th>Correct</th>
                <th>Acc</th>
                <th>Exec Fail</th>
                <th>Wrong Result</th>
              </tr>
            </thead>
            <tbody id="selected-db-difficulty-table"></tbody>
          </table>
        </div>
      </section>

      <section class="panel span-6">
        <div class="section-head">
          <div>
            <h2>Selected Database Error Types</h2>
            <p>Primary failure mix for the chosen database.</p>
          </div>
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr><th>Error Type</th><th>Count</th><th>Share</th></tr>
            </thead>
            <tbody id="selected-db-error-table"></tbody>
          </table>
        </div>
      </section>

      <section class="panel span-12">
        <div class="section-head">
          <div>
            <h2>Example Drill-Down</h2>
            <p>Filter questions and inspect the predicted SQL, gold SQL, and error type.</p>
          </div>
        </div>
        <div class="controls">
          <select id="example-db"></select>
          <select id="example-difficulty">
            <option value="all">All difficulties</option>
            <option value="easy">Easy</option>
            <option value="moderate">Moderate</option>
            <option value="hard">Hard</option>
          </select>
          <select id="example-error"></select>
          <input id="example-search" class="search" type="search" placeholder="Search question text or SQL">
        </div>
        <div class="scroll">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Database</th>
                <th>Difficulty</th>
                <th>Outcome</th>
                <th>Question</th>
                <th>Predicted SQL</th>
                <th>Gold SQL</th>
              </tr>
            </thead>
            <tbody id="examples-table"></tbody>
          </table>
        </div>
      </section>
    </div>
  </div>

  <script>
    const analysis = {payload};

    const byDb = new Map(analysis.database_summary.map(item => [item.db_id, item]));
    const difficultyOrder = ["easy", "moderate", "hard"];
    const overallTotal = analysis.overall_summary.total_examples;

    function fmtPct(value) {{
      return `${{Number(value).toFixed(2)}}%`;
    }}

    function escapeHtml(value) {{
      return (value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function renderOverview() {{
      document.getElementById("meta-eval-path").textContent = analysis.metadata.eval_report_path;
      document.getElementById("meta-example-path").textContent = analysis.metadata.per_example_path;
      document.getElementById("meta-dataset-path").textContent = analysis.metadata.dataset_jsonl_path;

      const src = analysis.overall_summary.source_summary || {{}};
      const cards = [
        ["Execution Accuracy", fmtPct(analysis.overall_summary.execution_accuracy), `${{analysis.overall_summary.correct}} / ${{overallTotal}} correct`],
        ["Exact Match", fmtPct(analysis.overall_summary.exact_match_accuracy), `${{analysis.overall_summary.exact_match_count}} exact matches`],
        ["Exec Fail Rate", fmtPct(src.exec_fail_rate || 0), `${{analysis.overall_summary.exec_fail_count}} execution failures`],
        ["Wrong Result Rate", fmtPct(src.wrong_result_rate || 0), `${{analysis.overall_summary.wrong_result_count}} wrong-result predictions`],
        ["Best Database", analysis.overall_summary.top_database || "-", analysis.top_databases[0] ? fmtPct(analysis.top_databases[0].accuracy) : ""],
        ["Worst Database", analysis.overall_summary.bottom_database || "-", analysis.bottom_databases[0] ? fmtPct(analysis.bottom_databases[0].accuracy) : ""],
      ];
      document.getElementById("overview-cards").innerHTML = cards.map(([label, value, detail]) => `
        <div class="card">
          <div class="label">${{escapeHtml(label)}}</div>
          <div class="value">${{escapeHtml(String(value))}}</div>
          <div class="detail">${{escapeHtml(detail)}}</div>
        </div>
      `).join("");
    }}

    function getSortedDatabases() {{
      const term = document.getElementById("db-search").value.trim().toLowerCase();
      const sort = document.getElementById("db-sort").value;
      const rows = analysis.database_summary.filter(row => row.db_id.toLowerCase().includes(term));
      rows.sort((a, b) => {{
        if (sort === "accuracy_desc") return b.accuracy - a.accuracy || b.total - a.total;
        if (sort === "accuracy_asc") return a.accuracy - b.accuracy || b.total - a.total;
        if (sort === "total_desc") return b.total - a.total || b.accuracy - a.accuracy;
        return a.db_id.localeCompare(b.db_id);
      }});
      return rows;
    }}

    function renderDatabaseTable() {{
      const tbody = document.getElementById("database-table");
      const rows = getSortedDatabases();
      tbody.innerHTML = rows.map(row => `
        <tr>
          <td><span class="db-name">${{escapeHtml(row.db_id)}}</span><div class="small">dataset: ${{row.dataset_total}}</div></td>
          <td>${{row.total}}</td>
          <td class="${{row.accuracy >= 50 ? "good" : row.accuracy < 25 ? "bad" : "warn"}}">${{fmtPct(row.accuracy)}}</td>
          <td>${{fmtPct(row.difficulty_split.easy.accuracy)}} <span class="small">(${{row.difficulty_split.easy.correct}}/${{row.difficulty_split.easy.total}})</span></td>
          <td>${{fmtPct(row.difficulty_split.moderate.accuracy)}} <span class="small">(${{row.difficulty_split.moderate.correct}}/${{row.difficulty_split.moderate.total}})</span></td>
          <td>${{fmtPct(row.difficulty_split.hard.accuracy)}} <span class="small">(${{row.difficulty_split.hard.correct}}/${{row.difficulty_split.hard.total}})</span></td>
          <td>${{row.exec_failed}}</td>
          <td>${{row.wrong_result}}</td>
        </tr>
      `).join("");
    }}

    function renderRankings() {{
      const ranking = [
        ...analysis.top_databases.map(row => [row, false, "Top"]),
        ...analysis.bottom_databases.map(row => [row, true, "Bottom"]),
      ];
      document.getElementById("ranking-bars").innerHTML = ranking.map(([row, bad, label]) => `
        <div class="bar-row ${{bad ? "bad" : ""}}">
          <div>
            <div class="db-name">${{escapeHtml(row.db_id)}}</div>
            <div class="small">${{label}} database</div>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${{row.accuracy}}%"></div></div>
          <div>${{fmtPct(row.accuracy)}}</div>
        </div>
      `).join("");
    }}

    function renderDifficultyCards() {{
      document.getElementById("difficulty-cards").innerHTML = analysis.difficulty_summary.map(item => `
        <div class="split-card">
          <div class="label">${{escapeHtml(item.difficulty)}}</div>
          <div class="big">${{fmtPct(item.accuracy)}}</div>
          <div class="small">${{item.correct}} / ${{item.total}} correct</div>
          <div class="small">exec fail: ${{item.exec_failed}} | wrong result: ${{item.wrong_result}}</div>
        </div>
      `).join("");
    }}

    function renderOverallErrors() {{
      const rows = Object.entries(analysis.overall_error_breakdown)
        .sort((a, b) => b[1] - a[1]);
      document.getElementById("overall-error-table").innerHTML = rows.map(([name, count]) => `
        <tr>
          <td>${{escapeHtml(name)}}</td>
          <td>${{count}}</td>
          <td>${{fmtPct((count / overallTotal) * 100)}}</td>
        </tr>
      `).join("");
    }}

    function populateSelectors() {{
      const databaseOptions = ['<option value="all">All databases</option>']
        .concat(analysis.database_summary.map(row => `<option value="${{row.db_id}}">${{row.db_id}}</option>`))
        .join("");
      document.getElementById("database-focus").innerHTML = analysis.database_summary
        .map(row => `<option value="${{row.db_id}}">${{row.db_id}}</option>`).join("");
      document.getElementById("example-db").innerHTML = databaseOptions;

      const errorOptions = ['all', ...Object.keys(analysis.overall_error_breakdown).sort()];
      document.getElementById("example-error").innerHTML = errorOptions
        .map(value => `<option value="${{value}}">${{value === 'all' ? 'All error types' : value}}</option>`)
        .join("");
    }}

    function renderSelectedDatabase() {{
      const dbId = document.getElementById("database-focus").value;
      const row = byDb.get(dbId);
      if (!row) return;

      document.getElementById("selected-db-summary").innerHTML = `
        <div class="split-grid">
          <div class="split-card">
            <div class="label">overall accuracy</div>
            <div class="big">${{fmtPct(row.accuracy)}}</div>
            <div class="small">${{row.correct}} / ${{row.total}} correct</div>
          </div>
          <div class="split-card">
            <div class="label">exec failures</div>
            <div class="big">${{row.exec_failed}}</div>
            <div class="small">${{fmtPct((row.exec_failed / row.total) * 100)}} of database</div>
          </div>
          <div class="split-card">
            <div class="label">wrong results</div>
            <div class="big">${{row.wrong_result}}</div>
            <div class="small">${{fmtPct((row.wrong_result / row.total) * 100)}} of database</div>
          </div>
        </div>
      `;

      document.getElementById("selected-db-difficulty-table").innerHTML = difficultyOrder.map(diff => {{
        const item = row.difficulty_split[diff];
        return `
          <tr>
            <td>${{escapeHtml(diff)}}</td>
            <td>${{item.total}}</td>
            <td>${{item.correct}}</td>
            <td>${{fmtPct(item.accuracy)}}</td>
            <td>${{item.exec_failed}}</td>
            <td>${{item.wrong_result}}</td>
          </tr>
        `;
      }}).join("");

      const errorRows = Object.entries(row.error_breakdown)
        .sort((a, b) => b[1] - a[1]);
      document.getElementById("selected-db-error-table").innerHTML = errorRows.length
        ? errorRows.map(([name, count]) => `
            <tr>
              <td>${{escapeHtml(name)}}</td>
              <td>${{count}}</td>
              <td>${{fmtPct((count / row.total) * 100)}}</td>
            </tr>
          `).join("")
        : '<tr><td colspan="3" class="empty">No examples available for this database.</td></tr>';
    }}

    function renderExamples() {{
      const dbValue = document.getElementById("example-db").value;
      const difficultyValue = document.getElementById("example-difficulty").value;
      const errorValue = document.getElementById("example-error").value;
      const search = document.getElementById("example-search").value.trim().toLowerCase();

      let rows = analysis.examples.filter(row => {{
        if (dbValue !== "all" && row.db_id !== dbValue) return false;
        if (difficultyValue !== "all" && row.difficulty !== difficultyValue) return false;
        if (errorValue !== "all" && row.primary_error !== errorValue) return false;
        if (!search) return true;
        const haystack = [
          row.question,
          row.predicted_sql,
          row.gold_sql,
          row.db_id,
          row.primary_error,
        ].join("\\n").toLowerCase();
        return haystack.includes(search);
      }});

      rows = rows.slice().sort((a, b) => {{
        if (a.correct !== b.correct) return Number(a.correct) - Number(b.correct);
        return a.question_id - b.question_id;
      }}).slice(0, 200);

      document.getElementById("examples-table").innerHTML = rows.length
        ? rows.map(row => `
            <tr>
              <td>${{row.question_id}}</td>
              <td>${{escapeHtml(row.db_id)}}</td>
              <td>${{escapeHtml(row.difficulty)}}</td>
              <td><span class="tag">${{escapeHtml(row.primary_error)}}</span></td>
              <td>
                <div>${{escapeHtml(row.question || "")}}</div>
                ${{row.evidence ? `<div class="small">${{escapeHtml(row.evidence)}}</div>` : ""}}
              </td>
              <td class="mono">${{escapeHtml(row.predicted_sql || "")}}</td>
              <td class="mono">${{escapeHtml(row.gold_sql || "")}}</td>
            </tr>
          `).join("")
        : '<tr><td colspan="7" class="empty">No examples match the selected filters.</td></tr>';
    }}

    renderOverview();
    renderDatabaseTable();
    renderRankings();
    renderDifficultyCards();
    renderOverallErrors();
    populateSelectors();
    renderSelectedDatabase();
    renderExamples();

    document.getElementById("db-search").addEventListener("input", renderDatabaseTable);
    document.getElementById("db-sort").addEventListener("change", renderDatabaseTable);
    document.getElementById("database-focus").addEventListener("change", renderSelectedDatabase);
    document.getElementById("example-db").addEventListener("change", renderExamples);
    document.getElementById("example-difficulty").addEventListener("change", renderExamples);
    document.getElementById("example-error").addEventListener("change", renderExamples);
    document.getElementById("example-search").addEventListener("input", renderExamples);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--per-example", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--bird-dev-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="T10 Interactive Eval Analysis",
        help="HTML page title",
    )
    args = parser.parse_args()

    analysis = build_analysis(
        eval_report_path=args.eval_report,
        per_example_path=args.per_example,
        dataset_jsonl_path=args.dataset_jsonl,
        bird_dev_json_path=args.bird_dev_json,
    )

    args.output_html.write_text(render_html(analysis, args.title))
    print(f"Wrote {args.output_html}")


if __name__ == "__main__":
    main()
