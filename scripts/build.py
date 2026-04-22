"""
AI & Organizational Change — daily digest builder.

Pipeline:
  1. Parse every RSS feed in sources.yaml.
  2. Skip already-seen URLs and entries older than LOOKBACK_DAYS.
  3. Cheap keyword pre-filter to cut the LLM bill.
  4. Claude Haiku scores relevance (1-10) and writes a Chinese summary.
  5. Entries with score >= MIN_SCORE are persisted to data/YYYY-MM-DD.json.
  6. Rebuild docs/index.html (last LOOKBACK_DAYS) and docs/archive/YYYY-MM-DD.html.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any

import feedparser
import yaml
from anthropic import Anthropic
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"
SEEN_PATH = DATA_DIR / "seen.json"
SOURCES_PATH = ROOT / "sources.yaml"

LOOKBACK_DAYS = 7          # only consider feed entries published within this window
RENDER_WINDOW_DAYS = 7     # how many past days to show on the index page (older → archive)
MIN_SCORE = 6              # items below this score are dropped
MAX_CONTENT_CHARS = 4000   # truncate article body sent to Claude
MODEL = "claude-haiku-4-5"

CNY_TZ = timezone(timedelta(hours=8))


@dataclass
class Entry:
    url: str
    title: str
    source: str
    category: str
    published: str        # ISO date (YYYY-MM-DD)
    fetched_at: str       # ISO datetime in CST
    score: int
    summary_zh: str
    key_points: list[str]
    tags: list[str]
    speakers: list[str]


# ---------- helpers ----------

def load_sources() -> dict[str, Any]:
    with open(SOURCES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seen() -> set[str]:
    if SEEN_PATH.exists():
        with open(SEEN_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_seen(seen: set[str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f, ensure_ascii=False, indent=2)


def strip_html(html: str) -> str:
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def entry_published(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        tm = entry.get(key)
        if tm:
            return datetime(*tm[:6], tzinfo=timezone.utc)
    return None


def keyword_hit(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


# ---------- Claude scoring ----------

SYSTEM_PROMPT = """你是一位资深内容编辑，专门筛选「AI（尤其是生成式 AI / 大模型 / Agent）对组织、管理、工作方式的影响」相关的英文内容，并用中文为中国读者做摘要。

评估维度（1-10 分，越高越相关）：
- 10：高层（CEO/CPO/CHRO/CTO）直接阐述 AI 如何改变组织结构、人员、流程、招聘、文化
- 8-9：严肃的咨询/研究/VC 文章，讨论 AI 对企业运营、劳动力、生产力的深层影响
- 6-7：有数据或案例的 AI 工作场景应用、效率变化、能力再定义
- 4-5：泛泛而谈的 AI 行业动态、模型发布，与组织变革关联弱
- 1-3：纯技术细节、纯产品发布、不涉及组织/人/工作方式的内容

请只输出一段 JSON（不要包裹 markdown 代码块），字段如下：
{
  "score": 1-10 的整数,
  "summary_zh": "150-200 字的中文摘要，突出对组织/管理/人员的影响与具体观点，避免套话",
  "key_points": ["3-5 条要点，每条不超过 40 字"],
  "tags": ["2-4 个中文标签，如 组织重构 / 人才策略 / 生产力 / Agent 落地 / 领导力"],
  "speakers": ["文中明确提到的有观点的人名或机构，如 Satya Nadella / Block / McKinsey；没有则为空数组"]
}

注意：如果文章与「AI 对组织/工作方式的影响」基本无关，给出较低 score（< 6），摘要仍按事实简短写。"""


def score_entry(client: Anthropic, title: str, source: str, content: str) -> dict[str, Any] | None:
    user_msg = (
        f"来源: {source}\n"
        f"标题: {title}\n"
        f"正文（可能截断）:\n{content[:MAX_CONTENT_CHARS]}"
    )
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        print(f"  ! Claude API error: {e}", file=sys.stderr)
        return None

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # Tolerate accidental code fences.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"  ! Could not parse JSON: {text[:200]}", file=sys.stderr)
        return None

    # Defensive normalization.
    data["score"] = int(data.get("score", 0))
    data["summary_zh"] = str(data.get("summary_zh", "")).strip()
    data["key_points"] = [str(x).strip() for x in data.get("key_points", []) if x]
    data["tags"] = [str(x).strip() for x in data.get("tags", []) if x]
    data["speakers"] = [str(x).strip() for x in data.get("speakers", []) if x]
    return data


# ---------- fetch + score pipeline ----------

def run_pipeline() -> list[Entry]:
    config = load_sources()
    seen = load_seen()
    keywords = config["keywords"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    client = Anthropic()  # uses ANTHROPIC_API_KEY env var
    results: list[Entry] = []

    for feed_def in config["feeds"]:
        name = feed_def["name"]
        url = feed_def["url"]
        category = feed_def.get("category", "other")
        print(f"\n== {name} ==")
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            print(f"  ! fetch failed: {e}", file=sys.stderr)
            continue
        if parsed.bozo and not parsed.entries:
            print(f"  ! feed returned no entries (bozo={parsed.bozo_exception})")
            continue

        for entry in parsed.entries[:30]:
            link = entry.get("link", "").split("#")[0].split("?utm")[0]
            if not link or link in seen:
                continue
            pub = entry_published(entry)
            if pub and pub < cutoff:
                continue

            title = strip_html(entry.get("title", "")).strip()
            body = entry.get("content", [{}])[0].get("value", "") if entry.get("content") else ""
            if not body:
                body = entry.get("summary", "") or entry.get("description", "")
            text = strip_html(body)

            blob = f"{title} {text}"
            if not keyword_hit(blob, keywords):
                seen.add(link)
                continue

            print(f"  -> scoring: {title[:80]}")
            scored = score_entry(client, title, name, text)
            seen.add(link)  # don't re-attempt even on low score
            if not scored or scored["score"] < MIN_SCORE:
                continue

            results.append(Entry(
                url=link,
                title=title,
                source=name,
                category=category,
                published=(pub.astimezone(CNY_TZ).date().isoformat()
                           if pub else datetime.now(CNY_TZ).date().isoformat()),
                fetched_at=datetime.now(CNY_TZ).isoformat(timespec="seconds"),
                score=scored["score"],
                summary_zh=scored["summary_zh"],
                key_points=scored["key_points"],
                tags=scored["tags"],
                speakers=scored["speakers"],
            ))
            time.sleep(0.3)  # gentle rate limit

    save_seen(seen)
    return results


# ---------- persistence ----------

def save_entries(new_entries: list[Entry]) -> None:
    """Append new entries into per-day JSON files keyed by fetch date (CST)."""
    today = datetime.now(CNY_TZ).date().isoformat()
    path = DATA_DIR / f"{today}.json"
    existing: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    existing_urls = {e["url"] for e in existing}
    for e in new_entries:
        if e.url not in existing_urls:
            existing.append(asdict(e))
    existing.sort(key=lambda e: (-e["score"], e["source"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_all_days(max_days: int) -> dict[str, list[dict]]:
    files = sorted(DATA_DIR.glob("20*.json"), reverse=True)[:max_days]
    out = {}
    for p in files:
        with open(p, encoding="utf-8") as f:
            out[p.stem] = json.load(f)
    return out


# ---------- rendering ----------

PAGE_CSS = """
:root {
  --bg: #0f1419; --panel: #161b22; --border: #2a313c;
  --fg: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --chip: #1f2937;
}
@media (prefers-color-scheme: light) {
  :root { --bg:#f7f8fa; --panel:#fff; --border:#e3e6eb; --fg:#1a1f2b; --muted:#6a7380; --accent:#1e5fd4; --chip:#eef1f5; }
}
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", "Microsoft YaHei", sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.6; }
.wrap { max-width: 900px; margin: 0 auto; padding: 32px 20px 80px; }
header h1 { font-size: 26px; margin: 0 0 6px; }
header p { color: var(--muted); margin: 0 0 24px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; padding: 12px;
           background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
.filters label { font-size: 13px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
.filters select, .filters input[type=range] { background: var(--chip); color: var(--fg); border: 1px solid var(--border);
           border-radius: 6px; padding: 4px 8px; font-size: 13px; }
.day { margin-bottom: 36px; }
.day h2 { font-size: 18px; border-bottom: 1px solid var(--border); padding-bottom: 6px; margin: 0 0 14px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
        padding: 16px 18px; margin-bottom: 14px; }
.card h3 { margin: 0 0 6px; font-size: 16px; line-height: 1.4; }
.card h3 a { color: var(--fg); text-decoration: none; }
.card h3 a:hover { color: var(--accent); }
.meta { display: flex; flex-wrap: wrap; gap: 10px; font-size: 12px; color: var(--muted); margin-bottom: 10px; }
.score { background: var(--accent); color: #fff; padding: 1px 7px; border-radius: 10px; font-weight: 600; }
.score.s-low { background: #888; }
.summary { margin: 8px 0 10px; }
.bullets { margin: 6px 0 10px; padding-left: 20px; }
.bullets li { margin-bottom: 2px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { background: var(--chip); color: var(--muted); padding: 2px 8px; border-radius: 10px;
        font-size: 12px; border: 1px solid var(--border); }
.chip.speaker { color: var(--accent); }
.empty { color: var(--muted); font-style: italic; padding: 40px 0; text-align: center; }
footer { color: var(--muted); font-size: 12px; margin-top: 40px; text-align: center; }
footer a { color: var(--muted); }
"""

FILTER_JS = """
function applyFilters() {
  const src = document.getElementById('f-source').value;
  const cat = document.getElementById('f-category').value;
  const min = parseInt(document.getElementById('f-score').value, 10);
  document.getElementById('f-score-val').textContent = min;
  document.querySelectorAll('.card').forEach(c => {
    const s = c.dataset.source, k = c.dataset.category, sc = parseInt(c.dataset.score, 10);
    const ok = (src === '*' || s === src) && (cat === '*' || k === cat) && sc >= min;
    c.style.display = ok ? '' : 'none';
  });
  document.querySelectorAll('.day').forEach(d => {
    const any = Array.from(d.querySelectorAll('.card')).some(c => c.style.display !== 'none');
    d.style.display = any ? '' : 'none';
  });
}
document.addEventListener('DOMContentLoaded', () => {
  ['f-source', 'f-category', 'f-score'].forEach(id => {
    document.getElementById(id).addEventListener('input', applyFilters);
  });
});
"""


def render_card(e: dict) -> str:
    score_class = "score" + (" s-low" if e["score"] < 7 else "")
    bullets = "".join(f"<li>{escape(p)}</li>" for p in e.get("key_points", []))
    tags = "".join(f'<span class="chip">{escape(t)}</span>' for t in e.get("tags", []))
    speakers = "".join(f'<span class="chip speaker">{escape(s)}</span>' for s in e.get("speakers", []))
    return f"""
    <article class="card" data-source="{escape(e['source'])}" data-category="{escape(e['category'])}" data-score="{e['score']}">
      <div class="meta">
        <span class="{score_class}">{e['score']}/10</span>
        <span>{escape(e['source'])}</span>
        <span>·</span>
        <span>{escape(e['category'])}</span>
        <span>·</span>
        <span>{escape(e['published'])}</span>
      </div>
      <h3><a href="{escape(e['url'])}" target="_blank" rel="noopener">{escape(e['title'])}</a></h3>
      <p class="summary">{escape(e['summary_zh'])}</p>
      <ul class="bullets">{bullets}</ul>
      <div class="chips">{speakers}{tags}</div>
    </article>
    """


def render_page(title: str, subtitle: str, days: dict[str, list[dict]], rel_root: str = "") -> str:
    sources = sorted({e["source"] for entries in days.values() for e in entries})
    categories = sorted({e["category"] for entries in days.values() for e in entries})
    opts_src = "\n".join(f'<option value="{escape(s)}">{escape(s)}</option>' for s in sources)
    opts_cat = "\n".join(f'<option value="{escape(c)}">{escape(c)}</option>' for c in categories)

    body = []
    for day, entries in days.items():
        if not entries:
            continue
        entries = sorted(entries, key=lambda e: (-e["score"], e["source"]))
        cards = "".join(render_card(e) for e in entries)
        body.append(f'<section class="day"><h2>{escape(day)} · {len(entries)} 条</h2>{cards}</section>')
    content = "\n".join(body) or '<div class="empty">暂无数据。首次运行后会在这里显示。</div>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>{escape(title)}</h1>
  <p>{escape(subtitle)} · <a href="{rel_root}index.html">最新</a> · <a href="{rel_root}archive/">归档</a></p>
</header>
<div class="filters">
  <label>来源
    <select id="f-source"><option value="*">全部</option>{opts_src}</select>
  </label>
  <label>分类
    <select id="f-category"><option value="*">全部</option>{opts_cat}</select>
  </label>
  <label>最低分 <span id="f-score-val">6</span>
    <input type="range" id="f-score" min="6" max="10" value="6">
  </label>
</div>
{content}
<footer>
  由 <a href="https://github.com/anthropics/claude-code" target="_blank">Claude Haiku 4.5</a> 每日构建 ·
  生成时间 {datetime.now(CNY_TZ).strftime('%Y-%m-%d %H:%M CST')}
</footer>
</div>
<script>{FILTER_JS}</script>
</body>
</html>
"""


_ARCHIVE_CSS_EXTRA = """
.archive-list { list-style: none; padding: 0; margin: 0; display: grid;
                grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 6px 16px; }
.archive-list li { padding: 4px 0; }
.archive-list a { color: var(--fg); text-decoration: none; }
.archive-list a:hover { color: var(--accent); }
.archive-list .count { color: var(--muted); font-size: 12px; margin-left: 6px; }
"""


def render_archive_index(days_with_counts: list[tuple[str, int]]) -> str:
    if not days_with_counts:
        body = '<p class="empty">暂无归档</p>'
    else:
        from collections import defaultdict
        by_month: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for d, c in days_with_counts:
            by_month[d[:7]].append((d, c))
        sections = []
        for month in sorted(by_month.keys(), reverse=True):
            days_in_month = sorted(by_month[month], key=lambda x: x[0], reverse=True)
            items = "".join(
                f'<li><a href="{d}.html">{d}</a><span class="count">{c} 条</span></li>'
                for d, c in days_in_month
            )
            total_entries = sum(c for _, c in days_in_month)
            sections.append(
                f'<section class="day"><h2>{month} · {len(days_in_month)} 天 / {total_entries} 条</h2>'
                f'<ul class="archive-list">{items}</ul></section>'
            )
        body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>归档 · AI 与组织变革</title>
<style>{PAGE_CSS}{_ARCHIVE_CSS_EXTRA}</style></head>
<body><div class="wrap">
<header><h1>归档</h1><p><a href="../index.html">← 返回最新</a></p></header>
{body}
</div></body></html>
"""


def rebuild_site() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    recent = load_all_days(RENDER_WINDOW_DAYS)
    index_html = render_page(
        "AI 与组织变革 · 每日精选",
        f"过去 {RENDER_WINDOW_DAYS} 天 · 来自 HBR / McKinsey / Sequoia / a16z / OpenAI / Anthropic 等",
        recent,
    )
    (DOCS_DIR / "index.html").write_text(index_html, encoding="utf-8")

    # Per-day archive pages (full history)
    all_files = sorted(DATA_DIR.glob("20*.json"), reverse=True)
    days_with_counts: list[tuple[str, int]] = []
    for p in all_files:
        day = p.stem
        with open(p, encoding="utf-8") as f:
            entries = json.load(f)
        days_with_counts.append((day, len(entries)))
        html = render_page(
            f"{day} · AI 与组织变革",
            f"当日 {len(entries)} 条",
            {day: entries},
            rel_root="../",
        )
        (ARCHIVE_DIR / f"{day}.html").write_text(html, encoding="utf-8")

    (ARCHIVE_DIR / "index.html").write_text(
        render_archive_index(days_with_counts), encoding="utf-8"
    )


# ---------- entrypoint ----------

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    new_entries = run_pipeline()
    print(f"\n== {len(new_entries)} new entries kept ==")
    if new_entries:
        save_entries(new_entries)
    rebuild_site()
    print("Site rebuilt at docs/")


if __name__ == "__main__":
    main()
