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
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0a0a0a; color: #999; font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; min-height: 100vh; }
a { color: inherit; text-decoration: none; }
a:hover { color: #ccc; }
.back-link { position: fixed; top: 18px; left: 20px; font-family: 'VT323', monospace; font-size: 1rem; color: #444; z-index: 100; letter-spacing: 0.02em; transition: color 0.15s; }
.back-link:hover { color: #888; }
header { padding: 60px 20px 32px; text-align: center; }
header h1 { font-family: 'VT323', monospace; font-size: clamp(1.8rem, 5.5vw, 3.6rem); color: #888; letter-spacing: 0.12em; line-height: 1.1; margin-bottom: 10px; word-spacing: 0.1em; }
header p { font-size: 0.78rem; color: #444; letter-spacing: 0.1em; }
header p a { color: #555; transition: color 0.15s; }
header p a:hover { color: #888; }
.controls { background: #0a0a0a; border-bottom: 1px solid #161616; position: sticky; top: 0; z-index: 50; }
.tabs { display: flex; overflow-x: auto; scrollbar-width: none; padding: 0 16px; max-width: 760px; margin: 0 auto; }
.tabs::-webkit-scrollbar { display: none; }
.tab { flex-shrink: 0; background: none; border: none; border-bottom: 2px solid transparent; padding: 12px 10px; font-size: 0.75rem; color: #555; cursor: pointer; font-family: system-ui, sans-serif; display: flex; align-items: center; gap: 4px; white-space: nowrap; transition: color 0.15s, border-color 0.15s; letter-spacing: 0.03em; }
.tab.active { color: #999; border-bottom-color: #555; }
.tab:hover { color: #888; }
.badge { font-size: 0.65rem; color: #444; }
.tab.active .badge { color: #777; }
.filter-bar { display: flex; align-items: center; gap: 10px; padding: 8px 16px; border-top: 1px solid #111; max-width: 760px; margin: 0 auto; font-size: 0.72rem; color: #444; flex-wrap: wrap; }
.filter-bar > span { letter-spacing: 0.05em; }
.filter-bar .sep { color: #222; }
.score-filter { display: flex; gap: 6px; flex-wrap: wrap; }
.sf-btn { background: none; border: 1px solid #222; border-radius: 3px; padding: 2px 10px; font-size: 0.7rem; color: #555; cursor: pointer; font-family: inherit; transition: color 0.15s, border-color 0.15s; letter-spacing: 0.05em; }
.sf-btn.active, .sf-btn:hover { color: #999; border-color: #555; }
.source-select { background: #0a0a0a; border: 1px solid #222; border-radius: 3px; padding: 2px 8px; font-size: 0.7rem; color: #777; font-family: inherit; cursor: pointer; outline: none; }
.source-select:hover { color: #999; border-color: #555; }
.source-select option { background: #0a0a0a; color: #999; }
.list { max-width: 760px; margin: 0 auto; padding: 8px 20px 80px; }
.day { padding-top: 28px; }
.day-head { font-size: 0.7rem; color: #444; letter-spacing: 0.12em; padding: 0 0 10px; border-bottom: 1px solid #161616; margin-bottom: 4px; text-transform: uppercase; }
.day-head .count { color: #333; margin-left: 6px; }
.entry { padding: 18px 0; border-bottom: 1px solid #111; }
.entry.hidden { display: none; }
.entry-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 0.7rem; color: #444; margin-bottom: 6px; letter-spacing: 0.04em; }
.entry-meta .sep { color: #222; }
.entry-score { color: #888; font-variant-numeric: tabular-nums; letter-spacing: 0.05em; }
.entry-score.s-low { color: #555; }
.entry-title { display: block; font-size: 0.95rem; color: #999; transition: color 0.15s; line-height: 1.5; margin-bottom: 8px; }
.entry-title:hover { color: #ccc; }
.entry-summary { font-size: 0.82rem; color: #666; line-height: 1.7; margin-bottom: 8px; }
.entry-bullets { list-style: none; padding: 0; margin: 6px 0 10px; }
.entry-bullets li { font-size: 0.78rem; color: #555; line-height: 1.7; padding-left: 14px; position: relative; }
.entry-bullets li::before { content: '·'; position: absolute; left: 4px; color: #444; }
.entry-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.entry-chip { font-size: 0.66rem; color: #555; border: 1px solid #222; padding: 1px 7px; border-radius: 2px; letter-spacing: 0.04em; }
.entry-chip.speaker { color: #777; border-color: #2a2a2a; }
.empty-state { text-align: center; padding: 60px 20px; color: #444; font-size: 0.82rem; }
@media (max-width: 480px) {
  .list { padding: 8px 14px 60px; }
  .entry-title { font-size: 0.88rem; }
  .filter-bar { font-size: 0.68rem; gap: 8px; }
}
footer { text-align: center; padding: 30px 20px; color: #333; font-size: 0.68rem; letter-spacing: 0.08em; }
footer a { color: #444; transition: color 0.15s; }
footer a:hover { color: #777; }
"""

FILTER_JS = """
function applyFilters() {
  const activeTab = document.querySelector('.tab.active');
  const cat = activeTab ? activeTab.dataset.cat : 'all';
  const activeScore = document.querySelector('.sf-btn.active');
  const minScore = activeScore ? parseInt(activeScore.dataset.score, 10) : 6;
  const srcEl = document.getElementById('f-source');
  const src = srcEl ? srcEl.value : '*';
  document.querySelectorAll('.entry').forEach(e => {
    const ec = e.dataset.category, es = e.dataset.source, esc = parseInt(e.dataset.score, 10);
    const show = (cat === 'all' || ec === cat) && (src === '*' || es === src) && esc >= minScore;
    e.classList.toggle('hidden', !show);
  });
  document.querySelectorAll('.day').forEach(d => {
    const any = Array.from(d.querySelectorAll('.entry')).some(e => !e.classList.contains('hidden'));
    d.style.display = any ? '' : 'none';
  });
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    applyFilters();
  }));
  document.querySelectorAll('.sf-btn').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.sf-btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    applyFilters();
  }));
  const srcEl = document.getElementById('f-source');
  if (srcEl) srcEl.addEventListener('change', applyFilters);
});
"""

CATEGORY_LABELS = {
    "ai-lab": "AI 实验室",
    "analyst": "分析师",
    "business": "商业",
    "vc": "投资",
    "podcast": "播客",
    "other": "其他",
}


def cat_label(slug: str) -> str:
    return CATEGORY_LABELS.get(slug, slug)


def render_entry(e: dict) -> str:
    score_class = "entry-score" + (" s-low" if e["score"] < 7 else "")
    bullets = "".join(f"<li>{escape(p)}</li>" for p in e.get("key_points", []))
    tags = "".join(f'<span class="entry-chip">{escape(t)}</span>' for t in e.get("tags", []))
    speakers = "".join(f'<span class="entry-chip speaker">{escape(s)}</span>' for s in e.get("speakers", []))
    chips = f'<div class="entry-chips">{speakers}{tags}</div>' if (tags or speakers) else ""
    bullets_html = f'<ul class="entry-bullets">{bullets}</ul>' if bullets else ""
    return f"""
    <article class="entry" data-source="{escape(e['source'])}" data-category="{escape(e['category'])}" data-score="{e['score']}">
      <div class="entry-meta">
        <span class="{score_class}">{e['score']}/10</span>
        <span class="sep">·</span>
        <span>{escape(e['source'])}</span>
        <span class="sep">·</span>
        <span>{escape(cat_label(e['category']))}</span>
        <span class="sep">·</span>
        <span>{escape(e['published'])}</span>
      </div>
      <a class="entry-title" href="{escape(e['url'])}" target="_blank" rel="noopener">{escape(e['title'])}</a>
      <p class="entry-summary">{escape(e['summary_zh'])}</p>
      {bullets_html}
      {chips}
    </article>
    """


_HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=VT323&display=swap" rel="stylesheet">"""


def render_page(
    heading: str,
    subtitle: str,
    days: dict[str, list[dict]],
    rel_root: str = "",
    page_title: str | None = None,
    show_filters: bool = True,
) -> str:
    all_entries = [e for entries in days.values() for e in entries]
    sources = sorted({e["source"] for e in all_entries})
    categories = sorted({e["category"] for e in all_entries})
    cat_counts: dict[str, int] = {c: 0 for c in categories}
    for e in all_entries:
        cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1

    cat_tabs = ['<button class="tab active" data-cat="all">全部'
                f'<span class="badge">{len(all_entries)}</span></button>']
    for c in categories:
        cat_tabs.append(
            f'<button class="tab" data-cat="{escape(c)}">{escape(cat_label(c))}'
            f'<span class="badge">{cat_counts[c]}</span></button>'
        )
    tabs_html = "".join(cat_tabs)

    opts_src = "".join(f'<option value="{escape(s)}">{escape(s)}</option>' for s in sources)

    body = []
    for day, entries in days.items():
        if not entries:
            continue
        entries = sorted(entries, key=lambda e: (-e["score"], e["source"]))
        items = "".join(render_entry(e) for e in entries)
        body.append(
            f'<section class="day">'
            f'<div class="day-head">{escape(day)}<span class="count">· {len(entries)} 条</span></div>'
            f'{items}</section>'
        )
    content = "\n".join(body) or '<div class="empty-state">暂无数据。</div>'

    controls_html = ""
    if show_filters and all_entries:
        controls_html = f"""<div class="controls">
  <div class="tabs">{tabs_html}</div>
  <div class="filter-bar">
    <span>评分</span>
    <div class="score-filter">
      <button class="sf-btn active" data-score="6">≥6</button>
      <button class="sf-btn" data-score="7">≥7</button>
      <button class="sf-btn" data-score="8">≥8</button>
      <button class="sf-btn" data-score="9">≥9</button>
    </div>
    <span class="sep">·</span>
    <span>来源</span>
    <select class="source-select" id="f-source"><option value="*">全部</option>{opts_src}</select>
  </div>
</div>"""

    final_title = page_title or heading
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
{_HEAD}
<title>{escape(final_title)}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<a class="back-link" href="https://www.tedzhong.com/">← tedzhong.com</a>
<header>
  <h1>{escape(heading)}</h1>
  <p>{subtitle}</p>
</header>
{controls_html}
<div class="list">
{content}
</div>
<footer>
  由 Claude Haiku 4.5 每日构建 · {datetime.now(CNY_TZ).strftime('%Y-%m-%d %H:%M CST')}
</footer>
<script>{FILTER_JS}</script>
</body>
</html>
"""


_ARCHIVE_CSS_EXTRA = """
.archive-list { list-style: none; padding: 0; margin: 0; display: grid;
                grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 4px 16px; }
.archive-list li { padding: 4px 0; font-size: 0.82rem; }
.archive-list a { color: #999; transition: color 0.15s; font-variant-numeric: tabular-nums; }
.archive-list a:hover { color: #ccc; }
.archive-list .count { color: #444; font-size: 0.7rem; margin-left: 6px; }
.archive-section { padding-top: 28px; }
.archive-section .day-head { margin-bottom: 12px; }
"""


def render_archive_index(days_with_counts: list[tuple[str, int]]) -> str:
    if not days_with_counts:
        inner = '<div class="empty-state">暂无归档</div>'
    else:
        from collections import defaultdict
        by_month: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for d, c in days_with_counts:
            by_month[d[:7]].append((d, c))
        sections = []
        for month in sorted(by_month.keys(), reverse=True):
            days_in_month = sorted(by_month[month], key=lambda x: x[0], reverse=True)
            items = "".join(
                f'<li><a href="{d}.html">{d}</a><span class="count">{c}</span></li>'
                for d, c in days_in_month
            )
            total_entries = sum(c for _, c in days_in_month)
            sections.append(
                f'<section class="archive-section">'
                f'<div class="day-head">{month}<span class="count">· {len(days_in_month)} 天 / {total_entries} 条</span></div>'
                f'<ul class="archive-list">{items}</ul></section>'
            )
        inner = "\n".join(sections)

    total_days = len(days_with_counts)
    total_entries = sum(c for _, c in days_with_counts)
    subtitle = (f'{total_days} 天 · {total_entries} 条 · '
                f'<a href="../index.html">← 返回最新</a>')

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
{_HEAD}
<title>ARCHIVE — AI 与组织变革</title>
<style>{PAGE_CSS}{_ARCHIVE_CSS_EXTRA}</style>
</head>
<body>
<a class="back-link" href="https://www.tedzhong.com/">← tedzhong.com</a>
<header>
  <h1>ARCHIVE</h1>
  <p>{subtitle}</p>
</header>
<div class="list">
{inner}
</div>
<footer>
  由 Claude Haiku 4.5 每日构建 · {datetime.now(CNY_TZ).strftime('%Y-%m-%d %H:%M CST')}
</footer>
</body>
</html>
"""


def rebuild_site() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    recent = load_all_days(RENDER_WINDOW_DAYS)
    total_recent = sum(len(v) for v in recent.values())
    index_html = render_page(
        heading="AI TRANSFORMATION",
        subtitle=(f'{total_recent} entries · 过去 {RENDER_WINDOW_DAYS} 天 · '
                  f'<a href="archive/">归档</a>'),
        days=recent,
        page_title="AI TRANSFORMATION — tedzhong.com",
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
            heading=day,
            subtitle=(f'{len(entries)} 条 · '
                      f'<a href="../index.html">← 返回最新</a> · '
                      f'<a href="./">归档</a>'),
            days={day: entries},
            rel_root="../",
            page_title=f"{day} — AI TRANSFORMATION",
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
