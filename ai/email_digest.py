import argparse
import html
import json
import os
import random
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import dotenv
import markdown
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_AI_DIR = os.path.dirname(os.path.abspath(__file__))
if _AI_DIR not in sys.path:
    sys.path.insert(0, _AI_DIR)
from digest_structure import DigestStructured, DigestTheme, DigestThemesOnly

SKILL_PATH = os.path.join(os.path.dirname(__file__), "skill.txt")

DIGEST_PRODUCT_TITLE = "每日硬凑师兄要求的创新点"

# 文献池中 [k] 引用（k 为 1–30），且非 Markdown 链接：](url) 已由 (?!\() 排除；
# 不要用 (?<!\])：否则 [13][14] 里 [14] 紧接在 ] 后无法匹配。
POOL_REF_RE = re.compile(r"(?<!\[)\[(?:[1-9]|[12][0-9]|30)\](?!\()")

_POOL_NEXT_BRACKET_RE = re.compile(
    r"(?<![0-9])(\d{1,2})(?=\[(?:[1-9]|[12][0-9]|30)\])"
)


def normalize_glued_numeric_pool_refs(md: str) -> str:
    """把模型输出的「13[14][17]」规范成「[13][14][17]」，便于后续识别池中编号。"""

    def norm(m: re.Match[str]) -> str:
        k = int(m.group(1))
        if k < 1 or k > 30:
            return m.group(0)
        return f"[{k}]"

    return _POOL_NEXT_BRACKET_RE.sub(norm, md)

DIGEST_CONSTRAINTS = """
---
# 每日邮件模式（覆盖 skill 中与文献数量冲突的条款）

你正在生成的是「当日 arXiv 抓取论文」的创新点构思与白皮书式 Markdown，不是让用户再去检索外部文献。

**文献池（唯一合法引用来源）**
- 用户消息中会给出若干篇论文，编号为 [1]、[2]、…、[N]（N≤30）。
- 你只能引用这些编号对应的论文（可用编号或 arXiv id 指代）。**禁止**引用、编造列表外的任何论文、链接或「待核验」占位文献。
- 原 skill 中「每个创新点至少十篇真实论文」「参考文献总表 60 篇」等数量要求**在本模式下不适用**，不得以凑数为目的虚构文献。

**引用与论述**
- 每个创新点至少引用池中 **3 篇不同编号**的论文来支撑「大多数 / 少数 / 为此」三层论述；同一篇池内论文可在多个创新点中再次出现。
- 正文里提及文献编号时，请使用单独的 `[k]` 形式（例如 `[3]`），不要使用 `[3](url)`，以便后台自动加上站内链接。
- 「大多数现有方法」「少数现有方法」是对**当日文献池内论文主题与方法共性**的归纳：只能依据池内摘要与 AI 字段归纳。

**结构化输出（必填；实现上可能拆成两步：先 themes、再单独输出 Markdown，内容要求不变）**
- `themes`：恰好 **3** 条，每条包含 `title`（短语标题）与 `blurb`（一句话概要）。三者对应三篇论文方向，用于邮件与网页顶部摘要。
- `markdown_digest`：**完整** Markdown 正文；从「## 当日文献池编号一览」开始写起；**不要**在 markdown 里再用小节重复罗列三条 theme 标题（顶部已由结构化字段展示）。
- Markdown 要求：`##` / `###`、pipe table、`- ` 列表；不要用裸 HTML；不要用围栏包住全文。

**markdown_digest 内容顺序**
1. ## 当日文献池编号一览（[k] + 题目 + id）
2. 按 skill：三篇论文方向、每篇两个创新点（共六个）、矩阵、池内去重检查、参考文献总表（仅池内）、自检（「十篇文献」改为「每创新点≥3 篇池内文献」）。
"""

THEMES_PHASE_NOTE = """
**本轮（第一步）**
- 仅通过结构化接口返回 **`themes`（恰好 3 条）**，不要生成长正文。
"""

MARKDOWN_PHASE_NOTE = """
**本轮（第二步）**
- **直接输出 Markdown 正文**（等价于前述 `markdown_digest`），从「## 当日文献池编号一览」写起。
- **禁止**输出 JSON；不要用围栏代码块包住全文。
- 下列三条主题已固定，正文与之呼应即可，**不要用同级小节标题再罗列这三条**：
{themes_block}
"""


def _ai_message_text(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(c)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="AI enhanced jsonl file")
    parser.add_argument("--out", type=str, required=True, help="full Markdown archive (.txt)")
    parser.add_argument("--digest-dir", type=str, default="digest", help="directory for Pages digest HTML")
    parser.add_argument("--max_papers", type=int, default=30, help="sample up to this many papers into the pool")
    parser.add_argument("--max_chars_per_paper", type=int, default=1200, help="truncate each paper context block")
    parser.add_argument("--date", type=str, default="", help="YYYY-MM-DD")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (env EMAIL_DIGEST_RANDOM_SEED if unset)",
    )
    return parser.parse_args()


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if n <= 0:
        return ""
    return s if len(s) <= n else (s[: n - 3] + "...")


def strip_yaml_frontmatter(raw: str) -> str:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return raw
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return raw


def load_skill_text() -> str:
    with open(SKILL_PATH, "r", encoding="utf-8") as f:
        return strip_yaml_frontmatter(f.read()).strip()


def load_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _digest_focus_query_from_items(items: List[Dict[str, Any]]) -> str:
    for it in items:
        q = (it.get("AI") or {}).get("digest_focus_query")
        if isinstance(q, str) and q.strip():
            return q.strip()
    return ""


def select_paper_pool(items: List[Dict[str, Any]], max_papers: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    rng = random.Random(seed) if seed is not None else random.Random()
    pool = list(items)
    focus_q = _digest_focus_query_from_items(items)
    if focus_q:
        matched = [it for it in pool if (it.get("AI") or {}).get("digest_theme_relevant") is True]
        if matched:
            pool = matched
            print(
                f"email_digest: pool filtered by digest_focus ({len(pool)}/{len(items)} relevant)",
                file=sys.stderr,
            )
        else:
            print(
                "email_digest: no digest_theme_relevant items, using full pool",
                file=sys.stderr,
            )
    k = min(max_papers, len(pool))
    if k <= 0:
        return []
    if len(pool) <= max_papers:
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, k)


def build_context(papers: List[Dict[str, Any]], max_chars_per_paper: int) -> str:
    blocks: List[str] = []
    for idx, it in enumerate(papers, start=1):
        title = _safe_str(it.get("title"))
        paper_id = _safe_str(it.get("id"))
        categories = it.get("categories")
        if isinstance(categories, list):
            categories_str = ", ".join(map(_safe_str, categories))
        else:
            categories_str = _safe_str(categories)

        summary = _safe_str(it.get("summary"))
        ai = it.get("AI") if isinstance(it.get("AI"), dict) else {}
        tldr = _safe_str(ai.get("tldr"))
        motivation = _safe_str(ai.get("motivation"))
        method = _safe_str(ai.get("method"))
        result = _safe_str(ai.get("result"))
        conclusion = _safe_str(ai.get("conclusion"))

        raw_block = "\n".join(
            [
                f"[{idx}] {title}",
                f"- id: {paper_id}",
                f"- categories: {categories_str}",
                f"- abstract: {summary}",
                f"- tldr: {tldr}",
                f"- motivation: {motivation}",
                f"- method: {method}",
                f"- result: {result}",
                f"- conclusion: {conclusion}",
            ]
        )
        blocks.append(_truncate(raw_block, max_chars_per_paper))
    return "\n\n".join(blocks).strip()


def default_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def unwrap_markdown_fence(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def markdown_to_html_fragment(md: str) -> str:
    return markdown.markdown(md, extensions=["tables", "nl2br", "sane_lists", "fenced_code"])


def site_pages_base() -> str:
    return os.environ.get("PAGES_BASE_URL", "").strip().rstrip("/")


def site_home_href() -> str:
    site = site_pages_base()
    return f"{site}/index.html" if site else "../index.html"


def digest_index_href() -> str:
    """创新点列表页在 main（与 jsonl 不同：digest 正文仍在 data，由 viewer 拉 raw）。"""
    site = site_pages_base()
    return f"{site}/digest-index.html" if site else "digest-index.html"


def digest_viewer_href(date_str: str) -> str:
    """邮件与外链入口：main 上的壳页，再从 data 分支 raw 拉 HTML。"""
    site = site_pages_base()
    if not site:
        return f"digest-viewer.html?date={date_str}"
    return f"{site}/digest-viewer.html?date={date_str}"


def paper_list_href(date_str: str, quoted_paper_id: str) -> str:
    site = site_pages_base()
    tail = f"index.html?date={date_str}&paperId={quoted_paper_id}"
    return f"{site}/{tail}" if site else f"../{tail}"


def linkify_pool_refs(md: str, papers: List[Dict[str, Any]], date_str: str) -> str:
    idx_to_id = {i + 1: _safe_str(p.get("id")) for i, p in enumerate(papers)}

    def repl(m: re.Match[str]) -> str:
        inner = m.group(0)[1:-1]
        k = int(inner)
        pid = idx_to_id.get(k)
        if not pid:
            return m.group(0)
        q = urllib.parse.quote(pid, safe="")
        href = paper_list_href(date_str, q)
        return f"[{k}]({href})"

    md = normalize_glued_numeric_pool_refs(md)
    return POOL_REF_RE.sub(repl, md)


def digest_page_shell(inner_article_html: str, date_str: str, themes: List[DigestTheme], papers: List[Dict[str, Any]]) -> str:
    headline = html.escape(f"{DIGEST_PRODUCT_TITLE} · {date_str}")
    theme_cards = []
    for i, t in enumerate(themes, start=1):
        theme_cards.append(
            f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;margin-bottom:12px;">
<div style="font-size:12px;color:#64748b;margin-bottom:6px;">主题 {i}</div>
<div style="font-size:17px;font-weight:700;color:#0f172a;margin-bottom:8px;">{html.escape(t.title)}</div>
<div style="font-size:14px;color:#475569;line-height:1.55;">{html.escape(t.blurb)}</div>
</div>"""
        )
    themes_html = "\n".join(theme_cards)

    rows = []
    for i, p in enumerate(papers, start=1):
        pid = _safe_str(p.get("id"))
        tit = _truncate(_safe_str(p.get("title")), 140)
        q = urllib.parse.quote(pid, safe="") if pid else ""
        href = paper_list_href(date_str, q) if q else "#"
        rows.append(
            f"<tr><td style='padding:10px 12px;border:1px solid #e5e7eb;'>[{i}]</td>"
            f"<td style='padding:10px 12px;border:1px solid #e5e7eb;'>{html.escape(tit)}</td>"
            f"<td style='padding:10px 12px;border:1px solid #e5e7eb;font-family:monospace;font-size:13px;'>{html.escape(pid)}</td>"
            f"<td style='padding:10px 12px;border:1px solid #e5e7eb;'><a href='{html.escape(href)}' style='color:#2563eb;font-weight:600;'>在论文主页打开</a></td></tr>"
        )
    pool_table = (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;margin:16px 0;'>"
        "<thead><tr style='background:#f3f4f6;'>"
        "<th style='padding:10px;border:1px solid #e5e7eb;text-align:left;'>编号</th>"
        "<th style='padding:10px;border:1px solid #e5e7eb;text-align:left;'>题目</th>"
        "<th style='padding:10px;border:1px solid #e5e7eb;text-align:left;'>ID</th>"
        "<th style='padding:10px;border:1px solid #e5e7eb;text-align:left;'>链接</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{headline}</title>
</head>
<body style="margin:0;background:#e8eef5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',sans-serif;color:#334155;">
<div style="max-width:900px;margin:0 auto;padding:28px 16px 48px;">
<nav style="font-size:14px;margin-bottom:18px;">
<a href="{html.escape(site_home_href())}" style="color:#2563eb;text-decoration:none;font-weight:600;">论文列表主页</a>
<span style="color:#94a3b8;margin:0 8px;">/</span>
<a href="{html.escape(digest_index_href())}" style="color:#2563eb;text-decoration:none;font-weight:600;">历史创新点</a>
</nav>
<header style="background:linear-gradient(135deg,#1e40af 0%,#3b82f6 100%);color:#fff;border-radius:14px;padding:22px 26px;margin-bottom:22px;">
<p style="margin:0;font-size:12px;opacity:0.88;">读文献 · 憋创新点 · GitHub Pages</p>
<h1 style="margin:10px 0 0;font-size:22px;font-weight:700;line-height:1.35;">{headline}</h1>
</header>
<section style="margin-bottom:22px;">
<h2 style="font-size:18px;color:#0f172a;margin:0 0 14px;border-bottom:2px solid #2563eb;padding-bottom:8px;">今日三个主题</h2>
{themes_html}
</section>
<section style="margin-bottom:22px;">
<h2 style="font-size:18px;color:#0f172a;margin:0 0 14px;border-bottom:2px solid #2563eb;padding-bottom:8px;">文献池（点击跳转当日论文详情）</h2>
{pool_table}
</section>
<article class="digest-body" style="background:#fff;border-radius:12px;padding:24px 26px 32px;box-shadow:0 4px 24px rgba(15,23,42,0.06);">
<style type="text/css">
.digest-body h2 {{ font-size:18px; color:#111827; margin:26px 0 12px; padding-bottom:8px; border-bottom:2px solid #cbd5e1; font-weight:700; }}
.digest-body h3 {{ font-size:15px; color:#1f2937; margin:18px 0 10px; font-weight:600; }}
.digest-body h4 {{ font-size:14px; color:#374151; margin:14px 0 8px; font-weight:600; }}
.digest-body p {{ margin:10px 0; line-height:1.65; }}
.digest-body ul, .digest-body ol {{ margin:10px 0; padding-left:22px; }}
.digest-body li {{ margin:5px 0; }}
.digest-body table {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:13px; }}
.digest-body th, .digest-body td {{ border:1px solid #e5e7eb; padding:10px 12px; vertical-align:top; }}
.digest-body th {{ background:#f3f4f6; font-weight:600; color:#111827; }}
.digest-body hr {{ border:none; border-top:1px solid #e5e7eb; margin:22px 0; }}
.digest-body blockquote {{ margin:12px 0; padding:10px 14px; border-left:4px solid #93c5fd; background:#f8fafc; color:#475569; }}
.digest-body a {{ color:#2563eb; }}
.digest-body code {{ font-size:13px; background:#f1f5f9; padding:2px 6px; border-radius:4px; }}
.digest-body pre {{ background:#1e293b; color:#e2e8f0; padding:14px; border-radius:8px; overflow-x:auto; font-size:13px; }}
.digest-body pre code {{ background:transparent; color:inherit; padding:0; }}
</style>
{inner_article_html}
<p style="margin-top:28px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:12px;color:#94a3b8;">内容由 AI 生成，请仔细甄别。</p>
</article>
</div>
</body>
</html>"""


def write_short_notice_files(
    notice_txt_path: str,
    notice_html_path: str,
    date_str: str,
    themes: List[DigestTheme],
    page_url: str,
) -> None:
    lines = [
        f"{DIGEST_PRODUCT_TITLE} {date_str}",
        "",
        "今日三个主题概要：",
    ]
    for i, t in enumerate(themes, start=1):
        lines.append(f"{i}. {t.title} — {t.blurb}")
    lines.extend(["", "完整创新点、矩阵与文献引用见：", page_url, ""])
    with open(notice_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    bullets = "".join(
        f"<li style='margin:8px 0;'><strong>{html.escape(t.title)}</strong><br/><span style='color:#64748b;font-size:14px;'>{html.escape(t.blurb)}</span></li>"
        for t in themes
    )
    html_page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;padding:20px;background:#f1f5f9;color:#334155;">
<div style="max-width:560px;margin:0 auto;background:#fff;padding:24px;border-radius:12px;">
<p style="margin:0 0 12px;font-size:13px;color:#64748b;">{html.escape(DIGEST_PRODUCT_TITLE)} · {html.escape(date_str)}</p>
<p style="margin:0 0 16px;">今日三个主题：</p>
<ul style="padding-left:18px;margin:0;">{bullets}</ul>
<p style="margin:22px 0 12px;">点击下方按钮查看当日完整创新点与可点击文献链接：</p>
<p style="margin:0;"><a href="{html.escape(page_url)}" style="display:inline-block;background:#2563eb;color:#fff;text-decoration:none;padding:12px 22px;border-radius:8px;font-weight:600;">打开完整页面</a></p>
<p style="margin-top:18px;font-size:12px;color:#94a3b8;">若按钮无效，请复制链接：{html.escape(page_url)}</p>
</div></body></html>"""
    with open(notice_html_path, "w", encoding="utf-8") as f:
        f.write(html_page)


def regenerate_digest_index(digest_dir: str) -> None:
    dates: List[str] = []
    if os.path.isdir(digest_dir):
        for fn in os.listdir(digest_dir):
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.html", fn):
                dates.append(fn[:-5])
    dates = sorted(set(dates), reverse=True)
    manifest_path = os.path.join(digest_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"dates": dates}, f, ensure_ascii=False, indent=2)

    items = "\n".join(
        f'<li style="margin:12px 0;"><a href="{html.escape(digest_viewer_href(d))}" style="color:#2563eb;font-size:16px;font-weight:600;text-decoration:none;">{html.escape(d)}</a></li>'
        for d in dates
    )
    if not items:
        items = '<li style="color:#64748b;">暂无归档</li>'

    idx_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{html.escape(DIGEST_PRODUCT_TITLE)} · 历史</title>
</head>
<body style="margin:0;background:#e8eef5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;">
<div style="max-width:720px;margin:0 auto;padding:32px 20px;">
<nav style="margin-bottom:20px;"><a href="{html.escape(site_home_href())}" style="color:#2563eb;font-weight:600;text-decoration:none;">论文列表主页</a></nav>
<h1 style="color:#0f172a;font-size:22px;">{html.escape(DIGEST_PRODUCT_TITLE)}</h1>
<p style="color:#64748b;">按日期浏览已生成的创新点页面。</p>
<ul style="list-style:none;padding:0;margin-top:24px;">
{items}
</ul>
</div>
</body>
</html>"""
    os.makedirs(digest_dir, exist_ok=True)
    with open(os.path.join(digest_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(idx_html)


def resolve_seed(cli_seed: Optional[int]) -> Optional[int]:
    if cli_seed is not None:
        return cli_seed
    env_seed = os.environ.get("EMAIL_DIGEST_RANDOM_SEED", "").strip()
    if env_seed == "":
        return None
    try:
        return int(env_seed)
    except ValueError:
        return None


def notice_paths(out_archive_txt: str, date_str: str) -> tuple[str, str]:
    d = os.path.dirname(os.path.abspath(out_archive_txt)) or "."
    return (
        os.path.join(d, f"email_notice_{date_str}.txt"),
        os.path.join(d, f"email_notice_{date_str}.html"),
    )


def pages_digest_url(date_str: str) -> str:
    site = site_pages_base()
    if not site:
        return f"(请配置 PAGES_BASE_URL)/digest-viewer.html?date={date_str}"
    return digest_viewer_href(date_str)


def main() -> None:
    if os.path.exists(".env"):
        dotenv.load_dotenv()

    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    language = os.environ.get("LANGUAGE", "Chinese")
    date_str = args.date.strip() or default_date_str()
    seed = resolve_seed(args.seed)

    digest_dir = args.digest_dir
    os.makedirs(digest_dir, exist_ok=True)

    notice_txt_path, notice_html_path = notice_paths(args.out, date_str)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    skill_body = load_skill_text()
    system_prompt = skill_body + "\n\n" + DIGEST_CONSTRAINTS.strip()

    items = load_items(args.data)
    papers = select_paper_pool(items, args.max_papers, seed)

    if not papers:
        stub = f"{DIGEST_PRODUCT_TITLE} {date_str}\n\n（无可用论文，跳过生成。）\n"
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(stub)
        empty_url = pages_digest_url(date_str)
        ph_themes = [
            DigestTheme(title="（无数据）", blurb="当日文献池为空，未生成创新点。"),
            DigestTheme(title="（无数据）", blurb=""),
            DigestTheme(title="（无数据）", blurb=""),
        ]
        write_short_notice_files(notice_txt_path, notice_html_path, date_str, ph_themes, empty_url)
        return

    context = build_context(papers, args.max_chars_per_paper)

    human_pool_block = (
        "日期：{date}\n"
        "语言配置：{language}\n"
        "当日抓取总篇数（文件内）：{total_in_file}\n"
        "本次随机抽取进入文献池的篇数：{pool_size}（编号 [1]…[{pool_size}]）。\n\n"
        "文献池内容：\n\n"
        "{context}"
    )

    invoke_in = {
        "date": date_str,
        "language": language,
        "total_in_file": len(items),
        "pool_size": len(papers),
        "context": context if context else "（无可用论文内容）",
    }

    llm = ChatOpenAI(model=model_name, temperature=0.2)

    prompt_themes = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\n" + THEMES_PHASE_NOTE.strip()),
            (
                "human",
                human_pool_block + "\n\n请先只产出结构化字段 **`themes`（恰好 3 条）**，不要写长 Markdown 正文。",
            ),
        ]
    )
    chain_themes = prompt_themes | llm.with_structured_output(DigestThemesOnly, method="function_calling")

    themes_only: Optional[DigestThemesOnly] = None
    last_parse_exc: Optional[OutputParserException] = None
    for _attempt in range(3):
        try:
            themes_only = chain_themes.invoke(invoke_in)
            break
        except OutputParserException as e:
            last_parse_exc = e
    if themes_only is None:
        raise RuntimeError("themes 结构化输出多次解析失败") from last_parse_exc

    themes_block = "\n".join(
        f"{i}. **{t.title}** — {t.blurb}" for i, t in enumerate(themes_only.themes, start=1)
    )
    system_md = system_prompt + "\n\n" + MARKDOWN_PHASE_NOTE.replace("{themes_block}", themes_block)

    prompt_md = ChatPromptTemplate.from_messages(
        [
            ("system", system_md),
            ("human", human_pool_block + "\n\n请根据系统说明输出完整 Markdown 正文（第二步）。"),
        ]
    )
    md_resp = (prompt_md | llm).invoke(invoke_in)
    md_raw = _ai_message_text(md_resp).strip()

    structured = DigestStructured(themes=themes_only.themes, markdown_digest=md_raw)

    md_body = unwrap_markdown_fence(structured.markdown_digest.strip())
    md_linked = linkify_pool_refs(md_body, papers, date_str)

    head_md_lines = [
        f"# {DIGEST_PRODUCT_TITLE} {date_str}",
        "",
        "## 今日三个主题（结构化）",
    ]
    for i, t in enumerate(structured.themes, start=1):
        head_md_lines.append(f"{i}. **{t.title}** — {t.blurb}")
    head_md_lines.extend(["", "---", ""])
    full_md = "\n".join(head_md_lines) + md_linked

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(full_md)
        f.write("\n")

    meta_path = os.path.join(digest_dir, f"{date_str}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": date_str,
                "title": DIGEST_PRODUCT_TITLE,
                "themes": [t.model_dump() for t in structured.themes],
                "pool": [
                    {"idx": i + 1, "id": _safe_str(p.get("id")), "title": _safe_str(p.get("title"))}
                    for i, p in enumerate(papers)
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    inner_html = markdown_to_html_fragment(md_linked)
    page_html = digest_page_shell(inner_html, date_str, structured.themes, papers)
    digest_html_path = os.path.join(digest_dir, f"{date_str}.html")
    with open(digest_html_path, "w", encoding="utf-8") as f:
        f.write(page_html)

    regenerate_digest_index(digest_dir)

    page_url = pages_digest_url(date_str)
    write_short_notice_files(notice_txt_path, notice_html_path, date_str, structured.themes, page_url)


if __name__ == "__main__":
    main()
