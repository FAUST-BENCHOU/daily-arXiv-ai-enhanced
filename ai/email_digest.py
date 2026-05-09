import argparse
import html
import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import dotenv
import markdown
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

SKILL_PATH = os.path.join(os.path.dirname(__file__), "skill.txt")

DIGEST_CONSTRAINTS = """
---
# 每日邮件模式（覆盖 skill 中与文献数量冲突的条款）

你正在生成的是「当日 arXiv 抓取论文」的邮件稿，不是让用户再去检索外部文献。

**文献池（唯一合法引用来源）**
- 用户消息中会给出若干篇论文，编号为 [1]、[2]、…、[N]（N≤30）。
- 你只能引用这些编号对应的论文（可用编号或 arXiv id 指代）。**禁止**引用、编造列表外的任何论文、链接或「待核验」占位文献。
- 原 skill 中「每个创新点至少十篇真实论文」「参考文献总表 60 篇」等数量要求**在本模式下不适用**，不得以凑数为目的虚构文献。

**引用与论述**
- 每个创新点至少引用池中 **3 篇不同编号**的论文来支撑「大多数 / 少数 / 为此」三层论述；同一篇池内论文可在多个创新点中再次出现。
- 「大多数现有方法」「少数现有方法」是对**当日文献池内论文主题与方法共性**的归纳：只能依据池内摘要与 AI 字段归纳，不得捏造池外综述结论。
- 若证据不足以支撑某一论断，须显著弱化或写明「依当日文献难以区分主次流派」，**禁止**编造实验结果或虚构论文。

**输出格式：必须使用 Markdown（不要使用裸 HTML 标签）**
- 一级章节用 `##`，次级用 `###`；段落之间空一行。
- 列表统一用 `- `（多级列表用缩进）。
- 去重矩阵、文献表等**必须使用 Markdown pipe table**（含表头分隔行 `|---|`）。
- 不要使用整块代码围栏包裹全文；表格单元格内避免未转义的 `|`。
- 数学公式尽量少用；必须用则用 `$...$` 简述，避免复杂 LaTeX。

**输出顺序（便于邮件阅读）**
1. 一行「邮件标题建议：……」（≤80 字）
2. 「当日文献池编号一览」：列出 [k] 题目（过长可截断）+ arXiv id（如有）
3. 随后严格按 skill 正文要求的结构输出：**三篇论文方向、每篇两个创新点（共六个创新点）**、去重矩阵、池内去重检查、参考文献总表（条目只能来自文献池，格式：[k] 题名，arXiv:id）、自检结果（自检项中关于「十篇文献」改为「每创新点≥3 篇池内文献」）。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="AI enhanced jsonl file")
    parser.add_argument("--out", type=str, required=True, help="output Markdown (.txt) path")
    parser.add_argument(
        "--html-out",
        type=str,
        default="",
        help="output HTML email path (default: same basename as --out with .html)",
    )
    parser.add_argument("--max_papers", type=int, default=30, help="sample up to this many papers into the pool")
    parser.add_argument("--max_chars_per_paper", type=int, default=1200, help="truncate each paper context block")
    parser.add_argument("--date", type=str, default="", help="date string like YYYY-MM-DD for email title/body")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling (default: env EMAIL_DIGEST_RANDOM_SEED or non-deterministic)",
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


def select_paper_pool(items: List[Dict[str, Any]], max_papers: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    rng = random.Random(seed) if seed is not None else random.Random()
    pool = list(items)
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
    return markdown.markdown(
        md,
        extensions=["tables", "nl2br", "sane_lists", "fenced_code"],
    )


def build_email_html(markdown_body: str, date_str: str) -> str:
    inner = markdown_to_html_fragment(markdown_body)
    headline = f"每日 arXiv 简报 · {date_str}"
    title_esc = html.escape(f"每日 arXiv 简报 {date_str}")
    # Double braces in CSS for .format
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_esc}</title>
</head>
<body style="margin:0;padding:0;background-color:#e8eef5;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color:#e8eef5;">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="max-width:680px;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 24px rgba(15,23,42,0.08);">
<tr><td style="padding:14px 24px;background:linear-gradient(135deg,#1e40af 0%,#3b82f6 100%);color:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',sans-serif;">
<p style="margin:0;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;opacity:0.88;">Daily arXiv Digest</p>
<h1 style="margin:8px 0 0;font-size:20px;font-weight:600;line-height:1.35;">{html.escape(headline)}</h1>
</td></tr>
<tr><td class="digest-body" style="padding:24px 28px 32px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',sans-serif;font-size:15px;line-height:1.65;color:#374151;">
<style type="text/css">
.digest-body h2 {{ font-size:18px; color:#111827; margin:28px 0 12px; padding-bottom:8px; border-bottom:2px solid #2563eb; font-weight:600; }}
.digest-body h3 {{ font-size:15px; color:#1f2937; margin:20px 0 10px; font-weight:600; }}
.digest-body h4 {{ font-size:14px; color:#374151; margin:14px 0 8px; font-weight:600; }}
.digest-body p {{ margin:10px 0; }}
.digest-body ul, .digest-body ol {{ margin:10px 0; padding-left:22px; }}
.digest-body li {{ margin:5px 0; }}
.digest-body table {{ border-collapse:collapse; width:100%; margin:16px 0; font-size:13px; }}
.digest-body th, .digest-body td {{ border:1px solid #e5e7eb; padding:10px 12px; vertical-align:top; }}
.digest-body th {{ background:#f3f4f6; font-weight:600; color:#111827; }}
.digest-body hr {{ border:none; border-top:1px solid #e5e7eb; margin:24px 0; }}
.digest-body blockquote {{ margin:12px 0; padding:10px 14px; border-left:4px solid #93c5fd; background:#f8fafc; color:#475569; }}
.digest-body code {{ font-size:13px; background:#f1f5f9; padding:2px 6px; border-radius:4px; }}
.digest-body pre {{ background:#1e293b; color:#e2e8f0; padding:14px; border-radius:8px; overflow-x:auto; font-size:13px; }}
.digest-body pre code {{ background:transparent; color:inherit; padding:0; }}
</style>
<div class="digest-content">
{inner}
</div>
<p style="margin-top:28px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:12px;color:#9ca3af;line-height:1.5;">内容由 AI 生成，请仔细甄别。</p>
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""


def build_empty_email_html(date_str: str, message: str) -> str:
    return build_email_html(f"## 提示\n\n{message}", date_str)


def html_output_path(out_txt: str, html_out: str) -> str:
    ho = (html_out or "").strip()
    if ho:
        return ho
    if out_txt.endswith(".txt"):
        return out_txt[:-4] + ".html"
    return out_txt + ".html"


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


def main() -> None:
    if os.path.exists(".env"):
        dotenv.load_dotenv()

    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    language = os.environ.get("LANGUAGE", "Chinese")
    date_str = args.date.strip() or default_date_str()
    seed = resolve_seed(args.seed)

    skill_body = load_skill_text()
    system_prompt = skill_body + "\n\n" + DIGEST_CONSTRAINTS.strip()

    items = load_items(args.data)
    html_path = html_output_path(args.out, args.html_out)

    papers = select_paper_pool(items, args.max_papers, seed)
    if not papers:
        stub = f"每日 arXiv 简报 {date_str}\n\n（无可用论文，跳过生成。）\n"
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(stub)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(build_empty_email_html(date_str, "（无可用论文，跳过生成。）"))
        return

    context = build_context(papers, args.max_chars_per_paper)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "日期：{date}\n"
                "语言配置：{language}\n"
                "当日抓取总篇数（文件内）：{total_in_file}\n"
                "本次随机抽取进入文献池的篇数：{pool_size}（编号 [1]…[{pool_size}]，三篇论文构想及六个创新点必须仅基于这些文献）\n\n"
                "研究方向请从下列文献池整体主题中自行提炼，无需用户另行提供课题名称。\n\n"
                "文献池内容（摘要 abstract 与 AI 增强字段）：\n\n"
                "{context}",
            ),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0.2)
    chain = prompt | llm
    resp = chain.invoke(
        {
            "date": date_str,
            "language": language,
            "total_in_file": len(items),
            "pool_size": len(papers),
            "context": context if context else "（无可用论文内容）",
        }
    )

    body_raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    body_md = unwrap_markdown_fence(body_raw)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"每日 arXiv 简报 {date_str}\n\n")
        f.write(body_md)
        f.write("\n")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_email_html(body_md, date_str))


if __name__ == "__main__":
    main()
