import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="AI enhanced jsonl file")
    parser.add_argument("--out", type=str, required=True, help="output text file path")
    parser.add_argument("--max_papers", type=int, default=30, help="max number of papers to include as context")
    parser.add_argument("--max_chars_per_paper", type=int, default=1200, help="truncate each paper context")
    parser.add_argument("--date", type=str, default="", help="date string like YYYY-MM-DD for email title/body")
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


def build_context(items: List[Dict[str, Any]], max_papers: int, max_chars_per_paper: int) -> str:
    blocks: List[str] = []
    for idx, it in enumerate(items[:max_papers], start=1):
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


def main() -> None:
    if os.path.exists(".env"):
        dotenv.load_dotenv()

    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")
    language = os.environ.get("LANGUAGE", "Chinese")
    date_str = args.date.strip() or default_date_str()

    items = load_items(args.data)
    context = build_context(items, args.max_papers, args.max_chars_per_paper)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是资深研究员助理。你将基于给定的一组论文内容信息，写一封可以直接发送到邮箱的每日简报。"
                "要求：中文、简洁、信息密度高、不要杜撰。"
                "输出结构固定为：\n"
                "1) 今日概览（3-6条要点）\n"
                "2) 主题分组（按主题列出，每组2-5条要点）\n"
                "3) 值得关注的论文（最多10篇，每篇1-2行：题目 + 关键贡献/结论）\n"
                "4) 可能的下一步（2-4条）\n"
                "禁止：空话套话、夸张措辞、编造实验结果。",
            ),
            (
                "human",
                "日期：{date}\n"
                "语言配置：{language}\n"
                "论文数量：{count}\n\n"
                "以下是论文内容信息（包含摘要与AI增强字段）。请严格基于这些信息生成简报：\n\n"
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
            "count": len(items),
            "context": context if context else "（无可用论文内容）",
        }
    )

    body = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    with open(args.out, "w") as f:
        f.write(f"每日 arXiv 简报 {date_str}\n\n")
        f.write(body)
        f.write("\n")


if __name__ == "__main__":
    main()

