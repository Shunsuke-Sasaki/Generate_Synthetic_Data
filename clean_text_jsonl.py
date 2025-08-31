#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jsonl から text 要素のみを抽出し、特殊文字を除去・整形して
50 文字未満の行を除外して JSONL として出力します。

Usage:
  python clean_text_jsonl.py --input in.jsonl --output out.jsonl --min-len 50
"""

import argparse, json, re, sys
from typing import Optional

CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")   # C0制御文字とDEL
WHITESPACE_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    # 1) 制御文字をスペースに置換（\n \r \t などを含む）
    s = CONTROL_CHARS_RE.sub(" ", s)
    # 2) 連続空白を1つに
    s = WHITESPACE_RE.sub(" ", s)
    # 3) 前後の空白をトリム
    return s.strip()

def process_line(line: str, min_len: int) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    txt = obj.get("text")
    if not isinstance(txt, str):
        return None
    cleaned = clean_text(txt)
    if len(cleaned) < min_len:
        return None
    return cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="入力JSONLパス")
    ap.add_argument("--output", "-o", required=True, help="出力JSONLパス（textのみ）")
    ap.add_argument("--min-len", type=int, default=50, help="最小文字数（既定: 50）")
    args = ap.parse_args()

    out_count = 0
    with open(args.input, "r", encoding="utf-8", errors="ignore") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            cleaned = process_line(line, args.min_len)
            if cleaned is None:
                continue
            fout.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
            out_count += 1

    print(f"done. kept {out_count} lines >= {args.min_len} chars.", file=sys.stderr)

if __name__ == "__main__":
    main()
