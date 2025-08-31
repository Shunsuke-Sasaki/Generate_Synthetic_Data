#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONLの各行で 'text' が max_len を超える場合に適切に分割して出力。

- 文境界（。．！？!?）での分割を優先
- 1文が max_len を超える場合は安全に文字数でハードラップ
- 既存フィールドは保持し、chunk_index / chunk_total を追加
- text が存在しない、文字列でない行はそのままスキップ（警告のみ）

Usage:
  python split_text_jsonl.py --input in.jsonl --output out.jsonl --max-len 4096
"""

import argparse
import json
import re
import sys
from typing import List, Dict, Any

# 文の終端とみなす文字（必要なら追加してください）
SENT_END_PATTERN = re.compile(r'(?<=[。．！？!?])')

def split_long_sentence(sentence: str, max_len: int) -> List[str]:
    """1文が長すぎる場合、max_len でハードラップ。"""
    return [sentence[i:i+max_len] for i in range(0, len(sentence), max_len)]

def sentence_split(text: str) -> List[str]:
    """
    シンプルな文分割。終端記号で区切るが、終端後の空文字は除去。
    改行はそのまま残す（必要に応じて前処理で潰してください）。
    """
    # まずはそのまま分割（終端記号を保持）
    parts = SENT_END_PATTERN.split(text)
    # re.split で区切り後に残る末尾空要素を除外
    parts = [p for p in parts if p]
    # 文末記号がない長塊対策：全体を1要素とするケースはそのまま返す
    return parts if parts else [text]

def chunk_by_max_len(text: str, max_len: int) -> List[str]:
    """
    文を積み上げて max_len を超えないようチャンク化。
    1文が max_len を超えるときはハードラップ。
    """
    sentences = sentence_split(text)
    chunks: List[str] = []
    buf = []

    current_len = 0
    for s in sentences:
        if len(s) > max_len:
            # まず今のバッファを確定
            if buf:
                chunks.append(''.join(buf))
                buf, current_len = [], 0
            # 超長文は分割して追加
            hard = split_long_sentence(s, max_len)
            # 末尾以外は確定チャンクとして出力
            chunks.extend(hard[:-1])
            # 最後の断片は次の文と一緒に詰める
            if hard:
                buf = [hard[-1]]
                current_len = len(hard[-1])
        else:
            if current_len + len(s) <= max_len:
                buf.append(s)
                current_len += len(s)
            else:
                # ここで一旦確定
                if buf:
                    chunks.append(''.join(buf))
                buf = [s]
                current_len = len(s)

    if buf:
        chunks.append(''.join(buf))
    return chunks

def process_record(obj: Dict[str, Any], max_len: int) -> List[Dict[str, Any]]:
    """
    1レコードを必要に応じて分割し、複数レコードにして返す。
    元のフィールドは保持し、chunk_index / chunk_total を付与。
    """
    txt = obj.get('text')
    if not isinstance(txt, str):
        return []  # スキップ

    if len(txt) <= max_len:
        out = dict(obj)
        out['chunk_index'] = 0
        out['chunk_total'] = 1
        return [out]

    chunks = chunk_by_max_len(txt, max_len)
    total = len(chunks)
    out_records = []
    for i, ch in enumerate(chunks):
        rec = dict(obj)
        rec['text'] = ch
        rec['chunk_index'] = i
        rec['chunk_total'] = total
        out_records.append(rec)
    return out_records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', required=True, help='入力 JSONL パス')
    ap.add_argument('--output', '-o', required=True, help='出力 JSONL パス')
    ap.add_argument('--max-len', type=int, default=4096, help='チャンク最大文字数（既定: 4096）')
    args = ap.parse_args()

    in_path, out_path = args.input, args.output
    max_len = args.max_len

    total_in = 0
    total_out = 0
    skipped = 0

    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            out_records = process_record(obj, max_len)
            if not out_records:
                skipped += 1
                continue

            for r in out_records:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
                total_out += 1

    print(
        f'done. input_lines={total_in}, output_lines={total_out}, skipped={skipped}, max_len={max_len}',
        file=sys.stderr
    )

if __name__ == '__main__':
    main()
