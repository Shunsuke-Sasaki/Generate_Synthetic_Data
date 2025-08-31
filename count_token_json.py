#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_tokens.py

Usage:
    python count_tokens.py [--tokenizer NAME_OR_PATH]
                           [--char-limit N]
                           [--batch-size N]
                           [--show-per-file]
                           PATH [PATH2 ...]

Examples:
    # ディレクトリを指定（再帰的に .json / .jsonl を収集）
    python count_tokens.py --char-limit 100000 --show-per-file data/json_dir

Options:
    --tokenizer NAME_OR_PATH   Hugging Face 形式のトークナイザ名またはローカルパス
                               省略時は 'cl-tohoku/bert-base-japanese-v2' を使用
    --char-limit N             トークンをカウントする各アイテムの最大文字数
                               （超えたものはトークン計算せず、文字数をstderrに出力）
    --batch-size N             一度にまとめてエンコードする件数 [default: 1024]
    --show-per-file            各ファイルごとのトークン総数も表示
"""
import argparse
import json
import pathlib
import sys
from typing import Iterable, List, Union, Any, Generator

from transformers import AutoTokenizer

# ----------------------------------------------------------------------
def iter_strings(obj: Any) -> Generator[str, None, None]:
    """任意のJSONオブジェクトから全ての文字列(str)を再帰的にyield"""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for v in obj:
            yield from iter_strings(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    # それ以外（数値・None等）は無視


def stream_texts_json(fp: pathlib.Path) -> Iterable[str]:
    """拡張子 .json をロードし、含まれる全ての文字列を再帰抽出して返す"""
    try:
        with fp.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON decode error: {fp}: {e}", file=sys.stderr)
        return
    except OSError as e:
        print(f"[WARN] File open error: {fp}: {e}", file=sys.stderr)
        return

    for s in iter_strings(data):
        # 念のため文字列化（上でstr限定だが型安全のため）
        yield str(s)


def stream_texts_jsonl(fp: pathlib.Path) -> Iterable[str]:
    """拡張子 .jsonl を1行ずつ読み、'text' フィールドがあれば返す"""
    with fp.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # フォーマット壊れ行を無視
                continue
            if isinstance(obj, dict) and "text" in obj:
                yield str(obj["text"])


def stream_texts(fp: pathlib.Path) -> Iterable[str]:
    """拡張子に応じてJSON/JSONLどちらも扱う"""
    suffix = fp.suffix.lower()
    if suffix == ".json":
        yield from stream_texts_json(fp)
    elif suffix == ".jsonl":
        yield from stream_texts_jsonl(fp)
    else:
        # 拡張子不明はJSONL互換として試みる（必要に応じて調整）
        yield from stream_texts_jsonl(fp)


def count_tokens(tokenizer, texts: List[str]) -> int:
    """リストの文章をまとめてエンコードしてトークン長を返す"""
    if not texts:
        return 0
    enc = tokenizer(texts, add_special_tokens=False)
    return sum(len(ids) for ids in enc["input_ids"])


def collect_input_files(paths: List[pathlib.Path]) -> List[pathlib.Path]:
    """与えられたパス群から、.json / .jsonl ファイル一覧を再帰的に収集"""
    files: List[pathlib.Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
            files.extend(sorted(p.rglob("*.json")))
        else:
            # 単一ファイル指定時は拡張子不問で受ける
            files.append(p)

    # 重複除去（順序維持）
    deduped = list(dict.fromkeys(files))
    return deduped

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="json/jsonl のテキスト・トークンカウンタ")
    parser.add_argument("paths", nargs="+", type=pathlib.Path,
                        help="ファイル/ディレクトリ を1つ以上。ディレクトリは再帰探索します。")
    parser.add_argument("--tokenizer", default="cl-tohoku/bert-base-japanese-v2")
    parser.add_argument("--char-limit", type=int,
                        help="トークンカウント対象の各アイテムの最大文字数")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--show-per-file", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    total_tokens = 0

    files = collect_input_files(args.paths)
    if not files:
        print("[ERROR] 対象ファイルが見つかりませんでした (.json / .jsonl)", file=sys.stderr)
        return 2

    for fp in files:
        file_tokens = 0
        batch: List[str] = []

        # アイテム番号（jsonlの行、jsonの抽出順）
        for idx, txt in enumerate(stream_texts(fp), start=1):
            if args.char_limit is not None and len(txt) > args.char_limit:
                print(f"[OVERSIZE] {fp}:{idx}: {len(txt)} chars", file=sys.stderr)
                continue

            batch.append(txt)
            if len(batch) >= args.batch_size:
                file_tokens += count_tokens(tokenizer, batch)
                batch.clear()

        # 残りのバッチを処理
        if batch:
            file_tokens += count_tokens(tokenizer, batch)

        total_tokens += file_tokens
        if args.show_per_file:
            print(f"{fp}: {file_tokens:,} tokens")

    print(f"TOTAL: {total_tokens:,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
