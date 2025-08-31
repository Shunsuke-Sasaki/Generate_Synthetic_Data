#!/usr/bin/env python3
"""
count_tokens.py

Usage:
    python count_tokens.py [--tokenizer NAME_OR_PATH]
                           [--char-limit N]
                           [--batch-size N]
                           [--show-per-file]
                           file1.jsonl file2.jsonl ...

Options:
    --tokenizer NAME_OR_PATH   Hugging Face 形式のトークナイザ名またはローカルパス
                               省略時は 'cl-tohoku/bert-base-japanese-v2' を使用
    --char-limit N             トークンをカウントする行の最大文字数
                               （超えた行はトークン計算せず、文字数を出力）
    --batch-size N             一度にまとめてエンコードする行数 [default: 1024]
    --show-per-file            各ファイルごとのトークン総数も表示
"""
import argparse, json, pathlib, sys
from typing import Iterable, List

from transformers import AutoTokenizer
from tqdm import tqdm

# ----------------------------------------------------------------------
def stream_texts(fname: pathlib.Path) -> Iterable[str]:
    """jsonl ファイルから 'text' フィールドを順に返すジェネレータ"""
    with fname.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "text" in obj:
                    yield str(obj["text"])
            except json.JSONDecodeError:
                # フォーマット壊れ行を無視
                continue

def count_tokens(tokenizer, texts: List[str]) -> int:
    """リストの文章をまとめてエンコードしてトークン長を返す"""
    enc = tokenizer(texts, add_special_tokens=False)
    return sum(len(ids) for ids in enc["input_ids"])

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="jsonl の text トークンカウンタ")
    parser.add_argument("files", nargs="+", type=pathlib.Path)
    parser.add_argument("--tokenizer", default="cl-tohoku/bert-base-japanese-v2")
    parser.add_argument("--char-limit", type=int,
                        help="トークンカウント対象の最大文字数")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--show-per-file", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    total_tokens = 0

    for fp in args.files:
        file_tokens = 0
        batch: List[str] = []
        # 行番号カウント用
        for idx, txt in enumerate(stream_texts(fp), start=1):
            # 文字数上限チェック
            if args.char_limit is not None and len(txt) > args.char_limit:
                # 超過した行の文字数を出力
                print(f"[OVERSIZE] {fp}:{idx}: {len(txt)} chars", file=sys.stderr)
                continue
            # バッチに追加
            batch.append(txt)
            if len(batch) >= args.batch_size:
                file_tokens += count_tokens(tokenizer, batch)
                batch.clear()

        # 残りバッチの処理
        if batch:
            file_tokens += count_tokens(tokenizer, batch)

        total_tokens += file_tokens
        print(f"{fp}: {file_tokens:,} tokens")

    print(f"TOTAL: {total_tokens:,} tokens")

if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())
