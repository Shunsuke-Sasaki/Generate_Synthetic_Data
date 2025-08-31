#!/usr/bin/env python3
# split_long_jsonl.py
import argparse, json, pathlib, sys

MAX_CHARS_DEFAULT = 100_000

# ------------------------------------------------------------
def split_text(text: str, max_chars: int):
    """text を改行単位でまとめ直し、max_chars 以下のチャンクに分割"""
    parts, buf, length = [], [], 0
    for line in text.split("\n"):
        # 1 行が丸ごと長すぎる場合の安全策
        while len(line) > max_chars:
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if not line:
            continue

        # +1 は改行を再結合するときの分を先に加算
        added = len(line) + (1 if buf else 0)
        if length + added > max_chars:
            parts.append("\n".join(buf))
            buf, length = [], 0
        buf.append(line)
        length += added
    if buf:
        parts.append("\n".join(buf))
    return parts

# ------------------------------------------------------------
def process_file(src: pathlib.Path, dst: pathlib.Path, max_chars: int):
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode error: {src}:{lineno}", file=sys.stderr)
                continue

            text = str(obj.get("text", ""))
            if len(text) <= max_chars:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # 長文を分割
            chunks = split_text(text, max_chars)
            total = len(chunks)
            for idx, chunk in enumerate(chunks, 1):
                new_obj = obj.copy()
                new_obj["text"] = chunk
                # 任意でメタ情報を付けたい場合は以下を有効化
                # new_obj["_split_idx"] = idx
                # new_obj["_split_total"] = total
                fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="jsonl の 'text' が 10 万文字を超える行を改行単位で分割"
    )
    ap.add_argument("files", nargs="+", type=pathlib.Path)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS_DEFAULT,
                    help="1 チャンクの最大文字数 (デフォルト: 100000)")
    ap.add_argument("--output-dir", type=pathlib.Path,
                    help="出力先ディレクトリ (指定が無ければ <name>_split.jsonl)")
    args = ap.parse_args()

    for src in args.files:
        if not src.is_file():
            print(f"[ERR] {src} が見つかりません", file=sys.stderr)
            continue

        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            dst = args.output_dir / src.name
        else:
            dst = src.with_name(f"{src.stem}_split{src.suffix}")

        process_file(src, dst, args.max_chars)
        print(f"[OK] written → {dst}", file=sys.stderr)

if __name__ == "__main__":
    main()
