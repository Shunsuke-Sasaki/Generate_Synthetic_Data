#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm

# 句点/文区切りの簡易パターン（日本語/英語混在を想定）
SENT_SPLIT_RE = re.compile(r'(?<=[。．.!?？!])\s+')
# 段落区切り（空行）
PARA_SPLIT_RE = re.compile(r'\n{2,}')
# 余分な空白整理
WS_RE = re.compile(r'[ \t\u3000]+')

# バックスラッシュで始まるエスケープ（例: \n, \r, \t, \xNN, \uFFFF, \N{...}, 8進）を削除
# ※ 実際の改行/タブ等の制御文字は削除しません（\\n 等の「文字列としてのシーケンス」を削除）
ESCAPE_PATTERNS = [
    re.compile(r'\\[abfnrtv]'),              # \a \b \f \n \r \t \v など
    re.compile(r'\\x[0-9A-Fa-f]{2}'),        # \xNN
    re.compile(r'\\u[0-9A-Fa-f]{4}'),        # \uFFFF
    re.compile(r'\\U[0-9A-Fa-f]{8}'),        # \UFFFFFFFF
    re.compile(r'\\N\{[^}]*\}'),             # \N{NAME}
    re.compile(r'\\[0-7]{1,3}'),             # 8進 \0 \07 \123
]

def strip_backslash_escapes(s: str) -> str:
    if not s:
        return s
    out = s
    for pat in ESCAPE_PATTERNS:
        out = pat.sub('', out)
    return out

def stable_uid(url: str, title: str | None) -> str:
    base = url or (title or "")
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:10]
    domain = ""
    try:
        domain = urlparse(url).netloc.split(":")[0]
        domain = domain.replace(".", "-")
    except Exception:
        pass
    return f"{domain}-{h}" if domain else h

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # 「文字としての」エスケープシーケンスを削除（例: \n -> "", \u3000 -> ""）
    s = strip_backslash_escapes(s)

    # 実際の制御文字の正規化（改行は維持）
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # タブや全角空白などを単一空白へ縮約
    s = WS_RE.sub(" ", s)

    # 行頭・行末の空白整形
    s = "\n".join(line.strip() for line in s.split("\n"))

    # 連続空行を最大2つに圧縮
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def chunk_text(body: str, max_chars: int) -> list[str]:
    """段落→文の順でできるだけ自然な境界で ~max_chars に収まるよう分割"""
    if len(body) <= max_chars:
        return [body]

    paras = [p.strip() for p in PARA_SPLIT_RE.split(body) if p.strip()]
    chunks, buf = [], ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in paras:
        if len(para) > max_chars:
            # 段落が大きすぎる場合は文単位で詰める
            sentences = [s.strip() for s in SENT_SPLIT_RE.split(para) if s.strip()]
            for sent in sentences:
                if not buf:
                    buf = sent
                elif len(buf) + 1 + len(sent) <= max_chars:
                    buf += " " + sent
                else:
                    flush()
                    buf = sent
            flush()
        else:
            if not buf:
                buf = para
            elif len(buf) + 2 + len(para) <= max_chars:
                buf += "\n\n" + para
            else:
                flush()
                buf = para
    flush()
    return chunks

def run(input_path: Path,
        output_path: Path,
        max_chars: int,
        min_chars: int,
        include_title_in_raw: bool,
        include_meta: bool):
    n_in, n_out, n_skip = 0, 0, 0
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Converting"):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj = json.loads(line)
            except Exception:
                n_skip += 1
                continue

            url = obj.get("url") or ""
            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()

            # 正規化（ここで \\n のような“エスケープ表現”は削除されます）
            title_norm = normalize_text(title)
            text_norm = normalize_text(text)

            if not text_norm or len(text_norm) < min_chars:
                n_skip += 1
                continue

            # raw を組み立て
            if include_title_in_raw and title_norm:
                raw_full = f"{title_norm}\n\n{text_norm}"
            else:
                raw_full = text_norm

            uid_base = stable_uid(url, title_norm)
            parts = chunk_text(raw_full, max_chars=max_chars)

            for pi, part in enumerate(parts, start=1):
                uid = uid_base if len(parts) == 1 else f"{uid_base}-p{pi}"
                rec = {"uid": uid, "raw": part}
                if include_meta:
                    rec["meta"] = {
                        "url": url,
                        "title": title_norm,
                        "mime": obj.get("mime"),
                        "scraped_at": obj.get("scraped_at"),
                        "source": urlparse(url).netloc if url else None,
                        "part_index": pi,
                        "part_count": len(parts),
                    }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"[done] input lines={n_in}, written records={n_out}, skipped={n_skip}")
    print(f"[hint] Output saved to: {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="元の scraped JSONL")
    ap.add_argument("--output", required=True, type=Path, help="EntiGraph 用 source JSONL の出力先（例: data/dataset/source/mycorpus.jsonl）")
    ap.add_argument("--max-chars", type=int, default=20000, help="1レコードあたりの最大文字数（長文は分割）")
    ap.add_argument("--min-chars", type=int, default=100, help="短すぎる本文を除外する閾値（文字）")
    ap.add_argument("--no-title", action="store_true", help="raw の先頭にタイトルを入れない")
    ap.add_argument("--include-meta", action="store_true", help="meta 情報を rec に含める（EntiGraph 自体は uid/raw しか使いません）")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(
        input_path=args.input,
        output_path=args.output,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        include_title_in_raw=not args.no_title,
        include_meta=args.include_meta,
    )

if __name__ == "__main__":
    main()
