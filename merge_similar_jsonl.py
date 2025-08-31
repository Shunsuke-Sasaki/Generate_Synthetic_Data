#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_similar_jsonl.py

入力 JSONL（各行: {"uid": "...", "raw": "..."}）を読み、
埋め込みのコサイン類似度で似ているテキストをグルーピングして連結。
各グループを 1 行の JSON として出力（JSONL）。
連結テキストは最大 max_chars（既定 20000）を超えた分は切り捨て。

グルーピングは「クラスタの重心と類似度 >= 閾値」で割り当てる貪欲法（高速・省メモリ）。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

# sentence-transformers（多言語E5）
from sentence_transformers import SentenceTransformer

def read_jsonl(path: Path) -> List[Tuple[str, str]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("uid")
            raw = obj.get("raw")
            if isinstance(uid, str) and isinstance(raw, str) and raw.strip():
                items.append((uid, raw.strip()))
    return items

def write_jsonl(path: Path, records: List[dict]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def embed_texts(model_name: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    E5 系は 'passage: ' を先頭に付ける運用が推奨。
    normalize_embeddings=True で L2 正規化 → cos類似度=内積
    """
    model = SentenceTransformer(model_name)
    prefixed = [f"passage: {t}" for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(embs, dtype=np.float32)

def greedy_cluster(
    embeddings: np.ndarray,
    uids: List[str],
    texts: List[str],
    threshold: float
):
    """
    貪欲クラスタリング：
    - 既存クラスタの重心（平均ベクトル）との cos 類似度が閾値以上ならそのクラスタに割り当て
    - どれも満たさなければ新規クラスタを作成
    計算量 ~ O(N*K)（Kはクラスタ数）で大規模でも現実的。
    """
    clusters = []            # 各クラスタのメンバー index のリスト
    centroid_sums = []       # 各クラスタのベクトル和（正規化前） shape=(d,)
    counts = []              # 各クラスタのメンバー数

    for i in tqdm(range(len(texts)), desc="Clustering"):
        v = embeddings[i]
        if not clusters:
            clusters.append([i])
            centroid_sums.append(v.astype(np.float32).copy())
            counts.append(1)
            continue

        # 重心の内積 = cos 類似度（embeddingsは正規化済み、重心は都度正規化して比較）
        # 速度のため、重心ベクトルは sum/count → 正規化してまとめて計算
        centroids = np.stack([(s / c) for s, c in zip(centroid_sums, counts)], axis=0)
        # 正規化
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        sims = centroids @ v  # (K, d) · (d,) -> (K,)

        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            clusters[best].append(i)
            centroid_sums[best] += v
            counts[best] += 1
        else:
            clusters.append([i])
            centroid_sums.append(v.astype(np.float32).copy())
            counts.append(1)

    return clusters

def build_records(
    clusters: List[List[int]],
    uids: List[str],
    texts: List[str],
    max_chars: int,
    group_uid_strategy: str = "first"  # or "concat"
) -> List[dict]:
    """
    各クラスタを 1 レコード化。
    - raw はメンバーの raw を「\n\n」で連結し、max_chars を超えた分は切り捨て
    - uid は 'first' ならクラスタ先頭の uid、'concat' なら uid を結合して短縮
    """
    out = []
    for idxs in clusters:
        parts = [texts[i] for i in idxs]
        joined = "\n\n".join(parts)
        if len(joined) > max_chars:
            joined = joined[:max_chars]

        if group_uid_strategy == "concat":
            # 代表 uid を短縮生成（先頭と件数を混ぜる等・用途に応じて調整可）
            rep_uid = f"{uids[idxs[0]]}-grp{len(idxs)}"
        else:
            rep_uid = uids[idxs[0]]

        out.append({"uid": rep_uid, "raw": joined})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="入力 JSONL（各行 {uid, raw}）")
    ap.add_argument("--output", required=True, type=Path, help="出力 JSONL（各行 {uid, raw}）")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base", help="埋め込みモデル名")
    ap.add_argument("--threshold", type=float, default=0.82, help="クラスタ割当の類似度下限（cos）")
    ap.add_argument("--batch-size", type=int, default=64, help="埋め込み計算のバッチサイズ")
    ap.add_argument("--max-chars", type=int, default=20000, help="連結後テキストの最大文字数（超過切り捨て）")
    ap.add_argument("--min-chars", type=int, default=1, help="短すぎる raw を除外するしきい値（文字数）")
    ap.add_argument("--uid-strategy", choices=["first", "concat"], default="first",
                    help="グループの代表 uid の付け方（first=先頭 / concat=先頭+件数）")
    args = ap.parse_args()

    items = read_jsonl(args.input)
    if not items:
        print("no valid records", file=sys.stderr)
        sys.exit(1)

    # 短文を除外（例: 壊れた 'Wh' のような行）
    items = [(u, t) for (u, t) in items if len(t) >= args.min_chars]
    uids = [u for u, _ in items]
    texts = [t for _, t in items]

    # 埋め込み
    embs = embed_texts(args.model, texts, batch_size=args.batch_size)

    # 貪欲クラスタリング
    clusters = greedy_cluster(embs, uids, texts, threshold=args.threshold)

    # 出力レコード作成（max_chars で切り捨て）
    records = build_records(clusters, uids, texts, max_chars=args.max_chars,
                            group_uid_strategy=args.uid_strategy)

    # 書き出し（JSONL: 1 行=1 レコード）
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)

    # 簡単なレポート
    sizes = [len(c) for c in clusters]
    print(f"[done] input={len(items)}, groups={len(clusters)}, "
          f"avg_group_size={np.mean(sizes):.2f}, threshold={args.threshold}")

if __name__ == "__main__":
    main()
