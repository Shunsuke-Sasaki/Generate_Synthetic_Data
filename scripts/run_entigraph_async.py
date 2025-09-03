#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, asyncio, json, sys, os, re
from pathlib import Path
from typing import List, Optional, Iterable, Tuple

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT   = SCRIPT_PATH.parents[1]          # .../Generate_Synthetic_Data
PROJECT_ROOT= REPO_ROOT.parent                # .../ax-cpt
ENTIGRAPH_PY= REPO_ROOT / "data" / "entigraph.py"  # .../Generate_Synthetic_Data/data/entigraph.py

# ------------------------------ helpers ------------------------------
def load_uids_from_jsonl(path: Path) -> List[str]:
    uids: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                uids.append("")
                continue
            try:
                obj = json.loads(line)
            except Exception:
                uids.append("")
                continue
            uids.append(str(obj.get("uid", "")))
    return uids

def parse_index_list(text: str) -> List[int]:
    """
    "1,2,5-9,20" -> [1,2,5,6,7,8,9,20]
    """
    s = text.strip()
    idxs: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b:
                idxs.extend(range(a, b+1))
            else:
                idxs.extend(range(b, a+1))
        else:
            idxs.append(int(part))
    # 重複除去＆昇順
    return sorted(set(idxs))

def resolve_under_root(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (PROJECT_ROOT / q)

# ------------------------------ runner ------------------------------
async def run_one(index: int, model_tag: str, timeout_sec: Optional[int], debug_fg: bool, uid: str = ""):
    if not ENTIGRAPH_PY.exists():
        raise FileNotFoundError(f"entigraph.py not found: {ENTIGRAPH_PY}")

    cmd = [sys.executable, str(ENTIGRAPH_PY), str(index), "--model-name", model_tag]
    env = os.environ.copy()

    if debug_fg:
        print(f"[debug] exec (fg): {cmd}  cwd={PROJECT_ROOT}")
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(PROJECT_ROOT))
        try:
            await asyncio.wait_for(proc.wait(), timeout=None if not timeout_sec else timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"[index={index}] timeout after {timeout_sec}s (fg)")
        if proc.returncode != 0:
            raise RuntimeError(f"[index={index}] entigraph.py failed (code={proc.returncode}) (fg)")
        return ""

    print(f"[start] i={index} uid={uid}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=None if not timeout_sec else timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(
            f"[index={index}] timeout after {timeout_sec}s\n"
            f"CMD: {' '.join(cmd)}\n"
            f"CWD: {PROJECT_ROOT}\n"
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"[index={index}] entigraph.py failed (code={proc.returncode})\n"
            f"STDOUT:\n{out.decode(errors='ignore')}\n"
            f"STDERR:\n{err.decode(errors='ignore')}\n"
        )
    print(f"[done ] i={index} uid={uid}")
    return out.decode(errors="ignore")

# ------------------------------ main ------------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-jsonl", required=True,
                    help="例: data/dataset/source/mycorpus.jsonl  (相対ならプロジェクトルート基準)")
    ap.add_argument("--out-dir-root", default="data/dataset/raw",
                    help="例: data/dataset/raw  (相対ならプロジェクトルート基準)")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--start", type=int, help="開始index（--index-list と併用不可）")
    ap.add_argument("--end", type=int, help="終了index（--index-list と併用不可）")
    ap.add_argument("--index-list", type=str, help="カンマ区切りindex/レンジ（例: 1,2,5-9）")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout-sec", type=int, default=600,
                    help="子プロセス（entigraph.py）の実行タイムアウト（秒）。0または負値で無制限。")
    ap.add_argument("--retries", type=int, default=2,
                    help="失敗・タイムアウト時の再試行回数")
    ap.add_argument("--debug-foreground", action="store_true",
                    help="子プロセスを前景で実行（出力をそのまま表示）。この場合は単発/低並列推奨。")
    args = ap.parse_args()

    src = resolve_under_root(args.source_jsonl)
    out_root = resolve_under_root(args.out_dir_root)

    if not src.exists():
        raise FileNotFoundError(f"source-jsonl not found: {src}")

    uids = load_uids_from_jsonl(src)
    n_total = len(uids)

    out_dir = out_root / f"quality_entigraph_{args.model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 実行対象 index の決定
    if args.index_list:
        todo = [i for i in parse_index_list(args.index_list) if 0 <= i < n_total]
    else:
        if args.start is None or args.end is None:
            raise SystemExit("--start/--end か --index-list のいずれかを指定してください")
        todo = [i for i in range(args.start, args.end) if 0 <= i < n_total]

    # 既存出力スキップ
    filtered: List[int] = []
    for i in todo:
        uid = uids[i]
        out_path = out_dir / f"{uid}.json" if uid else None
        if out_path and out_path.exists() and not args.force:
            print(f"[skip] {uid} (index={i})")
            continue
        filtered.append(i)

    if not filtered:
        print("Nothing to do. (All outputs exist or range empty)")
        return

    print(f"Plan: run entigraph.py for {len(filtered)} items "
          f"(concurrency={args.concurrency})")
    print(f"[paths] project_root={PROJECT_ROOT}")
    print(f"[paths] entigraph_py={ENTIGRAPH_PY}")
    print(f"[paths] source_jsonl={src}")
    print(f"[paths] out_dir={out_dir}")
    if args.timeout_sec and args.timeout_sec > 0:
        print(f"[cfg] timeout-sec={args.timeout_sec}  retries={args.retries}")
    if args.debug_foreground:
        print(f"[cfg] debug-foreground=ON (concurrency forced to 1)")
        args.concurrency = 1

    sem = asyncio.Semaphore(args.concurrency)
    ok = 0
    fail = 0

    async def bound_run(i: int):
        nonlocal ok, fail
        uid = uids[i]
        async with sem:
            attempt = 0
            while True:
                try:
                    _ = await run_one(i, args.model_tag,
                                      args.timeout_sec if args.timeout_sec>0 else None,
                                      args.debug_foreground,
                                      uid=uid)
                    ok += 1
                    if ok % 20 == 0:
                        print(f"[progress] ok={ok} fail={fail}")
                    return
                except Exception as e:
                    attempt += 1
                    if attempt > max(0, args.retries):
                        fail += 1
                        print(f"[error] index={i} uid={uid}: {e}")
                        return
                    print(f"[retry] index={i} uid={uid}: attempt {attempt}/{args.retries}")

    try:
        await asyncio.gather(*[bound_run(i) for i in filtered])
    except KeyboardInterrupt:
        print("\n[warn] KeyboardInterrupt: 中断要求を受けました。未完了プロセスは OS に残る可能性があります。")
        print("       必要なら次を実行してください:  pkill -f 'Generate_Synthetic_Data/data/entigraph.py'")
        raise
    print(f"Done. ok={ok} fail={fail}")
    if fail > 0:
        print("Some items failed. You can re-run with --force or specify them via --index-list.")

if __name__ == "__main__":
    asyncio.run(main())
