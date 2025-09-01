#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entigraph.py  —  Synthetic CPT (EntiGraph) 生成パイプライン（堅牢化版）

主な変更点：
- LLM 出力のサニタイズ強化：コードフェンス/前置き/＜think＞除去、JSON ブロック抽出、救済パース
- リトライ制御：上限回数＋指数バックオフ、無限ループ防止
- 長文入力の安全トリム：MAX_DOC_CHARS 超は先頭のみ使用
- 抽出エンティティの上限デフォルトを 20 に（env: MAX_ENTITIES で上書き可）

必要環境：
- inference.devapi.gptqa（OpenAI 互換 API ラッパ）
- utils.io_utils.{jload,set_openai_key}
- tqdm
"""

import os
import sys
import json
import random
import re
import time
from typing import List, Tuple, Optional, Iterable

# リポジトリ直下を import path に追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tqdm import tqdm  # noqa: E402
from inference.devapi import gptqa  # noqa: E402
from utils.io_utils import jload, set_openai_key  # noqa: E402

# QuALITY はオプショナル
try:
    from tasks.quality import QuALITY  # noqa: E402
except Exception:
    QuALITY = None

# ===== 設定 =====
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4-turbo")
SOURCE_JSONL = os.environ.get("SOURCE_JSONL", "data/dataset/source/mycorpus.jsonl")
OUT_DIR_TMPL = "data/dataset/raw/quality_entigraph_{model}"

# エンティティ上限（環境変数 MAX_ENTITIES でも指定可）— デフォルト 20
MAX_ENTITIES = int(os.environ.get("MAX_ENTITIES", "20"))

# コスト制御（必要なら上限を入れる / None で全組合せ）
MAX_PAIRS: Optional[int] = None
MAX_TRIPLES: Optional[int] = None

# 堅牢化用の閾値
MAX_JSON_RETRIES = int(os.environ.get("MAX_JSON_RETRIES", "6"))
RETRY_SLEEP_SECS = float(os.environ.get("RETRY_SLEEP_SECS", "1.5"))
MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "24000"))  # 入力が長すぎると空返し/失敗しやすい
DEBUG_SHOW_PREFIX = int(os.environ.get("DEBUG_SHOW_PREFIX", "180"))

# ===== UTF-8 保存ヘルパ =====
def jdump_utf8(obj, path, *, pretty: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

# ===== 生成文サニタイザ（思考過程/メタを除去） =====
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
CODEFENCE_RE = re.compile(r"^```.*?\n|```$", flags=re.MULTILINE | re.DOTALL)

HEADERS = [
    r"analysis", r"thought\s*process", r"reasoning", r"chain\s*of\s*thought",
    r"思考過程", r"推論(?:過程)?", r"考察"
]
SECTION_RES = [
    re.compile(rf"(?:^|\n)\s*(?:{h})\s*:\s*.*?(?=\n\s*\n|$)", flags=re.IGNORECASE | re.DOTALL)
    for h in HEADERS
]

def sanitize_completion(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = THINK_BLOCK_RE.sub("", text)
    t = THINK_TAG_RE.sub("", t)
    t = CODEFENCE_RE.sub("", t)
    for pat in SECTION_RES:
        t = pat.sub("", t)
    pos = text.lower().rfind("</think>")
    if pos != -1:
        tail = text[pos + len("</think>") :]
        if len(tail.strip()) >= 10:
            t = tail
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ===== JSON 抜き出し・救済ユーティリティ =====
def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_object(s: str) -> Optional[str]:
    """テキスト中の最初の top-level JSON オブジェクト {...} を抽出（括弧の深さで判定）"""
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None

def _try_parse_entities_obj(text: str) -> Optional[dict]:
    """text から JSON を取り出して dict に。ダメなら救済して返す。"""
    if not text:
        return None
    t = sanitize_completion(text)
    t = _strip_code_fences(t)

    # 1) 直 parse
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and "entities" in obj:
            return obj
    except Exception:
        pass

    # 2) {...} 抜き出し → parse
    block = _extract_json_object(t)
    if block:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "entities" in obj:
                return obj
        except Exception:
            pass

    # 3) 正規表現で "entities": [...] と "summary": "..." を救済抽出（最終手段）
    m = re.search(r'"entities"\s*:\s*(\[[^\]]*\])', t, flags=re.IGNORECASE | re.DOTALL)
    ents: List[str] = []
    if m:
        try:
            tmp = json.loads(m.group(1))
            if isinstance(tmp, list):
                ents = [x for x in tmp if isinstance(x, str)]
        except Exception:
            ents = []
    m2 = re.search(r'"summary"\s*:\s*"([^"]*)"', t, flags=re.IGNORECASE | re.DOTALL)
    summ = m2.group(1).strip() if m2 else ""

    if ents or summ:
        return {"entities": ents, "summary": summ}

    return None

# ===== 自前 JSONL 用システムメッセージ =====
SYS_GENERATE_ENTITIES = (
    "あなたはテキストから主要な固有表現・専門用語を抽出する抽出器です。"
    "入力文書から重複のないエンティティを10〜30個ほど抽出し、短い要約も返してください。"
    "必ず JSON 形式 {\"entities\": [\"...\"], \"summary\": \"...\"} で返答。"
    "出力はこの JSON 一個のみ。前置き、後置き、マークダウン、コードフェンス、<think> 等は一切禁止。"
    "本文にない情報は作らず、日本語で簡潔に。"
)
SYS_TWO_ENTITY_REL = (
    "以下の文書と2つのエンティティに基づき、両者の関係を日本語で1〜2段落で説明してください。"
    "本文記載に即し、推測や一般論は避けてください。"
    "出力は最終的な説明文のみ。前置き・見出し・思考過程・Reasoning・<think>タグなどは一切出力しないでください。"
)
SYS_THREE_ENTITY_REL = (
    "以下の文書と3つのエンティティに基づき、三者の関係（相互作用・因果・包含・対比など）を"
    "日本語で1〜2段落で具体的に説明してください。本文にない内容は避けてください。"
    "出力は最終的な説明文のみ。前置き・見出し・思考過程・Reasoning・<think>タグなどは一切出力しないでください。"
)

# ===== 低レベル I/O =====
def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

class _Doc:
    __slots__ = ("uid", "content")
    def __init__(self, uid: str, content: str):
        self.uid = uid
        self.content = content

def load_documents_from_jsonl(source_jsonl: str) -> List[_Doc]:
    if not os.path.exists(source_jsonl):
        raise FileNotFoundError(f"SOURCE_JSONL not found: {source_jsonl}")
    docs: List[_Doc] = []
    for rec in _read_jsonl(source_jsonl):
        uid = rec.get("uid")
        raw = rec.get("raw")
        if uid and raw:
            docs.append(_Doc(uid, raw))
    if not docs:
        raise RuntimeError(f"No valid records in {source_jsonl} (need uid/raw).")
    return docs

def load_documents_from_quality(split: str = "all") -> Tuple[List[_Doc], object]:
    if QuALITY is None:
        raise RuntimeError("tasks.quality.QuALITY is unavailable. Use --source jsonl or install deps.")
    task = QuALITY(split)
    docs = [_Doc(d.uid, d.content) for d in task.documents]
    return docs, task

# ===== LLM 呼び出し =====
def generate_entities(document_content: str,
                      system_message: str,
                      openai_model: str) -> dict:
    # 長文のときは安全に頭から切る（トークン上限超過→空返し/エラーを避ける）
    doc = document_content if len(document_content) <= MAX_DOC_CHARS else document_content[:MAX_DOC_CHARS]

    prompt = f"""
    ### Document Content:
    {doc}
    """

    last_sample = ""
    for attempt in range(1, MAX_JSON_RETRIES + 1):
        try:
            completion = gptqa(prompt, openai_model, system_message, json_format=True)
        except Exception as e:
            print(f"[warn] gptqa raised on attempt {attempt}/{MAX_JSON_RETRIES}: {e}")
            time.sleep(RETRY_SLEEP_SECS * attempt)
            continue

        if not completion or not str(completion).strip():
            print(f"[warn] Empty completion on attempt {attempt}/{MAX_JSON_RETRIES}")
            time.sleep(RETRY_SLEEP_SECS * attempt)
            continue

        last_sample = str(completion).strip()
        obj = _try_parse_entities_obj(last_sample)
        if obj and isinstance(obj, dict) and "entities" in obj:
            return obj

        head = last_sample[:DEBUG_SHOW_PREFIX].replace("\n", "\\n")
        print(f"[warn] Non-JSON or missing keys on attempt {attempt}/{MAX_JSON_RETRIES}: '{head}...'")
        time.sleep(RETRY_SLEEP_SECS * attempt)

    print("[error] Failed to obtain valid JSON for entities after retries.")
    if last_sample:
        head = last_sample[:DEBUG_SHOW_PREFIX].replace("\n", "\\n")
        print(f"[error] Last sample head: '{head}...'")
    return {"entities": [], "summary": ""}

def generate_two_entity_relations(document_content: str,
                                  entity1: str,
                                  entity2: str,
                                  system_message: str,
                                  openai_model: str) -> str:
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    """
    completion = gptqa(prompt, openai_model, system_message)
    return completion

def generate_three_entity_relations(document_content: str,
                                    entity1: str,
                                    entity2: str,
                                    entity3: str,
                                    system_message: str,
                                    openai_model: str) -> str:
    prompt = f"""
    ### Document Content:
    {document_content}
    ### Entities:
    - {entity1}
    - {entity2}
    - {entity3}
    """
    completion = gptqa(prompt, openai_model, system_message)
    return completion

# ===== エンティティ正規化（重複・空文字の除去 + 上限カット） =====
def normalize_entities(entities: List[str], max_n: int = MAX_ENTITIES) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for e in entities or []:
        if not isinstance(e, str):
            continue
        s = e.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
        if max_n and len(cleaned) >= max_n:
            break
    return cleaned

# ===== メイン：1ドキュメント処理 =====
def generate_synthetic_data_for_document(document_index: int,
                                         model_name: str = DEFAULT_MODEL_NAME,
                                         source_mode: str = "jsonl",
                                         quality_split: str = "all"):
    """
    document_index: 0-origin
    source_mode: "jsonl" | "quality"
    """
    random.seed(42)
    set_openai_key()

    # ソース読み込み
    task = None
    if source_mode == "quality":
        docs, task = load_documents_from_quality(split=quality_split)
        sys_msg_entities = task.openai_system_generate_entities
        sys_msg_two = task.openai_system_generate_two_entity_relations
        sys_msg_three = task.openai_system_generate_three_entity_relations
    else:
        docs = load_documents_from_jsonl(SOURCE_JSONL)
        sys_msg_entities = SYS_GENERATE_ENTITIES
        sys_msg_two = SYS_TWO_ENTITY_REL
        sys_msg_three = SYS_THREE_ENTITY_REL

    if document_index < 0 or document_index >= len(docs):
        raise IndexError(f"document_index out of range: {document_index} (0..{len(docs)-1})")

    document = docs[document_index]
    out_dir = OUT_DIR_TMPL.format(model=model_name)
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/{document.uid}.json"

    print(f"[info] Generating synthetic data for uid={document.uid} (idx={document_index})")
    print(f"[info] Source: {source_mode} | Model tag: {model_name}")
    print(f"[info] Output: {output_path}")
    print(f"[info] Doc length: {len(document.content)} chars (truncated to {min(len(document.content), MAX_DOC_CHARS)})")

    # 既存出力の読み込み/初期化
    if os.path.exists(output_path):
        output = jload(output_path)
        if not isinstance(output, list):
            output = [[]]
    else:
        output = [[]]  # [entities(list), summary(str), ...relations...]

    # エンティティ抽出（既存・新規どちらでも上限で制限）
    if isinstance(output[0], list) and len(output[0]) > 0:
        entities_raw = output[0]
        entities = normalize_entities(entities_raw, MAX_ENTITIES)
        if entities != entities_raw:
            print(f"[info] Trim entities: {len(entities_raw)} -> {len(entities)}")
            output[0] = entities
            jdump_utf8(output, output_path)
        # 既存 summary がなければ空を追加
        if len(output) == 1:
            output.append("")
            jdump_utf8(output, output_path)
    else:
        ent_obj = generate_entities(document.content, sys_msg_entities, model_name)
        entities = normalize_entities(ent_obj.get("entities", []), MAX_ENTITIES)
        summary = ent_obj.get("summary", "")
        print(f"[info] Entities extracted: {len(entities)} (capped at {MAX_ENTITIES})")
        output[0] = entities
        output.append(summary)
        jdump_utf8(output, output_path)

    # 2者関係
    pair_list: List[Tuple[str, str]] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            pair_list.append((entities[i], entities[j]))
    if MAX_PAIRS is not None and len(pair_list) > MAX_PAIRS:
        pair_list = pair_list[:MAX_PAIRS]

    for e1, e2 in tqdm(pair_list, desc="2-entity relations"):
        try:
            resp = generate_two_entity_relations(document.content, e1, e2, sys_msg_two, model_name)
            if resp:
                output.append(sanitize_completion(resp))
                jdump_utf8(output, output_path)
        except Exception as e:
            print(f"[warn] two-entity generation failed for ({e1}, {e2}): {e}")

    # 3者関係
    triple_list: List[Tuple[str, str, str]] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            for k in range(j + 1, len(entities)):
                triple_list.append((entities[i], entities[j], entities[k]))
    random.shuffle(triple_list)
    if MAX_TRIPLES is not None and len(triple_list) > MAX_TRIPLES:
        triple_list = triple_list[:MAX_TRIPLES]

    for e1, e2, e3 in tqdm(triple_list, desc="3-entity relations"):
        try:
            resp = generate_three_entity_relations(document.content, e1, e2, e3, sys_msg_three, model_name)
            if resp:
                output.append(sanitize_completion(resp))
                jdump_utf8(output, output_path)
        except Exception as e:
            print(f"[warn] three-entity generation failed for ({e1}, {e2}, {e3}): {e}")

# ===== エントリポイント =====
def _parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int, nargs="?", help="処理する記事のインデックス (0-origin)")
    ap.add_argument("--source", choices=["jsonl", "quality"], default="jsonl",
                    help="ソースの種類（jsonl: 自前 {uid,raw} / quality: QuALITY）")
    ap.add_argument("--quality-split", choices=["train", "dev", "all"], default="all",
                    help="--source quality のときの分割")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                    help="出力ディレクトリ名用のモデル名ラベル（例: gpt-4-turbo）")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.index is None:
        print("Usage:\n  python data/entigraph.py <index> [--source jsonl|quality] [--quality-split all] [--model-name gpt-4-turbo]")
        sys.exit(1)
    generate_synthetic_data_for_document(
        document_index=args.index,
        model_name=args.model_name,
        source_mode=args.source,
        quality_split=args.quality_split,
    )
