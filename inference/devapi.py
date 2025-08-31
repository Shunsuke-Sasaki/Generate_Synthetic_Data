import os, time
from typing import Optional
import httpx
from openai import OpenAI

# inference/devapi.py など gptqa を定義している場所

import os, httpx
from openai import OpenAI

def gptqa(prompt: str,
          openai_model_name: str,
          system_message: str,
          json_format: bool = False,
          temp: float = 0.2):
    # vLLM の OpenAI 互換サーバへ接続
    base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key  = os.getenv("OPENAI_API_KEY", "EMPTY")  # ダミーでOK
    client = OpenAI(base_url=base_url,
                    api_key=api_key,
                    http_client=httpx.Client(timeout=60.0))

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    kwargs = dict(model=openai_model_name, messages=messages, temperature=temp)
    if json_format:
        # vLLM のバージョンやモデルによっては未対応な場合があるため
        # まずは設定し、ダメなら後段のフォールバックで外すのが安全
        kwargs["response_format"] = {"type": "json_object"}

    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as e:
        # JSONモード非対応時のフォールバック（純プロンプトでJSON出力させる）
        if json_format and "response_format" in str(e):
            kwargs.pop("response_format", None)
            messages[0]["content"] = system_message + "\nReturn ONLY valid JSON."
            completion = client.chat.completions.create(**kwargs)
        else:
            raise
    return completion.choices[0].message.content



"""
def gptqa(
    prompt: str,
    openai_model_name: str,
    system_message: str,
    json_format: bool = False,
    temp: float = 0.2,
    timeout: float = 60.0,
    max_tokens: Optional[int] = None,
    max_retries: int = 5,
):
    
    Return: string (model output). entigraph.py 側で json.loads() される前提。
    set_openai_key() が先に呼ばれて OPENAI_API_KEY が入っていることを想定。
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    http_client = httpx.Client(proxies=proxy, timeout=timeout) if proxy else httpx.Client(timeout=timeout)
    client = OpenAI(api_key=api_key, http_client=http_client)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    kwargs = {
        "model": openai_model_name,
        "messages": messages,
        "temperature": temp,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if json_format:
        # JSON専用出力（対応モデルでは純JSONが返る）
        kwargs["response_format"] = {"type": "json_object"}

    backoff = 1.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            last_err = e
            # response_format非対応などで弾かれたらフォールバック
            if json_format and ("response_format" in msg or "not supported" in msg):
                kwargs.pop("response_format", None)
                # 出力をJSON限定する追加指示
                messages[0]["content"] = system_message + "\nReturn only valid JSON with no extra text."
                kwargs["messages"] = messages
            # レート/一時的エラーはリトライ
            if any(s in msg for s in ("429", "Rate limit", "502", "503", "504", "timeout", "Temporary")):
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            # それ以外は即時エラー
            raise
    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")
"""