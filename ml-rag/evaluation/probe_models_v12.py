"""v12 排障：qwen-plus 突然 403 AccessDenied.Unpurchased，逐模型发一次 1-token 探针定位范围。
只问"1+1=?"、max_tokens=8，单模型成本可忽略；不调用任何缓存路径，纯看权限。
用法：python evaluation/probe_models_v12.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI

from rag.config import load_credentials

MODELS = ["qwen-plus", "qwen-plus-latest", "qwen-turbo", "qwen-flash",
          "qwen-max", "qwen3-max", "qwen3.7-max", "deepseek-v3"]

key, base = load_credentials()
client = OpenAI(api_key=key, base_url=base, timeout=60)
print(f"端点：{base}\n")
for m in MODELS:
    try:
        r = client.chat.completions.create(model=m, max_tokens=8, temperature=0,
                                           messages=[{"role": "user", "content": "1+1=?"}])
        print(f"  ✓ {m:18s} 通：{r.choices[0].message.content.strip()[:20]}")
    except Exception as e:  # noqa: BLE001 — 探针就是要逐个模型吞异常看状态码
        code = getattr(e, "status_code", "")
        msg = str(e).split("\n")[0][:110]
        print(f"  ✗ {m:18s} {code} {msg}")
