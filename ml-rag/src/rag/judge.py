"""LLM-as-judge：让模型按评分细则给答案打分（0-10），带缓存避免重复计费。"""
import hashlib
import json
import time

from openai import OpenAI

from .config import CHAT_MODEL, INDEX_DIR, load_credentials

CRITERIA = """判卷纪律：事实核验只以"标准要点"为准，不得凭你自己记忆引入要点之外的数字；
篇幅长不等于分高，只按实质内容打分。
评分维度（各占1/3）：
1. 事实准确：关键数字/结论是否正确，无编造
2. 覆盖要点：是否答出问题问的核心点
3. 可操作：学习者能否据此行动（复现/避坑）
对两个回答 A/B 各打 0-10 分，只输出 JSON：{"A": x, "B": y, "reason": "一句话"}"""

_CACHE = INDEX_DIR / "judge_cache.json"


def _cache_load():
    return json.loads(_CACHE.read_text(encoding="utf-8")) if _CACHE.exists() else {}


def judge_pair(question: str, gold_points: list[str], ans_a: str, ans_b: str) -> dict:
    """gold_points 是出题时写好的标准要点，判卷有依据，不当纯裁判。"""
    prompt = (f"问题：{question}\n标准要点（人类教师提供）：{json.dumps(gold_points, ensure_ascii=False)}\n"
              f"回答A：{ans_a}\n---\n回答B：{ans_b}\n---\n{CRITERIA}")
    key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    c = _cache_load()
    if key in c:
        return c[key]
    cli_key, base = load_credentials()
    cli = OpenAI(api_key=cli_key, base_url=base, timeout=120)
    for attempt in range(3):
        try:
            r = cli.chat.completions.create(model=CHAT_MODEL, temperature=0, max_tokens=300,
                                            messages=[{"role": "user", "content": prompt}])
            raw = r.choices[0].message.content
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    s, e = raw.find("{"), raw.rfind("}")
    out = json.loads(raw[s:e + 1])
    c[key] = out
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return out
