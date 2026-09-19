"""LLM-as-judge：让模型按评分细则给答案打分（0-10），带缓存避免重复计费。"""
import hashlib
import json
import re
import time

from openai import OpenAI

from .config import CHAT_MODEL, INDEX_DIR, load_credentials

CRITERIA = """判卷纪律：事实核验只以"标准要点"为准，不得凭你自己记忆引入要点之外的数字；篇幅长不等于分高。
第一步·列事实清单：从回答A、B分别摘出可核验断言（具体数字、端口、阈值、百分比），逐条对照标准要点：
一致=✓；与要点冲突、或要点里根本没有却被当成事实给出的具体值=✗。两组同罪同罚，无豁免。每条不超过20字，不要引号。
第二步·锚定打分：从8分起评——每遗漏一个标准要点−2；每处✗−3；要点全覆盖且零✗+2；可操作建议+0~1（封顶10）。
10 分只发给"零✗且要点全覆盖"的答案；拿不准就扣。
只输出 JSON（reason 严格≤40字，防截断）：
{"A": x, "B": y, "fact_check": {"A": ["断言→✓或✗"], "B": ["…"]}, "deductions": {"A": [], "B": []}, "reason": "≤40字"}"""

GROUNDED_RULE = """\n新增"有据性"核验：回答中每个具体断言（数字、阈值、端口、百分比、机制细节）
都必须能在【原始资料】中找到出处；每发现一处"资料没写但回答说得斩钉截铁"的断言记为一条 violation
并 −2（标明"推测"的合理推断不扣）。"""

SINGLE_RULE = """\n打分从8起评：每遗漏一个标准要点−2；每条 violation −2；要点全覆盖且零 violation +2（封顶10）。"""

_CACHE = INDEX_DIR / "judge_cache.json"


def _cache_load():
    return json.loads(_CACHE.read_text(encoding="utf-8")) if _CACHE.exists() else {}


def _ask(prompt: str, required: tuple[str, ...]) -> dict:
    """判卷调用：md5 缓存 + 3 次重试 + 截断兜底（正则把分数抠出来）。"""
    key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    c = _cache_load()
    if key in c:
        return c[key]
    cli_key, base = load_credentials()
    cli = OpenAI(api_key=cli_key, base_url=base, timeout=120)
    raw = ""
    for attempt in range(3):
        try:
            r = cli.chat.completions.create(model=CHAT_MODEL, temperature=0, max_tokens=700,
                                            messages=[{"role": "user", "content": prompt}])
            raw = r.choices[0].message.content or ""
            s, e = raw.find("{"), raw.rfind("}")
            if s < 0 or e <= s:
                raise ValueError("judge 回复里没有 JSON")
            try:
                out = json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                found = {}
                for k in required:
                    m = re.search(rf'"{k}"\s*:\s*(\d+(?:\.\d+)?)', raw)
                    if not m:
                        raise
                    found[k] = float(m.group(1))
                out = {**found, "reason": raw[s:s + 60] + "(截断兜底)"}
            if any(k not in out for k in required):
                raise ValueError("judge JSON 缺字段")
            break
        except Exception:
            if attempt == 2:
                (INDEX_DIR / "judge_raw_last.txt").write_text(raw, encoding="utf-8")
                raise
            time.sleep(2 * (attempt + 1))
    c[key] = out
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    return out


def judge_pair(question: str, gold_points: list[str], ans_a: str, ans_b: str,
               context: str = "") -> dict:
    """gold_points 是出题时写好的标准要点，判卷有依据，不当纯裁判。
    context 非空时启用"有据性"核验：把检索到的原始资料一并给判卷官。"""
    prompt = (f"问题：{question}\n标准要点（人类教师提供）：{json.dumps(gold_points, ensure_ascii=False)}\n"
              f"回答A：{ans_a}\n---\n回答B：{ans_b}\n---\n{CRITERIA}")
    if context:
        prompt = f"【原始资料】\n{context}\n\n{prompt}{GROUNDED_RULE}"
    return _ask(prompt, ("A", "B"))


def judge_faithful(question: str, gold_points: list[str], ans: str, context: str) -> dict:
    """单一答案的有据性判卷：对照原始资料找幻觉（探针已验证 3/3 抓到注入错误）。"""
    prompt = (f"【原始资料】\n{context}\n\n问题：{question}\n"
              f"标准要点（人类教师提供）：{json.dumps(gold_points, ensure_ascii=False)}\n"
              f"回答：{ans}\n---\n只输出 JSON："
              f'{{"score": x, "violations": ["无据断言，每条不超过20字"], "reason": "不超过40字"}}'
              f"{GROUNDED_RULE}{SINGLE_RULE}")
    return _ask(prompt, ("score",))
