"""离线数字有据性核验：纯字符串，零 API。
答案里出现的每个数字，若在模型的整个输入（检索正文+题面）里找不到逐字出处，
标为可疑（疑似凭空生成）。生产答案走 SYSTEM_RAG，不带逐句【资料i】标注，
故这里按"任一输入"做集合式核验。
已知局限：模型做算术派生的数字（如 40→58 报"+18"）会被误标——所以它是提示器不是判决书。"""
import re

_NUM = re.compile(r"\d+\.\d+|\d{2,}")


def ungrounded_numbers(answer: str, hits, question: str = "") -> list[str]:
    """返回答案中"在模型的整个输入里都找不到逐字出处"的数字（去重、按出现序）。
    hits 为 retrieve/retrieve_hybrid 的返回，元素是 (chunk, score)；question 为原题，
    纳入依据集是因为答案复述题面自带数字属正常，不该算编造。
    仍会误报算术派生值（如两数之差）——故它是提示器，措辞要留余地。"""
    context = re.sub(r"\s+", "", "".join(c.text for c, _ in hits) + question)
    seen: set[str] = set()
    bad: list[str] = []
    for n in _NUM.findall(answer):
        if n in seen:
            continue
        seen.add(n)
        if n not in context:
            bad.append(n)
    return bad
