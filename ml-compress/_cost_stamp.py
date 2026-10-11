"""把「这一发花了多少次训练」落进档案——以侧车形式，零训练。

为什么是侧车而不是原地补 `cost` 键：板块的规矩是**产物只能升版、旧版留盘做逐字段对账**
（第廿一发之后每一次升版都留着上一本，闸⑥ 的配对、闸⑮ 的过期指针都靠这个）。往 41 本历史账里
补一个键 = 原地改写历史档案：那些 blob 与 HEAD 里的版本逐字段不同，而名字相同，读档的人分不出
"这本被回填过"。侧车把这件事变成"新增一件、指着那一本"，历史档案一个字节不动。

价的算法**不在这里**：调 `step4f_alpha_curve.price_tag`——那是 `--dry-run` 开跑前报价、
落盘后的『价签对账』用的同一把尺子。这里再写一遍就是两个数可以各自漂。

读数从**档案自己记的字段**现读（`ladder` / `seeds` / `thin_ladder`），并且同时数一遍档案里
**实际落了条目的那两个袋子**（`arms` 的键数、`gate_alpha_zero_identity` 的条数）。两个数一并不
相等时，本件不报错、只把两侧都写进侧车——判红不红由守卫那侧的闸⑱ 决定，本件是产物方。

限度（写在这里而不是藏在回执里）：
  ① 只覆盖 `compress_report_v4f*.json` 这一族（`step4f` 的阶梯账）。v4b/v4c/v4j/v4k/v4l 那些
     账的价签口径各是另一套算术，本件不替它们报价——它们当中 14 本自带 `cost` 键（v4j/v4k/v4l）。
  ② 侧车记的是"**按档案自己记的实验配置**应当训练几次"，不是"这台机器历史上真的跑了几天"。
     账实对账的那一票是运行末那行『[价签对账] 开跑前价签 N 次 → 实际 N 次』，它不落档 ⇒
     侧车能证"这本账的价是 144"，不能证"这一发真的付了 144 而没多跑"。
  ③ 内容不含时间戳：同一本账重跑本件得到逐字相同的侧车（幂等），所以"侧车 mtime 变了"
     永远意味着"账或算法变了"，不会意味着"又跑了一次工具"。

用法（零训练，只读）：
    python D:/ML_Projects/ml-tutorial/ml-compress/_cost_stamp.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import step4f_alpha_curve as sf   # noqa: E402  只用它的 price_tag，不触发任何训练

BOOK_GLOB = "compress_report_v4f*.json"
SIDECAR_PREFIX = "compress_cost_"
# 档案里那两个"实际落了条目"的袋子：键数/条数各自应当等于算法报出来的那两支。
ENUM_KD = "arms"
ENUM_ID = "gate_alpha_zero_identity"
REQUIRED = ("ladder", "seeds", "thin_ladder")


def sidecar_name(book: Path) -> str:
    return SIDECAR_PREFIX + book.name[len("compress_report_"):]


def stamp(book: Path) -> tuple[dict, list[str]]:
    """读一本账 ⇒ (侧车内容, 不可核原因列表)。原因非空时侧车不落盘。"""
    d = json.loads(book.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        return {}, [f"缺字段 {missing}"]
    if not isinstance(d.get("seeds"), list) or not d["seeds"]:
        return {}, ["seeds 不是非空名单"]
    if not isinstance(d.get("ladder"), list) or not d["ladder"]:
        return {}, ["ladder 不是非空名单"]
    lad, seeds, thin = list(d["ladder"]), list(d["seeds"]), bool(d["thin_ladder"])
    tag = sf.price_tag(lad, seeds, thin)
    enum_kd = len(d.get(ENUM_KD) or {})
    enum_id = len(d.get(ENUM_ID) or [])
    return {
        "subject": book.name,
        "produced_by": "_cost_stamp.py",
        "fields_read": {"ladder": lad, "seeds_count": len(seeds), "thin_ladder": thin,
                        "temps_fixed": d.get("temps_fixed")},
        "price_tag_source": "step4f_alpha_curve.price_tag（与 --dry-run、落盘后的价签对账同一把尺子）",
        "ce_trainings": tag["ce_trainings"],
        "kd_trainings": tag["kd_trainings"],
        "identity_gate_trainings": tag["identity_gate_trainings"],
        "total_trainings": tag["total"],
        "api_fee": 0,
        "enumerated_from_archive": {ENUM_KD: enum_kd, ENUM_ID: enum_id},
        "enumeration_matches_derivation": (enum_kd == tag["kd_trainings"]
                                           and enum_id == tag["identity_gate_trainings"]),
        "note": ("零训练补价：这一发实际付了几次训练由运行末那行『[价签对账] 开跑前价签 N 次 → 实际 N 次』"
                 "作证，本侧车证的是『按这本账自己记的配置，价应该是 N』。历史档案未被改写——"
                 "本件是新增，不是回填。"),
    }, []


def main() -> int:
    books = sorted(HERE.glob(BOOK_GLOB))
    if not books:
        # 分母读不到就不许报"干净"：这一族一本都没有时，"0 个侧车"与"侧车全齐"是同一个数。
        raise SystemExit(f"盘上 glob({BOOK_GLOB}) 读到 0 本 ⇒ 没有可补价的账。"
                         "这一句不是『已经补完』，是『压根没开始』")
    wrote, same, unchecked = [], [], []
    for b in books:
        payload, why = stamp(b)
        if why:
            unchecked.append((b.name, "；".join(why)))
            continue
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        out = HERE / sidecar_name(b)
        if out.exists() and out.read_text(encoding="utf-8") == text:
            same.append(out.name)
            continue
        out.write_text(text, encoding="utf-8")
        wrote.append(out.name)
    print(f"[补价签] 读 {len(books)} 本（glob {BOOK_GLOB}）⇒ 落侧车 {len(wrote)} 件／"
          f"逐字已存在不动 {len(same)} 件／不可核 {len(unchecked)} 本")
    for n in unchecked:
        print(f"    不可核：{n[0]}｜{n[1]}")
    total = sum(json.loads((HERE / n).read_text(encoding="utf-8"))["total_trainings"]
                for n in wrote + same)
    print(f"[对账] 侧车覆盖 {len(wrote) + len(same)} 本／合计报价 {total} 次训练／纯 CPU、零 API 费")
    if len(wrote) + len(same) + len(unchecked) != len(books):
        raise SystemExit(f"分母不守恒：落盘 {len(wrote)} + 已存在 {len(same)} + 不可核 {len(unchecked)} "
                         f"≠ 读到 {len(books)} 本 ⇒ 有账既没落侧车也没进点名单，这一句『覆盖 N 本』不许引用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
