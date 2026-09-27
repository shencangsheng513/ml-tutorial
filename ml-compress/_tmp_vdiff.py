# -*- coding: utf-8 -*-
"""一次性对账件：比对两份档案 JSON，把**不同的键路径**逐条列出来（升版时"除这几位之外必须逐字段相同"的凭据）。

为什么要有这件而不是"扫一眼 diff"：本轮要升版的两本账（step4i v2→v3、step4l v1→v2）各自只许动
少数几个键，其余每一位都必须一字不动。手工 diff 面对 3 万字节、上千键的档案只能看出"看着差不多"，
而"差不多"在这门课里就是没测到。

自检（先答一道已知答案的考题，再拿它去答没答过的）：
  python -u _tmp_vdiff.py compress_report_v4i_mechanism_v2.json compress_report_v4i_mechanism_v3.json
  ⇒ 已知差异只该出现在 repro_scan_T3_ladder 之下 + limitations（追加）。跑出来若还别处不同，
    要么档案错了，要么这件本身不灵。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if len(sys.argv) != 3:
    raise SystemExit("用法：python -u _tmp_vdiff.py <旧档.json> <新档.json>")
a = json.loads((HERE / sys.argv[1]).read_text(encoding="utf-8"))
b = json.loads((HERE / sys.argv[2]).read_text(encoding="utf-8"))

diff = []


def walk(x, y, path):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x:
                diff.append(f"+ {path}/{k}（只在新档里有）")
            elif k not in y:
                diff.append(f"- {path}/{k}（只在新档里没有 = 旧档被改过）")
            else:
                walk(x[k], y[k], f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list):
        shifted = False
        if len(x) != len(y):
            diff.append(f"~ {path} 长度 {len(x)} → {len(y)}")
            # 追加在头部会把整条列表逐项推位——那不是"改动"。旧项若全部原样在场，就只记这一条，不再逐位比
            try:
                shifted = set(x) <= set(y)
            except TypeError:                       # 项不可哈希（dict/list）⇒ 退回逐位比
                pass
        if not shifted:
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
    elif x != y:
        diff.append(f"~ {path}: {x!r} → {y!r}")


walk(a, b, "")
n_leaf = len(diff)
by_prefix: dict[str, int] = {}
for d in diff:
    top = d[2:].lstrip("/").split("/")[0].split("[")[0] or "（顶层）"
    by_prefix[top] = by_prefix.get(top, 0) + 1

def count_leaves(o) -> int:
    if isinstance(o, dict):
        return sum(count_leaves(v) for v in o.values()) or 1
    if isinstance(o, list):
        return sum(count_leaves(v) for v in o) or 1
    return 1


print(f"[分母] 旧档 {count_leaves(a)} 个叶子 / 新档 {count_leaves(b)} 个叶子")
print(f"[差异] 共 {n_leaf} 条，按顶层位置归堆：")
for k, v in sorted(by_prefix.items(), key=lambda kv: -kv[1]):
    print(f"   {k}: {v} 条")
for d in diff[:40]:
    print("   ", d)
if n_leaf > 40:
    print(f"    …另 {n_leaf - 40} 条（同一顶层位置下）")
if not diff:
    print("    （无差异 ⇒ 这一趟升版什么也没改，那它凭什么叫 v+1？）")
