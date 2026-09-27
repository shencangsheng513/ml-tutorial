# -*- coding: utf-8 -*-
"""一次性核对：闸⑫ 的旧判别器在**盘上真产物**上会红几条、新判别器红几条。
用途=给本轮"3 条假红"那个读数补一份可重跑的盘上原件（原件被同名重跑覆盖了）。"""
import glob, json, re

BRACE = re.compile(r"\{([^{}]*)\}")

def old(inner):
    return bool(re.search(r"[A-Za-z_]\w*\s*[(\[]", inner)) or "'" in inner or '"' in inner or "[" in inner

def new(inner):
    return bool(re.match(r"[A-Za-z_]\w*(?:[.!([:].*)?$", inner.strip()))

def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node

import sys
INJECT = "--inject" in sys.argv
oh, nh, nf, books = [], [], 0, 0
for p in sorted(glob.glob("compress_report_v*.json")):
    books += 1
    d = json.load(open(p, encoding="utf-8"))
    if INJECT:
        for i, x in enumerate(d.get("limitations") or []):
            if isinstance(x, str) and "种子底由" in x and "个点" in x:
                d["limitations"][i] = re.sub(r"(种子底由 )\d+( 个点)", r"{len(seeds)}", x)
    for path, s in walk(d):
        nf += 1
        for m in BRACE.finditer(s):
            inner = m.group(1)
            tag = "%s%s: {%.60s}" % (p, path, inner)
            if old(inner):
                oh.append(tag)
            if new(inner):
                nh.append(tag)
print(f"全册 {books} 本 / 展开 {nf} 条字符串字段")
print(f"旧判别器命中 {len(oh)} 处：")
for t in oh:
    print("   ", t)
print(f"新判别器命中 {len(nh)} 处：")
for t in nh:
    print("   ", t)
