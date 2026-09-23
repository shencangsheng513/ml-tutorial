"""第十四发的**零训练**预飞：n=3 那档预演走不到 n=16 那条分支，所以把 16 特有的三件事单独跑一遍。

为什么要它：第十三发立下的固定动作是"先 `--nseeds 3` 把整条路径走通再开真跑"，但
`SEEDS16` 的使用、"第 9~{nseeds} 条是新买的"那行措辞、档案名 `_n16` 后缀，全都只在
`nseeds==16` 时才执行 ⇒ 预演**覆盖不到这一发的新代码**。而上一轮的教训恰好是这一类：
`KeyError: 12` 在 84 次训练之后才炸（233 秒白烧）。这一档把"训练之后才碰得到"的那几处
在花钱之前先看一眼。全程不训练。
"""
import re
import sys

from step4_distill import D4, T95, check_flow_constants
from step4f_alpha_curve import SEEDS8, SEEDS12, SEEDS16

fails = []

# ① 流池本身：16 条、无重复、前 12 条一字不动（补流不许换掉已买的流，否则新账不是同一把尺子量得更精）
print(f"[① 流池] len(SEEDS16)={len(SEEDS16)} 去重后={len(set(SEEDS16))}")
if len(set(SEEDS16)) != 16:
    fails.append("SEEDS16 有重复种子")
if tuple(SEEDS16[:12]) != tuple(SEEDS12):
    fails.append("SEEDS16 的前 12 条与 SEEDS12 不一致（补流换掉了已买的流）")
if tuple(SEEDS16[:8]) != tuple(SEEDS8):
    fails.append("SEEDS16 的前 8 条不是板块共用的那批 ⇒ 跨脚本逐位对账会失效")
new_four = [s for s in SEEDS16 if s not in SEEDS12]
print(f"        新增四条 = {new_four}")

# ② 系数册：n=16 两把都在册（缺任何一把都会炸在全部训练之后）
try:
    check_flow_constants(16)
    print(f"[② 系数册] D4[16]={D4[16]} T95[16]={T95[16]} 在册")
except SystemExit as e:
    fails.append(f"check_flow_constants(16) 退出：{e}")

# ③ 档案名后缀：先把 step4f 源码里那两行原文抓出来打印（人眼核它有没有被我改过），
#    再按同一公式在本地算两个取值——两件事分开做，别用 eval 把源码绕进来。
src = open("step4f_alpha_curve.py", encoding="utf-8").read()
m = re.search(r"\n    extra = \(.*?\)\s*\n", src, re.S)
print(f"[③ 档案名] step4f 源码原文：{m.group(0).strip() if m else '没抓到 extra 那两行'}")
if m is None:
    fails.append("找不到 extra 赋值，无法核对档案名")
SHAPE_RULE = "v5"


def suffix(nseeds: int, thin: bool, t_col, high: bool) -> str:
    """逐字照抄 step4f 里 extra 的拼接顺序（含 _thin / _hi 的位置）：改了源码就要在这里露出来。"""
    return ("" if nseeds == 3 else f"_n{nseeds}") + ("_thin" if thin else "") + \
           ("" if t_col is None else f"_t{t_col:g}") + ("_hi" if high else "") + f"_{SHAPE_RULE}"


for got, want in ((suffix(16, False, 6.0, False), "_n16_t6_v5"),
                  (suffix(16, False, None, False), "_n16_v5")):
    print(f"        n=16 t={want[4:-3] or '—'} → _{got.lstrip('_')} 期望 _{want.lstrip('_')}")
    if got != want:
        fails.append(f"档案名后缀 {got} != {want}")

# 自我审计③ 那行措辞（nseeds>8 那一支现在对 12 与 16 共用，必须两个都读一遍）
for nseeds in (12, 16):
    line = (f"这一发已按第4课e 的办法扩到 {nseeds} 条流"
            + (f"（第 9~{nseeds} 条是新买的，不在板块共用的那 8 条里 ⇒ 新增的只是读数密度，"
               "跨脚本逐位对账仍然只发生在共用的那 8 条上）" if nseeds > 8 else ""))
    print(f"[措辞 n={nseeds}] {line}")

print("\n" + ("预飞失败：\n  - " + "\n  - ".join(fails) if fails else "预飞全绿（未训练）"))
sys.exit(1 if fails else 0)
