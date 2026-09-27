# -*- coding: utf-8 -*-
"""第廿四发 D 的零训练预检（我的临时件，收工时删）。

要证的四件事，全是"名字"和"闸门"，一次训练都不烧：
  1. step4b 刚登记的满数据档空后缀，与 step4c.v4b_path 对同一格算出的名字**逐字相同**
     （两处名字各写一套 = 第二把尺子；这一条不成立就别去烧 6 次训练）。
  2. 新登记的档案名不许撞上别家的本子：入库后由那本自己记的 (族, 流数, 档) 自证身份；未入库只报『待买』。
  3. C 修完之后 ④ 的硬约束成立，且**两种盘态都成立**：v4b_path 返回名字 ⟺ 那本在盘上且自记流数 = 问它的 n；
     缺那本 ⟹ 在第一次训练之前拒跑，绝不回落到别的流数。
     （只有 n>3 这一支有缺档即拒的闸；3 条流那支没有存在性检查——那一本正是 step4b 要新造的。）
  4. 新登记的空后缀没把 MLP 的两档共本历史炸开：真跑 CLI `--arch mlp --tier 满数据8500`，
     合并档案闸必须在训练前落下并且报成红点（不是"代码里看着会拦"）。

写这一份时踩到的口径（第一版就按它自己说的错了两处）：[2][3] 原先断言的是"这一格还空着 / v4b_path 必须炸"，
①② 一入库这两条立刻变成**两条假红**——那两条断言钉的是"跑预检的那一刻"，不是不变量。
预检要在链的每一段都能复跑，所以断言必须写成"两种盘态都成立"的形式（本轮已改）。
"""
import json
from pathlib import Path

import step4b_seed_variance as sb
import step4c_alpha_T_sweep as sc

HERE = Path(__file__).resolve().parent
fails: list[str] = []

print("[1] cnn × 满数据8500 × S128：两个脚本算出的同一个名字")
sb.ARCH, sb.TIER, sb.WIDTH = "cnn", "满数据8500", 128
sc.ARCH = "cnn"
b3, b8 = sb.book_name(3).name, sb.book_name(8).name
c3 = sc.v4b_path(128, "满数据8500", 3).name
try:                                   # ② 入库之前这本查无档案 = 该炸（[3] 单独钉），此处先记下待复核
    c8: str | None = sc.v4b_path(128, "满数据8500", 8).name
    deferred8 = False
except SystemExit:
    c8, deferred8 = None, True
print(f"  step4b 要写的名字   3流={b3}  8流={b8}")
print(f"  step4c 要读的名字   3流={c3}  8流={c8 if c8 else '（② 未入库，拒跑——待 ② 之后复跑本预检自明）'}")
if c3 != b3:
    fails.append(f"[1-1] 写盘名 {b3} ≠ 读数名 {c3}——3 流那本造出来也没人读得到")
else:
    print("  [OK] 3 流：写与读同名")
if deferred8:
    print("  [deferred] 8 流：这一条腿要等 ② 把那本底买回来才测得到，本预检在 ② 之后必须复跑")
elif c8 != b8:
    fails.append(f"[1-2] 写盘名 {b8} ≠ 读数名 {c8}——8 流那本造出来也没人读得到")
else:
    print("  [OK] 8 流：写与读同名")

def _flows_of(p: Path) -> tuple:
    """那本底自己记的身份：(学生架构, 流数, 它声称跑的那一档)。"""
    dd = json.loads(p.read_text(encoding="utf-8"))
    return (dd.get("student_arch"), len(dd["seeds"]), dd.get("tier_arg"))


print("\n[2] 新登记的档案名不许撞上别家的本子：入库后用那本自己记的身份自证（未入库记『待买』）")
for name, n_want in [(b3, 3), (b8, 8)]:
    p = HERE / name
    if not p.exists():
        print(f"  [待买] {name} 还没入库 ⇒ 这一格正是本次要跑的")
        continue
    fam, nflow, tier_arg = _flows_of(p)
    print(f"  [已入库] {name}：族={fam} 流数={nflow} 档={tier_arg}")
    if (fam, nflow, tier_arg) != ("cnn", n_want, "满数据8500"):
        fails.append(f"[2] {name} 自证的身份是 族={fam} × {nflow} 流 × {tier_arg}，"
                     f"不是 cnn × {n_want} 流 × 满数据8500 ⇒ 这个新名字撞上了别家的本子")

print("\n[3] C 修完之后的契约（两种状态都成立才算）：v4b_path 返回名字 ⟺ 那本底在盘上且流数对得上；"
      "缺那本 ⟹ 在训练之前拒跑，绝不回落")
n_want = 8
p8 = HERE / b8
try:
    got = sc.v4b_path(128, "满数据8500", n_want)
    if not p8.exists():
        fails.append(f"[3] v4b_path 返回了 {got.name} 而那本不在盘上 ⇒ 返回了一个查无此账的名字")
    elif _flows_of(got)[1] != n_want:
        fails.append(f"[3] v4b_path 返回 {got.name}，可它自己记的流数是 {_flows_of(got)[1]} ≠ {n_want}"
                     f"⇒ 借了别的流数的底，C 白修")
    else:
        print(f"  [OK] 返回 {got.name}，那本自记流数 = {_flows_of(got)[1]} = 问它的那个 n")
except SystemExit as e:
    if p8.exists():
        fails.append(f"[3] {b8} 明明在盘上却拒跑 ⇒ 名字两处不一致（写盘名 {b8} 不是读数名）")
    else:
        print(f"  [OK] ② 未入库，训练前拒跑：{str(e).splitlines()[0][:78]}…")

print("\n[4] MLP 族走这一档仍被合并档案闸拦在训练前（由 CLI 真跑，不看代码）")
import subprocess
import sys
r = subprocess.run([sys.executable, "step4b_seed_variance.py", "--arch", "mlp", "--tier", "满数据8500"],
                   cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
msg = (r.stdout + r.stderr)
hit = "属于合并档案" in msg
print(f"  rc={r.returncode}  拦下={hit}")
print("  " + msg.strip().splitlines()[-1][:110])
if not hit:
    fails.append("[4] MLP × 满数据 没被合并档案闸拦下 ⇒ 新登记的空后缀把两档共本的历史炸开了")
elif r.returncode == 0:
    fails.append("[4] 拦下了却退 0：拒绝必须是红点，退 0 等于放它去训练")

print("\n" + ("[预检] 全绿 ⇒ 这一段的训练可以开烧（下一段开跑前复跑本件，断言两种盘态都成立）"
              if not fails else "[预检] 有红，先别训练："))
for f in fails:
    print("  ✗ " + f)
raise SystemExit(1 if fails else 0)
