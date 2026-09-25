# -*- coding: utf-8 -*-
"""第4课j：把 α·‖∇(T²KL)‖ 钉成常数、只换温度——还第4课i 自己写在 `limitations` 里的那笔明码债。

第4课i（零训练）把 α=0.05 那一格的跨列差（T=6 比 T=3 多错 28.1 张）拆成两条候选通道，判的是
**出口 (b)：大头在内容通道**。但它的判法是**排除法**：拿第 0 批实测的 ‖∇(T²KL)‖ 算出
α*=0.080（把 T=6@0.05 的力度钉回 T=3 所需的 α），再用 T=3 阶梯夹住 α* 那一段的**局部斜率**折算成
share = −5.1 张 = 跨列差的 18%。那句债写得很直白："内容通道本课只量得出自己动了多少，量不出换算成几张
（单位是 nat 不是张），把它变成测量要一次训练（把 α·‖∇(T²KL)‖ 钉死、只换 T；阶梯 5 格 × 8 条流 =
40 次训练）。" **这一发就是那一次训练。**

做法：α 与 T 在 loss 里不正交（T² 补偿不精确，第4课c 已实测在册），但**单位 α 下的 KL 梯度范数
是 α 的线性系数**（‖∇(α·T²KL)‖ = α·‖∇(T²KL)‖，α 是纯标量前置），所以"钉死力度"这件事有闭式解：
T=3 阶梯上每一格 α 配一个 T=6 那一列的等力度剂量

        α'(α) = α · ‖∇(T²·KL)‖_T=3 / ‖∇(T²·KL)‖_T=6 = α × 1.41276 / 2.26708 = α × 0.6232

于是"同力度、只换温度"这一对比较里，权重通道被**逐格构造对齐**（不再是折算），剩下的差只能来自软标签
的形状。两列的每一格共用同一粒种子的 CE 臂 ⇒ 配对差里 CE 锚自动抵消，连减数都不需要重训。

  问句（只有一句）：**把 KD 项推参数的力度钉成同一个数之后，T=6 那一列还比 T=3 多错吗？**

闸（全部在第一次训练之前，任一红 ⇒ 退出、零落盘、不入库）：
  闸 I（读账复算）：两本在册阶梯账（T=3 / T=6，同 8 条流）每格均值必须从它自己的逐流 gap 现算命中，
      **并且**每格的逐流 gap 必须等于那本账 `arms` 里 `wrong_ce − wrong_kd` 的重排——因为这一发的
      配对差直接拿 `wrong_kd` 当货币（CE 抵消），arms 与 ladder_summary 不一致就是这本账自己打架。
  闸 II（配对前提）：两本账流名单逐位同序、档/宽度/学生族/种子底/闸格/锚点剂量/teacher_acc 全同、
      温度列必须不同、五格 α 必须两本都有。
  闸 III（CE 减数 + 老师身份）：两本账每粒种子的 `wrong_ce` 逐位相同；老师权重现算测试准确率必须
      等于两本账自报值 = 板块常量 0.9798。（沿用第4课i 闸 I 的理由：减数不唯一，差里掺 CE 自己的漂移。）
  闸 IV（**这把新尺子的接线闸**，也是最要紧的一道）：本脚本现算的第 0 批 ‖∇CE‖ / ‖∇(T²KL)‖（T=3、T=6）
      必须与 `compress_report_v4c_mid.json` 的 `gradient_probe.by_T` **且**与
      `compress_report_v4i_mechanism_v2.json` 的 `probe_by_T[*]["batch0"]` 逐位相同；
      并且由同一对梯度重算的顶层配平 α 必须等于第4课c 在册那个数（α=0.7 → **0.436**）。
      为什么这一道归本发独有：α' 全由这两个范数算出来，尺子错一格 = 五个剂量全错，
      而那五格正是本发唯一的新原料。第4课c 在册的 `alpha_to_match_anchor` 恰好就是本发阶梯顶那一格
      的 α' ⇒ **免费拿到一条"新算式复现在册配平数"的对账线**（不是新造的常数）。
  闸 V（跨脚本逐位，各 3 条老流）：本脚本重训 (α=0.3, T=3) 与 (α=0.3, T=6) 两格，`wrong_kd` 必须
      分别逐位命中两本账的对应臂。**这是本发的"恒等闸"**：配对差一头是盘上新训的、另一头是从档案读的，
      没有这道闸就没法说这两头出自同一台训练器。（恒等闸 α=0 不适用——阶梯里没有 0 那一格，
      判据也不吃它；接线证明由这道逐位闸承担，它比 α=0 强：它核的是 KD 分支。）

出口（预登记，跑之前写死，此后只许追加"落地"）：主格 = **α=0.05 那一格**（第4课i 的 α* 住在那里、
倒挂也住在那一列），其余四格是"这个残余是不是只有一格才出得来"的对照。承重口径沿用板块那条：
逐流配对差的 95% CI 不含 0（第二十发已证它与"出界幅度 > 半宽"逐字等价）。符号：d = wrong_kd(T3@α) −
wrong_kd(T6@α') ⇒ **d<0 = 同力度下 T=6 多错**。
  (a) 主格 CI 含 0 **且**五格全含 0 ⇒ 力度解释吃下全部跨列差，第4课i 的出口 (b) 被训练侧**否证**。
      这一支必须当**事故**处理并当场写明矛盾在哪：探针（初始一步、第 0 批）与 600 步训练对"有效剂量"
      给出不同答案 ⇒ 要改的是探针那句"权重通道"，不是这一发的数。
  (b) 主格 CI 不含 0 且 d<0 ⇒ 内容通道**第一次有张数**：报"同力度下 T=6 仍多错 |d| 张（CI、同号数）"。
      与第4课i 那句 share=−5.1 张并排读，但**不许相减当分解**（α 曲线非单调，见自我审计② 那条）。
  (c) 任一格 CI 不含 0 而 d>0（同力度下 T=6 反而更浅）⇒ 力度这一维被**高估**：那个范数不是 600 步的
      有效剂量（第4课c 自我审计④ 那条"起手配平不是全程配平"兑现成读数）⇒ 本发的归因作废方向是**探针**，
      不许读成"内容通道有益"。
  (d) 主格含 0 但其余格有不含 0 的（混合）⇒ 逐格不一致 = "匹配强度"这个概念在剂量上不守恒。
      这一支**不判哪个通道**（三分支里没有"部分承重"这一格位置），只登记成下一版判据的债；
      描述性读数不进判词，是板块老规矩。
零 API 费、纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe -u step4j_matched_strength.py --nseeds 3   # 预演，账验完即删、不入库
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe -u step4j_matched_strength.py --nseeds 8   # 真跑
  # 拆红（都在训练之前退出，零成本）：
  ... --nseeds 8 --probeblind    # 不许 backward，改用教科书假设「T² 补偿精确 ⇒ α'=α」⇒ 只有闸 IV 该红
  ... --nseeds 8 --scramble      # 给 T=6 那本账每格 gap +1 ⇒ 只有闸 I 该红

**价签（逐项列加法，加数全部进和——第十五发那条口径的第二次兑现）**：
  靶子 5 格 × 8 条流 = **40 次**（第4课i 欠的那个数）
  + 闸 V 的 T=3 格 3 条老流 = **3 次**
  + 闸 V 的 T=6 格 3 条老流 = **3 次**
  = 真跑 **46 次**；预演本（3 条流）另计 15 + 3 + 3 = **21 次**，验完即删。
  比菜单上那句"40 次"多 6 次，多出来的全是跨脚本逐位闸，且它们在最贵的一步之前就要么红要么绿。

维度账（本发买的是哪一种"买"）：**覆盖**——同一批 8 条流、同一档、同一副学生、同一个老师，
只把 α 搬到五个从未训过的坐标（0.0312/0.0623/0.1246/0.1870/0.4362）。它**不新增独立性维度**
（七维一维都没换）、**不买分辨率**（流数没动），所以出口 (b) 若兑现只能写成
"机制归因第一次有了直接读数"，**不许写成"内容通道被复证"**。它买的也不是判据版本（板块判据 v5 一字不动，
本发根本不进形状判据——判的是另一条量：同力度跨温度的配对差）。

落地（跑完之后追加，上面预登记原文一个字不动）：
  ① 五道闸全绿，两档拆红各自只红自己那道（**都在第一次训练之前退出，各 0 次训练**，RC 均为 1）：
     `--probeblind` 红在**闸 IV**（改用教科书"T² 补偿精确"推定时，T=6 的 ‖∇(T²KL)‖ 报 1.41276
     而 `v4c_mid`/`v4i_mechanism_v2` 两本基准账记 2.26708）；`--scramble` 红在**闸 I**
     （α=0.05 格均值现算 −72.8 ≠ 档案 −73.8）。
  ② **预演本抓到两个 bug，都崩在任何一次训练之前**（崩≠红，所以它们不算闸的功劳，但也不算白跑）：
     ⑴ 常量名拼错：六处 `RUNS` 应为 `RUNGS` ⇒ `NameError`，死在"顶层配平 α 对账"那一行、第一次 backward 之前。
     ⑵ **这一发最该记的一条**：`g3, g6 = gate_strength_map(...)` 解包了一个**按温度索引的字典** ⇒
        拿到的是键 `3.0/6.0` 而不是那两个范数，于是阶梯顶的等力度剂量算成 **0.350 而不是在册的 0.436**。
        它没有红在任何一道"设计来挡它"的闸上——拦住它的是**那条免费的"新算式复现在册配平数"对账线**
        （闸 IV 第二段）。若第4课c 没在册记过 0.436，五个剂量会全体偏移同一个常数因子，
        而下面每一道闸、每一条 CI、每一个同号数**照样全绿**：错的是剂量本身，没有一处能自证。
        可迁移形式：**能给一条新算式找到"复现某个在册数字"的自由对账线，就别省。**
  ③ 货币方向自纠（在真跑之前现读抓出来的，与第4课i 那条"散文方向反"同类，这次是**算式**方向反）：
     未配平对照原写成 `gap(T3) − gap(T6)`，与本发主表 `d = wrong_kd(T3) − wrong_kd(T6@α')` 符号相反 ⇒
     同一张表两行两种货币，读者会把"−32.9 vs +28.1"读成符号翻转。改成 `hi − lo` 后 α=0.05 那行读出
     **−28.1 张**，与第4课h/第4课i 那颗钉子**逐位相同** ⇒ 这条对照顺带成了第二次跨脚本对账（零成本）。
  ④ 读数（8 条流、46 次训练、产物 `compress_report_v4j_matched_n8.json`、实测 43 秒）：
     等力度比 g3/g6 = **0.62316** ⇒ 五格剂量 0.03116 / 0.06232 / 0.12463 / 0.18695 / 0.43621
     （顶格逐位复现在册 0.436）。**出口走 (b)**：主格 α=0.05（T=6 那一头是 α'=0.0312）
     d = **−32.9 张、CI [−42.3, −23.5]、8/8 同号、二项 p=0.0078** ⇒ **内容通道第一次有张数**：
     把 KD 项推参数的力度钉成同一个数、并且给 T=6 那一头**更低**的 α（0.0312 < 0.05）之后，
     T=6 仍然多错 33 张。α=0.1 那格也不含 0（−19.4 张、7/8、p=0.0703）⇒ 承重格数 2/5；
     α≥0.2 三格全含 0（−1.2 / +2.0 / −3.5 张）⇒ **残余只住在小剂量端**，这与第4课i 那个 α*=0.080
     住在的位置一致（但那一致是并排读出来的，不是本发的判据）。接线：闸 V 6/6 逐位命中两本账
     （794/797、795/786、821/820）。
  ⑤ **一条描述性读数，不进判词、而且它欠一次配对**：问"配平这一步换掉了多少"，点估计是**反号**的——
     未配平 −28.1 张 → 等力度 −32.9 张，按第4课i 的折算该收回 +5.1 张，实测却往深里走了 4.8 张。
     但这两行相减**不是**一个配对统计量（该配的是同一列上 `wrong_kd(T6@0.0312) − wrong_kd(T6@0.05)`），
     本发没为它算 CI，而两条 CI 大幅重叠（[−42.3,−23.5] vs [−42.3,−14.0]）⇒ 这句话只到**未判定**。
     欠账（配方已写死、**下一发零次训练**）：把本档 `matched_ladder["α=0.05"]` 每条记录的 `wrong_kd`
     与 `v4f_mid_n8_t6_v5.json` 的 `arms["α0.05 种子<seed>"]["wrong_kd"]` 按同 8 条流配对，
     报那一格的"配平换掉了多少张"（两头都是 T=6 列、同一批流、闸 V 已证同一台训练器）。
  ⑥ 边界（`limitations` 里已入库，这里只重复最要紧两条）：**起手配平不是全程配平**——‖∇(T²KL)‖ 是
     第 0 批、初始那一步的量，600 步里两项相对尺度会变（出口 (c) 那一支正是为这条边界留的，本次没走到）；
     以及坐标账：本发只买**覆盖**，七维一维未换 ⇒ "内容通道被复证"一个字都不能说。
  ⑦ **成本自纠（价签那句"预演 21 次"少算了一倍）**：预演实跑四趟——前两趟在第一次训练之前就崩/红
     （②那两条，各 0 次），第三趟绿（21 次）；③ 那处货币修正之后**又**完整预演了一趟（再 21 次，
     账验完即删、未入库）。⇒ 整链实付 **88 次 = 预演 42 + 真跑 46**，零 API 费。
     规规矩矩写全的原因还是那条：成本句漏加的数都是事后才发现的。
  ⑧ 债的状态：`compress_report_v4i_mechanism_v2.json` 的 `limitations[2]` 那句"把它变成测量要一次训练
     …40 次训练"从此**已付**。入库不改，那句话原样留在老档里，付账凭据是本档与第4课j 这本账。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (DATA_TIERS, FLOW_SEEDS, SEEDS, TEACHER_ARCHIVE_ACC, VAL_N,
                           check_flow_constants, logits_of, summarize, train_student)
from step4c_alpha_T_sweep import term_grad_norms
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
TIER, WIDTH, ARCH = "中数据4500", 128, "mlp"
T_LO, T_HI = 3.0, 6.0
RUNGS = (0.05, 0.1, 0.2, 0.3, 0.7)          # T=3 那一列在册阶梯的五格（0 是恒等闸，不是臂）
PRIMARY = 0.05                              # 主格：第4课i 的 α* 与倒挂都住在这附近
GATE_ALPHA = 0.3                            # 闸 V 重训那一格（两本账都有、且是老 3 条流可对账的格）
L3 = "compress_report_v4f_mid_n8_v5.json"      # T=3 列的阶梯（配对差另一头的来源）
L6 = "compress_report_v4f_mid_n8_t6_v5.json"   # T=6 列同阶梯（未配平的对照，本发只读）
V4C = "compress_report_v4c_mid.json"           # 探针在册基准 + alpha_to_match_anchor
V4I = "compress_report_v4i_mechanism_v2.json"  # 这笔债的原文所在


def out_name(nseeds: int) -> Path:
    """流数是本发的判别维度：不带 _n8 会让 3 条流那本预演账撞掉真跑那本。"""
    return HERE / f"compress_report_v4j_matched{'' if nseeds == 3 else f'_n{nseeds}'}.json"


# ---------- 读账（三本在册账；本发只新训 T=6 那一列的五个等力度格） ----------
def load_ladder(name: str, scramble: bool = False) -> dict:
    """一本阶梯账 → 每格逐流 gap / wrong_kd / 每粒种子的 CE 错图数。

    为什么把 wrong_kd 也读出来：本发的配对差以错图数为货币（CE 在同一条流内相减自动抵消），
    而 gap 只是它的派生量——两道复算（均值命中 + gap == wrong_ce − wrong_kd）都过才敢说这两个词指同一批数。
    """
    d = json.loads((HERE / name).read_text(encoding="utf-8"))
    seeds = [int(s) for s in d["seeds"]]
    arch = str(d.get("student_arch", "mlp"))
    cells, wrongs, armgap = {}, {}, {}
    for a, v in d["ladder_summary"].items():
        cells[float(a)] = dict(zip(seeds, [int(g) for g in v["gaps_imgs"]]))
    for k, row in d["arms"].items():
        af = float(k.split(" 种子")[0][1:])
        wrongs.setdefault(af, {})[int(row["seed"])] = int(row["wrong_kd"])
        armgap.setdefault(af, {})[int(row["seed"])] = int(row["wrong_ce"]) - int(row["wrong_kd"])
    if scramble:
        cells = {a: {s: g + 1 for s, g in c.items()} for a, c in cells.items()}
    ce: dict[int, set] = {}
    for row in d["arms"].values():
        ce.setdefault(int(row["seed"]), set()).add(int(row["wrong_ce"]))
    for sd, vals in sorted(ce.items()):
        if len(vals) > 1:
            raise SystemExit(f"{name}：种子 {sd} 的 CE 错图数有 {sorted(vals)} 个取值 ⇒ 减数不唯一")
    return {"name": name, "seeds": seeds, "arch": arch, "tier": d["tier"], "width": int(d["width"]),
            "t": float(d["temps_fixed"]), "floor": int(d["seed_floor_imgs"]),
            "teacher_acc": float(d["teacher_acc"]), "probe_alpha": float(d["probe_alpha"]),
            "anchor_alpha": float(d["anchor_alpha"]), "cells": cells, "wrongs": wrongs, "armgap": armgap,
            "means": {float(a): float(v["mean_gap_imgs"]) for a, v in d["ladder_summary"].items()},
            "ce": {sd: next(iter(v)) for sd, v in ce.items()}}


def gate_recalc(b: dict) -> None:
    """闸 I：每格均值现算命中 + 每格逐流 gap == arms 里的 wrong_ce − wrong_kd。"""
    for a, want in sorted(b["means"].items()):
        got = round(sum(b["cells"][a].values()) / len(b["cells"][a]), 1)
        if got != want:
            raise SystemExit(f"{b['name']}：α={a:g} 格均值现算 {got} ≠ 档案记的 {want} ⇒ 读错账")
        for sd in b["seeds"]:
            if b["armgap"][a][sd] != b["cells"][a][sd]:
                raise SystemExit(f"{b['name']}：α={a:g}/种子{sd} 的 ladder gap {b['cells'][a][sd]} ≠ "
                                 f"arms 里的 wrong_ce−wrong_kd {b['armgap'][a][sd]}"
                                 f" ⇒ 这本账的两种记法不是同一批数，本发的货币不成立")


def gate_pairing(lo: dict, hi: dict) -> None:
    """闸 II：两本账只许差一个温度列，且五格 α 都在。"""
    if lo["seeds"] != hi["seeds"]:
        raise SystemExit("两本阶梯账的流名单不同序/不同成员 ⇒ 逐流配对没有立足点")
    same = {k: (lo[k], hi[k]) for k in ("tier", "width", "arch", "floor", "teacher_acc",
                                        "probe_alpha", "anchor_alpha") if lo[k] != hi[k]}
    if same:
        raise SystemExit(f"这两本不是同一格只换温度列（{same}）⇒ 判别变量没对齐，差值不属于任何假说")
    if lo["t"] == hi["t"]:
        raise SystemExit(f"两本账温度列都是 T={lo['t']:g} ⇒ 跨列差恒为 0")
    for a in RUNGS:
        if a not in lo["cells"] or a not in hi["cells"]:
            raise SystemExit(f"α={a:g} 不是两本账都有 ⇒ 那一格的等力度配对没有原料")


def gate_ce_identity(lo: dict, hi: dict, teacher_acc: float) -> dict:
    """闸 III：老师身份 + 两列 CE 减数逐位相同。"""
    want = {lo["teacher_acc"], hi["teacher_acc"]}
    if len(want) != 1:
        raise SystemExit(f"两本账自报 teacher_acc 不一致（{want}）⇒ 不是同一副老师的账")
    got = round(teacher_acc, 4)
    if got != lo["teacher_acc"] or got != round(TEACHER_ARCHIVE_ACC, 4):
        raise SystemExit(f"老师现算测试准确率 {got:.4f} ≠ 两本账记的 {lo['teacher_acc']} / "
                         f"板块常量 {TEACHER_ARCHIVE_ACC} ⇒ 本课引用的账与盘上这副权重不是同一次实验")
    bad = [s for s in lo["seeds"] if lo["ce"].get(s) != hi["ce"].get(s)]
    if bad:
        raise SystemExit(f"CE 臂在流 {bad} 上两列不等 ⇒ 跨列差里掺着 CE 自己的漂移")
    return {"teacher_acc_recomputed": got, "teacher_acc_archives": lo["teacher_acc"],
            "ce_minus_number_checked": f"{len(lo['seeds'])}/{len(lo['seeds'])}"}


def gate_strength_map(g: dict, v4c: dict, v4i: dict, probeblind: bool) -> dict:
    """闸 IV：等力度映射的原料必须命中两本在册账，且顶层配平 α 复现在册那个数。

    `probeblind` 是它的拆红档：不许 backward、改用教科书假设「T² 补偿精确 ⇒ ‖∇‖ 与 T 无关」，
    于是 α' 恒等于 α、权重通道"按声称"整条消失 ⇒ 这一档红钉的是
    **"映射必须由一次实测的梯度算出，不能由 T² 推定"**（第4课i 的 --t2claim 的同族，
    区别是这里它直接决定五个剂量本身）。"""
    for t in (T_LO, T_HI):
        a = v4c["gradient_probe"]["by_T"][f"{t}"]
        if a["g_ce"] != v4i["probe_by_T"][f"{t}"]["batch0"]["g_ce"] \
                or a["g_kd_unit"] != v4i["probe_by_T"][f"{t}"]["batch0"]["g_kd_unit"]:
            raise SystemExit(f"两本基准账自己的探针就不一致（{V4C} vs {V4I}，T={t:g}）⇒ 谁都不能当尺子")
        if g[t]["g_ce"] != a["g_ce"] or g[t]["g_kd_unit"] != a["g_kd_unit"]:
            raise SystemExit(f"本脚本现算的第 0 批探针（T={t:g}）= {g[t]} ≠ {V4C}/{V4I} 记的 "
                             f"‖∇CE‖{a['g_ce']} / ‖∇(T²KL)‖{a['g_kd_unit']}"
                             f"{'（本档正在演示：用教科书 T² 推定代替一次 backward）' if probeblind else ''}"
                             f" ⇒ 这把尺子与产出那两本账的尺子不是同一把，五个剂量全不发")
    return {t: v4c["gradient_probe"]["by_T"][f"{t}"]["g_kd_unit"] for t in (T_LO, T_HI)}


def alpha_map(g3: float, g6: float) -> dict:
    """α'(α) = α · ‖∇(T²KL)‖_3 / ‖∇(T²KL)‖_6；α 是 loss 里 KL 梯度的纯标量前置，所以这是闭式而非近似。"""
    return {a: a * g3 / g6 for a in RUNGS}


def main(nseeds: int = 8, probeblind: bool = False, scramble: bool = False) -> None:
    mode = "probeblind" if probeblind else "scramble" if scramble else ""
    if probeblind:
        print("[验红] --probeblind：不许 backward，按教科书假设「T² 补偿精确」⇒ α'=α，只有闸 IV 该红")
    if scramble:
        print("[验红] --scramble：给 T=6 那本账每格 gap +1，只有闸 I 该红")
    if nseeds not in FLOW_SEEDS:
        raise SystemExit(f"复现流只有 {sorted(FLOW_SEEDS)} 这几档")
    seeds = FLOW_SEEDS[nseeds]
    check_flow_constants(len(seeds))     # 缺系数的 KeyError 会炸在 46 次训练之后（第4课n=12 那次的教训）
    out = out_name(nseeds)
    if out.exists() and not mode:
        raise SystemExit(f"{out.name} 已入库——重跑会盖掉这本判决档案（要重跑得先改名或删档）")

    lo, hi = load_ladder(L3), load_ladder(L6, scramble=scramble)
    gate_recalc(lo)                                              # 闸 I
    gate_recalc(hi)
    gate_pairing(lo, hi)                                         # 闸 II
    v4c = json.loads((HERE / V4C).read_text(encoding="utf-8"))
    v4i = json.loads((HERE / V4I).read_text(encoding="utf-8"))

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    t_acc = float((predict(teacher, xte, False) == yte).mean())
    ident = gate_ce_identity(lo, hi, t_acc)                       # 闸 III

    xd, yd = xt_all[:DATA_TIERS[TIER]], torch.tensor(yt_all[:DATA_TIERS[TIER]])
    soft = logits_of(teacher, xd)
    g = {t: term_grad_norms(xd, yd, soft, t, WIDTH, arch=ARCH) for t in (T_LO, T_HI)}
    if probeblind:
        g[T_HI] = {"g_ce": g[T_HI]["g_ce"], "g_kd_unit": g[T_LO]["g_kd_unit"]}
    smap = gate_strength_map(g, v4c, v4i, probeblind)         # 闸 IV（返回按温度索引的字典）
    g3, g6 = smap[T_LO], smap[T_HI]
    AM = alpha_map(g3, g6)
    top_booked = v4c["gradient_probe"]["by_T"][f"{T_HI}"]["alpha_to_match_anchor"]
    if round(AM[RUNGS[-1]], 3) != top_booked:
        raise SystemExit(f"由现算梯度重算的阶梯顶等力度 α 得 {AM[RUNGS[-1]]:.3f}，"
                         f"而 {V4C} 在册的 alpha_to_match_anchor（T={T_HI:g}）是 {top_booked}"
                         f" ⇒ 本发的映射式与第4课c 的配平臂不是同一个算式，五个剂量全不发")

    print(f"[{TIER} × S{WIDTH} × {ARCH} 族] 五格 × {len(seeds)} 条流 @ T={T_HI:g}（等力度）| "
          f"配对另一头读 {L3}（T={T_LO:g}）")
    print(f"[闸 I] 两本阶梯 {len(lo['cells'])}+{len(hi['cells'])} 格均值现算命中、"
          f"gap 与 arms 的 wrong_ce−wrong_kd 逐格逐流相同")
    print(f"[闸 II] 两本账 {len(seeds)} 条流同序、判别变量全同、只温度列不同")
    print(f"[闸 III] 老师身份：现算 {ident['teacher_acc_recomputed']} = 两本账自报 "
          f"{ident['teacher_acc_archives']} = 板块常量 {TEACHER_ARCHIVE_ACC}；"
          f"CE 减数逐位核过 {ident['ce_minus_number_checked']} 条")
    print(f"[闸 IV] 第 0 批探针逐位命中 {V4C} 与 {V4I}：‖∇(T²KL)‖ T=3 {g3:.5f} / T=6 {g6:.5f}"
          f"（‖∇CE‖ {g[T_LO]['g_ce']:.5f}，与 T 无关）⇒ 等力度比 g3/g6 = {g3 / g6:.4f}")
    print(f"[等力度映射] α'（T=6 上与本发各格同力度）= " +
          "、".join(f"α={a:g}→{AM[a]:.4f}" for a in RUNGS) +
          f"；阶梯顶那一格复现在册配平数 {top_booked}（第4课c 的 match_T6 臂用的就是它）")

    # ---------- 先立预言（此后不许改口） ----------
    print("\n[先立预言（此后不许改口）]")
    print(f"  H_力度（第4课i 的 18% 那句推到极端：跨列差全由 α·‖∇‖ 解释）→ 同力度五格 d≈0，"
          f"CI 应含 0；出口 (a)")
    print(f"  H_内容（第4课i 实际走的出口 (b)：形状本身带伤害）→ 主格 d<0 且 CI 不含 0，"
          f"量级参考：未配平的跨列差是 −28.1 张，权重折算只吃了 −5.1 张 ⇒ 剩下约 −23 张开外")
    print(f"  反向读数（同力度下 T=6 更浅）不是 H_内容 也不是 H_力度：那是探针把'有效剂量'估高了，出口 (c)")

    # ---------- 各臂 ----------
    rows = {a: [] for a in RUNGS}
    gates = []
    print(f"\n[开跑] 每条形如 (T=6, α'=等力度剂量) 一副；CE 锚不重训（配对里自动抵消），"
          f"接线由闸 V 的 {2 * len(SEEDS)} 次逐位对账承担：")
    for seed in seeds:
        line = []
        for a in RUNGS:
            ap = AM[a]
            m, _ = train_student(WIDTH, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=ap, t=T_HI, arch=ARCH)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            rows[a].append({"seed": seed, "alpha_matched": round(ap, 5), "T": T_HI,
                            "wrong_kd": wrong, "wrong_kd_T3": lo["wrongs"][a][seed],
                            "d_imgs": lo["wrongs"][a][seed] - wrong})
            line.append(f"a{a:g}(→{ap:.3f}):{wrong}")
        print(f"  种子 {seed:<5} " + " ".join(line))
        if seed in SEEDS:                       # 只有老 3 条有跨脚本账可核
            for t, ref, book in ((T_LO, lo, L3), (T_HI, hi, L6)):
                m, _ = train_student(WIDTH, xd, yd, soft, "KD", xv, yv, seed=seed,
                                     alpha=GATE_ALPHA, t=t, arch=ARCH)
                p = predict(m, xte, False)
                wrong = int((p != yte).sum())
                want = ref["wrongs"][GATE_ALPHA][seed]
                gates.append({"T": t, "alpha": GATE_ALPHA, "seed": seed, "archive": book,
                              "this_run_wrong_kd": wrong, "archive_wrong_kd": want,
                              "match": bool(wrong == want)})
    bad_gate = [x for x in gates if not x["match"]]
    if not gates:
        raise SystemExit(f"闸 V 一条记录都没有（老 3 条流 × 两列 = {2 * len(SEEDS)} 次）：空表『全过』不是通过")
    print(f"\n[闸 V 跨脚本逐位 (α={GATE_ALPHA:g})] {len(gates) - len(bad_gate)}/{len(gates)} 命中 "
          f"T=3 读 {L3}、T=6 读 {L6}"
          + ("" if not bad_gate else f" ⇒ **不通过：{bad_gate}**——本脚本与那两本账不是同一台训练器，"
                                     f"一头新训一头读档的配对差全部作废"))
    if bad_gate or mode:
        raise SystemExit(f"[{'验红模式 ' + mode if mode else '闸 V 红'}] 走到了这里就不发读数、不入库")

    # ---------- 判决 ----------
    diffs = {}
    print(f"\n[同力度跨温度·逐流配对差 d = 错图数(T3@α) − 错图数(T6@α')] 负 = 同力度下 T=6 多错")
    for a in RUNGS:
        d = [r["d_imgs"] for r in rows[a]]
        s = summarize(d, f"α={a:g} 等力度配对差（T3 − T6）")
        crosses = bool(s["ci95_imgs"][0] < 0 < s["ci95_imgs"][1])
        diffs[a] = {"alpha_matched_T6": round(AM[a], 5), "paired": s, "ci_crosses_zero": crosses}
        print(f"  α={a:<5}（α'={AM[a]:.4f}） d {d} 均值 {s['mean_gap_imgs']:+6.1f} 张 "
              f"CI [{s['ci95_imgs'][0]:+.1f}, {s['ci95_imgs'][1]:+.1f}] σ {s['sd_imgs']} "
              f"同号 {s['same_sign_seeds']} p={s['p_seeds_binomial']} ⇒ "
              + ("CI 含 0：这一格同力度下两列没分开" if crosses else
                 ("CI 不含 0 且 d<0：同力度下 T=6 仍多错" if s["mean_gap_imgs"] < 0 else
                  "CI 不含 0 且 **d>0**：同力度下 T=6 反而更浅 ⇒ 出口 (c)，探针被高估")))
    # 未配平那一头（零成本，从两本在册账现读）：给出"配平这一步换掉了多少"的对照
    # 货币与上面的 d 一致：gap(T6) − gap(T3) = wrong_kd(T3) − wrong_kd(T6)，负 = T=6 多错
    unbal = {}
    for a in RUNGS:
        s = summarize([hi["cells"][a][sd] - lo["cells"][a][sd] for sd in seeds], f"α={a:g} 未配平跨列差")
        unbal[a] = s
        print(f"  对照·未配平（固定 α、只换 T）α={a:<5} 差 {s['mean_gap_imgs']:+6.1f} 张 "
              f"CI [{s['ci95_imgs'][0]:+.1f}, {s['ci95_imgs'][1]:+.1f}] 同号 {s['same_sign_seeds']}")

    prim = diffs[PRIMARY]
    nonnull = [a for a, v in diffs.items() if not v["ci_crosses_zero"]]
    neg = [a for a in nonnull if diffs[a]["paired"]["mean_gap_imgs"] < 0]
    pos = [a for a in nonnull if diffs[a]["paired"]["mean_gap_imgs"] > 0]
    if pos:
        exit_rule = (f"(c) 探针被高估：{pos} 这些格同力度下 T=6 反而更浅（CI 不含 0）。"
                     f"⇒ 第 0 批那个梯度范数不是 600 步训练的有效剂量（第4课c 自我审计④ 那条"
                     f"'起手配平不是全程配平'兑现成读数），本发的归因作废方向是**探针**，"
                     f"不许读成『内容通道有益』，也不许拿它否掉第4课i 的出口 (b)")
    elif prim["ci_crosses_zero"] and not nonnull:
        exit_rule = (f"(a) 五格全部 CI 含 0 ⇒ 力度解释吃下全部跨列差，第4课i 的出口 (b)"
                     f"（大头在内容通道）被训练侧**否证**。这是**事故**不是发现：矛盾点在"
                     f"「初始一步、第 0 批的 ‖∇‖」与「600 步之后的结果」之间，"
                     f"要改的是探针那句『权重通道吃 18%』的换算，不是这一发的数")
    elif not prim["ci_crosses_zero"] and PRIMARY in neg:
        p = prim["paired"]
        exit_rule = (f"(b) 内容通道第一次有张数：把 KD 项力度钉成同一个数之后，主格 α={PRIMARY:g}"
                     f"（T=6 那一头是 α'={prim['alpha_matched_T6']:.4f}）T=6 仍多错 "
                     f"{abs(p['mean_gap_imgs']):.1f} 张（CI [{p['ci95_imgs'][0]:+.1f},"
                     f" {p['ci95_imgs'][1]:+.1f}]、同号 {p['same_sign_seeds']}、二项 p={p['p_seeds_binomial']}）"
                     f"⇒ 第4课i 那句『排除法给的归因』从此有了一次**直接测量**；"
                     f"承重的格数 {len(neg)}/{len(RUNGS)}（其余格见上表，不合并成一个数）")
    else:
        exit_rule = (f"(d) 混合读数：主格 CI 含 0 而 {nonnull} 不含 ⇒ 『匹配强度』这个概念在剂量上不守恒。"
                     f"本发的三分支里没有『部分承重』这一格位置 ⇒ 不判哪个通道，"
                     f"登记成下一版判据的债（描述性读数不进判词是老规矩）")
    print(f"\n[出口（预登记）] {exit_rule}")

    report = {"quantity": "同力度（α·‖∇(T²KL)‖ 相等）下 T=6 列与 T=3 列的逐流配对错图数之差，"
                          "五格剂量各一条（负 = 同力度下 T=6 多错）",
              "pays": f"{V4I} 的 limitations 第 3 条（『把它变成测量要一次训练…40 次训练』）",
              "tier": TIER, "width": WIDTH, "student_arch": ARCH, "seeds": list(seeds),
              "temps": [T_LO, T_HI], "rungs": list(RUNGS), "primary_rung": PRIMARY,
              "strength_map": {"g_kd_unit_T3": g3, "g_kd_unit_T6": g6,
                               "ratio_g3_over_g6": round(g3 / g6, 5),
                               "alpha_matched": {str(a): round(AM[a], 5) for a in RUNGS},
                               "parity_against_v4c_anchor": top_booked},
              "archives": {"T3_ladder": L3, "T6_ladder": L6, "probe_baseline": V4C, "debt_source": V4I},
              "gates": {**ident, "recalc_cells": f"{len(lo['cells'])}+{len(hi['cells'])}",
                        "pairing": "同 8 条流同序、判别变量全同、只温度列不同",
                        "probe_batch0_bitwise": True,
                        "cross_script": {"n": len(gates), "all_match": True, "detail": gates}},
              "predictions_ex_ante": {"H_strength_d_is_zero": "五格 CI 全含 0",
                                      "H_content_d_negative": "主格 d<0 且 CI 不含 0",
                                      "over_matched_probe_d_positive": "任一格 d>0 且 CI 不含 0"},
              "matched_ladder": {f"α={a:g}": rows[a] for a in RUNGS},
              "paired_diff": {f"α={a:g}": diffs[a] for a in RUNGS},
              "unmatched_cross_column_diff_read_from_books": {f"α={a:g}": unbal[a] for a in RUNGS},
              "exit": exit_rule, "verdict": exit_rule,
              "cost": {"arm_trainings": len(RUNGS) * len(seeds),
                       "cross_script_gate_trainings": len(gates),
                       "ce_trainings": 0,
                       "total": len(RUNGS) * len(seeds) + len(gates),
                       "api_fee": 0,
                       "note": "CE 锚不重训：配对差 d = wrong_kd(T3) − wrong_kd(T6)，同一条流内 CE 抵消"
                               "（与第4课c ③b 那条'CE 锚自动抵消'同一种读法）"},
              "limitations": [
                  f"配平是**起手配平**，不是全程配平：‖∇(T²KL)‖ 在第 0 批（{WIDTH} 张）、初始那一步测，"
                  "训练 600 步里两项的相对尺度会随学生逼近老师而变（与第4课c 自我审计④ 同一条）。"
                  "所以『同力度』指的是初始那一步的系数对齐，不是一个被全程守住的量——"
                  "出口 (c) 那一支正是这条边界的兑现方式，本发不假装它不存在",
                  f"这一发买的是**覆盖**（{len(RUNGS)} 个从未训过的 α 坐标 × 同一批 {len(seeds)} 条流）："
                  "七维独立性一维都没换、流数没动 ⇒ 出口 (b) 只写成『第一次直接读数』，"
                  "不许写成『内容通道被复证』",
                  "五格**不许合并成一个数**：α 曲线非单调（第4课i 自我审计那句），"
                  "所以没有一个"
                  "『平均同力度残余』是合法的机制量；承重的是主格那一格，其余四格是它的对照组",
                  "跨列差的换算不许做减法分解：|d| 与第4课i 的 share=−5.1 张量的是不同的东西"
                  "（一个是训练结果之差、一个是初始斜率×Δα），本发只把它们并排放，不相减",
                  f"读数只属于**一个坐标**（{TIER} × S{WIDTH} × {ARCH} 族 × 那 {len(seeds)} 条流）："
                  "满数据档/低数据档、S64、Fashion、CNN 学生各自要各自的等力度映射（g3/g6 是按坐标现算的，"
                  "第4课c 在 CNN 族上读到的是另一组数），本发一个字都没碰",
                  "判据版本不涉及：本发不进形状判据 v5（它判的是 α 阶梯的形状），"
                  "这一发的量是同力度跨温度的配对差 ⇒ 不改任何在册判词，也不进守卫的翻转集合参评对象",
                  f"多重比较：{len(RUNGS)} 格 × {len(seeds)} 条流，主格是**跑之前**点名的（α=0.05），"
                  "其余四格按对照读，不许事后从五格里挑最好看的那格当出口"],
              }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{report['cost']['total']} 次训练 = "
          f"靶子 {report['cost']['arm_trainings']} + 闸 V {report['cost']['cross_script_gate_trainings']}，"
          f"10000 张全量测试，纯 CPU，零 API 费）· {exit_rule[:60]}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="把 α·‖∇(T²KL)‖ 钉成常数、只换温度：等力度阶梯配平实验")
    ap.add_argument("--nseeds", type=int, choices=(3, 8), default=8,
                    help="3 = 预演/对账那批老流（账验完即删，不入库）；8 = 板块共用那批（真跑）")
    ap.add_argument("--probeblind", action="store_true",
                    help="拆红：不许 backward，改用教科书假设「T² 补偿精确 ⇒ α'=α」⇒ 只有闸 IV 该红")
    ap.add_argument("--scramble", action="store_true",
                    help="拆红：给 T=6 那本账每格 gap +1 ⇒ 只有闸 I 该红")
    a = ap.parse_args()
    main(nseeds=a.nseeds, probeblind=a.probeblind, scramble=a.scramble)
