# -*- coding: utf-8 -*-
"""第4课i：α→0 那一格的跨列之差，机制上能拆掉多少（**零训练、零 API 费**）

先交一条本课查出来的**散文方向错**（数字全对，话写反了）：第4课h 的档案字段 `quantity` 与它
docstring 落地④ 都写着"α=0.05 跨列差 −28.1 张 ⇒ T=6 那一列在最浅的一格**少错** 28 张"。方向是反的：
gap 的定义是 `wrong_ce − wrong_kd`（第4课 起就是这个符号约定，负=KD 多错），跨列差 = T6 的 gap
减 T3 的 gap，所以**负号=T=6 那一格多错 28 张**。本课把两本账的逐流 gap 与第4课h 的读数逐位复算过
（闸 III），错的只是"把差值翻回错图数"时的那句话，所以修的是**散文与字段标签**、不是任何数字。
本课的问句按修正后的方向写：**为什么 T=6 那一列在最浅的一格（α=0.05）比 T=3 多错 28 张？**

为什么这件事零训练就能动一部分（分解的穷尽性是算出来的，不是声称的）：α 固定时
        loss = (1−α)·CE + α·T²·KL(T)
第一项的权重 (1−α) **与 T 无关** ⇒ 在"同一条流池、同一档、同一副学生、同一个 α，只换温度列"这一对
比较里，"CE 那半的权重变了"这一支被构造排除。剩下的只有两处：
  ① **权重通道**：KL 项真正推参数的力度 = α·‖∇(T²·KL)‖。第4课c 已经在册量过它随 T 漂
     （T² 补偿不精确——同一本 `compress_report_v4c_mid.json` 的探针就是这一发的对账基准），
     所以固定 α 换 T **顺带换了权重**；
  ② **内容通道**：软标签的形状。argmax 与 T 无关（除以正数保序），变的只有 top1/第二名的配比。
⇒ 这一发的问句是**二分且穷尽**的：那 28 张要么能由"权重被 T 顺带抬上去"复现，要么不能。
   后者的意思是它住在内容通道——而内容通道本课**只量得出自己动了多少，量不出换算成几张**：
   把它变成测量要一次训练（把 α·‖∇(T²KL)‖ 钉死、只换 T），价签写在 `limitations`。

四道检查先于读数成立（能被拆红的才叫闸）：
  闸 I（老师身份 + 两列共用同一个减数）：老师权重现算的测试集准确率必须等于**两本账各自记着的**
      `teacher_acc`（也等于板块那个常量 0.9798——那个常量此前只在第4课训练前 assert 过一次，
      这里把它变成"引用这两本账之前必须再核一次"）；再核两本账每一粒种子的 `wrong_ce` **逐位相同**
      ——跨列差是拿 gap 相减的，如果两列的减数不是同一个数，那 28 张里就掺着 CE 臂自己的漂移。
  闸 II（探针与体检复算对账在册基准）：本脚本的零训练探针在第 0 批上必须与 `compress_report_v4c_mid.json`
      的 `gradient_probe.by_T` **逐位**相同（‖∇CE‖、‖∇(T²KL)‖ 三个温度各两项，外加由它们重算的
      `alpha_to_match_anchor`），软标签体检四个量 × 三个温度也必须与同档同宽的
      `soft_label_audit_by_T` 逐位相同。**这一条是探针的接线闸**：它不通过，说明本脚本读的
      不是那 4500 张、或不是那副老师、或初始权重/批次与第4课c 不同——那后面每一个"机制量"都是凭空的。
      复用第4课c 那支 `term_grad_norms` 本体，不重写一份（重写一份就等于养第二把尺子）。
  闸 III（阶梯复算 + 与第4课h 的钉子对账）：两本阶梯账的流名单必须**逐位同序**（配对的立足点），
      档位/宽度/温度列/种子底/探针格/锚点格这些判别变量必须全同；每一格均值必须从它自己的逐流 gap
      现算命中（八条流八个数）；跨列 α=0.05 那条差的 均值/CI/σ/同号/二项 p/**逐流序列**必须与
      `compress_report_v4h_crosscol_v6.json` 记着的读数逐位相同。**钉子是"与账对账"，不是预测。**
  闸 IV（两条通道至少一条必须真的动）：权重通道开没开 = α*（把 T=6 那一格的 KD 力度钉回 T=3 所需的 α）
      与 0.05 的差是否超出批次散布；内容通道开没开 = KL(p(T3)‖p(T6)) 与其变动图数是否大于 0。
      **两条都关 ⇒ 当场退出**：那时这一发没有判别对象，打印出来的"归因"是装饰。

拆红（每档只钉一道闸，跑不出红的模式＝那道闸没测过）：
  --teacherblind  把老师 checkpoint 换成 Fashion 那副（同一条读档代码、同一批图）⇒ 只有闸 I 该红
  --t2claim       不许 backward，改用教科书假设"T² 补偿精确 ⇒ ‖∇(T²KL)‖ 与 T 无关"⇒ 闸 II 该红，
                  而且它顺带把 α* 从 0.080 塌回 0.050（权重通道"按声称"直接关闭）⇒
                  这一档钉的是**"权重图必须来自一次实测的梯度，不能由 T² 推定"**
  --scramble      给 T=6 那一列所有 gap +1（模拟取错账/单位）⇒ 只有闸 III 该红

预登记的出口（跑之前写死，此后不许改口）：
  ① 四道检查全过才发读数；任一红 ⇒ 退出、不入库。
  ② 权重通道吃下的份额用**局部斜率**算，不用事后挑的格子：在 T=3 阶梯上找出夹住 α* 的那一段
     [a_lo, a_hi]，slope = (均值(a_hi) − 均值(a_lo))/(a_hi − a_lo)，share = slope × (α* − 0.05)。
     这一段自己的配对步长 CI 一并打印（它就是 share 的不确定度来源；跨 0 时 share 只能算派生估计）。
  ③ 判据（阈值 0.5 的理由：**一半**是"这条通道算不算主因"的最弱非平凡门槛，低于它连主因都算不上）：
     (a) |share| ≥ 0.5·|跨列差| ⇒ 大头在**权重通道**（温度在小剂量端偷偷是条权重轴）；
     (b) 否则 ⇒ 大头在**内容通道**，且要求 T=3 全阶梯**没有一格**能复现那一格的结果（逐格配对差的
         CI 全不含 0）——"权重与分配在 T=3 上穷尽也复现不出"是 (b) 的加强，不是另一件事；
     (c) 若 |share| < 0.5|跨列差| **但** T=3 阶梯上存在一格能复现（配对差 CI 含 0）⇒
         **本探针的权重图指认失败**，这一发的机制读数作废（要回去改探针，不许把 (c) 读成"两种说法都对"）。
  ④ 几何对照（多算的一列，不参与出口判定）：在 10 个在册格里找**总梯度**（两项都按各自权重加起来）
     与目标格相对距离最近的那一格。它给的是"初始那一步、那一个批次上，参数被推的方向与强度是否相同"，
     与 ② 的"结果是否相同"是两条独立的近亲判据——两者一致才敢说"权重通道"不是事后归因。

这一发买的是**判别量**（第四种买法，与第4课h 同族）：一条流都没加、七维里一维都没换、α 阶梯一格没长，
所以它**永不构成复证**——它只是把"多错 28 张"这句话从"只有一个观测事实"变成"两个候选通道各自有数、
并且其中一个当场被判掉"。所有读数都在**同一个坐标**（中数据4500 × S128 × 那 8 条流 × α=0.05 那一格），
把它外推到别的档/别的宽度需要那一维自己的账。

成本（实测）：0 次训练、0 元；探针 = 3 个温度 × 5 个批次各两次 backward + 10 个格各一次总梯度，
老师的软标签 = 4500 张一次前向 + 10000 张测试一次前向。

落地（跑完之后才写，只写实测到的事）：
  ① 四道检查全绿，三档拆红各自只红自己那道（RC 均为 1）：`--teacherblind` 红在闸 I（Fashion 那副老师
     在 MNIST 测试集上现算 0.1053 ≠ 0.9798——这一发把"老师身份"从注释变成行为），`--t2claim` 红在闸 II
     （用教科书 T² 推定时第 0 批的 g_kd_unit(T=1) 报 1.41276 而基准账记 0.48723），`--scramble` 红在闸 III
     （α=0.05 格均值现算 −72.8 ≠ 档案 −73.8）。
  ② **两处在"读到机制量之前"就崩掉的 bug，是这一发落地时抓的**（崩≠红，所以它们不算闸的功劳）：
     ⑴ `gate_probe_vs_archive` 里 `by_t[t]["g_kd_unit"]` 把每温度那个**按批的列表**当字典索 ⇒ TypeError；
     ⑵ `gate_recalc_v4h` 直接对第4课h 那份**按坐标分对**记的档案取顶层键 `cross_column_cellwise`
     ⇒ KeyError。第⑵处修完顺带把"认不出这一对"变成红着退出（按两个文件名认领，命中数≠1 就停），
     否则拿别的坐标的读数当钉子这件事只会崩、不会红。**教训：一条从没跑过正常模式的检查，
     在它是"红"还是"崩"这件事上是未被测过的**——三档拆红各自只走得到自己那一道的退出点，
     崩在后面那道闸上的 bug 拆红档看不见，只有正常模式跑通才暴露。
  ③ 权重通道的实测形状：‖∇(T²KL)‖（第 0 批）T=1 0.48723 / T=3 1.41276 / T=6 2.26708，
     ‖∇CE‖ 0.49343（与 T 无关，同一批同一初值）⇒ 从 T=3 加热到 T=6 时那一项的真实力度乘
     **1.60**，而"教科书 T² 补偿精确"预言乘 1.0。所以 α*=0.080（五批散布 0.0767~0.0802，
     对账用的第 0 批那一个数），比名义 α=0.05 多出 **+0.030 个 α 单位**——固定 α 换温度确实顺带换了权重。
     但把它换算成张数：夹住 α* 的那一段是 α=0.05→0.1，局部斜率 −168.0 张/单位 α ⇒ share = **−5.1 张**
     ＝跨列差 −28.1 张的 **18%**。
  ④ 出口走 **③(b)：大头在内容通道**，而且 (b) 那个"加强"版条件也成立——T=3 全阶梯五格逐格与目标格
     配对，**没有一格**的 CI 含 0（最接近的是 α=0.1：差 −19.8 张 CI [−34.7, −4.8]、7/8 同号），
     其余四格差 −23.9 ~ −35.2 张。所以"把 KD 项的力度在 T=3 上钉到 T=6 那一格的实际水平"这件事，
     连同"在 T=3 那一列上随便挑一格"，都复现不出那 28 张。
  ⑤ 事后核算（**不是新出口，判据 ③ 没用它**，只是把已入库的段内步长 CI 按同一比例折一下）：
     share = 段内步长 × (α*−0.05)/0.05 = ×0.604 ⇒ share 的区间约 **[−8.8, −1.3] 张**，
     而 (a) 的门槛是 |share| ≥ 14.05 张 ⇒ **即便取这段步长 CI 最极端的那一端，权重通道也够不到主因**。
     这一句的可迁移形式：排除法结论要顺手问"它用的那个量自己的不确定度能不能翻掉结论"，
     能翻就要改口，不能翻就把"不能翻"算出来——写在判据里而不是写在嘴上。
  ⑥ 内容通道本课只量得出形状，量不出张数：同一批 4500 张上 top1 0.9257→0.6921、塌缩率
     23.7%→0.0%、第二名概率 0.0409→0.1028，KL(p_T3‖p_T6)=**0.2050 nat**，4093 张图的真实类概率被
     加热压低超过 0.1，4500/4500 张分布都变了。其中一条把"老师把错图教给学生"那类机制当场压住：
     **老师在这个拟合集上本来就只错 10 张**（与 T 无关）⇒ 那 28 张里最多 10 张能由"老师在图上就是错的"
     解释，剩下至少 18 张必须来自分布在**正确图**上被抹平——这是"幅度上限"式的排除，不是测量，
     而且它只约束"错图污染"这一条通路，不约束内容通道的其他形状解释。
  ⑦ 预登记④ 那条几何对照与 ② 的结果**不一致**：总梯度离目标格最近的格子是 α=0.1×T3
     （相对距离 0.0378、夹角余弦 0.99936，在预登记的 0.10 闸门内 ⇒ 是"几何近亲"），
     可 ④ 那一扫描里恰恰是这一格的结果差 −19.8 张、CI 不含 0 ⇒ **初始那一步被推的方向与强度相同，
     不等于 600 步之后伤害相同**。这一列不参与出口判定，但它拿到的读数是"给权重通道加分失败"，
     与 ③(b) 同向。
  ⑧ 本课那句方向修正：两本账的逐流 gap 与第4课h 的读数逐位复算之后，α=0.05 那一格
     T=3 gap −45.6 张、T=6 gap −73.8 张 ⇒ 跨列差 −28.1 张按 gap 定义翻回错图数是
     **T=6 那一格多错 28.1 张**（同号 8/8、二项 p=0.0078）。第4课h 的档案字段文本按"入库不改"留着，
     修正写在它的 docstring 与板块文档里。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (BS, DATA_TIERS, SEED, TEACHER_ARCHIVE_ACC, build_mlp, logits_of,
                           soft_label_audit, summarize)
from step4c_alpha_T_sweep import term_grad_norms
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
# v2 = v1 + 顶层 `seeds`（本课用的那 8 条流的名字直接落档）。老档 v1 原样留盘，逐字段对着读。
OUT = HERE / "compress_report_v4i_mechanism_v2.json"

TIER, WIDTH = "中数据4500", 128
SMALL, T_LO, T_HI = 0.05, 3.0, 6.0
TEMPS = (1.0, 3.0, 6.0)
NBATCH = 5                                   # 第 0 批必须是第4课c 那一批（对账用），其余只报散布
ALPHA3 = "compress_report_v4f_mid_n8_v4.json"          # T=3 那一列的 α 阶梯（8 条流）
ALPHA6 = "compress_report_v4f_mid_n8_t6_v4.json"       # T=6 那一列（同 8 条流、同档同宽）
V4C = "compress_report_v4c_mid.json"                   # 探针 + 软标签体检的在册基准
V4H = "compress_report_v4h_crosscol_v6.json"           # 那条跨列差的钉子
GEO_TOL = 0.10                           # 总梯度相对距离 ≤ 0.10 才算"几何上的近亲"（预登记，非事后）


# ---------- 读账（三本在册账 + 一条钉子），全部先核对再取数 ----------
def load_ladder(name: str, scramble: bool = False) -> dict:
    """一本阶梯账：逐流 gap + 每一粒种子的 CE 错图数（两列的减数必须同一个数，闸 I 用）。"""
    d = json.loads((HERE / name).read_text(encoding="utf-8"))
    seeds = [int(s) for s in d["seeds"]]
    cells = {float(a): dict(zip(seeds, [int(g) for g in v["gaps_imgs"]]))
             for a, v in d["ladder_summary"].items()}
    if scramble:
        cells = {a: {s: g + 1 for s, g in c.items()} for a, c in cells.items()}
    ce: dict[int, set[int]] = {}
    for row in d["arms"].values():
        ce.setdefault(int(row["seed"]), set()).add(int(row["wrong_ce"]))
    for sd, vals in sorted(ce.items()):
        if len(vals) > 1:
            raise SystemExit(f"{name}：种子 {sd} 的 CE 错图数在这本账里有 {sorted(vals)} 个取值"
                             f" ⇒ 同一粒种子的减数不唯一，跨列差没有共同基准")
    return {"name": name, "seeds": seeds, "tier": d["tier"], "width": int(d["width"]),
            "t": float(d["temps_fixed"]), "floor": int(d["seed_floor_imgs"]),
            "teacher_acc": float(d["teacher_acc"]), "probe_alpha": float(d["probe_alpha"]),
            "anchor_alpha": float(d["anchor_alpha"]), "cells": cells,
            "cell_means": {float(a): float(v["mean_gap_imgs"]) for a, v in d["ladder_summary"].items()},
            "ce": {sd: next(iter(v)) for sd, v in ce.items()}}


def gate_pairing(lo: dict, hi: dict) -> None:
    """闸 III 的配对前提：两本账只许差一个温度列。"""
    if lo["seeds"] != hi["seeds"]:
        raise SystemExit(f"两本阶梯账的流名单不是同一批（成员或顺序不同）⇒ 逐流配对没有立足点，"
                         f"这一发只能退回两个均值之差")
    same = {k: (lo[k], hi[k]) for k in ("tier", "width", "floor", "teacher_acc", "probe_alpha", "anchor_alpha")
            if lo[k] != hi[k]}
    if same:
        raise SystemExit(f"这两本不是同一格只换温度列（{same}）⇒ 判别变量没对齐，差值不属于任何假说")
    if lo["t"] == hi["t"]:
        raise SystemExit(f"两本账的温度列都是 T={lo['t']:g} ⇒ 跨列差恒为 0，这一发什么都没测")
    if SMALL not in lo["cells"] or SMALL not in hi["cells"]:
        raise SystemExit(f"α={SMALL:g} 不是两本账都有 ⇒ 问句那一格没有原料")


def gate_ladder_recalc(b: dict) -> None:
    """闸 III 的复算：每一格均值必须从它自己的逐流 gap 现算命中（圆到 0.1 张）。"""
    for a, want in sorted(b["cell_means"].items()):
        got = round(sum(b["cells"][a].values()) / len(b["cells"][a]), 1)
        if got != want:
            raise SystemExit(f"{b['name']}：α={a:g} 格均值现算 {got} ≠ 档案记的 {want}"
                             f" ⇒ 读错账（串档/取错 α/单位不对），这一发的读数不发")


def gate_ce_identity(lo: dict, hi: dict, teacher_acc: float) -> dict:
    """闸 I：老师身份 + 两列的 CE 减数逐位相同。"""
    want = {lo["teacher_acc"], hi["teacher_acc"]}
    if len(want) != 1:
        raise SystemExit(f"两本账自报的 teacher_acc 不一致（{want}）⇒ 它们用的不是同一副老师，"
                         f"跨列的差里掺着软标签来源的变化")
    got = round(teacher_acc, 4)
    if got != lo["teacher_acc"] or got != round(TEACHER_ARCHIVE_ACC, 4):
        raise SystemExit(f"老师权重现算的测试准确率 {got:.4f} ≠ 两本账记着的 {lo['teacher_acc']} / "
                         f"板块常量 {TEACHER_ARCHIVE_ACC} ⇒ 本课引用的账与盘上这副权重不是同一次实验，"
                         f"全部机制读数作废")
    bad = [s for s in lo["seeds"] if lo["ce"].get(s) != hi["ce"].get(s)]
    if bad:
        raise SystemExit(f"CE 臂在流 {bad} 上两列不等 ⇒ 跨列差值的减数被换过，那 28 张里掺着 CE 自己的漂移")
    return {"teacher_acc_recomputed": got, "teacher_acc_archives": lo["teacher_acc"],
            "ce_minus_number_checked": f"{len(lo['seeds'])}/{len(lo['seeds'])}"}


def gate_recalc_v4h(lo: dict, hi: dict, nail: dict) -> dict:
    """闸 III 的钉子：跨列 α=0.05 那条差必须与第4课h 入库的读数逐位相同。

    第4课h 的档案是**按对**记的（`pairs` 的键是那一对的坐标标签），所以先按两个文件名认领这一对：
    认不出来＝在拿别的坐标的读数当本档的钉子（串档），当场退出而不回落到"随便挑一对"。
    """
    gaps = [hi["cells"][SMALL][s] - lo["cells"][SMALL][s] for s in lo["seeds"]]
    got = summarize(gaps, f"α={SMALL:g} 跨列差（T6−T3，按流配对）")
    hit = [p for p in nail["pairs"].values()
           if p["archives"] == {"T3": ALPHA3, "T6": ALPHA6}]
    if len(hit) != 1:
        raise SystemExit(f"{V4H} 里恰好认领到 {len(hit)} 对（文件名 {ALPHA3} × {ALPHA6}）"
                         f"⇒ 这一发对不上第4课h 的哪一个坐标，钉子没法钉")
    rec = hit[0]["cross_column_cellwise"][f"α={SMALL:g}"]
    bad = [k for k in ("mean_gap_imgs", "sd_imgs", "ci95_imgs", "same_sign_seeds", "p_seeds_binomial")
           if got[k] != rec[k]]
    if bad or [int(x) for x in got["gaps"]] != [int(x) for x in rec["gaps"]]:
        raise SystemExit(f"跨列差与 {V4H} 对不上（红键 {bad or ['gaps']}：现算 {got['mean_gap_imgs']} vs "
                         f"档案 {rec['mean_gap_imgs']}）⇒ 本脚本与第4课h 读的不是同一批数字")
    return got


# ---------- 零训练仪器：探针（复用第4课c 那支）+ 内容体检 ----------
def grads(xd, yd, soft, t: float, batch: int, width: int = WIDTH) -> tuple[torch.Tensor, torch.Tensor]:
    """第 batch 批上的两项梯度向量（同一副初始权重；batch=0 逐字等于第4课c 那一批）。"""
    torch.manual_seed(SEED)
    m = build_mlp(width)
    gen = torch.Generator().manual_seed(SEED)
    idx = None
    for _ in range(batch + 1):
        idx = torch.randint(0, len(xd), (BS,), generator=gen)
    z = m(xd[idx])
    ce = F.cross_entropy(z, yd[idx])
    kl = F.kl_div(torch.log_softmax(z / t, dim=1), torch.softmax(soft[idx] / t, dim=1),
                  reduction="batchmean") * t ** 2
    ps = list(m.parameters())
    g_ce = torch.autograd.grad(ce, ps, retain_graph=True)
    g_kl = torch.autograd.grad(kl, ps)
    flat = lambda gs: torch.cat([g.flatten() for g in gs])     # noqa: E731
    return flat(g_ce), flat(g_kl)


def probe(soft, xd, yd, t: float, batch: int) -> dict:
    """单位 α 下两项各自的梯度 L2 范数——α* 的唯一来源（不许由 T² 推定，见 --t2claim）。"""
    if batch:
        g_ce, g_kl = grads(xd, yd, soft, t, batch)
        return {"g_ce": round(float(g_ce.norm()), 5), "g_kd_unit": round(float(g_kl.norm()), 5)}
    return term_grad_norms(xd, yd, soft, t, WIDTH)             # 第 0 批：逐字复用第4课c 那支


def gate_probe_vs_archive(by_t: dict, arch: dict, t2claim: bool) -> None:
    """闸 II：第 0 批的探针与软标签体检必须命中第4课c 那本同档同宽的在册基准。"""
    for t in TEMPS:
        a = arch["gradient_probe"]["by_T"][f"{t}"]     # 键是 str(1.0)="1.0"：:g 会掉小数 ⇒ KeyError
        for k in ("g_ce", "g_kd_unit"):
            if by_t[t][0][k] != a[k]:
                raise SystemExit(f"探针第 0 批的 {k}（T={t:g}）现算 {by_t[t][0][k]} ≠ {V4C} 记的 {a[k]}"
                                 f"{'（本档正在演示：用教科书 T² 推定代替一次 backward）' if t2claim else ''}"
                                 f" ⇒ 这把尺子与产出那本账的尺子不是同一把，机制读数不发")
        want = a["alpha_to_match_anchor"]          # 档案记的：锚点 α=0.7 那一格的配平 α
        got = round(by_t[T_LO][0]["g_kd_unit"] * 0.7 / by_t[t][0]["g_kd_unit"], 3)
        if got != want:
            raise SystemExit(f"由现算梯度重算的 `alpha_to_match_anchor`（T={t:g}）得 {got}，"
                             f"档案记 {want} ⇒ 探针的用法与第4课c 不同（基准 α 或分母取错）")


def content_audit(soft, yd) -> dict:
    """内容通道自己动没动：逐温度的形状量 + T=3→T=6 的实际距离（单位=nat，不给张数的换算）。"""
    p = {t: torch.softmax(soft / t, dim=1) for t in (T_LO, T_HI)}
    kl = float((p[T_LO] * (p[T_LO].log() - p[T_HI].log())).sum(1).mean())
    p_true = {t: p[t].gather(1, yd[:, None]).squeeze(1) for t in p}
    drop = int((p_true[T_HI] < p_true[T_LO] - 0.1).sum())
    changed = int(((p[T_LO] - p[T_HI]).abs().max(1).values > 1e-4).sum())
    return {"mean_kl_pT3_given_pT6_nats": round(kl, 6), "imgs_p_true_drops_gt_0.1": drop,
            "imgs_dist_changed": changed, "pool_n": int(len(yd)),
            "teacher_wrong_imgs_on_pool": int((soft.argmax(1) != yd).sum())}


# ---------- 机制算式 ----------
def alpha_star(g3: float, g6: float) -> float:
    """把 (α=SMALL, T=6) 那一格的 KD 力度钉回 T=3 所需的 α：
    α* = SMALL × 实测‖∇(T²KL)‖在 T=6 的值 ÷ 同一批在 T=3 的值。"""
    return SMALL * g6 / g3


def weight_share(ladder: dict, a_star: float) -> dict:
    """预登记②：用夹住 α* 的那一段局部斜率算权重通道吃下的份额（不用事后挑的格子）。"""
    xs = sorted(ladder["cell_means"])
    seg = [(a, b) for a, b in zip(xs, xs[1:]) if a <= a_star <= b]
    if not seg:
        return {"alpha_star": round(a_star, 3), "share_imgs": None,
                "note": f"α*={a_star:.3f} 落在阶梯 [{xs[0]:g},{xs[-1]:g}] 之外 ⇒ 只能用端点外推，"
                        f"本课不外推（share 记 None，出口判定退回 ③ 的 (b)/(c) 两支）"}
    a, b = seg[0]
    slope = (ladder["cell_means"][b] - ladder["cell_means"][a]) / (b - a)
    step = summarize([ladder["cells"][b][s] - ladder["cells"][a][s] for s in ladder["seeds"]],
                     f"α={a:g}→{b:g} 段内步长（share 的不确定度来源）")
    return {"alpha_star": round(a_star, 3), "segment": [a, b], "slope_imgs_per_alpha": round(slope, 1),
            "share_imgs": round(slope * (a_star - SMALL), 1), "segment_step_paired": step}


def repro_scan(target: dict, ladder: dict, seeds: list[int]) -> dict:
    """T=3 阶梯上每一格与目标格的配对差：找"哪一格能复现那一格的结果"（出口 ③ 的 (b)/(c) 靠它）。"""
    out = {}
    for a, cell in sorted(ladder["cells"].items()):
        s = summarize([target[sd] - cell[sd] for sd in seeds], f"目标 − T3@α={a:g}")
        out[f"α={a:g}"] = {"residual_imgs": s["mean_gap_imgs"], "ci95_imgs": s["ci95_imgs"],
                           "same_sign": s["same_sign_seeds"], "p": s["p_seeds_binomial"],
                           "reproduces": bool(s["ci95_imgs"][0] <= 0 <= s["ci95_imgs"][1])}
    return out


def geo_scan(xd, yd, soft, cells: list[tuple[float, float]], target: tuple[float, float]) -> dict:
    """预登记④：10 个在册格各自的**总梯度** (1−α)·∇CE + α·∇(T²KL)，与目标格的相对距离与夹角。"""
    def total(a: float, t: float) -> torch.Tensor:
        g_ce, g_kl = grads(xd, yd, soft, t, 0)
        return (1 - a) * g_ce + a * g_kl
    gt = total(*target)
    nt = float(gt.norm())
    rows = {}
    for a, t in cells:
        if (a, t) == target:
            continue
        g = total(a, t)
        rel = float((g - gt).norm()) / nt
        cos = float(torch.dot(g, gt) / (g.norm() * nt))
        rows[f"α={a:g}×T{t:g}"] = {"rel_dist": round(rel, 4), "cos": round(cos, 5),
                                   "near_kin": bool(rel <= GEO_TOL)}
    best = min(rows, key=lambda k: rows[k]["rel_dist"])
    return {"target": f"α={target[0]:g}×T{target[1]:g}", "cells": rows, "nearest": best,
            "nearest_rel_dist": rows[best]["rel_dist"], "tol": GEO_TOL}


def main(teacherblind: bool = False, t2claim: bool = False, scramble: bool = False) -> None:
    mode = "teacherblind" if teacherblind else "t2claim" if t2claim else "scramble" if scramble else ""
    if teacherblind:
        print("[验红] --teacherblind：把老师换成 Fashion 那副（读档代码/图/账全不动）⇒ 只有闸 I 该红")
    if t2claim:
        print("[验红] --t2claim：不许 backward，用教科书假设『T² 补偿精确』推 ‖∇(T²KL)‖ ⇒ 只有闸 II 该红")
    if scramble:
        print("[验红] --scramble：给 T=6 那一列所有 gap +1 ⇒ 只有闸 III 该红")
    if OUT.exists() and not mode:
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉这本机制探针的判决（要重跑得先改名或删档）")

    lo, hi = load_ladder(ALPHA3), load_ladder(ALPHA6, scramble=scramble)
    gate_pairing(lo, hi)                                   # 闸 III（配对前提）
    gate_ladder_recalc(lo)                                 # 闸 III（两本各自复算）
    gate_ladder_recalc(hi)

    xt_all, yt_all, xte, yte = load_split()
    teacher = build_cnn()
    ckpt = MODELS / ("fashion_cnn_fp32.pt" if teacherblind else "mnist_cnn_fp32.pt")
    teacher.load_state_dict(torch.load(ckpt, weights_only=True))
    teacher.eval()
    t_acc = float((predict(teacher, xte, False) == yte).mean())
    ident = gate_ce_identity(lo, hi, t_acc)                # 闸 I

    n_data = DATA_TIERS[TIER]
    xd, yd = xt_all[:n_data], torch.tensor(yt_all[:n_data])
    soft = logits_of(teacher, xd)
    arch = json.loads((HERE / V4C).read_text(encoding="utf-8"))
    by_t = {t: [probe(soft, xd, yd, t, b) for b in range(NBATCH)] for t in TEMPS}
    if t2claim:
        # 教科书假设：T² 补偿之后 ‖∇(T²KL)‖ 与 T 无关 ⇒ 把 T≠基准 的实测值整批换成基准那一档的值
        for t in TEMPS:
            if t != T_LO:
                by_t[t] = [{"g_ce": by_t[t][b]["g_ce"], "g_kd_unit": by_t[T_LO][b]["g_kd_unit"]}
                           for b in range(NBATCH)]
    gate_probe_vs_archive(by_t, arch, t2claim)             # 闸 II
    aud = {t: soft_label_audit(soft, t=t) for t in TEMPS}
    for t in TEMPS:
        a = arch["soft_label_audit_by_T"][f"a1_T{t:g}"]["除以T"]
        for k in ("mean_top1", "pct_top1_gt_0.99", "mean_p2", "mean_margin_top1_top2"):
            if aud[t]["除以T"][k] != a[k]:
                raise SystemExit(f"软标签体检 T={t:g} 的 {k} 现算 {aud[t]['除以T'][k]} ≠ 档案 {a[k]}"
                                 f" ⇒ 读的不是那 4500 张 / 那副老师（闸 II 的体检半边）")
    cont = content_audit(soft, yd)

    # ---------- 闸 IV：两条通道至少一条要真的动 ----------
    a_star = [alpha_star(by_t[T_LO][b]["g_kd_unit"], by_t[T_HI][b]["g_kd_unit"])
              for b in range(NBATCH)]
    spread = max(a_star) - min(a_star)
    w_open = abs(a_star[0] - SMALL) > spread
    c_open = cont["mean_kl_pT3_given_pT6_nats"] > 0 and cont["imgs_dist_changed"] > 0
    if not (w_open or c_open):
        raise SystemExit(f"权重通道没开（α*={a_star[0]:.3f}，批次散布 {spread:.3f}）且内容通道没动"
                         f"（KL={cont['mean_kl_pT3_given_pT6_nats']}）⇒ 这一发没有判别对象，不发归因")

    nail = gate_recalc_v4h(lo, hi, json.loads((HERE / V4H).read_text(encoding="utf-8")))  # 闸 III
    wc = weight_share(lo, a_star[0])
    target = hi["cells"][SMALL]
    scan = repro_scan(target, lo, lo["seeds"])
    cells_all = [(a, t) for t, lad in ((T_LO, lo), (T_HI, hi)) for a in lad["cells"]]
    geo = geo_scan(xd, yd, soft, cells_all, (SMALL, T_HI))

    share = wc.get("share_imgs")
    frac = None if share is None else round(share / nail["mean_gap_imgs"], 3)
    hit = [k for k, v in scan.items() if v["reproduces"]]
    if share is not None and abs(share) >= 0.5 * abs(nail["mean_gap_imgs"]):
        exit_rule = f"(a) 大头在**权重通道**：α*={wc['alpha_star']}，share {share:+.1f} 张" \
                    f"＝跨列差的 {abs(frac):.0%} ⇒ 温度在 α={SMALL:g} 那一格偷偷是条权重轴"
    elif hit:
        exit_rule = (f"(c) **探针的权重图指认失败**：|share| 不够一半，但 T=3 阶梯上 {hit} "
                     f"的配对差 CI 含 0（能复现那一格的结果）⇒ 这一发的机制读数作废，回去改探针")
    else:
        worst = min(scan.items(), key=lambda kv: abs(kv[1]["residual_imgs"]))
        exit_rule = (f"(b) 大头在**内容通道**：share {share if share is not None else '—'} 张"
                     f"（{f'{abs(frac):.0%}' if frac is not None else '—'}），且 T=3 全阶梯没有一格能复现"
                     f"（最近的一格 {worst[0]} 仍差 {worst[1]['residual_imgs']:+.1f} 张 "
                     f"CI {worst[1]['ci95_imgs']}，{len(hit)}/{len(scan)} 格含 0）⇒ 权重与分配在 T=3 "
                     f"上穷尽也复现不出那一格的伤害")

    print(f"[闸 I] 老师身份：现算测试准确率 {ident['teacher_acc_recomputed']} = 两本账自报 "
          f"{ident['teacher_acc_archives']} = 板块常量 {TEACHER_ARCHIVE_ACC}；"
          f"CE 减数两列逐位核过 {ident['ce_minus_number_checked']} 条")
    print(f"[闸 II] 探针第 0 批与 {V4C} 的 gradient_probe 逐位命中（三温度×两范数 + α_to_match 重算）、"
          f"软标签体检四量×三温度命中")
    print(f"[闸 III] 两本阶梯 {len(lo['seeds'])} 条流同序、判别变量全同、{len(lo['cells'])}+"
          f"{len(hi['cells'])} 格均值现算命中、跨列差与 {V4H} 逐位相同")
    print(f"[闸 IV] 权重通道{'开' if w_open else '关'}（α*={a_star[0]:.3f}±{spread:.3f}）、"
          f"内容通道{'开' if c_open else '关'}（KL {cont['mean_kl_pT3_given_pT6_nats']:.4f} nat、"
          f"{cont['imgs_dist_changed']}/{cont['pool_n']} 张分布变了）")
    print(f"\n[目标格] α={SMALL:g}：T=3 gap {lo['cell_means'][SMALL]:+.1f} 张、T=6 gap "
          f"{hi['cell_means'][SMALL]:+.1f} 张 ⇒ 跨列差（T6−T3）{nail['mean_gap_imgs']:+.1f} 张 "
          f"CI {nail['ci95_imgs']}、同号 {nail['same_sign_seeds']}、二项 p={nail['p_seeds_binomial']}"
          f"\n           翻回错图数：负号 = **T=6 那一格多错 {-nail['mean_gap_imgs']:.1f} 张**"
          f"（这就是第4课h 那句『少错』要修正的方向）")
    print(f"[探针] ‖∇(T²KL)‖ 实测（第 0 批）：" + " / ".join(f"T{t:g} {by_t[t][0]['g_kd_unit']:.5f}"
                                                            for t in TEMPS)
          + f"；‖∇CE‖ {by_t[T_LO][0]['g_ce']:.5f}（与 T 无关，同一批同一初值）")
    print(f"[权重通道] α* = {SMALL:g} ×（T=6 实测梯度 ÷ T=3 实测梯度）= {a_star[0]:.3f}（{NBATCH} 批散布 "
          f"{min(a_star):.3f}~{max(a_star):.3f} ⇒ 用第 0 批（对账那一批）的数；教科书 T² 假设会给 "
          f"{SMALL:g} ⇒ 差 {a_star[0] - SMALL:+.3f}）"
          + (f"；夹住它的那一段 α={wc['segment'][0]:g}→{wc['segment'][1]:g}，斜率 "
             f"{wc['slope_imgs_per_alpha']} 张/单位α ⇒ share = {wc['share_imgs']:+.1f} 张"
             f"（段内步长 {wc['segment_step_paired']['mean_gap_imgs']:+.1f} 张 "
             f"CI {wc['segment_step_paired']['ci95_imgs']} ⇒ share 的不确定度就来自这里）"
             if share is not None else f"（{wc['note']}）"))
    print(f"[内容通道] 同一批图上的形状变化：top1 {aud[T_LO]['除以T']['mean_top1']}→"
          f"{aud[T_HI]['除以T']['mean_top1']}、塌缩率 {aud[T_LO]['除以T']['pct_top1_gt_0.99']}%→"
          f"{aud[T_HI]['除以T']['pct_top1_gt_0.99']}%、第二名概率 {aud[T_LO]['除以T']['mean_p2']}→"
          f"{aud[T_HI]['除以T']['mean_p2']}；KL(p_T3‖p_T6)={cont['mean_kl_pT3_given_pT6_nats']:.4f} nat、"
          f"{cont['imgs_p_true_drops_gt_0.1']} 张图的真实类概率被加热压低 >0.1；"
          f"老师在拟合集上本来就错 {cont['teacher_wrong_imgs_on_pool']} 张（污染上限，与 T 无关）")
    print(f"[几何对照] 总梯度离目标格最近的是 {geo['nearest']}（相对距离 {geo['nearest_rel_dist']}，"
          f"闸门 {GEO_TOL}）⇒ {'有' if any(v['near_kin'] for v in geo['cells'].values()) else '没有'}"
          f"近亲格")
    print("[T=3 阶梯逐格复现扫描]（配对差 = 目标格 − 该格；CI 含 0 才算复现）")
    for k, v in scan.items():
        print(f"   {k:<8} 差 {v['residual_imgs']:+7.1f} 张 CI {v['ci95_imgs']} 同号 {v['same_sign']}"
              f" p={v['p']}  {'← 复现' if v['reproduces'] else ''}")
    print(f"\n[出口（预登记③）] {exit_rule}")

    if mode:
        raise SystemExit(f"[验红模式 {mode}] 居然走到了发读数这一步 ⇒ 至少一道闸没拦住，不发不入库")
    out = {"quantity": "α=0.05 那一格的跨列之差（T=6 列 − T=3 列，gap 单位=CE错图−KD错图，"
                       "负=T=6 多错）拆成『权重通道』与『内容通道』两条候选各吃多少",
           "seeds": [int(s) for s in lo["seeds"]],
           "coordinate": {"tier": TIER, "width": WIDTH, "flow_pool": len(lo["seeds"]),
                          "cell": f"α={SMALL:g}", "temps": [T_LO, T_HI],
                          "archives": [ALPHA3, ALPHA6], "probe_baseline": V4C, "nail_archive": V4H},
           "sign_correction": "第4课h 的 `quantity` 字段与 docstring 落地④ 把这句话写成『T=6 少错 28 张』，"
                              "方向反了；本课逐位复算两本账的 gap 与那条跨列差之后按『多错 28 张』修正"
                              "（数字未动，旧档案 v1~v6 的字段文本按『入库不改』原样留着）",
           "decomposition": "α 固定 ⇒ (1−α)·CE 那半的权重与 T 无关（被构造排除）；剩下的两个通道："
                            "①KD 项力度 α·‖∇(T²KL)‖（T² 补偿不精确 ⇒ 换 T 顺带换权重）②软标签形状"
                            "（argmax 与 T 无关，只有配比变）。二者穷尽，所以出口是二分的。",
           "gates": ident, "probe_by_T": {f"{t}": {"batch0": by_t[t][0], "batches": by_t[t]}
                                          for t in TEMPS},
           "soft_label_audit_by_T": {f"{t}": aud[t]["除以T"] for t in TEMPS},
           "weight_channel": {"alpha_star": a_star[0], "alpha_star_batches": [round(v, 4) for v in a_star],
                              "batch_spread": round(spread, 4), "textbook_T2_says": SMALL,
                              "measured_vs_textbook": round(a_star[0] - SMALL, 4), **wc},
           "content_channel": cont, "geo_twin": geo, "nail_cross_column": nail,
           "repro_scan_T3_ladder": scan, "share_of_nail": frac, "exit": exit_rule,
           "verdict": exit_rule,
           "limitations": [
               "v2 相对 v1 只多一个顶层键 `seeds`：v1 只记了流池大小与两本来源账，"
               "『这一发用的是哪批流』要跳一跳（去那两本里读）才拿得到，守卫闸⑩ 的档案侧扫描因此看不见它。"
               "补上名字之后两本账除这个键之外必须逐字段相同——这一条是本轮 D 那笔仪器欠账的收据。",
               "这一发买的是**判别量**：没加流、没换那七维里任何一维、没长 α 格 ⇒ 出口 (b) 说"
               "『大头在内容通道』是**排除法**给的归因，不是测出来的归因。",
               "内容通道没有换算仪器：KL(p_T3‖p_T6) 的单位是 nat，不是张。要把它变成测量需要一发训练"
               "（把 α·‖∇(T²KL)‖ 钉死、只换 T；阶梯 5 格 × 8 条流 = **40 次训练**，本脚本一次都没付）。",
               "探针的一切读数都发生在**初始那一步、第 0 批（128 张）**：‖∇‖、α*、总梯度夹角都是"
               "「起点几何」，不是训练动力学（与第4课c 自我审计④ 同一条）。批次散布只报了 "
               f"{NBATCH} 批，它约束的是 α* 的稳定性，不约束『前 600 步的平均力度』。",
               "权重通道那一格用的是**局部斜率**（夹住 α* 的那一段），而 T=3 那条阶梯在 α=0.1 之后"
               "非单调（0.1 最差、0.2 回升）⇒ share 只在 α∈[0.05,0.1] 这一小段内合法，"
               "不许拿它去解释高剂量段。段内步长自己的 CI 就是 share 的不确定度。",
               f"读数只属于**一个坐标**（{TIER} × S{WIDTH} × 那 {len(lo['seeds'])} 条流 × α={SMALL:g}）："
               "满数据档与低数据档、S64 那副学生、Fashion 上同样那句话各自要各自的账，本课一个字都没碰。",
               "两本阶梯账跨判据版本（T=3 记在 `_v4`、T=6 记在 `_v4`；探针基准来自 3 条流那本 v4c）："
               "配的是训练路径上的逐流 gap，判据版本只改派生判词不改 gap ⇒ 可比，"
               "但『同一把尺子』在这里指同一个训练器，不指同一个判据。",
           ]}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（纯零训练：0 次训练、0 元）· {exit_rule}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="α→0 那一格跨列之差的机制拆解：零训练，读 "
                                             f"{ALPHA3} / {ALPHA6} / {V4C} / {V4H} 四本在册账"
                                             " + 现算老师软标签与梯度")
    ap.add_argument("--teacherblind", action="store_true", help="把老师换成 Fashion 那副，验闸 I 会红")
    ap.add_argument("--t2claim", action="store_true",
                    help="用教科书假设（T² 补偿精确）代替实测梯度，验闸 II 会红")
    ap.add_argument("--scramble", action="store_true", help="给 T=6 那一列 gap 全 +1，验闸 III 会红")
    a = ap.parse_args()
    main(teacherblind=a.teacherblind, t2claim=a.t2claim, scramble=a.scramble)
