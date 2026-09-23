# -*- coding: utf-8 -*-
"""第4课h：两列温度之间的斜率差——**按流配对**，不是两个点估计相减（纯读四本已入库的账，零训练、零 API 费）

欠账的原文（`面试版判决表.md` 第四节，上一轮新记的那条小账）：反号线在 MNIST 低数据档有两个坐标
（T=3 与 T=6），两列"小剂量→阶梯顶"的配对步长各自是 +48.6 与 +101.5 张 ⇒
"斜率随温度变陡"这句话**至今只是两个各自带 CI 的点估计相减**。同一批 8 条流、同一批图、
同一副学生、同一个老师，所以这两列本可以按流配对——配对了才有它自己的 CI。中数据档那一 pair
（倒挂的两个坐标，步长 7.1 与 29.8）顺手一起算，因为同一条规矩管两档。

三道闸必须先于读数成立（能被拆红的才叫闸，见下面"拆红"）：
  闸 A（**配对前提可核**）：两本账 α=0 那一格是 CE 臂，它与温度无关。如果同一条流在两列里
      真是同一次初始化、同一批图，那它的 `ce_wrong` 必须**逐位相同**。这一条把"我凭什么按种子配对"
      从口头假设变成盘上对账；核不到的流要如实报覆盖度（薄档只存 3 条恒等闸 ⇒ 低数据那一 pair
      只有 3/8 条流能逐位核，其余 5 条按同一构造推定，这句话必须打印出来）。
  闸 B（复算对账）：本脚本从 `ladder_summary[α]["gaps_imgs"]` 现列重算的**每一格均值**，以及
      α=小剂量→阶梯顶 那条步长的 均值 / σ / SE / 极差 / d4 折算 σ̂ / 二项 p / **逐流序列**，
      必须与**那本账自己记着的**读数逐位相同。对不上就是读错账（串档、取错 α、拿错档），当场退出——
      不许把错位来的差值当新读数打印出去。
  闸 C（两列必须真的不同）：两本账的 `temps_fixed` 相等 ⇒ 这一发什么都没测，退出。
      （"空表全过 = 没测到"的同一族：两列相同时斜率差恒为 0，那是装饰不是读数。）

拆红（每档只钉一道闸，跑不出红的模式＝那道闸没测过）：
  --red-check  把中数据那一对的 T=6 账换成另一本 T=3 账 ⇒ 只有闸 C 该红
  --unlink     只旋转 T=6 那本的**种子标签**（模拟「同名种子其实不是同一条流」）⇒ 红的第一道必须是闸 A。
               这一档由 `probe_gates` 把 A、B 各自的反应**实测打印**出来，不靠注释断言：B 的
               均值/σ/SE/极差/二项 p 都不读键 ⇒ 该照旧绿；但 B 的「逐流序列」那条子检查读键序 ⇒
               它也会红。所以 B 不是 A 的替身，可也不能说 B 完全看不见配对方向——这句话说的是实测。
  --scramble   给 T=6 那本的所有 gap +1（模拟取错档案/单位）⇒ 只有闸 B 该红（A 只看 CE 臂，不碰 KD gap）

预登记的出口（跑之前写死，此后不许改口；钉子数字现读自四本已入库档案）：
  ① 三道闸全过才发读数；任一红 ⇒ 退出、不入库。
  ② 斜率差的**点估计由算术钉死**，不许当成新信息：中数据 = 29.8 − 7.1 = **+22.7 张**、
     低数据 = 101.5 − 48.6 = **+52.9 张**。本脚本的点估计是从逐流 gap 现算的，而档案存的两个均值
     各自已圆到 0.1 张 ⇒ 允许**双舍入的 0.1 张**差（`NAIL_TOL`）；出这个界＝本脚本算错，不发读数。
     钉子是"和两本账自报的读数对账"，不是"预测"。这一发唯一的新信息是**它的 CI、同号计数与二项 p**。
  ③ 出口：(a) CI 不含 0 ⇒ "斜率随温度变陡/变浅"从此是**以流为单位的配对读数**（升格）；
         (b) CI 含 0 ⇒ 这句话仍然只是两个点估计之差，欠账改写成"按 1/√n 还要再一倍流"。
     两档各自判，一升一不升要如实分开报，不许挑好的说。
  ④ 顺手多算的那一列（逐 α 的跨列配对差 T6−T3）不许事后挑：它回答"变陡发生在哪一端"。
     判法写死在 `cellwise_verdict`，只看小剂量格与阶梯顶格这两格的不含 0 模式，四个分支全覆盖：
     小格不含 0 而顶格含 0 ⇒ 变化发生在**小剂量端**、两列的顶差不多；顶格不含 0 而小格含 0 ⇒
     发生在**顶端**；两格都不含 0 ⇒ 整段一起平移（读不出哪一端）；两格都含 0 ⇒ 这一列什么都没判。

这一发买的是什么，要说清：它**既不是分辨率**（没加一条流）**也不是独立性**（没换那七维里的任何一维）
**也不是覆盖**（没加一格 α）——它是**第四种买法：同一批流上换一个判别量**（把"两个点估计相减"
改成"以流为单位的配对差"）。这正是上一轮给第4课g 那条债写下的"或者换一个判别量"的做法第一次落地；
同一套配对可以指向那一发（三档 × 两条样本批次），本轮不做。

落地（跑完之后才写，只写实测到的事）：
  ① 三道闸各钉住一次拆红，没有一档是装饰：`--red-check` 红在闸 C（"两本账的温度列都是 T=3"）；
     `--unlink` 红在闸 A，而**同一档里实测打印**出闸 B 只红 1 条子检查——「步长的逐流序列」，
     五个格均值与步长的 6 个统计量全绿 ⇒ 「B 不读键序所以看不见配对方向」这句话本轮被推翻了一半：
     B 的均值族确实不读键，但它的序列子检查读，所以 B 与 A 有重叠、不能互相替身；
     `--scramble` 红在闸 B（第一条就是 α=0.05 格均值现算 −72.8 ≠ 档案的 −73.8）。
  ② 预登记②的钉子对上了：中数据现算 +22.6 vs 档案相减 +22.7（双舍入 0.1 之内）、低数据 +52.9 = +52.9。
     于是"点估计不是发现"这句话本轮是**被对账了一次**，不是被引用了一次。
  ③ 两档一升一不升，如实分开报：**低数据** +52.9 张 CI [+33.1, +72.7]、8/8 同号、二项 p=0.0078
     ⇒ 出口 (a)：「加热把 α→0 曲线的斜率变陡」在这一档从此是以流为单位的配对读数；
     **中数据** +22.6 张 CI [−0.4, +45.7]、6/8 ⇒ 出口 (b)：仍然只是两个点估计之差。
     中数据的下界离 0 只差 **0.4 张**——σ 维持 27.5 时 n=12 就把下界抬到约 +5.1（按现 σ 折算，
     不是读数；上两轮补流实测过 σ 会涨）。而 T=3 那一列的 12 条流账 `_mid_n12_v4` **已经在册**，
     缺的只有 T=6 那一列 ⇒ 新落一条有明确买法的分辨率债：`step4f --tier 中数据4500 --t 6 --nseeds 12`
     之后重跑本脚本，一发判干净。
  ④ 逐 α 那一列读出来的东西比"变陡"两个字更有信息，而且**两档不是同一件事**：
     中数据 ⇒ 变化在小剂量端（α=0.05 跨列差 −28.1 张 CI [−42.3, −14.0] 不含 0，阶梯顶 −5.5 张跨 0），
     符号是 **T=6 那一列在最浅的一格少错 28 张** ⇒ "斜率更陡"全部来自加热把小剂量端救回来，
     不是顶上有额外伤害；低数据 ⇒ 变化在顶端（α=0.05 差 −3.1 张跨 0，α=0.3 +50.5、α=0.7 +49.8 都不含 0）
     ⇒ 整个高剂量平台被抬起来。同名的一句话在两档是两种机制，这条不许合并成一句"温度越高斜率越陡"。
  ⑤ 配对前提的覆盖度按档不同且已打印在表头：中数据 CE 臂 8/8 条流逐位核过；薄档只存 3 条恒等闸
     ⇒ 低数据那一 pair 有 5 条流是按同一构造**推定**、不是逐位核过。
本脚本 0 次训练，产物 `compress_report_v4h_crosscol.json`。它是**纯读档的派生物**，不进守卫
`_red_v3.py` 的 `compress_report_v4f*` 在册账名单（那七道闸盯的是训练路径与判词，这里没有训练路径）；
它自己的守卫就是上面三道闸＋三档拆红，这一条边界写在 `ml-compress/README.md` 里。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from step4_distill import check_flow_constants, summarize

HERE = Path(__file__).resolve().parent
OUT = HERE / "compress_report_v4h_crosscol.json"
SEEDS = [42, 7, 2026, 11, 202, 3003, 40004, 55555]      # 四本账共用的随机流名单（顺序也一致）

# 每对 = 同档 / 同宽 / 同任务 / 同批 8 条流，只有温度列不同。布局差异只允许出现在这张表里。
PAIRS = {
    "中数据4500": ("compress_report_v4f_mid_n8_v4.json", "compress_report_v4f_mid_n8_t6_v4.json"),
    "低数据2000": ("compress_report_v4f_low_n8_thin_v3.json", "compress_report_v4f_low_n8_thin_t6_v4.json"),
}
# 预登记②的算术钉子（抄自两本账各自的 paired_step_small_to_anchor，跑完不许改）
EX_ANTE = {"中数据4500": 22.7, "低数据2000": 52.9}
NAIL_TOL = 0.1 + 1e-9        # 预登记②：档案两个均值各自圆到 0.1 ⇒ 两次取整最多差 0.1 张


def excl0(s: dict) -> bool:
    """CI 不含 0。`summarize` 不供这个键，只能由 ci95_imgs 两端现算（键名写错＝静默假绿）。"""
    return bool(s["ci95_imgs"][0] > 0 or s["ci95_imgs"][1] < 0)


def load(name: str, unlink: bool = False, bump: int = 0) -> dict:
    """读一本账：先核名单/档位/温度列，再把逐流序列摊成 {α: {seed: gap}}。

    `unlink` 只旋转**标签**不旋转数值：这正是「同名的种子在两列里其实不是同一条流」那个假说
    会长成的样子——册内读数（均值、册内步长）全都不动，只有跨册配对会散架。
    `bump` 给所有 KD gap 加常数（模拟取错档案/单位）。两者只服务拆红，正常跑时都是默认值。
    """
    p = HERE / name
    if not p.exists():
        raise SystemExit(f"缺 {name}——两列配对缺一本就没有「跨列」这回事，不许用一本账下一个斜率的判决")
    d = json.loads(p.read_text(encoding="utf-8"))
    seeds = [int(s) for s in d["seeds"]]
    if seeds != SEEDS:
        raise SystemExit(f"{name} 的种子名单/顺序与共用那批不一致 ⇒ 跨列的差不能按流配对")
    check_flow_constants(len(seeds))
    cells = {float(a): dict(zip(seeds, [int(g) for g in v["gaps_imgs"]]))
             for a, v in d["ladder_summary"].items()}
    ce = {int(e["seed"]): int(e["ce_wrong"]) for e in d["gate_alpha_zero_identity"]}
    if unlink:
        rot = seeds[1:] + seeds[:1]
        cells = {a: dict(zip(rot, g.values())) for a, g in cells.items()}
        ce = dict(zip(rot, ce.values()))
    if bump:
        cells = {a: {sd: g + bump for sd, g in c.items()} for a, c in cells.items()}
    return {"name": name, "dataset": d["dataset"], "tier": d["tier"], "width": int(d["width"]),
            "t": float(d["temps_fixed"]), "teacher_acc": float(d["teacher_acc"]),
            "probe_alpha": float(d["probe_alpha"]), "anchor_alpha": float(d["anchor_alpha"]),
            "floor": int(d["seed_floor_imgs"]), "cells": cells, "ce": ce,
            "seed_cell_means": {float(a): float(v["mean_gap_imgs"]) for a, v in d["ladder_summary"].items()},
            "step": d["paired_step_small_to_anchor"]}


def gate_pair(lo: dict, hi: dict, where: str) -> tuple[float, float]:
    """闸 C + 同格核：两列必须**不同温**、其余判别变量必须**全同**、判据吃的那两格 α 必须都在两本账里。"""
    if lo["t"] == hi["t"]:
        raise SystemExit(f"{where}：两本账的温度列都是 T={lo['t']:g} ⇒ 斜率差恒为 0，这一发什么都没测")
    same = {k: (lo[k], hi[k]) for k in ("dataset", "tier", "width", "teacher_acc", "floor",
                                        "probe_alpha", "anchor_alpha") if lo[k] != hi[k]}
    if same:
        raise SystemExit(f"{where}：这两本不是同一格只换列（{same}）——判别变量没对齐，差值不属于任何假说")
    small, top = min(lo["cells"]), max(lo["cells"])
    if top != lo["anchor_alpha"] or hi["anchor_alpha"] != top or hi["probe_alpha"] != lo["probe_alpha"]:
        raise SystemExit(f"{where}：阶梯顶/锚点剂量两本不一致（{top} vs {hi['anchor_alpha']}）⇒ 尺子不同")
    if small not in hi["cells"] or top not in hi["cells"]:
        raise SystemExit(f"{where}：α={small:g}/{top:g} 不是两本账都有，跨列配对没有原料")
    return small, top


def gate_ce_pairing(lo: dict, hi: dict, where: str) -> str:
    """闸 A：CE 臂与温度无关 ⇒ 同一条流在两列里的 ce_wrong 必须逐位相同。核不到的流如实报。"""
    both = sorted(set(lo["ce"]) & set(hi["ce"]))
    if not both:
        raise SystemExit(f"{where}：两本账没有共同的恒等闸流 ⇒ 配对前提一次都没核到（空表不算通过）")
    bad = [s for s in both if lo["ce"][s] != hi["ce"][s]]
    if bad:
        raise SystemExit(f"{where}：CE 臂在流 {bad} 上两列不等 ⇒ 同名的种子不是同一条流，"
                         f"跨列差值不许按流配对（只能退回不配对、报两个均值之差）")
    return f"{len(both)}/{len(SEEDS)}"


def step_recalc(b: dict, small: float, top: float) -> dict:
    """册内那条 α=小剂量→阶梯顶 的配对步长，从盘上的逐流 gap 现算（闸 B 与读数共用这一份代码）。"""
    return summarize([b["cells"][top][sd] - b["cells"][small][sd] for sd in SEEDS], "α=小→顶 配对步长")


def gate_recalc(b: dict, where: str, collect: bool = False) -> list[str]:
    """闸 B：现列重算必须逐位命中那本账自己记着的读数（每格均值 + 那条小剂量→阶梯顶的步长）。

    `collect=True` 时不退出，把违规攒成名单返回——--unlink 要靠它打印「B 的哪几条子检查动了」。
    """
    bad: list[str] = []

    def check(what: str, got, want) -> None:
        if got != want:
            bad.append(f"{what}：现算 {got} ≠ 档案记的 {want}")

    for a, recorded in b["seed_cell_means"].items():
        check(f"α={a:g} 格均值", round(sum(b["cells"][a].values()) / len(b["cells"][a]), 1),
              round(recorded, 1))
    small, top = min(b["cells"]), max(b["cells"])
    s = step_recalc(b, small, top)
    for k in ("mean_gap_imgs", "sd_imgs", "se_imgs", "range_imgs", "d4_sigma_hat", "p_seeds_binomial"):
        check(f"α={small:g}→{top:g} 步长的 {k}", s[k], b["step"][k])
    check("步长的逐流序列", [int(x) for x in s["gaps"]], [int(x) for x in b["step"]["gaps"]])
    if bad and not collect:
        raise SystemExit(f"{where}：{bad[0]} ⇒ 读错账（取错格子/串档/单位不对），这条读数不发")
    return bad


def cellwise_verdict(diff: dict, small: float, top: float) -> str:
    """预登记④：判法写死——只看小剂量格与阶梯顶格这两格的跨列差不含 0 的模式。"""
    a, b = excl0(diff[small]), excl0(diff[top])
    if a and not b:
        return "跨列的变化发生在**小剂量端**，两列的顶差不多"
    if b and not a:
        return "跨列的变化发生在**阶梯顶那一端**"
    if a and b:
        return "两格都不含 0 ⇒ 整段一起平移（读不出发生在哪一端）"
    return "两格都含 0 ⇒ 这一列什么都没判"


def probe_gates(lo: dict, hi: dict, where: str) -> None:
    """--unlink 专用：把 A、B 各自的反应实测打印出来，再红着退出（这一支永不入库）。

    为什么要替 B 也报一次：「B 看不见配对结构」是我上一版**口述**的断言，而 B 的逐流序列子检查
    其实读键序。只有跑出来才知道 B 到底动了哪几条——不跑就是把一句没验过的话留在仓库里当证据。
    """
    msg_a = ""
    try:
        gate_ce_pairing(lo, hi, where)
    except SystemExit as e:
        msg_a = str(e)
    v_b = gate_recalc(hi, where, collect=True)
    print(f"[实测] 闸 A：{'红（如期）' if msg_a else '不红 ← 这一档没测到，配对前提是装饰'}")
    if msg_a:
        print(f"        {msg_a}")
    print(f"[实测] 闸 B：{len(v_b)} 条子检查红（名单如下；一条都没红＝B 不读键序，那它就不能替 A 作证）")
    for v in v_b:
        print(f"        {v}")
    if not msg_a:
        raise SystemExit(f"{where}：--unlink 没能拆红闸 A")
    raise SystemExit(f"{where}：--unlink 验红结束（红的是闸 A），不发读数、不入库")


def main(mode: str = "") -> None:
    unlink, scramble = mode == "unlink", mode == "scramble"
    if mode == "red-check":
        # 验红：把中数据那一对里的 T=6 账换成另一本 T=3 账——档位对、宽度对、流对，只有温度列不对。
        # 若闸 C 恒绿，「两列」这两个字就是装饰。跑通即证明它会红着退出（不写档案）。
        PAIRS["中数据4500"] = (PAIRS["中数据4500"][0], "compress_report_v4f_mid_n8_v3.json")
        print("[验红] --red-check：中数据那一对的第二本换成 T=3 的 `_n8_v3`，闸 C 必须拦下")
    if unlink:
        print("[验红] --unlink：只旋 T=6 那本的种子标签 ⇒ 红的第一道必须是闸 A；闸 B 动了哪几条一并打印")
    if scramble:
        print("[验红] --scramble：只给 T=6 那本的 gap +1 ⇒ 只有闸 B 该红（闸 A 只看 CE 臂，该照旧绿）")
    if OUT.exists() and not mode:
        # 验红那一支永不入库（它必须走到读数断言），所以不能被这道闸先挡掉。
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉这本跨列配对的判决（要重跑得先改名或删档）")

    out = {"quantity": "同一批 8 条流上，T=6 那一列减 T=3 那一列（单位：张/10000，正=T=6 更往多错走）",
           "seed_order": SEEDS, "mode": mode or "（正常跑）",
           "prediction_ex_ante_imgs": dict(EX_ANTE), "pairs": {}}
    for tier, (n_lo, n_hi) in PAIRS.items():
        # 两列只有一本被动：两本同向旋转等于没旋，两本同加常数等于差里没动。
        lo = load(n_lo)
        hi = load(n_hi, unlink=unlink, bump=1 if scramble else 0)
        where = f"{tier}（{n_lo} vs {n_hi}）"
        small, top = gate_pair(lo, hi, where)          # 闸 C：不同温 + 其余判别变量全同
        if unlink:
            probe_gates(lo, hi, where)                 # 只有这一档替 A/B 各报一次实测再退出
        cov = gate_ce_pairing(lo, hi, where)           # 闸 A：配对前提
        gate_recalc(lo, where)                         # 闸 B：两本各对一次账
        gate_recalc(hi, where)
        if scramble:
            raise SystemExit(f"{where}：--scramble 走完三道闸居然没红 ⇒ 闸 B 是装饰")
        steps = {tag: step_recalc(b, small, top) for tag, b in (("T3", lo), ("T6", hi))}
        diff = {a: summarize([hi["cells"][a][sd] - lo["cells"][a][sd] for sd in SEEDS], f"α={a:g} 跨列差")
                for a in sorted(set(lo["cells"]) & set(hi["cells"]))}
        slope = summarize([(hi["cells"][top][sd] - hi["cells"][small][sd])
                           - (lo["cells"][top][sd] - lo["cells"][small][sd]) for sd in SEEDS],
                          "斜率差（T6 步长 − T3 步长，按流配对）")
        nail = round(steps["T6"]["mean_gap_imgs"] - steps["T3"]["mean_gap_imgs"], 1)
        if abs(nail - EX_ANTE[tier]) > 1e-9 or abs(slope["mean_gap_imgs"] - nail) > NAIL_TOL:
            raise SystemExit(f"{where}：斜率差的点估计 {slope['mean_gap_imgs']} 对不上预登记②——档案自报步长"
                             f"相减得 {nail}、钉子写的是 {EX_ANTE[tier]}、双舍入容差 {NAIL_TOL:g}"
                             f" ⇒ 本脚本算错了，这条读数不发（钉子是与账对账，不是预测）")
        exit_slope = "(a) 升格成以流为单位的配对读数" if excl0(slope) else "(b) 仍只是两个点估计之差"
        exit_cell = cellwise_verdict(diff, small, top)
        out["pairs"][tier] = {"archives": {"T3": n_lo, "T6": n_hi}, "criterion_alphas": [small, top],
                              "ce_pairing_checked": cov, "steps_within_column": steps,
                              "cross_column_cellwise": {f"α={a:g}": s for a, s in diff.items()},
                              "slope_diff_paired": slope, "ex_ante_nail_imgs": nail,
                              "exit_slope": exit_slope, "exit_cellwise": exit_cell}
        print(f"\n[{tier}] 配对前提：CE 臂 {cov} 条流逐位核过 · 判据吃的两格 α={small:g}/{top:g}"
              + ("（薄档只存 3 条恒等闸 ⇒ 其余按同一构造推定，不写进结论）" if cov.startswith("3/") else ""))
        for tag in ("T3", "T6"):
            st = steps[tag]
            print(f"   册内步长 {tag}：{st['mean_gap_imgs']:+.1f} 张 CI [{st['ci95_imgs'][0]:+.1f},"
                  f" {st['ci95_imgs'][1]:+.1f}] 同号 {st['same_sign_seeds']}（档案原读数，闸 B 已逐位对过）")
        print(f"   斜率差（按流配对）{slope['mean_gap_imgs']:+.1f} 张（预登记钉子 {nail:+.1f}）"
              f" CI [{slope['ci95_imgs'][0]:+.1f}, {slope['ci95_imgs'][1]:+.1f}] 同号 {slope['same_sign_seeds']}"
              f" 二项 p={slope['p_seeds_binomial']} sd={slope['sd_imgs']} 极差 {slope['range_imgs']} 张 ⇒ {exit_slope}")
        for a, s in diff.items():
            print(f"   α={a:<5g} 跨列差 {s['mean_gap_imgs']:+7.1f} 张 CI [{s['ci95_imgs'][0]:+.1f},"
                  f" {s['ci95_imgs'][1]:+.1f}] 同号 {s['same_sign_seeds']}"
                  f" {'不含 0' if excl0(s) else '跨 0'}")
        print(f"   逐 α 那一列按预登记④的判法 ⇒ {exit_cell}")

    out["verdict"] = ("；".join(f"{t} 的斜率差走出口 {v['exit_slope']}" for t, v in out["pairs"].items())
                      + " ——两档各自判，一升一不升如实分开报")
    out["limitations"] = [
        "这一发**没换任何一维**（同批图、同老师、同副学生、同批 8 条流、同一批 α）⇒ 它买到的是"
        "**判别量**，不是分辨率/独立性/覆盖中的任何一种；跨列差值再干净也不构成第二次复证",
        "配对前提的覆盖度按档不同：中数据 8/8 条流逐位核过（CE 臂），薄档只存 3 条恒等闸 ⇒ "
        "低数据那一 pair 有 5 条流是按同一构造推定、而非逐位核过",
        "低数据那一对的两本账跨判据版本（T=3 是 `_v3`、T=6 是 `_v4`）：配的是**训练路径上的逐流 gap**，"
        "判据版本只改派生判词不改 gap，所以可比；但这里的「同一把尺子」=同一个训练器，不是同一个判据",
        "闸 A 只核到 CE 臂那一格：两列的 CE 读数逐位相同，恰恰因为 CE 与 T 无关。它核到「同一次初始化、"
        "同一批图」，核不到 KL 那一侧——这是配对前提的下界，不是全证",
        "斜率差的点估计由算术钉死（见 prediction_ex_ante_imgs 与每对的 ex_ante_nail_imgs），"
        "所以它不是发现；发现只能是 CI / 同号计数 / 二项 p",
        "两档的斜率差共用同一批 8 条流（同一个随机源）⇒ 它们是同一条流上的两个读数，"
        "不能当两次独立复证用；要独立就得换流/换任务/换老师",
    ]
    if mode:
        raise SystemExit(f"[验红模式 {mode}] 居然走到了发读数这一步 ⇒ 至少一道闸没拦住，不发不入库")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（纯读档：0 次训练，零 API 费）· {out['verdict']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="两列温度之间的斜率差：纯读四本已入库的账，按流配对")
    ap.add_argument("--red-check", action="store_true", help="把 T=6 那本换成另一本 T=3 账，验闸 C 会红")
    ap.add_argument("--unlink", action="store_true", help="旋转 T=6 的种子标签，验闸 A（配对前提）会红")
    ap.add_argument("--scramble", action="store_true", help="给 T=6 的 gap 全 +1，验闸 B（复算对账）会红")
    a = ap.parse_args()
    main(mode="red-check" if a.red_check else "unlink" if a.unlink else "scramble" if a.scramble else "")
