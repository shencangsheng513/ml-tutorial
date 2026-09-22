# -*- coding: utf-8 -*-
"""第7课：任务 × 老师 的 2×2 低数据档裁决（纯读四本已入库的账，零训练、零 API 费）

"低数据档一加热就有益"这个翻号，目前只活在 MNIST×老师A 那一格里。四格各有 8 条配对随机流：
  MNIST×A（v4e_low_t1t6）/ MNIST×D（v4e_d_low_t1t6，同档换欠训练老师）
  Fashion×A（v4e_fashion_low_t1t6）/ Fashion×dry（v6b，任务内换干湿）
两说的分歧只看一件事：**同一档里换老师，符号换不换**。
  H_任务：翻号是"训练集小"这个处方的性质 ⇒ 四个格都该与所在任务同向（MNIST 两格同正）。
  H_软标签：翻号是老师 A 那份塌缩到 92.8% 的软标签的性质 ⇒ 换一份更湿的软标签（D 49.3%）就该塌回 0。
判据（看到 D 那发读数之前写死在 step4e 文档串里，此处照抄）：MNIST×D 若仍同号为正 → H_任务；
若转负或 CI 跨 0 → 翻号收进"MNIST×老师A 的这份软标签"，任务级的说法降级。

一件白送的仪器：四本账用的是**同一份随机流名单**（顺序也一致，函数里逐位核过才敢配），
所以"换老师"这一维的差是**按种子配对**的差，不是两个均值相减——A 与 D 各自有 CI 不等于它们之差有 CI。
跨任务那一行只报差值与各自己的 CI，不配：同名的种子在两个数据集里不是同一条流。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = [42, 7, 2026, 11, 202, 3003, 40004, 55555]      # 四本账共用的随机流名单
T95 = 2.365                                              # 双侧 95%，df=7
OUT = HERE / "compress_report_v7_interaction.json"
LOW_TIER = "低数据2000"
PAIRS = (("MNIST", "A", "D"), ("Fashion", "A", "dry"))   # 同任务内配对的两副老师

# 每格 = (档案名, 从档案里取"按种子配对的 T 主效应序列"的闭包, 确认这本账确实是低数据档 T1/T6 的断言)
# 布局差异只允许出现在这张表里：v4e 家族把序列放在 t_main_effect_paired，第6课b 放在顶层。
CELLS = {
    ("MNIST", "A"): ("compress_report_v4e_low_t1t6.json",
                     lambda d: d["t_main_effect_paired"]["per_seed_imgs"],
                     lambda d: d.get("tier") == LOW_TIER and d.get("target_Ts") == [1.0, 6.0]),
    ("MNIST", "D"): ("compress_report_v4e_d_low_t1t6.json",
                     lambda d: d["t_main_effect_paired"]["per_seed_imgs"],
                     lambda d: d.get("tier") == LOW_TIER and d.get("target_Ts") == [1.0, 6.0]),
    ("Fashion", "A"): ("compress_report_v4e_fashion_low_t1t6.json",
                       lambda d: d["t_main_effect_paired"]["per_seed_imgs"],
                       lambda d: d.get("tier") == LOW_TIER and d.get("target_Ts") == [1.0, 6.0]),
    ("Fashion", "dry"): ("compress_report_v6b_fashion_dry.json",
                         lambda d: d["paired_t_effect_per_seed_imgs"],
                         lambda d: d.get("tier") == LOW_TIER),
}


def stats(vals: list[float]) -> dict:
    """配序列的自报读数：均值、σ、95% CI（t 分布 df=n−1）、同号计数、双侧二项。"""
    n = len(vals)
    assert n == len(SEEDS), f"这条序列 {n} 个数，不是 {len(SEEDS)} 条流——配对不成立"
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    ci = T95 * sd / math.sqrt(n)
    pos, neg = sum(1 for v in vals if v > 0), sum(1 for v in vals if v < 0)
    k = max(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    return {"mean_imgs": round(m, 1), "sd_imgs": round(sd, 1),
            "ci95_imgs": [round(m - ci, 1), round(m + ci, 1)],
            "per_seed_imgs": [round(v, 1) for v in vals],
            "same_sign": f"{k}/{n}（{'+' if pos >= neg else '−'}）", "p_binomial": round(p, 4),
            "excludes_zero": bool(m - ci > 0 or m + ci < 0)}


def load(cell: tuple[str, str]) -> list[float]:
    """读一本账：先核它是这一格（缺文件/档位不对/种子名单不对，都零训练退出）。"""
    name, get, check = CELLS[cell]
    p = HERE / name
    if not p.exists():
        raise SystemExit(f"缺 {name}——2×2 缺 {cell} 这一角，不许用三格下一个两说的判决")
    d = json.loads(p.read_text(encoding="utf-8"))
    if not check(d):
        raise SystemExit(f"{name} 不是 {LOW_TIER} 的 T=1/T=6 配对主效应档案，读错账的对账会绿")
    if [int(s) for s in d["seeds"]] != SEEDS:
        raise SystemExit(f"{name} 的种子名单/顺序与 {SEEDS} 不一致 ⇒ 跨老师的差不能按种子配对")
    vals = [float(v) for v in get(d)]
    if d.get("archive_parity_all_pass") is False or d.get("ce_anchor_parity_failed"):
        raise SystemExit(f"{name} 自己记着对账没过，它的读数不进这张表")
    return vals


def main(red_check: bool = False) -> None:
    if red_check:
        # 验红：故意把 Fashion×A 那一角指向 MNIST 的 t=3 那本账，档位对、温度列不对。
        # 这条闸若恒绿，2×2 就是在读四本互不相干的账；跑通即证明它会红着退出（不写档案）。
        print("[验红] --red-check：把 Fashion×A 换成 MNIST 的 compress_report_v4e_low_t3.json，"
              "断言必须拦下（T 列不是 1/6）；本课读数不入库")
        CELLS[("Fashion", "A")] = ("compress_report_v4e_low_t3.json", *list(CELLS[("Fashion", "A")])[1:])
    if OUT.exists() and not red_check:
        # 验红那一支永不入库（它必须走到读数断言），所以不能先被这道闸挡掉——
        # 否则 --red-check 只验了"档案已存在"这道闸，读数断言那条闸其实没测过。
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉这本 2×2 判决")
    vals = {c: load(c) for c in CELLS}
    st = {f"{t}×{k}": stats(v) for (t, k), v in vals.items()}
    print("[第7课 2×2] 低数据档 · 按种子配对的 T 主效应（正=加热有益，负=加热有害），单位：张/10000")
    for name, s in st.items():
        print(f"  {name:<14} {s['mean_imgs']:+7.1f} 张  95% CI [{s['ci95_imgs'][0]:+.1f},"
              f" {s['ci95_imgs'][1]:+.1f}]  同号 {s['same_sign']}  二项 p={s['p_binomial']}"
              f"  {'CI 不含 0' if s['excludes_zero'] else 'CI 跨 0 → 这一格读不出方向'}")

    con = {}
    print("\n[换老师这一维的读数：按种子配对，不是两个均值相减]")
    for task, a, b in PAIRS:
        d = stats([y - x for x, y in zip(vals[(task, a)], vals[(task, b)])])
        con[f"{task}: {b}−{a}"] = d
        print(f"  {task}（{b} 减 {a}）{d['mean_imgs']:+7.1f} 张  95% CI [{d['ci95_imgs'][0]:+.1f},"
              f" {d['ci95_imgs'][1]:+.1f}]  同号 {d['same_sign']}  p={d['p_binomial']} → "
              f"{'换老师换出了读数' if d['excludes_zero'] else '换老师没换出读数（这一维测不动）'}")

    key_d = "MNIST: D−A"
    flip_kept = st["MNIST×D"]["excludes_zero"] and st["MNIST×D"]["mean_imgs"] > 0
    teacher_seen = con[key_d]["excludes_zero"]
    both_neg = all(st[f"Fashion×{k}"]["mean_imgs"] < 0 for k in ("A", "dry"))
    if flip_kept:
        verdict = ("H_任务 拿到一票：换到欠训练老师 D，MNIST 低数据档的加热正号仍在"
                   "（两副老师同号且各自 CI 不含 0）")
    elif teacher_seen:
        verdict = (f"翻号收进『MNIST × 老师 A 的那份软标签』：换到同档更湿的老师 D，"
                   f"正号从 {st['MNIST×A']['mean_imgs']:+.1f} 塌到 {st['MNIST×D']['mean_imgs']:+.1f} 张"
                   f"（CI 跨 0 = 这一格读不出方向），而两副老师之差按种子配对是 "
                   f"{con[key_d]['mean_imgs']:+.1f} 张、CI 不含 0 ⇒ 老师这一维有读数。"
                   f"『符号由数据量决定』在 MNIST 内部就站不住，任务级的说法降级为"
                   f"『数据量 × 这份软标签』的交互；而 Fashion 那两格"
                   f"{'两格全负' if both_neg else '并非两格全负'}：换任务连符号都不跟")
    else:
        verdict = ("换老师后仍测不动，且两副老师之差的 CI 也跨 0 ⇒ 这一发没买到判别量，"
                   "两说都没判决，先补老师或补流")
    print(f"\n[判决] {verdict}")

    lim = ["每格 8 条流只覆盖同一副学生、同一个数据档、同一对温度（T=1/T=6）：这是"
           "『四角齐全』，不是『四维齐全』——中间档与 α 轴不在这张表里",
           "MNIST×D 的底是借来的 37 张（v5_d 现读）：两副老师共用一把尺，配对 CI 不吃这个底，"
           "但单格读数拿它当闸门时要记得它是老师 A 那侧量的",
           "跨任务那一行不是配对的：同名种子在两个数据集里是两条不同的流，只报差值不报差的 CI",
           "老师只有两副、且 D 的精度低于 A（94.05% vs 97.98%）：拆开的是知识源，"
           "拆不掉老师强弱这一维（与第5课同一句代价）",
           "两行换老师的**干湿方向是相反的**（MNIST：A 塌缩 92.8% → D 49.3%，更湿 −43.5pp；"
           "Fashion：A 42.5% → dry 64.0%，更干 +21.5pp）：所以这两行不是同一假说的两次同向复证，"
           "而是两个不同方向的各一次读数——MNIST 那一侧『往湿走』把正号抹掉了，"
           "Fashion 这一侧『往干走』没动那个负号（CI 宽度 −46.6~+35.0，测不动不等于没效应）",
           "干湿只各有一副对照老师、且都是单一配方旋钮（轮数）造的：这一发能否掉『符号只跟数据量』，"
           "不能给『符号跟干湿』定斜率"]
    report = {"cells": st, "within_task_paired_contrasts": con, "seed_order": SEEDS,
              "quantity": "低数据档按种子配对的 T 主效应（mean over α 的 gap@T6 − gap@T1），单位张/10000",
              "archives": {f"{t}×{k}": CELLS[(t, k)][0] for (t, k) in CELLS},
              "teacher_softlabel_shift": {
                  "MNIST: A→D": "塌缩 92.8% → 49.3%（更湿 −43.5pp，读自 v5_d 的干湿闸）",
                  "Fashion: A→dry": "塌缩 42.5% → 64.0%（更干 +21.5pp，读自 v6b 的干湿闸）"},
              "prediction_ex_ante": "判据写在 step4e 文档串第五发那一段（跑 D 那发之前），此处照抄执行",
              "verdict": verdict, "limitations": lim}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（纯读档：0 次训练，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="任务 × 老师 2×2 低数据档裁决：纯读四本已入库的账")
    ap.add_argument("--red-check", action="store_true",
                    help="验红：故意读错一本账，确认档位/温度列断言会红着退出（不写档案）")
    main(red_check=ap.parse_args().red_check)
