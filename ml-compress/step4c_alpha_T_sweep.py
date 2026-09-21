# -*- coding: utf-8 -*-
"""第4课的第三笔债：把「α=0.7、T=3 单点」这条边界，从怀疑变成判决。

第4课测到 KD 显著有害（−0.63pp，种子层也扛住了），但那一轮只取了一组超参，
limitations 里自己写着「α/T 未扫参」。这一课补齐，并且带着两个**互相打架**的假设去扫：

  H1 稀释说：软标签本身没错，错在 α=0.7 把 70% 的权重给了它、抢走了硬标签的信号。
     ⇒ 伤害 ∝ α（α=0.3 时约 −27 张），而 T 轴是平的（每列内部不分高低）。
  H2 空洞说：第4课 Q3a 量到老师的软标签≈one-hot（测试集 92% 的图 top1>0.99），
     KL 项里没有 CE 给不出的信息，纯粹是噪声源。温度才是唯一能把暗知识放出来的旋钮。
     ⇒ 伤害随 T↑ 收缩（T=6 时暗知识真正上桌），α 轴的斜率反而没那么要紧。

两说的可判别处就在 **α=1.0 那一列的 T 轴**：H1 说那一列平（都是"没用硬标签"），
H2 说那一列随 T 涨。顺带还能回答第4课留的那个"如果 α→0 伤害就消失"的反事实。

但扫参之前先要拆一颗地雷：α 和 T **不是正交的两个旋钮**。标准写法里 KL 项乘了 T²，
这个补偿因子只要不精确，升 T 就在同时改 KD 项的有效权重——那"温度越高越坏"里就掺着
"权重越偏越坏"，H1 会以借温度之名的形式混进 T 轴的读数里。所以本课先做一次零训练的
梯度范数探针（loss 两项各自 backward，看 KD 梯度随 T 漂多少），漂移出闸门就补一组
**配平臂**：每个温度用自己那个能把 KD 梯度钉回锚点水平的 α，再看 T 轴还斜不斜。

仪器规矩沿用前几课：同种子配对（CE 与 KD 每个种子一副权重同起点）、逐图符号检验、
幅度只跟**种子底**比（底不硬编码，从对应宽度的 compress_report_v4b*.json 读——锚点档案值、
H1 的伤害基数、种子底三样全都跟着宽度走，S64 那一份是 0.9224/776、−55 张、底 53 张）。

`--width 64` 是后补的一发：S128 那张表判出"伤害的旋钮是 T 不是 α"，但那只在一副学生上成立。
这一发问的是**这条归因跨不跨架构**（同一老师、同一批数据、同一份软标签，只换学生宽度；
所以同号=结论不随学生宽度变，不等于换个数据集还成立）。
零 API 费，纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4c_alpha_T_sweep.py
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4c_alpha_T_sweep.py --width 64
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (BS, KD_ALPHA, KD_T, SEED, TIERS, VAL_N, build_mlp, logits_of,
                           sign_test_p, soft_label_audit, train_student)
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
TIER = "满数据8500"                                 # 只扫主档：两档一起扫翻倍，且低数据那一档
WIDTH = 128                                         # 默认副；已被第4课b 降级为"不显著"的低数据档不追
ALPHAS = (0.3, 0.7, 1.0)
TEMPS = (1.0, 3.0, 6.0)
SEEDS = (42, 7, 2026)
ALPHA_ANCHOR = KD_ALPHA                             # 0.7：第4课那格的权重分配
ANCHOR_T = KD_T                                     # 3.0：探针的基准温度，也是锚点格
T2_TOL = 0.15                                       # KD 梯度范数随 T 漂移超过 ±15% 就算 T 轴被污染


def v4b_path(width: int) -> Path:
    return HERE / ("compress_report_v4b.json" if width == 128 else f"compress_report_v4b_s{width}.json")


def out_path(width: int) -> Path:
    return HERE / ("compress_report_v4c.json" if width == 128 else f"compress_report_v4c_s{width}.json")


def load_anchor(v4b: Path, width: int) -> dict:
    """锚点档案值 + 种子底 + H1 的伤害基数，三样全从第4课b 那一副宽度的报告读。

    为什么不写第二套字面量：换一副宽度这些数字**全都变**（S64 的锚是 0.9224/776/128/73、
    伤害基数 −55 张、底 53 张），而"对账"这件事的效力完全来自它比的是档案真值。
    抄一份数字进代码，档案一改就对上老账了——所以档案是唯一来源，这里只负责读。"""
    d = json.loads(v4b.read_text(encoding="utf-8"))
    if int(d["width"]) != width:
        raise SystemExit(f"{v4b.name} 是 S{d['width']} 的报告，不是 S{width}——换宽度请连档案一起换")
    arm = d["arms"][f"{TIER} 种子{SEEDS[0]}"]        # α=0.7/T=3 那一格，种子 42
    return {"kd": {"acc": arm["acc_kd"], "wrong": arm["wrong_kd"],
                   "broken": arm["flips"]["broken"], "fixed": arm["flips"]["fixed"]},
            "ce_acc": arm["acc_ce"],
            "gap_imgs": arm["gap_imgs"],            # KD−CE，负=KD 更差，与本课 gap 同号向
            "floor": int(d["verdicts"][TIER]["seed_floor_imgs"]["KD"])}


def h1_pred(gap_imgs: int) -> dict:
    """H1 的可打数字：把锚点那一格的伤害按 α 线性缩放（α/α_anchor × gap）。"""
    return {a: round(a / ALPHA_ANCHOR * gap_imgs) for a in ALPHAS}


def key(a: float, t: float) -> str:
    return f"a{a:g}_T{t:g}"


def term_grad_norms(xd, yd, soft, t: float, width: int, seed: int = SEED) -> dict:
    """零训练探针：同一副初始权重、同一批数据，把 loss 两项各自 backward，
    量它们真正作用到学生参数上的梯度 L2 范数（KL 按标准写法带 T² 补偿）。

    为什么训练前必须量这个：α 和 T 不是正交的两个旋钮——T² 这个补偿因子只要不精确，
    升 T 就会顺带把 KD 项的**有效权重**抬起来。那样"温度越高越坏"里就掺着
    "权重分配越偏越坏"（H1 借温度之名发生），事后再也分不开。"""
    torch.manual_seed(seed)
    m = build_mlp(width)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(xd), (BS,), generator=gen)
    z = m(xd[idx])
    ce = nn.functional.cross_entropy(z, yd[idx])
    kl = nn.functional.kl_div(torch.log_softmax(z / t, dim=1),
                              torch.softmax(soft[idx] / t, dim=1),
                              reduction="batchmean") * t ** 2
    ps = list(m.parameters())
    g_ce = torch.autograd.grad(ce, ps, retain_graph=True)
    g_kd = torch.autograd.grad(kl, ps)
    flat = lambda gs: float(torch.cat([g.flatten() for g in gs]).norm())   # noqa: E731
    return {"g_ce": round(flat(g_ce), 5), "g_kd_unit": round(flat(g_kd), 5)}


def main(width: int = WIDTH) -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    v4b = v4b_path(width)
    if not v4b.exists():
        raise SystemExit(f"缺 {v4b.name}——锚点档案值和种子底都从它读，先跑 "
                         f"python step4b_seed_variance.py"
                         + ("" if width == 128 else " --width 64"))
    out = out_path(width)
    anchor = load_anchor(v4b, width)
    floor = anchor["floor"]
    ANCHOR_KD, ANCHOR_CE_ACC = anchor["kd"], anchor["ce_acc"]
    H1_PRED = h1_pred(anchor["gap_imgs"])

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    n_data = TIERS[TIER]
    xd, yd = xt_all[:n_data], torch.tensor(yt_all[:n_data])
    soft = logits_of(teacher, xd)                   # 软标签与 T 无关：温度在 loss 里加，全程共用一份

    print(f"teacher+数据就位 | {TIER} 训练 {n_data} 张 | 测试 {len(yte)} | 学生 S{width} | "
          f"网格 {len(ALPHAS)}×{len(TEMPS)}×{len(SEEDS)} = "
          f"{len(ALPHAS) * len(TEMPS) * len(SEEDS)} 次 KD + {len(SEEDS)} 次 CE 对照")
    print(f"[仪器] 锚点档案值来自 {v4b.name}：α={ALPHA_ANCHOR}/T={ANCHOR_T:g}/种子{SEEDS[0]} "
          f"KD {ANCHOR_KD['acc']}/错{ANCHOR_KD['wrong']}/坏{ANCHOR_KD['broken']}/正{ANCHOR_KD['fixed']}，"
          f"CE {ANCHOR_CE_ACC}，那一格伤害 {anchor['gap_imgs']} 张（H1 按 α 线性缩放它）")
    print(f"[仪器] 种子底（第4课b 的 KD 臂，换随机流的错图数极差）= {floor} 张，"
          f"任何 |均值 gap| ≤ {floor} 张在本机上不算疗效")

    # ---------- 零训练先验：三个温度下老师的软标签到底"软"了多少 ----------
    print("\n[零训练先验] 老师在拟合集上的软标签体检，逐温度：")
    audit_by_t = {}
    for t in TEMPS:
        a = soft_label_audit(soft, t=t)
        audit_by_t[key(1.0, t)] = {"T": t, "原始": a["原始"], "除以T": a["除以T"]}
        o, s = a["原始"], a["除以T"]
        print(f"  T={t:<4} 原始 top1 {o['mean_top1']:.4f}（塌缩 {o['pct_top1_gt_0.99']}%，p2 {o['mean_p2']:.4f}）"
              f" → /T 后 top1 {s['mean_top1']:.4f}（塌缩 {s['pct_top1_gt_0.99']}%，p2 {s['mean_p2']:.4f}）")
    lo, hi = audit_by_t[key(1.0, 1.0)]["除以T"], audit_by_t[key(1.0, TEMPS[-1])]["除以T"]
    print(f"  → 从 T=1 到 T={TEMPS[-1]:g}：top1 {lo['mean_top1']:.4f}→{hi['mean_top1']:.4f}，"
          f"塌缩率 {lo['pct_top1_gt_0.99']}%→{hi['pct_top1_gt_0.99']}%，"
          f"第二名概率 {lo['mean_p2']:.4f}→{hi['mean_p2']:.4f}"
          f"（KL 能拿到的「暗知识」总量就这么多，先记着）")

    # ---------- 零训练探针：T 是温度轴，还是偷偷的权重轴？ ----------
    assert ALPHA_ANCHOR in ALPHAS and ANCHOR_T in TEMPS, "锚点格不在网格里，探针没有基准"
    probe = {t: term_grad_norms(xd, yd, soft, t, width) for t in TEMPS}
    ref_g = probe[ANCHOR_T]["g_kd_unit"] * ALPHA_ANCHOR      # 第4课那格真正推参数的 KD 梯度
    print("\n[零训练探针] 同一副初始权重、同一批数据，loss 两项各自 backward 的梯度 L2 范数：")
    for t in TEMPS:
        p = probe[t]
        g_kd = ALPHA_ANCHOR * p["g_kd_unit"]
        share = g_kd / (g_kd + (1 - ALPHA_ANCHOR) * p["g_ce"])
        p["kd_share_at_anchor_alpha"] = round(share, 3)
        p["g_kd_drift_vs_anchor_pct"] = round((ALPHA_ANCHOR * p["g_kd_unit"] / ref_g - 1) * 100, 1)
        p["alpha_to_match_anchor"] = round(ref_g / p["g_kd_unit"], 3)
        print(f"  T={t:<4} ‖∇CE‖ {p['g_ce']:.4f} | ‖∇(T²·KL)‖ {p['g_kd_unit']:.4f}"
              f"（α 取 {ALPHA_ANCHOR} 时是锚点的 {p['g_kd_drift_vs_anchor_pct']:+.1f}%）"
              f" | KD 份额 {share:.1%} | 把 KD 梯度钉回锚点需要 α={p['alpha_to_match_anchor']}")
    drift = max(abs(probe[t]["g_kd_drift_vs_anchor_pct"]) for t in TEMPS) / 100
    # α>1 会给硬标签项负权重，那是另一个实验而不是对照——配不平的温度直接排除
    for t in TEMPS:
        probe[t]["matchable"] = bool(0.0 <= probe[t]["alpha_to_match_anchor"] <= 1.0)
    unmatched = [t for t in TEMPS if not probe[t]["matchable"]]
    matched = ([t for t in TEMPS if probe[t]["matchable"]
                and abs(probe[t]["alpha_to_match_anchor"] - ALPHA_ANCHOR) > 0.05]
               if drift > T2_TOL else [])
    print(f"  → T² 补偿的精确度：网格内 KD 梯度最大漂移 {drift:.1%}（闸门 ±{T2_TOL:.0%}）→ "
          + ("T 轴是干净的单变量，不必配平" if not matched
             else f"T 轴同时是条权重轴，加跑配平臂 {matched}：每 T 用自己的 α 把 KD 梯度钉回锚点水平")
          + (f"；配不平（需要 α>1）的温度：{unmatched}" if unmatched else ""))

    # ---------- 训练之前把两说的数字写死，之后不许改口 ----------
    pred_lines = []
    for a in ALPHAS:
        h1 = H1_PRED[a]
        pred_lines.append({"alpha": a, "h1_gap_imgs": h1,
                           "h1_expect_T_flat": f"该 α 三个 T 之差 ≤ 种子底 {floor} 张"})
    print("\n[先立预言（此后不许改口）]")
    print(f"  H1 稀释说 → α 轴斜率：{pred_lines}")
    print(f"  H1 还说 α=1.0 那一列**平**：T=1 与 T=6 的 gap 差 ≤ {floor} 张（都没用硬标签，一样惨）")
    print(f"  H2 空洞说 → α=1.0 那一列**随 T 涨**：T={TEMPS[-1]:g} 比 T=1 至少收回 {floor} 张以上；"
          f"且 α 轴不该更陡（既然锅在软标签空洞，给多少权重都是噪声）")
    print(f"  两说的公共判决点：α={ALPHAS[0]} T=3 若 |gap| < {floor} 张 → 伤害主要来自权重分配（H1 侧）")
    if matched:
        print(f"  配平臂（每个漂移出闸门的温度用自己的 α 把 KD 梯度钉回锚点水平，T={matched}）："
              f"钉回去若把该温度相对锚点的**额外伤害**收回超过一个底（{floor} 张）"
              f"→ 第4课读到的高 T 之害主要是 T² 把 KD 项抬了权重，H1 以「温度伪装成权重」的形式复活；"
              f"收不回来（≤ {floor} 张）→ 权重解释不了，伤害跟着温度本身走")

    # ---------- 各臂：CE 是同种子的锚，每个 KD 格逐格配对 ----------
    specs = [(key(a, t), a, t) for a in ALPHAS for t in TEMPS]
    specs += [(f"match_T{t:g}", probe[t]["alpha_to_match_anchor"], t) for t in matched]
    ce_ref: dict[int, dict] = {}
    cells: dict[str, list[dict]] = {k: [] for k, _, _ in specs}
    print(f"\n[开跑] 每个种子一副 CE + {len(specs)} 副 KD（同种子同初始权重、同批次流，只差目标函数）：")
    for seed in SEEDS:
        ce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
        ce_pred = predict(ce, xte, False)
        ce_wrong = int((ce_pred != yte).sum())
        ce_ref[seed] = {"acc": round(1 - ce_wrong / len(yte), 4), "wrong": ce_wrong}
        line = [f"  种子 {seed:<5} CE 错 {ce_wrong} 张"]
        mline = []
        for k, a, t in specs:
            m, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=a, t=t)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            broken = int(((ce_pred == yte) & (p != yte)).sum())
            fixed = int(((ce_pred != yte) & (p == yte)).sum())
            cells[k].append({"seed": seed, "alpha": a, "T": t,
                             "wrong_kd": wrong, "wrong_ce": ce_wrong,
                             "acc_kd": round(1 - wrong / len(yte), 4),
                             "gap_imgs": ce_wrong - wrong,          # 正=KD 比 CE 少错
                             "broken": broken, "fixed": fixed,
                             "p_sign": round(sign_test_p(broken, fixed), 6)})
            (mline if k.startswith("match_") else line).append(f"{k}:{ce_wrong - wrong:+d}")
        print(" ".join(line))
        if mline:
            print(" " * 12 + "（配平臂）" + "  ".join(mline))
        if seed == SEEDS[0]:
            r = cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]
            ok = (abs(r["acc_kd"] - ANCHOR_KD["acc"]) < 5e-5 and r["wrong_kd"] == ANCHOR_KD["wrong"]
                  and r["broken"] == ANCHOR_KD["broken"] and r["fixed"] == ANCHOR_KD["fixed"]
                  and ce_ref[seed]["acc"] == ANCHOR_CE_ACC)
            print(f"  [锚点逐位对账 α={ALPHA_ANCHOR} T={ANCHOR_T} 种子{SEEDS[0]}] KD {r['acc_kd']:.4f}"
                  f"/错{r['wrong_kd']}/坏{r['broken']}/正{r['fixed']} vs 档案 {ANCHOR_KD} → "
                  + ("通过：本课的 loss 就是第4课那个 loss，只是多了两个旋钮" if ok else "**不通过：全部结论作废**"))

    # ---------- 汇总与判决（分支算出来的，不是写死的） ----------
    grid = {}
    for k, rows in cells.items():
        gaps = [r["gap_imgs"] for r in rows]
        tb = sum(r["broken"] for r in rows)
        tf = sum(r["fixed"] for r in rows)
        grid[k] = {"per_seed": rows, "gaps_imgs": gaps,
                   "mean_gap_imgs": round(sum(gaps) / len(gaps), 1),
                   "range_imgs": max(gaps) - min(gaps),
                   "same_dir_as_h1": sum(1 for g in gaps if g < 0),
                   "pooled": {"broken": tb, "fixed": tf},
                   "p_pooled": round(sign_test_p(tb, tf), 8)}
    at = lambda a, t: grid[key(a, t)]["mean_gap_imgs"]      # noqa: E731

    print("\n[网格] 每格三个种子的 KD−CE 平均（张，负=KD 更差）；括号内是该格跨种子极差：")
    print("        T=" + "        ".join(f"{t:<7g}" for t in TEMPS))
    for a in ALPHAS:
        print(f"  α={a:<4} " + "        ".join(f"{at(a, t):+6.1f}({grid[key(a, t)]['range_imgs']:2d})"
                                              for t in TEMPS))

    verdicts: dict[str, dict | str] = {}
    a_max = ALPHAS[-1]
    h1_resid = [abs(at(a, ANCHOR_T) - H1_PRED[a]) for a in ALPHAS]
    alpha_slope = at(a_max, ANCHOR_T) - at(ALPHAS[0], ANCHOR_T)   # α 从 0.3 到 1.0，gap 往哪走
    verdicts["H1_alpha_axis"] = {
        "pred": {str(a): H1_PRED[a] for a in ALPHAS},
        "measured_anchor_T": {str(a): at(a, ANCHOR_T) for a in ALPHAS},
        "alpha_slope_imgs": round(alpha_slope, 1),
        "abs_error_vs_pred": [round(x, 1) for x in h1_resid],
        "monotone_worse_with_alpha": bool(at(ALPHAS[0], ANCHOR_T) > at(ALPHA_ANCHOR, ANCHOR_T)
                                          > at(a_max, ANCHOR_T)),
        "slope_beats_seed_floor": bool(abs(alpha_slope) > floor)}
    col_t = [at(a_max, t) for t in TEMPS]
    verdicts["H1_or_H2_alpha_max_T_column"] = {
        "measured": {str(t): at(a_max, t) for t in TEMPS},
        "T_span_imgs": round(max(col_t) - min(col_t), 1),
        "T_max_minus_T_min_imgs": round(at(a_max, TEMPS[-1]) - at(a_max, TEMPS[0]), 1),
        "flat_as_H1_says": bool(abs(at(a_max, TEMPS[-1]) - at(a_max, TEMPS[0])) <= floor),
        "rises_as_H2_says": bool(at(a_max, TEMPS[-1]) - at(a_max, TEMPS[0]) > floor)}
    t_span_all = [round(sum(at(a, TEMPS[-1]) for a in ALPHAS) / len(ALPHAS)
                        - sum(at(a, TEMPS[0]) for a in ALPHAS) / len(ALPHAS), 1)]
    steps = {}
    for t_lo, t_hi in zip(TEMPS[:-1], TEMPS[1:]):
        d = round(sum(at(a, t_hi) - at(a, t_lo) for a in ALPHAS) / len(ALPHAS), 1)
        steps[f"{t_lo:g}→{t_hi:g}"] = {"imgs": d, "in_seed_floor_units": round(abs(d) / floor, 2)}
    verdicts["T_main_effect"] = {"T_max_minus_T_min_pooled_over_alpha": t_span_all[0],
                                 "beats_seed_floor": bool(abs(t_span_all[0]) > floor),
                                 "adjacent_steps": steps,
                                 "where_the_damage_lives": min(steps, key=lambda k: steps[k]["imgs"])}
    v_pure = at(a_max, TEMPS[0])
    verdicts["pure_KD_low_T_vs_CE"] = {"mean_gap_imgs": v_pure,
                                       "vs_floor": round(abs(v_pure) / floor, 2),
                                       "sign": "KD 更好" if v_pure > 0 else "CE 更好",
                                       "counts_as_effect": bool(abs(v_pure) > floor)}
    v_low = at(ALPHAS[0], ANCHOR_T)
    verdicts["alpha_03_at_anchor_T"] = {"mean_gap_imgs": v_low,
                                        "vs_floor": round(abs(v_low) / floor, 2),
                                        "survives_seed": bool(abs(v_low) > floor),
                                        "reading": "α→0 方向上伤害没消失" if abs(v_low) > floor
                                        else "α→0 伤害消失"}
    free = {k: g for k, g in grid.items() if not k.startswith("match_")}
    best = max(free, key=lambda k: free[k]["mean_gap_imgs"])
    worst = min(free, key=lambda k: free[k]["mean_gap_imgs"])
    beating = [k for k, g in free.items() if g["mean_gap_imgs"] > 0]
    verdicts["grid_extremes"] = {"best": {"cell": best, "mean_gap_imgs": grid[best]["mean_gap_imgs"]},
                                 "worst": {"cell": worst, "mean_gap_imgs": grid[worst]["mean_gap_imgs"]},
                                 "cells_beating_ce": sorted(beating),
                                 "n_cells": len(free)}
    if matched:
        # 配平判的是"把 KD 梯度钉回锚点后，这一档自己的伤害还剩下多少"——
        # 不是"配平各档之间的跨度"（漂移出闸门的往往只有一档，跨度恒为 0 会读成"压平"）
        per_T = {str(t): {"alpha_matched": probe[t]["alpha_to_match_anchor"],
                          "gap_at_anchor_alpha": at(ALPHA_ANCHOR, t),
                          "gap_gradient_matched": grid[f"match_T{t:g}"]["mean_gap_imgs"],
                          "damage_recovered_imgs": round(grid[f"match_T{t:g}"]["mean_gap_imgs"]
                                                         - at(ALPHA_ANCHOR, t), 1)}
                 for t in matched}
        recovered = round(sum(v["damage_recovered_imgs"] for v in per_T.values()) / len(per_T), 1)
        verdicts["weight_or_temperature"] = {
            "per_T": per_T, "mean_damage_recovered_imgs": recovered,
            "weight_explains_the_T_dose": bool(recovered > floor),
            "reading": ("钉回梯度后伤害大部分回来了（回收 "
                        f"{recovered} 张 > 底 {floor} 张）→ 高 T 之害主要来自 T² 抬权重，H1 借温度复活"
                        if recovered > floor else
                        f"钉回梯度只换回 {recovered} 张（底 {floor} 张）→ 权重解释不了这档伤害，"
                        "锅在温度本身：软化的目标分布真把学生带偏了")}
    else:
        verdicts["weight_or_temperature"] = {
            "max_kd_grad_drift_pct": round(max(abs(probe[t]["g_kd_drift_vs_anchor_pct"])
                                                for t in TEMPS), 1),
            "gate_pct": round(T2_TOL * 100, 1),
            "reading": "T² 补偿在闸门内，T 轴是干净的单变量：网格里的 T 斜率就是温度的作用"}

    print("\n[判决]")
    v = verdicts["H1_alpha_axis"]
    print(f"  ① α 轴（H1 的主张）：α={ALPHAS} 在锚点温度实测 {v['measured_anchor_T']}，"
          f"预测 {v['pred']}，|误差| {v['abs_error_vs_pred']}；"
          f"单调变坏 {v['monotone_worse_with_alpha']}，斜率 {v['alpha_slope_imgs']} 张"
          f"（种子底 {floor} 张）→ {'α 轴是真信号' if v['slope_beats_seed_floor'] else 'α 轴在底以下，谈不上斜率'}")
    v = verdicts["H1_or_H2_alpha_max_T_column"]
    print(f"  ② **判别格** α={a_max} 那一列 {v['measured']}：最高温比最低温差 {v['T_max_minus_T_min_imgs']} 张，"
          f"列内极差 {v['T_span_imgs']} 张 → 平（H1 赢）={v['flat_as_H1_says']}／随 T 涨（H2 赢）={v['rises_as_H2_says']}")
    v = verdicts["T_main_effect"]
    print(f"  ③ T 轴主效应（{len(ALPHAS)} 列平均）：高温 − 低温 = {v['T_max_minus_T_min_pooled_over_alpha']} 张，"
          f"压过种子底={v['beats_seed_floor']}；逐档走一步："
          + "、".join(f"{k} {d['imgs']:+.1f} 张（{d['in_seed_floor_units']}x 底）"
                      for k, d in v["adjacent_steps"].items())
          + f" → 伤害主要住在 {v['where_the_damage_lives']} 这一步")
    v = verdicts["pure_KD_low_T_vs_CE"]
    print(f"  ④ α={a_max}/T={TEMPS[0]:g}（只用老师 argmax 的软标签、不加热）：{v['sign']}，gap {v['mean_gap_imgs']} 张 = "
          f"种子底的 {v['vs_floor']}x，算疗效={v['counts_as_effect']}")
    v = verdicts["alpha_03_at_anchor_T"]
    print(f"  ⑤ 第4课那句反事实「α→0 伤害就消失」：α={ALPHAS[0]}/T={ANCHOR_T:g} 实测 {v['mean_gap_imgs']} 张 = 底的 "
          f"{v['vs_floor']}x → {v['reading']}")
    v = verdicts["grid_extremes"]
    print(f"  ⑥ {v['n_cells']} 格里最优点 {v['best']['cell']}（{v['best']['mean_gap_imgs']:+.1f} 张）、"
          f"最差 {v['worst']['cell']}（{v['worst']['mean_gap_imgs']:+.1f} 张）；"
          f"跑赢 CE 的格数 {len(v['cells_beating_ce'])}/{v['n_cells']}"
          f"{'' if not v['cells_beating_ce'] else '：' + '、'.join(v['cells_beating_ce'])}")
    v = verdicts["weight_or_temperature"]
    if "per_T" in v:
        print(f"  ⑦ 配平臂（α 由探针算出，把 KD 梯度钉回锚点水平）："
              + "；".join(f"T={k} α={d['alpha_matched']}：α={ALPHA_ANCHOR} 时 {d['gap_at_anchor_alpha']:+.1f} 张 → "
                          f"配平后 {d['gap_gradient_matched']:+.1f} 张（换回 {d['damage_recovered_imgs']:+.1f}）"
                          for k, d in v["per_T"].items())
              + f" → {v['reading']}")
    else:
        print(f"  ⑦ 权重/温度解耦探针：KD 梯度随 T 最大漂移 {v['max_kd_grad_drift_pct']}%"
              f"（闸门 ±{v['gate_pct']}%）→ {v['reading']}")

    print("\n[自我审计① 底是别课的] 种子底取自第4课b 的 KD 臂（α=0.7/T=3），本课改的是目标函数的两个系数，"
          "严格说各格自己的种子底要各扫各的；这里用一格的底当全表的闸，是**借来的仪器**，会低估 T 大端的方差。")
    print(f"[自我审计② 多重比较] {len(free)} 格 × 三种子，p_pooled 只看方向不看在底之上（同一批 10000 张测试图，"
          f"种子之间翻牌事件不独立，p 偏乐观）。全表只有一件事是硬判决：{floor} 张这条线。")
    print(f"[自我审计③ 只扫了满数据档与 S{width}] 低数据那一档第4课b 已降级为不显著；"
          "若 H1/H2 想在低数据档有不同答案，本课没测。")
    if width != WIDTH:
        print(f"[自我审计③ b 跨架构 ≠ 独立复证] 这一副 S{width} 与默认那副 S{WIDTH} 共用**同一个老师、同一批数据、"
              "同一份软标签**，只有学生那侧换了。所以两份报告若同号，说明的是「结论不随学生宽度变」，"
              "而不是「换个数据集还成立」——数据那侧的独立性要到第5课（另一副 teacher）才有。")
    print("[自我审计④ 探针只管到梯度幅值] 权重/温度解耦靠的是初始那一步、一个批次、一副初始权重上"
          "两项梯度的 L2 范数之比；训练过程中两项的相对尺度会随学生逼近老师而变（KL 变小），"
          "所以配平是**起手配平**，不是全程配平。")
    print(f"[自我审计⑤ 探针顺手量到 T² 补偿过头了] KD 项梯度范数随 T："
          + " → ".join(f"T{t:g} {probe[t]['g_kd_unit']:.4f}" for t in TEMPS)
          + f"（锚点 α={ALPHA_ANCHOR} 下占 {probe[TEMPS[0]]['kd_share_at_anchor_alpha']:.0%} → "
          f"{probe[TEMPS[-1]]['kd_share_at_anchor_alpha']:.0%}）——教科书说「α 管权重、T 管温度」，"
          "在这个实现里不成立：升 T 确实在抬 KD 的实际话语权。低温端反而比锚点低，"
          f"α≤1 配不平（需要 α={probe[TEMPS[0]]['alpha_to_match_anchor']}），"
          "这一格按「不可配平」记账而不是悄悄删掉。")

    other = 64 if width == 128 else 128
    report = {"tier": TIER, "width": width, "alphas": list(ALPHAS), "temps": list(TEMPS),
              "seeds": list(SEEDS), "test_n": int(len(yte)), "train_fit_n": cut,
              "anchor_hyperparams": {"alpha": ALPHA_ANCHOR, "T": ANCHOR_T},
              "anchor_archive_from": v4b.name,
              "seed_floor_imgs_from_v4b": floor,
              "soft_label_audit_by_T": audit_by_t,
              "gradient_probe": {"t2_tolerance_pct": round(T2_TOL * 100, 1),
                                 "matched_temps": matched, "unmatchable_temps": unmatched,
                                 "by_T": probe},
              "predictions_ex_ante": {"H1_alpha_axis": pred_lines,
                                      "H1_alpha_max_T_column_flat": True,
                                      "H2_alpha_max_T_column_rises": True,
                                      "H1_pred_gap_imgs": {str(a): H1_PRED[a] for a in ALPHAS}},
              "parity_anchor": {"this_run": {"acc": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["acc_kd"],
                                             "wrong": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["wrong_kd"],
                                             "broken": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["broken"],
                                             "fixed": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["fixed"]},
                                "archive": ANCHOR_KD,
                                "ce_acc_this_run": ce_ref[SEEDS[0]]["acc"],
                                "ce_acc_archive": ANCHOR_CE_ACC},
              "ce_ref_by_seed": ce_ref, "grid": grid, "verdicts": verdicts,
              "limitations": ["种子底借自第4课b 的 α=0.7/T=3 一格，未逐格重量",
                              f"只扫满数据 8500 + S{width}；低数据档未扫，另一副宽度是"
                              f"{out_path(other).name}（各写各的表，跨架构结论要两份对照着读）",
                              "每格 3 个种子，均值本身没有二次底",
                              "权重/温度解耦只在初始一步的一个批次上配平，不是全程配平",
                              "p_pooled 偏乐观（三种子共用同一批测试图）"]}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{len(SEEDS) * (len(cells) + 1)} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="α×T 扫参归因（默认 S128；--width 64 测结论跨不跨架构）")
    ap.add_argument("--width", type=int, choices=(128, 64), default=WIDTH,
                    help="学生宽度；锚点档案值/伤害基数/种子底都从该宽度的 compress_report_v4b*.json 读")
    main(ap.parse_args().width)
