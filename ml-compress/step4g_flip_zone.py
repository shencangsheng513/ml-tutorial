# -*- coding: utf-8 -*-
"""第4课g 夹逼翻号点：在 2000 与 4500 之间补三档，问 T 轴的符号是**斜坡**还是**台阶**。

三个已入库读数把符号钉成了一条会翻的轴（同一副学生 S128、同一个老师 A）：
  2000 张 → +56.4（加热有益） | 4500 → −57.3 | 8500 → −124.3 张（加热有害）
两端之间只隔 2.25 倍数据量，而"数据量决定符号"这句话**没说自己是怎么变的**。只测两端等于在曲线上
画两个点就说知道它弯；补三刀，才分得清"连续调制"和"存在门槛"。

训练之前把两说的数字写死（此后不许改口）：
  H_斜坡：按 n 单调下降、按两端线性内插 ⇒ 斜率 (−57.3−56.4)/2500 = −0.0455 张/张，
          **预言 2500≈+33.7、3000≈+10.9、3500≈−11.8**，三档绝对值都明显小于两端的 ~56。
  H_台阶：门槛之前贴住 +56 量级、之后贴住 −57 量级 ⇒ 出现 |读数| ≥ 40 张且偏离内插超过本档底。
判别量是**幅度住在哪儿**，不是符号（两说可以给出同一个符号）。

三条仪器规矩（前几课的债换来的）：
  ① 底不借：每档现算自己的底 = 同目标函数只换随机流的错图数极差，取 CE 臂与四格 KD 臂的最大值
     （第4课d 量过"借来的底对单格读数偏大、会埋效应"）。
  ② α 只留 0.7/1.0：判别量是 T 的符号与幅度，α 轴已在四份档案里量平（斜率 4.7 张 = 底 0.11x）。
     与两端锚点的"三档 α 平均"那点口径差 ≤ α 全轴斜率的一半（≈2 张），写在 limitations 里。
  ③ 已入库拒绝重跑：档案 compress_report_v4g.json。

零费、纯 CPU（45 次学生训练）。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4g_flip_zone.py

第二发（`--sample disjoint`）还的是第一发自己点名的那笔账：自我审计④ 写着"2000/2500/3000/3500/4500
是同一批图的**嵌套前缀**，不是独立样本 ⇒ 只读形状、不做回归"，而这一发的判决恰好是"0/3 过闸"。
一个全在底以内的读数，最需要的就是知道它是不是"这批图"的特产。

先算账再动手（这笔账决定设计，不决定措辞）：要让各档**互不重叠**，三档需要
2500+3000+3500 = 9000 张；而整条曲线（含两个端点）已经吃掉了池子的前 4500 张、验证尾占掉最后
1500 张 ⇒ 10000 张的池子里只剩 **4000 张**没被用过。所以"档档独立"在这个池子上**买不起**，
硬做只会做出一条各档仍然互相重叠、却声称独立的假账。能买得起的是把独立性做在**整条曲线**
这一层：拿那 4000 张重造一条同形状的两端内插曲线（各档仍是这条曲线内部的嵌套前缀 —— 这是
"曲线"的定义，不是缺陷），于是它与第一条曲线的**每一档都不共享一张图**。它回答的问题是：
"三档全在底以内"是这批图的特产，还是这条轴在这个区间上本来就读不出方向。
两个自动推出来的边界（不写字面量）：下界 = 已入库曲线上端点吃过的最大档
（`max(DATA_TIERS 里非满数据档)` = 4500），上界 = 池子去掉验证尾（`len − VAL_N` = 8500）。
代价逐条入账：① 这 4000 张与**满数据档那 8500 张**必然重叠（池子只剩这么多），所以这条曲线
不能和满数据档并排声明独立性，只能和 2000~4500 那条翻号区间并排；② 换样本同时换了"抽到哪些图"，
两条曲线的档位差不再组成匹配 ⇒ 只能整条比整条，不能逐档回归拼起来。
产物另起一名 `compress_report_v4g_disjoint.json`（**档案名跟着判别变量走**，否则第二次跑
会无声盖掉第一次那条"夹紧失败"的判决）。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4g_flip_zone.py --sample disjoint
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import DATA_TIERS, VAL_N, logits_of, sign_test_p, soft_label_audit, train_student
from step4c_alpha_T_sweep import TIER as FULL_TIER
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
LO_TIER, HI_TIER = "低数据2000", "中数据4500"        # 两个端点锚：从各自档案读，不抄字面量
FLIP_TIERS = {"低2500": 2500, "低3000": 3000, "低3500": 3500}
ALPHAS = (0.7, 1.0)
TEMPS = (1.0, 6.0)                                   # 只测两端：主效应就是这两端之差
SEEDS = (42, 7, 2026)
WIDTH = 128
D4 = 1.693
STEP_LINE = 40.0                                     # 台阶说的"贴住端点"线（张）
SAMPLES = ("prefix", "disjoint")


def out_path(sample: str) -> Path:
    """档案名跟着**取样方式**走：第一本保持原名（旧账不改名），第二本另起一名。
    两本共用一个名字的话，第二条曲线会无声盖掉第一条那条"夹紧失败"的判决。"""
    return HERE / ("compress_report_v4g.json" if sample == "prefix"
                   else f"compress_report_v4g_{sample}.json")


def free_region(cut: int) -> tuple[int, int]:
    """disjoint 曲线只准用**已入库曲线任何一档都没训过**的图。下界 = 非满数据档里最大的那一档
    （翻号区间的上端点吃过 4500 张），上界 = 池子去掉验证尾。两个数都从常量推出来：
    写死 4500/8500 的话，将来加一档端点，这条"独立"曲线就会悄悄开始重用它说要避开的图。"""
    return max(n for t, n in DATA_TIERS.items() if t != FULL_TIER), cut


def data_slice(xt_all, yt_all, sample: str, n: int, cut: int):
    lo = 0 if sample == "prefix" else free_region(cut)[0]
    if sample != "prefix" and lo + n > cut:
        raise SystemExit(f"{sample} 档要 {n} 张，而池子里没被任何已入库档训过的只剩 {cut - lo} 张："
                         f"要么换更小的档，要么扩池——扩池会改老师拟合集的定义，那是另一课")
    return xt_all[lo:lo + n], torch.tensor(yt_all[lo:lo + n])


def key(a: float, t: float) -> str:
    return f"a{a:g}_T{t:g}"


def rng(xs: list) -> int:
    return max(xs) - min(xs)


def anchors() -> dict:
    """两端锚点的 T 主效应、当年用的底、α 口径，全部从入库档案现读。
    写死字面量的话，档案一改判词照样打印得出来——只是没了意义。"""
    src = {LO_TIER: "compress_report_v4c_low.json", HI_TIER: "compress_report_v4c_mid.json"}
    out = {}
    for tier, fname in src.items():
        d = json.loads((HERE / fname).read_text(encoding="utf-8"))
        if int(d["width"]) != WIDTH or d["tier"] != tier:
            raise SystemExit(f"{fname} 自称 S{d.get('width')}/{d.get('tier')!r}，"
                             f"本课当 {tier} 用——先看一眼档案再跑")
        out[tier] = {"t_main_imgs": float(d["verdicts"]["T_main_effect"]
                                          ["T_max_minus_T_min_pooled_over_alpha"]),
                     "alphas": [float(a) for a in d["alphas"]],
                     "borrowed_floor_imgs": int(d["seed_floor_imgs_from_v4b"]),
                     "n_imgs": DATA_TIERS[tier], "from": fname}
    return out


def main(sample: str = "prefix") -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    out = out_path(sample)
    if out.exists():
        raise SystemExit(f"{out.name} 已入库——重跑会盖掉那本账。要重跑先删或改名")
    anc = anchors()
    lo, hi = anc[LO_TIER]["t_main_imgs"], anc[HI_TIER]["t_main_imgs"]
    n_lo, n_hi = anc[LO_TIER]["n_imgs"], anc[HI_TIER]["n_imgs"]
    slope = (hi - lo) / (n_hi - n_lo)
    pred = {t: round(lo + slope * (n - n_lo), 1) for t, n in FLIP_TIERS.items()}

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    free_lo, _ = free_region(cut)
    seg_lo = 0 if sample == "prefix" else free_lo
    print(f"[取样] {sample}：这条曲线用池子里第 {seg_lo}~{seg_lo + max(FLIP_TIERS.values())} 号图"
          f"（验证尾从第 {cut} 号开始，两不相交）")
    print(f"端点锚（档案现读）：{anc[LO_TIER]['from']} {lo:+.1f} / {anc[HI_TIER]['from']} {hi:+.1f} 张"
          f" | 内插斜率 {slope * 1000:+.1f} 张每千张 | 口径：端点 {len(anc[LO_TIER]['alphas'])} 档 α、"
          f"本发 {len(ALPHAS)} 档")
    print("\n软标签干湿（各档拟合集现算，只读不判）：")
    audits = {}
    for tier, n in FLIP_TIERS.items():
        xd, _ = data_slice(xt_all, yt_all, sample, n, cut)
        a = soft_label_audit(logits_of(teacher, xd))["原始"]
        audits[tier] = a
        print(f"  {tier}（{n} 张）：top1 {a['mean_top1']}，塌缩率 {a['pct_top1_gt_0.99']}%，p2 {a['mean_p2']}")

    print("\n[先立预言（此后不许改口）]")
    print(f"  H_斜坡：单调下降 + 按两端内插 = {pred}（三档绝对值都 < 两端的 ~56）")
    print(f"  H_台阶：某一刀直接跳，那档 |T6−T1| ≥ {STEP_LINE:g} 张 且偏离内插 > 本档底")
    print("  判别量 = 幅度住在哪儿，不是符号；别拿符号当裁决")
    if sample != "prefix":
        print(f"  [第二发另立一条，且它才是本课的判别量] 上一发三档全在底以内（0/3 过闸）。换一批不相交的图重跑同形状：")
        print(f"    判据 = 至少一档过本档自己的底 → 上一发那三条读数是**这批图**造成的（它当时不该写成"
              f"「这条轴读不出」）；仍 0/3 过闸 → 夹紧失败是这条轴在这个区间的性质，换样本救不回来。")
        print(f"    内插/台阶那两说在这一发只算附带读数：两端锚来自**另一批图**，跨样本的偏差里混了抽样差异。")

    result: dict[str, dict] = {}
    for tier, n in FLIP_TIERS.items():
        xd, yd = data_slice(xt_all, yt_all, sample, n, cut)
        soft = logits_of(teacher, xd)
        print(f"\n[开跑] {tier} | S{WIDTH} | {len(ALPHAS)}×{len(TEMPS)}×{len(SEEDS)} = "
              f"{len(ALPHAS) * len(TEMPS) * len(SEEDS)} 次 KD + {len(SEEDS)} 次 CE")
        rows: dict[str, list[dict]] = {key(a, t): [] for a in ALPHAS for t in TEMPS}
        ce_wrong: list[int] = []
        for seed in SEEDS:
            ce, _ = train_student(WIDTH, xd, yd, soft, "CE", xv, yv, seed=seed)
            ce_pred = predict(ce, xte, False)
            cw = int((ce_pred != yte).sum())
            ce_wrong.append(cw)
            line = [f"  种子 {seed:<5} CE 错 {cw} 张"]
            for a in ALPHAS:
                for t in TEMPS:
                    m, _ = train_student(WIDTH, xd, yd, soft, "KD", xv, yv,
                                         seed=seed, alpha=a, t=t)
                    p = predict(m, xte, False)
                    wrong = int((p != yte).sum())
                    broken = int(((ce_pred == yte) & (p != yte)).sum())
                    fixed = int(((ce_pred != yte) & (p == yte)).sum())
                    rows[key(a, t)].append({"seed": seed, "alpha": a, "T": t,
                                            "wrong_kd": wrong, "wrong_ce": cw,
                                            "acc_kd": round(1 - wrong / len(yte), 4),
                                            "gap_imgs": cw - wrong,
                                            "broken": broken, "fixed": fixed,
                                            "p_sign": round(sign_test_p(broken, fixed), 6)})
                    line.append(f"{key(a, t)}:{cw - wrong:+d}")
            print(" ".join(line))
        mean = {k: round(sum(r["gap_imgs"] for r in v) / len(v), 1) for k, v in rows.items()}
        wrange = {k: rng([r["wrong_kd"] for r in v]) for k, v in rows.items()}
        grange = {k: rng([r["gap_imgs"] for r in v]) for k, v in rows.items()}
        own_floor = max([rng(ce_wrong)] + list(wrange.values()))
        gate = {k: round(2 * (grange[k] / D4) / math.sqrt(len(SEEDS)), 1) for k in rows}
        t_main = round(sum(mean[key(a, TEMPS[-1])] - mean[key(a, TEMPS[0])] for a in ALPHAS)
                       / len(ALPHAS), 1)
        tb = sum(r["broken"] for v in rows.values() for r in v)
        tf = sum(r["fixed"] for v in rows.values() for r in v)
        print("  网格（平均 gap，括号=该格 ±2SE）："
              + "  ".join(f"{k} {mean[k]:+.1f}(±{gate[k]})" for k in rows))
        borrowed = " / ".join(f"{t} {anc[t]['borrowed_floor_imgs']}" for t in (LO_TIER, HI_TIER))
        print(f"  [本档现算的底] CE 极差 {rng(ce_wrong)} | KD 四格错图极差 {wrange} → 闸 = {own_floor} 张"
              f"（第4课c 当年在两端锚那本账里借的底：{borrowed}）")
        print(f"  [主效应] {tier}：T6−T1 = **{t_main:+.1f} 张** = 本档底 {own_floor} 的 "
              f"{round(abs(t_main) / own_floor, 2)}x → "
              + ("过闸" if abs(t_main) > own_floor else "在底以内（这一档读不出符号）"))
        # 配对读法（只当敏感性）：同一条随机流里先算 T6−T1 再平均，量的是这条轴自己的散布
        per_seed = [sum(rows[key(a, TEMPS[-1])][i]["gap_imgs"] - rows[key(a, TEMPS[0])][i]["gap_imgs"]
                        for a in ALPHAS) / len(ALPHAS) for i in range(len(SEEDS))]
        pe_gate = round(2 * (rng(per_seed) / D4) / math.sqrt(len(SEEDS)), 1)
        print(f"  [配对读法·敏感性] 逐种子 T6−T1 = {[round(v, 1) for v in per_seed]} → 平均 {t_main:+.1f} 张"
              f"（与上面逐格口径同值，自检通过 = {abs(t_main - sum(per_seed) / len(per_seed)) <= 0.2}）"
              f"| 它自己的 ±2SE = {pe_gate} 张 → "
              + ("这一把过" if abs(t_main) > pe_gate else "这一把也不过")
              + "；**判决不用它**（立判据时用的是本档底）")
        result[tier] = {"n_train_imgs": n, "soft_label_audit": audits[tier],
                        "cells": {k: {"per_seed": v, "mean_gap_imgs": mean[k],
                                      "wrong_range_imgs": wrange[k], "gap_range_imgs": grange[k],
                                      "gate_2se_imgs": gate[k]} for k, v in rows.items()},
                        "interpolated_pred_imgs": pred[tier],
                        "deviation_from_interp_imgs": round(t_main - pred[tier], 1),
                        "seed_floor_imgs_own": own_floor,
                        "seed_floor_detail": {"ce_range": rng(ce_wrong), "kd_wrong_ranges": wrange},
                        "t_main_effect_imgs": t_main,
                        "vs_own_floor": round(abs(t_main) / own_floor, 2),
                        "beats_own_floor": bool(abs(t_main) > own_floor),
                        "paired_per_seed_imgs": [round(v, 1) for v in per_seed],
                        "paired_gate_2se_imgs": pe_gate,
                        "paired_beats_gate_sensitivity_only": bool(abs(t_main) > pe_gate),
                        "pooled_flips": {"broken": tb, "fixed": tf,
                                         "p_sign": round(sign_test_p(tb, tf), 8)}}

    print("\n[判决]")
    for tier in FLIP_TIERS:
        r = result[tier]
        print(f"  ① {tier}: 实测 {r['t_main_effect_imgs']:+7.1f} 张（底 {r['seed_floor_imgs_own']}，"
              f"过闸 {r['beats_own_floor']}）| 内插预言 {r['interpolated_pred_imgs']:+6.1f} "
              f"→ 偏差 {r['deviation_from_interp_imgs']:+.1f} 张")
    vals = [result[t]["t_main_effect_imgs"] for t in FLIP_TIERS]
    monotone = bool(all(a > b for a, b in zip(vals, vals[1:])))
    ramp = bool(monotone and all(abs(result[t]["deviation_from_interp_imgs"])
                                 <= result[t]["seed_floor_imgs_own"] for t in FLIP_TIERS))
    step_like = bool(any(result[t]["beats_own_floor"] and abs(result[t]["t_main_effect_imgs"]) >= STEP_LINE
                         and abs(result[t]["deviation_from_interp_imgs"]) > result[t]["seed_floor_imgs_own"]
                         for t in FLIP_TIERS))
    readable = [t for t in FLIP_TIERS if result[t]["beats_own_floor"]]
    print(f"  ② 单调下降 {vals} = {monotone}｜三档偏差都在各自底以内（斜坡形状）= {ramp}"
          f"｜有档贴端点且偏离内插超底（台阶形状）= {step_like}")
    verdict = ("斜坡：翻号是数据量连续调出来的，中间值就是中间值" if ramp else
               "台阶：至少一档贴住端点量级、落在内插线以外 ⇒ 存在门槛而不是一条斜线" if step_like else
               "两说都没钉死：过闸的档不够、或形状两边都不像——这一发只夹紧了区间，没给出形状")
    print(f"  ③ 终判：{verdict}")
    print(f"     过自己底的档 {len(readable)}/3（{'、'.join(readable) or '无'}）；"
          "没过底的档不当 0，当'这一档没量到'")
    if sample == "prefix":
        print("  ④ 自我审计：2000/2500/3000/3500/4500 是同一批图的**嵌套前缀**，不是独立样本 ⇒ "
              "这条曲线读形状、不做回归；端点锚是三档 α 平均、本发是两档（口径差 ≈2 张，见 limitations）")
    else:
        print(f"  ④ 本课判别量（换了一批不相交的图，第 {seg_lo}~{seg_lo + max(FLIP_TIERS.values())} 号）："
              + (f"**{len(readable)}/3 过本档底 → 上一发那三条读数是「这批图」造成的**，"
                 f"它当时不该写成「这条轴读不出」；过闸的档 = {'、'.join(readable)}"
                 if readable else
                 "仍然 0/3 过闸 ⇒ 夹紧失败不是这批图的特产，是这条轴在 2000~4500 之间本来就读不出方向；"
                 "两条曲线各自都只站在自己的三个点上"))
        print("     自我审计：这条曲线与第一条**逐档不共享一张图**（独立性只做在曲线这一层），"
              "但它自己三档仍是嵌套前缀；且它与满数据档那 8500 张必然重叠 ⇒ 不能拿它去并排满数据档；"
              "两端锚来自另一批图，所以跨样本的「偏差」里混了抽样差异，α 口径差也仍在那里（见 limitations）")

    report = {"width": WIDTH, "sample": sample,
              "sample_slice": {"start": seg_lo, "free_region_start": free_lo,
                               "val_tail_start": int(cut)},
              "tiers": FLIP_TIERS, "alphas": list(ALPHAS), "temps": list(TEMPS),
              "seeds": list(SEEDS), "test_n": int(len(yte)), "train_val_n": VAL_N,
              "anchors_from_archives": anc, "interp_slope_imgs_per_1k": round(slope * 1000, 1),
              "prediction_ex_ante": {"ramp_pred_imgs": pred,
                                     "step_pred": f"某一档 |T6−T1| ≥ {STEP_LINE:g} 且偏离内插 > 本档底",
                                     "discriminator": "幅度住在哪儿，不是符号",
                                     **({} if sample == "prefix" else {
                                         "sample_pred": "上一发 0/3 过闸：换一批不相交的图，"
                                                        "至少一档过本档底 ⇒ 那三条是这批图造成的；"
                                                        "仍 0/3 ⇒ 夹紧失败是这条轴的性质",
                                         "result": f"{len(readable)}/3 过闸",
                                         "verdict_this_shot": ("上一发那三条读数是这批图造成的"
                                                              if readable else
                                                              "夹紧失败跨样本存活：这条轴在这个区间读不出方向")})},
              "result": result,
              "verdict": {"monotone_in_n": monotone, "reads_as_ramp": ramp,
                          "reads_as_step": step_like, "tiers_over_own_floor": readable,
                          "sentence": verdict},
              "limitations": ([
                  "三档是嵌套前缀（同一批图越切越多），不是独立样本 ⇒ 读形状不做回归"]
                  if sample == "prefix" else [
                  "独立性只做在**曲线这一层**：这条曲线与第一条逐档不共享一张图，但它自己三档仍是嵌套前缀；"
                  "两条曲线不能拼成一条六档曲线（各档之间的差里混了抽样差异）",
                  f"这一批发用的第 {seg_lo}~ 号图与满数据档（前 8500 张）重叠 ⇒ 本课的读数只能与"
                  f"翻号区间（2000~4500）并排，不能与满数据档并排声明独立",
                  "两端锚来自另一批图：本课的「偏离内插」里同时含数据量差与样本差异，只读大小不读符号"]
              ) + [f"α 只留 {ALPHAS}：与端点锚的三档平均口径差 ≤2 张（α 轴四份档案实测平）",
                   "底是本档现算并对 CE 与四格 KD 取 max，偏保守；格数越多这个 max 越膨胀",
                   "每档 3 条随机流：±2SE 按独立合成是下界（第4课d 同一告警）",
                   "本档底量的是**单臂**重训的漂移，而 T 主效应是**配对差**——第4课e 在 Fashion 上"
                   "刚量过这把尺子偏宽（底 106 vs 配对 σ 38.6 张），所以表里另给一列配对 ±2SE 做敏感性，"
                   "**但判决仍用训练前立好的那把（本档底）**，不能因为另一把尺子更严就换",
                   "老师固定 A、学生固定 S128：这条曲线是**这一副老师**的曲线"
                   "（第5课 D 已经把低数据那个正号在换老师后打回底以内 ⇒ 翻号点本身可能跟着老师走）",
                   "换一批图只买到「这批图 vs 那批图」这一维独立性，任务、老师、架构、配方一个都没换"]}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    n_tr = sum(len(r["cells"]) * len(SEEDS) + len(SEEDS) for r in result.values())
    print(f"\n已存 {out.name}（{n_tr} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="第4课g：夹逼翻号点（--sample disjoint 换一批不相交的图重跑同一条曲线）")
    ap.add_argument("--sample", choices=SAMPLES, default="prefix",
                    help="prefix = 已入库那一发（同批图的前缀档）；disjoint = 整条曲线换一批没被用过的图")
    args = ap.parse_args()
    main(args.sample)
