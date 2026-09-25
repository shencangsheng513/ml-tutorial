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

第三、四发（`--nseeds 8`，两种取样各一本）还的是**分辨率债**，不是独立性债：三档各自只有 3 条随机流，"过不过本档底"
这件事本身的抽样误差就比两说的差还粗（判决表第四节把它列为最后一处"只有 3 条流"）。三条预先写死、
此后不许改口的规矩：
  ① **换流数＝换实验**：产物一律另起名 `compress_report_v4g_n8.json` / `compress_report_v4g_disjoint_n8.json`，
     已入库那两本一个字都不动，v4g 那三档的判词也不回头改。
  ② **老 3 条流只当逐位对账的锚**：新 5 条（第4课e 那批）才是这一发买到的东西。训练之前先打一枪
     canary（2500 档 × α0.7/T6 × 种子 42，1 次训练）与老档案比 wrong_kd/wrong_ce/broken/fixed，
     对不上就直接退出——省掉后面三百次训练；跑完再逐格逐流做全套对账（闸⑥ 同一条腿）。
     **空表不算通过**：一条可核记录都没有 ⇒ 报错退出（第4课f 的规矩）。
  ③ **跨流数只准读配对 CI**：单臂极差底（max−min）随流数机械变宽（E[极差]=D4(n)·σ，
     D4 从 1.693 跳到 2.847），所以 8 条流那本的"过本档底"比 3 条流那本**天然更难**，
     两本之间唯一的比较线是逐流配对差的均值 ±95% t CI（T95 同表随 n 走）。
     系数表单点在 `step4_distill.D4 / T95`，本脚本不自带常数：回落成默认值＝用错刻度（第4课e 的教训）。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (D4, T95, DATA_TIERS, VAL_N, check_flow_constants, flow_tiers, flows,
                           logits_of, sign_test_p, soft_label_audit, train_student)
from step4c_alpha_T_sweep import TIER as FULL_TIER
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
LO_TIER, HI_TIER = "低数据2000", "中数据4500"        # 两个端点锚：从各自档案读，不抄字面量
FLIP_TIERS = {"低2500": 2500, "低3000": 3000, "低3500": 3500}
ALPHAS = (0.7, 1.0)
TEMPS = (1.0, 6.0)                                   # 只测两端：主效应就是这两端之差
# 流名单唯一来源 = step4_distill.FLOW_SEEDS：本课只挑要用哪一档，不养第二份副本。
SEEDS = flows(3)
SEEDS8 = flows(8)                                    # 第4课e 那五条：老 3 条只对账，新 5 条才是这一发买到的
NSEEDS = (3, 8)
if set(NSEEDS) - set(flow_tiers()):
    raise SystemExit(f"--nseeds 档位 {NSEEDS} 里有未登记的档位（在册 {flow_tiers()}）")
SEED_SET = {n: flows(n) for n in NSEEDS}
WIDTH = 128
STEP_LINE = 40.0                                     # 台阶说的"贴住端点"线（张）
SAMPLES = ("prefix", "disjoint")
# 逐位对账吃的字段：四个都是"这条流自己算出来的整数"，换台机器/换版本只要有一位不同就会红
CHECK_FIELDS = ("wrong_kd", "wrong_ce", "broken", "fixed")


def out_path(sample: str, nseeds: int = 3) -> Path:
    """档案名跟着**取样方式 + 流数**走：第一本保持原名（旧账不改名），后续各发另起一名。
    两本共用一个名字的话，第二条曲线会无声盖掉第一条那条"夹紧失败"的判决。"""
    extra = "" if nseeds == 3 else f"_n{nseeds}"
    base = "compress_report_v4g" if sample == "prefix" else f"compress_report_v4g_{sample}"
    return HERE / f"{base}{extra}.json"


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


def old_flow_refs(sample: str) -> tuple[dict, dict, str]:
    """补流那本必须与老那本在**共有的 3 条流**上逐位相同，否则两本之间连"同一台仪器"都不成立。
    摊平成 tier → 格 → 种子 → 待核字段；比对交给 canary 与收尾的全套闸。
    老账自己的配对逐流读数**从档案里的逐格逐流原始记录重算**（不是读那个现成字段）：
    第一本 v4g 早于 paired_per_seed_imgs 这个字段，重算两本都能做，而且顺手证明了"公式对得上"；
    字段存在时再拿重算值对一次，对不上就是摊平 refs 时挑错了账。"""
    ref_path = out_path(sample, 3)
    if not ref_path.exists():
        raise SystemExit(f"缺基准账 {ref_path.name}：{n_seed_hint(sample)}"
                         f"没有它就没有逐位对账的锚，这一发不许跑")
    d = json.loads(ref_path.read_text(encoding="utf-8"))
    # 取样方式：第一本（prefix）早于 sample 字段，档案名本身就是它的取样方式，缺字段不算不一致
    if (int(d["width"]) != WIDTH or d.get("sample", "prefix") != sample
            or list(d["seeds"]) != list(SEEDS)
            or [float(a) for a in d["alphas"]] != list(ALPHAS)
            or [float(t) for t in d["temps"]] != list(TEMPS)
            or d["tiers"] != FLIP_TIERS):
        raise SystemExit(f"{ref_path.name} 的配置与本次要跑的不一致"
                         f"（width={d.get('width')} sample={d.get('sample')} seeds={d.get('seeds')} "
                         f"alphas={d.get('alphas')} temps={d.get('temps')}）：基准账挑错了，先看一眼")
    refs, old_sum = {}, {}
    for tier, book in d["result"].items():
        refs[tier] = {k: {int(r["seed"]): {f: r[f] for f in CHECK_FIELDS} for r in v["per_seed"]}
                      for k, v in book["cells"].items()}
        by_seed = {k: {int(r["seed"]): r["gap_imgs"] for r in v["per_seed"]}
                   for k, v in book["cells"].items()}
        paired = [sum(by_seed[key(a, TEMPS[-1])][s] - by_seed[key(a, TEMPS[0])][s]
                      for a in ALPHAS) / len(ALPHAS) for s in SEEDS]
        t_main = float(book["t_main_effect_imgs"])
        if abs(paired_ci(paired, len(paired))["mean_imgs"] - t_main) > 0.2:
            raise SystemExit(f"{ref_path.name} 的 {tier}：按逐格原始记录重算的 T 主效应 "
                             f"{sum(paired) / len(paired):+.1f} ≠ 档案记的 {t_main:+.1f} "
                             f"⇒ 本次的配对公式与那本账不同源，先查 key(a, T) 的接法")
        stored = book.get("paired_per_seed_imgs")
        if stored is not None and [round(float(v), 1) for v in stored] != [round(v, 1) for v in paired]:
            raise SystemExit(f"{ref_path.name} 的 {tier}：档案自带的 paired_per_seed_imgs {stored} "
                             f"≠ 重算值 {paired} ⇒ 摊平 refs 时挑错了账")
        old_sum[tier] = {"t_main_imgs": t_main,
                         "paired_per_seed_imgs": [round(v, 1) for v in paired],
                         "paired_recomputed_from_cells": True,
                         "stored_paired_field": stored is not None}
    return refs, old_sum, ref_path.name


def n_seed_hint(sample: str) -> str:
    return f"--sample {sample} 的 3 条流那一本从没跑过。"


def paired_ci(values: list[float], n: int) -> dict:
    """逐流配对差的均值 ±95% t CI（T95 单点在 step4_distill，缺 n 直接 KeyError：回落=用错刻度）。
    跨流数读这两本时**只准用它**，单臂极差底随流数机械变宽。"""
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    ci = T95[n] * sd / math.sqrt(n)
    return {"n_flows": n, "mean_imgs": round(mean, 1), "sd_imgs": round(sd, 1),
            "ci95_imgs": [round(mean - ci, 1), round(mean + ci, 1)],
            "same_sign": f"{sum(1 for v in values if v > 0)}/{n}（+）· "
                         f"{sum(1 for v in values if v < 0)}/{n}（−）",
            "excludes_zero": bool(mean - ci > 0 or mean + ci < 0)}


def main(sample: str = "prefix", nseeds: int = 3) -> None:
    if nseeds not in NSEEDS:
        raise SystemExit(f"--nseeds 只认 {NSEEDS}（流数是判别变量，不接受顺手改第三个值）")
    seeds = SEED_SET[nseeds]
    check_flow_constants(len(seeds))
    d4 = D4[len(seeds)]
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    out = out_path(sample, nseeds)
    if out.exists():
        raise SystemExit(f"{out.name} 已入库——重跑会盖掉那本账。要重跑先删或改名")
    refs, old_sum, ref_name = ((None, None, None) if nseeds == 3
                               else old_flow_refs(sample))
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
    canary: dict | None = None
    if refs is not None:
        # 训练之前先打一枪：一次训练换掉三百次训练的浪费——老流对不上 ⇒ 这台仪器已经不是当初那台
        cname, cn = next(iter(FLIP_TIERS.items()))
        cd, cy = data_slice(xt_all, yt_all, sample, cn, cut)
        cmodel, _ = train_student(WIDTH, cd, cy, logits_of(teacher, cd), "KD", xv, yv,
                                  seed=SEEDS[0], alpha=ALPHAS[0], t=TEMPS[-1])
        cp = predict(cmodel, xte, False)
        cce, _ = train_student(WIDTH, cd, cy, logits_of(teacher, cd), "CE", xv, yv, seed=SEEDS[0])
        ccep = predict(cce, xte, False)
        got = {"wrong_kd": int((cp != yte).sum()), "wrong_ce": int((ccep != yte).sum()),
               "broken": int(((ccep == yte) & (cp != yte)).sum()),
               "fixed": int(((ccep != yte) & (cp == yte)).sum())}
        want = refs[cname][key(ALPHAS[0], TEMPS[-1])][SEEDS[0]]
        canary = {"tier": cname, "cell": key(ALPHAS[0], TEMPS[-1]), "seed": SEEDS[0],
                  "this_run": got, "archive": want, "book": ref_name,
                  "match": all(got[f] == want[f] for f in CHECK_FIELDS)}
        print(f"\n[逐位对账·开跑前的一枪] {ref_name} {cname}/{canary['cell']}/种子{SEEDS[0]}："
              f"本次 {got} vs 档案 {want} → " + ("**一致**，往下跑" if canary["match"] else "**不一致**"))
        if not canary["match"]:
            raise SystemExit("老的那条随机流在新代码里复现不出来 ⇒ 训练路径变了，"
                             "这一发读到的 8 条流与已入库那 3 条不在同一台仪器上，先查改动再跑")
    for tier, n in FLIP_TIERS.items():
        xd, yd = data_slice(xt_all, yt_all, sample, n, cut)
        soft = logits_of(teacher, xd)
        print(f"\n[开跑] {tier} | S{WIDTH} | {len(ALPHAS)}×{len(TEMPS)}×{len(seeds)} = "
              f"{len(ALPHAS) * len(TEMPS) * len(seeds)} 次 KD + {len(seeds)} 次 CE")
        rows: dict[str, list[dict]] = {key(a, t): [] for a in ALPHAS for t in TEMPS}
        ce_wrong: list[int] = []
        for seed in seeds:
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
        gate = {k: round(2 * (grange[k] / d4) / math.sqrt(len(seeds)), 1) for k in rows}
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
                        for a in ALPHAS) / len(ALPHAS) for i in range(len(seeds))]
        pe_gate = round(2 * (rng(per_seed) / d4) / math.sqrt(len(seeds)), 1)
        ci_all = paired_ci(per_seed, len(per_seed))
        print(f"  [配对读法·敏感性] 逐流 T6−T1 = {[round(v, 1) for v in per_seed]} → 平均 {t_main:+.1f} 张"
              f"（与上面逐格口径同值，自检通过 = {abs(t_main - sum(per_seed) / len(per_seed)) <= 0.2}）"
              f"| 它自己的 ±2SE = {pe_gate} 张 → "
              + ("这一把过" if abs(t_main) > pe_gate else "这一把也不过")
              + "；**判决不用它**（立判据时用的是本档底）")
        print(f"  [跨流数唯一可比的那条线] 逐流配对差 {len(per_seed)} 条 → 均值 {ci_all['mean_imgs']:+.1f} 张"
              f" 95% CI {ci_all['ci95_imgs']}｜同号 {ci_all['same_sign']}｜CI 不含 0 = {ci_all['excludes_zero']}"
              f"（单臂底 {own_floor} 张随流数机械变宽，所以**与 3 条流那本只能比这个数**）")
        tier_book = {"n_train_imgs": n, "soft_label_audit": audits[tier],
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
                     "paired_ci95_imgs": ci_all,
                     "paired_beats_gate_sensitivity_only": bool(abs(t_main) > pe_gate),
                     "pooled_flips": {"broken": tb, "fixed": tf,
                                      "p_sign": round(sign_test_p(tb, tf), 8)}}
        if refs is not None:
            # 共有流逐位对账（闸⑥ 同一条腿）：老 3 条 × 四格，四个整数字段一个都不许差
            checked, bad = 0, []
            for k, v in rows.items():
                for r in v:
                    if r["seed"] not in SEEDS:
                        continue
                    want = refs[tier][k][r["seed"]]
                    checked += 1
                    if any(r[f] != want[f] for f in CHECK_FIELDS):
                        bad.append({"cell": k, "seed": r["seed"],
                                    "this_run": {f: r[f] for f in CHECK_FIELDS}, "archive": want})
            if checked == 0:
                raise SystemExit(f"{tier}：逐位对账一条记录都没有（基准账 {ref_name} 里没有可核的格）"
                                 "——空表『全过』不是通过，是没测到")
            if bad:
                print(f"  [共有流逐位对账] FAIL {len(bad)}/{checked}：{bad[:3]}")
                raise SystemExit("老的随机流在新代码里复现不出来 ⇒ 两本账不是同一台仪器，"
                                 "跨流数那把配对 CI 也失去可比性，先查改动")
            old3 = [v for v, s in zip(per_seed, seeds) if s in SEEDS]
            ci_old3 = paired_ci(old3, len(old3))
            ref_ci = paired_ci(old_sum[tier]["paired_per_seed_imgs"], len(SEEDS))
            if abs(ci_old3["mean_imgs"] - old_sum[tier]["t_main_imgs"]) > 0.2:
                raise SystemExit(f"{tier}：共有流子集重算出的 T 主效应 {ci_old3['mean_imgs']:+.1f} "
                                 f"≠ 基准账记的 {old_sum[tier]['t_main_imgs']:+.1f} ⇒ 逐位对账过了但公式没对上，查 per_seed 接线")
            if ref_ci["mean_imgs"] != ci_old3["mean_imgs"] or ref_ci["ci95_imgs"] != ci_old3["ci95_imgs"]:
                raise SystemExit(f"{tier}：基准账自己的配对读数在本次重算出的 CI {ref_ci['ci95_imgs']} "
                                 f"≠ 逐位复现出的那 3 条流 {ci_old3['ci95_imgs']} ⇒ 摊平 refs 时挑错了账")
            print(f"  [共有流逐位对账] {checked}/{checked} 条（老 {len(SEEDS)} 流 × {len(rows)} 格 × "
                  f"{len(CHECK_FIELDS)} 个整数字段）与 {ref_name} 逐位一致"
                  f"| 老 3 流子集重算 T 主效应 {ci_old3['mean_imgs']:+.1f} 张"
                  f" = 基准账 {old_sum[tier]['t_main_imgs']:+.1f} ✓")
            print(f"  [跨流数只读这条] {ref_name}（3 流）配对 {ref_ci['mean_imgs']:+.1f} "
                  f"CI {ref_ci['ci95_imgs']} → 本次（{len(seeds)} 流）{ci_all['mean_imgs']:+.1f} "
                  f"CI {ci_all['ci95_imgs']}｜CI 是否含 0 变了 = "
                  f"{ref_ci['excludes_zero'] != ci_all['excludes_zero']}")
            tier_book["parity_vs_3flow_book"] = {"book": ref_name, "records_checked": checked,
                                                 "fields": list(CHECK_FIELDS), "all_match": True,
                                                 "old_subset_paired": ci_old3,
                                                 "old_book_paired_recomputed": ref_ci,
                                                 "old_subset_t_main_matches_archive": True}
            tier_book["paired_ci95_old_subset_imgs"] = ci_old3["ci95_imgs"]
            tier_book["cross_flow_count_read"] = {
                "rule": "与 3 流那本唯一可比的读数是逐流配对差的均值 ±95% t CI；单臂底随流数机械变宽，不许比",
                "old_book": ref_name, "old_paired": ref_ci, "new_paired": ci_all,
                "mean_shift_imgs": round(ci_all["mean_imgs"] - ref_ci["mean_imgs"], 1)}
        result[tier] = tier_book

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
    if refs is not None:
        print(f"  ⑤ 补流本读法（{ref_name} 3 流 → 本次 {len(seeds)} 流）："
              + "；".join(f"{t} {r['cross_flow_count_read']['old_paired']['mean_imgs']:+.1f}"
                          f"{r['cross_flow_count_read']['old_paired']['ci95_imgs']} → "
                          f"{r['cross_flow_count_read']['new_paired']['mean_imgs']:+.1f}"
                          f"{r['cross_flow_count_read']['new_paired']['ci95_imgs']}"
                          f"（不含0={r['cross_flow_count_read']['new_paired']['excludes_zero']}）"
                          for t, r in result.items()))
        print("     只有这一行可以跨两本读；上面①②③那把「过本档底」是单臂极差底，"
              f"随流数机械变宽（{D4[len(SEEDS)]}→{d4}），两本的 beats_own_floor 布尔值**不可比**。"
              "换流数＝换实验：这一本不给老判词改口，只把三档读数钉得更细")

    lims = ([
        "三档是嵌套前缀（同一批图越切越多），不是独立样本 ⇒ 读形状不做回归"]
        if sample == "prefix" else [
        "独立性只做在**曲线这一层**：这条曲线与第一条逐档不共享一张图，但它自己三档仍是嵌套前缀；"
        "两条曲线不能拼成一条六档曲线（各档之间的差里混了抽样差异）",
        f"这一批发用的第 {seg_lo}~ 号图与满数据档（前 8500 张）重叠 ⇒ 本课的读数只能与"
        f"翻号区间（2000~4500）并排，不能与满数据档并排声明独立",
        "两端锚来自另一批图：本课的「偏离内插」里同时含数据量差与样本差异，只读大小不读符号"]) + [
        f"α 只留 {ALPHAS}：与端点锚的三档平均口径差 ≤2 张（α 轴四份档案实测平）",
        "底是本档现算并对 CE 与四格 KD 取 max，偏保守；格数越多这个 max 越膨胀",
        f"每档 {len(seeds)} 条随机流：±2SE 按独立合成是下界（第4课d 同一告警）",
        "本档底量的是**单臂**重训的漂移，而 T 主效应是**配对差**——第4课e 在 Fashion 上"
        "刚量过这把尺子偏宽（底 106 vs 配对 σ 38.6 张），所以表里另给一列配对 ±2SE 做敏感性，"
        "**但判决仍用训练前立好的那把（本档底）**，不能因为另一把尺子更严就换",
        "老师固定 A、学生固定 S128：这条曲线是**这一副老师**的曲线"
        "（第5课 D 已经把低数据那个正号在换老师后打回底以内 ⇒ 翻号点本身可能跟着老师走）"]
    if refs is not None:
        lims += [f"这一本是**补流本**（{len(SEEDS)}→{len(seeds)} 条流）：买到的是分辨率不是独立性——"
                 f"任务、老师、架构、数据档、取样方式一个都没换，而且单臂极差底随流数机械变宽"
                 f"（D4 {D4[len(SEEDS)]}→{d4}），所以「过本档底」比 {ref_name} 那一本天然更难，"
                 f"两本之间只准比逐流配对差的均值 ±95% t CI（见各档 cross_flow_count_read）"]
    if sample != "prefix":
        lims += ["换一批图只买到「这批图 vs 那批图」这一维独立性，任务、老师、架构、配方一个都没换"]

    report = {"width": WIDTH, "sample": sample,
              "sample_slice": {"start": seg_lo, "free_region_start": free_lo,
                               "val_tail_start": int(cut)},
              "tiers": FLIP_TIERS, "alphas": list(ALPHAS), "temps": list(TEMPS),
              "seeds": list(seeds), "n_seeds": len(seeds),
              "d4_sigma_hat": d4, "t95": T95[len(seeds)],
              "resolution_shot": (None if refs is None else {
                  "against_book": ref_name, "old_flows_are_anchor_only": True,
                  "new_flows": [s for s in seeds if s not in SEEDS],
                  "canary_pre_training": canary,
                  "why": "补流买的是分辨率不是独立性：老 3 条流逐位对账，跨流数只读配对 CI"}),
              "test_n": int(len(yte)), "train_val_n": VAL_N,
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
              "limitations": lims}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    n_tr = sum(len(r["cells"]) * len(seeds) + len(seeds) for r in result.values())
    if refs is not None:
        n_tr += 2                       # canary 那两枪（一枪 KD、一枪 CE）也是真训练出来的
    print(f"\n已存 {out.name}（{n_tr} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="第4课g：夹逼翻号点"
                                             "（--sample disjoint 换一批不相交的图；--nseeds 8 把三档补到 8 条流）")
    ap.add_argument("--sample", choices=SAMPLES, default="prefix",
                    help="prefix = 已入库那一发（同批图的前缀档）；disjoint = 整条曲线换一批没被用过的图")
    ap.add_argument("--nseeds", type=int, choices=NSEEDS, default=3,
                    help="8 = 加第4课e 那五条随机流（老 3 条只当逐位对账的锚）；档案名加 _n8")
    args = ap.parse_args()
    main(args.sample, args.nseeds)
