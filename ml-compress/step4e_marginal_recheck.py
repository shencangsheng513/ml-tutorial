# -*- coding: utf-8 -*-
"""第4课d 的续账：把闸门读成"边缘"的那几格补到 8 条随机流，是疗效还是刻度假象？（只补种子，不扫参）

第4课d（`step4d_cell_floors.py`）拿每格自己的配对误差棒重算了一遍闸门，八条判决里只有一条翻档：
**α=1.0/T=1 平均 +9.7 张**（S128）——过自己的 ±2SE（9.5 张），没过借来的底（41 张）。
同一档另一格 α=0.7/T=1 是 +9.0 张（p=0.012），S64 上 α=1.0/T=1 也是同号 +8.0 张。
三处同号，方向是"不加热的 KD 比 CE 略好"，而这正好和第4课c 的措辞（"与 CE 打平"）相反方向。

所以这一发只做一件事：**把误差棒做到能判这件事的大小**。不扫 α、不扫 T、不加架构，
只把目标那一格的随机流从 3 条加到 8 条。为什么加种子就够：
  · 老 3 条必须**逐位复现**（错图数/翻坏/翻正与 v4c 档案一字不差），否则说明本课改坏了 loss；
  · 判据换成**以种子为复现单位**的检验——逐图 p 在三种子间不独立（审计②那条），
    而"8 条独立随机流里几条同号"是干净的复现级证据。

第二发（`--tier 低数据2000 --t 6`）是同一件仪器**复用**，理由别写反：低数据档九格扫出来
9/9 全是正号、α=1.0/T=6 到 +71.7 张 = 该档借来的底（37 张）的 1.9x，这是本板块第一条正号判决。
但它每种子上只训过一次、n 仍是 3——**同一把尺子不能只对不利的读数严格**，
所以那条比"打平"大得多的正号也得免检通过：补到 8 条再判。

训练前把判决标准写死（此后不许改口）：
  H+（这一格的软标签确实值几张图）：8 种子平均 gap ≥ +5 张，**且 95% CI 下界 > 0**；
  H0（第4课c 那句"打平/不算疗效"站得住，边缘读数是 ±2SE 下界的假信号）：CI 跨 0；
  H−（反向，几乎不可能但照写）：CI 上界 < 0。
零 API 费，纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4e_marginal_recheck.py
  # 低数据档 T=6 那两格（对账基准读 compress_report_v4c_low.json）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4e_marginal_recheck.py --tier 低数据2000 --t 6
  # 第三发：低数据档 T=3 那一列（a0.7_T3 +17.7 张 p=0.052、a1_T3 +22.0 张 p=0.0196，
  # 两格都还在自己的 ±2SE 里）——第4课g 量到中间档"符号读不出"之后，这一列是低数据档最有肉的两格，
  # 值得用 8 条流而不是 3 条来看。产物另起一名（compress_report_v4e_low_t3.json），
  # 因为 t=6 那一本已入库：**同一件仪器换温度必须换档案名**，否则第二次跑无声盖掉第一次的判决。

第四发（`--dataset fashion --tier 低数据2000 --t 1 6`）把同一件仪器搬到第二个任务上，
对象是第6课那句被自己判成"测不动"的读数：Fashion 低数据档 T 主效应 **−94.7 张**，只有本档
自己的底 106 张的 0.89x。这一发不判"翻号是不是 MNIST 特产"（那是 n=3 那发按预立判据没资格
说的话），只回答一个更窄的问题：**把 −94.7 的误差棒从"借来的 106 张极差"换成它自己的
8 条配对随机流，它到底是接近零还是一个真负号**。为此必须**同时跑 T=1 与 T=6 两列**——
主效应是配对差之差，复现单位是"同一条随机流里 (gap@T6 − gap@T1)"，只补 T=6 一列量不到它
（那一列自己的三格均值才 +29.7/−47.3 张，读数其实住在 T=1 那一列太好：+97.7/+74.0 张）。
换数据集时逐位对账的基准交给第6课那本账（`compress_report_v6_fashion.json` 的 raw_rows），
种子底也换成**它自己那档的 own floor**——MNIST 的 41/37 张在这里既没被量过、也不该被借。

第五发（`--teacher D --tier 低数据2000 --t 1 6`）不动数据集、不动档位，只在这件仪器上开一个
**老师**维度，为的是补 2×2（任务 × 老师）低数据档缺的那一格。四格里先前只有三格有 8 条配对流：
MNIST×A = **+60.4 张**（CI [+39.3, +81.5]，8/8 同号 ⇒ 加热有益）、Fashion×A = −79.1、Fashion×dry = −84.9，
而 MNIST×D 只有第5课那发 n=3 的 −20.3 张（在借来的底 37 张以内 = 读不出）。这一格是**唯一能把
"翻号跟任务"与"翻号跟这份软标签"分开的读数**：若它补到 8 条仍是负号（或 CI 跨 0），那 MNIST 那个
正号就不是 MNIST 这个任务的性质，而是老师 A 那份软标签的性质——2×2 的两种解释到此才第一次可判。
换老师要换的三样全跟着开关走，一处一份：①**权重**（A = 20 epoch 基线，D = 同配方只训 3 epoch）；
②**逐位对账的账本**（D 的旧三条种子住在第5课那本 `compress_report_v5_d.json`，不在 v4c）；
③**干湿闸门**——训练前先从那本账现读两副老师的塌缩率，差不到 10pp 直接退出：
"换了老师但软标签没换"的读数里，两副老师的差没有知识源的解释权（第6课b 同一把刀）。
CE 锚不吃软标签 ⇒ 旧三条种子的 wrong_ce 必须与 v5_d 同数，这条对账顺带证明
"换老师的同时没换随机流、没换 loader、没换测试集"。**档案名跟着老师走**（`_d` 后缀），
否则这一发会无声盖掉老师 A 那一本的判决。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (D4, T95, DATA_TIERS, VAL_N, flows, logits_of, sign_test_p, summarize,
                         train_student)
from step4c_alpha_T_sweep import ALPHA_ANCHOR, TIER, V4C_SUFFIX, WIDTH, key
from step4c_alpha_T_sweep import out_path as v4c_out_path
from step6_fashion_replication import TEACHER_PT, load_fashion
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
# 流名单唯一来源是 step4_distill.FLOW_SEEDS（第4课b/c/f 已经并进去过）：本课那 8 条从这里取，
# 自己不养第二份副本——第二份副本一旦与在册池漂移，"补进来的那几条才是复现"这句就悄悄不成立了。
OLD_SEEDS = flows(3)                         # v4c/v6 已跑过的三条：本课只当对账锚，不重算结论
NEW_SEEDS = flows(8)[len(flows(3)):]         # 补进来的五条随机流
H_MIN_EFFECT = 5.0                            # 训练前立的下限：低于 5 张不值得为它开一课
FIELDS = ("wrong_kd", "wrong_ce", "broken", "fixed")

# 换数据集/换老师要换的东西全在这里，一处一份：**没有第二套字面量**。
# loader 直接 import 第6课那个函数（自己再抄一份装载代码，逐位对账就只证明了我抄得对，
# 没证明仪器对）；ref 是旧三条种子的原始读数所在那本账，底也从同一本现读，
# 所以 Fashion 用的是它自己那档的 own floor（106 张），不借 MNIST 的 41/37。
# 老师按 (数据集, 老师) 登记：换老师 = 换知识源，账本/权重/干湿基准三样一起换，
# 少换一样就是对回旧账——而且会绿着绿到训练结束。
MARGIN_PP = 10.0      # 干湿闸：两副老师的塌缩率至少差这么多才算真换了变量（第6课b 同一条）

TEACHERS = {                           # 外层按数据集、内层按老师字母；A = 各数据集默认那副
    "mnist": {
        "A": {"pt": MODELS / "mnist_cnn_fp32.pt", "layout": "v4c",
              "ref": lambda w, tier: v4c_out_path(w, tier),
              "floor_kind": "借来的单臂种子底（v4b 经 v4c 转录）", "audit": None},
        # D 的第③样（干湿基准）不用现算：第5课已经把两副老师在**同一批拟合集**上的塌缩率
        # 写进了它自己的档案，从同一本账现读即可——自己再算一遍等于信代码胜过信入库的判决。
        "D": {"pt": MODELS / "mnist_cnn_underfit3.pt", "layout": "v5",
              "ref": lambda w, tier: HERE / "compress_report_v5_d.json",
              "floor_kind": "老师 A 在同一档借来的单臂种子底（v5_d 现读，两副老师共用一把尺）",
              "audit": ("A", "D")},
    },
    "fashion": {
        "A": {"pt": TEACHER_PT, "layout": "v6",
              "ref": lambda w, tier: HERE / "compress_report_v6_fashion.json",
              "floor_kind": "该档自己的底（第4课g 规矩：CE 臂 + 各 KD 格跨种子极差取 max）",
              "audit": None},
    },
}

DATASETS = {
    "mnist": {"loader": load_split, "teachers": TEACHERS["mnist"]},
    "fashion": {"loader": load_fashion, "teachers": TEACHERS["fashion"]},
}


def teacher_spec(dataset: str, teacher: str) -> dict:
    """老师按数据集查：D 那副权重只在 MNIST 上造过，硬搬到 Fashion 等于凭空换了一份知识源。"""
    avail = DATASETS[dataset]["teachers"]
    if teacher not in avail:
        raise SystemExit(f"{dataset} 没有登记过老师 {teacher}（只有 {'、'.join(avail)}）——"
                         f"换数据集要连老师的权重一起造，不然这一发跑的是不存在的知识源")
    return avail[teacher]


def cells_for(temps: tuple[float, ...]) -> list[tuple[str, float, float]]:
    """目标格：每个温度两格（α=1.0 与 α=0.7，v4d 里被埋掉的是 T=1、低数据档要复检的是 T=6）。
    给两个温度就是四格——第6课那条 −94.7 是**配对差之差**，只有同一条随机流的两列都在手上
    才量得到它；只补 T=6 一列，误差棒量的是别的东西。"""
    return [(key(a, t), a, t) for t in temps for a in (1.0, ALPHA_ANCHOR)]


def archive_cells(dataset: str, teacher: str, width: int, tier: str,
                  cells: tuple[str, ...]) -> dict:
    """旧三条种子的原始读数 + 该档的底 + 它是什么底，三样全从**同一本账**现读。
    先核这本账属于这个数据集/这副宽度/这一档（换老师时还核它记的是不是这位老师）：
    读错账的对账比没对账更危险——它会绿。"""
    spec = teacher_spec(dataset, teacher)
    lay = spec["layout"]
    dryness: dict | None = None
    p = spec["ref"](width, tier)
    if not p.exists():
        raise SystemExit(f"缺 {p.name}——逐位对账的基准只能从它读；先跑 {dataset} 那一课的父脚本")
    d = json.loads(p.read_text(encoding="utf-8"))
    if lay == "v4c":
        if int(d["width"]) != width or d["tier"] != tier:
            raise SystemExit(f"{p.name} 是 S{d['width']}/{d['tier']} 的档案，不是 S{width}/{tier}")
        floor, where = int(d["seed_floor_imgs_from_v4b"]), "grid"
        get = lambda k: d["grid"][k]["per_seed"]                      # noqa: E731
        missing = [k for k in cells if k not in d["grid"]]
    elif lay == "v5":
        # 老师 D 的旧三条种子住在第5课那本账里（档 → grid → 格 → per_seed）；同一本账还记着
        # 两副老师在拟合集上的塌缩率，干湿闸从这里现读——自己再算一遍等于信代码胜过信判决。
        if d.get("teacher_name") != teacher or int(d["width"]) != width:
            raise SystemExit(f"{p.name} 记的是老师 {d.get('teacher_name')}/S{d.get('width')}，"
                             f"不是 {teacher}/S{width}——换老师要连账本一起换")
        if tier not in d["tiers"]:
            raise SystemExit(f"{p.name} 没有 {tier} 这一档，只有 {'、'.join(d['tiers'])}")
        pt = d["tiers"][tier]
        floor, where = int(pt["borrowed_floor_imgs"]), f"tiers[{tier}].grid"
        get = lambda k: pt["grid"][k]["per_seed"]                     # noqa: E731
        missing = [k for k in cells if k not in pt["grid"]]
        a_name, b_name = spec["audit"]
        aud = d["soft_label_audit"][tier]
        dryness = {"from": p.name, "tier": tier, "a_teacher": a_name, "b_teacher": b_name,
                   "a_collapse_pct": aud[a_name]["原始"]["pct_top1_gt_0.99"],
                   "b_collapse_pct": aud[b_name]["原始"]["pct_top1_gt_0.99"]}
    else:
        if d.get("dataset") != "FashionMNIST" or int(d["width"]) != width:
            raise SystemExit(f"{p.name} 记的是 {d.get('dataset')}/S{d.get('width')}，"
                             f"不是 {dataset}/S{width}——换数据集/换宽度要连账本一起换")
        if tier not in d["per_tier"]:
            raise SystemExit(f"{p.name} 没有 {tier} 这一档，只有 {'、'.join(d['per_tier'])}")
        pt = d["per_tier"][tier]
        floor, where = int(pt["own_floor_imgs"]), "per_tier[tier].raw_rows"
        get = lambda k: pt["raw_rows"][k]                             # noqa: E731
        missing = [k for k in cells if k not in pt["raw_rows"]]
    if missing:
        raise SystemExit(f"{p.name} 的 {where} 里没有 {missing} 这几格：新 α/新温度得先在父课扫出来")
    return {"file": p.name, "dataset": dataset, "teacher": teacher, "tier": tier,
            "floor_kind": spec["floor_kind"], "floor_imgs": floor, "dryness": dryness,
            "cells": {k: {r["seed"]: {f: r[f] for f in FIELDS} for r in get(k)} for k in cells}}


def dryness_gate(arc: dict) -> dict | None:
    """换老师那一发必须**先证明软标签真的换了**：两副老师在同一批拟合集上的塌缩率若差不到
    MARGIN_PP，这一格与老师 A 那格量的就是同一份知识，2×2 里根本没有"老师"这一维——
    在训练前退出，不烧那 40 次。CE 锚对账管的是"没换仪器"，管不了"没换知识源"。"""
    dy = arc["dryness"]
    if dy is None:
        return None
    gap = round(dy["b_collapse_pct"] - dy["a_collapse_pct"], 1)
    dy.update(gap_pp=gap, margin_pp=MARGIN_PP, moved_abs=abs(gap) >= MARGIN_PP,
              b_drier_than_a=gap > 0)
    print(f"[闸① 干湿必须真的动] {dy['tier']} 拟合集 T=1 塌缩率："
          f"{dy['a_teacher']} {dy['a_collapse_pct']}% -> {dy['b_teacher']} {dy['b_collapse_pct']}%"
          f"（差 {gap:+}pp，闸门 >= {MARGIN_PP:g}pp）")
    if not dy["moved_abs"]:
        raise SystemExit(f"两副老师的软标签一样干（差 {abs(gap)}pp < {MARGIN_PP:g}pp）=> 这一发"
                         f"没有判别量，{dy['b_teacher']} 与 {dy['a_teacher']} 的差里没有知识源的"
                         f"解释权；零训练退出")
    return dy


def v4e_path(dataset: str, teacher: str, width: int, tier: str,
             temps: tuple[float, ...]) -> Path:
    """档案名必须跟着**每一个判别变量**走：数据集、老师、宽度、档位、温度列。已入库的三本保持
    原名不动（旧账不许换名字），其余组合一律由它们拼出来。后缀规则只有一个来源
    （step4c 的 V4C_SUFFIX）：这里再抄一份，下一档进来就会和满数据档那本撞名。"""
    archived = {("mnist", "A", 128, TIER, (1.0,)): "compress_report_v4e.json",
                ("mnist", "A", 128, "低数据2000", (6.0,)): "compress_report_v4e_low.json",
                ("mnist", "A", 128, "低数据2000", (3.0,)): "compress_report_v4e_low_t3.json"}
    k = (dataset, teacher, width, tier, tuple(temps))
    if k in archived:
        return HERE / archived[k]
    if tier != TIER and tier not in V4C_SUFFIX:
        raise SystemExit(f"档 {tier} 没登记档案名后缀（V4C_SUFFIX 只有 {list(V4C_SUFFIX)}）："
                         f"回落到默认名会和 {TIER} 那本撞名，无声盖掉它的判决")
    return HERE / ("compress_report_v4e" + ("" if dataset == "mnist" else f"_{dataset}")
                   + ("" if teacher == "A" else f"_{teacher.lower()}")
                   + V4C_SUFFIX.get(tier, "") + ("" if width == 128 else f"_s{width}")
                   + "_t" + "t".join(f"{t:g}" for t in temps) + ".json")




def t_effects_by_seed(spec, rows, seeds, temps: tuple[float, ...]) -> list[float]:
    """按种子配对的 T 主效应：同一条随机流里 mean over α 的（gap@T_max − gap@T_min）。
    复现单位是**流**不是**格**——第6课那条 −94.7 本来就是配对差之差，拿单格的 gap 序列
    去量它，量的又是别的东西（种子底那个错犯的同一个）。"""
    by = {(k, r["seed"]): r["gap_imgs"] for k, _, _ in spec for r in rows[k]}
    alphas = sorted({a for _, a, _ in spec})
    hi, lo = max(temps), min(temps)
    return [sum(by[(key(a, hi), s)] - by[(key(a, lo), s)] for a in alphas) / len(alphas)
            for s in seeds]


def effect_verdict(mean: float, ci: list[float], temps: tuple[float, ...], what: str) -> str:
    """判词里不写死方向：同一个读数在满数据档叫"加热有害"、在低数据档叫"加热有益"，
    把符号和 min/max 一起硬编码，换一档就打印出反向的话（第4课c 抓到过的那处报表 bug）。"""
    lo_ci, hi_ci = ci
    span = f"T{min(temps):g}→T{max(temps):g}"
    if lo_ci > 0 or hi_ci < 0:
        noun = "加热有益（温度越高越少错）" if lo_ci > 0 else "加热有害（温度越高越多错）"
        if abs(mean) < H_MIN_EFFECT:
            return f"有方向但够不上线：{span} 均值 {mean:+.1f} 张 < {H_MIN_EFFECT:g} 张的下限（{what}）"
        return f"H_有方向：{span} 这一刀是{noun}，{mean:+.1f} 张，CI 不含 0（{what}）"
    return f"H0：CI 跨 0 → {span} 在 {what} 上仍测不动，不许替它编方向"


def main(dataset: str = "mnist", teacher: str = "A", width: int = WIDTH, tier: str = TIER,
         temps: tuple[float, ...] = (1.0,), red_check: bool = False) -> None:
    temps = tuple(sorted(set(temps)))
    tlist = "、".join(f"{t:g}" for t in temps)
    spec = cells_for(temps)
    cells = tuple(k for k, _, _ in spec)
    ds = DATASETS[dataset]
    tsp = teacher_spec(dataset, teacher)
    arc = archive_cells(dataset, teacher, width, tier, cells)
    seeds = list(OLD_SEEDS) + list(NEW_SEEDS)
    out = v4e_path(dataset, teacher, width, tier, temps)
    if out.exists():
        raise SystemExit(f"{out.name} 已入库——重跑会盖掉那本账。要重跑先删或改名，"
                         f"别在没对过账的情况下换掉已入库的判决")
    if red_check:
        # 验红：故意把两副老师当成同一副（塌缩率照抄），闸必须红着退出且不写档案。
        # 不验这一条，"闸门通过"这句话没有内容——恒绿的守卫等于没有守卫。
        print("[验红] --red-check：把两副老师的塌缩率改成同一个数，闸① 必须红着退出；"
              "跑通即说明干湿闸不是恒绿（本课读数不入库）")
        dy = arc["dryness"]
        if dy is not None:
            dy["b_collapse_pct"], dy["b_teacher"] = dy["a_collapse_pct"], dy["a_teacher"]
    dry = dryness_gate(arc)          # 训练前退出：换老师却没换干湿，这一发就没有判别量
    print(f"[开跑] {dataset} | 老师 {teacher} | S{width} | {tier} | "
          f"复检 T={tlist} 共 {len(cells)} 格 {list(cells)}"
          f" × {len(seeds)} 条随机流（老 {len(OLD_SEEDS)} 条只对账、新 {len(NEW_SEEDS)} 条才是复现）"
          f"| 每种子一副 CE 锚")
    print(f"[档案] 逐位对账基准来自 {arc['file']}；该档的底 {arc['floor_imgs']} 张 = {arc['floor_kind']}")

    xt_all, yt_all, xte, yte = ds["loader"]()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    tnet = build_cnn()
    tnet.load_state_dict(torch.load(tsp["pt"], weights_only=True))
    tnet.eval()
    xd, yd = xt_all[:DATA_TIERS[tier]], torch.tensor(yt_all[:DATA_TIERS[tier]])
    soft = logits_of(tnet, xd)

    rows: dict[str, list[dict]] = {k: [] for k in cells}
    parity: dict[str, bool] = {}
    for seed in seeds:
        line = [f"  种子 {seed:<6}"]
        ce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
        ce_pred = predict(ce, xte, False)
        ce_wrong = int((ce_pred != yte).sum())
        line.append(f"CE 错 {ce_wrong:>4}")
        for k, a, t in spec:
            m, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=a, t=t)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            broken = int(((ce_pred == yte) & (p != yte)).sum())
            fixed = int(((ce_pred != yte) & (p == yte)).sum())
            r = {"seed": seed, "alpha": a, "T": t, "wrong_kd": wrong, "wrong_ce": ce_wrong,
                 "gap_imgs": ce_wrong - wrong, "broken": broken, "fixed": fixed,
                 "p_sign_img": round(sign_test_p(broken, fixed), 6), "new": seed not in OLD_SEEDS}
            rows[k].append(r)
            line.append(f"{k}:{ce_wrong - wrong:+d}")
            if seed in OLD_SEEDS:
                ref = arc["cells"][k].get(seed)
                ok = ref is not None and all(r[x] == ref[x] for x in FIELDS)
                parity[f"{k} 种子{seed}"] = bool(ok)
        print("  ".join(line))

    print(f"\n[对账] 旧种子逐位复现（错图数/翻坏/翻正四项全等，基准 = {arc['file']}）：")
    for k, v in parity.items():
        print(f"  {'✓' if v else '✗'} {k}")
    n_ok = sum(parity.values())
    if n_ok != len(parity):
        print(f"  **{n_ok}/{len(parity)} 通过——不通过则本课与那本账不是同一台仪器，下面所有读数作废**")

    stats: dict[str, dict] = {}
    print(f"\n[判决] 判据（训练前立好）：mean ≥ +{H_MIN_EFFECT} 张且 95% CI 下界 > 0 → H+；CI 跨 0 → H0")
    for k in cells:
        for tag, sub in (("8 条全用", rows[k]),
                         ("只用新 5 条", [r for r in rows[k] if r["new"]])):
            s = summarize([r["gap_imgs"] for r in sub], f"{k} | {tag}")
            s["counts_as_verdict"] = tag == "8 条全用"
            s["verdict"] = (f"{'H+ 这一格确实值几张图（原措辞要改口）' if s['ci95_imgs'][0] > 0 and s['mean_gap_imgs'] >= H_MIN_EFFECT else ('H− 反向，先查仪器' if s['ci95_imgs'][1] < 0 else 'H0：CI 跨 0 → 边缘读数不算疗效')}"
                            f"（{k} @ {tier}）")
            stats.setdefault(k, {})[tag] = s
            print(f"  {k}（{tag}，n={s['n_seeds']}）平均 {s['mean_gap_imgs']:+.1f} 张，"
                  f"95% CI [{s['ci95_imgs'][0]:+.1f}, {s['ci95_imgs'][1]:+.1f}]，"
                  f"同号 {s['same_sign_seeds']} 条（二项 p={s['p_seeds_binomial']}）→ {s['verdict']}")

    eff = None
    eff_stats: dict[str, dict] = {}
    if len(temps) > 1:
        vals = t_effects_by_seed(spec, rows, seeds, temps)
        print(f"\n[T 主效应·按种子配对] 每格各自有 CI 不等于主效应有 CI：这一行才是 {tlist} 那一刀的误差棒")
        for tag, sub in (("8 条全用", vals), ("只用新 5 条", vals[len(OLD_SEEDS):])):
            s = summarize(sub, f"T主效应 | {tag}")
            s["counts_as_verdict"] = tag == "8 条全用"
            s["verdict"] = effect_verdict(s["mean_gap_imgs"], s["ci95_imgs"], temps, f"{dataset}/{tier}（{tag}）")
            eff_stats[tag] = s
            print(f"  {tag}（n={s['n_seeds']}）平均 {s['mean_gap_imgs']:+.1f} 张，"
                  f"95% CI [{s['ci95_imgs'][0]:+.1f}, {s['ci95_imgs'][1]:+.1f}]，"
                  f"同号 {s['same_sign_seeds']} 条（二项 p={s['p_seeds_binomial']}，"
                  f"σ {s['sd_imgs']} 张）→ {s['verdict']}")
        eff = {"per_seed_imgs": [round(v, 1) for v in vals], "stats": eff_stats,
               "n_seed_floor_proxy_imgs": arc["floor_imgs"],
               "note": f"对照：第6课那条 n=3 的主效应 = {arc['file']} 里的读数，"
                       f"它的闸门用的是本档极差底 {arc['floor_imgs']} 张（单臂漂移），"
                       f"而这里是配对序列自己的 σ {eff_stats['8 条全用']['sd_imgs']} 张"}

    lim = [f"只补了 T={tlist} 这几列的格，其余格的种子数仍是 3 条",
           "「只用新 5 条」那一档的样本标准差只有 4 个自由度，CI 宽度自己就不确定，"
           "它只当敏感性检查（结论别被对账那 3 条主导），不参与判决",
           "8 条同架构同数据同老师同任务：这是「同一副仪器把读数做精」，"
           "**新增的不是独立性，是分辨率**",
           "逐图 p 在种子间不独立，只报进表不参与判决；判决用 CI + 以种子为单位的二项检验"]
    if dataset == "fashion":
        lim += ["只低数据档：满数据档那一刀第6课已经 1.12x 过闸，这一发不重复花那份钱",
                "只一副老师、只 S128、没跑 α→0 阶梯（与第6课同）",
                "第6课的自我审计仍然成立：换数据集同时换了干湿与任务难度，"
                "这一发把『这一档有没有方向』量清了，仍没把『符号跟数据量』与『符号跟干湿』分开"]
    elif teacher != "A":
        lim += [f"只换了老师这一维：数据集/宽度/档位/α 对/种子名单与老师 A 那一本**逐字相同**，"
                f"所以两本的差可以读成软标签之差（干湿闸 {dry['gap_pp']:+}pp 已先过）",
                "D 是欠训练 3 epoch 的同族 CNN：它的精度也比 A 低（94.05% vs 97.98%），"
                "所以这一维拆开的是「知识源」，拆不掉「老师强弱」——与第5课同一句代价",
                "两副老师都是 MNIST 上训的：这一格配 Fashion 那两格只能判 2×2 的角，"
                "不能判「翻号跟任务」是任务的哪一条性质"]
    else:
        lim += ["满数据档那一发效力量级 +10 张 = 0.1pp，对选型结论不构成改口，只改措辞；"
                "低数据档若成立则是 +0.7pp 量级，会改进选"]
    report = {"dataset": dataset, "teacher": teacher, "dryness_gate": dry,
              "tier": tier, "width": width, "target_Ts": list(temps),
              "cells": list(cells), "seeds": seeds,
              "old_seeds_for_parity": list(OLD_SEEDS), "new_seeds": list(NEW_SEEDS),
              "h_min_effect_imgs": H_MIN_EFFECT, "archive_parity": parity,
              "archive_parity_all_pass": bool(n_ok == len(parity)),
              "archive_file": arc["file"], "floor_imgs": arc["floor_imgs"],
              "floor_kind": arc["floor_kind"], "teacher_weight": tsp["pt"].name,
              "test_n": int(len(yte)), "raw_rows": rows, "stats": stats,
              "t_main_effect_paired": eff,
              "why_more_seeds": "第4课d 用每格自己的 ±2SE 重算闸门，T=1 那两格从「0.24x 底=不算」"
                                "翻成「1.01x=刚好过」——但 ±2SE 是按独立合成的下界，n=3 的极差又是 σ 的粗估，"
                                "所以翻档本身不能当结论，只能当「值得把种子加到 8 条」的理由。"
                                "低数据档 T=6 那条正号（+71.7 张 = 该档底 1.9x）同理：不能因为它大就免检。",
              "limitations": lim}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{len(seeds) * (len(cells) + 1)} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="闸门边缘读数复检：补五条随机流，判它是疗效还是刻度假象")
    ap.add_argument("--dataset", choices=tuple(DATASETS), default="mnist",
                    help="换任务时 loader/老师权重/对账账本/该档的底四样一起换（见 DATASETS）")
    ap.add_argument("--teacher", default="A",
                    help="用哪一副老师的软标签（每个数据集登记的老师见 TEACHERS）；"
                         "换老师时权重/对账账本/干湿闸基准三样一起换，档案名加 _老师 后缀")
    ap.add_argument("--red-check", action="store_true",
                    help="验红：把干湿闸的两个基准数改成同一个，确认它会红着退出（不写档案）")
    ap.add_argument("--width", type=int, choices=(128, 64), default=WIDTH)
    ap.add_argument("--tier", choices=tuple(DATA_TIERS), default=TIER,
                    help="数据档位；对账基准与产物文件名都跟着档位走（低数据 → *_low）")
    ap.add_argument("--t", type=float, nargs="+", choices=(1.0, 3.0, 6.0), default=[1.0],
                    help="复检哪一列/哪几列温度的两格（α=1.0 与 α=0.7）；"
                         "给两个温度就额外算**按种子配对的 T 主效应**")
    args = ap.parse_args()
    main(args.dataset, args.teacher, args.width, args.tier, tuple(args.t),
         red_check=args.red_check)
