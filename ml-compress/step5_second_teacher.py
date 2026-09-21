# -*- coding: utf-8 -*-
"""第5课 换老师：把"符号由数据量决定"这句话交给第二份知识源去判（`step5_second_teacher.py`）

第4课的三笔复证全部共享同一副老师：换宽度（S64）、换数据档（低数据）、换随机流（n=8）——
它们都在**同一份软标签**上做文章，所以严格说只复证了"学生那侧"和"样本流那侧"。
本课换掉唯一没动过的东西：老师。

第二副老师不是外部模型，而是本板块自己那把刀：**第3课的手术刀（NetU L1 重要性剪 conv2
一半输出通道 + 微调 8 epoch）** 造出来的 R50F。"它是另一份知识而不是 A 的影子"这件事要现场
证明（两副老师在测试集上不一致多少张、软标签之间的 KL），不然换老师等于换个种子。

要判的两说（在看到 B 的零训练体检之后、任何学生训练之前写死，此后不许改口）：
  H_数据量：T 的符号跟着**学生能看到多少张**走（满数据有害、低数据有益），换老师不改。
  H_老师个性：T 的符号跟着**老师软标签干不干**走——剪过一半的老师更不自信 ⇒ 暗知识更多 ⇒
             满数据档那点稀释之害应当收缩，而低数据档那个正号本来就不是温度的功劳。

仪器上两件白送的检查：
  ① **CE 锚与第4课c 必须逐位相同**（CE 臂不吃软标签，同种子同数据同宽度 ⇒ 与老师无关）。
     不过则说明两副老师的读数差里混了随机流，本课作废。
  ② **两把尺一起报**：第4课d 量过"借来的底对单格偏大、跨格 ±2SE 是下界"，所以跨格量（T 主效应）
     用借来的底（保守），单格同时报自己的 ±2SE；只过自己那把的读数标"待补种子"，不当判决。

`--width 64` 是把"换老师"与"换宽度"两维叠起来的一发：老师 A 的对照读数、种子底、CE 锚
全部从 **S64 自己的** v4b/v4c 档案读（不抄字面量，抄了就对回 S128 的账）。要说清楚它买到的是什么：
S64 从没跑过低数据档，而两副老师的塌缩率差不到 10pp（判别闸门），于是这一发**没有判别量**——
它只回答"满数据档那个负号，换一副学生 *并且* 换一份软标签之后还在不在"，属复现不属裁决。

`--teacher C` 是第三副老师：同一把 NetU 刀、**只剪 20%** 的轻一刀（第3课量过它更准），
于是「换老师」从 1 个重复升到 2 个，并且多出一条只有两个以上重复才有的读数——
⑥ 把各副老师的**塌缩率**与它的**满数据 T 主效应幅度**并排，看干湿排序与伤害排序同不同向
（老师个性说的核心主张就是这个排序；三副老师全同一族，所以它只算一票支持）。

`--teacher D` 是第四副老师，也是**第二族旋钮**：不动刀。同一副未剪枝 CNN、同一批图、
同一配方逐行照搬，**只把 epoch 从 20 砍到 3**（欠训练 ⇒ 软标签应当更湿）。为什么要它：
⑥ 那条"越干越伤"目前由 B/A/C 三点拿到，而三点全出自同一把刀，**干湿、剪枝剂量、老师精度
三者同向变化，分不开**——补一个由"训练轮数"决定的干湿点，才有一样东西可判：
若 D 明显更湿而满数据之害照旧（≈A 的幅度），⑥ 的排序就被判成同族刀的巧合；
若 D 更湿且伤害按序收缩，"幅度跟软标签干湿"就跨过了一个独立旋钮。
代价照旧要写清：D 的精度也更低（欠训练），所以它拆开的是"剂量"这一维，拆不掉"老师强弱"这一维。
"逐行照搬"这句话当场验过（一次性，不在本课跑）：把 train_recipe 的 epochs 换回 20 重训一遍，
得到的权重与 models/mnist_cnn_fp32.pt **逐位相同** ⇒ D 确是 A 的优化轨迹截在第 3 轮那个点，
换掉的真的只有轮数。（torch 版本或线程数一变就要重验这条，它不是恒成立的。）

档案名跟着（宽度, 老师）走，**已入库就拒绝重跑**：同一把刀重跑会无声盖掉入库的判决。

跑法（每发约 4~15 分钟，纯 CPU，零 API 费）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step5_second_teacher.py
  # 叠"换学生"这一维（读 S64 自己的 v4b/v4c 档案，写 compress_report_v5_s64.json）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step5_second_teacher.py --width 64
  # 第三副老师（轻一刀，写 compress_report_v5_c.json，并把 B 的账并进来读 ⑥）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step5_second_teacher.py --teacher C
  # 第四副老师（不动刀、只砍训练轮数 ⇒ 干湿换了第二族旋钮，写 compress_report_v5_d.json）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step5_second_teacher.py --teacher D
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from step1_quantize import MODELS, build_cnn, payload_kb, predict
from step3_prune import channel_importance, fine_tune, keep_indices, rebuild
from step4_distill import TIERS, VAL_N, logits_of, sign_test_p, soft_label_audit, train_student
from step4c_alpha_T_sweep import v4b_path
from step4c_alpha_T_sweep import out_path as v4c_path
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
WIDTH = 128                                    # 默认副；--width 64 叠上"换宽度"这一维
ALPHAS = (0.7, 1.0)                            # 只留两档：本课的判别量是 T 的**符号**，两档够定符号
TEMPS = (1.0, 3.0, 6.0)
SEEDS = (42, 7, 2026)                          # 与第4课c 同一批种子：CE 锚可逐位对账
D4 = 1.693                                     # n=3 正态极差期望，极差 → σ̂（第4课d 的公式）
# 每一行是一把**不同的刀**（不是换种子）：换老师要换的是知识源。B 就是第3课那把，
# 它的档案（compress_report_v5*.json）已经入库，所以这一行的剂量/文件名一动，旧账就对不上了。
TEACHERS = {"B": (0.5, "mnist_cnn_pruned50.pt", "第3课手术刀（NetU L1 剪 conv2 50% + 微调 8 epoch，种子 42）"),
            "C": (0.2, "mnist_cnn_pruned20.pt", "更轻的一刀（NetU L1 只剪 conv2 20% + 同一套微调，种子 42）")}
# 老师 D 的干湿由"训练轮数"决定，与剪枝剂量无关——⑥ 那条排序因此才有第二族的点可判（设计见文档串）。
TRAINED = {"D": (3, "mnist_cnn_underfit3.pt", "未剪枝 CNN、同配方同种子，只训 3 epoch（欠训练 ⇒ 软标签应当更湿）")}
ALL_TEACHERS = {**TEACHERS, **TRAINED}
FAMILY = {"A": "未剪枝基线（20 epoch）", "B": "剪枝剂量", "C": "剪枝剂量", "D": "训练轮数（欠训练）"}
UNPRUNED_C2 = 3                                   # build_cnn 里 conv2 的下标；不动刀那把没剪，通道数照实读


def key(a: float, t: float) -> str:
    return f"a{a:g}_T{t:g}"


def baselines(width: int) -> dict[str, dict]:
    """老师 A 在这一副宽度上的全部对照读数，逐档从**那一档自己的** v4b/v4c 档案读：
    借来的种子底、T 主效应、CE 锚错图数。三样全在档案里，抄一份进代码就等于对回旧账。

    只收两本档案**都在**的档：S64 从没跑过低数据档，硬跑等于拿 S128 的底去闸 S64 的读数。"""
    out: dict[str, dict] = {}
    for tier in TIERS:
        v4b, v4c = v4b_path(width, tier), v4c_path(width, tier)
        if not v4c.exists():
            continue
        d = json.loads(v4c.read_text(encoding="utf-8"))
        if int(d["width"]) != width or d.get("tier") != tier:
            raise SystemExit(f"{v4c.name} 自称 S{d.get('width')}/{d.get('tier')!r}，"
                             f"本课当 S{width}/{tier} 用——先看一眼档案再跑")
        b = v4b.exists() and json.loads(v4b.read_text(encoding="utf-8")).get("verdicts", {}).get(tier)
        if not b:
            continue
        out[tier] = {"floor": int(b["seed_floor_imgs"]["KD"]),
                     "t_main_A": float(d["verdicts"]["T_main_effect"]
                                        ["T_max_minus_T_min_pooled_over_alpha"]),
                     "ce_anchors": {(tier, int(s)): int(r["wrong"])
                                    for s, r in d["ce_ref_by_seed"].items()}}
    if not out:
        raise SystemExit(f"S{width} 一本可用的对照档案都没有（需要 compress_report_v4b*/v4c* 成对）")
    return out


def cells_table(rows: dict[str, list[dict]]) -> tuple[dict, dict]:
    """每格：三种子平均 gap、跨种子极差、**自己的** ±2SE（第4d 的公式）、合并符号检验 p。"""
    grid, stats = {}, {}
    for k, rs in rows.items():
        gaps = [r["gap_imgs"] for r in rs]
        tb, tf = sum(r["broken"] for r in rs), sum(r["fixed"] for r in rs)
        rng = max(gaps) - min(gaps)
        gate = round(2 * (rng / D4) / math.sqrt(len(gaps)), 1)
        mean = round(sum(gaps) / len(gaps), 1)
        grid[k] = {"per_seed": rs, "gaps_imgs": gaps, "mean_gap_imgs": mean, "range_imgs": rng,
                   "pooled": {"broken": tb, "fixed": tf}, "p_pooled": round(sign_test_p(tb, tf), 8)}
        stats[k] = {"mean": mean, "range": rng, "gate": gate, "p_pooled": grid[k]["p_pooled"],
                    "over_own_gate": bool(abs(mean) > gate),
                    "n_positive_seeds": sum(1 for g in gaps if g > 0)}
    return grid, stats


def train_recipe(epochs: int, xt: torch.Tensor, yt) -> nn.Module:
    """train_baseline 的配方逐行照搬（同 seed 起点、同 Adam 1e-3、同 bs 128、同 randperm 流），
    **只把 20 epoch 换成 epochs**。于是"训 3 轮"这副权重就是 A 的训练轨迹在第 3 轮的那个点：
    唯一的旋钮是轮数，初始化、数据顺序、批次划分全都没动（动刀那两把做不到这一点）。"""
    torch.manual_seed(42)
    m = build_cnn()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(xt)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            k = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(m(xt[k]), torch.tensor(yt[k]))
            loss.backward()
            opt.step()
    return m


def teacher_knife(name: str, force_retrain: bool = False) -> tuple[torch.nn.Module, dict]:
    """指定那把刀。TEACHERS 里的走第3课的手术刀 + 同一套微调（keep 索引由 fp32 权重算出，
    确定性，所以缓存的权重能原样装回去）；TRAINED 里的**不动刀**，只换训练轮数。
    两条路都把权重存档，复用与新建如实写在 info['action'] 里（打印不许说"新剪"来糊弄没剪的那把）。"""
    fname = ALL_TEACHERS[name][1]
    pt = MODELS / fname
    base = build_cnn()
    base.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    if name in TRAINED:
        epochs, _, source = TRAINED[name]
        if pt.exists() and not force_retrain:
            m = build_cnn()
            m.load_state_dict(torch.load(pt, weights_only=True))
            return m.eval(), {"reused": True, "kept_channels": int(base[UNPRUNED_C2].out_channels),
                              "path": fname, "source": source, "action": "复用缓存权重（本轮没重训）"}
        xt_all, yt_all, _, _ = load_split()
        m = train_recipe(epochs, xt_all, yt_all)
        torch.save(m.state_dict(), pt)
        return m.eval(), {"reused": False, "kept_channels": int(base[UNPRUNED_C2].out_channels),
                          "path": fname, "source": source,
                          "action": f"本轮按基线配方从头训 {epochs} epoch 并存档"}
    ratio, _, source = TEACHERS[name]
    keep = keep_indices(channel_importance(base), ratio)
    if pt.exists() and not force_retrain:
        m = rebuild(base, keep)
        m.load_state_dict(torch.load(pt, weights_only=True))
        return m.eval(), {"reused": True, "kept_channels": int(keep.numel()),
                           "path": fname, "source": source, "action": "复用缓存权重（本轮没剪没微调）"}
    xt_all, yt_all, _, _ = load_split()
    cut = len(xt_all) - VAL_N
    torch.manual_seed(42)
    m, hist = fine_tune(rebuild(base, keep), xt_all[:cut], yt_all[:cut], xt_all[cut:], yt_all[cut:])
    torch.save(m.state_dict(), pt)
    return m.eval(), {"reused": False, "kept_channels": int(keep.numel()),
                      "path": fname, "source": source, "last_val_acc": hist[-1]["val_acc"],
                      "action": "本轮新剪+微调并存档"}


def out_name(width: int, teacher: str) -> str:
    """档案名带上**是哪把刀**：第三副老师的读数不能盖掉第5课入库的那本账。"""
    stem = "compress_report_v5" if teacher == "B" else f"compress_report_v5_{teacher.lower()}"
    return stem + ("" if width == 128 else f"_s{width}") + ".json"


def sibling_readings(width: int, name: str) -> dict[str, dict]:
    """同宽度下**已入库**的其他老师的（塌缩率, 满数据 T 主效应）。只读同宽度的账：
    把 S128 的 T 主效应拿去和 S64 的并排，就是把"换学生"和"换老师"两维混成一维。"""
    pts: dict[str, dict] = {}
    for other in sorted(set(ALL_TEACHERS) - {name}):
        p = HERE / out_name(width, other)
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        declared = d.get("teacher_name", "B")   # 第5课入库的两本没有这个字段，它们就是 B
        if int(d.get("width", 0)) != width or declared != other:
            raise SystemExit(f"{p.name} 自称老师 {declared!r}/S{d.get('width')}，"
                             f"本课当 {other}/S{width} 用——先看一眼那本账再并表")
        full = d["tiers"].get("满数据8500")
        if not full:
            continue
        pts[other] = {"collapse_pct": d["soft_label_audit"]["满数据8500"][other]["原始"]["pct_top1_gt_0.99"],
                      "t_main_imgs": full["t_main_effect_imgs"], "from": p.name}
    return pts


def main(width: int = WIDTH, name: str = "B") -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    base = baselines(width)
    tiers = [t for t in TIERS if t in base]                     # 这一副宽度有据可查的档
    floor = {t: base[t]["floor"] for t in tiers}
    T_MAIN_A = {t: base[t]["t_main_A"] for t in tiers}
    ce_arch = {k: v for t in tiers for k, v in base[t]["ce_anchors"].items()}
    missing = [t for t in TIERS if t not in tiers]
    # 收缩判据 = 老师 A 满数据之害的一半：跟着档案算，换宽度换的是它自己的那个数
    SHRINK_LINE = round(abs(T_MAIN_A["满数据8500"]) / 2, 1)
    OUT = HERE / out_name(width, name)
    if OUT.exists():
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉那本账。要重跑先删或改名，"
                         f"别在没对过账的情况下换掉已入库的判决")

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    ta = build_cnn()
    ta.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    ta.eval()
    tb, info = teacher_knife(name)
    print(f"老师 {name}（{info['kept_channels']} 通道，{info['source']}）就位 | "
          f"{info['action']} {info['path']} | 体积 {payload_kb(tb):.1f} KB")

    # ---------- 零训练一：这副新老师是"另一份知识"还是 A 的影子 ----------
    pa, pb = predict(ta, xte, False), predict(tb, xte, False)
    agree = int((pa == pb).sum())
    kl_ab = float(torch.nn.functional.kl_div(torch.log_softmax(logits_of(ta, xte), 1),
                                             torch.softmax(logits_of(tb, xte), 1),
                                             reduction="batchmean"))
    acc_a = 1 - int((pa != yte).sum()) / len(yte)
    acc_b = 1 - int((pb != yte).sum()) / len(yte)
    print(f"[独立性体检] 老师 A 与 {name} 在 {len(yte)} 张测试图上判得不一致 **{len(yte) - agree} 张**"
          f"（相同率 {agree / len(yte):.1%}）；软标签 KL(A‖{name}) = {kl_ab:.4f}（≈0 就是影子，换老师等于没换）")
    print(f"[独立性体检] 各自测试精度 A {acc_a:.4f} / {name} {acc_b:.4f}")
    versus_prev: dict[str, dict] = {}
    if name != "B" and (MODELS / TEACHERS["B"][1]).exists():
        tp, _ = teacher_knife("B")
        p2 = predict(tp, xte, False)
        versus_prev["B"] = {
            "disagree_imgs": len(yte) - int((p2 == pb).sum()),
            "kl_this_given_B": round(float(torch.nn.functional.kl_div(
                torch.log_softmax(logits_of(tb, xte), 1),
                torch.softmax(logits_of(tp, xte), 1), reduction="batchmean")), 4),
            "test_acc": round(1 - int((p2 != yte).sum()) / len(yte), 4)}
        print(f"[独立性体检] {name} 与 B：判得不一致 {versus_prev['B']['disagree_imgs']} 张，"
              f"KL({name}‖B) = {versus_prev['B']['kl_this_given_B']:.4f}，"
              f"B 的精度 {versus_prev['B']['test_acc']:.4f}"
              f"（{name} 若几乎是 B 的复刻，这一发就只是把同一份软标签喂了两遍，买不到第二个重复）")

    # ---------- 零训练二：新老师手里有没有暗知识（预言看着这一栏立） ----------
    audit_a, audit_b = {}, {}
    for tier, n in TIERS.items():
        xd = xt_all[:n]
        audit_a[tier] = soft_label_audit(logits_of(ta, xd))
        audit_b[tier] = soft_label_audit(logits_of(tb, xd))
    ra = audit_a["满数据8500"]["原始"]
    rb = audit_b["满数据8500"]["原始"]
    gap_pp = round(ra["pct_top1_gt_0.99"] - rb["pct_top1_gt_0.99"], 1)
    drier = gap_pp < 0
    print("\n[软标签体检（拟合集 8500 张，原始温度）]")
    print(f"  A：top1 {ra['mean_top1']}，塌缩率 {ra['pct_top1_gt_0.99']}%，第二名概率 {ra['mean_p2']}")
    print(f"  {name}：top1 {rb['mean_top1']}，塌缩率 {rb['pct_top1_gt_0.99']}%，第二名概率 {rb['mean_p2']}")

    discriminating = abs(gap_pp) >= 10.0
    print("\n[先立预言（此后不许改口）]")
    print(f"  判别量 = 每档的 T 主效应（α∈{ALPHAS} 平均，T=6 − T=1，单位张，负=KD 更差）；"
          f"老师 A 的实测：{T_MAIN_A}")
    print(f"  {name} 比 A 的塌缩率{'低' if not drier else '高'} {abs(gap_pp)}pp"
          f"（p2 {ra['mean_p2']}→{rb['mean_p2']}）→ 这一副软标签**{'更干' if drier else '更湿'}**")
    if discriminating:
        print(f"  → 这一发有判别力（干湿差 {abs(gap_pp)}pp ≥ 10pp）。"
              + (f"H_老师个性：{name} 的软标签比 A 干，满数据之害应当**更大**"
                 f"（|T 主效应| 比 A 的 {T_MAIN_A['满数据8500']:.1f} 至少再坏一个该档底 "
                 f"{floor['满数据8500']} 张）；H_数据量：幅度仍在 A 的 ±{SHRINK_LINE} 张邻域内、符号为负"
                 if drier else
                 f"H_老师个性：满数据 T 主效应收缩到 ±{SHRINK_LINE} 张以内"
                 f"（软标签不那么干，稀释就没那么伤）；H_数据量：符号仍为负、幅度同量级（>{SHRINK_LINE} 张）")
              + f"。低数据档是第二个判别点：老师个性说 {'暗知识更稀，正号应当比 A 小' if drier else '新老师的暗知识更值钱、正号应当更大'}，"
                f"数据量说正号照搬、幅度不必变大")
    else:
        print(f"  → 两副老师的软标签一样干（塌缩率只差 {abs(gap_pp)}pp < 10pp），于是两说对**满数据档**"
              f"给出同一个预测（T 主效应为负、幅度同量级）：这一档只算「形状复现」，不算判别。"
              + (f"本课唯一的判别量是低数据档的符号：数据量说 → 翻正且过闸；"
                 f"老师个性说 → A 那发的 {T_MAIN_A['低数据2000']:+.1f} 张是这副老师的运气，"
                 f"{name} 这里应当回落到 ±底 以内" if "低数据2000" in tiers else
                 f"而 S{width} 从没跑过低数据档（档案里没有），所以这一发**连那一个判别量都没有**："
                 f"它只回答『满数据档的负号换一副学生、再换一份软标签之后还在不在』，属复现不属裁决"))
    for tier in tiers:
        print(f"  闸门（{tier}）：借来的底 {floor[tier]} 张（跨格量用它）；单格同时报自己的 ±2SE（D4={D4}）")
    if missing:
        print(f"  （{ '、'.join(missing) } 在 S{width} 没有对照档案 → 本课不跑，也不拿别的档的底来闸它）")

    # ---------- 开跑 ----------
    result: dict[str, dict] = {}
    ce_parity: list[dict] = []
    for tier in tiers:
        n = TIERS[tier]
        xd, yd = xt_all[:n], torch.tensor(yt_all[:n])
        soft = logits_of(tb, xd)                    # 全流程唯一与老师 A 不同的一行
        print(f"\n[开跑] {tier} | 学生 S{width} | {len(ALPHAS)}×{len(TEMPS)}×{len(SEEDS)} = "
              f"{len(ALPHAS) * len(TEMPS) * len(SEEDS)} 次 KD + {len(SEEDS)} 次 CE 锚")
        rows: dict[str, list[dict]] = {key(a, t): [] for a in ALPHAS for t in TEMPS}
        ce_ref: dict[int, int] = {}
        for seed in SEEDS:
            ce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
            ce_pred = predict(ce, xte, False)
            ce_wrong = int((ce_pred != yte).sum())
            ce_ref[seed] = ce_wrong
            want = ce_arch.get((tier, seed))
            ok = ce_wrong == want
            ce_parity.append({"tier": tier, "seed": seed, "wrong_ce": ce_wrong,
                              "wrong_ce_v4c": want, "match": bool(ok)})
            line = [f"  种子 {seed:<5} CE 错 {ce_wrong} 张" + ("" if ok else f"（≠第4课c 的 {want}）")]
            for a in ALPHAS:
                for t in TEMPS:
                    m, _ = train_student(width, xd, yd, soft, "KD", xv, yv,
                                         seed=seed, alpha=a, t=t)
                    p = predict(m, xte, False)
                    wrong = int((p != yte).sum())
                    broken = int(((ce_pred == yte) & (p != yte)).sum())
                    fixed = int(((ce_pred != yte) & (p == yte)).sum())
                    rows[key(a, t)].append({"seed": seed, "alpha": a, "T": t,
                                            "wrong_kd": wrong, "wrong_ce": ce_wrong,
                                            "acc_kd": round(1 - wrong / len(yte), 4),
                                            "gap_imgs": ce_wrong - wrong,
                                            "broken": broken, "fixed": fixed,
                                            "p_sign": round(sign_test_p(broken, fixed), 6)})
                    line.append(f"{key(a, t)}:{ce_wrong - wrong:+d}")
            print(" ".join(line))
        grid, stats = cells_table(rows)
        t_main = round(sum(stats[key(a, TEMPS[-1])]["mean"] - stats[key(a, TEMPS[0])]["mean"]
                           for a in ALPHAS) / len(ALPHAS), 1)
        print(f"\n[网格] {tier}（KD−CE 三种子平均，括号=该格自己的 ±2SE）：")
        for a in ALPHAS:
            print(f"  α={a:<5} " + "   ".join(
                f"T={t:<4g}{stats[key(a, t)]['mean']:+7.1f}(±{stats[key(a, t)]['gate']})" for t in TEMPS))
        ratio = round(abs(t_main) / floor[tier], 2)
        print(f"  T 主效应（T6−T1，{len(ALPHAS)} 档 α 平均）= **{t_main:+.1f} 张** "
              f"= 借来的底 {floor[tier]} 张的 {ratio}x → "
              + ("过闸" if abs(t_main) > floor[tier] else "在底以内（谈不上符号）"))
        result[tier] = {"grid": grid, "stats": stats, "t_main_effect_imgs": t_main,
                        "vs_borrowed_floor": ratio, "beats_floor": bool(abs(t_main) > floor[tier]),
                        "teacher_A_t_main_imgs": T_MAIN_A[tier],
                        "sign_agrees_with_A": bool(t_main * T_MAIN_A[tier] > 0),
                        "soft_label_audit_B": audit_b[tier], "ce_ref_by_seed": ce_ref,
                        "borrowed_floor_imgs": floor[tier]}

    # ---------- 判决 ----------
    n_pass = sum(1 for x in ce_parity if x["match"])
    print("\n[判决]")
    print(f"  ① CE 锚逐位对账（不吃软标签的那一臂，与第4课c 比）：**{n_pass}/{len(ce_parity)} 通过**"
          + ("→ 两副老师的 KD 读数差只来自软标签" if n_pass == len(ce_parity)
             else "→ 不通过：随机流都变了，本课所有跨老师对比作废"))
    for tier in tiers:
        r = result[tier]
        print(f"  ② {tier}：老师 A 的 T 主效应 {r['teacher_A_t_main_imgs']:+.1f} 张 → 老师 {name} "
              f"{r['t_main_effect_imgs']:+.1f} 张"
              f"（{r['vs_borrowed_floor']}x 底，同号 {r['sign_agrees_with_A']}，过闸 {r['beats_floor']}）")
    if n_pass != len(ce_parity):
        raise SystemExit("CE 锚没逐位复现第4课c → 两副老师的差里混进了随机流，"
                         "跨老师对比全部作废；先查 train_student/load_split 有没有被改坏，"
                         "别把这一轮的数当复证用")
    full, low = result["满数据8500"], result.get("低数据2000")
    got_positive, pred_holds = None, None
    if discriminating:
        if drier:
            bar = round(abs(T_MAIN_A["满数据8500"]) + floor["满数据8500"], 1)
            worse = bool(abs(full["t_main_effect_imgs"]) > bar)
            pred_holds = worse
            v3 = (f"软标签更干确实把满数据之害放大（|{full['t_main_effect_imgs']:+.1f}| > A 的 "
                  f"{abs(T_MAIN_A['满数据8500']):.1f} + 底 {floor['满数据8500']} = {bar}）"
                  "→ H_老师个性 拿到一票：伤害幅度跟着软标签的干湿走" if worse else
                  f"换了副更干的新老师，满数据之害没跟着放大（|{full['t_main_effect_imgs']:+.1f}| ≤ {bar}）"
                  "→ H_数据量 拿到一票：幅度钉在数据量上，不在软标签干湿上")
            print(f"  ③ 终判（干湿差 {abs(gap_pp)}pp，{name} 比 A 干）：{v3}")
        else:
            shrink = abs(full["t_main_effect_imgs"]) < SHRINK_LINE
            pred_holds = shrink
            v3 = ("软标签形状确实管着伤害幅度：H_老师个性 拿到一票，"
                  "第4课那句『符号由数据量决定』要降级为『在这一族老师上成立』" if shrink else
                  "幅度没收缩：换了一整份软标签，满数据之害还是那么大 → H_数据量 拿到一票")
            print(f"  ③ 终判（干湿差 {abs(gap_pp)}pp，{name} 比 A 湿）：{name} 的满数据 T 主效应收缩到 "
                  f"±{SHRINK_LINE} 张以内 = {shrink} → {v3}")
    elif low is None:
        survived = bool(full["sign_agrees_with_A"] and full["beats_floor"])
        v3 = (f"这一发**没有判别量**（塌缩率差 {abs(gap_pp)}pp 不到 10pp，而 S{width} 从没跑过低数据档），"
              f"所以只读形状不读裁决：老师 A 的满数据 {full['teacher_A_t_main_imgs']:+.1f} 张之害，"
              f"在换一副学生（S{width}）**并且**换一份软标签（老师 {name}）之后"
              + ("还在（同号且过该档自己的底）→ 这条负号同时扛住了架构与软标签两个维度"
                 if survived else "没扛住（反号或落进底以内）→ 负号只在 S128 那一副学生上复现过"))
        print(f"  ③ 终判（无判别量，只读形状）：{v3}")
    else:
        got_positive = bool(low["sign_agrees_with_A"] and low["beats_floor"])
        v3 = ("翻正且过闸 → 正号跨老师复现，数据量说占优：两副软标签一样干却给出同一个翻转，"
              "说明这个翻转不是这副老师的运气" if got_positive else
              "没有跨老师复现 → 第4课那条 H+ 只属于老师 A，"
              "「符号由数据量决定」这句要改口成「由这一副老师决定」")
        print(f"  ③ 终判（本课只有低数据档这一个判别量）：{name} 在 2000 张档的 T 主效应{v3}")
    for t, r in result.items():
        over = [f"{k} {s['mean']:+.1f}(±{s['gate']}, p={s['p_pooled']})"
                for k, s in r["stats"].items() if s["over_own_gate"]]
        print(f"  ④ 单格读数（{t}，只报待追不当判决）：" + ("、".join(over) or "无格过自己的闸"))
    kn = sorted(o for o in ALL_TEACHERS
                if o != name and (HERE / out_name(width, o)).exists())
    print("  ⑤ 独立性账：" + (f"换老师买到 {len(kn) + 1} 份软标签（{'、'.join(kn + [name])} 各一副，"
                              f"来自 {len({FAMILY[k] for k in kn + [name]})} 族不同的旋钮）。"
                              if name != "B" else
                              "换老师买到的只有一份软标签（1 个重复）。")
          + ("本课另叠了一维换学生（S" + str(width) + "），但两维都不在数据那侧——"
             if width != WIDTH else "")
          + "仍不是新数据集、新任务。")

    # ---------- ⑥ 多副老师并排：干湿排序与伤害幅度排序同不同向（"换老师"升到 2 个重复才有的读数） ----------
    pts = sibling_readings(width, name)
    pts["A"] = {"collapse_pct": ra["pct_top1_gt_0.99"], "t_main_imgs": T_MAIN_A["满数据8500"],
                "from": "第4课c 档案（塌缩率本轮现算）"}
    pts[name] = {"collapse_pct": rb["pct_top1_gt_0.99"], "t_main_imgs": full["t_main_effect_imgs"],
                 "from": OUT.name}
    hs = {k: abs(v["t_main_imgs"]) for k, v in pts.items()}
    order = sorted(pts, key=lambda k: (pts[k]["collapse_pct"], k))
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]]
    up = sum(1 for a, b in pairs if hs[a] < hs[b])
    down = len(pairs) - up - sum(1 for a, b in pairs if hs[a] == hs[b])
    mono = bool(up == len(pairs) and len(pairs) > 0)
    fams = {k: FAMILY[k] for k in pts}
    distinct = sorted(set(fams.values()))
    caveat = (f"只有 {len(pts)} 副老师、全都出自同一把 NetU 刀的不同剂量，支持不等于裁决"
              if len(distinct) == 1 else
              f"{len(pts)} 副老师、{len(distinct)} 族不同的旋钮（"
              + "、".join(f"{k}={fams[k]}" for k in order)
              + "）；同向不再能由'同一把刀的剂量'一句话打发，但每族仍只是个别点，支持不等于裁决")
    print("  ⑥ 老师并排（塌缩率升序 → |满数据 T 主效应|）："
          + " < ".join(f"{k} {pts[k]['collapse_pct']}%→{hs[k]:.1f}张" for k in order)
          + f" | {len(pairs)} 对里同向 {up}、反向 {down} → "
          + ("**全同向**：软标签越干、满数据之害越大，H_老师个性 在这些旋钮上拿到"
             f"{len(pts)} 个点的支持" if mono else
             "不同向：干湿排序解释不了幅度差，这条负号的幅度不跟着软标签干湿走")
          + f"（{caveat}）")
    teacher_ordering = {"teachers": {k: {**v, "abs_t_main_imgs": hs[k], "family": fams[k]}
                                     for k, v in pts.items()},
                        "knob_families": distinct,
                        "pairs": len(pairs), "concordant_pairs": up, "discordant_pairs": down,
                        "dryness_harm_monotone": mono}

    report = {"teacher": {"name": name, "source": info["source"], "family": FAMILY[name],
                          "kept_channels": info["kept_channels"], "kb": round(payload_kb(tb), 1),
                          "created_by": info["action"],
                          "reused_cache": bool(info["reused"]), "path": info["path"],
                          "test_acc": round(acc_b, 4), "teacher_a_test_acc": round(acc_a, 4),
                          "disagree_with_A_imgs": len(yte) - agree, "kl_A_given_this": round(kl_ab, 4),
                          **({"versus_previous": versus_prev} if versus_prev else {})},
              "teacher_ordering": teacher_ordering,
              "soft_label_audit": {t: {"A": audit_a[t], name: audit_b[t]} for t in TIERS},
              "prediction_ex_ante": {"discriminating": discriminating,
                                     "shrink_line_imgs": SHRINK_LINE, "collapse_gap_pp": gap_pp,
                                     "teacher_drier_than_A": bool(drier),
                                     "gate": "跨格量用借来的底，单格同时报自己的 ±2SE（第4课d 的两把尺）"},
              "ce_parity_vs_v4c": ce_parity, "ce_parity_all_pass": bool(n_pass == len(ce_parity)),
              "verdict": {"discriminating": discriminating,
                          "low_data_positive_across_teacher": (
                              None if low is None else
                              bool(low["sign_agrees_with_A"] and low["beats_floor"])),
                          "no_discriminating_quantity": bool(not discriminating and low is None),
                          "full_data_sign_survives_both_swaps": bool(
                              full["sign_agrees_with_A"] and full["beats_floor"]),
                          "full_data_shrunk": bool(discriminating
                                                   and abs(full["t_main_effect_imgs"]) < SHRINK_LINE),
                          "sentence": v3,
                          "teacher_direction": "drier_than_A" if drier else "wetter_than_A",
                          "directional_prediction_holds": (None if pred_holds is None else bool(pred_holds))},
              "width": width, "teacher_name": name, "alphas": list(ALPHAS), "temps": list(TEMPS),
              "seeds": list(SEEDS),
              "tiers": result, "tiers_skipped_no_archive": missing, "test_n": int(len(yte)),
              "limitations": ["α 只留两档（0.7/1.0）：本课的判别量是 T 的符号，两档够定符号；"
                              "α 轴的形状第4课已在三个档案里量过（0.11x/0.11x/0.33x），不在本课范围",
                              "每格 3 条随机流，且与第4课c 共用同种子 ⇒ 各副老师的 gap 可逐种子配对；"
                              + (f"「换老师」有 {len(kn) + 1} 个重复（{'、'.join(kn + [name])}），"
                                 + ("但它们同族同起点，只换了剂量：彼此之间的散布是这一族刀的下界，"
                                    "不是任意知识源的下界"
                                    if len({FAMILY[k] for k in kn + [name]}) == 1 else
                                    "且已跨 " + str(len({FAMILY[k] for k in kn + [name]}))
                                    + " 族不同的旋钮；每族仍只 1~2 个点，散布的下界仍谈不上任意知识源")
                                 if name != "B" else
                                 "但「换老师」本身只有 1 个重复：B 是个例还是这一族稀疏 CNN 的代表，没测"),
                              "±2SE 是独立合成的下界（三种子跨格共用同一批随机流），与第4课d 同一告警",
                              f"预言是在看完 {name} 的零训练软标签体检之后立的（学生训练之前，但不是看到 {name} 之前）；"
                              "体检差 <10pp 时脚本自动把本课降级为「形状复现」而不算判别，这条写在分支里",
                              "档案已入库就拒绝重跑（同一把刀重跑会盖掉入库的判决）；"
                              "要重跑先删或改名，别在没对过账的情况下换掉旧账",
                              (f"只 {('、'.join(missing))} 档在 S{width} 没有对照档案 → 本课换不了那一档，"
                               "跨老师的判决仍只在 S128 的两档上成立" if missing else
                               "学生固定 S128：换老师与换宽度不叠加，S64 上的跨老师复证要另跑一发")]}
    n_train = sum(len(r["ce_ref_by_seed"]) + len(r["grid"]) * len(SEEDS) for r in result.values())
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（{n_train} 次学生训练，老师 {name}：{info['action']}，"
          f"{len(yte)} 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="第5课 换老师（默认老师 B · S128 两档；--width 64 叠上换学生，"
                    "--teacher C/D 换第三/第四副老师）")
    ap.add_argument("--width", type=int, choices=(128, 64), default=WIDTH,
                    help="学生宽度；老师 A 的 T 主效应、种子底、CE 锚全从该宽度的 v4b/v4c 档案读")
    ap.add_argument("--teacher", choices=tuple(ALL_TEACHERS), default="B",
                    help="哪把刀造的老师：B=剪 50%、C=剪 20%（都是剪枝剂量），"
                         "D=不动刀、只把训练轮数从 20 砍到 3（欠训练）；档案名跟着老师走")
    args = ap.parse_args()
    main(args.width, args.teacher)
