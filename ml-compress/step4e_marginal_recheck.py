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
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import TIERS, VAL_N, logits_of, sign_test_p, train_student
from step4c_alpha_T_sweep import ALPHA_ANCHOR, TIER, WIDTH, key
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
OLD_SEEDS = (42, 7, 2026)                    # v4c 已跑过的三条：本课只当对账锚，不重算结论
NEW_SEEDS = (11, 202, 3003, 40004, 55555)    # 补进来的五条随机流
D4 = {3: 1.693, 5: 2.326, 8: 2.847}                # E[极差]/σ
T95 = {3: 4.303, 5: 2.776, 8: 2.365}               # 双侧 95% t 分位数
H_MIN_EFFECT = 5.0                            # 训练前立的下限：低于 5 张不值得为它开一课


def cells_for(t: float) -> list[tuple[str, float]]:
    """目标两格：α=1.0 与 α=0.7 各一格（v4d 里被埋掉的是 T=1，低数据档要复检的是 T=6）。"""
    return [(key(1.0, t), 1.0), (key(ALPHA_ANCHOR, t), ALPHA_ANCHOR)]


def archive_name(width: int, tier: str) -> str:
    """v4c 各档档案名**只有一个来源**：step4c 的 out_path。这里再抄一份后缀规则，
    下一档（_mid 这种）一进来就会去读一本不存在（或更糟：存在但是别档的）的账。"""
    return out_path(width, tier).name


def archive_cells(width: int, tier: str, cells: tuple[str, ...]) -> dict:
    """从对应宽度、对应档位的 v4c 档案里读目标两格的三种子原始读数，用来做逐位对账。"""
    p = HERE / archive_name(width, tier)
    if not p.exists():
        raise SystemExit(f"缺 {p.name}——逐位对账的基准只能从它读，先跑 step4c_alpha_T_sweep.py"
                         f"{' --width ' + str(width) if width != 128 else ''}"
                         f"{' --tier ' + tier if tier != TIER else ''}")
    d = json.loads(p.read_text(encoding="utf-8"))
    if int(d["width"]) != width or d["tier"] != tier:
        raise SystemExit(f"{p.name} 是 S{d['width']}/{d['tier']} 的档案，不是 S{width}/{tier}")
    return {"file": p.name, "tier": tier,
            "floor_imgs": int(d["seed_floor_imgs_from_v4b"]),
            "cells": {k: {r["seed"]: {"wrong_kd": r["wrong_kd"], "wrong_ce": r["wrong_ce"],
                                      "broken": r["broken"], "fixed": r["fixed"]}
                          for r in d["grid"][k]["per_seed"]} for k in cells}}


def out_path(width: int, tier: str) -> Path:
    return HERE / ("compress_report_v4e.json" if tier == TIER else "compress_report_v4e_low.json")


def summarize(gaps: list[int], label: str) -> dict:
    """以种子为单位的估计：均值 ± 95% CI（t 分布，SE 用样本标准差），外加同号计数。"""
    n = len(gaps)
    mean = sum(gaps) / n
    sd = math.sqrt(sum((g - mean) ** 2 for g in gaps) / (n - 1))
    se = sd / math.sqrt(n)
    ci = T95[n] * se          # 故意不用 .get：加种子时必须连系数表一起改，回落=用错刻度
    pos = sum(1 for g in gaps if g > 0)
    neg = sum(1 for g in gaps if g < 0)
    # 复现级检验：把"这条随机流站哪边"当 Bernoulli，双侧二项
    p_seed = min(1.0, 2 * sum(math.comb(n, k) for k in range(max(pos, neg), n + 1)) / 2 ** n)
    return {"label": label, "n_seeds": n, "mean_gap_imgs": round(mean, 1),
            "sd_imgs": round(sd, 1), "se_imgs": round(se, 1),
            "ci95_imgs": [round(mean - ci, 1), round(mean + ci, 1)],
            "gaps": gaps, "same_sign_seeds": f"{max(pos, neg)}/{n}（{'+' if pos >= neg else '−'}）",
            "p_seeds_binomial": round(p_seed, 4),
            "range_imgs": max(gaps) - min(gaps), "d4_sigma_hat": round((max(gaps) - min(gaps)) / D4[n], 1)}


def main(width: int = WIDTH, tier: str = TIER, t: float = 1.0) -> None:
    spec = cells_for(t)
    cells = tuple(k for k, _ in spec)
    arc = archive_cells(width, tier, cells)
    seeds = list(OLD_SEEDS) + list(NEW_SEEDS)
    out = out_path(width, tier)
    print(f"[开跑] S{width} | {tier} | 复检 T={t:g} 两格 {list(cells)} × {len(seeds)} 条随机流"
          f"（老 {len(OLD_SEEDS)} 条只对账、新 {len(NEW_SEEDS)} 条才是复现）| 每格一副同种子 CE 锚")
    print(f"[档案] 逐位对账基准来自 {arc['file']}；该档借来的种子底 {arc['floor_imgs']} 张")

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    xd, yd = xt_all[:TIERS[tier]], torch.tensor(yt_all[:TIERS[tier]])
    soft = logits_of(teacher, xd)

    rows: dict[str, list[dict]] = {k: [] for k in cells}
    parity: dict[str, bool] = {}
    for seed in seeds:
        line = [f"  种子 {seed:<6}"]
        ce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
        ce_pred = predict(ce, xte, False)
        ce_wrong = int((ce_pred != yte).sum())
        line.append(f"CE 错 {ce_wrong:>4}")
        for k, a in spec:
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
                ok = ref is not None and all(r[x] == ref[x] for x in
                                             ("wrong_kd", "wrong_ce", "broken", "fixed"))
                parity[f"{k} 种子{seed}"] = bool(ok)
        print("  ".join(line))

    print("\n[对账] 老三格逐位复现（错图数/翻坏/翻正四项全等）：")
    for k, v in parity.items():
        print(f"  {'✓' if v else '✗'} {k}")
    n_ok = sum(parity.values())
    if n_ok != len(parity):
        print(f"  **{n_ok}/{len(parity)} 通过——不通过则本课与 v4c 不是同一台仪器，下面所有读数作废**")

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

    report = {"tier": tier, "width": width, "target_T": t, "cells": list(cells), "seeds": seeds,
              "old_seeds_for_parity": list(OLD_SEEDS), "new_seeds": list(NEW_SEEDS),
              "h_min_effect_imgs": H_MIN_EFFECT, "archive_parity": parity,
              "archive_parity_all_pass": bool(n_ok == len(parity)),
              "archive_file": arc["file"], "borrowed_floor_imgs": arc["floor_imgs"],
              "test_n": int(len(yte)), "raw_rows": rows, "stats": stats,
              "why_more_seeds": "第4课d 用每格自己的 ±2SE 重算闸门，T=1 那两格从「0.24x 底=不算」"
                                "翻成「1.01x=刚好过」——但 ±2SE 是按独立合成的下界，n=3 的极差又是 σ 的粗估，"
                                "所以翻档本身不能当结论，只能当「值得把种子加到 8 条」的理由。"
                                "低数据档 T=6 那条正号（+71.7 张 = 该档底 1.9x）同理：不能因为它大就免检。",
              "limitations": [f"只补了 T={t:g} 两格，其余格的种子数仍是 3 条",
                              "「只用新 5 条」那一档的样本标准差只有 4 个自由度，CI 宽度自己就不确定，"
                              "它只当敏感性检查（结论别被对账那 3 条主导），不参与判决",
                              "8 条同架构同数据同老师：这是「同一副仪器把读数做精」，不是独立复证",
                              "逐图 p 在种子间不独立，只报进表不参与判决；判决用 CI + 以种子为单位的二项检验",
                              "满数据档那一发效力量级 +10 张 = 0.1pp，对选型结论不构成改口，只改措辞；"
                              "低数据档若成立则是 +0.7pp 量级，会改进选"]}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{len(seeds) * (len(cells) + 1)} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="闸门边缘读数复检：补五条随机流，判它是疗效还是刻度假象")
    ap.add_argument("--width", type=int, choices=(128, 64), default=WIDTH)
    ap.add_argument("--tier", choices=tuple(TIERS), default=TIER,
                    help="数据档位；对账基准与产物文件名都跟着档位走（低数据 → *_low）")
    ap.add_argument("--t", type=float, choices=(1.0, 3.0, 6.0), default=1.0,
                    help="复检哪个温度那一列的两格（α=1.0 与 α=0.7）")
    args = ap.parse_args()
    main(args.width, args.tier, args.t)
