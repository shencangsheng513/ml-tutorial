# -*- coding: utf-8 -*-
"""第4课的续账：还「每臂只训一次」这笔债——换随机流重训，看 KD 判决活不活。

第4课 limitations 里自己写着：**符号检验只管图间配对，不管种子间方差**。
而固定种子下重跑是逐位复现（精度/翻牌/KB 三次运行一字不差，只有墙钟在漂），
所以"再跑一遍"根本测不到那条方差——**必须换随机流**。这一课就做这一件事：

  判决 A（满数据 S128）：KD −0.63pp、翻坏131/翻正68、p=9.4e-06  → 换种子还同向吗？
  判决 B（低数据 S128）：KD +0.27pp、翻坏135/翻正162、p=0.131   → 那个正号稳不稳？

顺带产出本板块缺的一件仪器：**种子底**——同一份数据、同一个目标函数，只换随机流，
精度能差几张图。有了它，以后任何"小于 N 张"的单臂读数才知道算不算数。

零 API 费，纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4b_seed_variance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import TIERS, VAL_N, logits_of, sign_test_p, train_student
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
OUT = HERE / "compress_report_v4b.json"
V4 = HERE / "compress_report_v4.json"
SEEDS = (42, 7, 2026)                             # 42 是第4课那一轮，留着当逐位对账的锚
WIDTH = 128
# 第4课档案值（committed run）——种子 42 必须逐位复现，否则说明本课改坏了 train_student
ARCHIVE = {"满数据8500": {"CE": 0.9326, "KD": 0.9263}, "低数据2000": {"CE": 0.9045, "KD": 0.9072}}


def one_pair(teacher: torch.nn.Module, xd, yd, soft, xv, yv, xte, yte, seed: int) -> dict:
    """一个种子内的 CE/KD 配对：与第4课同款（同种子重播初始化 + 私有 Generator 批次流）。"""
    preds = {}
    for mode in ("CE", "KD"):
        m, _ = train_student(WIDTH, xd, yd, soft, mode, xv, yv, seed=seed)
        preds[mode] = predict(m, xte, False)
    ce, kd = preds["CE"], preds["KD"]
    acc = {k: float((p == yte).mean()) for k, p in preds.items()}
    broken = int(((ce == yte) & (kd != yte)).sum())
    fixed = int(((ce != yte) & (kd == yte)).sum())
    return {"seed": seed, "acc_ce": round(acc["CE"], 4), "acc_kd": round(acc["KD"], 4),
            "gap_pp": round((acc["KD"] - acc["CE"]) * 100, 2),
            "gap_imgs": int(round((acc["KD"] - acc["CE"]) * 10000)),
            "flips": {"broken": broken, "fixed": fixed},
            "p_sign": round(sign_test_p(broken, fixed), 6),
            "wrong_ce": int((ce != yte).sum()), "wrong_kd": int((kd != yte).sum())}


def spread(xs: list[int]) -> dict:
    """跨种子的极差与均值——这就是「重训一次」的误差棒，单位是张。"""
    return {"min": min(xs), "max": max(xs), "range": max(xs) - min(xs),
            "mean": round(sum(xs) / len(xs), 2)}


def main() -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()

    print(f"[先立仪器] 种子 42 必须逐位复现第4课那一轮，否则是本课把 train_student 改坏了："
          f"对照 {V4.name}（committed）")
    parity = {}
    tiers: dict[str, list[dict]] = {}
    for tier, n_data in TIERS.items():
        xd, yd = xt_all[:n_data], torch.tensor(yt_all[:n_data])
        soft = logits_of(teacher, xd)             # 软标签与随机流无关：各种子共用同一份
        rows = [one_pair(teacher, xd, yd, soft, xv, yv, xte, yte, s) for s in SEEDS]
        tiers[tier] = rows
        for r in rows:
            if r["seed"] == 42:
                for mode in ("CE", "KD"):
                    v = r[f"acc_{mode.lower()}"]
                    parity[f"{tier} {mode}"] = (v == ARCHIVE[tier][mode],
                                                v, ARCHIVE[tier][mode])
        print(f"\n[{tier} · S{WIDTH}] 三个随机流下的 KD−CE 配对（10000 张，翻牌逐图算）：")
        for r in rows:
            ok = "  ←第4课那一轮" if r["seed"] == 42 else ""
            print(f"  种子 {r['seed']:<5} CE {r['acc_ce']:.4f}（错 {r['wrong_ce']}） | "
                  f"KD {r['acc_kd']:.4f}（错 {r['wrong_kd']}） | gap {r['gap_pp']:+.2f}pp "
                  f"（翻坏 {r['flips']['broken']}/翻正 {r['flips']['fixed']}，"
                  f"p={r['p_sign']:.4g}）{ok}")

    bad = {k: v for k, v in parity.items() if not v[0]}
    print(f"\n[仪器对账] {'逐位复现 4/4 通过 → 换种子测到的确实是随机流，不是代码漂了' if not bad else 'FAIL：' + str(bad)}")

    verdicts = {}
    for tier, rows in tiers.items():
        gaps = [r["gap_imgs"] for r in rows]
        signs = sum(1 for g in gaps if g < 0)
        tb = sum(r["flips"]["broken"] for r in rows)
        tf = sum(r["flips"]["fixed"] for r in rows)
        verdicts[tier] = {
            "gaps_imgs": gaps, "gap_spread": spread(gaps),
            "same_dir_seeds": f"{max(signs, len(gaps) - signs)}/{len(gaps)}",
            "p_seeds": round(sign_test_p(min(signs, len(gaps) - signs), max(signs, len(gaps) - signs)), 4),
            "pooled_flips": {"broken": tb, "fixed": tf},
            "p_pooled": round(sign_test_p(tb, tf), 8),
            "seed_floor_imgs": {"CE": spread([r["wrong_ce"] for r in rows])["range"],
                                "KD": spread([r["wrong_kd"] for r in rows])["range"]}}
        v = verdicts[tier]
        floor = max(v["seed_floor_imgs"]["CE"], v["seed_floor_imgs"]["KD"])
        v["gap_over_floor"] = round(abs(v["gap_spread"]["mean"]) / floor, 2)
        v["survives_seed"] = bool(abs(v["gap_spread"]["mean"]) > floor)
        print(f"\n[判决 {tier}] gap 跨种子 {v['gaps_imgs']} 张 → 极差 {v['gap_spread']['range']} 张、"
              f"均值 {v['gap_spread']['mean']:+.1f} 张")
        print(f"  · 符号维度的复现：{v['same_dir_seeds']} 个种子同向，种子层二项 p={v['p_seeds']}"
              f"（n=3 最多只能到 0.25，**方向稳 ≠ 种子层显著**）")
        print(f"  · 三次复现的翻牌合并：坏 {v['pooled_flips']['broken']} / 正 {v['pooled_flips']['fixed']}，"
              f"p={v['p_pooled']:.4g}（可合并的前提：三组是同一份数据的不同随机流，交换性成立）")
        print(f"  · **种子底：同一目标函数只换随机流，错图数极差 CE {v['seed_floor_imgs']['CE']} 张 / "
              f"KD {v['seed_floor_imgs']['KD']} 张**——以后任何单臂读数小于这个数，在这台机器上都不算疗效")
        print(f"  · 疗效对底：|均值 gap| {abs(v['gap_spread']['mean']):.1f} 张 ÷ 种子底 {floor} 张 = "
              f"{v['gap_over_floor']}x → "
              + ("**扛住换随机流**" if v["survives_seed"] else "**没扛住：幅度与换一次权重的抖动同量级，不算疗效**"))

    print("\n[自我审计① 重跑不等于复现] 第4课固定种子，精度/翻牌/KB 跨运行逐位相同，"
          "所以「再跑一遍 step4」只能量到墙钟漂移、量不到种子方差；"
          "这一课必须改随机流才是还那笔债——别把「跑过两次」当成「复现过」。")
    print(f"[自我审计② 两层误差别混] 图配对检验（p 很小）说的是「同一副权重下 KD 与 CE 谁判对更多」，"
          f"种子层（n={len(SEEDS)}）说的是「换一副权重还成不成立」。前者管系统误差、后者管重训方差，"
          "两者不能互相顶替，也不能乘在一起吹成「p 更小」。")
    print(f"[自我审计③ 仍未做] α/T 扫参、宽度 S64 的种子复现、"
          f"每种子只训一次（种子底本身没有二次底，n={len(SEEDS)} 个点的极差是欠估的）。")
    pool = "、".join(f"{t} {v['pooled_flips']['broken']}+{v['pooled_flips']['fixed']}"
                     for t, v in verdicts.items())
    print(f"[自我审计④ 合并 p 偏乐观] 三个种子吃的是同一批 10000 张测试图：同一张图可以在三种子里都翻坏，"
          f"所以合并起来的翻牌数（{pool}）不是独立事件，p_pooled 只是「比单种子更稳的方向感」，"
          f"不能当第二个独立复现。真正干净的复现统计只有上面那行「{len(SEEDS)} 个种子同向」。")

    report = {"seeds": list(SEEDS), "width": WIDTH, "test_n": int(len(yte)),
              "train_fit_n": cut, "parity_seed42_vs_archive":
                  {k: {"match": v[0], "this_run": v[1], "archive": v[2]} for k, v in parity.items()},
              "arms": {f"{t} 种子{s}": r for t, rows in tiers.items() for s, r in zip(SEEDS, rows)},
              "verdicts": verdicts,
              "limitations": ["每种子只训一次：种子底由 3 个点的极差估，欠估",
                              "只复现了 S128；S64 两档判决未换种子",
                              "α/T 未扫参（第4课同款边界，本轮没动）",
                              "p_pooled 偏乐观：三个种子共用同一批 10000 张测试图，翻牌事件跨种子不独立",
                              "本轮不测墙钟：精度是确定性读数，墙钟要另开交错配对流程"]}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（12 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    main()
