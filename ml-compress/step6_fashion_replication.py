# -*- coding: utf-8 -*-
"""第6课 换数据集：把「T 的符号跟数据量走」交给第二个任务去判（`step6_fashion_replication.py`）

前五课的复证全在**同一个任务**里发生：换随机流（4b/4e）、换学生宽度（4c --width）、
换数据档（4c --tier、4g）、换老师（第5课四副）。它们守得住的边界是"这条结论不随
学生/样本流/知识源变"，但没有任何一发问过它跟 MNIST 本身有没有关系——而本板块最响的那条
判决恰好是最容易被任务特性解释的那条：满数据档加热有害（−95.7 张 = 该档底 2.3x）、
低数据档加热有益（+71.7 张 = 1.9x），**一个符号随数据量翻转**。MNIST 上"背得出"和"猜不出"
分得很开，所以这条翻转完全可能是 0.98 精度那个任务的特产。

选 FashionMNIST：同尺寸（28×28 灰度、10 类、60k/10k）、同架构能跑、类别语义难一档
（T-shirt/top 与 Shirt、Sandal 与 Sneaker 本来就混），于是**不改一行训练代码**就把"任务"换掉了。
凡是要改架构才能做的对比，改掉的就不止"任务"一个变量。

要判的两说（看到 Fashion 老师的零训练体检之后、任何学生训练之前写死，此后不许改口）：
  H_任务通用：T 主效应的符号只跟学生能看到多少张有关（满负、低正），换数据集不改。
  H_MNIST 特产：Fashion 上老师背不出（塌缩率低、精度低），"加热"就没有可倒出来的干东西，
              两档可能同为负、或同为不显著。

三件仪器，训练前立好：
  ① **MNIST 逐位对账闸**（`--gate-mnist`）：本脚本自带一份数据装载，所以先证明
     "装载 + 调用训练器"与第4课c 完全一致——重跑 α=0.7/T=3/种子42/满数据档那一格，
     与 `compress_report_v4c.json` 四项读数（错图数/翻坏/翻正/CE 错图数）逐位比。
     不过 ⇒ 换数据同时换了仪器，Fashion 读数无从对照，**直接退出**。
  ② **老师体检闸**：Fashion 老师测试精度低于 0.80 就中止——一副不合格的老师做软标签，
     判出来的"KD 没用"要先归给老师。
  ③ **本档自己的种子底**（第4课g 立的规矩）：CE 臂 + 四个 KD 格各自的跨种子极差取 max，
     不借 MNIST 那两本账里的 41/37 张；借来的底偏小，会把"测不动"读成"有方向"。

产物 `compress_report_v6_fashion.json`，**已入库就拒绝重跑**。

跑法（约 25~40 分钟，纯 CPU，零 API 费；首次会训一次 Fashion 老师并缓存）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step6_fashion_replication.py
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step6_fashion_replication.py --gate-mnist
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import FashionMNIST

from step1_quantize import DATA, MODELS, build_cnn, predict
from step4_distill import TEACHER_ARCHIVE_ACC, VAL_N, logits_of, sign_test_p, soft_label_audit, train_student
from step4c_alpha_T_sweep import ANCHOR_T, ALPHA_ANCHOR, SEEDS, TIER, key, out_path, v4b_path
from step5_second_teacher import train_recipe
from train_baseline import N_TR, load_split

HERE = Path(__file__).resolve().parent
OUT = HERE / "compress_report_v6_fashion.json"
TEACHER_PT = MODELS / "fashion_cnn_fp32.pt"
TEACHER_EPOCHS = 20                      # 与 MNIST 基线逐行同款（train_recipe 就是那份配方）
FULL, LOW = "满数据8500", "低数据2000"
TIERS = (FULL, LOW)
N_FULL, N_LOW = 8500, 2000
ALPHAS = (0.7, 1.0)
TEMPS = (1.0, 6.0)                       # T 主效应的两个端点：翻号读数就长在这一对上
TEACHER_ACC_GATE = 0.80


def load_fashion():
    """逐行照搬 train_baseline.load_split：同 rng(42)、同抽 10000 张、测试集全量，只换数据集类。
    download=False 是故意的——数据已落盘并核过 MD5，留着 True 等于允许一次无声的网络替换。"""
    tr = FashionMNIST(str(DATA), train=True, download=False)
    te = FashionMNIST(str(DATA), train=False, download=False)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(tr), N_TR, replace=False)
    xtr = tr.data[idx].numpy().astype(np.float32) / 255.0
    ytr = np.array(tr.targets)[idx]
    xte = te.data.numpy().astype(np.float32) / 255.0
    yte = np.array(te.targets)
    return (torch.tensor(xtr).unsqueeze(1), ytr, torch.tensor(xte).unsqueeze(1), yte)


def gate_mnist_parity() -> dict:
    """① 对账闸：本脚本的装载路径 + 训练器调用必须逐位复现第4课c 的锚点格。"""
    v4c, v4b = out_path(128, TIER), v4b_path(128, TIER)
    if not v4c.exists() or not v4b.exists():
        raise SystemExit(f"缺 {v4c.name} / {v4b.name}——对账基准只能从档案读")
    cell = json.loads(v4c.read_text(encoding="utf-8"))["grid"][key(ALPHA_ANCHOR, ANCHOR_T)]["per_seed"][0]
    arm = json.loads(v4b.read_text(encoding="utf-8"))["arms"][f"{TIER} 种子{SEEDS[0]}"]

    xt, yt, xte, yte = load_split()
    cut = len(xt) - VAL_N
    xv, yv = xt[cut:], torch.tensor(yt[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    xd, yd = xt[:N_FULL], torch.tensor(yt[:N_FULL])
    soft = logits_of(teacher, xd)
    ce, _ = train_student(128, xd, yd, soft, "CE", xv, yv, seed=SEEDS[0])
    pe = predict(ce, xte, False)
    m, _ = train_student(128, xd, yd, soft, "KD", xv, yv, seed=SEEDS[0],
                         alpha=ALPHA_ANCHOR, t=ANCHOR_T)
    p = predict(m, xte, False)
    got = {"wrong_kd": int((p != yte).sum()), "wrong_ce": int((pe != yte).sum()),
           "broken": int(((pe == yte) & (p != yte)).sum()),
           "fixed": int(((pe != yte) & (p == yte)).sum())}
    ref = {k: cell[k] for k in ("wrong_kd", "wrong_ce", "broken", "fixed")}
    ok = all(got[k] == ref[k] for k in ref) and got["wrong_ce"] == int(arm["wrong_ce"])
    print(f"[闸① MNIST 逐位对账 {key(ALPHA_ANCHOR, ANCHOR_T)} 种子{SEEDS[0]}] 本脚本 {got} "
          f"vs v4c 档案 {ref}（CE 另核 v4b {arm['wrong_ce']}）→ "
          + ("通过：换数据集只换了数据，没换仪器" if ok else "不通过"))
    if not ok:
        raise SystemExit("对账不过 ⇒ 本脚本与第4课c 不是同一台仪器，Fashion 读数无从对照，先修接线")
    return {"cell": key(ALPHA_ANCHOR, ANCHOR_T), "seed": SEEDS[0], "this_script": got,
            "archive_v4c": ref, "ce_wrong_v4b_archive": int(arm["wrong_ce"]), "match": True}


def get_teacher(force: bool = False):
    """Fashion 老师：同一副 CNN、同一份配方、同一批 10000 张图，只换任务；训一次就缓存。"""
    xt, yt, xte, yte = load_fashion()
    if TEACHER_PT.exists() and not force:
        net = build_cnn()
        net.load_state_dict(torch.load(TEACHER_PT, weights_only=True))
        net.eval()
        src = "复用缓存权重（本轮没重训）"
    else:
        t0 = time.time()
        net = train_recipe(TEACHER_EPOCHS, xt, yt).eval()
        torch.save(net.state_dict(), TEACHER_PT)
        src = f"本轮按基线配方从头训 {TEACHER_EPOCHS} epoch 并存档（{time.time() - t0:.0f}s）"
    acc = float((predict(net, xte, False) == yte).mean())
    return net, {"test_acc": round(acc, 4), "created_by": src, "epochs": TEACHER_EPOCHS,
                 "weight": TEACHER_PT.name}


def own_floor(spans: dict[str, list[int]]) -> dict:
    """③ 本档自己的底 = 各目标函数跨种子错图数极差的最大值（第4课g 的规矩）。"""
    per = {k: int(max(v) - min(v)) for k, v in spans.items()}
    return {"own_floor_imgs": max(per.values()), "per_target_span_imgs": per}


def mnist_refs() -> dict:
    """对照数字**从两本 v4c 档案现读**，一个都不抄。抄进来过一次就出过事：
    −95.7/+71.7 那两个数一个是别的温度格、一个是别的组合，92.5% 是别的温度下的塌缩率，
    拿假基准去比真复证比不比对还糟。同类可比也要自己算：档案里那条 T 主效应是对
    α∈{0.3,0.7,1.0} 取的均值，本课的阶梯只有 α∈{0.7,1.0}，所以按本课的 α 集合重算一份
    （两个数都进报告，差多少也让人看见）。"""
    refs = {}
    for tier in TIERS:
        p = out_path(128, tier)
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["tier"] != tier or int(d["width"]) != 128:
            raise SystemExit(f"{p.name} 是 {d['tier']}/S{d['width']} 的表，不是 {tier}/S128——对照读错了账")
        g = d["grid"]
        missing = [key(a, t) for a in ALPHAS for t in TEMPS if key(a, t) not in g]
        if missing:
            raise SystemExit(f"{p.name} 里没有 {missing} 这几格，无法算同类可比的 T 主效应")
        refs[tier] = {
            "archive": p.name,
            "T_main_effect_like_for_like_imgs": round(
                sum(g[key(a, TEMPS[-1])]["mean_gap_imgs"] - g[key(a, TEMPS[0])]["mean_gap_imgs"]
                    for a in ALPHAS) / len(ALPHAS), 1),
            "T_main_effect_as_archived_imgs": d["verdicts"]["T_main_effect"]["T_max_minus_T_min_pooled_over_alpha"],
            "seed_floor_imgs": int(d["seed_floor_imgs_from_v4b"]),
            "collapse_pct_after_divT": {
                f"T{t:g}": d["soft_label_audit_by_T"][key(1.0, t)]["除以T"]["pct_top1_gt_0.99"] for t in TEMPS},
            "mean_p2_after_divT": {f"T{t:g}": d["soft_label_audit_by_T"][key(1.0, t)]["除以T"]["mean_p2"] for t in TEMPS},
        }
    return refs


def run_tier(tier: str, n: int, teacher, xt_all, yt_all, xv, yv, xte, yte) -> dict:
    xd, yd = xt_all[:n], torch.tensor(yt_all[:n])
    soft = logits_of(teacher, xd)
    specs = [(key(a, t), a, t) for a in ALPHAS for t in TEMPS]
    cells = {k: [] for k, _, _ in specs}
    ce_wrong = []
    print(f"\n[{tier} · Fashion] 训练 {n} 张 | 测试 {len(yte)} | {len(specs)} 格 × {len(SEEDS)} 种子 + 每种子一副 CE 锚")
    for seed in SEEDS:
        ce, _ = train_student(128, xd, yd, soft, "CE", xv, yv, seed=seed)
        pe = predict(ce, xte, False)
        cw = int((pe != yte).sum())
        ce_wrong.append(cw)
        line = [f"  种子 {seed:<5} CE 错 {cw:>4}"]
        for k, a, t in specs:
            m, _ = train_student(128, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=a, t=t)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            broken = int(((pe == yte) & (p != yte)).sum())
            fixed = int(((pe != yte) & (p == yte)).sum())
            cells[k].append({"seed": seed, "alpha": a, "T": t, "wrong_kd": wrong, "wrong_ce": cw,
                             "gap_imgs": cw - wrong, "broken": broken, "fixed": fixed,
                             "p_sign": round(sign_test_p(broken, fixed), 6)})
            line.append(f"{k}:{cw - wrong:+d}")
        print("  ".join(line))

    fl = own_floor({"CE": ce_wrong, **{k: [r["wrong_kd"] for r in v] for k, v in cells.items()}})
    floor = fl["own_floor_imgs"]
    grid = {}
    for k, v in cells.items():
        gaps = [r["gap_imgs"] for r in v]
        tb, tf = sum(r["broken"] for r in v), sum(r["fixed"] for r in v)
        grid[k] = {"gaps_imgs": gaps, "mean_gap_imgs": round(sum(gaps) / len(gaps), 1),
                   "pooled_flips": {"broken": tb, "fixed": tf},
                   "p_pooled": round(sign_test_p(tb, tf), 6),
                   "vs_own_floor": round(abs(sum(gaps) / len(gaps)) / floor, 2)}
        g = grid[k]
        print(f"  {k:<10} 均值 {g['mean_gap_imgs']:+6.1f} 张（本档底 {floor} 的 {g['vs_own_floor']}x）"
              f" 合并翻坏 {g['pooled_flips']['broken']}/翻正 {g['pooled_flips']['fixed']} p={g['p_pooled']:.4g}")
    mean_at = lambda a, t: grid[key(a, t)]["mean_gap_imgs"]          # noqa: E731
    main_eff = round(sum(mean_at(a, TEMPS[-1]) - mean_at(a, TEMPS[0]) for a in ALPHAS) / len(ALPHAS), 1)
    has = bool(abs(main_eff) > floor)
    print(f"  → 本档 T 主效应 {main_eff:+.1f} 张 = 自己的底（{floor} 张）的 {abs(main_eff) / floor:.2f}x → "
          + (f"**有方向：{'加热有益' if main_eff > 0 else '加热有害'}**" if has else "在底以内 = 测不动"))
    return {"T_main_effect_imgs": main_eff, **fl, "has_direction": has,
            "in_units_of_own_floor": round(abs(main_eff) / floor, 2),
            "sign": "正（加热有益）" if main_eff > 0 else "负（加热有害）",
            "grid": grid, "raw_rows": cells}


def main(force_teacher: bool = False) -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉那本判决账；要重跑先改名或删，别无声换账")
    parity = gate_mnist_parity()
    refs = mnist_refs()          # 对照全从 v4c 档案现读：抄进代码的基准会在档案改口后对上老账

    xt_all, yt_all, xte, yte = load_fashion()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher, tinfo = get_teacher(force_teacher)
    print(f"\n[闸② 老师体检] Fashion 老师测试精度 {tinfo['test_acc']}（MNIST 那副 {TEACHER_ARCHIVE_ACC}）| {tinfo['created_by']}")
    if tinfo["test_acc"] < TEACHER_ACC_GATE:
        raise SystemExit(f"老师只有 {tinfo['test_acc']}：判出来的『KD 没用』要先归给老师，本课作废")

    dry = {}
    print("[零训练先验] Fashion 老师在拟合集（它背过的 8500 张）上的软标签干湿，逐温度：")
    soft_full = logits_of(teacher, xt_all[:N_FULL])
    for t in TEMPS:
        a = soft_label_audit(soft_full, t=t)
        o, s = a["原始"], a["除以T"]
        dry[str(t)] = {"raw_top1": o["mean_top1"], "raw_collapse_pct": o["pct_top1_gt_0.99"],
                       "raw_p2": o["mean_p2"], "divT_top1": s["mean_top1"],
                       "divT_collapse_pct": s["pct_top1_gt_0.99"], "divT_p2": s["mean_p2"]}
        print(f"  T={t:<4} 原始 top1 {o['mean_top1']:.4f}（塌缩 {o['pct_top1_gt_0.99']}%，p2 {o['mean_p2']:.4f}）"
              f" → /T 后 top1 {s['mean_top1']:.4f}（塌缩 {s['pct_top1_gt_0.99']}%，p2 {s['mean_p2']:.4f}）")
    m = refs[FULL]
    print(f"  对照（MNIST 老师 A，{m['archive']} 现读，同为拟合集 8500 张所以可比）："
          + "、".join(f"T={t:g} 塌缩 {m['collapse_pct_after_divT'][f'T{t:g}']}%"
                      f"（p2 {m['mean_p2_after_divT'][f'T{t:g}']:.4f}）" for t in TEMPS))

    print(f"\n[先立判据（此后不许改口）] 每档 T 主效应 = mean over α 的（gap@T={TEMPS[-1]:g} − gap@T={TEMPS[0]:g}），"
          f"与**该档自己的底**比：|主效应| > 底 才算有方向；两档方向相反 → 翻号跨数据集复现；"
          f"两档同负 → 翻号是 MNIST 特产；任一档落在底内 → 那一档测不动，不许替它编方向。")

    res = {FULL: run_tier(FULL, N_FULL, teacher, xt_all, yt_all, xv, yv, xte, yte),
           LOW: run_tier(LOW, N_LOW, teacher, xt_all, yt_all, xv, yv, xte, yte)}
    hi, lo = res[FULL], res[LOW]
    both = hi["has_direction"] and lo["has_direction"]
    if both and (hi["T_main_effect_imgs"] > 0) != (lo["T_main_effect_imgs"] > 0):
        verdict = "翻号跨数据集复现 → 支 H_任务通用"
    elif both and hi["T_main_effect_imgs"] < 0 and lo["T_main_effect_imgs"] < 0:
        verdict = "两档同为负 → 翻号是 MNIST 特产（支 H_MNIST 特产）"
    elif both:
        verdict = "两档同正 → MNIST 那个满数据负号没复现"
    else:
        weak = [t for t in TIERS if not res[t]["has_direction"]]
        verdict = f"测不动：{'、'.join(weak)} 的主效应落在本档自己的底以内"
    print(f"\n[判决] {verdict}")
    print("[判决的分量] 同两档并排（Fashion 用本档自己的底，MNIST 那本账用的是它借来的底，两个数不能混读）：")
    for tier in TIERS:
        r, rr = res[tier], refs[tier]
        print(f"  {tier:<10} Fashion T 主效应 {r['T_main_effect_imgs']:+7.1f} 张 / 自己的底 {r['own_floor_imgs']:3d} 张"
              f"（{r['in_units_of_own_floor']}x） | MNIST 同类可比 {rr['T_main_effect_like_for_like_imgs']:+7.1f} 张"
              f" / 借来的底 {rr['seed_floor_imgs']:3d} 张（档案那条对三档 α 取平均是 "
              f"{rr['T_main_effect_as_archived_imgs']:+.1f} 张）")

    report = {"dataset": "FashionMNIST", "n_pool": int(len(xt_all)), "test_n": int(len(yte)),
              "teacher": tinfo, "teacher_dryness_by_T": dry,
              "mnist_refs_read_from_archive": refs,
              "gate_mnist_parity": parity, "width": 128, "alphas": list(ALPHAS), "temps": list(TEMPS),
              "seeds": list(SEEDS),
              "floor_rule": "第4课g：每档用自己的底（CE 臂 + 各 KD 格跨种子极差取 max），不借 MNIST 账里的 41/37 张",
              "per_tier": res, "verdict": verdict,
              "ref_note": f"对照里的 MNIST 主效应按本课的 α∈{ALPHAS} 重算（同类可比）；"
                          f"档案原那条对三档 α 取平均，两个数都留在表里",
              "limitations": [
                  "每格 3 条随机流、每流训一次：主效应只有本档底这一个误差棒代理，没扩到 n=8",
                  "只 S128、只两个数据档（中段三档第4课g 已量出在底内，这里不再花那份钱）",
                  "换数据集同时换了老师的干湿与任务难度，『符号跟数据量』与『符号跟干湿』在这一发里"
                  "分不开——第5课那把刀在 Fashion 上没有对应物",
                  "只一档老师（无跨老师复证）：本课的独立性来自**换任务**，不来自换知识源",
                  "Fashion 上没跑 α→0 阶梯，污染说是否跨任务未测"]}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（{len(SEEDS) * (len(ALPHAS) * len(TEMPS) + 1) * len(TIERS)} 次学生训练 "
          f"+ 1 次老师训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="第6课：FashionMNIST 换数据集复证 T 轴翻号")
    ap.add_argument("--gate-mnist", action="store_true", help="只跑 MNIST 逐位对账闸（约 1 分钟）")
    ap.add_argument("--retrain-teacher", action="store_true", help="强制重训 Fashion 老师")
    args = ap.parse_args()
    if args.gate_mnist:
        gate_mnist_parity()
    else:
        main(args.retrain_teacher)
