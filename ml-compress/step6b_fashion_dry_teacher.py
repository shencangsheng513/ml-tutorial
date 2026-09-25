# -*- coding: utf-8 -*-
"""第6课的续账：给 Fashion 造一副**更干**的老师，让"低数据那个符号跟软标签干湿走"在第二个任务里拿到判别量。

第6课换任务那一发的收获是**否定的**：满数据档之害跨任务复现了，低数据档的翻号没有——
Fashion 低数据档的配对 T 主效应是 −79.1 张（8 条随机流、95% CI [−111.4, −46.8]、8/8 同号，
见 compress_report_v4e_fashion_low_t1t6.json），而 MNIST 低数据档同两格是 **+71.7 张**。
符号在换任务时翻回去了，这一刀本身没判什么：它同时换掉了任务、老师的精度**和老师的干湿**
（Fashion 满档老师在拟合集上 T=1 塌缩率 42.5%，MNIST 那副 93.3% ⇒ Fashion 的老师明显更湿）。

板块里"幅度跟干湿"那条排序（越干越伤）目前站在 5 个点上，全部来自 MNIST 那四副老师；
第5课已经点破它的软肋：**干湿、剪枝剂量、老师精度同向变化，分不开**。要把它从"支持"推到"裁决"，
缺的正是一个**同一个任务内**的干湿对照——所以这一发不换任务，只换老师的干湿：

  H_干湿（要拿判别量的那一侧）：把 Fashion 老师训得**更干**（拟死同批图 ⇒ 软标签更 one-hot），
        低数据档的 T 主效应应当跟着从负往正走；翻到正号 = 干湿解释在第二个任务里也成立。
  H_任务（这一发的对手）：符号是这个任务的特产（类间语义混、老师天花板低），
        换干湿不动它，改训多少 epoch 也还是负号。

三件仪器，训练前立好：
  ① **干湿必须真的动了**（判别量的前置闸）：新老师在**同一批拟合集**上的 T=1 塌缩率必须比
     A 那副高 ≥10pp（A 的读数从 v6 档案现读，不抄字面量）。没动 ⇒ 这一发**没有判别量**，
     直接退出，不许把"读不出差别"读成"两说都无所谓"。
  ② **CE 锚逐位对账**：CE 臂不吃软标签 ⇒ 同种子同数据同宽度下新老师的 CE 错图数必须与 v6 档案
     低数据档那三条逐一相同。对不上 ⇒ 换老师的同时换了仪器，两副老师的差里没有干湿的解释权。
  ③ **本档自己的底**（第4课g 规矩）：8 条随机流下 CE 臂 + 四格 KD 各自跨种子极差取 max，不借任何账。

判据（训练前写死，此后不许改口）：读数单位是**逐种子配对的 T 主效应**
（同一副随机流内 (gap@T=6 − gap@T=1) 对 α 取平均），不是跨重训的极差——第6课已经演示过拿单臂底
去量配对量会把真效应读成"测不动"（−79.1 张 = 该档底 0.75x 却 8/8 同号）。
  · CI 下界 > 0 ⇒ 干湿说拿到任务内翻转（H_任务这一发被削弱）；
  · CI 上界 < 0 ⇒ 换干湿不换符号 ⇒ H_任务占优，"幅度跟干湿"要么另有机制要么只在 MNIST 那族刀下成立；
  · CI 跨 0 ⇒ 这一发没判决，别替它编方向。
零 API 费，纯 CPU。跑法（约 10 分钟，其中一副新老师要训一次并缓存）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step6b_fashion_dry_teacher.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import step6_fashion_replication as s6
from step1_quantize import MODELS, build_cnn, predict
from step4_distill import VAL_N, flows, logits_of, soft_label_audit, summarize
from step5_second_teacher import train_recipe
from step6_fashion_replication import LOW, N_LOW, TEACHER_ACC_GATE, TEACHER_PT, load_fashion, run_tier

HERE = Path(__file__).resolve().parent
DRY_EPOCHS = 40                                   # A 那副是 20：只把轮数翻倍，刀不动
DRY_PT = MODELS / "fashion_cnn_ep40.pt"
OUT = HERE / "compress_report_v6b_fashion_dry.json"
MARGIN_PP = 10.0                                  # 干湿闸：塌缩率至少差这么多才算换了变量
SEEDS8 = flows(8)             # 名单只有一个出处：登记处缺这一档会直接 SystemExit，不会静默少跑一条流


def get_dry_teacher(xt, yt):
    """同一副 CNN、同一份基线配方，只把轮数从 20 抬到 40 ⇒ 更拟死 ⇒ 软标签应当更干。"""
    if DRY_PT.exists():
        net = build_cnn()
        net.load_state_dict(torch.load(DRY_PT, weights_only=True))
        return net.eval(), "复用缓存权重（本轮没重训）"
    net = train_recipe(DRY_EPOCHS, xt, yt).eval()
    torch.save(net.state_dict(), DRY_PT)
    return net, f"本轮按基线配方训 {DRY_EPOCHS} epoch（A 是 {s6.TEACHER_EPOCHS}）并存档"


def main(red_check: bool = False) -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉那本账；要重跑先改名或删")
    if not TEACHER_PT.exists():
        raise SystemExit(f"缺 {TEACHER_PT.name}——A 那副 Fashion 老师只能复用，先跑 step6")
    v6 = json.loads((HERE / "compress_report_v6_fashion.json").read_text(encoding="utf-8"))
    if v6["dataset"] != "FashionMNIST":
        raise SystemExit("对照只能读 Fashion 那本账")
    a_collapse = v6["teacher_dryness_by_T"]["1.0"]["raw_collapse_pct"]          # 干湿闸的基准
    a_low = v6["per_tier"][s6.FULL if red_check else LOW]
    if red_check:
        print("[验红] --red-check：CE 对账故意拿另一档（满数据档）的锚当基准，闸② 必须红着退出；"
              "跑通即说明这条闸不是恒绿（本课的读数不入库，退出码非 0）")

    xt_all, yt_all, xte, yte = load_fashion()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    dry, src = get_dry_teacher(xt_all, yt_all)
    acc_dry = round(float((predict(dry, xte, False) == yte).mean()), 4)
    print(f"[老师] dry {DRY_EPOCHS} epoch | {src} | 测试精度 {acc_dry}（A 那副 {v6['teacher']['test_acc']}）")
    if acc_dry < TEACHER_ACC_GATE:
        raise SystemExit(f"新老师只有 {acc_dry} < {TEACHER_ACC_GATE}：判出来的『符号』要先归给不合格的老师")

    dry_collapse = soft_label_audit(logits_of(dry, xt_all[:s6.N_FULL]), t=1.0)["原始"]["pct_top1_gt_0.99"]
    moved = round(dry_collapse - a_collapse, 1)
    print(f"[闸① 干湿必须真的动] 拟合集 {s6.N_FULL} 张 T=1 塌缩率：A {a_collapse}% → dry {dry_collapse}%"
          f"（差 {moved:+.1f}pp，闸门 ≥ +{MARGIN_PP:g}pp）")
    if moved < MARGIN_PP:
        raise SystemExit(f"只动了 {moved:+.1f}pp < {MARGIN_PP:g}pp ⇒ 这一发没有判别量：两副老师的软标签"
                         "实质同干，读不出差别不等于两说都无所谓。先换剂量再来")

    s6.SEEDS = SEEDS8        # run_tier 的循环跟着它走：本课把流数抬到 8（判据要 CI）
    print(f"\n[开跑] {LOW} · dry 老师 | {len(s6.ALPHAS) * len(s6.TEMPS)} 格 × {len(SEEDS8)} 条流 + 每流一副 CE")
    res = run_tier(LOW, N_LOW, dry, xt_all, yt_all, xv, yv, xte, yte)

    k0 = s6.key(s6.ALPHAS[0], s6.TEMPS[0])
    ce_this = {r["seed"]: r["wrong_ce"] for r in res["raw_rows"][k0]}
    parity = [{"seed": r["seed"], "this_run_ce_wrong": ce_this[r["seed"]], "v6_archive_ce_wrong": r["wrong_ce"],
               "match": ce_this[r["seed"]] == r["wrong_ce"]}
              for r in a_low["raw_rows"][k0] if r["seed"] in ce_this]
    bad = [p for p in parity if not p["match"]]
    print(f"\n[闸② CE 锚逐位对账（CE 不吃软标签 ⇒ 必须与 A 那发同数）] "
          + (f"{len(parity)}/{len(parity)} 通过" if parity and not bad else f"FAIL/空表：{bad or parity}"))
    if not parity or bad:
        raise SystemExit("CE 锚对不上（或一条都没对上）⇒ 换老师的同时换了仪器，两副老师的差里没有干湿的解释权")
    if red_check:
        raise SystemExit("验红失败：故意拿另一档的 CE 锚当基准居然全对上了 ⇒ 闸② 是恒绿的，"
                         "它拦不住'对错了账'，下面所有读数不可信")

    hi, lo = max(s6.TEMPS), min(s6.TEMPS)
    by = {(k, r["seed"]): r["gap_imgs"] for k in (s6.key(a, t) for a in s6.ALPHAS for t in s6.TEMPS)
          for r in res["raw_rows"][k]}
    eff = [round(sum(by[(s6.key(a, hi), s)] - by[(s6.key(a, lo), s)] for a in s6.ALPHAS) / len(s6.ALPHAS), 1)
           for s in SEEDS8]
    st = summarize(eff, f"dry 老师 | {LOW} 配对 T 主效应")
    floor = res["own_floor_imgs"]
    print(f"\n[配对读数] 逐种子（T={hi:g}−T={lo:g} 对 α 平均）：{eff}")
    print(f"  均值 {st['mean_gap_imgs']:+.1f} 张，95% CI [{st['ci95_imgs'][0]:+.1f}, {st['ci95_imgs'][1]:+.1f}]，"
          f"同号 {st['same_sign_seeds']}（二项 p={st['p_seeds_binomial']}）| 单臂底 {floor} 张"
          f"（{st['mean_gap_imgs'] / floor:+.2f}x，只当仪器刻度不当判决）")
    a_eff = a_low["T_main_effect_imgs"]
    if st["ci95_imgs"][0] > 0:
        verdict = "干湿说拿到任务内翻转：同一任务同一批图，只把老师训干 ⇒ 低数据 T 主效应由负转正"
    elif st["ci95_imgs"][1] < 0:
        verdict = "换干湿不换符号：dry 老师仍为负 ⇒ H_任务占优，『幅度跟干湿』在第二个任务里没拿到因果"
    else:
        verdict = "CI 跨 0：这一发没判决，别替它编方向"
    print(f"[判决] {verdict}")
    print(f"[对照] A（塌缩 {a_collapse}%）低数据 T 主效应 {a_eff:+.1f} 张（v6 那发 3 条流）；"
          f"dry（塌缩 {dry_collapse}%）{st['mean_gap_imgs']:+.1f} 张（本课 {len(SEEDS8)} 条流）"
          f"→ 干湿差 {moved:+.1f}pp 换到效应差 {st['mean_gap_imgs'] - a_eff:+.1f} 张")

    report = {"dataset": "FashionMNIST", "tier": LOW, "n_train_imgs": N_LOW,
              "teacher": {"epochs": DRY_EPOCHS, "weight": DRY_PT.name, "test_acc": acc_dry, "created_by": src},
              "dryness_gate": {"a_collapse_pct": a_collapse, "dry_collapse_pct": dry_collapse,
                               "moved_pp": moved, "required_pp": MARGIN_PP, "pass": True},
              "ce_anchor_parity": parity, "seeds": list(SEEDS8),
              "alphas": list(s6.ALPHAS), "temps": list(s6.TEMPS),
              "paired_t_effect_per_seed_imgs": eff, "paired_t_effect_stats": st,
              "single_arm_own_floor_imgs": floor, "teacher_a_low_tier_ref_from_v6_imgs": a_eff,
              "prediction_ex_ante": {"H_dryness": "训干 ⇒ 低数据 T 主效应往正走（任务内翻号）",
                                     "H_task": "换干湿不换符号（仍负）",
                                     "yardstick": "逐种子配对 + 95% CI；单臂底只当仪器刻度"},
              "verdict": verdict, "per_tier_raw": res,
              "limitations": ["干湿由 epoch 轮数决定，同时改变了老师精度与拟合度：拆开的是轮数，"
                              "拆不掉『老师强弱』这一维（第5课 D 同一告警）",
                              "只一个任务、只低数据档、只一副新老师：这是干湿说的第一次任务内判别，不是收敛",
                              f"温度列只有 T∈{s6.TEMPS} 两端，中间形状未测",
                              f"{len(SEEDS8)} 条流同架构同数据同老师：把读数做精，不是独立复证"]}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（{len(SEEDS8) * (len(s6.ALPHAS) * len(s6.TEMPS) + 1)} 次学生训练 "
          f"+ 1 次老师训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fashion 第二副老师（更干）：低数据档 T 主效应的任务内干湿判别")
    ap.add_argument("--red-check", action="store_true",
                    help="验红：CE 对账故意拿另一档的锚当基准，闸② 必须红着退出（不写档）")
    args = ap.parse_args()
    main(args.red_check)
