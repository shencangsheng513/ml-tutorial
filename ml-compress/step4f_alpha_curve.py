# -*- coding: utf-8 -*-
"""第4课c 欠的最后一发：把 α 从 0.3 往下探到 0，看伤害是"剂量"还是"污染"。

第4课c 扫了 α∈{0.3, 0.7, 1.0}（T=3 那一行实测 −72.0 / −75.0 / −76.7 张），判词是
"α 轴平、斜率 −4.7 张 = 底的 0.11x"。但**平在 0.3~1.0 之间，不等于平在 0~0.3 之间**——
稀释说（H1：伤害 ∝ 给了软标签多少权重）没有被那场扫参杀掉，它只是退到了最后一块地方：

  H1 的残城：老师项是**按份额**掺进来的毒，α=0.05 只该坏 5 张（−75 × 0.05/0.7），
            α=0.3 才坏 32 张。那么从 0 走到 0.3 这一段必须是**陡的**。
  污染说（H1 已死）：softmax 目标分布一旦参与，学生学到的就是那副分布的**形状**，
            份额只决定它多快生效。α=0.05 就该看到接近 −72 的伤害，而 0.3→1.0 加不了多少。

两说的分歧点只有一个数字：**α=0.05 那一格离 0 近，还是离 −72 近。**
（H1 预测 −5.4 张，污染说预测 −70 张开外，相差 13 倍——这是本板块最便宜的一次判决。）

两件免费仪器，训练前立好：
  · **α=0 恒等闸**：loss 在 α=0 时按定义就是纯 CE。它必须与 CE 臂**逐位相同**
    （错图数一样、与 CE 的不同预测数=0）。这不是一个臂，是 α 这根线**接没接对**的证明——
    接错了下面五格全是废数。
  · **α=0.3 跨脚本闸**：第4课c 已经跑过这一格并把逐种子读数存进 v4c 档案，
    本课重跑一遍，与档案逐位对账。对不上说明新增的这条路径动了 train_student 的调用方式。

判据（训练前写死，此后不许改口）：小 α 那一头每格的均值 gap 都低于种子底（41 张），
所以**不能用"过不过底"当判决**——那等于宣布这一发没判决。用逐图配对：同一副权重、同一批数据，
两臂只有目标函数不同，翻坏/翻正的不对称比"重训抖动"敏感得多（第4课的老道理）。
  · 若 α≤0.1 两格配对检验不出方向（p>0.05 且坏≈正）而 α=0.2/0.3 出方向 → 有阈值，H1 半活；
  · 若 α=0.05 就同号且合并 p 显著 → 稀释说最后一块地没了，伤害是**污染型**。
零 API 费，纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4f_alpha_curve.py
  # 换档（档案驱动：底与 α=0.3 基准都从那一档自己的账里读）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4f_alpha_curve.py --tier 中数据4500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import DATA_TIERS, VAL_N, logits_of, sign_test_p, train_student
from step4c_alpha_T_sweep import ALPHA_ANCHOR, ANCHOR_T, TIER, load_anchor, out_path, v4b_path
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
LADDER = (0.0, 0.05, 0.1, 0.2, 0.3)               # 0 是恒等闸不是臂；0.3 是跨脚本闸
SEEDS = (42, 7, 2026)                              # 与第4课b/c 同一批随机流，才能与档案逐位对
PROBE_ALPHA = 0.3                                   # 已在 v4c 档案里的那一格
P_SIG = 0.05


def v4c_name(width: int, tier: str) -> Path:
    return out_path(width, tier)


def main(width: int = 128, tier: str = TIER) -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    v4b = v4b_path(width, tier)
    v4c = v4c_name(width, tier)
    if not v4b.exists() or not v4c.exists():
        raise SystemExit(f"缺 {v4b.name} / {v4c.name}——底和 α={PROBE_ALPHA} 基准全从这两本读，"
                         f"先跑 step4b_seed_variance.py 与 step4c_alpha_T_sweep.py（带上同一组 --tier）")
    anchor = load_anchor(v4b, width, tier)
    floor = anchor["floor"]
    arch = json.loads(v4c.read_text(encoding="utf-8"))
    if int(arch["width"]) != width or arch["tier"] != tier:
        raise SystemExit(f"{v4c.name} 是 {arch['tier']}/S{arch['width']} 的表，不是 {tier}/S{width}")
    probe = arch["grid"][f"a{PROBE_ALPHA:g}_T{ANCHOR_T:g}"]          # 逐种子原始读数，用来对账
    anchor_mean = arch["grid"][f"a{ALPHA_ANCHOR:g}_T{ANCHOR_T:g}"]["mean_gap_imgs"]

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    xd, yd = xt_all[:DATA_TIERS[tier]], torch.tensor(yt_all[:DATA_TIERS[tier]])
    soft = logits_of(teacher, xd)

    h1_line = {a: round(a / ALPHA_ANCHOR * anchor_mean) for a in LADDER if a > 0}
    print(f"[{tier} · S{width}] 训练 {DATA_TIERS[tier]} 张 | 测试 {len(yte)} | α 阶梯 {LADDER} @ T={ANCHOR_T:g} | "
          f"{len(LADDER) * len(SEEDS) + len(SEEDS)} 次训练")
    print(f"[仪器] 种子底（{v4b.name} 的 KD 臂）= {floor} 张；"
          f"锚点格 α={ALPHA_ANCHOR}/T={ANCHOR_T} 三种子均值 {anchor_mean:+.1f} 张（{v4c.name}）")
    print(f"[仪器] α={PROBE_ALPHA} 档案读数（跨脚本对账基准）：均值 {probe['mean_gap_imgs']:+.1f} 张，"
          f"逐种子 {[r['gap_imgs'] for r in probe['per_seed']]}")
    print("\n[先立预言（此后不许改口）]")
    print(f"  H1 残城（伤害 ∝ α，直线过原点）→ " +
          "、".join(f"α={a:g} {h1_line[a]:+.1f} 张" for a in LADDER if a > 0))
    print(f"  污染说（份额只决定多快生效）→ α=0.05 就有 {anchor_mean:+.1f} 张量级的伤害，"
          f"且阶梯前四格彼此差不超过一个底（{floor} 张）")
    print(f"  判别线：α=0.05 实测距 0 更近（|gap| < {abs(h1_line[0.05]) * 2:.0f} 张）→ H1 活；"
          f"距 {anchor_mean:+.1f} 更近（|gap| > {floor}）→ H1 死在最后一块地上")

    ce_pred, rows = {}, {a: [] for a in LADDER if a > 0}
    id_gate, probe_gate = [], []
    print("\n[开跑] 每个种子一副 CE + 阶梯各一副（同起点同批次流）：")
    for seed in SEEDS:
        mce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
        pe = predict(mce, xte, False)
        ce_pred[seed] = pe
        ce_wrong = int((pe != yte).sum())
        line = [f"  种子 {seed:<5} CE 错 {ce_wrong} 张"]
        for a in LADDER:
            if a == 0.0:                       # 恒等闸：这一格不许当臂用，它测的是接线
                m0, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=0.0, t=ANCHOR_T)
                p0 = predict(m0, xte, False)
                n_diff = int((p0 != pe).sum())
                id_gate.append({"seed": seed, "wrong": int((p0 != yte).sum()),
                                "ce_wrong": ce_wrong, "diff_vs_ce_imgs": n_diff,
                                "bit_identical": bool(n_diff == 0 and (p0 != yte).sum().item() == ce_wrong)})
                line.append(f"a0(恒等):差{n_diff}")
                continue
            m, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=a, t=ANCHOR_T)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            broken = int(((pe == yte) & (p != yte)).sum())
            fixed = int(((pe != yte) & (p == yte)).sum())
            rows[a].append({"seed": seed, "alpha": a, "wrong_kd": wrong, "wrong_ce": ce_wrong,
                            "gap_imgs": ce_wrong - wrong, "broken": broken, "fixed": fixed,
                            "diff_vs_ce_imgs": int((p != pe).sum()),
                            "p_sign": round(sign_test_p(broken, fixed), 6)})
            line.append(f"a{a:g}:{ce_wrong - wrong:+d}(差{int((p != pe).sum())})")
            if a == PROBE_ALPHA:
                r = rows[a][-1]
                ref = probe["per_seed"][SEEDS.index(seed)]
                probe_gate.append({"seed": seed,
                                   "match": bool(r["wrong_kd"] == ref["wrong_kd"] and r["broken"] == ref["broken"]
                                                 and r["fixed"] == ref["fixed"]),
                                   "this_run": {k: r[k] for k in ("wrong_kd", "broken", "fixed")},
                                   "archive": {k: ref[k] for k in ("wrong_kd", "broken", "fixed")}})
        print(" ".join(line))

    bad_id = [g for g in id_gate if not g["bit_identical"]]
    print(f"\n[恒等闸 α=0 必须逐位等于 CE] {'3/3 通过：α 这根线接对了，阶梯上的数才是剂量的数' if not bad_id else 'FAIL：' + str(bad_id)}")
    if bad_id:
        print("  α=0 与 CE 不同 ⇒ loss 里 α 的位置不是教科书那个位置（比如 (1−α) 也乘在别处、"
              "或 KL 项被无条件加了进来）。**下面所有阶梯读数作废**，先修接线再谈剂量。")
        raise SystemExit(1)
    bad_probe = [g for g in probe_gate if not g["match"]]
    print(f"[跨脚本闸 α={PROBE_ALPHA}] {'3/3 与 v4c 档案逐位一致 → 本课与第4课c 是同一个训练器' if not bad_probe else 'FAIL：' + str(bad_probe)}")

    print("\n[阶梯] 每格三种子（10000 张；差=与同种子 CE 预测不同的图数，是**分辨率**不是疗效）：")
    summary = {}
    for a, rr in rows.items():
        gaps = [r["gap_imgs"] for r in rr]
        tb, tf = sum(r["broken"] for r in rr), sum(r["fixed"] for r in rr)
        pooled = round(sign_test_p(tb, tf), 6)
        summary[a] = {"gaps_imgs": gaps, "mean_gap_imgs": round(sum(gaps) / len(gaps), 1),
                      "vs_floor": round(abs(sum(gaps) / len(gaps)) / floor, 2),
                      "mean_diff_vs_ce_imgs": round(sum(r["diff_vs_ce_imgs"] for r in rr) / len(rr), 1),
                      "pooled_flips": {"broken": tb, "fixed": tf}, "p_pooled": pooled,
                      "same_dir_seeds": f"{sum(1 for g in gaps if g < 0)}/{len(gaps)}",
                      "h1_pred_imgs": h1_line[a],
                      "beats_h1_line": bool(sum(gaps) / len(gaps) < 2 * h1_line[a])}   # 比直线坏一倍以上
        s = summary[a]
        print(f"  α={a:<5} gap {gaps} 张 均值 {s['mean_gap_imgs']:+6.1f}（底 {floor} 的 {s['vs_floor']}x）| "
              f"合并翻坏 {tb}/翻正 {tf} p={pooled:.4g} | 与 CE 预测不同 {s['mean_diff_vs_ce_imgs']:.0f} 张 | "
              f"H1 该格预测 {h1_line[a]:+.1f} → {'**比直线坏得多：污染型**' if s['beats_h1_line'] else '与直线同量级：H1 撑住'}")

    # ---------- 判决：形状由数字挑，不由人挑 ----------
    small = min(a for a in rows)
    s_small, s_probe = summary[small], summary[PROBE_ALPHA]
    all_same = f"{len(SEEDS)}/{len(SEEDS)}"
    if s_small["p_pooled"] < P_SIG and s_small["same_dir_seeds"] == all_same:
        shape = "污染型（有伤害不必有剂量）"
        reading = (f"α={small:g} 已经把方向摆出来（{all_same} 同号、合并 p={s_small['p_pooled']:.4g}、"
                   f"均值 {s_small['mean_gap_imgs']:+.1f} 张），而它是锚点剂量的 {small / ALPHA_ANCHOR:.0%}"
                   f"——H1 的『按份额掺毒』在最后一块地上也没了：软标签一旦进目标函数，"
                   f"坏的是它**在不在**，不是它占多少")
    elif s_small["p_pooled"] >= P_SIG and s_probe["p_pooled"] < P_SIG:
        shape = "有阈值（H1 半活）"
        reading = (f"小剂量端（α≤{small:g}）配对检验不出方向（p={s_small['p_pooled']:.3g}），"
                   f"到 α={PROBE_ALPHA} 才出（p={s_probe['p_pooled']:.3g}）→ 存在一个 0~{PROBE_ALPHA} 之间的"
                   f"起点，之前测不出、之后基本饱和；说『伤害随 α 线性增长』仍然没有证据")
    else:
        shape = "测不动（这一发没判决）"
        reading = (f"两端都出不了方向（α={small:g} p={s_small['p_pooled']:.3g}、"
                   f"α={PROBE_ALPHA} p={s_probe['p_pooled']:.3g}）→ 本课的分辨率不足以判，别硬解")
    print(f"\n[判决] {shape}：{reading}")
    lo_step = round(s_probe["mean_gap_imgs"] - s_small["mean_gap_imgs"], 1)
    hi_step = round(anchor_mean - probe["mean_gap_imgs"], 1)
    share = round(abs(s_small["mean_gap_imgs"]) / abs(anchor_mean) * 100)
    print(f"[判决的分量] 阶梯第一格 α={small:g}（锚点剂量的 {small / ALPHA_ANCHOR:.0%}）"
          f"就已经买到锚点伤害的 **{share}%**：{s_small['mean_gap_imgs']:+.1f} / {anchor_mean:+.1f} 张")
    print(f"[判决的边界] 剂量往上再加：α={small:g}→{PROBE_ALPHA:g}（剂量 ×{PROBE_ALPHA / small:.0f}）"
          f"只再走 {lo_step:+.1f} 张；α={PROBE_ALPHA:g}→{ALPHA_ANCHOR:g}（剂量 ×{ALPHA_ANCHOR / PROBE_ALPHA:.2f}）"
          f"只再走 {hi_step:+.1f} 张"
          + (f" → 前一段的剂量倍数是后一段的 {PROBE_ALPHA / small / (ALPHA_ANCHOR / PROBE_ALPHA):.2f} 倍，"
             f"换来的增量却是它的 {abs(lo_step / hi_step):.1f} 倍：**曲线在第一个读数之后基本就平了**"
             if hi_step else ""))

    under = [a for a, s in summary.items() if abs(s["mean_gap_imgs"]) <= floor]
    print(f"\n[自我审计① 判决单位是配对，不是底] "
          + (f"α∈{under} 各格的 |gap| 在种子底（{floor} 张）以下——这是**设计使然**："
             "H1 自己就预测它们小。所以本课的判决单位是逐图配对（同一副权重内比翻牌），"
             "不是跨重训的极差。反过来也成立：谁要是拿『不到底』来否掉这些格，"
             "也就是在否掉自己用来立锚点的那套配对检验。" if under else
             f"这一轮每一格都过了种子底（{floor} 张）——连 α={small:g} 那格都过了，"
             "所以本课不需要拿配对检验去救小读数；判决的力气全在第一格的**方向**上（同号 + 合并 p），"
             "不在它离底多远。"))
    print("[自我审计② 恒等闸只证明接线，不证明灵敏度] α=0 逐位等于 CE 说的是『α 确实乘在该乘的地方』；"
          "它不给仪器定分辨率。分辨率由每格与 CE 不同的预测图数给出（上面『差 xxx 张』那一列）——"
          "那一列如果本身就是几十张，任何『这一格与 CE 打平』的说法都只是没看见。")
    print("[自我审计③ 阶梯是 3 条随机流 × 每流一次] 与第4课b/c 同病：每种子只训一次，"
          "跨种子的极差没有二次底。要把它当正式判决用，得先照第4课e 的办法扩到 8 条流。")
    t1 = arch["grid"].get(f"a{ALPHA_ANCHOR:g}_T1")
    print(f"[自我审计④ 只在 T={ANCHOR_T:g} 上扫 α] 第4课c 判过 T 才是主效应、α 是平的，"
          "所以本阶梯是**在伤害最大的那个温度上**看剂量；"
          + (f"同一本档案里 α={ALPHA_ANCHOR}/T=1 那格 {t1['mean_gap_imgs']:+.1f} 张，"
             "按污染说它也该跟着饱和，但本课没测。" if t1 else
             f"这份档案里没有 α={ALPHA_ANCHOR}/T=1 那格，无从对照。"))
    print("[自我审计⑤ 对账基准跟着档案走] 本课的 α=0.3 格与 v4c 同种子同数据同起点，"
          "所以能逐位对账；换了 --tier 或 --width 之后这条闸连着换档案——"
          "拿 A 档的账对 B 档的数就是自欺。")

    report = {"tier": tier, "width": width, "temps_fixed": ANCHOR_T, "ladder": list(LADDER),
              "seeds": list(SEEDS), "probe_alpha": PROBE_ALPHA,
              "floor_from": v4b.name, "seed_floor_imgs": floor,
              "anchor_mean_from": v4c.name, "anchor_mean_gap_imgs": anchor_mean,
              "predictions_ex_ante": {"H1_line_through_origin": h1_line,
                                      "contamination_pred": f"α={LADDER[1]:g} 即接近 {anchor_mean:+.1f} 张"},
              "gate_alpha_zero_identity": id_gate, "gate_probe_vs_v4c": probe_gate,
              "ladder_summary": {str(a): summary[a] for a in summary},
              "arms": {f"α{a:g} 种子{r['seed']}": r for a, rr in rows.items() for r in rr},
              "verdict": {"shape": shape, "reading": reading,
                          "first_cell_alpha": small,
                          "first_cell_share_of_anchor_damage_pct": share,
                          "step_low_imgs": lo_step, "step_low_dose_x": round(PROBE_ALPHA / small, 2),
                          "step_high_imgs": hi_step,
                          "step_high_dose_x": round(ALPHA_ANCHOR / PROBE_ALPHA, 2)},
              "limitations": ["每种子每格只训一次：跨种子极差没有二次底",
                              f"只扫 T={ANCHOR_T:g} 一列，T=1/T=6 上的剂量响应未测",
                              "小 α 各格均值本就低于种子底（H1 的预测就是这个量级），"
                              "判决依赖逐图配对检验而非重训抖动",
                              f"只 {tier}+S{width}：跨档/跨架构要各自带 --tier/--width 重跑，"
                              "档案不同不能混读"]}
    out = HERE / v4c.name.replace("v4c", "v4f")     # 命名只有一处权威：档名跟着 v4c 走
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{len(SEEDS) * (len(LADDER) + 1)} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="α→0 剂量曲线（默认 S128+满数据，T=3 一列）")
    ap.add_argument("--width", type=int, choices=(128, 64), default=128)
    ap.add_argument("--tier", choices=tuple(DATA_TIERS), default=TIER,
                    help="数据档位；底读该档 v4b、α=0.3 基准读该档 v4c，两本缺一就退出")
    args = ap.parse_args()
    main(args.width, args.tier)
