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

跑法（约 8~10 分钟，纯 CPU，零 API 费）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step5_second_teacher.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, payload_kb, predict
from step3_prune import channel_importance, fine_tune, keep_indices, rebuild
from step4_distill import TIERS, VAL_N, logits_of, sign_test_p, soft_label_audit, train_student
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
WIDTH = 128                                    # 学生固定 S128：本课换的是老师，不是学生
RATIO = 0.5                                    # 与第3课 R50F 同一刀（conv2 输出通道 32→16）
ALPHAS = (0.7, 1.0)                            # 只留两档：本课的判别量是 T 的**符号**，两档够定符号
TEMPS = (1.0, 3.0, 6.0)
SEEDS = (42, 7, 2026)                          # 与第4课c 同一批种子：CE 锚可逐位对账
D4 = 1.693                                     # n=3 正态极差期望，极差 → σ̂（第4课d 的公式）
OUT = HERE / "compress_report_v5.json"
TEACHER_B_PT = MODELS / "mnist_cnn_pruned50.pt"
V4C = {"满数据8500": "compress_report_v4c.json", "低数据2000": "compress_report_v4c_low.json"}
T_MAIN_A = {"满数据8500": -124.3, "低数据2000": 56.4}    # 第4课c 两档 T 主效应（老师 A）
SHRINK_LINE = 62.0                             # A 的满数据之害的一半，收缩判据的尺


def key(a: float, t: float) -> str:
    return f"a{a:g}_T{t:g}"


def floors_from_v4b() -> dict[str, int]:
    """两档各自的借来的底（第4课b 的 KD 臂错图极差）——从档案读，不抄字面量。"""
    d = json.loads((HERE / "compress_report_v4b.json").read_text(encoding="utf-8"))
    if int(d["width"]) != WIDTH:
        raise SystemExit(f"compress_report_v4b.json 是 S{d['width']} 的，本课学生固定 S{WIDTH}")
    out = {}
    for tier in TIERS:
        try:
            out[tier] = int(d["verdicts"][tier]["seed_floor_imgs"]["KD"])
        except KeyError:
            raise SystemExit(f"档案里没有 {tier} 的种子底，换档位之前先把第4课b 那一档跑出来")
    return out


def ce_anchors_from_v4c() -> dict[tuple[str, int], int]:
    """第4课c 两档的 CE 锚错图数：CE 臂不吃软标签，所以它是本课的免费对账基准。"""
    out = {}
    for tier, fname in V4C.items():
        p = HERE / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("tier") != tier:
            raise SystemExit(f"{fname} 自称 {d.get('tier')!r}，本课当 {tier} 用——先看一眼档案")
        for s, r in d["ce_ref_by_seed"].items():
            out[(tier, int(s))] = int(r["wrong"])
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


def teacher_b(force_retrain: bool = False) -> tuple[torch.nn.Module, dict]:
    """第二副老师 = 第3课那把手术刀 + 同一套微调。keep 索引由 fp32 权重算出（确定性），
    所以缓存的权重能原样装回去；微调的批次流种子写死，换机器不重跑也是同一副。"""
    base = build_cnn()
    base.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    keep = keep_indices(channel_importance(base), RATIO)
    if TEACHER_B_PT.exists() and not force_retrain:
        m = rebuild(base, keep)
        m.load_state_dict(torch.load(TEACHER_B_PT, weights_only=True))
        return m.eval(), {"reused": True, "kept_channels": int(keep.numel())}
    xt_all, yt_all, _, _ = load_split()
    cut = len(xt_all) - VAL_N
    torch.manual_seed(42)
    m, hist = fine_tune(rebuild(base, keep), xt_all[:cut], yt_all[:cut], xt_all[cut:], yt_all[cut:])
    torch.save(m.state_dict(), TEACHER_B_PT)
    return m.eval(), {"reused": False, "kept_channels": int(keep.numel()),
                      "last_val_acc": hist[-1]["val_acc"]}


def main() -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    floor = floors_from_v4b()
    ce_arch = ce_anchors_from_v4c()

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    ta = build_cnn()
    ta.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    ta.eval()
    tb, info = teacher_b()
    print(f"第二副老师就位（{info['kept_channels']} 通道的 R50F）| "
          f"{'复用缓存权重' if info.get('reused') else '本轮新训并存档'} {TEACHER_B_PT.name} | "
          f"体积 {payload_kb(tb):.1f} KB")

    # ---------- 零训练一：B 是"另一份知识"还是 A 的影子 ----------
    pa, pb = predict(ta, xte, False), predict(tb, xte, False)
    agree = int((pa == pb).sum())
    kl_ab = float(torch.nn.functional.kl_div(torch.log_softmax(logits_of(ta, xte), 1),
                                             torch.softmax(logits_of(tb, xte), 1),
                                             reduction="batchmean"))
    acc_a = 1 - int((pa != yte).sum()) / len(yte)
    acc_b = 1 - int((pb != yte).sum()) / len(yte)
    print(f"[独立性体检] 两副老师在 {len(yte)} 张测试图上判得不一致 **{len(yte) - agree} 张**；"
          f"软标签 KL(A‖B) = {kl_ab:.4f}（≈0 就是影子，换老师等于没换）")
    print(f"[独立性体检] 各自测试精度 A {acc_a:.4f} / B {acc_b:.4f}")

    # ---------- 零训练二：B 手里有没有暗知识（预言看着这一栏立） ----------
    audit_a, audit_b = {}, {}
    for tier, n in TIERS.items():
        xd = xt_all[:n]
        audit_a[tier] = soft_label_audit(logits_of(ta, xd))
        audit_b[tier] = soft_label_audit(logits_of(tb, xd))
    ra = audit_a["满数据8500"]["原始"]
    rb = audit_b["满数据8500"]["原始"]
    gap_pp = round(ra["pct_top1_gt_0.99"] - rb["pct_top1_gt_0.99"], 1)
    print("\n[软标签体检（拟合集 8500 张，原始温度）]")
    print(f"  A：top1 {ra['mean_top1']}，塌缩率 {ra['pct_top1_gt_0.99']}%，第二名概率 {ra['mean_p2']}")
    print(f"  B：top1 {rb['mean_top1']}，塌缩率 {rb['pct_top1_gt_0.99']}%，第二名概率 {rb['mean_p2']}")

    discriminating = gap_pp >= 10.0
    print("\n[先立预言（此后不许改口）]")
    print(f"  判别量 = 每档的 T 主效应（α∈{ALPHAS} 平均，T=6 − T=1，单位张，负=KD 更差）；"
          f"老师 A 的实测：{T_MAIN_A}")
    print(f"  B 比 A 的塌缩率低 {gap_pp}pp（p2 {ra['mean_p2']}→{rb['mean_p2']}）")
    if discriminating:
        print(f"  → 这一发有判别力。H_老师个性 预言满数据 T 主效应收缩到 ±{SHRINK_LINE} 张以内"
              f"（软标签不那么干，稀释就没那么伤）；H_数据量 预言符号仍为负、幅度同量级（>{SHRINK_LINE} 张）。"
              f"低数据档是第二个判别点：老师个性说 B 的暗知识更值钱、正号应当更大，"
              f"数据量说正号照搬、幅度不必变大")
    else:
        print(f"  → 两副老师的软标签一样干（塌缩率只差 {gap_pp}pp < 10pp），于是两说对**满数据档**"
              f"给出同一个预测（T 主效应为负、幅度同量级）：这一档只算「形状复现」，不算判别。"
              f"本课唯一的判别量是低数据档的符号：数据量说 → 翻正且过闸；"
              f"老师个性说 → A 那发的 +56.4 张是这副老师的运气，B 这里应当回落到 ±底 以内")
    for tier in TIERS:
        print(f"  闸门（{tier}）：借来的底 {floor[tier]} 张（跨格量用它）；单格同时报自己的 ±2SE（D4={D4}）")

    # ---------- 开跑 ----------
    result: dict[str, dict] = {}
    ce_parity: list[dict] = []
    for tier, n in TIERS.items():
        xd, yd = xt_all[:n], torch.tensor(yt_all[:n])
        soft = logits_of(tb, xd)                    # 全流程唯一与老师 A 不同的一行
        print(f"\n[开跑] {tier} | 学生 S{WIDTH} | {len(ALPHAS)}×{len(TEMPS)}×{len(SEEDS)} = "
              f"{len(ALPHAS) * len(TEMPS) * len(SEEDS)} 次 KD + {len(SEEDS)} 次 CE 锚")
        rows: dict[str, list[dict]] = {key(a, t): [] for a in ALPHAS for t in TEMPS}
        ce_ref: dict[int, int] = {}
        for seed in SEEDS:
            ce, _ = train_student(WIDTH, xd, yd, soft, "CE", xv, yv, seed=seed)
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
                    m, _ = train_student(WIDTH, xd, yd, soft, "KD", xv, yv,
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
    for tier in TIERS:
        r = result[tier]
        print(f"  ② {tier}：老师 A 的 T 主效应 {r['teacher_A_t_main_imgs']:+.1f} 张 → 老师 B {r['t_main_effect_imgs']:+.1f} 张"
              f"（{r['vs_borrowed_floor']}x 底，同号 {r['sign_agrees_with_A']}，过闸 {r['beats_floor']}）")
    low, full = result["低数据2000"], result["满数据8500"]
    if discriminating:
        shrink = abs(full["t_main_effect_imgs"]) < SHRINK_LINE
        v3 = ("软标签形状确实管着伤害幅度：H_老师个性 拿到一票，"
              "第4课那句『符号由数据量决定』要降级为『在这一族老师上成立』" if shrink else
              "幅度没收缩：换了一整份软标签，满数据之害还是那么大 → H_数据量 拿到一票")
        print(f"  ③ 终判：B 的满数据 T 主效应收缩到 ±{SHRINK_LINE} 张以内 = {shrink} → {v3}")
    else:
        got_positive = low["sign_agrees_with_A"] and low["beats_floor"]
        v3 = ("翻正且过闸 → 正号跨老师复现，数据量说占优：两副软标签一样干却给出同一个翻转，"
              "说明这个翻转不是这副老师的运气" if got_positive else
              "没有跨老师复现 → 第4课那条 H+ 只属于老师 A，"
              "「符号由数据量决定」这句要改口成「由这一副老师决定」")
        print(f"  ③ 终判（本课只有低数据档这一个判别量）：B 在 2000 张档的 T 主效应{v3}")
    for t, r in result.items():
        over = [f"{k} {s['mean']:+.1f}(±{s['gate']}, p={s['p_pooled']})"
                for k, s in r["stats"].items() if s["over_own_gate"]]
        print(f"  ④ 单格读数（{t}，只报待追不当判决）：" + ("、".join(over) or "无格过自己的闸"))
    print("  ⑤ 独立性账：换老师买到的只有一份软标签（1 个重复）。"
          "仍不是新数据集、新任务、新学生架构。")

    report = {"teacher_b": {"source": "第3课手术刀（NetU L1 剪 conv2 50% + 微调 8 epoch，种子 42）",
                            "kept_channels": info["kept_channels"], "kb": round(payload_kb(tb), 1),
                            "reused_cache": bool(info.get("reused")), "path": TEACHER_B_PT.name,
                            "test_acc": round(acc_b, 4), "teacher_a_test_acc": round(acc_a, 4),
                            "disagree_imgs": len(yte) - agree, "kl_A_given_B": round(kl_ab, 4)},
              "soft_label_audit": {t: {"A": audit_a[t], "B": audit_b[t]} for t in TIERS},
              "prediction_ex_ante": {"discriminating": discriminating,
                                     "shrink_line_imgs": SHRINK_LINE, "collapse_gap_pp": gap_pp,
                                     "gate": "跨格量用借来的底，单格同时报自己的 ±2SE（第4课d 的两把尺）"},
              "ce_parity_vs_v4c": ce_parity, "ce_parity_all_pass": bool(n_pass == len(ce_parity)),
              "verdict": {"discriminating": discriminating,
                          "low_data_positive_across_teacher": bool(
                              result["低数据2000"]["sign_agrees_with_A"]
                              and result["低数据2000"]["beats_floor"]),
                          "full_data_shrunk": bool(discriminating
                                                   and abs(full["t_main_effect_imgs"]) < SHRINK_LINE),
                          "sentence": v3},
              "width": WIDTH, "alphas": list(ALPHAS), "temps": list(TEMPS), "seeds": list(SEEDS),
              "tiers": result, "test_n": int(len(yte)),
              "limitations": ["α 只留两档（0.7/1.0）：本课的判别量是 T 的符号，两档够定符号；"
                              "α 轴的形状第4课已在三个档案里量过（0.11x/0.11x/0.33x），不在本课范围",
                              "每格 3 条随机流，且与第4课c 共用同种子 ⇒ 两副老师的 gap 可逐种子配对；"
                              "但「换老师」本身只有 1 个重复：B 是个例还是这一族稀疏 CNN 的代表，没测",
                              "±2SE 是独立合成的下界（三种子跨格共用同一批随机流），与第4课d 同一告警",
                              "预言是在看完 B 的零训练软标签体检之后立的（学生训练之前，但不是看到 B 之前）；"
                              "体检差 <10pp 时脚本自动把本课降级为「形状复现」而不算判别，这条写在分支里",
                              "学生固定 S128：换老师与换宽度不叠加，S64 上的跨老师复证要另跑一发"]}
    n_train = sum(len(r["ce_ref_by_seed"]) + len(r["grid"]) * len(SEEDS) for r in result.values())
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（{n_train} 次学生训练 + 1 副老师的剪枝微调，"
          f"{len(yte)} 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    main()
