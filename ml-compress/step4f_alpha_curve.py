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

第二发（`--dataset fashion`）问的是这条判决的**适用范围**："伤害是污染型不是剂量型"目前只站在
MNIST 的两副学生上（S128 79% / S64 63%），任务那一维一次都没换过。换任务要连三样东西一起换，
而且**一样都不许抄**：
  · **温度列**：MNIST 这条阶梯铺在 T=3（锚点温度、也是第4课c 判定的伤害列）上，可 Fashion 那本账
    只有 T∈{1,6}。所以阶梯所在那一列由 `max(v6.temps)` 现读 = **T=6**（伤害最大那一端，
    a1_T6 = −141.0 张），不是把 3.0 抄过来——抄过来会得到一格不存在的锚点、以及一个对不上的闸。
  · **锚点剂量与跨脚本闸**：Fashion 没跑过 α=0.3，那本账里最小的 α 是 0.7 ⇒ 阶梯的最高一档
    （=锚点=闸格）取 `min(v6.alphas)` = **0.7**，它同时是 H1 直线的那把尺子和逐位对账的基准。
    于是这一发比 MNIST 那一发多一格（阶梯 0.0/0.05/0.1/0.2/0.3/**0.7**），而"锚点格就是闸格"
    这件事必须写在报告里：它意味着这一发没有独立于锚点的第二条对账线。
  · **底**：读 Fashion 该档自己的 own floor（满数据档 112 张），不借 MNIST 的 41/37。
α=0 恒等闸跨任务照样成立（它测的是接线，不是世界）。产物 `compress_report_v4f_fashion.json`。
跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4f_alpha_curve.py --dataset fashion --tier 满数据8500

第三发（`--dataset fashion --tier 满数据8500 --nseeds 8 --thin --t 1`）补的是第二发自己留下的那条边界：
上一发的阶梯铺在 `max(v6.temps)` = T=6 上，而 v6 那本账里 α=0.7/T=1 那格读数是 **+44.7 张（正号：
KD 比 CE 少犯错）**。所以"有阈值"当时只是 **T=6 那一列**的形状，不能往别的温度列搬。把阶梯搬到 T=1，
两说的分歧点照旧只有一个数字，只是符号反了：
  · 污染说 → α=0.05 就该吃到 +44.7 量级的增益（这一列的阶梯是平的）；
  · H1 直线 → α=0.05 只有 +3.2 张（这一列的阶梯是陡的）。
`--t` 换的是**列**不是档，所以档案名带 `_t1`，且 `check_t` 只允许开在这一本账已有的温度列上：
拿一本没有那一列的账铺阶梯，锚点格与闸格都会凭空造出来（最坏的一种绿）。产物
`compress_report_v4f_fashion_n8_thin_t1.json`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import DATA_TIERS, TEACHER_ARCHIVE_ACC, VAL_N, logits_of, sign_test_p, train_student
from step4e_marginal_recheck import OLD_SEEDS, summarize          # 复现单位是流：CI 用同一套系数表
from step4c_alpha_T_sweep import ALPHA_ANCHOR, ANCHOR_T, TIER, V4C_SUFFIX, key, load_anchor, out_path, v4b_path
from step6_fashion_replication import TEACHER_PT, load_fashion
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
LADDER = (0.0, 0.05, 0.1, 0.2, 0.3)               # 0 是恒等闸不是臂；0.3 是跨脚本闸
SEEDS = (42, 7, 2026)                              # 与第4课b/c 同一批随机流，才能与档案逐位对
SEEDS8 = SEEDS + (11, 202, 3003, 40004, 55555)     # 第4课e 那五条：老 3 条只对账，新 5 条才是复现
PROBE_ALPHA = 0.3                                   # 已在 v4c 档案里的那一格
P_SIG = 0.05
V6 = HERE / "compress_report_v6_fashion.json"

DATASETS = {
    "mnist": {"loader": load_split, "teacher_pt": MODELS / "mnist_cnn_fp32.pt"},
    "fashion": {"loader": load_fashion, "teacher_pt": TEACHER_PT},
}


def v4c_name(width: int, tier: str) -> Path:
    return out_path(width, tier)


def name_suffix(tier: str, width: int) -> str:
    """产物名跟着每一个判别变量走；后缀规则只有一个来源（step4c 的 V4C_SUFFIX）。"""
    if tier != TIER and tier not in V4C_SUFFIX:
        raise SystemExit(f"档 {tier} 没登记档案名后缀（V4C_SUFFIX 只有 {list(V4C_SUFFIX)}）："
                         f"回落到默认名会和 {TIER} 那本撞名，无声盖掉它的判决")
    return V4C_SUFFIX.get(tier, "") + ("" if width == 128 else f"_s{width}")


def check_t(t: float, grid: dict, where: str) -> float:
    """阶梯只能铺在这一本账**已有**的温度列上：没有那一列就没有锚点格，
    也就没有尺子更没有闸——换列等于换一本账。"""
    cols = sorted({float(k.split("_T")[1]) for k in grid})
    if t not in cols:
        raise SystemExit(f"{where} 这本账只有温度列 {cols}，没有 T={t:g}："
                         f"先把那一列跑进账里再来铺阶梯")
    return t


def setup(dataset: str, width: int, tier: str, extra: str = "",
          t_col: float | None = None) -> dict:
    """六样读数（温度列 / 锚点剂量 / 闸格 / 该档的底 / 锚点伤害 / 产物名）全从**那一本账**现读。

    为什么必须现读而不是各写一套：Fashion 那本账没有 T=3、也没有 α=0.3——把 MNIST 的两个常量
    抄过来，会得到一个不存在的锚点格和一格对不上的闸（而"对不上的闸"是最坏的一种绿）。
    账本字段布局的差异只允许出现在这一个函数里。"""
    if dataset == "mnist":
        v4b, v4c = v4b_path(width, tier), v4c_name(width, tier)
        missing = [p.name for p in (v4b, v4c) if not p.exists()]
        if missing:
            raise SystemExit(f"缺 {'、'.join(missing)}——底和 α={PROBE_ALPHA} 基准全从这两本读，"
                             f"先跑 step4b_seed_variance.py 与 step4c_alpha_T_sweep.py（带上同一组 --tier）")
        arch = json.loads(v4c.read_text(encoding="utf-8"))
        if int(arch["width"]) != width or arch["tier"] != tier:
            raise SystemExit(f"{v4c.name} 是 {arch['tier']}/S{arch['width']} 的表，不是 {tier}/S{width}")
        anc_a, gate_a, lad = ALPHA_ANCHOR, PROBE_ALPHA, LADDER
        t = ANCHOR_T if t_col is None else t_col
        floor, floor_kind, floor_from = load_anchor(v4b, width, tier)["floor"], "v4b 的 KD 臂跨种子极差", v4b.name
        grid, anchor_from = arch["grid"], v4c.name
        t = check_t(t, grid, v4c.name)
        out = HERE / v4c.name.replace("v4c", "v4f").replace(".json", f"{extra}.json")
        rows_of = lambda k: grid[k]["per_seed"]                        # noqa: E731
        need = [key(anc_a, t), key(gate_a, t)]
    else:
        if not V6.exists():
            raise SystemExit(f"缺 {V6.name}——换任务的阶梯只能对着那一本任务自己的账铺，先跑 step6")
        d = json.loads(V6.read_text(encoding="utf-8"))
        if d.get("dataset") != "FashionMNIST" or int(d["width"]) != width or tier not in d["per_tier"]:
            raise SystemExit(f"{V6.name} 是 {d.get('dataset')}/S{d.get('width')}"
                             f"（档位 {list(d.get('per_tier', {}))}），装不下 {dataset}/S{width}/{tier}")
        pt, lad_t = d["per_tier"][tier], [float(x) for x in d["temps"]]
        t = max(lad_t) if t_col is None else t_col   # 默认铺在伤害最大那一端；--t 换列
        anc_a = min(float(x) for x in d["alphas"])  # 那本账里最小的 α 就是锚点剂量（0.7）
        gate_a, lad = anc_a, LADDER + (anc_a,)      # 没有 α=0.3 可对账 ⇒ 闸落在锚点格上
        floor, floor_kind, floor_from = int(pt["own_floor_imgs"]), "该档自己的底（第4课g 规矩）", V6.name
        grid, anchor_from = pt["grid"], f"{V6.name} 的 {tier}"
        t = check_t(t, grid, V6.name)
        rows_of = lambda k: pt["raw_rows"][k]                          # noqa: E731
        out = HERE / f"compress_report_v4f_{dataset}{name_suffix(tier, width)}{extra}.json"
        need = [key(anc_a, t), key(gate_a, t)]
    miss = [k for k in dict.fromkeys(need) if k not in grid]
    if miss:
        raise SystemExit(f"{anchor_from} 里没有 {miss} 这几格：这一列温度/这个剂量那一本账没跑过，"
                         f"阶梯的尺子和闸就没有基准（有 {sorted(grid)}）")
    peers = {tt: grid[key(anc_a, tt)]["mean_gap_imgs"] for tt in
             sorted({float(k.split("_T")[1]) for k in grid} - {t})
             if key(anc_a, tt) in grid}
    gate = rows_of(key(gate_a, t))
    return {"t": t, "anchor_alpha": anc_a, "gate_alpha": gate_a, "ladder": lad,
            "teacher_acc": (float(d["teacher"]["test_acc"]) if dataset == "fashion" else TEACHER_ARCHIVE_ACC),
            "anchor_mean": float(grid[key(anc_a, t)]["mean_gap_imgs"]),
            "gate_rows": gate, "gate_mean": round(sum(r["gap_imgs"] for r in gate) / len(gate), 1),
            "floor": floor, "floor_kind": floor_kind, "floor_from": floor_from, "anchor_from": anchor_from,
            "peers": peers, "out": out,
            "gate_is_anchor": bool(gate_a == anc_a), "grid_keys": sorted(grid)}


def main(dataset: str = "mnist", width: int = 128, tier: str = TIER,
         nseeds: int = 3, thin: bool = False, t_col: float | None = None) -> None:
    ds = DATASETS[dataset]
    if nseeds not in (3, 8):
        raise SystemExit(f"复现流只有 3（对账那批）与 8（加第4课e 那五条）两档")
    seeds = SEEDS if nseeds == 3 else SEEDS8
    extra = ("" if nseeds == 3 else "_n8") + ("_thin" if thin else "") + \
            ("" if t_col is None else f"_t{t_col:g}")
    if not ds["teacher_pt"].exists():
        raise SystemExit(f"缺 {ds['teacher_pt'].name}——这一课的老师权重只能复用，别现造")
    L = setup(dataset, width, tier, extra, t_col)
    T, LAD, PROBE, ANC = L["t"], L["ladder"], L["gate_alpha"], L["anchor_alpha"]
    if thin:
        # 薄阶梯：只留判决真正吃的那两格（最小剂量 + 锚点剂量），恒等闸只在老 3 条流上跑。
        # 为什么可以砍：这一发要判的是"从 0.05 走到 0.7 这一步陡不陡"，中间三格不参与那个配对差。
        LAD = (0.0, LAD[1], ANC)
        print(f"[薄阶梯] 只跑 {LAD}：两格剂量差 = 污染说与 H1 的全部分歧所在")
    floor, anchor_mean = L["floor"], L["anchor_mean"]
    probe = {"mean_gap_imgs": L["gate_mean"], "per_seed": L["gate_rows"]}
    out = L["out"]
    if out.exists():
        raise SystemExit(f"{out.name} 已入库——重跑会盖掉那本账。要重跑先删或改名")

    xt_all, yt_all, xte, yte = ds["loader"]()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(ds["teacher_pt"], weights_only=True))
    teacher.eval()
    xd, yd = xt_all[:DATA_TIERS[tier]], torch.tensor(yt_all[:DATA_TIERS[tier]])
    soft = logits_of(teacher, xd)

    h1_line = {a: round(a / ANC * anchor_mean) for a in LAD if a > 0}
    print(f"[{dataset} · {tier} · S{width}] 训练 {DATA_TIERS[tier]} 张 | 测试 {len(yte)} | "
          f"α 阶梯 {LAD} @ T={T:g} | {len(seeds)} 条流")
    print(f"[仪器] 底（{L['floor_from']} 的{L['floor_kind']}）= {floor} 张；"
          f"锚点格 α={ANC:g}/T={T:g} 三种子均值 {anchor_mean:+.1f} 张（{L['anchor_from']}）")
    print(f"[仪器] α={PROBE:g} 档案读数（跨脚本对账基准）：均值 {probe['mean_gap_imgs']:+.1f} 张，"
          f"逐种子 {[r['gap_imgs'] for r in probe['per_seed']]}")
    if L["gate_is_anchor"]:
        print(f"[仪器] **闸格就是锚点格**（α={PROBE:g}=锚点剂量）：这一发只剩恒等闸是独立于锚点的对账线")
    print("\n[先立预言（此后不许改口）]")
    print(f"  H1 残城（伤害 ∝ α，直线过原点）→ " +
          "、".join(f"α={a:g} {h1_line[a]:+.1f} 张" for a in LAD if a > 0))
    print(f"  污染说（份额只决定多快生效）→ α={LAD[1]:g} 就有 {anchor_mean:+.1f} 张量级的效应（方向随锚点符号），"
          f"且阶梯小剂量各格彼此差不超过一个底（{floor} 张）")
    print(f"  判别线：α={LAD[1]:g} 实测距 0 更近（|gap| < {abs(h1_line[LAD[1]]) * 2:.0f} 张）→ H1 活；"
          f"距 {anchor_mean:+.1f} 更近（|gap| > {floor}）→ H1 死在最后一块地上")

    ce_pred, rows = {}, {a: [] for a in LAD if a > 0}
    id_gate, probe_gate = [], []
    print("\n[开跑] 每个种子一副 CE + 阶梯各一副（同起点同批次流）：")
    for seed in seeds:
        mce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
        pe = predict(mce, xte, False)
        ce_pred[seed] = pe
        ce_wrong = int((pe != yte).sum())
        line = [f"  种子 {seed:<5} CE 错 {ce_wrong} 张"]
        for a in LAD:
            if a == 0.0 and thin and seed not in SEEDS:
                continue                     # 薄阶梯：接线证明在老 3 条上跑过就够（它不随种子变）
            if a == 0.0:                       # 恒等闸：这一格不许当臂用，它测的是接线
                m0, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=0.0, t=T)
                p0 = predict(m0, xte, False)
                n_diff = int((p0 != pe).sum())
                id_gate.append({"seed": seed, "wrong": int((p0 != yte).sum()),
                                "ce_wrong": ce_wrong, "diff_vs_ce_imgs": n_diff,
                                "bit_identical": bool(n_diff == 0 and int((p0 != yte).sum()) == ce_wrong)})
                line.append(f"a0(恒等):差{n_diff}")
                continue
            m, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=a, t=T)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            broken = int(((pe == yte) & (p != yte)).sum())
            fixed = int(((pe != yte) & (p == yte)).sum())
            rows[a].append({"seed": seed, "alpha": a, "wrong_kd": wrong, "wrong_ce": ce_wrong,
                            "gap_imgs": ce_wrong - wrong, "broken": broken, "fixed": fixed,
                            "diff_vs_ce_imgs": int((p != pe).sum()),
                            "p_sign": round(sign_test_p(broken, fixed), 6)})
            line.append(f"a{a:g}:{ce_wrong - wrong:+d}(差{int((p != pe).sum())})")
            if a == PROBE and seed in SEEDS:   # 只有老 3 条有跨脚本账可核
                r = rows[a][-1]
                ref = probe["per_seed"][SEEDS.index(seed)]
                probe_gate.append({"seed": seed,
                                   "match": bool(r["wrong_kd"] == ref["wrong_kd"] and r["broken"] == ref["broken"]
                                                 and r["fixed"] == ref["fixed"]),
                                   "this_run": {k: r[k] for k in ("wrong_kd", "broken", "fixed")},
                                   "archive": {k: ref[k] for k in ("wrong_kd", "broken", "fixed")}})
        print(" ".join(line))

    bad_id = [g for g in id_gate if not g["bit_identical"]]
    src = "v4c" if dataset == "mnist" else "v6"
    print(f"\n[恒等闸 α=0 必须逐位等于 CE] {f'{len(id_gate)}/{len(id_gate)} 通过：α 这根线接对了，阶梯上的数才是剂量的数' if not bad_id else 'FAIL：' + str(bad_id)}")
    if bad_id:
        print("  α=0 与 CE 不同 ⇒ loss 里 α 的位置不是教科书那个位置（比如 (1−α) 也乘在别处、"
              "或 KL 项被无条件加了进来）。**下面所有阶梯读数作废**，先修接线再谈剂量。")
        raise SystemExit(1)
    bad_probe = [g for g in probe_gate if not g["match"]]
    if not probe_gate:
        raise SystemExit(f"跨脚本闸一条记录都没有（阶梯 {list(rows)} 里没有闸格 α={PROBE:g}）："
                         "空表『全过』不是通过，是没测到")
    print(f"[跨脚本闸 α={PROBE:g}] {f'{len(probe_gate)}/{len(probe_gate)} 与 {src} 档案逐位一致 → 本课与那一课是同一台训练器' if not bad_probe else 'FAIL：' + str(bad_probe)}")
    if bad_probe:
        print(f"  对不上 ⇒ 本课与 {src} 不是同一台仪器，锚点伤害就没有基准，阶梯读数只能重跑")
        raise SystemExit(1)

    dirn = -1.0 if anchor_mean < 0 else 1.0      # 伤害方向跟着锚点符号：T=1 那列锚点可能是正号
    print(f"\n[阶梯] 每格 {len(seeds)} 条随机流（10000 张全量测试；差=与同种子 CE 预测不同的图数，是**分辨率**不是疗效）：")
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
                      "beats_h1_line": bool(dirn * (sum(gaps) / len(gaps)) > 2 * abs(h1_line[a]))}
        s = summary[a]
        print(f"  α={a:<5} gap {gaps} 张 均值 {s['mean_gap_imgs']:+6.1f}（底 {floor} 的 {s['vs_floor']}x）| "
              f"合并翻坏 {tb}/翻正 {tf} p={pooled:.4g} | 与 CE 预测不同 {s['mean_diff_vs_ce_imgs']:.0f} 张 | "
              f"H1 该格预测 {h1_line[a]:+.1f} → {'**比直线坏得多：污染型**' if s['beats_h1_line'] else '与直线同量级：H1 撑住'}")

    # ---------- 配对步长（第4课e 那把尺）：同一条流里 锚点格 − 最小剂量格 ----------
    # 污染说预测这一步 ≈ 0（两格都早饱和），H1 直线预测它 ≈ 锚点伤害的绝大部分。
    # 老 3 条流那发只有逐图配对（合并 p）：同一批 10000 张图上重复数牌，复现单位是图不是流。
    small = min(a for a in rows)
    by_seed = {a: {r["seed"]: r["gap_imgs"] for r in rows[a]} for a in (small, ANC)}
    step = [by_seed[ANC][sd] - by_seed[small][sd] for sd in seeds]
    pair_step = summarize(step, f"α={small:g}→{ANC:g} 配对步长")
    crosses = bool(pair_step["ci95_imgs"][0] < 0 < pair_step["ci95_imgs"][1])
    print(f"\n[配对步长·以流为单位] α={small:g}→{ANC:g}（剂量 ×{ANC / small:.0f}）"
          f"每条形如（锚点格 − 小剂量格）：{step}")
    print(f"  均值 {pair_step['mean_gap_imgs']:+.1f} 张，95% CI [{pair_step['ci95_imgs'][0]:+.1f}, "
          f"{pair_step['ci95_imgs'][1]:+.1f}]，σ {pair_step['sd_imgs']} 张，同号 {pair_step['same_sign_seeds']} 条"
          f"（二项 p={pair_step['p_seeds_binomial']}）")
    print(f"  污染说预测这一步 ≈ 0，H1 直线预测 {h1_line[ANC] - h1_line[small]:+.1f} 张 → "
          + ("CI 跨 0：这一段掉多少说不清，两说都没判决" if crosses else
             ("CI 不含 0 且为负：这一段真的在掉 ⇒ 阶梯不是平到锚点的，污染说在这一列站不住"
              if pair_step["mean_gap_imgs"] < 0 else
              "CI 不含 0 且为正：小剂量格比锚点格更差，先查仪器再谈形状")))

    # ---------- 判决：形状由数字挑，不由人挑 ----------
    s_small, s_probe = summary[small], summary[PROBE]
    all_same = f"{len(seeds)}/{len(seeds)}"
    if s_small["p_pooled"] < P_SIG and s_small["same_dir_seeds"] == all_same:
        shape = "污染型（有伤害不必有剂量）"
        reading = (f"α={small:g} 已经把方向摆出来（{all_same} 同号、合并 p={s_small['p_pooled']:.4g}、"
                   f"均值 {s_small['mean_gap_imgs']:+.1f} 张），而它是锚点剂量的 {small / ANC:.0%}"
                   f"——H1 的『按份额掺毒』在最后一块地上也没了：软标签一旦进目标函数，"
                   f"坏的是它**在不在**，不是它占多少")
    elif s_small["p_pooled"] >= P_SIG and s_probe["p_pooled"] < P_SIG:
        shape = "有阈值（H1 半活）"
        reading = (f"小剂量端（α≤{small:g}）配对检验不出方向（p={s_small['p_pooled']:.3g}），"
                   f"到 α={PROBE:g} 才出（p={s_probe['p_pooled']:.3g}）→ 存在一个 0~{PROBE:g} 之间的"
                   f"起点，之前测不出、之后基本饱和；说『伤害随 α 线性增长』仍然没有证据")
    else:
        shape = "测不动（这一发没判决）"
        reading = (f"两端都出不了方向（α={small:g} p={s_small['p_pooled']:.3g}、"
                   f"α={PROBE:g} p={s_probe['p_pooled']:.3g}）→ 本课的分辨率不足以判，别硬解")
    print(f"\n[判决] {shape}：{reading}")
    lo_step = round(s_probe["mean_gap_imgs"] - s_small["mean_gap_imgs"], 1)
    hi_step = round(anchor_mean - probe["mean_gap_imgs"], 1)
    share = round(abs(s_small["mean_gap_imgs"]) / abs(anchor_mean) * 100)
    print(f"[判决的分量] 阶梯第一格 α={small:g}（锚点剂量的 {small / ANC:.0%}）"
          f"就已经买到锚点伤害的 **{share}%**：{s_small['mean_gap_imgs']:+.1f} / {anchor_mean:+.1f} 张")
    if L["gate_is_anchor"]:
        print(f"[判决的边界] α={small:g}→{PROBE:g}（剂量 ×{PROBE / small:.0f}）再走 {lo_step:+.1f} 张；"
              f"往上没有第二格可看——闸格就是锚点格，所以这一发判的是『0~{PROBE:g} 这一段陡不陡』，"
              f"判不了『{PROBE:g} 以上还涨不涨』（那一问留在 MNIST 的 α={PROBE:g}→{ALPHA_ANCHOR:g} 上）")
    else:
        print(f"[判决的边界] 剂量往上再加：α={small:g}→{PROBE:g}（剂量 ×{PROBE / small:.0f}）"
              f"只再走 {lo_step:+.1f} 张；α={PROBE:g}→{ANC:g}（剂量 ×{ANC / PROBE:.2f}）"
              f"只再走 {hi_step:+.1f} 张"
              + (f" → 前一段的剂量倍数是后一段的 {PROBE / small / (ANC / PROBE):.2f} 倍，"
                 f"换来的增量却是它的 {abs(lo_step / hi_step):.1f} 倍：**曲线在第一个读数之后基本就平了**"
                 if hi_step else ""))

    under = [a for a, s in summary.items() if abs(s["mean_gap_imgs"]) <= floor]
    print(f"\n[自我审计① 判决单位是配对，不是底] "
          + (f"α∈{under} 各格的 |gap| 在底（{floor} 张）以下——这是**设计使然**："
             "H1 自己就预测它们小。所以本课的判决单位是逐图配对（同一副权重内比翻牌），"
             "不是跨重训的极差。反过来也成立：谁要是拿『不到底』来否掉这些格，"
             "也就是在否掉自己用来立锚点的那套配对检验。" if under else
             f"这一轮每一格都过了底（{floor} 张）——连 α={small:g} 那格都过了，"
             "所以本课不需要拿配对检验去救小读数；判决的力气全在第一格的**方向**上（同号 + 合并 p），"
             "不在它离底多远。"))
    print("[自我审计② 恒等闸只证明接线，不证明灵敏度] α=0 逐位等于 CE 说的是『α 确实乘在该乘的地方』；"
          "它不给仪器定分辨率。分辨率由每格与 CE 不同的预测图数给出（上面『差 xxx 张』那一列）——"
          "那一列如果本身就是几十张，任何『这一格与 CE 打平』的说法都只是没看见。")
    print(f"[自我审计③ 阶梯是 {len(seeds)} 条随机流 × 每流一次] 与第4课b/c 同病：每种子只训一次，"
          "跨种子的极差没有二次底。"
          + ("这一发已按第4课e 的办法扩到 8 条流：极差仍是粗估，但判决改吃配对步长的 CI。"
             if nseeds == 8 else "要把它当正式判决用，得先照第4课e 的办法扩到 8 条流。"))
    peers = {tt: m for tt, m in L["peers"].items() if tt != T}
    print(f"[自我审计④ 只在 T={T:g} 上扫 α] 第4课c 判过 T 才是主效应、α 是平的，"
          f"所以本阶梯是**在 T={T:g} 这一列上**看剂量。"
          + ("同一本档案里 " + "、".join(f"α={ANC:g}/T={tt:g} 那格 {m:+.1f} 张"
                                        for tt, m in peers.items())
             + " 本课没测——阶梯的形状结论不许往这些列搬。" if peers else
             f"这本账只有 T={T:g} 一列，无其它温度可对照。"))
    print(f"[自我审计⑤ 对账基准跟着档案走] 本课的 α={PROBE:g} 格与 {src} 同种子同数据同起点，"
          "所以能逐位对账；换了 --dataset / --tier / --width 之后这条闸连着换档案——"
          "拿 A 档的账对 B 档的数就是自欺。")
    if dataset != "mnist":
        print(f"[自我审计⑥ 换任务同时换了三样] 任务的难易、老师的干湿（Fashion 满档老师 "
              f"{L['teacher_acc']}，MNIST 那副 {TEACHER_ARCHIVE_ACC}）、以及那本账里可用的温度列。"
              "所以这一发判的是『污染型伤害在第二个任务上还在不在』，不是『污染型伤害与 MNIST 无关』。")

    report = {"dataset": dataset, "tier": tier, "width": width, "temps_fixed": T, "ladder": list(LAD),
              "seeds": list(seeds), "probe_alpha": PROBE, "anchor_alpha": ANC,
              "paired_step_small_to_anchor": pair_step,
              "t_column_override": t_col, "thin_ladder": bool(thin), "identity_gate_seeds": len(id_gate),
              "gate_is_anchor": L["gate_is_anchor"], "teacher_acc": L["teacher_acc"],
              "floor_from": L["floor_from"], "floor_kind": L["floor_kind"], "seed_floor_imgs": floor,
              "anchor_mean_from": L["anchor_from"], "anchor_mean_gap_imgs": anchor_mean,
              "predictions_ex_ante": {"H1_line_through_origin": h1_line,
                                      "contamination_pred": f"α={LAD[1]:g} 即接近 {anchor_mean:+.1f} 张"},
              "gate_alpha_zero_identity": id_gate, "gate_probe_vs_archive": probe_gate,
              "ladder_summary": {str(a): summary[a] for a in summary},
              "arms": {f"α{a:g} 种子{r['seed']}": r for a, rr in rows.items() for r in rr},
              "verdict": {"shape": shape, "reading": reading,
                          "first_cell_alpha": small,
                          "first_cell_share_of_anchor_damage_pct": share,
                          "step_low_imgs": lo_step, "step_low_dose_x": round(PROBE / small, 2),
                          "step_high_imgs": hi_step,
                          "step_high_dose_x": (None if L["gate_is_anchor"] else round(ANC / PROBE, 2))},
              "limitations": ["每种子每格只训一次：跨种子极差没有二次底" if nseeds == 3 else
                              f"{len(seeds)} 条流各训一次：这是把分辨率做精，新增的不是独立性",
                              f"只扫 T={T:g} 一列，其余温度上的剂量响应未测",
                              "小 α 各格均值本就低于底（H1 的预测就是这个量级），"
                              "判决依赖逐图配对检验而非重训抖动",
                              f"只 {tier}+S{width}：跨档/跨架构要各自带 --tier/--width 重跑，"
                              "档案不同不能混读"]
              + ([] if dataset == "mnist" else
                 ["换任务同时换了任务难度/老师干湿/那本账里可用的温度列：判的是『第二个任务上还在』，"
                  "不是『与 MNIST 无关』",
                  f"闸格就是锚点格（{src} 账里没有 α=0.3）：只剩恒等闸一条独立于锚点的对账线",
                  f"底是 {src} 该档自己的 own floor（{floor} 张），比 MNIST 两本账借来的 41/37 张宽"])}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{len(seeds) + len(seeds) * len([a for a in LAD if a > 0])}"
          f" + 恒等闸 {len(id_gate)} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="α→0 剂量曲线（默认 MNIST S128+满数据，T=3 一列）")
    ap.add_argument("--dataset", choices=tuple(DATASETS), default="mnist",
                    help="换任务：fashion 时温度列/锚点剂量/底全部从 compress_report_v6_fashion.json 现读")
    ap.add_argument("--width", type=int, choices=(128, 64), default=128)
    ap.add_argument("--nseeds", type=int, choices=(3, 8), default=3,
                    help="8 = 加第4课e 那五条随机流（老 3 条仍只当逐位对账的锚）；档案名加 _n8")
    ap.add_argument("--thin", action="store_true",
                    help="薄阶梯：只跑最小剂量格与锚点格两格（配对步长只吃这两格），"
                         "恒等闸只在老 3 条流上跑；档案名加 _thin")
    ap.add_argument("--t", type=float, choices=(1.0, 3.0, 6.0), default=None,
                    help="把阶梯铺在哪一列温度上；默认 MNIST=T=3、Fashion=该账最大温度列。"
                         "显式给值时档案名加 _t<值>（换列=换一本账，不能与默认那本撞名）")
    ap.add_argument("--tier", choices=tuple(DATA_TIERS), default=TIER,
                    help="数据档位；底与跨脚本闸基准都从那一档（那一本任务）自己的账里读，缺格就退出")
    args = ap.parse_args()
    main(args.dataset, args.width, args.tier, nseeds=args.nseeds, thin=args.thin, t_col=args.t)
