# -*- coding: utf-8 -*-
"""第4课的第三笔债：把「α=0.7、T=3 单点」这条边界，从怀疑变成判决。

第4课测到 KD 显著有害（−0.63pp，种子层也扛住了），但那一轮只取了一组超参，
limitations 里自己写着「α/T 未扫参」。这一课补齐，并且带着两个**互相打架**的假设去扫：

  H1 稀释说：软标签本身没错，错在 α=0.7 把 70% 的权重给了它、抢走了硬标签的信号。
     ⇒ 伤害 ∝ α（α=0.3 时约 −27 张），而 T 轴是平的（每列内部不分高低）。
  H2 空洞说：第4课 Q3a 量到老师的软标签≈one-hot（测试集 92% 的图 top1>0.99），
     KL 项里没有 CE 给不出的信息，纯粹是噪声源。温度才是唯一能把暗知识放出来的旋钮。
     ⇒ 伤害随 T↑ 收缩（T=6 时暗知识真正上桌），α 轴的斜率反而没那么要紧。

两说的可判别处就在 **α=1.0 那一列的 T 轴**：H1 说那一列平（都是"没用硬标签"），
H2 说那一列随 T 涨。顺带还能回答第4课留的那个"如果 α→0 伤害就消失"的反事实。

但扫参之前先要拆一颗地雷：α 和 T **不是正交的两个旋钮**。标准写法里 KL 项乘了 T²，
这个补偿因子只要不精确，升 T 就在同时改 KD 项的有效权重——那"温度越高越坏"里就掺着
"权重越偏越坏"，H1 会以借温度之名的形式混进 T 轴的读数里。所以本课先做一次零训练的
梯度范数探针（loss 两项各自 backward，看 KD 梯度随 T 漂多少），漂移出闸门就补一组
**配平臂**：每个温度用自己那个能把 KD 梯度钉回锚点水平的 α，再看 T 轴还斜不斜。

仪器规矩沿用前几课：同种子配对（CE 与 KD 每个种子一副权重同起点）、逐图符号检验、
幅度只跟**种子底**比（底不硬编码，从对应宽度的 compress_report_v4b*.json 读——锚点档案值、
H1 的伤害基数、种子底三样全都跟着宽度走，S64 那一份是 0.9224/776、−55 张、底 53 张）。

`--width 64` 是后补的一发：S128 那张表判出"伤害的旋钮是 T 不是 α"，但那只在一副学生上成立。
这一发问的是**这条归因跨不跨架构**（同一老师、同一批数据、同一份软标签，只换学生宽度；
所以同号=结论不随学生宽度变，不等于换个数据集还成立）。
`--tier 低数据2000` 补的是自我审计③ 欠的那一发：同一副学生、同一个老师，把训练数据从 8500 张
砍到 2000 张，问 **T 轴那条斜率换数据量还在不在**（结果：**反号了**——满数据档加热有害、
低数据档加热有益，于是这一发从"补漏"升成了本课最有意思的一条判决）。
`--tier 中数据4500` 是顺着那条判决追的一发：两个端点反号，中间只隔一次 4.25 倍的数据量变化，
**符号到底在哪一档翻**？只测两端等于在一条曲线上画了两个点就说知道它弯；补一个中点，
才分得清"随数据量单调变坏"和"两端不同、中间还有一次翻转"。
`--nseeds 8` 还的是**分辨率债**：这张 3×3 表一直只有 3 条随机流，而 3/3 全同号的双侧二项 p 恰好是
0.25——"无反例"不等于"显著"。扩到 8 条流（与第4课b/e/f 同一批种子，不许自创）之后，T 主效应
除了那个随流数机械变宽的极差型种子底，还多一条**按流配对**的 CI，能跨 3 条/8 两本账并读。
零 API 费，纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4c_alpha_T_sweep.py
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4c_alpha_T_sweep.py --width 64
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4c_alpha_T_sweep.py --tier 低数据2000
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4c_alpha_T_sweep.py --tier 中数据4500
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4c_alpha_T_sweep.py --nseeds 8
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (BS, DATA_TIERS, KD_ALPHA, KD_T, SEED, TIERS, VAL_N, build_mlp,
                           logits_of, sign_test_p, soft_label_audit, summarize, train_student)
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
TIER = "满数据8500"                                 # 默认档；--tier 低数据2000 兑现自我审计③ 欠的那一发
WIDTH = 128                                         # 默认副；已被第4课b 降级为"不显著"的低数据档不追
ALPHAS = (0.3, 0.7, 1.0)
TEMPS = (1.0, 3.0, 6.0)
SEEDS = (42, 7, 2026)
SEEDS8 = SEEDS + (11, 202, 3003, 40004, 55555)     # 第4课e/f 那批流：扩种子只能扩这一套，换一批就等于换一把尺
NSEEDS = 3                                          # 可被 --nseeds 覆盖
ALPHA_ANCHOR = KD_ALPHA                             # 0.7：第4课那格的权重分配
ANCHOR_T = KD_T                                     # 3.0：探针的基准温度，也是锚点格
T2_TOL = 0.15                                       # KD 梯度范数随 T 漂移超过 ±15% 就算 T 轴被污染


# 种子底档案的位置：**第4课那两档记在同一本合并账里**（step4b 默认跑法一次写两档，
# 换档名会读到一本不存在的档案，而 step4b 又拒绝单跑这两档——两头堵死）。
# 只有第4课没跑过的档（中数据4500）才有自己的一本。输出名不一样：v4c 每档各写各的表。
V4B_SUFFIX = {"中数据4500": "_mid"}
V4C_SUFFIX = {"低数据2000": "_low", "中数据4500": "_mid"}


def v4b_path(width: int, tier: str) -> Path:
    """这一档、这一副宽度的种子底档案。S64 只跑过满数据档，所以那本里没有档位后缀。"""
    if NSEEDS == 8:                                     # 8 条流那本底只覆盖 S128 的两档，别的组合直接拒
        if width != 128 or tier not in TIERS:
            raise SystemExit(f"--nseeds 8 的种子底档案只有 S128 × {list(TIERS)}（compress_report_v4b_n8.json）；"
                             f"要测 S{width} 或 {tier}，得先给那一副把那本底扩出来，不许借别档的底")
        return HERE / "compress_report_v4b_n8.json"
    stem = "compress_report_v4b" if width == 128 else f"compress_report_v4b_s{width}"
    if tier not in TIERS and tier not in V4B_SUFFIX:
        raise SystemExit(f"档 {tier} 没登记种子底档案名：它要么在第4课那两档的合并账里（那就别新加名字），"
                         f"要么有自己的 v4b 单档账（在这里登记后缀）。"
                         f"悄悄回落到合并账等于拿别的档的底来判这一档")
    # 档位后缀**不随宽度消失**：旧写法只在 width==128 时贴，写死的假设是"S64 只跑过满数据档"，
    # 于是 64×中数据会去读 v4b_s64.json（满数据那本）——那本里真有一个档对的臂，load_anchor 拦不到，
    # 结果是借别的档的底来判这一档的闸门。老路径逐字不变（64×满数据仍回落 v4b_s64.json）。
    return HERE / (stem + V4B_SUFFIX.get(tier, "") + ".json")


def out_path(width: int, tier: str = TIER) -> Path:
    stem = "compress_report_v4c" if width == 128 else f"compress_report_v4c_s{width}"
    # 流数是这本账的判别维度：不带 _n8 就会把板块一路在用的那本 3 条流的判决无声盖掉
    return HERE / (stem + V4C_SUFFIX.get(tier, "") + ("_n8" if NSEEDS == 8 else "") + ".json")


def load_anchor(v4b: Path, width: int, tier: str = TIER) -> dict:
    """锚点档案值 + 种子底 + H1 的伤害基数，三样全从第4课b 那一副宽度、那一档的报告读。

    为什么不写第二套字面量：换一副宽度这些数字**全都变**（S64 的锚是 0.9224/776/128/73、
    伤害基数 −55 张、底 53 张），换一档数据也全变（S128 低数据档的锚是 0.9072/928、
    伤害基数 +27 张——符号都反了），而"对账"这件事的效力完全来自它比的是档案真值。
    抄一份数字进代码，档案一改就对上老账了——所以档案是唯一来源，这里只负责读。"""
    d = json.loads(v4b.read_text(encoding="utf-8"))
    if NSEEDS == 8:
        # 借来的底必须先证明它没换尺子：v4b_n8 自己带那道跨脚本闸，这里读它，不在这里重算
        got = (d.get("cross_script_parity") or {}).get(f"{tier} 老3条流×4项读数")
        if got != "逐位相同":
            raise SystemExit(f"{v4b.name} 里 {tier} 的跨脚本对账不是「逐位相同」（读到 {got!r}）："
                             f"这本 8 条流的底与 3 条流那本不可并读，先查上游那一发")
    if int(d["width"]) != width:
        raise SystemExit(f"{v4b.name} 是 S{d['width']} 的报告，不是 S{width}——换宽度请连档案一起换")
    arm_key = f"{tier} 种子{SEEDS[0]}"
    if arm_key not in d["arms"]:
        raise SystemExit(f"{v4b.name} 里没有 {arm_key} 这一格（该档案只有 "
                         f"{'、'.join(sorted({k.split(' 种子')[0] for k in d['arms']}))}）——"
                         f"这一档没在这副宽度上造过底，换档位请连档案一起换回 {TIER}")
    arm = d["arms"][arm_key]
    return {"kd": {"acc": arm["acc_kd"], "wrong": arm["wrong_kd"],
                   "broken": arm["flips"]["broken"], "fixed": arm["flips"]["fixed"]},
            "ce_acc": arm["acc_ce"],
            "gap_imgs": arm["gap_imgs"],            # KD−CE，负=KD 更差，与本课 gap 同号向
            "floor": int(d["verdicts"][tier]["seed_floor_imgs"]["KD"]),
            "floor_ce": int(d["verdicts"][tier]["seed_floor_imgs"]["CE"]),
            # 8 条流那本底把锚点格的**逐流 gap** 也记着：那一列就是这一课锚点格的同批读数，
            # 逐位对上 ⇒ 两本账量的是同 8 条随机流，而不只是同一个种子 42
            "paired_gaps": ({"seeds": d["seeds"],
                            "gaps": d["verdicts"][tier]["gaps_imgs"],
                            "label": f"{v4b.name} verdicts.{tier}.gaps_imgs"}
                            if NSEEDS == 8 else None)}


def h1_pred(gap_imgs: int) -> dict:
    """H1 的可打数字：把锚点那一格的伤害按 α 线性缩放（α/α_anchor × gap）。"""
    return {a: round(a / ALPHA_ANCHOR * gap_imgs) for a in ALPHAS}


def key(a: float, t: float) -> str:
    return f"a{a:g}_T{t:g}"


def term_grad_norms(xd, yd, soft, t: float, width: int, seed: int = SEED) -> dict:
    """零训练探针：同一副初始权重、同一批数据，把 loss 两项各自 backward，
    量它们真正作用到学生参数上的梯度 L2 范数（KL 按标准写法带 T² 补偿）。

    为什么训练前必须量这个：α 和 T 不是正交的两个旋钮——T² 这个补偿因子只要不精确，
    升 T 就会顺带把 KD 项的**有效权重**抬起来。那样"温度越高越坏"里就掺着
    "权重分配越偏越坏"（H1 借温度之名发生），事后再也分不开。"""
    torch.manual_seed(seed)
    m = build_mlp(width)
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(xd), (BS,), generator=gen)
    z = m(xd[idx])
    ce = nn.functional.cross_entropy(z, yd[idx])
    kl = nn.functional.kl_div(torch.log_softmax(z / t, dim=1),
                              torch.softmax(soft[idx] / t, dim=1),
                              reduction="batchmean") * t ** 2
    ps = list(m.parameters())
    g_ce = torch.autograd.grad(ce, ps, retain_graph=True)
    g_kd = torch.autograd.grad(kl, ps)
    flat = lambda gs: float(torch.cat([g.flatten() for g in gs]).norm())   # noqa: E731
    return {"g_ce": round(flat(g_ce), 5), "g_kd_unit": round(flat(g_kd), 5)}


def main(width: int = WIDTH, tier: str = TIER) -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    v4b = v4b_path(width, tier)
    if not v4b.exists():
        hint = "python step4b_seed_variance.py" if tier == TIER else \
            f"python step4b_seed_variance.py --tier {tier}"
        raise SystemExit(f"缺 {v4b.name}——锚点档案值与**这一档自己的**种子底都从它读，先跑 "
                         + (hint + ("" if width == 128 else " --width 64")))
    out = out_path(width, tier)
    anchor = load_anchor(v4b, width, tier)
    floor = anchor["floor"]
    ANCHOR_KD, ANCHOR_CE_ACC = anchor["kd"], anchor["ce_acc"]
    H1_PRED = h1_pred(anchor["gap_imgs"])

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    n_data = DATA_TIERS[tier]
    xd, yd = xt_all[:n_data], torch.tensor(yt_all[:n_data])
    soft = logits_of(teacher, xd)                   # 软标签与 T 无关：温度在 loss 里加，全程共用一份
    seeds = SEEDS if NSEEDS == 3 else SEEDS8

    print(f"teacher+数据就位 | {tier} 训练 {n_data} 张 | 测试 {len(yte)} | 学生 S{width} | "
          f"网格 {len(ALPHAS)}×{len(TEMPS)}×{len(seeds)} = "
          f"{len(ALPHAS) * len(TEMPS) * len(seeds)} 次 KD + {len(seeds)} 次 CE 对照")
    print(f"[仪器] 锚点档案值来自 {v4b.name}：α={ALPHA_ANCHOR}/T={ANCHOR_T:g}/种子{SEEDS[0]} "
          f"KD {ANCHOR_KD['acc']}/错{ANCHOR_KD['wrong']}/坏{ANCHOR_KD['broken']}/正{ANCHOR_KD['fixed']}，"
          f"CE {ANCHOR_CE_ACC}，那一格 gap {anchor['gap_imgs']:+d} 张（H1 按 α 线性缩放它）")
    if anchor["gap_imgs"] > 0:
        print(f"       ⚠ 这一档的锚点本身是**正向**的（KD 比 CE 少错 {anchor['gap_imgs']} 张），"
              f"所以 H1 的三个预测全带正号。但别把符号读成疗效：第4课b 已经把这一格降级为不显著"
              f"（幅度只有该档 KD 种子底 {floor} 张的 {abs(anchor['gap_imgs']) / floor:.2f}x）。"
              f"本课在低数据档要判的是「T 轴的斜率形状换档还在不在」，不是「KD 有没有用」。")
    print(f"[仪器] 种子底（第4课b 的 KD 臂，换随机流的错图数极差）= {floor} 张，"
          f"任何 |均值 gap| ≤ {floor} 张在本机上不算疗效")
    if anchor["floor_ce"] > floor:
        strict = anchor["floor_ce"]
        print(f"       ⚠ **这一档的闸比前三档松**：被门控的量是 KD−CE 之差，可这一档 CE 臂自己的底"
              f"（{anchor['floor_ce']} 张）比 KD 臂（{floor} 张）大——三档里第一次出现。"
              f"沿用 KD 底是为了和三份已入库的 v4c 表可比，那就得把话说清楚："
              f"落在 {floor}~{strict} 张之间的读数，换个更保守的底就翻，不能当疗效。")

    # ---------- 零训练先验：三个温度下老师的软标签到底"软"了多少 ----------
    print("\n[零训练先验] 老师在拟合集上的软标签体检，逐温度：")
    audit_by_t = {}
    for t in TEMPS:
        a = soft_label_audit(soft, t=t)
        audit_by_t[key(1.0, t)] = {"T": t, "原始": a["原始"], "除以T": a["除以T"]}
        o, s = a["原始"], a["除以T"]
        print(f"  T={t:<4} 原始 top1 {o['mean_top1']:.4f}（塌缩 {o['pct_top1_gt_0.99']}%，p2 {o['mean_p2']:.4f}）"
              f" → /T 后 top1 {s['mean_top1']:.4f}（塌缩 {s['pct_top1_gt_0.99']}%，p2 {s['mean_p2']:.4f}）")
    lo, hi = audit_by_t[key(1.0, 1.0)]["除以T"], audit_by_t[key(1.0, TEMPS[-1])]["除以T"]
    print(f"  → 从 T=1 到 T={TEMPS[-1]:g}：top1 {lo['mean_top1']:.4f}→{hi['mean_top1']:.4f}，"
          f"塌缩率 {lo['pct_top1_gt_0.99']}%→{hi['pct_top1_gt_0.99']}%，"
          f"第二名概率 {lo['mean_p2']:.4f}→{hi['mean_p2']:.4f}"
          f"（KL 能拿到的「暗知识」总量就这么多，先记着）")

    # ---------- 零训练探针：T 是温度轴，还是偷偷的权重轴？ ----------
    assert ALPHA_ANCHOR in ALPHAS and ANCHOR_T in TEMPS, "锚点格不在网格里，探针没有基准"
    probe = {t: term_grad_norms(xd, yd, soft, t, width) for t in TEMPS}
    ref_g = probe[ANCHOR_T]["g_kd_unit"] * ALPHA_ANCHOR      # 第4课那格真正推参数的 KD 梯度
    print("\n[零训练探针] 同一副初始权重、同一批数据，loss 两项各自 backward 的梯度 L2 范数：")
    for t in TEMPS:
        p = probe[t]
        g_kd = ALPHA_ANCHOR * p["g_kd_unit"]
        share = g_kd / (g_kd + (1 - ALPHA_ANCHOR) * p["g_ce"])
        p["kd_share_at_anchor_alpha"] = round(share, 3)
        p["g_kd_drift_vs_anchor_pct"] = round((ALPHA_ANCHOR * p["g_kd_unit"] / ref_g - 1) * 100, 1)
        p["alpha_to_match_anchor"] = round(ref_g / p["g_kd_unit"], 3)
        print(f"  T={t:<4} ‖∇CE‖ {p['g_ce']:.4f} | ‖∇(T²·KL)‖ {p['g_kd_unit']:.4f}"
              f"（α 取 {ALPHA_ANCHOR} 时是锚点的 {p['g_kd_drift_vs_anchor_pct']:+.1f}%）"
              f" | KD 份额 {share:.1%} | 把 KD 梯度钉回锚点需要 α={p['alpha_to_match_anchor']}")
    drift = max(abs(probe[t]["g_kd_drift_vs_anchor_pct"]) for t in TEMPS) / 100
    # α>1 会给硬标签项负权重，那是另一个实验而不是对照——配不平的温度直接排除
    for t in TEMPS:
        probe[t]["matchable"] = bool(0.0 <= probe[t]["alpha_to_match_anchor"] <= 1.0)
    unmatched = [t for t in TEMPS if not probe[t]["matchable"]]
    matched = ([t for t in TEMPS if probe[t]["matchable"]
                and abs(probe[t]["alpha_to_match_anchor"] - ALPHA_ANCHOR) > 0.05]
               if drift > T2_TOL else [])
    print(f"  → T² 补偿的精确度：网格内 KD 梯度最大漂移 {drift:.1%}（闸门 ±{T2_TOL:.0%}）→ "
          + ("T 轴是干净的单变量，不必配平" if not matched
             else f"T 轴同时是条权重轴，加跑配平臂 {matched}：每 T 用自己的 α 把 KD 梯度钉回锚点水平")
          + (f"；配不平（需要 α>1）的温度：{unmatched}" if unmatched else ""))

    # ---------- 训练之前把两说的数字写死，之后不许改口 ----------
    pred_lines = []
    for a in ALPHAS:
        h1 = H1_PRED[a]
        pred_lines.append({"alpha": a, "h1_gap_imgs": h1,
                           "h1_expect_T_flat": f"该 α 三个 T 之差 ≤ 种子底 {floor} 张"})
    print("\n[先立预言（此后不许改口）]")
    print(f"  H1 稀释说 → α 轴斜率：{pred_lines}")
    print(f"  H1 还说 α=1.0 那一列**平**：T=1 与 T=6 的 gap 差 ≤ {floor} 张（都没用硬标签，一样惨）")
    print(f"  H2 空洞说 → α=1.0 那一列**随 T 涨**：T={TEMPS[-1]:g} 比 T=1 至少收回 {floor} 张以上；"
          f"且 α 轴不该更陡（既然锅在软标签空洞，给多少权重都是噪声）")
    print(f"  两说的公共判决点：α={ALPHAS[0]} T=3 若 |gap| < {floor} 张 → 伤害主要来自权重分配（H1 侧）")
    if matched:
        print(f"  配平臂（每个漂移出闸门的温度用自己的 α 把 KD 梯度钉回锚点水平，T={matched}）："
              f"钉回去若把该温度相对锚点的**额外伤害**收回超过一个底（{floor} 张）"
              f"→ 第4课读到的高 T 之害主要是 T² 把 KD 项抬了权重，H1 以「温度伪装成权重」的形式复活；"
              f"收不回来（≤ {floor} 张）→ 权重解释不了，伤害跟着温度本身走")

    # ---------- 各臂：CE 是同种子的锚，每个 KD 格逐格配对 ----------
    specs = [(key(a, t), a, t) for a in ALPHAS for t in TEMPS]
    specs += [(f"match_T{t:g}", probe[t]["alpha_to_match_anchor"], t) for t in matched]
    ce_ref: dict[int, dict] = {}
    cells: dict[str, list[dict]] = {k: [] for k, _, _ in specs}
    print(f"\n[开跑] 每个种子一副 CE + {len(specs)} 副 KD（同种子同初始权重、同批次流，只差目标函数）：")
    for seed in seeds:
        ce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
        ce_pred = predict(ce, xte, False)
        ce_wrong = int((ce_pred != yte).sum())
        ce_ref[seed] = {"acc": round(1 - ce_wrong / len(yte), 4), "wrong": ce_wrong}
        line = [f"  种子 {seed:<5} CE 错 {ce_wrong} 张"]
        mline = []
        for k, a, t in specs:
            m, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=a, t=t)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            broken = int(((ce_pred == yte) & (p != yte)).sum())
            fixed = int(((ce_pred != yte) & (p == yte)).sum())
            cells[k].append({"seed": seed, "alpha": a, "T": t,
                             "wrong_kd": wrong, "wrong_ce": ce_wrong,
                             "acc_kd": round(1 - wrong / len(yte), 4),
                             "gap_imgs": ce_wrong - wrong,          # 正=KD 比 CE 少错
                             "broken": broken, "fixed": fixed,
                             "p_sign": round(sign_test_p(broken, fixed), 6)})
            (mline if k.startswith("match_") else line).append(f"{k}:{ce_wrong - wrong:+d}")
        print(" ".join(line))
        if mline:
            print(" " * 12 + "（配平臂）" + "  ".join(mline))
        if seed == seeds[0]:
            r = cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]
            ok = (abs(r["acc_kd"] - ANCHOR_KD["acc"]) < 5e-5 and r["wrong_kd"] == ANCHOR_KD["wrong"]
                  and r["broken"] == ANCHOR_KD["broken"] and r["fixed"] == ANCHOR_KD["fixed"]
                  and ce_ref[seed]["acc"] == ANCHOR_CE_ACC)
            print(f"  [锚点逐位对账 α={ALPHA_ANCHOR} T={ANCHOR_T} 种子{SEEDS[0]}] KD {r['acc_kd']:.4f}"
                  f"/错{r['wrong_kd']}/坏{r['broken']}/正{r['fixed']} vs 档案 {ANCHOR_KD} → "
                  + ("通过：本课的 loss 就是第4课那个 loss，只是多了两个旋钮" if ok else "**不通过：全部结论作废**"))

    # ---------- 汇总与判决（分支算出来的，不是写死的） ----------
    grid = {}
    for k, rows in cells.items():
        gaps = [r["gap_imgs"] for r in rows]
        tb = sum(r["broken"] for r in rows)
        tf = sum(r["fixed"] for r in rows)
        grid[k] = {"per_seed": rows, "gaps_imgs": gaps,
                   "mean_gap_imgs": round(sum(gaps) / len(gaps), 1),
                   "range_imgs": max(gaps) - min(gaps),
                   "same_dir_as_h1": sum(1 for g in gaps if g < 0),
                   "pooled": {"broken": tb, "fixed": tf},
                   "p_pooled": round(sign_test_p(tb, tf), 8)}
    at = lambda a, t: grid[key(a, t)]["mean_gap_imgs"]      # noqa: E731

    # ---------- 跨脚本**逐流**对账（零额外训练，只在借了 8 条流那本底的一发可做） ----------
    pg = anchor.get("paired_gaps")
    if pg:
        mine = [r["gap_imgs"] for r in cells[key(ALPHA_ANCHOR, ANCHOR_T)]]
        same_flows = pg["seeds"] == list(seeds)
        ok = same_flows and mine == pg["gaps"]
        print(f"\n[跨脚本逐流对账 锚点格 α={ALPHA_ANCHOR}/T={ANCHOR_T:g}] 本课 {mine} vs {pg['label']} {pg['gaps']}"
              f" → " + ("逐位相同：这两本账量的是**同 8 条随机流**，「换流数」这一维对得上尺子"
                        if ok else
                        f"**不同**（流序一致={same_flows}）：两本账不是同一批流，"
                        "与 v4b_n8 并读的一切结论作废"))
        gap_parity = {"this_run_gaps": mine, "archive_gaps": pg["gaps"],
                      "flow_order_matches": same_flows, "from": pg["label"],
                      "verdict": "逐位相同" if ok else "不同：作废"}
    else:
        gap_parity = {"verdict": "这一发没有可借的逐流账（3 条流那本底只记极差，不记逐流 gap）",
                      "flow_order_matches": None}
        print("\n[跨脚本逐流对账] 跳过：3 条流那本 v4b 底不记逐流 gap，只有恒等闸与锚点单格对账。")

    print(f"\n[网格] 每格{len(seeds)} 条流的 KD−CE 平均（张，负=KD 更差）；括号内是该格跨种子极差：")
    print("        T=" + "        ".join(f"{t:<7g}" for t in TEMPS))
    for a in ALPHAS:
        print(f"  α={a:<4} " + "        ".join(f"{at(a, t):+6.1f}({grid[key(a, t)]['range_imgs']:2d})"
                                              for t in TEMPS))

    verdicts: dict[str, dict | str] = {}
    a_max = ALPHAS[-1]
    h1_resid = [abs(at(a, ANCHOR_T) - H1_PRED[a]) for a in ALPHAS]
    alpha_slope = at(a_max, ANCHOR_T) - at(ALPHAS[0], ANCHOR_T)   # α 从 0.3 到 1.0，gap 往哪走
    verdicts["H1_alpha_axis"] = {
        "pred": {str(a): H1_PRED[a] for a in ALPHAS},
        "measured_anchor_T": {str(a): at(a, ANCHOR_T) for a in ALPHAS},
        "alpha_slope_imgs": round(alpha_slope, 1),
        "abs_error_vs_pred": [round(x, 1) for x in h1_resid],
        "monotone_worse_with_alpha": bool(at(ALPHAS[0], ANCHOR_T) > at(ALPHA_ANCHOR, ANCHOR_T)
                                          > at(a_max, ANCHOR_T)),
        "slope_beats_seed_floor": bool(abs(alpha_slope) > floor)}
    col_t = [at(a_max, t) for t in TEMPS]
    verdicts["H1_or_H2_alpha_max_T_column"] = {
        "measured": {str(t): at(a_max, t) for t in TEMPS},
        "T_span_imgs": round(max(col_t) - min(col_t), 1),
        "T_max_minus_T_min_imgs": round(at(a_max, TEMPS[-1]) - at(a_max, TEMPS[0]), 1),
        "flat_as_H1_says": bool(abs(at(a_max, TEMPS[-1]) - at(a_max, TEMPS[0])) <= floor),
        "rises_as_H2_says": bool(at(a_max, TEMPS[-1]) - at(a_max, TEMPS[0]) > floor)}
    t_span_all = [round(sum(at(a, TEMPS[-1]) for a in ALPHAS) / len(ALPHAS)
                        - sum(at(a, TEMPS[0]) for a in ALPHAS) / len(ALPHAS), 1)]
    # 同一个量的**配对**读法：每条流先自己「高温 − 低温」（同一条流的两格相减，CE 锚自动抵消），
    # 再按 α 平均。为什么要它：种子底是 max−min 型统计量，流数一多就机械变宽，
    # 所以下面那个 beats_seed_floor 不能用来跨流数确认或推翻判决——能跨的只有这条配对 CI。
    per_flow_t = [round(sum(grid[key(a, TEMPS[-1])]["per_seed"][i]["gap_imgs"]
                            - grid[key(a, TEMPS[0])]["per_seed"][i]["gap_imgs"]
                            for a in ALPHAS) / len(ALPHAS), 1) for i in range(len(seeds))]
    t_paired = summarize(per_flow_t, f"T{TEMPS[-1]:g}−T{TEMPS[0]:g}（每流先配对再按 α 平均）")
    # T 轴的符号在两档是反的：满数据档加热有害（锚点 gap<0，读"伤害"），低数据档加热有益
    # （gap>0，读"收益"）。写死"伤害"会让低数据档把最大的一步挑反（挑成最小的那个正号）。
    axis_noun = "伤害" if at(ALPHA_ANCHOR, ANCHOR_T) < 0 else "收益"
    steps = {}
    for t_lo, t_hi in zip(TEMPS[:-1], TEMPS[1:]):
        d = round(sum(at(a, t_hi) - at(a, t_lo) for a in ALPHAS) / len(ALPHAS), 1)
        steps[f"{t_lo:g}→{t_hi:g}"] = {"imgs": d, "in_seed_floor_units": round(abs(d) / floor, 2)}
    pick = max if t_span_all[0] > 0 else min
    verdicts["T_main_effect"] = {"T_max_minus_T_min_pooled_over_alpha": t_span_all[0],
                                 "beats_seed_floor": bool(abs(t_span_all[0]) > floor),
                                 "seed_floor_gate_is_n_dependent": (
                                     f"底 {floor} 张是 {len(seeds)} 条流的 max−min，换流数不可比"
                                     f"（{v4b.name}）；跨流数只能读 paired_per_flow"),
                                 "paired_per_flow": t_paired,
                                 "adjacent_steps": steps, "axis_noun": axis_noun,
                                 "where_the_effect_lives": pick(steps, key=lambda k: steps[k]["imgs"])}
    v_pure = at(a_max, TEMPS[0])
    verdicts["pure_KD_low_T_vs_CE"] = {"mean_gap_imgs": v_pure,
                                       "vs_floor": round(abs(v_pure) / floor, 2),
                                       "sign": "KD 更好" if v_pure > 0 else "CE 更好",
                                       "counts_as_effect": bool(abs(v_pure) > floor)}
    v_low = at(ALPHAS[0], ANCHOR_T)
    low_paired = summarize([r["gap_imgs"] for r in cells[key(ALPHAS[0], ANCHOR_T)]],
                           f"α={ALPHAS[0]:g}/T={ANCHOR_T:g} 按流配对")
    ci_low = low_paired["ci95_imgs"]
    low_floor, low_ci = bool(abs(v_low) > floor), bool(not (ci_low[0] <= 0 <= ci_low[1]))
    if axis_noun != "伤害":
        low_reading = (f"这句反事实在本档用不上：锚点格自身 {at(ALPHA_ANCHOR, ANCHOR_T):+.1f} 张"
                       f"也在底（{floor} 张）以内，α={ALPHAS[0]:g} 与 α={ALPHA_ANCHOR:g} 都还没和纯 CE 分出胜负，"
                       "所以「往 α→0 走会怎样」在本档是两个不显著读数之间的比较")
    elif low_floor and low_ci:
        low_reading = "α→0 方向上伤害没消失（两把尺同判）"
    elif low_ci:
        low_reading = (f"**两把尺打架，而判决站在配对那一侧**：极差型底（{floor} 张，随流数机械变宽）"
                       f"说 {abs(v_low)} 张没穿过，但按流配对 {low_paired['mean_gap_imgs']:+.1f} 张 "
                       f"CI [{ci_low[0]:+.1f}, {ci_low[1]:+.1f}]、{low_paired['same_sign_seeds']} 条流同号不含 0 "
                       f"⇒ 伤害没消失；「α→0 伤害就消失」这句反事实仍然不成立。"
                       f"（3 条流那本底只有 {floor} 张的一半量级，同一个读数当时判的是「穿过」）"
                       if NSEEDS == 8 else f"α→0 方向上伤害没消失（底没穿过但 CI 不含 0：{floor} 张这把尺太粗）")
    elif low_floor:
        low_reading = (f"穿过了底但配对 CI 含 0：底是随流数变宽的粗尺、CI 是同一条流内相减的细尺，"
                       "两把不同意时按 CI 读 ⇒ 这一格只能说「说不清」")
    else:
        low_reading = "α→0 伤害消失（两把尺同判）"
    verdicts["alpha_03_at_anchor_T"] = {
        "mean_gap_imgs": v_low, "vs_floor": round(abs(v_low) / floor, 2),
        "survives_seed": low_floor, "paired_per_flow": low_paired,
        "paired_ci_excludes_zero": low_ci, "yardsticks_agree": bool(low_floor == low_ci),
        "reading": low_reading}
    free = {k: g for k, g in grid.items() if not k.startswith("match_")}
    best = max(free, key=lambda k: free[k]["mean_gap_imgs"])
    worst = min(free, key=lambda k: free[k]["mean_gap_imgs"])
    beating = [k for k, g in free.items() if g["mean_gap_imgs"] > 0]
    verdicts["grid_extremes"] = {"best": {"cell": best, "mean_gap_imgs": grid[best]["mean_gap_imgs"]},
                                 "worst": {"cell": worst, "mean_gap_imgs": grid[worst]["mean_gap_imgs"]},
                                 "cells_beating_ce": sorted(beating),
                                 "n_cells": len(free)}
    if matched:
        # 配平判的是"把 KD 梯度钉回锚点后，这一档自己的伤害还剩下多少"——
        # 不是"配平各档之间的跨度"（漂移出闸门的往往只有一档，跨度恒为 0 会读成"压平"）
        per_T = {str(t): {"alpha_matched": probe[t]["alpha_to_match_anchor"],
                          "gap_at_anchor_alpha": at(ALPHA_ANCHOR, t),
                          "gap_gradient_matched": grid[f"match_T{t:g}"]["mean_gap_imgs"],
                          "damage_recovered_imgs": round(grid[f"match_T{t:g}"]["mean_gap_imgs"]
                                                         - at(ALPHA_ANCHOR, t), 1)}
                 for t in matched}
        recovered = round(sum(v["damage_recovered_imgs"] for v in per_T.values()) / len(per_T), 1)
        if axis_noun == "伤害":
            explains, reading = bool(recovered > floor), (
                ("钉回梯度后伤害大部分回来了（回收 "
                 f"{recovered} 张 > 底 {floor} 张）→ 高 T 之害主要来自 T² 抬权重，H1 借温度复活"
                 if recovered > floor else
                 f"钉回梯度只换回 {recovered} 张（底 {floor} 张）→ 权重解释不了这档伤害，"
                 "锅在温度本身：软化的目标分布真把学生带偏了"))
        else:
            explains, reading = bool(abs(recovered) > floor), (
                f"本档 T 轴是{axis_noun}：把 KD 梯度钉回锚点（降 α）后收益变化 {recovered:+.1f} 张，"
                f"幅度 {abs(recovered)} 张 > 底 {floor} 张 → 这档收益里确实有一部分是「T² 顺带抬了权重」"
                "买来的，温度与权重在本档没完全分开" if abs(recovered) > floor else
                f"本档 T 轴是{axis_noun}：钉回梯度（降 α）后收益只动了 {recovered:+.1f} 张，"
                f"在底（{floor} 张）以内 → 收益也不是「T² 顺带抬权重」买来的：把权重降回去，收益没跟着降；"
                "它跟着温度本身走——低数据下真被摊软的目标分布确实给学生带了东西")
        verdicts["weight_or_temperature"] = {
            "per_T": per_T, "mean_damage_recovered_imgs": recovered,
            "axis_noun": axis_noun, "weight_explains_the_T_dose": explains, "reading": reading}
    else:
        verdicts["weight_or_temperature"] = {
            "max_kd_grad_drift_pct": round(max(abs(probe[t]["g_kd_drift_vs_anchor_pct"])
                                                for t in TEMPS), 1),
            "gate_pct": round(T2_TOL * 100, 1),
            "reading": "T² 补偿在闸门内，T 轴是干净的单变量：网格里的 T 斜率就是温度的作用"}

    print("\n[判决]")
    v = verdicts["H1_alpha_axis"]
    print(f"  ① α 轴（H1 的主张）：α={ALPHAS} 在锚点温度实测 {v['measured_anchor_T']}，"
          f"预测 {v['pred']}，|误差| {v['abs_error_vs_pred']}；"
          f"单调变坏 {v['monotone_worse_with_alpha']}，斜率 {v['alpha_slope_imgs']} 张"
          f"（种子底 {floor} 张）→ {'α 轴是真信号' if v['slope_beats_seed_floor'] else 'α 轴在底以下，谈不上斜率'}")
    v = verdicts["H1_or_H2_alpha_max_T_column"]
    print(f"  ② **判别格** α={a_max} 那一列 {v['measured']}：最高温比最低温差 {v['T_max_minus_T_min_imgs']} 张，"
          f"列内极差 {v['T_span_imgs']} 张 → 平（H1 赢）={v['flat_as_H1_says']}／随 T 涨（H2 赢）={v['rises_as_H2_says']}")
    v = verdicts["T_main_effect"]
    print(f"  ③ T 轴主效应（{len(ALPHAS)} 列平均）：高温 − 低温 = {v['T_max_minus_T_min_pooled_over_alpha']} 张，"
          f"压过种子底={v['beats_seed_floor']}；逐档走一步："
          + "、".join(f"{k} {d['imgs']:+.1f} 张（{d['in_seed_floor_units']}x 底）"
                      for k, d in v["adjacent_steps"].items())
          + f" → {v['axis_noun']}主要住在 {v['where_the_effect_lives']} 这一步")
    p = v["paired_per_flow"]
    print(f"     ③ b 同一量的**按流配对**读法（{p['n_seeds']} 条流，CE 锚在同一条流内相减自动抵消）："
          f"{p['mean_gap_imgs']:+.1f} 张 CI95 [{p['ci95_imgs'][0]:+.1f}, {p['ci95_imgs'][1]:+.1f}]"
          f"（SD {p['sd_imgs']}）、同号 {p['same_sign_seeds']}、流级二项 p={p['p_seeds_binomial']}"
          f" → CI 不含 0 = {not (p['ci95_imgs'][0] <= 0 <= p['ci95_imgs'][1])}。"
          "种子底那条闸门随流数变宽，只在本表内部可比；跨流数（3 条 vs 8 条）只能读这一条。")
    v = verdicts["pure_KD_low_T_vs_CE"]
    print(f"  ④ α={a_max}/T={TEMPS[0]:g}（只用老师 argmax 的软标签、不加热）：{v['sign']}，gap {v['mean_gap_imgs']} 张 = "
          f"种子底的 {v['vs_floor']}x，算疗效={v['counts_as_effect']}")
    v = verdicts["alpha_03_at_anchor_T"]
    print(f"  ⑤ 第4课那句反事实「α→0 伤害就消失」：α={ALPHAS[0]}/T={ANCHOR_T:g} 实测 {v['mean_gap_imgs']} 张 = 底的 "
          f"{v['vs_floor']}x → {v['reading']}")
    v = verdicts["grid_extremes"]
    print(f"  ⑥ {v['n_cells']} 格里最优点 {v['best']['cell']}（{v['best']['mean_gap_imgs']:+.1f} 张）、"
          f"最差 {v['worst']['cell']}（{v['worst']['mean_gap_imgs']:+.1f} 张）；"
          f"跑赢 CE 的格数 {len(v['cells_beating_ce'])}/{v['n_cells']}"
          f"{'' if not v['cells_beating_ce'] else '：' + '、'.join(v['cells_beating_ce'])}")
    v = verdicts["weight_or_temperature"]
    if "per_T" in v:
        print(f"  ⑦ 配平臂（α 由探针算出，把 KD 梯度钉回锚点水平）："
              + "；".join(f"T={k} α={d['alpha_matched']}：α={ALPHA_ANCHOR} 时 {d['gap_at_anchor_alpha']:+.1f} 张 → "
                          f"配平后 {d['gap_gradient_matched']:+.1f} 张（换回 {d['damage_recovered_imgs']:+.1f}）"
                          for k, d in v["per_T"].items())
              + f" → {v['reading']}")
    else:
        print(f"  ⑦ 权重/温度解耦探针：KD 梯度随 T 最大漂移 {v['max_kd_grad_drift_pct']}%"
              f"（闸门 ±{v['gate_pct']}%）→ {v['reading']}")

    print("\n[自我审计① 底是别课的] 种子底取自第4课b 的 KD 臂（α=0.7/T=3），本课改的是目标函数的两个系数，"
          "严格说各格自己的种子底要各扫各的；这里用一格的底当全表的闸，是**借来的仪器**。"
          "方向本课当时猜「低估 T 大端方差」，第4课d/e 已实测：猜反了——借来的底对**单格读数偏大**"
          "（换成每格自己的 ±2SE，T=1 两格立刻翻成显著），而那个翻档在扩到 8 条随机流后又消失了 ⇒ "
          "借来的底当闸这轮是对的，但「谁埋了谁」得先量再猜（±2SE 按独立合成是下界，见第4课d limitations）。")
    print(f"[自我审计② 多重比较] {len(free)} 格 × {len(seeds)} 条流，p_pooled 只看方向不看在底之上（同一批 10000 张测试图，"
          f"种子之间翻牌事件不独立，p 偏乐观）。全表只有一件事是硬判决：{floor} 张这条线。")
    if tier == TIER:
        others = [t for t in DATA_TIERS if t != tier]
        print(f"[自我审计③ 只扫了 {tier} 与 S{width}] 其余档位 {others} 各自加 `--tier` 补扫；"
              "补扫前那一档必须先有自己的种子底（step4b --tier 造），拿本档的底去判别档的读数，"
              "等于用一把不知道刻度的尺量东西。")
    else:
        g, strict = anchor["gap_imgs"], max(floor, anchor["floor_ce"])
        print(f"[自我审计③ 这一发扫的是 {tier}] 该档锚点格在第4课b 只有 {len(seeds)} 条随机流，"
              f"幅度 {g:+d} 张；这一档自己的底 = KD {floor} / CE {anchor['floor_ce']} 张 → |gap| 是它们的 "
              f"{abs(g) / floor:.2f}x / {abs(g) / strict:.2f}x。"
              + ("压在底以下：本档只能读「T 轴形状换档后还在不在」，不能读成「KD 在这一档有疗效」。"
                 if abs(g) <= floor else
                 "过了 KD 底、没过更保守的 CE 底：方向可以读，**幅度得按会被底翻掉的量报**——"
                 "换把保守的尺就不显著。" if abs(g) <= strict else
                 "两把底都过：这一档的锚点读数是硬的。"))
    if width != WIDTH:
        print(f"[自我审计③ b 跨架构 ≠ 独立复证] 这一副 S{width} 与默认那副 S{WIDTH} 共用**同一个老师、同一批数据、"
              "同一份软标签**，只有学生那侧换了。所以两份报告若同号，说明的是「结论不随学生宽度变」，"
              "而不是「换个数据集还成立」——数据那侧的独立性要到第5课（另一副 teacher）才有。")
    print("[自我审计④ 探针只管到梯度幅值] 权重/温度解耦靠的是初始那一步、一个批次、一副初始权重上"
          "两项梯度的 L2 范数之比；训练过程中两项的相对尺度会随学生逼近老师而变（KL 变小），"
          "所以配平是**起手配平**，不是全程配平。")
    print(f"[自我审计⑤ 探针顺手量到 T² 补偿过头了] KD 项梯度范数随 T："
          + " → ".join(f"T{t:g} {probe[t]['g_kd_unit']:.4f}" for t in TEMPS)
          + f"（锚点 α={ALPHA_ANCHOR} 下占 {probe[TEMPS[0]]['kd_share_at_anchor_alpha']:.0%} → "
          f"{probe[TEMPS[-1]]['kd_share_at_anchor_alpha']:.0%}）——教科书说「α 管权重、T 管温度」，"
          "在这个实现里不成立：升 T 确实在抬 KD 的实际话语权。低温端反而比锚点低，"
          f"α≤1 配不平（需要 α={probe[TEMPS[0]]['alpha_to_match_anchor']}），"
          "这一格按「不可配平」记账而不是悄悄删掉。")

    other = 64 if width == 128 else 128
    report = {"tier": tier, "width": width, "alphas": list(ALPHAS), "temps": list(TEMPS),
              "seeds": list(seeds), "n_seeds_arg": NSEEDS, "test_n": int(len(yte)), "train_fit_n": cut,
              "anchor_hyperparams": {"alpha": ALPHA_ANCHOR, "T": ANCHOR_T},
              "anchor_archive_from": v4b.name,
              "seed_floor_imgs_from_v4b": floor,
              "seed_floor_imgs_detail": {"gate_used_kd_arm": floor, "ce_arm": anchor["floor_ce"],
                                         "note": ("被门控量是 KD−CE 之差；CE 底更大时，"
                                                  "介于两数之间的读数按保守底就不显著")},
              "soft_label_audit_by_T": audit_by_t,
              "gradient_probe": {"t2_tolerance_pct": round(T2_TOL * 100, 1),
                                 "matched_temps": matched, "unmatchable_temps": unmatched,
                                 "by_T": probe},
              "predictions_ex_ante": {"H1_alpha_axis": pred_lines,
                                      "H1_alpha_max_T_column_flat": True,
                                      "H2_alpha_max_T_column_rises": True,
                                      "H1_pred_gap_imgs": {str(a): H1_PRED[a] for a in ALPHAS}},
              "parity_anchor": {"this_run": {"acc": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["acc_kd"],
                                             "wrong": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["wrong_kd"],
                                             "broken": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["broken"],
                                             "fixed": cells[key(ALPHA_ANCHOR, ANCHOR_T)][0]["fixed"]},
                                "archive": ANCHOR_KD,
                                "ce_acc_this_run": ce_ref[SEEDS[0]]["acc"],
                                "ce_acc_archive": ANCHOR_CE_ACC},
              "cross_script_gap_parity": gap_parity,
              "ce_ref_by_seed": ce_ref, "grid": grid, "verdicts": verdicts,
              "limitations": ["种子底借自第4课b 的 α=0.7/T=3 一格；第4课d 已逐格量过自己的配对误差棒"
                              "（借来的底对单格读数偏大，±2SE 又因跨格共用随机流是下界）",
                              (f"只扫 {tier} + S{width}；" +
                               ("其余档位各自补扫（每档要先有 step4b --tier 造的种子底），另一副宽度是"
                                if tier == TIER else
                                f"另一副宽度（S{other}）只造过 {TIER} 一档的底，档位与宽度没法同时换；")
                               + f"compress_report_v4c{'' if width == 64 else '_s64'}.json"
                                 f"{'_n8' if NSEEDS == 8 else ''}（3 条流那本，各写各的表；"
                                 "8 条流只做过 S128 满/低两档，跨架构/跨档结论要几份对照着读，"
                                 "而跨流数只能读配对 CI）"),
                              f"每格 {len(seeds)} 条流，均值本身没有二次底",
                              "权重/温度解耦只在初始一步的一个批次上配平，不是全程配平",
                              "p_pooled 偏乐观（多条流共用同一批测试图（极差型种子底还随流数变宽））"]}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{len(seeds) * (len(cells) + 1)} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="α×T 扫参归因（默认 S128+满数据；--width 64 测跨架构，"
                                             "--tier 低数据2000 测跨数据量）")
    ap.add_argument("--width", type=int, choices=(128, 64), default=WIDTH,
                    help="学生宽度；锚点档案值/伤害基数/种子底都从该宽度的 compress_report_v4b*.json 读")
    ap.add_argument("--tier", choices=tuple(DATA_TIERS), default=TIER,
                    help="数据档位；每档的种子底/锚点档案值从该档自己的 compress_report_v4b*.json 读。"
                         "低数据与中数据目前只有 S128 造过底，配 --width 64 会直接报错")
    ap.add_argument("--nseeds", type=int, choices=(3, 8), default=3,
                    help="每个格子几条随机流；8 只服务「老账只有 3 条流」那笔分辨率债，"
                         "种子底必须换到对应的 compress_report_v4b_n8.json")
    args = ap.parse_args()
    NSEEDS = args.nseeds
    # 闸门在训练之前：跑完 80 次训练再发现要覆盖老档案，那一发就白烧了
    if out_path(args.width, args.tier).exists():
        raise SystemExit(f"{out_path(args.width, args.tier).name} 已入库（判决档案不覆盖重跑）；"
                         f"要重跑得先把它 mv 成 _prev_*，并且事后逐位核对只有派生字段变")
    main(args.width, args.tier)
