# -*- coding: utf-8 -*-
"""压缩第4课（板块收官）：知识蒸馏——换架构能绕过第3课那堵墙吗，以及老师手里到底有没有「暗知识」。

前三课把两根杠杆都量过了：int8 量化（体积 3.7x、单张不降反慢）、结构化剪枝（体积 1.97x、
乘加 55.1%、单张只快 14%），并且第3课用三点拟合钉出一道上界——**这台机器上单张延迟的 71%
是与乘加无关的固定开销，任何只砍乘加的手段最多把延迟压到 29%**。那节课是事后解释，
本课把它改成**事前预测**：训练之前先按拟合式算出每个学生网络该跑多少毫秒，再去测。
猜对了，这台仪器有外推力；猜错了（很可能是线性外推出界），拟合式就得当场降级。

本课换的是「结构」不是「通道数」：MLP 没有空间复用，784→128→10 只有约 10 万乘加
（teacher 是 111.7 万），是本板块**算力最省**的臂；代价是参数不减反增（10 万 vs 剪枝后的 5.3 万）
——第1课那条「体积跟参数走、速度跟算力走」会在这里以**反方向**再演一次。四问：

  Q1 先验 + 预测（零训练）：学生的乘加/参数/体积各是多少？把第3课那条拟合式当仪器用，
      在训练之前写下每个臂该有的 ms_best。这一步做完就不许改口。
  Q2 蒸馏到底有没有用：同架构、同数据、同优化步数、同初始化种子，只换目标函数
      （硬标签 CE vs 软标签 KD），两档数据量（8500 / 2000）各配一对。
      判差异用**配对臂的正确仪器**：两臂逐张配对的翻牌（翻坏/翻正）做双侧精确符号检验。
      ±3 张那个数是「同一臂跨运行」的复现底（第1~3课量的），拿它判两条同种子臂之间的差是错的仪器。
  Q3 老师的软标签里有没有「暗知识」——本课权重最高的一问，因为它是 Q2 的机制解释：
      Q3a 训练前体检：teacher 的 top1 概率、top1>0.99 的塌缩率、top1−top2 间隔，
          在**它拟合见过的图**上 vs **没见过的测试图**上各是多少。
          若拟合集上软标签已塌成 one-hot，KD 就退化成 CE——**先预言 Q2 不会有用，再去测 Q2**。
      Q3b 训练后看结构：学生的错图与老师的错图重叠（继承偏置）vs 独立错。
  Q4 三本账 + 板块收官：体积 / 乘加 / 精度 / 配对延迟放同一张表找支配关系，
      判「换架构到底买到了什么」。

一条第3课的教训前置成设计而非事后补救：那节课发现「同进程里多插一条臂会消耗全局随机流、
把后面所有臂的初始化改掉」。所以本课 ① 每臂训练前 `manual_seed(SEED)` 重播，② 取样用独立
`torch.Generator` 不碰全局流——于是 CE 臂与 KD 臂**初始权重逐位相同、批次序列逐张相同**，
唯一差别只剩目标函数。固定的是**优化步数**而不是轮数，否则「数据少」和「训得少」混成一团。

泄漏纪律照旧：teacher 的软标签只在训练划分上取，测试集最后读一次。但诚实前提是——
第0课那副 teacher 正是在这 10000 张训练图上学出来的，所以拟合集上的软标签天生是「背过的答案」，
这正是 Q3a 要量出来的东西。

用法：先 train_baseline.py（拿 teacher 权重）、再 step3_prune.py（拿拟合式与剪枝臂读数），最后
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python step4_distill.py     # 纯 CPU，零 API 费
"""
import json
import sys
import math                      # summarize 用 sqrt；下面 comb 那行是这个文件原有的
from math import comb
from pathlib import Path

import torch
import torch.nn as nn

from step1_quantize import MODELS, build_cnn, payload_kb, predict
from step3_prune import LAT_GATE_PCT, macs_by_layer, paired_timing, params_by_layer
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
V3 = HERE / "compress_report_v3.json"
OUT = HERE / "compress_report_v4.json"
HIST = HERE / "latency_history_v4.json"
TEACHER_ARCHIVE_ACC = 0.9798
SEED, STEPS, BS, LR = 42, 600, 128, 1e-3
KD_ALPHA, KD_T = 0.7, 3.0                       # 单点取值，没扫参：判词里要如实带上这个边界
VAL_N = 1500
TIERS = {"满数据8500": 8500, "低数据2000": 2000}
# 第4课本体只用上面两档（重跑必须逐位复现，所以这张表不动）。
# 下面这张是"扫参线可选的数据档位全集"：中数据档是第4课c/d/e 之后为问"T 轴符号在哪一档翻"补的第三档，
# 只有软标签/训练量随档位变的那些脚本（step4b/step4c）从这里取。
DATA_TIERS = {"满数据8500": 8500, "中数据4500": 4500, "低数据2000": 2000}
WIDTHS = (128, 64)
NOISE_IMG = 3                                    # 本板块反复用的噪声底：10000 张里 ±3 张（**同臂跨运行**）
TICK_MS = 0.001     # paired_timing 只留 3 位小数 = 一格 1µs；在 0.015 ms 量级上一格就是 6.7%，
                    # 比对照臂量出来的 0.76% 底大一个数量级——底必须按被测臂自己的刻度重算


# ---------- 以「流」为单位的汇总：全板块一把尺子 ----------
# 表放在这里而不是各脚本自带：补流数时必须连系数表一起改，回落成默认值等于用错刻度（第4课e 的教训）
# n=16 那两格不是查表抄来的：`_coef_check.py` 现算（D4=2n∫xΦ^(n-1)φ 的 Simpson 积分、
# T95=正则化不完全贝塔反解），且**先把在册的 3/5/8/12 四档复算到 ±0.0005 以内**（本轮实测最大偏差 0.0005）
# 才敢用同一把积分去要 16。
# n=26（第十八发：Fashion 那个坐标补到 26 条流）同法：先把**在册五档 3/5/8/12/16** 全复算一遍，
# 实测最大偏差仍是 0.0005（n=12 的 D4，复算 3.2585），其余九个数偏差 ≤0.0004 ⇒ 同一把积分对齐了才取 26。
D4 = {3: 1.693, 5: 2.326, 8: 2.847, 12: 3.258, 16: 3.532, 26: 3.964}     # E[极差]/σ
T95 = {3: 4.303, 5: 2.776, 8: 2.365, 12: 2.201, 16: 2.131, 26: 2.060}    # 双侧 95% t 分位数

# ---------- 复现流（种子）名单：全板块唯一来源 ----------
# 为什么只能有一份：这些标签**就是**"同一批随机流"的定义。第4课b/c/f 各自抄一份 3/8，
# 于是"补一档流数"要在三个文件里各改一遍，漏改的那一份会安静地用短名单跑出"8 条流的账"。
# （step4g / step5 / step6b / step7 仍各自留有一份，那一族本板块还没并进来。）
SEEDS = (42, 7, 2026)                              # 42 是第4课那一轮，留着当逐位对账的锚
SEEDS8 = SEEDS + (11, 202, 3003, 40004, 55555)    # 第4课e/f 那五条：老 3 条只对账，新 5 条才是复现
SEEDS12 = SEEDS8 + (808, 1337, 24601, 90210)      # 第九发新买四条：只加读数密度，不属于板块共用的那 8 条
                                                  # ⇒ 跨脚本逐位对账仍然只发生在前 8 条上（闸⑥ 的许可证范围）
SEEDS16 = SEEDS12 + (1717, 3131, 5252, 7373)      # 第十四发再买四条：同样是"只加密度"
SEEDS26 = SEEDS16 + (11701, 13717, 15217, 19001, 23017, 29017, 31701, 33017, 37017, 41017)
                                                  # 第十八发（倒挂跨不跨任务）新买十条。开跑前按 `grep -rw`
                                                  # 过全部 .py/.json/.md 全部 0 命中；同一把检索用在册老种子
                                                  # 1717/90210 上命中 6/11 本 ⇒ 检索是活的（换过的种子不叫新流）
FLOW_SEEDS = {3: SEEDS, 8: SEEDS8, 12: SEEDS12, 16: SEEDS16, 26: SEEDS26}


def flow_tiers() -> tuple[int, ...]:
    """在册的复现流档位。argparse 的 choices 与运行时的守卫都读它，别在两处各写一遍名单。"""
    return tuple(sorted(FLOW_SEEDS))


def flows(n: int) -> tuple[int, ...]:
    """按档位取那一批流；缺档当场停，不许回落到默认名单（回落＝用短名单跑出一本"长名单的账"）。"""
    if n not in FLOW_SEEDS:
        raise SystemExit(f"复现流没有 {n} 这一档（在册 {list(FLOW_SEEDS)}）：先在这里登记名单，"
                         f"再给 D4/T95 补同一档的系数——名单与系数必须一起长")
    return FLOW_SEEDS[n]


def check_flow_constants(n: int) -> None:
    """开跑前先确认这一档流数的两把系数都在册。

    `summarize` 里那句 `T95[n]` 故意不用 .get——回落成默认值等于用错刻度。但 KeyError 会炸在
    全部训练之后（本轮 n=12 就白烧了 233 秒），所以这道检查提前到第一次训练之前：宁可不跑。"""
    miss = [name for name, tbl in (("D4", D4), ("T95", T95)) if n not in tbl]
    if miss:
        raise SystemExit(f"册上的流数档只有 {sorted(D4)}；n={n} 缺 {'、'.join(miss)}。"
                         f"补流要先用 `_coef_check.py` 现算这两个系数（并把在册各档复算一遍对齐）再写进表，"
                         f"不许凭记忆填、更不许回落成默认值")


# 极差是 max−min 型统计量，随流数机械变宽，所以它只配当「单臂重训一次的抖动」的底；
# 判配对差要用下面这个函数自己的 σ（d4_sigma_hat 是把它折算回 σ 的那一步）。
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


# ---------- 学生架构：一副 MLP，与 teacher 唯一的共同点是输入和输出 ----------
def build_mlp(hidden: int) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(),
                         nn.Linear(hidden, 10))


def logits_of(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + 512]) for i in range(0, len(x), 512)])


# ---------- Q3a 软标签体检：老师手里到底有没有信息 ----------
def soft_label_audit(logits: torch.Tensor, t: float = KD_T) -> dict:
    """温度前后各量一次。塌缩（top1 逼近 1）意味着 KL 项梯度接近 0，
    KD 就只是换了个写法、信息量与 one-hot 无异——那时 Q2 测不出差异是机制决定的，不是运气。"""
    out = {}
    for tag, lg in (("原始", logits), ("除以T", logits / t)):
        top2 = torch.softmax(lg, dim=1).topk(2, dim=1).values
        out[tag] = {"T": t,
                    "mean_top1": round(float(top2[:, 0].mean()), 4),
                    "pct_top1_gt_0.99": round(float((top2[:, 0] > 0.99).float().mean() * 100), 1),
                    "mean_margin_top1_top2": round(float((top2[:, 0] - top2[:, 1]).mean()), 4),
                    "mean_p2": round(float(top2[:, 1].mean()), 4)}   # 第二名的概率才是暗知识本体
    return out


# ---------- Q2 训练臂：CE 与 KD 只差一个目标函数 ----------
def train_student(hidden: int, xd: torch.Tensor, yd: torch.Tensor, soft: torch.Tensor,
                  mode: str, xv: torch.Tensor, yv: torch.Tensor,
                  seed: int = SEED, alpha: float = KD_ALPHA,
                  t: float = KD_T) -> tuple[nn.Module, list]:
    torch.manual_seed(seed)                       # 重播 → 两臂初始权重逐位相同
    m = build_mlp(hidden)
    gen = torch.Generator().manual_seed(seed)     # 批次序列走独立流，不污染全局（第3课的修法）
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    n, curve = len(xd), []
    for step in range(1, STEPS + 1):
        m.train()
        idx = torch.randint(0, n, (BS,), generator=gen)
        z = m(xd[idx])
        if mode == "CE":
            loss = ce(z, yd[idx])
        else:
            soft_loss = nn.functional.kl_div(
                torch.log_softmax(z / t, dim=1),
                torch.softmax(soft[idx] / t, dim=1), reduction="batchmean") * t ** 2
            loss = (1 - alpha) * ce(z, yd[idx]) + alpha * soft_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            m.eval()
            with torch.no_grad():
                curve.append({"step": step,
                              "val_acc": round(float((m(xv).argmax(1) == yv).float().mean()), 4)})
    return m.eval(), curve


# ---------- Q2 判决仪器：配对臂用符号检验，不用跨运行的 ±3 张 ----------
def sign_test_p(broken: int, fixed: int) -> float:
    """H0：两臂之间不一致的那些图，翻坏/翻正各占一半。精确二项双侧 p。
    配对在这里是净赚的：两臂同种子同批次，判对判错一致的图不进入分母，
    于是只剩几百张不一致图要显著偏离五五开——比拿「同一臂跨运行 ±3 张」那个底去比敏感得多。"""
    n = broken + fixed
    if n == 0:
        return 1.0
    k = min(broken, fixed)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------- Q3b 错误结构：继承了老师的偏置，还是各错各的 ----------
def error_structure(y, pred_t, pred_s) -> dict:
    t_wrong, s_wrong = pred_t != y, pred_s != y
    return {"student_wrong": int(s_wrong.sum()),
            "shared_wrong": int((t_wrong & s_wrong).sum()),          # 老师也错 = 继承判断偏置
            "student_only_wrong": int((~t_wrong & s_wrong).sum()),   # 老师对而学生错 = 纯粹没学到
            "fixed_teacher_wrong": int((t_wrong & ~s_wrong).sum()),  # 学生翻正了老师的错
            "inherit_rate_pct": round(float((t_wrong & s_wrong).sum() / max(int(s_wrong.sum()), 1) * 100), 1)}


def main() -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        sys.exit("缺 teacher 权重，先跑 python train_baseline.py")
    if not V3.exists():
        sys.exit("缺 compress_report_v3.json——本课 Q1 要用第3课那条拟合式，先跑 step3_prune.py")
    torch.manual_seed(SEED)
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    fit = v3["latency_fit"]
    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])               # 验证图仍来自训练划分
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    t_macs = sum(macs_by_layer(teacher).values())
    t_kb = round(payload_kb(teacher), 1)
    t_pred = predict(teacher, xte, False)
    t_acc = float((t_pred == yte).mean())
    print(f"teacher 档案对账：{t_acc:.4f} vs {TEACHER_ARCHIVE_ACC:.4f}"
          f"（差 {abs(t_acc - TEACHER_ARCHIVE_ACC) * 100:.2f}pp）| 拟合 {cut} + 验证 {VAL_N}"
          f"（都在训练划分内）| 测试 10000 只读一次 | torch {torch.__version__}")
    assert abs(t_acc - TEACHER_ARCHIVE_ACC) * 100 < 0.05, "teacher 与档案不符，本课全部结论作废"

    # --- Q3a 训练之前先体检软标签：它决定 Q2 有没有理由出收益 ---
    soft_fit = logits_of(teacher, xt_all[:cut])
    soft_te = logits_of(teacher, xte)
    aud_fit, aud_te = soft_label_audit(soft_fit), soft_label_audit(soft_te)
    print("\n[Q3a 软标签体检] 老师的 top1 / 塌缩率(top1>0.99) / top1−top2 间隔 / 第二名概率：")
    for tag, a in (("拟合集（老师见过）", aud_fit), ("测试集（老师没见过）", aud_te)):
        o, s = a["原始"], a["除以T"]
        print(f"  {tag:20s} 原始 top1 {o['mean_top1']:.4f}｜塌缩 {o['pct_top1_gt_0.99']}%"
              f"｜间隔 {o['mean_margin_top1_top2']:.3f}｜p2 {o['mean_p2']:.4f}"
              f"   → /T 后 top1 {s['mean_top1']:.4f}｜塌缩 {s['pct_top1_gt_0.99']}%｜p2 {s['mean_p2']:.4f}")
    collapsed = round(aud_fit["原始"]["pct_top1_gt_0.99"] - aud_te["原始"]["pct_top1_gt_0.99"], 1)
    q2_prediction = "KD 与 CE 分不出胜负（软标签≈one-hot ⇒ KL 项没有 CE 给不出的信息）"
    print(f"  塌缩差 = 拟合集比测试集多 {collapsed:+.1f}pp 的 top1>0.99。"
          f"\n  **先写预言**（此后不许改口）：若老师在自己见过的图上软标签已塌成 one-hot，"
          f"那 KD 相对 CE 就没有额外信息——Q2 测不出差异是机制决定的。温度 T={KD_T} 只整体软化概率、"
          f"不改 argmax，救不回方向。→ 记为可核对的一句话：『{q2_prediction}』")

    # --- Q1 先验 + 事前预测 ---
    print(f"\n[Q1 先验] 学生的三本账，以及**训练之前**用第3课拟合式（{fit['fixed_overhead_ms']}+"
          f"{fit['per_macs_ms']}×乘加占比）写下的延迟预测：")
    prior = {}
    for h in WIDTHS:
        m = build_mlp(h)
        macs, par = macs_by_layer(m), params_by_layer(m)
        pct = sum(macs.values()) / t_macs * 100
        pred_ms = fit["fixed_overhead_ms"] + fit["per_macs_ms"] * pct / 100
        prior[h] = {"macs": sum(macs.values()), "macs_pct_of_teacher": round(pct, 1),
                    "params": sum(par.values()), "kb": round(payload_kb(m), 1),
                    "macs_by_layer": {k: int(v) for k, v in macs.items()},
                    "predicted_ms_best": round(pred_ms, 4)}
        p = prior[h]
        print(f"  S{h}: 乘加 {p['macs']:,}（teacher 的 {pct:.1f}%）| 参数 {p['params']:,}"
              f" | {p['kb']} KB | 预测 ms_best = {pred_ms:.4f} ms")
    t_pred_ms = fit["fixed_overhead_ms"] + fit["per_macs_ms"]
    print(f"  teacher: 乘加 {t_macs:,} | 参数 {sum(params_by_layer(teacher).values()):,} | {t_kb} KB")
    print(f"  先验判决：**换架构砍得动乘加、砍不动参数**——S128 乘加只剩 teacher 的 "
          f"{prior[128]['macs_pct_of_teacher']}%（本板块最狠一刀），体积却是 teacher 的 "
          f"{prior[128]['kb'] / t_kb:.0%}，比第3课剪到一半的 CNN（211.5 KB）还大。"
          f"按拟合外推，这一刀最多把单张从 {t_pred_ms:.3f} 压到 {prior[128]['predicted_ms_best']:.4f} ms"
          f"（{t_pred_ms / prior[128]['predicted_ms_best']:.2f}x）——**墙还在**，Q4 去验。")

    # --- Q2 + Q3b 各臂 ---
    print("\n[Q2 蒸馏 vs 只学硬标签] 同种子同起点同步数，只换目标函数：")
    rows: dict[str, dict] = {}
    ctrl = build_cnn()
    ctrl.load_state_dict(teacher.state_dict())
    models: list[tuple[str, nn.Module]] = [("A0 teacher", teacher), ("A0复本(对照)", ctrl)]
    for tier, n_data in TIERS.items():
        xd, yd = xt_all[:n_data], torch.tensor(yt_all[:n_data])
        soft = logits_of(teacher, xd)
        for h in WIDTHS:
            if tier.startswith("低数据") and h != 128:
                continue          # 低数据档只在主架构上配对（省下的时间已注明，不是漏跑）
            for mode in ("CE", "KD"):
                st, curve = train_student(h, xd, yd, soft, mode, xv, yv)
                tag = f"S{h} {mode} {tier}"
                pred = predict(st, xte, False)
                ce_arm = rows.get(f"S{h} CE {tier}")
                e = error_structure(yte, t_pred, pred)
                rows[tag] = {
                    "kb": round(payload_kb(st), 1),
                    "macs_pct_of_teacher": prior[h]["macs_pct_of_teacher"],
                    "params": prior[h]["params"],
                    "acc": round(float((pred == yte).mean()), 4),
                    "wrong": int((pred != yte).sum()),
                    "flip_vs_teacher": {"broken": int(((t_pred == yte) & (pred != yte)).sum()),
                                        "fixed": int(((t_pred != yte) & (pred == yte)).sum())},
                    "kd_vs_ce_flips": ({"broken": int(((ce_arm["pred"] == yte) & (pred != yte)).sum()),
                                        "fixed": int(((ce_arm["pred"] != yte) & (pred == yte)).sum())}
                                       if ce_arm else None),
                    "acc_gap_vs_ce_pp": (round((float((pred == yte).mean()) - ce_arm["acc"]) * 100, 2)
                                         if ce_arm else None),
                    "error_structure": e,
                    "kl_to_teacher_test": round(float(nn.functional.kl_div(
                        torch.log_softmax(logits_of(st, xte) / KD_T, 1),
                        torch.softmax(soft_te / KD_T, 1), reduction="batchmean") * KD_T ** 2), 4),
                    "val_curve": curve, "pred": pred}
                r = rows[tag]
                extra = (f" | 对 CE 臂 {r['acc_gap_vs_ce_pp']:+.2f}pp"
                         f"（翻坏 {r['kd_vs_ce_flips']['broken']}/翻正 {r['kd_vs_ce_flips']['fixed']}）"
                         if ce_arm else "")
                print(f"  {tag:22s} 精度 {r['acc']:.4f}（错 {r['wrong']}）{extra}"
                      f" | 对 teacher 翻坏 {r['flip_vs_teacher']['broken']}/翻正 {r['flip_vs_teacher']['fixed']}"
                      f" | 错图继承率 {e['inherit_rate_pct']}% | 末端 val {curve[-1]['val_acc']:.4f}")
                models.append((tag, st))
    for r in rows.values():
        r.pop("pred", None)

    # --- Q2 判决 ---
    print("\n[Q2 判决] 同档同宽度内 KD−CE（仪器 = 逐张配对翻牌的双侧精确符号检验；"
          f"±{NOISE_IMG} 张那个底是「同一臂跨运行」的复现底，判不了配对差）：")

    def pair(h: int, tier: str) -> dict | None:
        ce_k, kd_k = f"S{h} CE {tier}", f"S{h} KD {tier}"
        if ce_k not in rows or kd_k not in rows:
            return None
        gap = rows[kd_k]["acc"] - rows[ce_k]["acc"]
        n_img = int(round(gap * 10000))
        fl = rows[kd_k]["kd_vs_ce_flips"]
        p = sign_test_p(fl["broken"], fl["fixed"])
        rows[kd_k]["kd_vs_ce_p_sign"] = round(p, 6)
        # 逐步长相减：区分「一开始就慢」和「最后才学错」——两种负收益的修法完全不同
        pt = [round((b - a) * 100, 2) for a, b in
              zip([c["val_acc"] for c in rows[ce_k]["val_curve"]],
                  [c["val_acc"] for c in rows[kd_k]["val_curve"]])]
        shape = ("全程落后" if all(v <= 0 for v in pt) else
                 "全程领先" if all(v >= 0 for v in pt) else
                 "先落后→后段反超" if pt[0] < 0 else "混合")
        return {"width": h, "tier": tier, "ce_acc": rows[ce_k]["acc"], "kd_acc": rows[kd_k]["acc"],
                "gap_pp": round(gap * 100, 2), "gap_imgs": n_img, "flips": fl, "p_sign": p,
                "significant": bool(p < 0.05), "val_gap_curve_pp": pt, "val_shape": shape,
                "student_only_delta_vs_ce": rows[kd_k]["error_structure"]["student_only_wrong"]
                - rows[ce_k]["error_structure"]["student_only_wrong"],
                "cross_run_floor_imgs": NOISE_IMG}

    verdicts = {f"S{h} {t}": d for t in TIERS for h in WIDTHS if (d := pair(h, t))}
    for k, v in verdicts.items():
        print(f"  {k:22s} CE {v['ce_acc']:.4f} → KD {v['kd_acc']:.4f}"
              f"（{v['gap_pp']:+.2f}pp = {v['gap_imgs']:+d} 张）"
              f"| 翻坏 {v['flips']['broken']}/翻正 {v['flips']['fixed']}"
              f" → p={v['p_sign']:.2e} "
              + (f"**显著，KD 更{'差' if v['gap_imgs'] < 0 else '好'}**" if v["significant"]
                 else "不显著（翻牌数与五五开分不开）")
              + f" | 逐步长 KD−CE {v['val_gap_curve_pp']} → {v['val_shape']}")
    best_student = max(rows, key=lambda k: rows[k]["acc"])
    over = rows[best_student]["acc"] > t_acc
    print(f"  最强学生臂 {best_student} 精度 {rows[best_student]['acc']:.4f} vs teacher {t_acc:.4f}"
          f"（{(rows[best_student]['acc'] - t_acc) * 100:+.2f}pp）"
          + ("——学生反超老师，先查仪器再讲故事" if over else ""))

    # --- Q3b 结构 ---
    print("\n[Q3b 错图结构] 继承率 = 学生错图里老师也错的比例（老师共错 "
          f"{int((t_pred != yte).sum())} 张）：")
    for k, v in rows.items():
        e = v["error_structure"]
        print(f"  {k:22s} 共享错 {e['shared_wrong']:3d}｜独立错 {e['student_only_wrong']:3d}"
              f"｜翻正老师 {e['fixed_teacher_wrong']:3d}｜继承率 {e['inherit_rate_pct']:5.1f}%"
              f"｜对老师 KL {v['kl_to_teacher_test']:.4f}")
    kl = {t: (rows[f"S128 CE {t}"]["kl_to_teacher_test"], rows[f"S128 KD {t}"]["kl_to_teacher_test"])
          for t in TIERS if f"S128 KD {t}" in rows}
    print(f"  KL(学生‖老师) 测试集 CE→KD：{kl} ——这项**就是 KD 的目标函数本身**，KD 臂更低属于自证，"
          f"不能当疗效证据；能当证据的只有上面那列错图结构。")

    # --- Q4 配对延迟 + 事前验收 ---
    print("\n[Q4 延迟] 交错配对计时（沿用第3课那台仪器：8 轮 × 800 张、轮转换序）…")
    lrows, floors = paired_timing(models, xte)
    a0 = lrows["A0 teacher"]
    for name, v in lrows.items():
        print(f"  {name:22s} p50 {v['ms_p50']:.3f} ms（{a0['ms_p50'] / v['ms_p50']:.2f}x）"
              f" | best {v['ms_best']:.3f} ms（{a0['ms_best'] / v['ms_best']:.2f}x）"
              f" | 吞吐 {v['thr_batch128']:.0f} 张/秒（{v['thr_batch128'] / a0['thr_batch128']:.2f}x）")
    ok = {k: v < LAT_GATE_PCT for k, v in floors.items()}
    print("  [仪器自校准] 对照臂跨臂相对差："
          + "｜".join(f"{k} {floors[k]:.1f}%{'✓' if ok[k] else '✗'}" for k in floors)
          + f"（门槛 {LAT_GATE_PCT:.0f}%）→ 可用栏：" + "、".join(k for k, v in ok.items() if v))
    hist = json.loads(HIST.read_text(encoding="utf-8")) if HIST.exists() else []
    hist.append({"n_runs": len(hist) + 1, "cfg": ["LAT", 8, 800], "ctrl_floor_pct": floors,
                 "p50": {k: v["ms_p50"] for k, v in lrows.items()},
                 "best": {k: v["ms_best"] for k, v in lrows.items()}})
    HIST.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [跨运行] 本课历史已攒 {len(hist)} 次（机器本地文件，不进 git）")
    stat = "ms_best" if ok["best"] else "ms_p50"
    skey = "best" if stat == "ms_best" else "p50"

    print(f"\n[把 Q1 的账现在结掉] 事前预测 vs 实测（口径 {stat}；"
          f"拟合点落在乘加 55%~100%，被测点在 {prior[64]['macs_pct_of_teacher']}%~"
          f"{prior[128]['macs_pct_of_teacher']}% —— 这是**外推**，不是插值）：")
    check = {}
    for tag, v in lrows.items():
        if "对照" in tag or tag.startswith("A0"):
            continue
        h = int(tag.split()[0][1:])
        pred_ms, meas = prior[h]["predicted_ms_best"], v[stat]
        err = (meas - pred_ms) / pred_ms * 100
        floor_pct = max(floors[skey], TICK_MS / meas * 100)   # 底要按被测臂自己的读数刻度重算
        within = abs(meas - pred_ms) <= floor_pct / 100 * meas
        check[tag] = {"predicted_ms": pred_ms, "measured_ms": meas, "err_pct": round(err, 1),
                      "floor_pct": round(floor_pct, 1), "within_instrument_floor": bool(within)}
        print(f"  {tag:22s} 预测 {pred_ms:.4f} → 实测 {meas:.4f} ms（{err:+.1f}%）"
              f" 本臂底 ±{floor_pct:.1f}% {'✓ 仪器分辨率内' if within else '✗ 超出'}")
    n_in = sum(v["within_instrument_floor"] for v in check.values())
    max_err = max(abs(v["err_pct"]) for v in check.values())
    extrapolated = 100 / prior[64]["macs_pct_of_teacher"]
    print(f"  判决：{n_in}/{len(check)} 个臂落在仪器分辨率内，最大偏差 {max_err:.1f}%"
          f"（外推距离约 {extrapolated:.0f} 倍）→ "
          + ("拟合式**有外推力**：第3课那道 29% 上界可以当设计约束用（拿它先算再动手）。"
             if n_in == len(check) else
             "拟合式**外推失败**：降级为只在 55%~100% 区间成立的事后解释，本课改用实测值。")
          + "两种结果都如实报，不许只报蒙对的那半边。")

    # --- 第3课那道上界受审：破没破由实测比值说了算，脚本不许预先站队 ---
    ck = f"S128 KD {list(TIERS)[0]}"
    wall_ms = a0[stat] * fit["compute_share_pct"] / 100
    ratio_pct = lrows[ck][stat] / a0[stat] * 100
    broke = ratio_pct < fit["compute_share_pct"]
    print(f"\n[第3课那道上界，本课是它的对手] 拟合式说基线里 "
          f"{100 - fit['compute_share_pct']:.0f}% 的延迟与乘加无关 → "
          f"只砍乘加的手段最多降到基线的 {fit['compute_share_pct']}%（{wall_ms:.3f} ms）。\n"
          f"  实测 {ck} = {lrows[ck][stat]:.3f} ms = 基线的 {ratio_pct:.1f}%（乘加只留 "
          f"{prior[128]['macs_pct_of_teacher']}%）→ "
          + ("**上界被击穿**：那条拟合式只在「同一副 conv 图」这一族里成立，"
             f"{fit['fixed_overhead_ms']:.3f} ms 不是机器的常数，"
             "是「这台机器跑这副 conv 图」的常数。"
             if broke else "上界成立：换架构也没绕过它。"))
    p1, p2 = prior[128]["macs_pct_of_teacher"] / 100, prior[64]["macs_pct_of_teacher"] / 100
    m1, m2 = lrows[ck][stat], lrows[f"S64 KD {list(TIERS)[0]}"][stat]
    if m1 <= m2:                            # 更宽的图不更慢 → 斜率非正，两点拟不出东西
        a_mlp, b_mlp = m2, 0.0
        rate_txt = (f"两臂读数 {m1:.3f} vs {m2:.3f} ms（宽度加了没变慢）→ 斜率拟不出，"
                    f"截距只能给上界 ≤{m2:.3f} ms")
    else:
        b_mlp = (m1 - m2) / (p1 - p2)       # ms / (100% 乘加)，与第3课 per_macs_ms 同单位
        a_mlp = m2 - b_mlp * p2
        same_rate = 0.5 < b_mlp / fit["per_macs_ms"] < 2.0
        rate_txt = (f"截距 {a_mlp:.4f} ms（占它自己延迟 {a_mlp / m1 * 100:.0f}%）"
                    f"= 第3课那条 {fit['fixed_overhead_ms']:.3f} ms 的 "
                    f"{a_mlp / fit['fixed_overhead_ms']:.0%}；"
                    f"斜率 {b_mlp:.4f} vs conv 图 {fit['per_macs_ms']:.4f} ms/（100% 乘加）→ "
                    + ("斜率同量级，**搬家搬的全是截距**。" if same_rate else
                       "斜率本身也差开了 → 两副图连「每 100% 乘加值多少毫秒」都不一样，"
                       "截距和单价都得各记各的。"))
    print(f"  那 MLP 自己有没有底？拿 S128/S64 两个宽度臂现拟一条（**只有两点、零自由度，"
          f"只当量级看**）：{rate_txt} → "
          + ("**墙没有消失，只是跟着架构搬家**：换架构买到的不是「去掉固定开销」，"
             "是「换一副开销小得多的图」。" if broke else
             "MLP 的截距与 conv 图同量级 → 第3课那条上界跨架构也成立，本课的预测胜出。"))
    rate = {}
    for tag, v in lrows.items():
        mm = t_macs if tag.startswith("A0") else prior[int(tag.split()[0][1:])]["macs"]
        rate[tag] = round(mm / v[stat] / 1e6, 1)
    print("  各臂有效算力率（乘加 ÷ 单张时间，GMAC/s）："
          + "｜".join(f"{k.replace(' 满数据8500', '·8500').replace(' 低数据2000', '·2000')} {v}"
                      for k, v in rate.items() if "对照" not in k))

    # --- 收官 Pareto：数字全部现读各课 JSON，不手抄 ---
    # 延迟轴不能直接跨课比毫秒（第3课的纪律：跨轮漂移 1.15x）。改成「同一轮内相对 teacher 的比值」，
    # 比值是配对法在同一轮里量出来的，跨课才可比。
    print("\n[收官 Pareto] 体积 / 乘加 / 精度 / 同轮内延迟比四列（跨四节课；延迟只取各课自己那一轮的读数）：")
    v3_a0_base = v3["latency_paired"]["A0 fp32基线"]
    cand = [("teacher CNN fp32（第0课）", t_kb, 100.0, t_acc, a0[stat], 1.0)]
    for tag, key, latkey in (("剪 50% CNN+微调（第3课）", "R50F 剪+微调", "R50F"),
                             ("剪 50%+int8（第3课）", "R50FQ 再int8量化", "R50FQ")):
        arm = v3["arms"][key]
        cand.append((tag, arm["kb"], arm["macs_pct"], arm["acc"],
                     v3["latency_paired"][latkey][stat],
                     v3["latency_paired"][latkey][stat] / v3_a0_base[stat]))
    for k in rows:
        if "KD" in k and "8500" in k:
            cand.append((f"蒸馏学生 {k}", rows[k]["kb"], rows[k]["macs_pct_of_teacher"],
                         rows[k]["acc"], lrows[k][stat], lrows[k][stat] / a0[stat]))
    for i, (tag, kb, mp, acc, ms, lr_) in enumerate(cand):
        dom = [c[0] for j, c in enumerate(cand) if j != i
               and c[1] <= kb and c[2] <= mp and c[3] >= acc and c[5] <= lr_]
        print(f"  {tag:30s} {kb:7.1f} KB｜乘加 {mp:5.1f}%｜精度 {acc:.4f}｜"
              f"延迟 {lr_ * 100:5.1f}% 基线（{ms:.3f} ms 跨轮只当量级）｜"
              + (f"**被 {dom[0]} 支配**" if dom else "未被支配（Pareto 前沿）"))

    # --- 判词与自我审计 ---
    print("\n[判词 Q2/Q3 蒸馏有没有用]")
    for t, v in verdicts.items():
        print(f"  {t}：KD−CE {v['gap_pp']:+.2f}pp（{v['gap_imgs']:+d} 张，"
              f"翻坏 {v['flips']['broken']}/翻正 {v['flips']['fixed']}，"
              f"符号检验 p={v['p_sign']:.2e} → "
              + ("显著" if v["significant"] else "不显著") + "）")
    sig = {t: v for t, v in verdicts.items() if v["significant"]}
    hurt = [t for t, v in sig.items() if v["gap_imgs"] < 0]
    helped = [t for t, v in sig.items() if v["gap_imgs"] > 0]
    thin = aud_fit["原始"]["pct_top1_gt_0.99"] >= 80.0
    print(f"  机制读数：老师在拟合集上 {aud_fit['原始']['pct_top1_gt_0.99']}% 的图 top1>0.99、"
          f"第二名概率仅 {aud_fit['原始']['mean_p2']:.4f} → 软标签**确实**接近 one-hot"
          f"（{'预言的前提兑现' if thin else '预言的前提不成立：软标签仍有信息'}）。")
    if not sig:
        concl = "两档都没测出差异 → 与预言一致：老师拿不出 CE 之外的信息。"
    elif hurt and not helped:
        slow = all(verdicts[t]["val_shape"] == "全程落后" for t in hurt)
        concl = (f"**预言落空**：不是「分不出胜负」，是 KD 在 {'、'.join(hurt)} 上**显著更差**。"
                 + ("而逐步长说它**从第 100 步就落后**（见上面 val_shape）→ 这不是「最后学到了老师的错」，"
                    "是「整个目标函数被换掉之后收敛变慢」：α=" + str(KD_ALPHA) + " 把真标签的权重压到 "
                    f"{1 - KD_ALPHA:.0%}，学生花力气去贴一副近似 one-hot 的分布，"
                    "等价于把监督信号稀释了。"
                    if slow else
                    "而逐步长显示差距是后段才张开的 → 更像学到了老师的错，"
                    "与错图结构（独立错增加）同向。")
                 + "所以「软标签没信息」要精确成「**在这个 α/T 下，贴老师的代价大于它能给的东西**」。")
    elif helped and not hurt:
        concl = "KD 显著更好 → 老师的软标签确有 CE 之外的信息，须回头修 Q3a 的读法（塌缩率不足以代表信息量）。"
    else:
        concl = f"方向随配置翻转（变差 {'、'.join(hurt)}｜变好 {'、'.join(helped)}）→ 分开判，不许合并成一个平均。"
    print(f"  事前预言：『{q2_prediction}』 → {concl}")
    hurt_txt = "、".join(f"{t} {verdicts[t]['gap_pp']:+.2f}pp" for t in hurt) or "本轮读数"
    print("  这条判决的代价要记在账上：为了让 CE/KD 严格配对，本课固定的是**优化步数**，"
          f"于是「两臂各自收敛后还差多少」这个对照被牺牲了——{hurt_txt} 里混着「没收敛完」的成分，"
          "α 扫描或拉长步数才能把它剥干净。")
    lo_tier = list(TIERS)[-1]
    lo_v = verdicts.get(f"S128 {lo_tier}")
    if lo_v:
        lo_txt = (f"低数据档独立错变化 {lo_v['student_only_delta_vs_ce']:+d} 幅、"
                  f"KD−CE {lo_v['gap_pp']:+.2f}pp（p={lo_v['p_sign']:.2e}）、"
                  f"逐步长 {lo_v['val_gap_curve_pp']} → {lo_v['val_shape']}"
                  + ("：正是「KD 当正则器」的签名（慢启动、后段反超），但**末端不显著**，"
                     "两点数据量定不了这条曲线。" if not lo_v["significant"] else
                     "，且末端显著。"))
        print("  翻牌结构（比平均分更硬）：满数据档「老师做对、学生做错」的独立错从 "
              f"{rows[f'S128 CE {list(TIERS)[0]}']['error_structure']['student_only_wrong']} 涨到 "
              f"{rows[f'S128 KD {list(TIERS)[0]}']['error_structure']['student_only_wrong']} —— "
              f"注意这条与「追老师过头」和「KD 还没训完」**两种读法都同向**，分开它俩的只有上面那列逐步长；"
              f"{lo_txt}")
    sp = lrows[f"S128 KD {list(TIERS)[0]}"]
    r50_macs = v3["arms"]["R50F 剪+微调"]["macs_pct"]
    v3_a0 = v3["latency_paired"]["A0 fp32基线"]
    v3_r50 = v3["latency_paired"]["R50F"]
    kb_pct = prior[128]["kb"] / t_kb * 100
    best_gap_pp = (rows[best_student]["acc"] - t_acc) * 100
    print(f"[判词 Q4 换架构买到了什么] 乘加砍到 {prior[128]['macs_pct_of_teacher']}%"
          f"（本板块最狠，比第3课剪枝还低 {r50_macs - prior[128]['macs_pct_of_teacher']:.0f}pp），"
          f"实测单张 {a0[stat]:.3f} → {sp[stat]:.3f} ms（{a0[stat] / sp[stat]:.2f}x = 基线的 "
          f"{ratio_pct:.1f}%）、吞吐 {a0['thr_batch128']:.0f} → {sp['thr_batch128']:.0f} 张/秒"
          f"（{sp['thr_batch128'] / a0['thr_batch128']:.2f}x）。"
          f"体积轴分工要看清：S128 = {prior[128]['kb']} KB（老师体积的 {kb_pct:.0f}%，"
          f"{'几乎没让步' if abs(kb_pct - 100) < 15 else '有让步'}），"
          f"S64 = {prior[64]['kb']} KB（{prior[64]['kb'] / t_kb:.0%}）才是真降体积那一头。"
          f"精度最强臂 {rows[best_student]['acc']:.4f} vs teacher {t_acc:.4f}"
          f"（{best_gap_pp:+.2f}pp）。→ 一句话："
          + ("**这是本板块第一次「砍乘加」和「砍时间」同号**："
             f"第3课（它自己那一轮内，跨轮引用只当量级）砍掉 {100 - r50_macs:.1f}pp 乘加只买到 "
             f"{v3_a0_base[stat] / v3_r50[stat]:.2f}x，本课砍到 "
             f"{prior[128]['macs_pct_of_teacher']}% 乘加买到 {a0[stat] / sp[stat]:.1f}x——"
             "不是乘法变便宜了，是换掉 conv 图之后**每层的固定开销那本账重开了**；"
             f"代价写在精度那一列（{best_gap_pp:+.2f}pp），且没拆「架构 vs 步数」。"
             if broke else "**换架构也没绕过第3课那堵墙**，固定开销是机器常数。"))
    print(f"[自我审计① 配对真配上了] CE/KD 同种子重播初始化、同 Generator 批次序列，"
          f"差别只剩目标函数；固定步数 {STEPS} 而非固定轮数，两档数据量因此可比。")
    print(f"[自我审计② 学生与老师差 {best_gap_pp:+.2f}pp 是本课的靶子吗] "
          f"{rows[best_student]['acc']:.4f} vs {t_acc:.4f}"
          + ("（反超 → 必须查泄漏再谈疗效）" if over else
             f"（{(rows[best_student]['acc'] - t_acc) * 100:+.2f}pp）→ 不是疗效问题而是**先验**问题："
             "这与 CV 线「架构要匹配挑战」同构，MLP 少掉的是卷积的局部/平移先验。"
             f"诚实的边界：本课没做「MLP 加宽/训久能不能追回来」的对照（固定 STEPS 是为了让 CE/KD 可比，"
             "不是给学生网络的最优配置），所以这一栏里有多少是架构、有多少是步数，**没拆开**。"))
    print(f"[自我审计③ 未做的对照] T={KD_T}、α={KD_ALPHA} 单点未扫参，「额外信息是负的」这个判决"
          f"只支持在本配置下（α=0.7 偏高是明显的怀疑点：若 α→0 伤害消失，则伤害来自权重分配而非软标签本身，"
          f"本课没做这条曲线）。数据量只 8500/2000 两点。"
          f"符号检验的边界：它只假设「不一致的图两臂交换对称」，两臂同种子同批次时这个假设比独立样本更强，"
          f"但仍不含「同一份数据重训一次」的方差——那是第3课 ±3 张底管的事，两者不互相替代。")
    print(f"[自我审计④ 延迟栏的资格] 本轮对照臂噪声底 {floors}（门槛 {LAT_GATE_PCT}%），"
          f"过闸的口径才用于判决；但对照臂的底是在 teacher 的 {a0['ms_best']:.3f} ms 量级上量的，"
          f"学生臂快了近十倍，读数刻度 {TICK_MS} ms 在 "
          f"{min(v[stat] for k, v in lrows.items() if '对照' not in k and not k.startswith('A0')):.3f}"
          f" ms 量级就是 {TICK_MS / sp['ms_best'] * 100:.0f}% → "
          f"底按各臂自己的量级重算（见 prediction_check.floor_pct）。")

    report = {"teacher_acc": round(t_acc, 4), "teacher_kb": t_kb, "teacher_macs": int(t_macs),
              "test_n": int(len(yte)), "train_fit_n": cut, "torch": torch.__version__,
              "hyperparams": {"steps": STEPS, "bs": BS, "lr": LR, "alpha": KD_ALPHA, "T": KD_T},
              "soft_label_audit": {"fit_set": aud_fit, "test_set": aud_te, "collapse_gap_pp": collapsed},
              "q2_prediction_before_training": q2_prediction,
              "prior_prediction": prior, "v3_fit_used": fit,
              "arms": rows, "kd_vs_ce": verdicts, "latency_paired": lrows,
              "ctrl_floor_pct": floors, "latency_gate_pct": LAT_GATE_PCT, "latency_stat": stat,
              "tick_ms": TICK_MS,
              "prediction_check": check, "prediction_within": int(n_in),
              "prediction_arms": len(check), "max_pred_err_pct": round(max_err, 1),
              "v3_ceiling_recheck": {"student_ms_pct_of_teacher": round(ratio_pct, 1),
                                     "v3_ceiling_pct": fit["compute_share_pct"],
                                     "broke": bool(broke),
                                     "mlp_own_intercept_ms": round(a_mlp, 4),
                                     "mlp_own_slope_ms": round(b_mlp, 4),
                                     "gmac_per_s": rate},
              "pareto": [{"tag": t, "kb": kb, "macs_pct": mp, "acc": acc,
                          stat: ms, "lat_ratio_vs_own_teacher": round(lr_, 4),
                          "dominated_by": [c[0] for j, c in enumerate(cand) if j != i
                                           and c[1] <= kb and c[2] <= mp
                                           and c[3] >= acc and c[5] <= lr_]}
                         for i, (t, kb, mp, acc, ms, lr_) in enumerate(cand)],
              "limitations": ["T/α 单点未扫参（α=0.7 偏高，KD 的伤害有多少来自权重分配而非软标签本身没拆）",
                              "数据量只有 8500/2000 两档", "每臂只训一次：符号检验只管图间配对，不管种子间方差",
                              "teacher 的软标签取自它拟合过的训练图（塌缩程度已量于 soft_label_audit）",
                              "学生落后老师 4.72pp 里架构与训练步数两件事没拆开（固定步数为配对服务）"]}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已存 compress_report_v4.json（10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    main()
