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
  # S64 那一档也换随机流复现（第4课：CE 0.9279 / KD 0.9224，−55 张）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4b_seed_variance.py --width 64
  # 给第4课没跑过的数据档造种子底（扫参线要拿它当闸门；只产底，不产复现）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4b_seed_variance.py --tier 中数据4500
  # 第八发：把这本种子底本身扩到第4课e/f 那批 8 条流（老 3 条先与 compress_report_v4b.json 逐位对账，
  # 档案名带 _n8 与老账并存）——板块一路用 3 个点的极差当闸门，这一发就是还那笔粗估债：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4b_seed_variance.py --nseeds 8
  # 第二十一发（板块计数）：第十九发之后欠的那条外推轴——换模型族（第八维），
  #   学生从 MLP 换成老师同族的 CNN。下面整段是**开跑前的预登记**，事后不改一个字。
  # 【预登记 · 只写不变量，读数事后现报】
  # 买什么：前七维买的是"同一副学生换个条件"（数据档/温度/α/宽度/流/标签/链），
  #   学生架构从来是常量。这一发把 MLP 这层地板换掉，检验"KD 输掉的张数"是不是
  #   MLP 这个特定几何的产物——如果是，换族后符号或量级应当翻；如果不是，它就成了
  #   一条与架构无关的观察。**这一发只买地板和逐流配对 CI，不买机制解释**：
  #   机制那一路（step4i）的量都是 logits/梯度层面的，与学生的几何无关，换族不重跑它。
  # 盘上有没有底：查过——30 本在册账**没有任何一本记录学生架构**（档案里没有这个字段），
  #   也不存在 v4b_cnn* 这类账本。所以 CNN 这一发没有"复现"可言，只有"第一次落账"，
  #   故 --arch cnn 时提前把跨脚本数值对账撤到只对新学生族的第一本账生效（家族闸）。
  # 实测价签（同一台机器、同一份数据、CPU，跑一次测一次，不是估的）：
  #   MLP S128 学生 101,770 参数 / CE 臂 2.55s / KD 臂 2.84s
  #   CNN 学生     105,866 参数 / CE 臂 24.94s / KD 臂 25.28s  ← 参数只多 4%，墙钟 8.8×
  #   CNN 没有宽度旋钮（build_cnn 不接受 hidden）→ --arch cnn 时 --width 无定义，
  #   所以 --arch cnn --width 64 必须当场退出，不能默默按 128 算。
  # 链条：这一发只跑 step4b（n=3 是 6 次训练、n=8 是 16 次）。完整链 v4b→v4c→v4f 约 95 次
  #   训练、按上面价签约 40 分钟——那两本是否补，等这一本的地板数字出来再决定，不预先承诺。
  # 四个拆红开关各自该红在哪道闸（红点不唯一 = 那道检查没在生效）：
  #   --arch cnn --width 64   → 训练前退出，0 次训练
  #   --arch cnn --blankrun   → 空账闸（不落盘），0 次训练
  #   --arch cnn --blindcommit → 仪器对账（数值），12 次训练
  #   --arch cnn --blindfamily → 跨脚本数值闸，16 次训练
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4b_seed_variance.py --arch cnn --tier 中数据4500 --nseeds 3
  # 【落地 · 两本都入了库，此后只追加不改口】
  #   compress_report_v4b_cnn_mid.json（3 条流，6 次训练）与 compress_report_v4b_cnn_mid_n8.json
  #   （8 条流，16 次训练）。跨种子的 KD−CE 之差：n=3 是 [37, −13, 4] 张（极差 50），
  #   n=8 是 [37, −13, 4, 51, 26, 42, 5, 25] 张（极差 64、7/8 同号、pooled p=1.11e-06）。
  #   底（极差型）：n=3 为 KD 63 / CE 30，n=8 为 KD 83 / CE 103 —— 同一条"极差随流数机械变宽"
  #   在本族照样成立，所以 gap_over_floor 只能在同一本账内部读（这本档案里那句 `floor_is_n_dependent`
  #   就是干这件事的），拿 3 条流与 8 条流两本的比值互相翻案就是拿 n 当效应。
  #   **本族第一对"补流不改老读数"的凭据**：`cross_script_parity` 现读"老3条流×4项读数 逐位相同"，
  #   也就是上面那条不变量真的落到了盘上（守卫闸⑥ 从此有跨族那一对 15 格可核）。
  #   **本族没有第4课历史判决可对**这件事是档案自己说的：`parity_no_archive` = ["中数据4500 CE",
  #   "中数据4500 KD"]、`parity_seed42_vs_archive` 是空字典 ⇒ 别去这两个字段里找"跨脚本逐位相同"，
  #   这一族的对账线在别处（α=0 恒等闸 + step4c/step4f 之间的 6/6 那两格）。
  #   上面预登记里那三个注入开关（--blankrun / --blindcommit / --blindfamily）**本轮未复测**：
  #   这一发真跑的只有正常两本 + `--arch cnn --width 64` 那一档（训练前退出，实测 0 次训练）。
  #   它们是给下一次改动的红档，本轮没跑就不要引用它们的"预期红数"当实测。
  #   修 bug 一处：这一本文件的 limitations 第一句曾少一个 `f` 前缀（`step4b_seed_variance.py:326`），
  #   于是**五本已入库档案**里印着字面量 `{len(seeds)}` 而不是数字——没有任何数字因此变红，坏掉的
  #   恰好是唯一那句告诉后来人"这个底欠估到什么程度"的话。守卫由此新增**闸⑫**（腿 A 扫未渲染占位符、
  #   腿 B 核那句里的 N 与该本自己的 `len(seeds)`），专属拆红 `--unformatted`（实测红 8 条）与
  #   `--wrongcount`（实测红 1 条，且验两腿分离）。**修它的脚本自己也误伤过一次**：见 README 第二十一发。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (DATA_TIERS, SEEDS, SEEDS8, STUDENT_ARCHS, TIERS, VAL_N,
                           check_flow_constants, flow_tiers, flows, logits_of, sign_test_p,
                           train_student)
from step4e_marginal_recheck import summarize          # CI 用全板块同一套 t 系数表，不在这里另刻一把
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
V4 = HERE / "compress_report_v4.json"
# 复现流名单 SEEDS / SEEDS8 / … 住在 step4_distill（唯一来源），由上面的 import 带进来
NSEEDS = 3                                         # 可被 --nseeds 覆盖
WIDTH = 128                                        # 可被 --width 覆盖（S64 复现用）
ARCH = "mlp"                                       # 可被 --arch 覆盖；cnn = 换模型族那一发（学生与老师同架构）
# 三个拆红开关：各自撤掉家族化改造里的一道检查，只用来证明"那道检查真在生效"，不参与任何读数
BLANK_RUN = BLIND_COMMIT = BLIND_FAMILY = False
TIER = ""                                          # 可被 --tier 覆盖；空 = 第4课那两档全跑
# 第4课档案值（committed run）——种子 42 必须逐位复现，否则说明本课改坏了 train_student
# 低数据档第4课只在 S128 上配对，所以 64 这一副里只有满数据档可查
ARCHIVE = {128: {"满数据8500": {"CE": 0.9326, "KD": 0.9263}, "低数据2000": {"CE": 0.9045, "KD": 0.9072}},
           64: {"满数据8500": {"CE": 0.9279, "KD": 0.9224}}}


# 单档复现的档案名。第4课那两档**故意不在这里**：它们的账本是一份合并档案（两档都在里面），
# 允许 --tier 单跑写回同名文件，就等于让一次单档运行把另一档的账抹掉。
# _low 只服务 cnn 族：MLP 那一族走 --tier 会先被下面那道"合并档案"闸拦掉（那里它仍是两档共本），
# 而 cnn 族压根没有合并档案（默认跑法在训练前就退出），所以它每一档都必须有自己的本子。
OUT_SUFFIX = {"中数据4500": "_mid", "低数据2000": "_low"}
# 学生架构是这本账的判别变量，后缀只在这一个函数里出现（两处名字各写一套就是第二把尺子）。
# mlp 不贴后缀：板块那 30 本账的名字是历史，改一次名就等于把所有在册账的身份全换掉。
FAMILY_SUFFIX = {"mlp": "", "cnn": "_cnn"}


def book_name(nseeds: int) -> Path:
    """这一族、这一档、这副宽度、这条流数的那本账叫什么名字——唯一来源。"""
    if TIER == "":
        suffix = ""
    elif TIER in OUT_SUFFIX:
        suffix = OUT_SUFFIX[TIER]
    else:
        raise SystemExit(f"--tier {TIER!r} 没有登记档案名：它要么该并进合并档案（那别用 --tier），"
                         f"要么该有自己的档案名（在这里登记）。回落成通用文件名等于把两档记进同一本")
    stem = "compress_report_v4b" if WIDTH == 128 else f"compress_report_v4b_s{WIDTH}"
    # 流数是这本账的判别维度：补过流那几本若写回同名文件，就无声盖掉了板块一路在用的那把 3 点底
    return HERE / (stem + FAMILY_SUFFIX[ARCH] + suffix
                   + ("" if nseeds == 3 else f"_n{nseeds}") + ".json")


def out_path() -> Path:
    return book_name(NSEEDS)


def ref_path() -> Path:
    """跨脚本对账的基准：**同一族、同一档、同一副宽度**那本 3 条流的账。

    旧写法把文件名写死成 `compress_report_v4b.json`（合并账，只有满数据与低数据两档），
    于是 `--tier 中数据4500 --nseeds 8` 这个组合根本没有基准——它会在训练之后才炸
    "没有记录可比"。这一支此前从未跑到过（板块从没给 mid 那本补过流数），
    换模型族那一发第一次要跑它，所以先把基准名接到 book_name(3) 上。"""
    base = book_name(3)
    if BLIND_FAMILY:            # 拆红专用：把基准强行接回 MLP 那本，模拟"名字接错家族"
        base = base.with_name(base.name.replace(FAMILY_SUFFIX[ARCH], ""))
    return base



def one_pair(teacher: torch.nn.Module, xd, yd, soft, xv, yv, xte, yte, seed: int) -> dict:
    """一个种子内的 CE/KD 配对：与第4课同款（同种子重播初始化 + 私有 Generator 批次流）。
    两臂共用同一个 ARCH ⇒ 换学生架构不改"CE 与 KD 只差一个目标函数"这条配对定义。"""
    preds = {}
    for mode in ("CE", "KD"):
        m, _ = train_student(WIDTH, xd, yd, soft, mode, xv, yv, seed=seed, arch=ARCH)
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


def committed(tier: str) -> dict | None:
    """第4课那一轮的 committed 精度值（种子 42 逐位对账用）。

    它**只对 mlp 学生存在**：ARCHIVE 里那些数是 S128/S64 那两副 MLP 跑出来的，换一副架构
    就没有任何数可比。按家族分叉写在这一个函数里，是为了让"这一本没有独立于本课的锚点"
    成为算出来的事实，而不是靠人在 main 里记得每处都加 if。"""
    if ARCH == "mlp" or BLIND_COMMIT:      # 后者是拆红专用：假装忘了按家族分叉
        return ARCHIVE[WIDTH].get(tier)
    return None


def stu_tag() -> str:
    """这一本的学生叫什么：宽度旋钮只对 MLP 有定义，CNN 那一副没有"多宽"这件事。"""
    return f"S{WIDTH}" if ARCH == "mlp" else "CNN学生（无宽度旋钮）"


def main() -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        raise SystemExit("缺 teacher 权重，先跑 python train_baseline.py")
    seeds = flows(NSEEDS)
    check_flow_constants(len(seeds))    # 缺系数在这一步就停：KeyError 会炸在 64 次训练之后
    # 补过流那几本的全部价值在于它们能与老账**并起来读**（同一批流、同一顺序）。所以训练之前先立一道
    # 跨脚本闸：老 3 条流必须与已入库那本逐位相同。不相同只有两种解释——代码漂了，或这批流不是那批流；
    # 两种都不允许把新底报成"同一把尺子量得更精"。
    v4b_ref = ref_path()
    if NSEEDS > 3:
        if not v4b_ref.exists():
            raise SystemExit(f"--nseeds {NSEEDS} 要拿 {v4b_ref.name} 当跨脚本基准，它不在盘上：先确认那一族那一档的 3 条流底在不在")
        ref_doc = json.loads(v4b_ref.read_text(encoding="utf-8"))
        # 家族对账放在数值对账之前：基准那本自己记着它是哪一族（老账没这个字段=mlp 那族）。
        # 这一道是零训练的，先拦掉才轮得到下面那烧十几分钟的数值闸——两道闸各管一段。
        ref_arch = ref_doc.get("student_arch", "mlp")
        if ref_arch != ARCH and not BLIND_FAMILY:
            raise SystemExit(f"[家族对账] {v4b_ref.name} 记的学生架构是 {ref_arch}，本课是 {ARCH}："
                             f"跨脚本基准必须同族，否则老 3 条流本来就对不上，红的是接错线不是代码漂")
        ref = ref_doc["arms"]
        for tier in (TIERS if TIER == "" else {TIER: None}):
            for s in SEEDS:
                if f"{tier} 种子{s}" not in ref:
                    raise SystemExit(f"跨脚本闸没有记录可比（{tier} 种子{s}）：空表在 all() 下恒真，那不算测过")
    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()

    selected = TIERS if TIER == "" else {TIER: DATA_TIERS[TIER]}
    archivable = any(committed(t) for t in selected)
    print(f"[先立仪器] 学生架构 = {ARCH}"
          + ("（宽度旋钮只对 MLP 学生有定义，CNN 学生那一副没有宽度可言）" if ARCH == "cnn" else f" · S{WIDTH}")
          + "。"
          + ("种子 42 必须逐位复现第4课那一轮，否则是本课把 train_student 改坏了："
             f"对照 {V4.name}（committed）" if archivable else
             f"本次只跑 {TIER}（第4课没有这一档的档案）→ 对账栏无值可比，"
             "代码是否漂移由默认跑法代测；本轮只交付一件仪器：这一档的种子底"))
    parity: dict[str, tuple] = {}
    parity_na: list[str] = []
    tiers: dict[str, list[dict]] = {}
    xcheck: dict[str, str] = {}
    for tier, n_data in selected.items():
        arch = committed(tier)
        if arch is None and TIER == "":
            continue        # 默认跑法只跑第4课配过对的那几档，不往合并档案里塞新档
        if arch is None:
            print(f"  [{tier}] 学生架构 {ARCH} 在第4课没有 committed 读数"
                  f"（{'S64 低数据档第4课本就没跑' if ARCH == 'mlp' else '那本账全是 MLP 学生的'}）"
                  f"→ 没有可逐位对账的档案值，parity 记 n/a；本课在这里只产**种子底**，不产复现")
        xd, yd = xt_all[:n_data], torch.tensor(yt_all[:n_data])
        soft = logits_of(teacher, xd)             # 软标签与随机流无关：各种子共用同一份
        rows = [one_pair(teacher, xd, yd, soft, xv, yv, xte, yte, s) for s in seeds]
        tiers[tier] = rows
        if NSEEDS > 3:                              # 跨脚本闸：老 3 条流逐位，才配把新买的那几条报成"同一把尺量得更精"
            for r in rows[:len(SEEDS)]:
                o = ref[f"{tier} 种子{r['seed']}"]
                four = (r["wrong_ce"], r["wrong_kd"], r["flips"]["broken"], r["flips"]["fixed"])
                if four != (o["wrong_ce"], o["wrong_kd"], o["flips"]["broken"], o["flips"]["fixed"]):
                    raise SystemExit(f"[跨脚本对账] FAIL {tier} 种子{r['seed']}：本发（学生架构 {ARCH}）{four} vs "
                                     f"{v4b_ref.name} {tuple(o[k] for k in ('wrong_ce', 'wrong_kd')) + (o['flips']['broken'], o['flips']['fixed'])}"
                                     f"\n老 3 条流没复现老账 → 这 {NSEEDS} 条与那 3 条不可并读，"
                                     f"先查 train_student/load_split 有没有被改坏（基准那本已由上面那道家族闸核过）")
            xcheck[f"{tier} 老3条流×4项读数"] = "逐位相同"
        for r in rows:
            if r["seed"] == 42:
                for mode in ("CE", "KD"):
                    v = r[f"acc_{mode.lower()}"]
                    if arch is None:
                        parity_na.append(f"{tier} {mode}")
                    else:
                        parity[f"{tier} {mode}"] = (v == arch[mode], v, arch[mode])
        print(f"\n[{tier} · {stu_tag()}] {len(seeds)} 个随机流下的 KD−CE 配对（10000 张，翻牌逐图算）：")
        for r in rows:
            ok = "  ←第4课那一轮" if (r["seed"] == 42 and arch) else ""
            print(f"  种子 {r['seed']:<5} CE {r['acc_ce']:.4f}（错 {r['wrong_ce']}） | "
                  f"KD {r['acc_kd']:.4f}（错 {r['wrong_kd']}） | gap {r['gap_pp']:+.2f}pp "
                  f"（翻坏 {r['flips']['broken']}/翻正 {r['flips']['fixed']}，"
                  f"p={r['p_sign']:.4g}）{ok}")

    bad = {k: v for k, v in parity.items() if not v[0]}
    na = f"（另有 {len(parity_na)} 项无档案可对：{ '、'.join(parity_na) }）" if parity_na else ""
    if bad:
        raise SystemExit(f"[仪器对账] FAIL：{bad}\n种子 42 没复现第4课 → 这一轮的数不能当复现用，"
                         f"先查是不是 train_student/load_split 被改坏了")
    if parity:
        print(f"\n[仪器对账] 逐位复现 {len(parity)}/{len(parity)} 通过 → 换种子测到的确实是随机流，不是代码漂了{na}")
    else:
        print(f"\n[仪器对账] **本轮没有对账可做**（{len(parity_na)} 项全 n/a：这一档第4课没跑过，"
              f"没有档案值可逐位比）→ 这一栏没测到任何事，别当闸门过了；"
              f"本课在这里只产**种子底**，「代码有没有漂」由默认跑法（有档案那两档）代测")
    if xcheck:
        print(f"[跨脚本对账] { '；'.join(xcheck) } —— 这 {len(xcheck)} 本老账逐位相同，"
              f"所以新 5 条流与老 3 条是同一把尺子的更精读数，不是换了一把")

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
        v["floor_is_n_dependent"] = (
            f"底是 max−min 型统计量，随流数机械变宽：n={len(seeds)} 时它必然 ≥ n=3 时的值，"
            f"所以 gap_over_floor 这个比值**跨本不可比**，换流数的那一发不能用它翻案或定案")
        v["paired_gap"] = summarize(gaps, f"{tier} · {stu_tag()} 按流配对")
        pg = v["paired_gap"]
        print(f"\n[判决 {tier}] gap 跨种子 {v['gaps_imgs']} 张 → 极差 {v['gap_spread']['range']} 张、"
              f"均值 {v['gap_spread']['mean']:+.1f} 张")
        print(f"  · 符号维度的复现：{v['same_dir_seeds']} 个种子同向，种子层二项 p={v['p_seeds']}"
              f"（{len(seeds)} 条流全同号也只对应 p={2 * 0.5 ** len(seeds):.4g}，**方向稳不等于种子层显著**）")
        print(f"  · {len(seeds)} 次复现的翻牌合并：坏 {v['pooled_flips']['broken']} / 正 {v['pooled_flips']['fixed']}，"
              f"p={v['p_pooled']:.4g}（可合并的前提：它们是同一份数据的不同随机流，交换性成立）")
        print(f"  · **种子底：同一目标函数只换随机流，错图数极差 CE {v['seed_floor_imgs']['CE']} 张 / "
              f"KD {v['seed_floor_imgs']['KD']} 张**——以后任何单臂读数小于这个数，在这台机器上都不算疗效")
        print(f"  · 疗效对底：|均值 gap| {abs(v['gap_spread']['mean']):.1f} 张 ÷ 种子底 {floor} 张 = "
              f"{v['gap_over_floor']}x → "
              + ("**扛住换随机流**" if v["survives_seed"] else "**没扛住：幅度与换一次权重的抖动同量级，不算疗效**"))
        print(f"  · 同一条腿换配对尺子（这 {len(seeds)} 条 gap 是**同起点同批次流**的配对差，"
              f"该用它自己的 σ 而不是单臂极差）：均值 {pg['mean_gap_imgs']:+.1f} 张、"
              f"σ {pg['sd_imgs']} 张、95% CI [{pg['ci95_imgs'][0]:+.1f}, {pg['ci95_imgs'][1]:+.1f}]、"
              f"{pg['same_sign_seeds']} 同号、二项 p={pg['p_seeds_binomial']}")
        if not v["survives_seed"] and pg["ci95_imgs"][0] * pg["ci95_imgs"][1] > 0:
            print(f"       ⚠ **两把尺子判反了，而且反得可以预期**：极差是 max−min，流越多它越宽（这本 {floor} 张 vs "
                  f"3 条流那本的同档底），所以『疗效 ÷ 极差』这个比值会随着你补流**自己往下掉**——"
                  f"它是 n 依赖的分母，不能拿来翻案。承重的是上面那条配对 CI：它不含 0，"
                  f"说明『换随机流之后 KD 与 CE 谁更强』这件事**有方向**；"
                  f"而『比一次重训的抖动大』这件事在这个 n 上没通过，报出来必须连着两把尺子一起报。")

    print("\n[自我审计① 重跑不等于复现] 第4课固定种子，精度/翻牌/KB 跨运行逐位相同，"
          "所以「再跑一遍 step4」只能量到墙钟漂移、量不到种子方差；"
          "这一课必须改随机流才是还那笔债——别把「跑过两次」当成「复现过」。")
    print(f"[自我审计② 两层误差别混] 图配对检验（p 很小）说的是「同一副权重下 KD 与 CE 谁判对更多」，"
          f"种子层（n={len(seeds)}）说的是「换一副权重还成不成立」。前者管系统误差、后者管重训方差，"
          "两者不能互相顶替，也不能乘在一起吹成「p 更小」。")
    print(f"[自我审计③ 仍未做] 每种子只训一次（种子底本身没有二次底，n={len(seeds)} 个点的极差是欠估的）；"
          + ("另一副宽度 S64 未换种子。" if WIDTH == 128
             else "S128 已复现，但 S64 低数据档第4课本就没跑，无从复现。")
          + "α/T 扫参见 step4c_alpha_T_sweep.py。")
    pool = "、".join(f"{t} {v['pooled_flips']['broken']}+{v['pooled_flips']['fixed']}"
                     for t, v in verdicts.items())
    print(f"[自我审计④ 合并 p 偏乐观] {len(seeds)} 个种子吃的是同一批 10000 张测试图：同一张图可以在多个种子里都翻坏，"
          f"所以合并起来的翻牌数（{pool}）不是独立事件，p_pooled 只是「比单种子更稳的方向感」，"
          f"不能当第二个独立复现。真正干净的复现统计只有上面那行「{len(seeds)} 个种子同向」。")

    report = {"seeds": list(seeds), "cross_script_parity": xcheck, "width": WIDTH, "test_n": int(len(yte)),
              "student_arch": ARCH, "student_tag": stu_tag(),
              "tier_arg": TIER or "默认（第4课两档）", "n_seeds_arg": NSEEDS,
              "parity_no_archive": parity_na,
              "train_fit_n": cut, "parity_seed42_vs_archive":
                  {k: {"match": v[0], "this_run": v[1], "archive": v[2]} for k, v in parity.items()},
              "arms": {f"{t} 种子{s}": r for t, rows in tiers.items() for s, r in zip(seeds, rows)},
              "verdicts": verdicts,
              "limitations": [f"每种子只训一次：种子底由 {len(seeds)} 个点的极差估，仍是欠估",
                              f"只复现了 {stu_tag()} + 档位 {list(tiers)}；"
                              f"其余臂（{'S64 满数据' if WIDTH == 128 else '低数据档 S64 本就没跑'}）未换种子",
                              "α/T 未扫参（本课只测随机流；扫参见 step4c_alpha_T_sweep.py）",
                              "p_pooled 偏乐观：多个种子共用同一批 10000 张测试图，翻牌事件跨种子不独立",
                              "本轮不测墙钟：精度是确定性读数，墙钟要另开交错配对流程",
                              "种子底用的是 max−min 型极差：它随流数机械变宽，所以 gap_over_floor 只能在同一本账内部读，拿 3 条流那本与 8 条流这本的比值互相翻案就是拿 n 当效应"]}
    if parity_na:
        report["limitations"].append(
            f"{'、'.join(parity_na)} 没有第4课档案可对：这一档只产种子底，"
            f"「换随机流还同向吗」这件事对它没有历史判决可复核")
    n_train = len(seeds) * 2 * len(tiers)
    if not tiers:
        raise SystemExit(f"[空账闸] 这一轮没有任何一档产出臂行（tiers 空）→ {out_path().name} 不能落盘。"
                         f"它是「{'--arch cnn 必须带 --tier' if ARCH == 'cnn' else '默认跑法只跑有档案那几档'}」"
                         f"那道提前闸的备份：提前闸被撤掉时，这里必须红，否则写出去的是一本没有读数的账")
    out_path().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out_path().name}（{n_train} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="第4课判决的种子复现（默认 S128 两档）")
    ap.add_argument("--width", type=int, choices=(128, 64), default=128,
                    help="学生 MLP 宽度；64 只复现满数据档（第4课低数据档没跑 S64）")
    ap.add_argument("--tier", choices=tuple(DATA_TIERS), default="",
                    help="只复现这一档（给扫参线造种子底）。第4课那两档走默认合并档案，"
                         "不在这里单跑——单跑一次就把另一档的账抹了")
    ap.add_argument("--nseeds", type=int, choices=flow_tiers(), default=3,
                    help="8/12/16/26 = 把这本种子底扩到对应那批流（名单见 step4_distill）；"
                         "档案名带 _n<流数>，与 3 条流那本并存不覆盖")
    ap.add_argument("--arch", choices=STUDENT_ARCHS, default="mlp",
                    help="学生架构。cnn = 换模型族那一发：学生换成与老师**同架构**的那副 CNN；"
                         "此时 --width 无定义，且必须带 --tier（默认合并账是 MLP 的读数）")
    ap.add_argument("--blankrun", action="store_true",
                    help="**拆红专用**：撤掉「cnn 不许走默认合并跑法」那道提前闸 ⇒ 该由写盘前的空账闸接住"
                         "（两道闸互为备份，红在第二道上）")
    ap.add_argument("--blindcommit", action="store_true",
                    help="**拆红专用**：让 committed() 无视家族，把 MLP 的档案值当成 cnn 的锚点 ⇒ 只有仪器对账该红")
    ap.add_argument("--blindfamily", action="store_true",
                    help="**拆红专用**：把跨脚本对账的基准强行回落到 MLP 那本 ⇒ 老 3 条流"
                         "对不上账，证明基准名真的按家族走而不是恰好都能对上")
    args = ap.parse_args()
    ARCH = args.arch
    BLANK_RUN, BLIND_COMMIT, BLIND_FAMILY = args.blankrun, args.blindcommit, args.blindfamily
    # 闸门必须在这里、训练之前：out_path() 是流程最后一行，那时才炸就白烧十几分钟 CPU
    if ARCH == "cnn" and args.width != 128:
        raise SystemExit("--arch cnn 没有宽度可言：build_cnn() 是固定架构，"
                         f"--width {args.width} 会让档案名（_s{args.width}_cnn）指着一个与 _cnn 完全同实验的对象"
                         "——两本名字不同的账共用一个身份，正是这个板块在 v4b_path 上摔过的那一类错")
    if ARCH == "cnn" and args.tier == "" and not (BLANK_RUN or BLIND_COMMIT):
        raise SystemExit("--arch cnn 不能走默认合并跑法（要带 --tier）：合并档案 "
                         f"{(HERE / 'compress_report_v4b.json').name} 那两档全是 MLP 学生的 committed 读数，"
                         f"换架构以后无值可对，默认跑法只会产出一本空账")
        # 两档拆红各自撤掉这道闸，但红点不同，正好量出两道备份各管一段：
        #   --blankrun    只撤这一道（家族分叉还在）⇒ 没有任何一档可跑 ⇒ 写盘前的[空账闸]红，零训练
        #   --blindcommit 连 committed() 的家族分叉一起撤（同一个错误的两面）⇒ 真去训练 12 次
        #                 ⇒ 由[仪器对账]在数值层抓住"借了别族的锚点"
    if args.width == 64 and args.tier in TIERS:
        # 旧写法在这里一刀切拒绝 --width 64，写死的假设是"S64 只跑过满数据档"。第十五发要在这副
        # 宽度上造中数据档的底，所以放行条件换成"这个组合会不会撞名/有没有档案名"：撞名由紧接着的
        # exists() 拦，没登记后缀的档由 out_path() 拦，都不必在这里连坐整副宽度。
        raise SystemExit(f"--tier {args.tier} 属于 S128 那两档的合并档案，S64 这一副没有它的位置："
                         f"要老档案就别带 --tier，要新档请先在 OUT_SUFFIX 里登记档案名")
    if args.tier in TIERS and ARCH == "mlp":
        raise SystemExit(f"--tier {args.tier} 是第4课那两档之一：它属于合并档案 "
                         f"compress_report_v4b.json（两档一起写）。单跑一档会把另一档的账抹掉——"
                         f"要老档案就别带 --tier")
    if args.width != 128 and args.tier and not ARCHIVE[args.width].get(args.tier):
        # 少一道闸必须说在训练之前：这一档在这一副宽度上没有第4课的 committed 读数，于是
        # "种子 42 逐位复现"那条独立对账线**结构上不存在**，这一本只剩自己的极差底。
        print(f"[闸缺] S{args.width} × {args.tier} 没有第4课的档案值可逐位对账"
              f"（ARCHIVE 这一副只登记了 {list(ARCHIVE[args.width])}）⇒ 这一本底没有独立于本课的锚点")
    WIDTH = args.width
    TIER = args.tier
    NSEEDS = args.nseeds
    if out_path().exists():
        raise SystemExit(f"{out_path().name} 已入库：重跑会无声盖掉那本种子底（板块所有判决的闸门都建在它上面）。"
                         f"要重跑得先自己改名或删档，并说清为什么")
    main()
