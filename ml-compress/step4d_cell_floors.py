# -*- coding: utf-8 -*-
"""第4课c 的仪器复检：把"借来的种子底"换成每格自己的底，判决站得住吗？（零训练，纯读档）

第4课c 的自我审计① 自己写着：全表 10 格共用第4课b 那一格（α=0.7/T=3）的底（S128 41 张 /
S64 53 张 / S128 低数据档 37 张），"是借来的仪器，会低估 T 大端的方差"。这一支不训练，
只把在册的 v4c 档案（`ARCHIVES` 那份清单）里**每格自己的**配对原始读数翻出来，按每格自己的
配对极差重算误差棒，再拿同一批判决过一遍闸。

要回答两问：
  Q1 有没有哪条判决会因为换成自己的底而翻档（显著↔不显著）？
  Q2 借来的底在哪些格上偏得最多——它有没有把**真实的小效应埋掉**（底偏大同样是仪器误差）？
低数据档也在表里，是因为它给出了本板块第一条正号判决（T=6 一格 +71.7 张）：
**同一把尺子不能只对不利的读数严格**，正号判决也得过一遍别人的闸。

为什么"每格自己的底"要用配对极差而不是错图极差：本课每条统计量的单位都是 KD−CE 的差，
而 CE 是同一种子的锚（同初始权重、同批次流），差值的跨种子波动比错图数的绝对波动小，
用后者的极差当闸门是**用错了刻度**（第3课就吃过"拿老师的百分比套学生"这个亏）。

零 API 费、零训练。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4d_cell_floors.py

本轮（第十五发 D 项）换的两件事，为什么不是改名重跑：
  ① **σ̂ 的换算系数必须跟着流数走**。极差 → σ̂ 的那把尺子是 E[极差]/σ，它随 n 单调变
     （n=3 是 1.693，n=8 是 2.847，n=16 是 3.532，见 `step4_distill.D4`）。原来这一支把
     `D4 = 1.693` 写死在模块里——在 n=3 的账上它对，把同一行代码喂给 8 条流的账，σ̂ 会被
     **高估 2.847/1.693 = 1.68 倍**（分母用小了），于是每格自己的 ±2SE 闸**集体变松**。
     变松不是"更保守"而是**朝错误的一侧保守**：它恰好把本课 Q2 要抓的那件事（借来的底埋掉了
     哪些真实的小效应）又埋了一次。
     这一档专属红**实测在 Q2，不在 Q1**（8 条流那本，两种系数并排读）：
       · Q1 翻档名单 **两种系数下完全相同**（都是 3/8：T 逐档 1→3、T 逐档 3→6、反事实臂 α=0.3/T=3）
         ⇒ 拿"翻档条数没变"说这一改无所谓，是读错了这条修复钉的是哪一问。
       · Q2 被埋名单 **4 本 → 3 本**：`a0.7_T1` 平均 +6.4 张，正确的闸 6.0 张（过闸 ⇒ 它被借来的
         77 张埋着），写死的闸 10.0 张（不过闸 ⇒ 从名单上消失）。
       · 每格闸值本身逐格差 1.68 倍（4.5~15.4 张 ↔ 7.5~25.9 张），所以这条红随时可以在档案里复算：
         把 `sd.D4` 临时换成 `{n: 1.693}` 再跑一次 `analyse()` 就是错的那一版。
     现在系数从册上按**这本账自己的 per_seed 条数**现取，并走 `check_flow_constants`：缺档流数在
     最贵的事之前就停，不许回落默认值。
     **n=3 的三本账逐位未动**（与旧产物 `compress_report_v4d.json` 并排核过：每条判决的值/闸/倍数/
     翻档、每格闸值、被埋名单全部相同）⇒ 这一改的所有新读数只来自"8 条流那本终于能用"，不来自重算。
  ② **同一坐标的 3 条流与 8 条流两本账并排进表**，另把 S128×中数据、S64×中数据两本在册账纳表。
     这不加训练（六本都在盘上），但它把 Q1 从"换底翻了几条"升级成"换底 × 换流数 各翻几条"。
     旧册的档案键叫 `S128 | 满数据8500`，新册同一坐标改叫 `S128 | 满数据8500 | 3 条流`
     （多出的后缀是必需的：同一坐标现在有两本账）。
     旧产物 `compress_report_v4d.json` 是 n=3 那一批的历史读数，**不就地覆盖**，新账另起一名。
  ③ **16 条流那本在第十七发买到了**：`step4c` 的 `--nseeds` 加了 16 这一档，代价是 **176 次训练**
     （每条流 11 次 = 9 格 KD + 1 CE + 1 配平臂；n=3 那本档案自报 33 次 = 11×3，所以 16 条流是
     11×16），产物 `compress_report_v4c_n16.json`。它喂得进 `analyse()`：10 格 × 每格 16 条
     `per_seed` 配对读数，借来的底 90 张取自 `compress_report_v4b_n16.json`（同坐标同流数）。
     代价与读数分开记账：这一支本身仍然**零训练、零 API 费**，它只是把已经付过钱的档案换把尺子重读。
     纳入 16 条流这一本后，Q1「换底翻档」第一次能在**同一坐标的三把尺子**（3/8/16 条流）之间对照，
     而 Q2「被埋名单」的闸值随 n 收窄（σ̂/√n）——但注意 ③ 的警告照旧：跨流数**不许比翻档条数**。

**同一坐标跨流数不许直接比翻档条数**：借来的底自己也随流数涨（41 → 77 张），每格自己的闸由
σ̂/√n 给，两本账的"倍数"分母不同。跨流数能对着读的只有各自的配对 CI，这里两栏都报。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from step4_distill import D4, check_flow_constants     # 换算系数的唯一事实源

HERE = Path(__file__).resolve().parent
ALPHAS = (0.3, 0.7, 1.0)
TEMPS = (1.0, 3.0, 6.0)
ANCHOR_T, ALPHA_ANCHOR = 3.0, 0.7
# σ̂ 的换算系数 E[极差]/σ 是**流数的函数**，不是一常数 ⇒ 从系数册按每本账自己的 per_seed 条数现取。
# 原来这里写死 `D4 = 1.693`（n=3 那一档），见文档串第十五发 D 项那一段。
ARCHIVES = (
    {"label": "S128 | 满数据8500 | 3 条流", "file": "compress_report_v4c.json",
     "width": 128, "tier": "满数据8500"},
    {"label": "S128 | 满数据8500 | 8 条流", "file": "compress_report_v4c_n8.json",
     "width": 128, "tier": "满数据8500"},
    {"label": "S128 | 满数据8500 | 16 条流", "file": "compress_report_v4c_n16.json",
     "width": 128, "tier": "满数据8500"},
    {"label": "S128 | 中数据4500 | 3 条流", "file": "compress_report_v4c_mid.json",
     "width": 128, "tier": "中数据4500"},
    {"label": "S64 | 满数据8500 | 3 条流", "file": "compress_report_v4c_s64.json",
     "width": 64, "tier": "满数据8500"},
    {"label": "S64 | 中数据4500 | 3 条流", "file": "compress_report_v4c_s64_mid.json",
     "width": 64, "tier": "中数据4500"},
    {"label": "S128 | 低数据2000 | 3 条流", "file": "compress_report_v4c_low.json",
     "width": 128, "tier": "低数据2000"},
)
# gap 的符号两处一致（正=KD 少错图=KD 更好），但「配平臂换回多少」那句话在两档是反的：
# 满数据档那一列是伤害，配平换回来是好事；低数据档 T 轴整个翻了号（高温有益），
# 所以同一个负数在这里读作「钉回梯度把收益弄丢了 7 张」而不是「换回伤害」。
NOTE = {"满数据8500": "配平臂换回的是**伤害**（该档 T↑ 有害）",
        "中数据4500": "该档 T↑ 同样有害，配平臂换回的是伤害",
        "低数据2000": "该档 T 轴已翻号（T↑ 有益），所以配平臂换回 −7.0 张要读成**丢掉 7 张收益**"}
OUT = HERE / "compress_report_v4d_n16.json"      # 前两本 v4d（n=3 的 compress_report_v4d.json、n=8 的 _n8.json）都是历史读数，一律不覆盖


def key(a: float, t: float) -> str:
    return f"a{a:g}_T{t:g}"


def cell_bars(g: dict) -> dict:
    """一格自己的仪器：该格**自己那本账**的配对差值极差 → σ̂ → 均值的 SE → ±2SE 就是这格自己的闸。

    换算系数按流数从册上现取（n=3 → 1.693，n=8 → 2.847，…）：E[极差]/σ 随 n 变，
    拿 n=3 的系数去换算 8 条流的极差等于用小的分母去除，σ̂ 高估 1.68 倍、闸门集体变松。"""
    gaps = [r["gap_imgs"] for r in g["per_seed"]]
    wrongs = [r["wrong_kd"] for r in g["per_seed"]]
    n = len(gaps)
    check_flow_constants(n)
    rng = max(gaps) - min(gaps)
    sigma_hat = rng / D4[n]
    se = sigma_hat / math.sqrt(n)
    return {"mean_gap_imgs": g["mean_gap_imgs"], "paired_range_imgs": rng,
            "d4_divisor_imgs": D4[n],
            "sigma_hat_imgs": round(sigma_hat, 1), "se_mean_imgs": round(se, 1),
            "gate_2se_imgs": round(2 * se, 1),
            "unpaired_kd_range_imgs": max(wrongs) - min(wrongs),
            "p_pooled": g["p_pooled"], "n_seeds": n}


def combine(se: dict[str, float], terms: list[tuple[str, float]]) -> float:
    """独立合成 SE：sqrt(Σ(w·SE)^2)。三种子跨格是同一批，误差不独立，所以这是**下界**（对备择有利）。"""
    return round(math.sqrt(sum((w * se[k]) ** 2 for k, w in terms)), 1)


def reading(name: str, value: float, terms: list[tuple[str, float]], se: dict, floor: int) -> dict:
    """一条判决在两种闸门下的读数：借来的底（第4课c 用的）vs 每格自己的底（±2SE 下界）。"""
    own = combine(se, terms)
    gate = round(2 * own, 1)
    return {"statistic": name, "value_imgs": round(value, 1),
            "cells": [k for k, _ in terms],
            "borrowed_floor_imgs": floor, "under_borrowed": bool(abs(value) > floor),
            "ratio_borrowed": round(abs(value) / floor, 2),
            "own_2se_imgs": gate, "under_own": bool(abs(value) > gate),
            "ratio_own": round(abs(value) / gate, 2) if gate else None,
            "flips": bool(abs(value) > floor) != bool(abs(value) > gate)}


def analyse(width: int, report: dict) -> dict:
    grid = report["grid"]
    bars = {k: cell_bars(g) for k, g in grid.items()}
    se = {k: b["se_mean_imgs"] for k, b in bars.items()}
    at = lambda a, t: grid[key(a, t)]["mean_gap_imgs"]      # noqa: E731
    floor = int(report["seed_floor_imgs_from_v4b"])

    col = [at(ALPHAS[-1], t) for t in TEMPS]
    hi, lo = max(col), min(col)
    k_hi = key(ALPHAS[-1], TEMPS[col.index(hi)])
    k_lo = key(ALPHAS[-1], TEMPS[col.index(lo)])

    readings = [
        reading("α 轴斜率（α=1.0 − α=0.3，锚点温度）",
                at(ALPHAS[-1], ANCHOR_T) - at(ALPHAS[0], ANCHOR_T),
                [(key(ALPHAS[-1], ANCHOR_T), 1.0), (key(ALPHAS[0], ANCHOR_T), 1.0)], se, floor),
        reading(f"判别格 α={ALPHAS[-1]:g} 列内 T 跨度（最高温 − 最低温）", hi - lo,
                [(k_hi, 1.0), (k_lo, -1.0)], se, floor),
        reading("T 主效应（跨 α 平均，T=6 − T=1）",
                sum(at(a, TEMPS[-1]) - at(a, TEMPS[0]) for a in ALPHAS) / len(ALPHAS),
                [(key(a, TEMPS[-1]), 1 / len(ALPHAS)) for a in ALPHAS]
                + [(key(a, TEMPS[0]), -1 / len(ALPHAS)) for a in ALPHAS], se, floor),
    ]
    for t_lo, t_hi in zip(TEMPS[:-1], TEMPS[1:]):
        readings.append(
            reading(f"T 逐档 {t_lo:g}→{t_hi:g}（跨 α 平均）",
                    sum(at(a, t_hi) - at(a, t_lo) for a in ALPHAS) / len(ALPHAS),
                    [(key(a, t_hi), 1 / len(ALPHAS)) for a in ALPHAS]
                    + [(key(a, t_lo), -1 / len(ALPHAS)) for a in ALPHAS], se, floor))
    readings.append(reading(f"不加热只借 argmax（α={ALPHAS[-1]:g}/T={TEMPS[0]:g}）",
                            at(ALPHAS[-1], TEMPS[0]), [(key(ALPHAS[-1], TEMPS[0]), 1.0)], se, floor))
    readings.append(reading(f"第4课反事实臂（α={ALPHAS[0]:g}/T={ANCHOR_T:g}）",
                            at(ALPHAS[0], ANCHOR_T), [(key(ALPHAS[0], ANCHOR_T), 1.0)], se, floor))
    wt = report["verdicts"].get("weight_or_temperature", {})
    per_t = wt.get("per_T", {})
    if per_t:
        terms = [(f"match_T{float(t):g}", 1 / len(per_t)) for t in per_t] \
            + [(key(ALPHA_ANCHOR, float(t)), -1 / len(per_t)) for t in per_t]
        readings.append(reading("配平臂换回的伤害（钉回 KD 梯度 vs 锚点 α=0.7）",
                                sum(v["damage_recovered_imgs"] for v in per_t.values()) / len(per_t),
                                terms, se, floor))

    # Q2：借来的底把哪些格埋了——|mean| 过自己的闸但没过借来的底
    buried = {k: b for k, b in bars.items()
              if abs(b["mean_gap_imgs"]) > b["gate_2se_imgs"] and abs(b["mean_gap_imgs"]) <= floor}
    n_flows = max(b["n_seeds"] for b in bars.values())
    return {"width": width, "borrowed_floor_imgs": floor,
            "n_flows": n_flows, "d4_divisor_imgs": D4[n_flows],
            "own_bars": {str(TEMP): {str(a): bars[key(a, TEMP)]["gate_2se_imgs"]
                                    for a in ALPHAS} for TEMP in TEMPS},
            "cells": bars, "readings": readings,
            "n_flips": sum(1 for r in readings if r["flips"]),
            "flipped_readings": [r["statistic"] for r in readings if r["flips"]],
            "cells_buried_by_borrowed_floor": {k: {"mean_gap_imgs": b["mean_gap_imgs"],
                                                   "own_2se_imgs": b["gate_2se_imgs"],
                                                   "p_pooled": b["p_pooled"]}
                                               for k, b in sorted(buried.items())},
            "bar_spread": {"min_own_2se": min(b["gate_2se_imgs"] for b in bars.values()),
                           "max_own_2se": max(b["gate_2se_imgs"] for b in bars.values()),
                           "vs_borrowed": [round(min(b["gate_2se_imgs"] for b in bars.values()) / floor, 2),
                                           round(max(b["gate_2se_imgs"] for b in bars.values()) / floor, 2)]}}


def main() -> None:
    report: dict = {"method": "每格配对差值极差 → σ̂=rng/D4[n]（n=该本账自己的流数，系数取自 step4_distill 的册）"
                            "→ SE=σ̂/√n → 闸门=±2SE；跨格统计量按独立合成（同批流跨格复用 ⇒ 下界，对备择有利）",
                    "d4_divisor_by_n": {str(k): v for k, v in sorted(D4.items())},
                    "borrowed_floor_source": "第4课b 的 α=0.7/T=3 那一格 KD 臂错图极差",
                    "archives": {}}
    for spec in ARCHIVES:
        fname, label = spec["file"], spec["label"]
        p = HERE / fname
        if not p.exists():
            print(f"跳过 {fname}：文件不在")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        width, tier = spec["width"], spec["tier"]
        if int(d["width"]) != width or d.get("tier") != tier:
            raise SystemExit(f"{fname} 声称是 S{d.get('width')} / {d.get('tier')!r}，"
                             f"清单里却写着 S{width} / {tier!r}——档案对不上，先看一眼再算")
        r = analyse(width, d)
        r["tier"] = tier
        r["flows"] = r["n_flows"]
        r["file"] = fname
        r["note"] = NOTE[tier]
        report["archives"][label] = r
        print(f"\n===== {label}（借来的底 {r['borrowed_floor_imgs']} 张；"
              f"σ̂ 换算系数 D4[{r['n_flows']}]={r['d4_divisor_imgs']}）=====")
        print(f"{'判决':<38}{'值(张)':>9}{'借来的底':>10}{'自己±2SE':>10}{'倍数(借)':>10}{'倍数(己)':>10}  翻档")
        for x in r["readings"]:
            print(f"{x['statistic']:<38}{x['value_imgs']:>9}{x['borrowed_floor_imgs']:>10}"
                  f"{x['own_2se_imgs']:>10}{x['ratio_borrowed']:>10}{x['ratio_own']:>10}"
                  + ("  ←翻" if x["flips"] else ""))
        print(f"  每格自己的闸：最松 {r['bar_spread']['min_own_2se']} 张 / 最紧 {r['bar_spread']['max_own_2se']} 张"
              f"（借来的底 = 它们的 {r['bar_spread']['vs_borrowed'][0]}x ~ {r['bar_spread']['vs_borrowed'][1]}x）")
        print(f"  [读法] {r['note']}")
        if r["cells_buried_by_borrowed_floor"]:
            print("  [Q2 被借来的底埋掉的格]（过自己的 ±2SE，没过借来的底）：")
            for k, b in r["cells_buried_by_borrowed_floor"].items():
                print(f"    {k}: 平均 {b['mean_gap_imgs']:+.1f} 张，自己的闸 {b['own_2se_imgs']} 张，"
                      f"借来的 {r['borrowed_floor_imgs']} 张，逐图合并 p={b['p_pooled']}")
        else:
            print("  [Q2 被埋掉的格] 无")
        print(f"  [Q1 翻档判决数] {r['n_flips']}/{len(r['readings'])}"
              + ("：" + "；".join(r["flipped_readings"]) if r["flipped_readings"] else ""))

    report["limitations"] = ["每格自己的底由**该本账自己的流数**条配对差值算：3 条流时极差是 σ 的粗估"
                             "（±约 40%）且没有二次底；8 条流时 σ̂ 本身仍是"
                             "「一次极差 ÷ 册上系数」，不是对分布的估计",
                             "同一坐标并排的 3 条流 / 8 条流两本账**不能比翻档条数**：借来的底随流数涨"
                             "（41 → 77 张），每格自己的闸是 σ̂/√n，两栏分母不同；跨流数只能各读各的配对 CI",
                             "同批流跨格复用 ⇒ 跨格统计量的 SE 是下界，倍数(己)偏乐观",
                             "逐图 p_pooled 同审计②：种子间翻牌不独立，只当方向参考",
                             "刻度只到『过不过自己的闸』，没做多重比较校正（每本账 9 条判决一起看）",
                             "S64 × 中数据4500 那一本的底（52 张）来自第4课b 同坐标那本，而那一本"
                             "自己声明『没有第4课的档案值可逐位对账』⇒ 这一列的底没有独立于本课的锚点，"
                             "换底翻档的解释力比另外几本弱一档",
                             "本表只重算闸门，不改判：翻档的判决要么补种子（第4课e）要么降级为措辞"]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（零训练，纯读 {len(report['archives'])}/{len(ARCHIVES)} 份 v4c 档案；"
          f"旧账 compress_report_v4d.json / _v4d_n8.json 保持不动）")


if __name__ == "__main__":
    main()
