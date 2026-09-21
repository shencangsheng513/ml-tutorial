# -*- coding: utf-8 -*-
"""第4课c 的仪器复检：把"借来的种子底"换成每格自己的底，判决站得住吗？（零训练，纯读档）

第4课c 的自我审计① 自己写着：全表 10 格共用第4课b 那一格（α=0.7/T=3）的底（S128 41 张 /
S64 53 张 / S128 低数据档 37 张），"是借来的仪器，会低估 T 大端的方差"。这一支不训练，
只把三份 v4c 档案里每格三种子的原始读数翻出来，按**每格自己的配对极差**重算误差棒，
再拿同一批判决过一遍闸。

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
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALPHAS = (0.3, 0.7, 1.0)
TEMPS = (1.0, 3.0, 6.0)
ANCHOR_T, ALPHA_ANCHOR = 3.0, 0.7
D4 = 1.693                      # n=3 时正态极差的期望 = 1.693σ，用来把极差换算成 σ̂
FILES = {(128, "满数据8500"): "compress_report_v4c.json",
         (128, "低数据2000"): "compress_report_v4c_low.json",
         (64, "满数据8500"): "compress_report_v4c_s64.json"}
# gap 的符号两处一致（正=KD 少错图=KD 更好），但「配平臂换回多少」那句话在两档是反的：
# 满数据档那一列是伤害，配平换回来是好事；低数据档 T 轴整个翻了号（高温有益），
# 所以同一个负数在这里读作「钉回梯度把收益弄丢了 7 张」而不是「换回伤害」。
NOTE = {"满数据8500": "配平臂换回的是**伤害**（该档 T↑ 有害）",
        "低数据2000": "该档 T 轴已翻号（T↑ 有益），所以配平臂换回 −7.0 张要读成**丢掉 7 张收益**"}
OUT = HERE / "compress_report_v4d.json"


def key(a: float, t: float) -> str:
    return f"a{a:g}_T{t:g}"


def cell_bars(g: dict) -> dict:
    """一格自己的仪器：三种子 KD−CE 差值的极差 → σ̂ → 均值的 SE → ±2SE 就是这格自己的闸。"""
    gaps = [r["gap_imgs"] for r in g["per_seed"]]
    wrongs = [r["wrong_kd"] for r in g["per_seed"]]
    rng = max(gaps) - min(gaps)
    se = (rng / D4) / math.sqrt(len(gaps))
    return {"mean_gap_imgs": g["mean_gap_imgs"], "paired_range_imgs": rng,
            "sigma_hat_imgs": round(rng / D4, 1), "se_mean_imgs": round(se, 1),
            "gate_2se_imgs": round(2 * se, 1),
            "unpaired_kd_range_imgs": max(wrongs) - min(wrongs),
            "p_pooled": g["p_pooled"], "n_seeds": len(gaps)}


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
    return {"width": width, "borrowed_floor_imgs": floor,
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
    report: dict = {"method": "每格三种子配对差值极差 → σ̂=rng/1.693 → SE=σ̂/√3 → 闸门=±2SE；"
                            "跨格统计量按独立合成（同三种子跨格复用 ⇒ 这是下界，对备择有利）",
                    "borrowed_floor_source": "第4课b 的 α=0.7/T=3 那一格 KD 臂错图极差",
                    "archives": {}}
    for (width, tier), fname in FILES.items():
        p = HERE / fname
        if not p.exists():
            print(f"跳过 {fname}：文件不在")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if int(d["width"]) != width or d.get("tier") != tier:
            raise SystemExit(f"{fname} 声称是 S{d.get('width')} / {d.get('tier')!r}，"
                             f"文件名却写着 S{width} / {tier!r}——档案对不上，先看一眼再算")
        r = analyse(width, d)
        label = f"S{width} | {tier}"
        r["tier"] = tier
        r["note"] = NOTE[tier]
        report["archives"][label] = r
        print(f"\n===== {label}（借来的底 {r['borrowed_floor_imgs']} 张）=====")
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

    report["limitations"] = ["每格只有 3 个种子：极差本身是 σ 的粗估（±约 40%），没有二次底",
                             "三种子跨格复用同一批随机流 ⇒ 跨格统计量的 SE 是下界，倍数(己)偏乐观",
                             "逐图 p_pooled 同审计②：种子间翻牌不独立，只当方向参考",
                             "刻度只到『过不过自己的闸』，没做多重比较校正（9 条判决一起看）",
                             "本表只重算闸门，不改判：翻档的判决要么补种子（第4课e）要么降级为措辞"]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（零训练，纯读三份 v4c 档案）")


if __name__ == "__main__":
    main()
