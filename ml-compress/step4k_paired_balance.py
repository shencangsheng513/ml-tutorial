"""第4课k（零训练）：把第4课j 落地⑤ 那句"配平换掉了多少张"从**两行相减**还成一次**配对读数**。

配方不是这里发明的，是 `step4j_matched_strength.py` 落地⑤ 写死的那一段，这里逐字兑现：
    把 v4j 档 `matched_ladder["α=0.05"]` 每条记录的 `wrong_kd`（= T=6 列在 α'=0.0312 上重训的结果）
    与 `v4f_mid_n8_t6_v5.json` 的 `arms["α0.05 种子<seed>"]["wrong_kd"]`（= 同一列在 α=0.05 上的结果）
    按同 8 条流配对，报"配平换掉了多少张"。两头都是 T=6 列、同一批流、闸 V 已证同一台训练器。

为什么这一发值一次零训练：落地⑤ 那两行（未配平 −28.1 / 等力度 −32.9）各自的 CI 大幅重叠
（[−42.3,−14.0] vs [−42.3,−23.5]），拿它们相减得到的 4.8 张**没有误差棒**，所以那句当时只能记
"未判定"。把同一批流的两条错图数直接相减，T=3 臂与 CE 臂都在配对里抵消 ⇒ 剩下的才是量"配平"这件事
的那个统计量，它有自己的 CI、同号数与二项 p。

出口（**跑之前**写死在这里，事后只许追加"落地"）：
  (a) 配对 CI **不含 0** 且均值为正 ⇒ "配平这一步确实往深里走了 N 张"从此是配对读数，
      落地⑤ 那句"未判定"就地作废；
  (b) 配对 CI **含 0** ⇒ 债还清了（有 CI 可报了），但结论仍是**未判定**，而且这一回是**测过之后**
      的未判定：那 4.8 张落在同批流的抽样噪声里。两种"未判定"必须分清——落地⑤ 当时是"没算"，
      这一发是"算了、测不动"；
  (c) 配对 CI 不含 0 且均值为负 ⇒ 与 v4j 那两行点估计**反号**（配平其实减轻了伤害）
      ⇒ 本脚本立刻当红停下：要么第4课j 的账有错，要么配对配错了流，两种都不许留下判词。

仪器（四道闸，全在读档，一次训练都不做）：
  闸 I   三本账的流名单必须**逐位同序**且等于登记处 `FLOW_SEEDS[8]` —— 配对的资格；
  闸 II  逐流恒等式：用三本账的原始 `wrong_kd` 现算 u_s = T3_s − T6@0.05_s、m_s = T3_s − T6@0.0312_s，
         必须与 v4j 在册的两列 gaps（以及 `matched_ladder` 的 `d_imgs`）**逐位相同** ——
         这一条证的是"T3 那一臂在两行里真是同一个数"，否则 u−m 不等于 T6 两头的差，配对就配错了东西；
  闸 III 两行自己的均值复现在册（四舍五入到 0.1 后逐字相等）—— 对账比的是档案真值，不是这里的字面量；
  闸 IV  配对差的均值必须**精确**等于两行均值之差（整数流级恒等式，不引入第二个数）。

落地（跑完之后追加，上面预登记原文一个字不动）：
  ① 四道闸全绿、**0 次训练、零 API 费**（档案 `cost.total = 0`）：闸 I 8 条流逐位同序且等于
     `FLOW_SEEDS[8]`；闸 II 逐流恒等式 u/m 复现在册 **8/8 逐位**；闸 III 两行均值现算 =
     在册 **−28.1 / −32.9**（四舍五入到 0.1 后逐字相等）；闸 IV 配对差之和 **38 == 两行之差**
     （= 4.8 × 8 的整数版本，所以出口那句 "+4.8 张" 确实是同一批流里搬出来的，不是新算的）。
  ② **出口走 (b)**：配对读数 = **+4.8 张、CI [−2.7, +12.2]、半宽 7.4、同号 6/8（+）、
     二项 p=0.2891、σ̂(D4)=9.5 张** ⇒ 落地⑤ 那句"未判定"**兑现的方式是给它算了误差棒，
     而误差棒吃下了它**。这一发把那句话的性质改写了：从"没算"变成"算了、测不动"，
     两者都写作"未判定"但不是同一件事，今后引用必须带这个区分。
  ③ 逐流 8 个数原样入档（`per_flow_diff_imgs` = 18, −4, 5, 4, −9, 15, 2, 7；极差 27 张）：
     正负两侧都住着流，所以 6/8 那个同号数**不是**"八条流都往一个方向走"的形态，
     不许顺嘴写成"多数流一致"以外的更强表述。
  ④ 三条不许读成的边界（都写进档案 `limitations`）：⑴ 这个数只属于 **中数据4500 × S128 ×
     MLP × α=0.05 那一格**（其余四格 v4j 没留同列原始错图数口径，本发不顺手扩），
     所以它不是"配平的平均代价"；⑵ 配对只消掉 T=3 臂与 CE 臂，**消不掉两个 T=6 臂各自的
     训练噪声**，剩下那份就是 σ̂(D4)=9.5；⑶ 两行 CI 重叠而配对 CI 含 0 **不是矛盾**，
     是同批流相减消共同噪声的那个方向。
  ⑤ 与第廿五发的关系（写在这里防串账）：本发还的是**同族内**"两行相减没误差棒"那条债；
     第廿五发 `step4l` 还的是 `step4f` 落地⑤ 那条**跨族**（MLP↔CNN 偏移）债，
     两笔债的出口分别是 (b) 与 (a)，**不许互相引用成对方的结论**。
"""
from __future__ import annotations

import json
from pathlib import Path

from step4_distill import FLOW_SEEDS, summarize

HERE = Path(__file__).resolve().parent
V4J = "compress_report_v4j_matched_n8.json"
T6_BOOK = "compress_report_v4f_mid_n8_t6_v5.json"      # T=6 列（α=0.05 那一头 + 它自己的 CE 臂）
T3_BOOK = "compress_report_v4f_mid_n8_v5.json"         # T=3 列：v4j 两行共用的那一臂，这里只用来核恒等式


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def arm_wrong(book: dict, alpha: float, seed: int) -> int:
    key = f"α{alpha:g} 种子{seed}"
    if key not in book["arms"]:
        raise SystemExit(f"{book.get('_name', '?')} 里没有 {key} 这一格（有 "
                         f"{'、'.join(sorted({k.split(' 种子')[0] for k in book['arms']})[:6])}…）")
    return int(book["arms"][key]["wrong_kd"])


def main() -> None:
    v4j, t6, t3 = read(V4J), read(T6_BOOK), read(T3_BOOK)
    for b, n in ((v4j, V4J), (t6, T6_BOOK), (t3, T3_BOOK)):
        b["_name"] = n
    seeds = [int(s) for s in v4j["seeds"]]
    rung = float(v4j["primary_rung"])                       # 点名主格 = 跑之前定那一格，不在这里改
    rows = {int(r["seed"]): r for r in v4j["matched_ladder"][f"α={rung:g}"]}
    a_hi = float(v4j["paired_diff"][f"α={rung:g}"]["alpha_matched_T6"])   # α'=0.0312，从档案读不写死

    # ---- 闸 I：配对的资格 ----
    for b in (t6, t3):
        if [int(s) for s in b["seeds"]] != seeds:
            raise SystemExit(f"闸 I {b['_name']} 的流名单与 {V4J} 不逐位相同 ⇒ 这两本不可配对，本发停")
    if seeds != list(FLOW_SEEDS[len(seeds)]):
        raise SystemExit(f"闸 I 那 {len(seeds)} 条流不是登记处 `FLOW_SEEDS[{len(seeds)}]` 那一档 ⇒ 换尺子了")
    if len(rows) != len(seeds):
        raise SystemExit(f"闸 I {V4J} 的 matched_ladder[α={rung:g}] 有 {len(rows)} 条，流数 {len(seeds)} ⇒ 缺流")
    print(f"[闸 I] 三本账同 {len(seeds)} 条流、同序、等于登记处那一档 ⇒ 有配对资格")

    # ---- 逐流配对 + 闸 II：先证明 T3 那一臂在两行里是同一个数 ----
    u_row = v4j["unmatched_cross_column_diff_read_from_books"][f"α={rung:g}"]
    m_row = v4j["paired_diff"][f"α={rung:g}"]["paired"]      # 这一格外面还套一层（剂量与 CI 旗在那层）
    diff, u_gaps, m_gaps = [], [], []
    for i, s in enumerate(seeds):
        w_t3 = arm_wrong(t3, rung, s)
        w_un = arm_wrong(t6, rung, s)                        # T=6 列 @ α=0.05（未配平那一头）
        w_mt = int(rows[s]["wrong_kd"])                      # T=6 列 @ α'=0.0312（配平后重训）
        if w_t3 != int(rows[s]["wrong_kd_T3"]):
            raise SystemExit(f"闸 II 种子{s}：T=3 臂在 v4j 记的是 {rows[s]['wrong_kd_T3']}，"
                             f"现读 {T3_BOOK} 是 {w_t3} ⇒ 两行共用的那一臂不是同一个数，u−m 不是配对差")
        u, m = w_t3 - w_un, w_t3 - w_mt
        if u != int(u_row["gaps"][i]) or m != int(m_row["gaps"][i]) or m != int(rows[s]["d_imgs"]):
            raise SystemExit(f"闸 II 种子{s}：现算 u={u} / m={m} 与在册 {u_row['gaps'][i]} / "
                             f"{m_row['gaps'][i]}（d_imgs {rows[s]['d_imgs']}）不逐位相同 ⇒ 配对配错了流")
        if w_mt - w_un != u - m:
            raise SystemExit(f"闸 II 种子{s}：T6 两头之差 {w_mt - w_un} ≠ u−m {u - m}")
        diff.append(w_mt - w_un)
        u_gaps.append(u)
        m_gaps.append(m)
    print(f"[闸 II] {len(seeds)} 条流逐位命中：u_s=T3−T6@{rung:g} 与 m_s=T3−T6@{a_hi:g} 都等于在册那一列"
          f"，且 (T6 两头之差) == u−m ⇒ 配对差的定义域成立")

    # ---- 闸 III / 闸 IV：对账比档案真值 ----
    for tag, mine, booked in (("未配平", u_gaps, u_row), ("等力度", m_gaps, m_row)):
        if round(sum(mine) / len(mine), 1) != float(booked["mean_gap_imgs"]):
            raise SystemExit(f"闸 III {tag}那行现算均值 {sum(mine) / len(mine):.1f} ≠ 在册 "
                              f"{booked['mean_gap_imgs']} ⇒ 复现失败，本发不停在判词上而停在仪器上")
    st = summarize(diff, f"α={rung:g} 配平换掉了多少张（T6 列内配对：wrong_kd(α'={a_hi:g}) − wrong_kd({rung:g}）")
    if sum(diff) != sum(u_gaps) - sum(m_gaps):
        raise SystemExit(f"闸 IV 配对差之和 {sum(diff)} ≠ 两行之差 {sum(u_gaps) - sum(m_gaps)}")
    print(f"[闸 III+IV] 两行各自复现在册（{u_row['mean_gap_imgs']} / {m_row['mean_gap_imgs']} 张），"
          f"配对差之和 {sum(diff)} == 两行之差 ⇒ 那个数是同一批流里搬出来的，不是新算的")

    half = round((st["ci95_imgs"][1] - st["ci95_imgs"][0]) / 2, 1)
    crosses = st["ci95_imgs"][0] <= 0 <= st["ci95_imgs"][1]
    if st["mean_gap_imgs"] < 0 and not crosses:
        raise SystemExit(f"出口 (c)：配对 CI [{st['ci95_imgs'][0]}, {st['ci95_imgs'][1]}] 不含 0 且均值为负 "
                         f"⇒ 与 v4j 那两行点估计反号（配平反而减轻了伤害）。本发不许留判词："
                         f"先查 {V4J} 的账或这里的配对")
    exit_rule = (
        f"(a) 配对读数落地：配平那一步让 T=6 那一头多错 {st['mean_gap_imgs']:+.1f} 张"
        f"（CI [{st['ci95_imgs'][0]}, {st['ci95_imgs'][1]}] 不含 0、同号 {st['same_sign_seeds']}、"
        f"二项 p={st['p_seeds_binomial']}）⇒ 落地⑤ 那句『未判定（没算 CI）』就地作废"
        if not crosses else
        f"(b) 债还清、结论仍是未判定，但这一回是**测过之后**的未判定：配平换掉的 "
        f"{st['mean_gap_imgs']:+.1f} 张落在抽样噪声里（CI [{st['ci95_imgs'][0]}, {st['ci95_imgs'][1]}] 含 0、"
        f"同号 {st['same_sign_seeds']}、二项 p={st['p_seeds_binomial']}）⇒ 落地⑤ 那句『未判定』的原意（两行相减"
        f"没有误差棒）已被兑现，兑现的方式是给它算了误差棒，而误差棒吃下了它")
    print(f"\n[出口（预登记）] {exit_rule}")

    report = {"quantity": f"配平这一步换掉的张数：同一列（T={6:g}）、同 {len(seeds)} 条流上"
                          f" wrong_kd(α'={a_hi:g}) − wrong_kd(α={rung:g}) 的逐流配对差"
                          "（正 = 钉成同力度之后 T=6 那一头反而多错）",
              "pays": f"{V4J} 落地⑤（『两行相减不是配对统计量』那条欠账，配方写死在 step4j 落地⑤）",
              "tier": t6["tier"], "width": t6["width"], "student_arch": t6.get("student_arch", "mlp"),
              "seeds": seeds, "primary_rung": rung, "alpha_matched_T6": a_hi,
              "archives": {"v4j_matched": V4J, "T6_ladder": T6_BOOK, "T3_ladder": T3_BOOK},
              "gates": {"同批流同序且等于登记处": f"{len(seeds)} 条",
                        "逐流恒等式 u/m 复现在册": f"{len(seeds)}/{len(seeds)} 逐位",
                        "两行均值复现在册": [u_row["mean_gap_imgs"], m_row["mean_gap_imgs"]],
                        "配对差之和等于两行之差": sum(diff)},
              "per_flow_diff_imgs": diff,
              "read_from_books": {"unmatched_T6_at_alpha": u_row["mean_gap_imgs"],
                                 "matched_T6_at_alpha_prime": m_row["mean_gap_imgs"],
                                 "difference_of_the_two_rows": round(
                                     u_row["mean_gap_imgs"] - m_row["mean_gap_imgs"], 1)},
              "paired": st, "half_width_imgs": half, "ci_crosses_zero": crosses,
              "exit": exit_rule, "verdict": exit_rule,
              "cost": {"arm_trainings": 0, "total": 0, "api_fee": 0,
                       "note": "纯读档：三本在册账的 8 组原始 wrong_kd 逐流相减，一次训练都不做"
                               "（这正是落地⑤ 把配方写成『下一发零次训练』的理由）"},
              "limitations": [
                  f"这个数只属于**一个坐标、一个剂量**：{t6['tier']} × S{t6['width']} × "
                  f"mlp 族 × α={rung:g} 那一格（其余四格 v4j 没有为它们留同列的原始错图数对账口径，"
                  "本发不顺手扩），所以它不是『配平的平均代价』",
                  "量的是**起手配平**这一步换掉了多少，不是『温度效应有百分之多少来自配平』："
                  "v4j 那条『α·‖∇(T²KL)‖ 只在第 0 批对齐』的边界在这里原样成立",
                  f"配对差与两行各自 CI 的关系不许读成矛盾：两行 CI 重叠而配对 CI {'含' if crosses else '不含'} 0，"
                  "这正是『同一批流相减会消掉共同噪声』那笔账的方向——但也只消掉 T3 与 CE 那两臂，"
                  "消不掉两个 T=6 臂各自的训练噪声（σ̂(D4) = "
                  f"{st['d4_sigma_hat']} 张就是那剩下的那份）",
                  "不改任何在册判词：这一发不进形状判据 v5 的参评对象（它量的是同列两次训练之差，"
                  "不是 α 阶梯的形状），也不进守卫的翻转集合"],
              }
    out = HERE / "compress_report_v4k_paired_balance.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（0 次训练、零 API 费）· {exit_rule[:70]}…")


if __name__ == "__main__":
    main()
