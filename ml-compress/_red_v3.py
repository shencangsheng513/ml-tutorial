# -*- coding: utf-8 -*-
"""v3/v4 形状判据的离线验红（零训练）：把已入库的 v4f 档案按 v1/v2/v3/v4 各算一遍，看翻转集合落在哪。

七道闸（沿用第四发的规矩：一道闸只钉一件修复，红要知道是哪件修复没装上）：
  ① 复算自证     每本档案**记录在案**的那句判词 = 产出它那一版的复算值（钉"喂的是同一批数字"）
  ② 翻转集合     v3 相对 v2 只许翻掉预登记的那一本（钉"v3 不是装饰，也不是连坐"）
  ③ 只翻该翻的   那一本必须从 测不动 翻到 反号线（钉翻的是对的方向）
  ④ 不连坐       板块全部在册『污染型』判词逐字不变（条数现算不写死；钉分支顺序与反号条件）
                 注意它**只按 v2/v3 复算**，所以它结构性地看不见 v4 的连坐——v4 的连坐由 ⑦ 独扛
  ⑤ 落地自证     真的 _v3 / _v4 **同流重跑**本：判词落在预登记预言上，且训练路径字段逐位不动
                 （钉"翻案是判据换来的，不是重训换来的"；换了流数=另一发实验，不在这一条里）
                 基准账 = 同流同配置里**版本号最高的前一版**（不是 glob 顺序最老那本：v1→v2 时
                 判据合法换过 `same_dir_seeds` 这类派生字段，拿老本当基准会把它报成"数字动了"）
  ⑥ 扩流不改老流 两本同配置账之间，**谁老是新由流集合/阶梯的严格包含决定，不看文件名**
                 （`_n12` 字典序在 `_n8` 前面，按文件名定向会结构性地漏掉"8 条流 ⊆ 12 条流"这一对）：
                 小账的每一格必须在大账里逐位重现。这一条是"3/8/12 条流的账能对着读"的**唯一许可证**——
                 本轮所有"换流数后判决变了/没变"的说法都默认了它，此前只是口头默认。
                 实测在册 7 对（本轮 +3：`_mid` 与 `_mid_n8_v3/_v4` 各对 `_n12` 一本）。
  ⑦ v4 的翻转集合 v4 相对 v3 只许翻掉预登记那一本（份额 118%、唯一出界的账）。只对**阶梯顶在账里**
                 的账参评：顶不在账时这条腿恒等于 v3，那种"通过"是没测到（本轮实测 14 本里参评 11 本）

自我拆红（每条各拆一道闸，跑了不红的模式=那条闸没测到，不许拿它当通过。下面全是**本轮实测**，
不是预期——规矩是实测再写回这张表，写死的计数只对"当前在册的那几本"成立：
本轮又入库一本（15→16 本）：`--decorative` 7→8、`--misfeed` 19→23、`--v4blind` 3→4，
因为**每一本新账一入库就成了复算闸① 与落地闸⑤ 的新被测对象**。
`--drift` 1→3 不是这个原因：那一档换的是**挑哪一对账去注入**的规则（见下面那行）。
加一本账就重跑九档再改这张表，沿用上轮的数字=这张表已经在说谎。）：
  --decorative  v3 瞎掉阶梯顶（s_top=None）→ 实测 8 条红：①×3 ② ③ ⑦ ⑤×2
                （谁都翻不动=装饰；⑦ 顺带红，因为瞎顶的 v4 连"倒挂"这一格原料都没有；
                 新落地的 `_v4`/`_n12` 本因此同时被 ① 与 ⑤ 点名）
  --destroy     把流级腿放松成"最多一条流反向"→ 实测 0 条红：它给的是"这批在册账的方向判定
                不靠最后一条流"这一条敏感性读数，**不算任何一道闸的验红通过**（承重由另八档拆）
  --collateral  第 2 支忘记要求反号、且挪到第 1 支前面 → 实测 5 条红：①×2 ② ④ ⑦
                （顺序与反号条件都承重）
  --misfeed     复算时把伤害方向 dirn 取反 → 实测 23 条红：①×20 + ④ + ⑤ + ⑦（读错数字当场被抓）
  --drift       把扩出来的那本某一格的 wrong_kd/gap 挪 +1 → 实测 3 条红：⑥×3
                （扩流本偷改老读数必须被抓。**注入点按包含关系挑、且优先挑"老那本文件名排在后面"
                 的那一对**（本轮是 `_n8_v3 ⊆ _n12_v4`）：老规则按文件名谁小谁老，
                 `_n12` 恰好排在 `_n8` 前面 ⇒ "8 条流 ⊆ 12 条流"这一对整个不在对账里，
                 本轮预登记的②就是这么被漏掉的。这一档现在钉的是**定向规则本身**，
                 三道同时红是因为 12 条流那本同时是三本老账的扩展）
  --sibdrift    给重跑本的一个训练路径读数（seed_floor_imgs）偷偷 +7 → 实测 3 条红：⑤ 逐本点名
                （钉"数字真动了的时候闸⑤ 还活着"，与下面那条一起构成 ⑤ 的两端）
  --oldsib      兄弟账退回 glob 顺序最老那本 → 实测 1 条红：⑤（跨了判据版本 ⇒ 把派生字段的正常
                演进报成漂移）。这一档钉的就是"基准挑哪一版"本身；它是**假红**，
                出现即说明选择规则被改坏了。
  --v4blind     v4 瞎掉倒挂腿（inverted_leg=False）→ 实测 4 条红：①×2 + ⑦ + ⑤
                （上一版只有 ⑦：那时 `_v4` 本还没落地，① 与 ⑤ 没有对象可查。
                 现在这条腿一旦被摘掉，档案里那句"倒挂"复算不出来、落地预言也落空）
  --v4loose     倒挂腿忘掉"同号且更大"，只要判到污染型就改口 → 实测 1 条红：只有 ⑦，
                红的内容是"参评三本污染型全翻成倒挂"（连坐）。**④ 抓不到它**——不是 ④ 松，
                是 ④ 定义在 v2/v3 复算上，结构性地看不见 v4 这条腿。
"""
import json
import glob
import sys

import step4f_alpha_curve as sf
from step4f_alpha_curve import shape_verdict as _orig_sv, shape_verdict_v1

MODE = next((a for a in sys.argv[1:] if a.startswith("--")), "")
_orig_leg = sf.flow_leg
shape_verdict = _orig_sv

if MODE == "--decorative":                       # v3 看不见阶梯顶
    def shape_verdict(s_small, s_probe, n, dirn, a_small, a_probe, anc_a,   # noqa: F811
                      s_top=None, pair_step=None):
        return _orig_sv(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, None, pair_step)
elif MODE == "--drift":                          # 扩流本"动了老读数"：注入在 docs 读完后做
    pass
elif MODE == "--destroy":                        # 流级腿放松成"最多一条流反向"
    def flow_leg(s, n):                           # noqa: D103
        out = _orig_leg(s, n)
        k, m = (int(x) for x in out["same_dir_flows"].split("/"))
        out["has_direction"] = bool(s["p_pooled"] < sf.P_SIG and k >= m - 1)
        return out
    sf.flow_leg = flow_leg                        # shape_verdict 走模块属性，这里换掉它才生效
elif MODE == "--collateral":                     # 反号条件忘掉的分支顺序 bug
    def shape_verdict(s_small, s_probe, n, dirn, a_small, a_probe, anc_a,   # noqa: F811
                      s_top=None, pair_step=None):
        lo, hi = sf.flow_leg(s_small, n), sf.flow_leg(s_probe, n)
        top = hi if s_top is None else sf.flow_leg(s_top, n)
        if top["has_direction"] and lo["p_pooled"] < sf.P_SIG:      # ← 少了 s_small*s_top<0
            return ("反号线（小剂量端与阶梯顶反向：两说都不合）", "拆红用：没要求反号", {})
        return _orig_sv(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, s_top, pair_step)

SHORT = lambda s: s.split("（")[0]      # noqa: E731


def v4_of(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, s_top, pair_step):
    """把 v4 那条腿装上（默认＝诚实实现：第 1 支成立 + 顶在账里 + 同号 + |第一格|>|顶|）。"""
    return _orig_sv(s_small, s_probe, n, dirn, a_small, a_probe, anc_a,
                    s_top, pair_step, inverted_leg=True)[0]


if MODE == "--v4blind":                # v4 瞎掉倒挂腿：它与 v3 逐字同判 → 该被闸⑦ 报成装饰
    def v4_of(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, s_top, pair_step):   # noqa: F811
        return _orig_sv(s_small, s_probe, n, dirn, a_small, a_probe, anc_a,
                        s_top, pair_step, inverted_leg=False)[0]
elif MODE == "--v4loose":              # 倒挂腿忘掉"同号且更大"：只要判到污染型就改口
    def v4_of(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, s_top, pair_step):   # noqa: F811
        sh, rd, lg = _orig_sv(s_small, s_probe, n, dirn, a_small, a_probe, anc_a,
                              s_top, pair_step, inverted_leg=True)
        if s_top is not None and SHORT(sh) == "污染型":
            return "污染型·倒挂（拆红用：没要求同号与大小）"
        return sh


def pruned(new, old):
    """把新账按旧账的字段形状裁剪：旧账没有的键＝后加的**报表字段**，不参与逐位对账；
    旧账有的键必须存在且值逐位相同（递归穿过 dict/list）。返回 (裁剪后的新账, 旧账缺的键, 多出的键)。"""
    lack, extra = [], []
    if isinstance(old, dict) and isinstance(new, dict):
        out = {}
        for k, v in old.items():
            if k not in new:
                lack.append(k)
                continue
            n_v, l2, e2 = pruned(new[k], v)
            out[k] = n_v
            lack += [f"{k}.{x}" for x in l2]
            extra += [f"{k}.{x}" for x in e2]
        extra += [k for k in new if k not in old]
        return out, lack, extra
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return None, ["长度"], []
        out, l3, e3 = [], [], []
        for i, (o_v, n_v) in enumerate(zip(old, new)):
            c, l2, e2 = pruned(n_v, o_v)
            out.append(c)
            lack_i = [f"[{i}]{x}" for x in l2]
            extra_i = [f"[{i}]{x}" for x in e2]
            l3 += lack_i
            e3 += extra_i
        return out, l3, e3
    return new, [], []

fails = []                                          # 一道闸红不提前退出：把五道都印出来才知道红了谁
rows = []
docs = {}


def setup_of(d):
    """一发实验的**判别配置**（不含流数、不含判据版本）：这两样换了是另一发，不是重跑。"""
    return (d.get("dataset", "mnist"), d["tier"], d["width"], str(d["probe_alpha"]),
            str(d.get("temps_fixed")), tuple(float(a) for a in d["ladder"] if float(a) != 0.0))


def same_cfg(a, b):
    """同一发配置的**重跑**：判据版本与流数可以不同，配置五项必须逐字相同。"""
    return setup_of(a)[:5] == setup_of(b)[:5]


def ladder_within(a, b):
    """a 的阶梯是 b 的阶梯的子集（b 是"扩了格"的那本）。"""
    return set(setup_of(a)[5]).issubset(set(setup_of(b)[5]))


for path in sorted(glob.glob("compress_report_v4f*.json")):
    d = json.loads(open(path, encoding="utf-8").read())
    docs[path] = d
    lad = {float(a): s for a, s in d["ladder_summary"].items()}
    small = min(lad)
    a_probe = float(d["probe_alpha"])
    anc = float(d.get("anchor_alpha") or 0.7)
    n = len(d["seeds"])
    if "anchor_mean_gap_imgs" in d:
        anchor_src, anchor_mean = "档案 anchor_mean_gap_imgs", float(d["anchor_mean_gap_imgs"])
    else:                                        # v1-era 非薄阶梯那三本没记这个字段
        anchor_src, anchor_mean = "账内最大剂量格（非薄阶梯没训到锚点）", lad[max(lad)]["mean_gap_imgs"]
    dirn = (1.0 if anchor_mean > 0 else -1.0) * (-1 if MODE == "--misfeed" else 1)
    s_small, s_probe = lad[small], lad[a_probe]
    s_top = lad.get(anc)
    rec = d["verdict"]["shape"]
    rule = d["verdict"].get("shape_rule") or "v1"
    v1 = shape_verdict_v1(s_small, s_probe, dirn)
    v2 = _orig_sv(s_small, s_probe, n, dirn, small, a_probe, anc)     # v2 = v3  machinery 不吃顶
    v3 = shape_verdict(s_small, s_probe, n, dirn, small, a_probe, anc,
                       None if MODE == "--decorative" else s_top,
                       d.get("paired_step_small_to_anchor"))[0]
    v4 = v4_of(s_small, s_probe, n, dirn, small, a_probe, anc,
               None if MODE == "--decorative" else s_top,
               d.get("paired_step_small_to_anchor"))
    v2 = v2[0]
    rows.append((path, rule, rec, v1, v2, v3, s_top is not None, anchor_src, n, small, a_probe, anc, v4))

    same = SHORT(rec) == SHORT({"v1": v1, "v2": v2, "v3": v3, "v4": v4}[rule])
    print(f"{path}  [{n} 条流 | 阶梯顶 {'在账里' if s_top else '不在账里'} | dirn 来源：{anchor_src} "
          f"| α {small:g}/{a_probe:g}/{anc:g}]")
    print(f"   复算  v1={SHORT(v1):<4} v2={SHORT(v2):<4} v3={SHORT(v3):<4} v4={SHORT(v4):<6}   "
          f"档案记（{rule}）={SHORT(rec)}  {'== 复算' if same else '**不等：喂错数字了**'}")
    if d["verdict"].get("verdict_under_v1_rule") and \
            SHORT(d["verdict"]["verdict_under_v1_rule"]) != SHORT(v1):
        fails.append(f"闸① {path} 档案里的 v1 并排判词与复算不符（记 "
                     f"{SHORT(d['verdict']['verdict_under_v1_rule'])}，复算 {SHORT(v1)}）")
    if d["verdict"].get("verdict_under_v2_rule") and \
            SHORT(d["verdict"]["verdict_under_v2_rule"]) != SHORT(v2):
        fails.append(f"闸① {path} 档案里的 v2 并排判词与复算不符（记 "
                     f"{SHORT(d['verdict']['verdict_under_v2_rule'])}，复算 {SHORT(v2)}）")
    if d["verdict"].get("verdict_under_v3_rule") and \
            SHORT(d["verdict"]["verdict_under_v3_rule"]) != SHORT(v3):
        fails.append(f"闸① {path} 档案里的 v3 并排判词与复算不符（记 "
                     f"{SHORT(d['verdict']['verdict_under_v3_rule'])}，复算 {SHORT(v3)}）")
    if not same:
        fails.append(f"闸① {path} 的档案判词复算不出来（档案记 {SHORT(rec)}，{rule} 复算 "
                     f"{SHORT({'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4}[rule])}）——"
                     f"先查是不是读了同一批数字，别谈 {rule}")

# 翻转集合只在**旧账**（记录在 v1/v2 名下那几本）上算：同一发实验的 _v3 重跑本不是"新的一翻"，
# 它是闸⑤要验的"预言落地"，算进翻转集合就是把一件事数两次。
flips_v2_v3 = [p for p, rule, rec, v1, v2, v3, *_ in rows
               if rule not in ("v3", "v4") and SHORT(v2) != SHORT(v3)]
expected = ["compress_report_v4f_low_n8_thin_v2.json"]
print(f"\n[闸① 复算自证] {len(rows)} 本档案："
      f"{'判词全部由产出它的那一版复算出来 → 通过' if not any(f.startswith('闸①') for f in fails) else '见上：有复算不出的'}")

print(f"[闸② v3 相对 v2 的翻转集合（只在旧账上算，"
      f"{sum(1 for r in rows if r[1] not in ('v3', 'v4'))} 本）] {flips_v2_v3 or '（空）'}")
if sorted(flips_v2_v3) != expected:
    fails.append(f"闸② 翻转集合应为 {expected}，实为 {flips_v2_v3}——谁都翻不动是装饰，连坐翻掉板块最硬结论是把债转成损失")

flip_row = [r for r in rows if r[0] == expected[0]]
if not flip_row:
    fails.append(f"闸③ 预期该翻的 {expected[0]} 不在表里")
else:
    got = SHORT(flip_row[0][4]) + " → " + SHORT(flip_row[0][5])
    print(f"[闸③ 只翻该翻的那一本] {expected[0]}：{got}")
    if got != "测不动 → 反号线":
        fails.append(f"闸③ 那一本应「测不动 → 反号线」，实为 {got}——翻错方向等于换了个说法而不是换了视野")

poll = [r for r in rows if r[2].startswith("污染型")]
hard = [r[0] for r in poll if SHORT(r[4]) != "污染型" or SHORT(r[5]) != "污染型"]
print(f"[闸④ 板块 {len(poll)} 本在册『污染型』有没有被连坐] {hard or '没有：逐字保持污染型'}")
if hard:
    fails.append(f"闸④ 被连坐：{hard}")

# 闸⑦：v4 相对 v3 只许翻掉预登记的那一本（份额>100%、唯一出界的那发）。
# 与闸② 同构：全翻是这条腿连坐（"只要判到污染型就说倒挂"），一本不翻是这条腿装饰。
expected_v4 = ["compress_report_v4f_mid_n8_v3.json"]
flips_v3_v4 = [r[0] for r in rows if r[1] != "v4" and SHORT(r[5]) != SHORT(r[12])]
can_flip = [r for r in rows if r[1] != "v4" and r[6]]       # 阶梯顶不在账里的本**参评不了**这条腿
print(f"[闸⑦ v4 相对 v3 的翻转集合（v3 及更早的 {sum(1 for r in rows if r[1] != 'v4')} 本里，"
      f"阶梯顶在账因而**参评**的 {len(can_flip)} 本；三本 v1-era 非薄账没训到锚点，"
      f"这条腿对它们恒等于 v3，不算通过也不算红）] {flips_v3_v4 or '（空）'}")
if sorted(flips_v3_v4) != expected_v4:
    fails.append(f"闸⑦ v4 的翻转集合应为 {expected_v4}，实为 {flips_v3_v4}——"
                 f"谁都翻不动是装饰，连坐翻掉别本是破坏")
if not can_flip:
    fails.append("闸⑦ 一本参评的账都没有——不是通过，是没测到")
else:
    part_poll = [r for r in can_flip if SHORT(r[2]).startswith("污染型")]
    stayed = [r for r in part_poll if SHORT(r[12]) == "污染型"]
    by_share = {r[0]: docs[r[0]]["verdict"]["first_cell_share_of_anchor_damage_pct"] for r in stayed}
    near = max(by_share.items(), key=lambda kv: kv[1]) if by_share else (None, None)
    print(f"  参评 {len(can_flip)} 本里在册『污染型』的 {len(part_poll)} 本："
          f"改口 {len(part_poll) - len(stayed)} 本、保持 {len(stayed)} 本"
          + (f"（保持的那些里离界最近的是 {near[1]}%，即 `{near[0]}`）" if by_share else ""))
    print("  近邻对照要分清两类：`compress_report_v4f_mid.json`(97%) 份额更贴界，但它是 3 条流的账、"
          "阶梯顶不在账里 ⇒ 这条腿对它恒等于 v3，**不能**当这条判据的近邻")

# 闸⑤：真实重跑必须落在预登记预言的那一格上，而且**只有判词能变**。
# 为什么这条比闸②硬：闸② 是拿旧账的数字离线复算，重跑可能因为训练路径变了而给出别的判词；
# 这一条同时钉住"预言=实测"和"数字没动"，动了数字的翻案就不是判据换到的东西，是重训换来的。
TRAIN_PATH = ("dataset", "tier", "width", "seeds", "ladder", "ladder_summary", "arms",
              "identity_gate_seeds", "probe_alpha", "anchor_alpha", "temps_fixed",
              "paired_step_small_to_anchor", "seed_floor_imgs", "predictions_ex_ante")
by_path = {r[0]: r for r in rows}


def ver_rank(q):                     # 文件名尾的 _v<N>；无后缀 = v1 时代，记 0
    _head, sep, tail = q.rsplit(".json", 1)[0].rpartition("_v")
    return int(tail) if sep and tail.isdigit() else 0


fresh = [p for p, rule, *_ in rows if rule in ("v3", "v4")]
if not fresh:
    fails.append("闸⑤ 一本 _v3/_v4 重跑账都没有——预言没人验过就等于没落地，别宣布新判据换到了东西")
else:
    print(f"[闸⑤ 落地自证] 已重跑 {len(fresh)} 本：{[p for p in fresh]}")
for p, rule, rec, v1, v2, v3, *_ in rows:
    if rule not in ("v3", "v4"):
        continue
    d_new = docs[p]
    if MODE == "--sibdrift" and isinstance(d_new.get("seed_floor_imgs"), (int, float)):
        # 突变：重跑本偷偷动了训练路径的一个数（不改判词，只改读数）→ 闸⑤ 必须逐本点名。
        d_new["seed_floor_imgs"] = d_new["seed_floor_imgs"] + 7
    # 同配置同流的那本才是"重跑本"：名字推出来的兄弟账可能是**另一发实验**（换了流数=换了实验），
    # 拿它当基准逐位对账，红了也不是判据的错。所以按**流集合**认，不按文件名认。
    sibs = [q for q in docs if q != p and same_cfg(d_new, docs[q])]
    same_flow = [q for q in sibs if docs[q]["seeds"] == d_new["seeds"]]
    if not same_flow:
        print(f"  {p}：{len(sibs)} 本同配置账、但没有一本流集合相同 → 这是该配置的**首跑**，"
              f"不是重跑本，闸⑤ 不管（它的对账线是恒等闸＋跨脚本闸）")
        continue
    pick_oldest = MODE == "--oldsib"
    earlier = [q for q in same_flow if ver_rank(q) < ver_rank(p)]
    if not earlier and not pick_oldest:
        print(f"  {p}：同流同配置 {len(same_flow)} 本，但没有一本判据版本**低于**本作 → "
              f"它是这一发在新判据下的首跑，前一版账要么不存在要么是**后继**（后继不能当基准）")
        continue
    sib = (same_flow[0] if pick_oldest
           else max(earlier, key=lambda q: (ver_rank(q), q)))
    print(f"  基准兄弟账 = {sib}"
          f"（同流同配置 {len(same_flow)} 本、版本低于本作的 {len(earlier)} 本，v 序 "
          f"{[ver_rank(q) for q in sorted(same_flow)]}，"
          f"按{'**最老**一本（--oldsib 突变：跨了判据版本）' if pick_oldest else '版本号最高的前一版'}挑）")
    srow = by_path[sib]
    d_old = docs[sib]
    pred_col = 12 if rule == "v4" else 5        # 落地要落的是"对兄弟账施**本作**判据"的那个预言
    on_pred = SHORT(rec) == SHORT(srow[pred_col])
    moved, shed, added = [], [], []
    for k in TRAIN_PATH:
        if k not in d_old:
            continue
        cut, lack, ext = pruned(d_new.get(k, None), d_old[k])
        if lack:
            shed += [f"{k}.{x}" for x in lack]
        if cut != d_old[k]:
            moved.append(k)
        added += [f"{k}.{x}" for x in ext]
    print(f"  {p}\n    重跑判词 = {SHORT(rec)}；离线预登记（对 {sib} 施 {rule}）= {SHORT(srow[pred_col])} "
          f"{'→ 落在预言上' if on_pred else '**偏离预言**'}")
    print(f"    训练路径逐位：{len([k for k in TRAIN_PATH if k in d_old])} 个字段"
          f"{'全同（只有判词与判据说明变了）' if not moved else f'**动了：{moved}**'}"
          + (f"｜旧账独有的键被删：{shed}" if shed else "")
          + (f"｜新账多出的报表字段 {len(added)} 个：{sorted(set(added))[:6]}" if added else ""))
    if not on_pred:
        fails.append(f"闸⑤ {p} 没落在预登记的预言上")
    if moved:
        fails.append(f"闸⑤ {p} 连训练路径一起换了（{moved}）——换数字换来的翻案不能记在判据头上")
    if shed:
        fails.append(f"闸⑤ {p} 旧账的字段在新账里找不到（{shed}）——报表缩水也算动了仪器")

# 闸⑥：扩流/扩格**不许改动老读数**。这一条是"3 条流的账与 8 条流的账能对着读"的唯一许可证——
# 本轮所有"换流数后判决变了/没变"的说法，都默认了同一批老流在两本账里是同一批人。
def arm_index(d):
    out = {}
    for rec in (d["arms"].values() if isinstance(d["arms"], dict) else d["arms"]):
        out[(round(float(rec["alpha"]), 4), int(rec["seed"]))] = rec
    return out


def pair_dirs(da, db):
    """两本同配置账之间"只扩不缩"的方向 → (老那本, 新那本, 各自的流数/格数)；不可比则 None。

    定向只看流集合与阶梯的**严格包含**，不看文件名：`_n12` 字典序排在 `_n8` 前面，
    按文件名谁小谁老会让"8 条流 ⊆ 12 条流"这一对结构上不存在——本轮预登记的②就这样被漏掉过。"""
    for px, py in ((da, db), (db, da)):
        sa, sb = set(px["seeds"]), set(py["seeds"])
        la = {round(float(x), 4) for x in px["ladder"] if float(x) != 0.0}
        lb = {round(float(x), 4) for x in py["ladder"] if float(x) != 0.0}
        if sa <= sb and la <= lb and (len(sa) < len(sb) or len(la) < len(lb)):
            return px, py, sa, sb, la, lb
    return None


print("[闸⑥ 扩流/扩格不改老读数（同配置账两两对账）]")
if MODE == "--drift":
    cand = []
    for pa in sorted(docs):
        for pb in sorted(docs):
            if pa >= pb or not same_cfg(docs[pa], docs[pb]):
                continue
            got = pair_dirs(docs[pa], docs[pb])
            if got:
                cand.append((got[0] is docs[pb], pa, pb))   # True = 老那本名字排在后面
    # 优先拆"文件名顺序与包含顺序相反"的那一对：按文件名挑对的老规则对它无感，
    # 退回任意一对的话，这一档就只是在测"闸⑥ 还活着"，测不到这次修的东西。
    hit = next(((pa, pb) for rev, pa, pb in cand if rev), None) or (cand[0][1:] if cand else None)
    if hit is None:
        fails.append("闸⑥（拆红模式 --drift）找不到可比的账对，注入没处放——这条红不算通过")
    else:
        pa, pb = hit                                        # 字典序 pa < pb，谁是老那本由包含关系定
        old, new = (pb, pa) if pair_dirs(docs[pa], docs[pb])[0] is docs[pb] else (pa, pb)
        kk = next(iter(arm_index(docs[old])))
        r = arm_index(docs[new])[kk]
        r["wrong_kd"] += 1
        r["gap_imgs"] += 1
        print(f"  （拆红注入）把扩出来的那本 {new} 的 α={kk[0]:g}/种子{kk[1]} 那格 wrong_kd 与 gap 各挪 +1"
              f"（基准 {old} 同格不动）——只有这一道闸会看见它"
              + (f"；这一对的**老那本文件名排在后面**（{old}），按文件名定向的旧规则看不见它"
                 if old > new else ""))
pairs = 0
for pa, da in sorted(docs.items()):
    for pb, db in sorted(docs.items()):
        if pa >= pb or not same_cfg(da, db):
            continue
        # 定向交给 pair_dirs()：与上面 --drift 的挑对规则**同一份代码**——两处各写一遍，
        # 就是本轮那个漏网的对（`_n8 ⊆ _n12`）的来历。
        got = pair_dirs(da, db)
        if got is None:
            continue                                    # 同流同格＝重跑，那是闸⑤ 的对象不是这条
        dx, dy, sa, sb, la, lb = got
        px, py = (pa, pb) if dx is da else (pb, pa)     # 老那本 / 新那本
        ia, ib = arm_index(dx), arm_index(dy)
        miss, bad = [], []
        for kk, ra in ia.items():
            rb = ib.get(kk)
            if rb is None:
                miss.append(kk)
                continue
            cut, lack, _ = pruned(rb, ra)
            if lack or cut != ra:
                bad.append(f"{kk}(缺{lack or ''}{'' if cut == ra else '值变'})")
        pairs += 1
        tag = f"  {px}（{len(sa)} 流×{len(la)} 格） ⊆ {py}（{len(sb)} 流×{len(lb)} 格）"
        if miss or bad:
            fails.append(f"闸⑥ {px} → {py}：缺格 {miss}、值变 {bad}——扩流那本动了老流，跨流数比较作废")
            print(f"{tag} → **红**：缺 {len(miss)} 格、变 {len(bad)} 格")
        else:
            print(f"{tag} → {len(ia)} 格逐位相同")
if pairs == 0:
    fails.append("闸⑥ 一对可比的账都没有——不是通过，是没测到")
print(f"  共对 {pairs} 对")

print(f"\n模式：{MODE or '（正常验红）'}")
if fails:
    print(f"\n>>> 验红 {len(fails)} 条红：")
    for f in fails:
        print("   ", f)
    raise SystemExit(1)
print("\n结论：v3 只买『判据看得见阶梯顶』这一件事，v4 只买『出界的份额话改口成倒挂』这一件事"
      "——其余各本判词逐字不动。")
