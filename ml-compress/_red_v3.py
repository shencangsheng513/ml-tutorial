# -*- coding: utf-8 -*-
"""v3 形状判据的离线验红（零训练）：把已入库的 v4f 档案按 v1/v2/v3 各算一遍，看翻转集合落在哪。

六道闸（沿用第四发的规矩：一道闸只钉一件修复，红要知道是哪件修复没装上）：
  ① 复算自证     每本档案**记录在案**的那句判词 = 产出它那一版的复算值（钉"喂的是同一批数字"）
  ② 翻转集合     v3 相对 v2 只许翻掉预登记的那一本（钉"v3 不是装饰，也不是连坐"）
  ③ 只翻该翻的   那一本必须从 测不动 翻到 反号线（钉翻的是对的方向）
  ④ 不连坐       板块全部在册『污染型』判词逐字不变（条数现算不写死；钉分支顺序与反号条件）
  ⑤ 落地自证     真的 _v3 **同流重跑**本：判词落在预登记预言上，且训练路径字段逐位不动
                 （钉"翻案是判据换来的，不是重训换来的"；换了流数=另一发实验，不在这一条里）
                 基准账 = 同流同配置里**版本号最高的前一版**（不是 glob 顺序最老那本：v1→v2 时
                 判据合法换过 `same_dir_seeds` 这类派生字段，拿老本当基准会把它报成"数字动了"）
  ⑥ 扩流不改老流 同配置的两本账（老 3 条流 ⊆ 新 8 条流、老阶梯 ⊆ 新阶梯）：小账的每一格必须在
                 大账里逐位重现。这一条是"3 条流的账与 8 条流的账能对着读"的**唯一许可证**——
                 本轮所有"换流数后判决变了/没变"的说法都默认了它，此前只是口头默认。

自我拆红（每条各拆一道闸，跑了不红的模式=那条闸没测到，不许拿它当通过）：
  --decorative  v3 瞎掉阶梯顶（s_top=None）→ 预期 ①②③⑤ 红（谁都翻不动=装饰）
  --destroy     把流级腿放松成"最多一条流反向"→ **本轮实测不红**：它给的是"这批在册账的方向判定
                不靠最后一条流"这一条敏感性读数，**不算任何一道闸的验红通过**（承重由另五档拆）
  --collateral  第 2 支忘记要求反号、且挪到第 1 支前面 → 预期 ①②④ 红（顺序与反号条件都承重）
  --misfeed     复算时把伤害方向 dirn 取反 → 预期 ① 红（读错数字当场被抓；实测 7 条含 ④）
  --drift       把大账某一格的 wrong_kd/gap 挪 +1 → 预期 ⑥ 红（扩流本偷改老读数必须被抓）
  --sibdrift    给重跑本的一个训练路径读数（seed_floor_imgs）偷偷 +7 → 预期 ⑤ 逐本红
                （钉"数字真动了的时候闸⑤ 还活着"，与下面那条一起构成 ⑤ 的两端）
  --oldsib      兄弟账退回 glob 顺序最老那本 → 预期 ⑤ 红（跨了判据版本 ⇒ 把派生字段的正常
                演进报成漂移）。这一档钉的就是"基准挑哪一版"本身；它是**假红**，
                出现即说明选择规则被改坏了。
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
    v2 = v2[0]
    rows.append((path, rule, rec, v1, v2, v3, s_top is not None, anchor_src, n, small, a_probe, anc))

    same = SHORT(rec) == SHORT({"v1": v1, "v2": v2, "v3": v3}[rule])
    print(f"{path}  [{n} 条流 | 阶梯顶 {'在账里' if s_top else '不在账里'} | dirn 来源：{anchor_src} "
          f"| α {small:g}/{a_probe:g}/{anc:g}]")
    print(f"   复算  v1={SHORT(v1):<4} v2={SHORT(v2):<4} v3={SHORT(v3)}   "
          f"档案记（{rule}）={SHORT(rec)}  {'== 复算' if same else '**不等：喂错数字了**'}")
    if d["verdict"].get("verdict_under_v1_rule") and \
            SHORT(d["verdict"]["verdict_under_v1_rule"]) != SHORT(v1):
        fails.append(f"闸① {path} 档案里的 v1 并排判词与复算不符（记 "
                     f"{SHORT(d['verdict']['verdict_under_v1_rule'])}，复算 {SHORT(v1)}）")
    if d["verdict"].get("verdict_under_v2_rule") and \
            SHORT(d["verdict"]["verdict_under_v2_rule"]) != SHORT(v2):
        fails.append(f"闸① {path} 档案里的 v2 并排判词与复算不符（记 "
                     f"{SHORT(d['verdict']['verdict_under_v2_rule'])}，复算 {SHORT(v2)}）")
    if not same:
        fails.append(f"闸① {path} 的档案判词复算不出来（档案记 {SHORT(rec)}，{rule} 复算 "
                     f"{SHORT({'v1': v1, 'v2': v2, 'v3': v3}[rule])}）——先查是不是读了同一批数字，别谈 v3")

# 翻转集合只在**旧账**（记录在 v1/v2 名下那几本）上算：同一发实验的 _v3 重跑本不是"新的一翻"，
# 它是闸⑤要验的"预言落地"，算进翻转集合就是把一件事数两次。
flips_v2_v3 = [p for p, rule, rec, v1, v2, v3, *_ in rows if rule != "v3" and SHORT(v2) != SHORT(v3)]
expected = ["compress_report_v4f_low_n8_thin_v2.json"]
print(f"\n[闸① 复算自证] {len(rows)} 本档案："
      f"{'判词全部由产出它的那一版复算出来 → 通过' if not any(f.startswith('闸①') for f in fails) else '见上：有复算不出的'}")

print(f"[闸② v3 相对 v2 的翻转集合（只在旧账上算，{sum(1 for r in rows if r[1] != 'v3')} 本）] {flips_v2_v3 or '（空）'}")
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


fresh = [p for p, rule, *_ in rows if rule == "v3"]
if not fresh:
    fails.append("闸⑤ 一本 _v3 重跑账都没有——预言没人验过就等于没落地，别宣布 v3 换到了东西")
else:
    print(f"[闸⑤ 落地自证] 已重跑 {len(fresh)} 本：{[p for p in fresh]}")
for p, rule, rec, v1, v2, v3, *_ in rows:
    if rule != "v3":
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
    sib = (same_flow[0] if pick_oldest
           else max(same_flow, key=lambda q: (ver_rank(q), q)))
    print(f"  基准兄弟账 = {sib}"
          f"（同流同配置 {len(same_flow)} 本，v 序 {[ver_rank(q) for q in sorted(same_flow)]}，"
          f"按{'**最老**一本（--oldsib 突变：跨了判据版本）' if pick_oldest else '版本号最高的前一版'}挑）")
    srow = by_path[sib]
    d_old = docs[sib]
    on_pred = SHORT(rec) == SHORT(srow[5])
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
    print(f"  {p}\n    重跑判词 = {SHORT(rec)}；离线预登记（对 {sib} 施 v3）= {SHORT(srow[5])} "
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


print(f"[闸⑥ 扩流/扩格不改老读数（同配置账两两对账）]")
if MODE == "--drift":
    hit = next(((pa, pb) for pa in sorted(docs) for pb in sorted(docs)
                if pa < pb and same_cfg(docs[pa], docs[pb])
                and set(docs[pa]["seeds"]) <= set(docs[pb]["seeds"])
                and {round(float(x), 4) for x in docs[pa]["ladder"] if float(x)}
                <= {round(float(x), 4) for x in docs[pb]["ladder"] if float(x)}), None)
    if hit is None:
        fails.append("闸⑥（拆红模式 --drift）找不到可比的账对，注入没处放——这条红不算通过")
    else:
        pa, pb = hit
        kk = next(iter(arm_index(docs[pa])))
        r = arm_index(docs[pb])[kk]
        r["wrong_kd"] += 1
        r["gap_imgs"] += 1
        print(f"  （拆红注入）把 {pb} 的 α={kk[0]:g}/种子{kk[1]} 那格 wrong_kd 与 gap 各挪 +1"
              f"（基准 {pa} 同格不动）——只有这一道闸会看见它")
pairs = 0
for pa, da in sorted(docs.items()):
    for pb, db in sorted(docs.items()):
        if pa >= pb or not same_cfg(da, db):
            continue
        sa, sb = set(da["seeds"]), set(db["seeds"])
        la = {round(float(x), 4) for x in da["ladder"] if float(x) != 0.0}
        lb = {round(float(x), 4) for x in db["ladder"] if float(x) != 0.0}
        if not (sa <= sb and la <= lb and (len(sa) < len(sb) or la < lb)):
            continue                        # 只扩不缩才是"同一批老流加了几条"
        ia, ib = arm_index(da), arm_index(db)
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
        tag = f"  {pa}（{len(sa)} 流×{len(la)} 格） ⊆ {pb}（{len(sb)} 流×{len(lb)} 格）"
        if miss or bad:
            fails.append(f"闸⑥ {pa} → {pb}：缺格 {miss}、值变 {bad}——扩流那本动了老流，跨流数比较作废")
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
print("\n结论：v3 装对了——它只买『判据看得见阶梯顶』这一件事。")
