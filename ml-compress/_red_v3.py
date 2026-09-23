# -*- coding: utf-8 -*-
"""v3/v4/v5 形状判据的离线验红（零训练）：把已入库的 v4f 档案按 v1/v2/v3/v4/v5 各算一遍，看翻转集合落在哪。

八道闸（沿用第四发的规矩：一道闸只钉一件修复，红要知道是哪件修复没装上）：
  ① 复算自证     每本档案**记录在案**的那句判词 = 产出它那一版的复算值（钉"喂的是同一批数字"）
  ② 翻转集合     v3 相对 v2 只许翻掉预登记的那一本（钉"v3 不是装饰，也不是连坐"）
  ③ 只翻该翻的   那一本必须从 测不动 翻到 反号线（钉翻的是对的方向）
  ④ 不连坐       板块全部在册『污染型』判词逐字不变（条数现算不写死；钉分支顺序与反号条件）
                 注意它**只按 v2/v3 复算**，所以它结构性地看不见 v4 的连坐——v4 的连坐由 ⑦ 独扛
  ⑤ 落地自证     真的 _v3 / _v4 / _v5 **同流重跑**本：判词落在预登记预言上，且训练路径字段逐位不动
                 （钉"翻案是判据换来的，不是重训换来的"；换了流数=另一发实验，不在这一条里）
                 基准账 = 同流同配置里**版本号最高的前一版**（不是 glob 顺序最老那本：v1→v2 时
                 判据合法换过 `same_dir_seeds` 这类派生字段，拿老本当基准会把它报成"数字动了"）
                 **重跑本还要求阶梯相同**：阶梯不同的那本是"往上扩了格"的另一发（`--ladder-high`），
                 让 ⑤ 去比它，红了的是 `ladder` 的长度、那是设计使然不是漂移；老格子的逐位对账归 ⑥。
                 `--laddersib` 把这条排除关掉，专门钉"扩格本冒充重跑本"这一端。
  ⑥ 扩流/扩格不改老读数 两本同配置账之间，**谁老是新由流集合/阶梯的严格包含决定，不看文件名**
                 （`_n12` 字典序在 `_n8` 前面，按文件名定向会结构性地漏掉"8 条流 ⊆ 12 条流"这一对）：
                 小账的每一格必须在大账里逐位重现。这一条是"3/8/12/16 条流的账能对着读"的**唯一许可证**——
                 本轮所有"换流数后判决变了/没变"的说法都默认了它，此前只是口头默认。
                 实测在册 **21 对**（第十四发两本 16 条流的账入库后 +7：`_mid_n16_v5` 一本就同时是
                 `_mid`/`_mid_n8_v3`/`_mid_n8_v4`/`_mid_n8_v5`/`_mid_n12_v4` **五本**老账的扩展，
                 `_mid_n16_t6_v5` 又接上 `_mid_n8_t6_v4` 与 `_mid_n12_t6_v5` 两本——60 格逐位相同是
                 "12 条流与 16 条流的账能对着读"这句话唯一有盘上凭据的形式）。
                 上一轮（`_mid_n12_v4` +3、`_hi` +3）与本轮（+7）的差要说清：**对数由这本账在包含关系里
                 站的位置决定，不由"这一发买了哪一维"决定**——第十一轮三本 `_t6` 首跑账一对都没多，
                 而第十四发"什么维都没换、只补分辨率"反倒一次多买 7 对。
                 但**反过来也成立**：v5 那本同流重跑账一入库就多买 3 对（`_mid.json` ⊆ 它、它 ⊆ `_n12_v4`、
                 它 ⊆ `_hi_v4`）——一本"什么维都没换"的账恰恰是**三对**"同一批流对着读"的中间人。
                 我预登记时按 +1 猜的，实测 +3。**许可证对数与"买了什么维"没有单调关系，只能现读。**）
  ⑦ v4 的翻转集合 v4 相对 v3 只许翻掉预登记那一本（份额 118%，**预登记当时**唯一出界的账）。只对**阶梯顶在账里**
                 的账参评：顶不在账时这条腿恒等于 v3，那种"通过"是没测到（本轮实测 14 本里参评 11 本）。
                 本轮新入库的两本 v4-era 出界账（`_mid_n8_t6_v4` 168%、`_n8_t6_v4` 101%）**不进这条集合**：
                 它们从落地那天就是 v4 判词、没有"v3 时代的旧判词"可翻 ⇒ 闸⑦ 对它们既不算通过也不算红，
                 它们的"倒挂"承重靠的是自己那本里的配对步长 CI，不是这道闸。
  ⑧ v5 的翻转集合 v5 相对 v4 只许翻掉预登记那**四本**（出界幅度 < 那条配对步长的 CI 半宽 ⇒ 从『倒挂』退回
                 『污染型』），且 `_mid_n8_t6_v4`（出界 29.8 > 半宽 22.4）**必须不翻**——它是在"记录在 v4
                 名下"这个参评口径里唯一还站着的倒挂。**注意"唯一"的口径**：第十四发之后，记录在 v5 名下、
                 同样判倒挂的账已有两本（`_mid_n12_t6_v5`、`_mid_n16_t6_v5`，同一坐标的 12 条流与 16 条流读数）
                 ⇒ "唯一一本账"早已不真，"唯一一个坐标"仍然真。脚本把这件事**现读打印**出来
                 （v5 复算判倒挂的账共几本、按配置归并成几个坐标、每个坐标几本），不靠注释里的形容词。
                 参评口径与 ⑦ 同族：只看**判词记录在 v4 名下、阶梯顶在账**的账（实测 24 本里参评 6 本——
                 第十四发那两本记在 v5 名下 ⇒ 结构上一本都不进参评集合，所以 6 本这个数没随册子涨）；
                 v5 落地那一本（`_mid_n8_v5`）是"预言兑现"，它的对账线是 ⑤，不进这里的参评表。
                 参评为空 = **没测到**而不是通过。附带一条不连坐检查：v4 复算不在第 1 支里的参评账，
                 v5 判词必须逐字相同（这条腿嵌在第 1 支内部，碰到别支就是实现走样）。
                 旧闸接新版本这件事本轮做齐了：① 加 v5 复算列、②⑤⑦ 共用的"旧账/后继账"名单 `RERUN_RULES`
                 含 v5、⑤ 的预言列按 `pred_col` 指向**本作**判据那一列。

自我拆红（每条各拆一道闸，跑了不红的模式=那条闸没测到，不许拿它当通过。下面全是**本轮实测**，
不是预期——规矩是实测再写回这张表，写死的计数只对"当前在册的那几本"成立：
本轮又入库**两本**（22→24 本，`compress_report_v4f_mid_n16_v5.json` 与 `_mid_n16_t6_v5.json`：
中数据 × T=3 / T=6 各补到 16 条流的**首跑**账，什么判据都没换、七维一维都没换，买的是分辨率），
拆红仍**十二档**、但每一档都要重测。计数变了六条：
**每一本新账一入库就成了复算闸① 与落地闸⑤ 的新被测对象**（一本 v5-era 账在 ① 的并排表里最多被点五次名：
v1/v2/v3/v4 并排列各一次＋本发判词一次），所以 `--misfeed` 48→58（①×52，两本新账各被点满）、
`--decorative` 18→21、`--collateral` 8→9、`--v4blind` 10→12、`--v5blind` 3→4
（多出来的那条 ① 是 `_mid_n16_v5`：它按 v4 复算判『倒挂』（份额 105% 出界），而档案记的是 v5 那句
『污染型』⇒ "把 CI 腿摘掉"这件事现在有一本**当场复算不出来**的新账，上一轮还没有这种对象）。
**第六条不是被测对象变多，是这一档的强度被修回来了**：`--drift` 4→5。上一轮把注入点改成"挑扩展出度
最大的 hub"，但排序键第一位用的是**反向对条数**而不是**注入实际红几条**（=这本扩展的老账总数）；
第十四发入库后 `_mid_n16_v5`（扩展 5 本）与 `_mid_n8_hi_v4`（扩展 4 本）反向对同为 3 条 ⇒ 平票被
文件名倒序判给小的那本，红数停在 4 而这一档本来该红 5。**同一类漂移第二次出现**（第一次是账册一涨
注入点漂到"只扩展一本"的新账），所以键改成 (len(hub[nw]), len(rev), 名字)：强度由结构定，不由平票裁决。
**没动的六档**：`--sibdrift` 3、`--oldsib` 1、`--laddersib` 2、`--v4loose` 1、`--v5strict` 3、
`--destroy` 仍 0 红（⑥ 的对账 14 对 → **21 对**，见上面那一条）。
加一本账就重跑十二档再改这张表，沿用上轮的数字=这张表已经在说谎。）：
  --decorative  v3 瞎掉阶梯顶（s_top=None）→ 实测 21 条红：①×14 ② ③ ⑤×2 ⑦ ⑧×2
                （谁都翻不动=装饰；⑦ 顺带红，因为瞎顶的 v4 连"倒挂"这一格原料都没有，⑧ 跟着红两次——
                 翻转集合变空 ＋ 唯一该站着的倒挂也站不住。① 从 11 涨到 14（实测点名）：多出的三条全在
                 两本新账上——`_mid_n16_t6_v5` 两次（v4 并排列 + 本发判词都写着『倒挂』），
                 `_mid_n16_v5` 一次（它档案的 v4 并排列也写着『倒挂』：份额 105% 在 v4 下越界 ⇒
                 瞎掉顶之后连那一列都复算不出来））
  --destroy     把流级腿放松成"最多一条流反向"→ 实测 0 条红：它给的是"这批在册账的方向判定
                不靠最后一条流"这一条敏感性读数，**不算任何一道闸的验红通过**（承重由另十一档拆）
  --collateral  第 2 支忘记要求反号、且挪到第 1 支前面 → 实测 9 条红：①×6 ② ④ ⑦
                （顺序与反号条件都承重；① 多出来的那一条是本轮新账 `_mid_n16_t6_v5`——它的 v3
                 并排列写着『污染型·倒挂』，分支顺序一挪就被读成反号线，所以"第 2 支必须在前"
                 这条约束现在又多一个被测对象）
  --misfeed     复算时把伤害方向 dirn 取反 → 实测 58 条红：①×52 + ④ + ⑤×2 + ⑦ + ⑧×2（读错数字当场被抓）
  --drift       把扩出来的那本某一格的 wrong_kd/gap 挪 +1 → 实测 5 条红：⑥×5
                （扩流/扩格本偷改老读数必须被抓。**注入点按包含关系挑、且挑"扩展出的老账最多"的那一本**
                 （本轮是 `_mid_n16_v5`，它同时是 5 本老账的扩展：`_mid`/`_mid_n8_v3`/`_mid_n8_v4`/
                 `_mid_n8_v5`/`_mid_n12_v4` ⇒ 一次注入该有 5 条红，**红数由账册结构定，不由字典序定**）。
                 第二类自纠（**同一类漂移第二次出现**）：上一轮定这条规则时把排序键第一位写成了
                 "反向对条数"，可打印的强度是"注入实际红几条"（=`len(hub[nw])`）——两本新账入库后
                 `_mid_n16_v5`（扩展 5 本）与 `_mid_n8_hi_v4`（扩展 4 本）反向对**都是 3 条**，
                 平票被文件名倒序判给小的那本 ⇒ 这一档停在 4 条红，而它本来该红 5。
                 教训：**"挑最强的一组"这句规则必须把"强"定义成这一档自己要打印的那个数**，
                 否则它会在账册增长时又一次悄悄降级，而且降级的方式恰好藏在平票里。
                 现在键是 (len(hub[nw]), 反向对数, 名字)。
                 这里有一处**自纠**：上一版挑的是"文件名排在后面的第一个反向对"，本轮那本 12 条流的新账一入库，
                 满足旧挑法的第一个候选就变成 `_mid_n8_t6_v4 ⊆ _mid_n12_t6_v5`——它只扩展**一本**老账，
                 于是这一档会从 4 条红悄悄降到 1 条红，**表上只写"4 条"的话就看不出来了**。
                 为什么旧规则要排除"老账名字排在后面"的对：老对账规则按文件名谁小谁老，`_n12` 恰好排在
                 `_n8` 前面 ⇒ "8 条流 ⊆ 12 条流"这一对整个不在对账里，上一轮预登记的②就是这么被漏掉的。
                 注意它**只挑一本**注入：其余 hub（`_n8_v3/_n8_v4/_n8_v5 ⊆ _mid_n12_v4` 等）同样该红，
                 这一档测的是"偷改老读数必须被抓"这件事本身，不是把 21 对全跑一遍）
  --sibdrift    给重跑本的一个训练路径读数（seed_floor_imgs）偷偷 +7 → 实测 3 条红：⑤ 逐本点名
                （钉"数字真动了的时候闸⑤ 还活着"，与下面两条一起构成 ⑤ 的三端）
  --oldsib      兄弟账退回 glob 顺序最老那本 → 实测 1 条红：⑤（跨了判据版本 ⇒ 把派生字段的正常
                演进报成漂移）。这一档钉的就是"基准挑哪一版"本身；它是**假红**，
                出现即说明选择规则被改坏了。
  --laddersib   把闸⑤"重跑本必须同阶梯"那条排除关掉 ⇒ 实测 2 条红：⑤ 点名 `_mid_n8_hi_v4`
                （红的内容是 `ladder` 长度与随之找不到的字段）。这一档钉的是**扩格本不许冒充重跑本**
                 那一端：没有它，`--ladder-high` 一入库闸⑤ 就会把"多买了两格"报成"训练路径动了"；
                 有了它，这条排除是被一档红盯着的规矩，不是一句注释）
  --v4blind     v4 瞎掉倒挂腿（inverted_leg=False）→ 实测 12 条红：①×9 + ⑤ + ⑦ + ⑧
                （在册"倒挂"账被 ① 点名——这条腿一旦被摘，档案里那句"倒挂"就复算不出来、落地预言也落空。
                 ⑧ 那条红的仍是"旧腿瞎了会连带让新腿没原料"；本轮 ① 多的两条是 `_mid_n16_t6_v5`
                 与 `_mid_n16_v5`——前者档案判『倒挂』，后者档案 v4 并排列也写着『倒挂』，
                 所以摘腿现在多两个当场可抓的对象）
  --v4loose     倒挂腿忘掉"同号且更大"，只要判到污染型就改口 → 实测 1 条红：只有 ⑦，
                红的内容是"参评三本污染型全翻成倒挂"（连坐）。**④ 抓不到它**——不是 ④ 松，
                是 ④ 定义在 v2/v3 复算上，结构性地看不见 v4 这条腿。
  --v5blind     v5 瞎掉那条 CI 腿（`ci_leg=False` ⇒ 与 v4 逐字同判）→ 实测 4 条红：①×2 + ⑤ + ⑧
                （装饰端：v5 若与 v4 一字不差，⑧ 的翻转集合就是空的，"换了判据"什么都没换到；
                 ① 与 ⑤ 那条红是 v5 落地本自己的——它的档案判词按 v4 复算不出来，
                 所以这一档同时钉住了"落地本必须真的被 v5 判过"。本轮多出的那条 ① 是 `_mid_n16_v5`：
                 它按 v4 复算判『倒挂』（份额 105% 越界）而档案记的是 v5 那句『污染型』⇒
                 **v5 那条腿第一次在一本"当场复算不出来"的新账上承重**，不再只是翻旧账的口径）
  --v5strict    把 v5 那条腿的分母从 CI **半宽**换成**全宽** `hi − lo` → 实测 3 条红：① + ⑧×2
                （过头端：全宽之下 `_mid_n8_t6_v4`（29.8 vs 44.7）也跟着翻 ⇒ 翻转集合变五本、
                 该站着的那本站不住；本轮多的一条 ① 是 `compress_report_v4f_mid_n12_t6_v5.json`
                 这本 12 条流的新账——它出界 28.0 张 < 全宽 28.8 张，同一把"过宽的尺"喂进去，
                 它自己档案里那句"倒挂"也当场复算不出来。
                 **但同一支坐标在 16 条流那本上不翻**：`_mid_n16_t6_v5` 出界 29.2 张 > 全宽 22.7 张
                 ⇒ 全宽这把尺在 8/12/16 条流上依次判 翻 / 翻 / 不翻。这条不翻**不计进红数**，
                 它如实记作：补流把半宽收窄的同时也收窄了"全宽"，过头端因此少了一条当场可抓的红。
                 **这一档钉的是这条腿里唯一没被预言钉死的算术自由度**——
                 不拆它，"半宽还是全宽"就成了"实现怎么写都算通过"）
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

# "重跑/后继账"的名单只写这一处：闸②⑤⑦⑧ 都拿它区分"旧账"与"新判据下落地的账"。
# 漏一个版本，那本账就会被算成"新一次翻案"，把预登记的翻转集合算脏（第十一发④ 的教训：
# 老闸不会自动获得对新版本的判决权，必须显式接进去）。
RERUN_RULES = ("v3", "v4", "v5")


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


def v5_of(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, s_top, pair_step):
    """v5 = v4 的全部条件 **且** 出界幅度 > 配对步长 CI 的**半宽**（默认即诚实实现）。"""
    return _orig_sv(s_small, s_probe, n, dirn, a_small, a_probe, anc_a,
                    s_top, pair_step, inverted_leg=True, ci_leg=True)[0]


if MODE == "--v5blind":                # v5 瞎掉那条 CI 腿：与 v4 逐字同判 → 该被闸⑧ 报成装饰
    def v5_of(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, s_top, pair_step):   # noqa: F811
        return v4_of(s_small, s_probe, n, dirn, a_small, a_probe, anc_a, s_top, pair_step)
elif MODE == "--v5strict":             # 半宽换成全宽：唯一还站着的倒挂也跟着翻 → 该被闸⑧ 报成过头
    sf.FULL_WIDTH_MODE = True


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
    v5 = v5_of(s_small, s_probe, n, dirn, small, a_probe, anc,
               None if MODE == "--decorative" else s_top,
               d.get("paired_step_small_to_anchor"))
    v2 = v2[0]
    rows.append((path, rule, rec, v1, v2, v3, s_top is not None, anchor_src, n, small, a_probe, anc, v4, v5))

    same = SHORT(rec) == SHORT({"v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5}[rule])
    print(f"{path}  [{n} 条流 | 阶梯顶 {'在账里' if s_top else '不在账里'} | dirn 来源：{anchor_src} "
          f"| α {small:g}/{a_probe:g}/{anc:g}]")
    print(f"   复算  v1={SHORT(v1):<4} v2={SHORT(v2):<4} v3={SHORT(v3):<4} v4={SHORT(v4):<6} "
          f"v5={SHORT(v5):<6}   档案记（{rule}）={SHORT(rec)}  {'== 复算' if same else '**不等：喂错数字了**'}")
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
    if d["verdict"].get("verdict_under_v4_rule") and \
            SHORT(d["verdict"]["verdict_under_v4_rule"]) != SHORT(v4):
        fails.append(f"闸① {path} 档案里的 v4 并排判词与复算不符（记 "
                     f"{SHORT(d['verdict']['verdict_under_v4_rule'])}，复算 {SHORT(v4)}）")
    if not same:
        fails.append(f"闸① {path} 的档案判词复算不出来（档案记 {SHORT(rec)}，{rule} 复算 "
                     f"{SHORT({'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4, 'v5': v5}[rule])}）——"
                     f"先查是不是读了同一批数字，别谈 {rule}")

# 翻转集合只在**旧账**（记录在 v1/v2 名下那几本）上算：同一发实验的 _v3 重跑本不是"新的一翻"，
# 它是闸⑤要验的"预言落地"，算进翻转集合就是把一件事数两次。
flips_v2_v3 = [p for p, rule, rec, v1, v2, v3, *_ in rows
               if rule not in RERUN_RULES and SHORT(v2) != SHORT(v3)]
expected = ["compress_report_v4f_low_n8_thin_v2.json"]
print(f"\n[闸① 复算自证] {len(rows)} 本档案："
      f"{'判词全部由产出它的那一版复算出来 → 通过' if not any(f.startswith('闸①') for f in fails) else '见上：有复算不出的'}")

print(f"[闸② v3 相对 v2 的翻转集合（只在旧账上算，"
      f"{sum(1 for r in rows if r[1] not in RERUN_RULES)} 本）] {flips_v2_v3 or '（空）'}")
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
flips_v3_v4 = [r[0] for r in rows if r[1] not in ("v4", "v5") and SHORT(r[5]) != SHORT(r[12])]
can_flip = [r for r in rows if r[1] not in ("v4", "v5") and r[6]]   # 阶梯顶不在账里的本**参评不了**这条腿
print(f"[闸⑦ v4 相对 v3 的翻转集合（v3 及更早的 {sum(1 for r in rows if r[1] not in ('v4', 'v5'))} 本里，"
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

# 闸⑧：v5 相对 v4 只许翻掉预登记那四本（出界幅度压不过"把两端钉在一起的这一步"自己的 CI 半宽）。
# 参评＝**判词记录在 v4 名下**且阶梯顶在账里的那些本：v5 落地的那本是"预言兑现"，归闸⑤ 管，
# 算进这里就是把同一次翻案数两遍（与闸②⑦ 同一套规矩）。
EXPECTED_V5 = ["compress_report_v4f_mid_n12_v4.json", "compress_report_v4f_mid_n8_hi_v4.json",
               "compress_report_v4f_mid_n8_v4.json", "compress_report_v4f_n8_t6_v4.json"]
MUST_STAY = "compress_report_v4f_mid_n8_t6_v4.json"      # 出界 29.8 张 > 半宽 ±22.4 张 ⇒ v5 之下唯一还站着的倒挂
can_flip_v5 = [r for r in rows if r[1] == "v4" and r[6]]
flips_v4_v5 = [r[0] for r in can_flip_v5 if SHORT(r[12]) != SHORT(r[13])]
print(f"\n[闸⑧ v5 相对 v4 的翻转集合（记录在 v4 名下、阶梯顶在账因而**参评**的 {len(can_flip_v5)} 本；"
      f"v5 落地那本归闸⑤）] {flips_v4_v5 or '（空）'}")
if not can_flip_v5:
    fails.append("闸⑧ 一本参评的账都没有——不是通过，是没测到")
if sorted(flips_v4_v5) != sorted(EXPECTED_V5):
    fails.append(f"闸⑧ v5 的翻转集合应为 {sorted(EXPECTED_V5)}，实为 {flips_v4_v5}——"
                 f"谁都翻不动是这条 CI 腿装饰，多翻是连坐（把界内或别的支的账一起改了）")
stay = [r for r in can_flip_v5 if r[0] == MUST_STAY]
if not stay:
    fails.append(f"闸⑧ 该保持倒挂的 {MUST_STAY} 不在参评表里——它没了就等于这条腿没人验过")
elif SHORT(stay[0][13]) != "污染型·倒挂":
    fails.append(f"闸⑧ {MUST_STAY} 在 v5 下判成了 {SHORT(stay[0][13])}：它的出界幅度 29.8 张 > 半宽 22.4 张，"
                 f"翻掉它说明这条腿被写松了（全宽？还是根本没比大小？）")
else:
    print(f"  在「记录在 v4 名下」这个参评口径里唯一还站着的倒挂：{MUST_STAY}（v5 复算仍是{SHORT(stay[0][13])}）")
inv5 = {}
for r in rows:
    if SHORT(r[13]) == "污染型·倒挂":
        inv5.setdefault(setup_of(docs[r[0]])[:5], []).append(r[0])
print(f"  【现读】v5 复算判『倒挂』的账共 {sum(len(v) for v in inv5.values())} 本，"
      f"按配置（档/宽/温度列）归并成 {len(inv5)} 个坐标：")
for k, v in inv5.items():
    note = f"（同一坐标的 {len(v)} 本：只差流数，逐位对账线在 ⑥）" if len(v) > 1 else ""
    print(f"        {k[1]} × S{k[2]} × T={k[4]} ⇒ {', '.join(v)}{note}")
beside = [r[0] for r in can_flip_v5 if not SHORT(r[12]).startswith("污染型") and SHORT(r[13]) != SHORT(r[12])]
print(f"[闸⑧ 的附带不连坐检查] v4 复算不在第 1 支里的参评账 {len([r for r in can_flip_v5 if not SHORT(r[12]).startswith('污染型')])} 本："
      + ("v5 判词逐字相同（这条腿结构上碰不到别的支）" if not beside else f"**被连坐：{beside}**"))
if beside:
    fails.append(f"闸⑧ 别的支被连坐：{beside}——v5 只许动倒挂那条腿的触发界")


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


fresh = [p for p, rule, *_ in rows if rule in RERUN_RULES]
if not fresh:
    fails.append("闸⑤ 一本 _v3/_v4 重跑账都没有——预言没人验过就等于没落地，别宣布新判据换到了东西")
else:
    print(f"[闸⑤ 落地自证] 已重跑 {len(fresh)} 本：{[p for p in fresh]}")
for p, rule, rec, v1, v2, v3, *_ in rows:
    if rule not in RERUN_RULES:
        continue
    d_new = docs[p]
    if MODE == "--sibdrift" and isinstance(d_new.get("seed_floor_imgs"), (int, float)):
        # 突变：重跑本偷偷动了训练路径的一个数（不改判词，只改读数）→ 闸⑤ 必须逐本点名。
        d_new["seed_floor_imgs"] = d_new["seed_floor_imgs"] + 7
    # 同配置同流的那本才是"重跑本"：名字推出来的兄弟账可能是**另一发实验**（换了流数=换了实验），
    # 拿它当基准逐位对账，红了也不是判据的错。所以按**流集合**认，不按文件名认。
    sibs = [q for q in docs if q != p and same_cfg(d_new, docs[q])]

    def is_rerun(q, dn=d_new):
        """重跑本 = 同流同配置**且同阶梯**。阶梯不同是另一发（扩格：老格子重测、新格子是靶子），
        它的逐位对账线是闸⑥，不是这里——让闸⑤ 去比扩格本，红了的是 `ladder` 的长度，
        而那是设计使然不是漂移。`--laddersib` 把这条排除关掉，用来钉"扩格本冒充重跑"这一端。"""
        return (docs[q]["seeds"] == dn["seeds"]
                and (MODE == "--laddersib"
                     or sorted(map(float, docs[q]["ladder"])) == sorted(map(float, dn["ladder"]))))

    same_flow = [q for q in sibs if is_rerun(q)]
    wid = [q for q in sibs if q not in same_flow and docs[q]["seeds"] == d_new["seeds"]]
    if wid:
        print(f"  {p}：同流同配置但有 {len(wid)} 本**阶梯不同**（扩格本）：{wid} → 不当重跑本，"
              f"老格子的逐位对账交给闸⑥")
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
    pred_col = {"v4": 12, "v5": 13}.get(rule, 5)   # 落地要落的是"对兄弟账施**本作**判据"的那个预言
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
    hub = {}                                   # 大账 → 它扩展了的那些老账
    for rev, pa, pb in cand:
        o, nw = (pa, pb) if pair_dirs(docs[pa], docs[pb])[0] is docs[pa] else (pb, pa)
        hub.setdefault(nw, []).append(o)
    # 优先拆"文件名顺序与包含顺序相反"的那一对：按文件名挑对的老规则对它无感，
    # 退回任意一对的话，这一档就只是在测"闸⑥ 还活着"，测不到这次修的东西。
    # 反向对里再挑**扩展老账最多的那本大账**：上一版取"第一个反向对"，于是账册一涨、注入点就漂到
    # 新入库那本（它只扩展一本老账）⇒ 红条数从 4 掉到 1，强度悄悄降了而表上看不出来。
    rev_new = {nw: [o for o in olds if o > nw] for nw, olds in hub.items() if any(o > nw for o in olds)}
    # 排序键的第一位是**注入一次实际红几条**（=len(hub[nw])，这本扩展的老账总数），不是反向对条数：
    # 第十四发入库后 `_mid_n16_v5`（扩展 5 本）与 `_mid_n8_hi_v4`（扩展 4 本）反向对都是 3 条，
    # 上一版按反向对数排 ⇒ 平票被文件名倒序判给小的那本，强度又悄悄降了一条——正是这条规则要防的漂移。
    best = (max(rev_new, key=lambda nw: (len(hub[nw]), len(rev_new[nw]), nw)) if rev_new else None)
    best = (best, rev_new[best]) if best else None
    if best is None:
        fails.append("闸⑥（拆红模式 --drift）找不到可比的账对，注入没处放——这条红不算通过")
    else:
        new, rev_olds = best                                   # new=注入的那本大账，rev_olds=文件名排在它后面的老账
        old = sorted(rev_olds)[0]                              # 基准挑反向的那本（按文件名定向的旧规则对它无感）
        kk = next(iter(arm_index(docs[old])))
        r = arm_index(docs[new])[kk]
        r["wrong_kd"] += 1
        r["gap_imgs"] += 1
        print(f"  （拆红注入）把扩出来的那本 {new} 的 α={kk[0]:g}/种子{kk[1]} 那格 wrong_kd 与 gap 各挪 +1"
              f"（基准 {old} 同格不动）——只有这一道闸会看见它；这一本同时是 {len(hub[new])} 本老账的扩展"
              f"⇒ 注入一次该有 {len(hub[new])} 条红（条数由账册结构定，不由字典序定）"
              + (f"。这一对的老那本文件名排在后面（{old} > {new}），按文件名定向的旧规则看不见它"
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
