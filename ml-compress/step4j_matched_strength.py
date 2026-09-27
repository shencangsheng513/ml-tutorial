# -*- coding: utf-8 -*-
"""第4课j：把 α·‖∇(T²KL)‖ 钉成常数、只换温度——还第4课i 自己写在 `limitations` 里的那笔明码债。

第4课i（零训练）把 α=0.05 那一格的跨列差（T=6 比 T=3 多错 28.1 张）拆成两条候选通道，判的是
**出口 (b)：大头在内容通道**。但它的判法是**排除法**：拿第 0 批实测的 ‖∇(T²KL)‖ 算出
α*=0.080（把 T=6@0.05 的力度钉回 T=3 所需的 α），再用 T=3 阶梯夹住 α* 那一段的**局部斜率**折算成
share = −5.1 张 = 跨列差的 18%。那句债逐字抄自那本账（第廿九发 B 之前这里是一段转述，冒充成引号里的原文）：
"内容通道没有换算仪器：KL(p_T3‖p_T6) 的单位是 nat，不是张。要把它变成测量需要一发训练
（把 α·‖∇(T²KL)‖ 钉死、只换 T；阶梯 5 格 × 8 条流 = **40 次训练**，本脚本一次都没付）。"
**这一发就是那一次训练。**

做法：α 与 T 在 loss 里不正交（T² 补偿不精确，第4课c 已实测在册），但**单位 α 下的 KL 梯度范数
是 α 的线性系数**（‖∇(α·T²KL)‖ = α·‖∇(T²KL)‖，α 是纯标量前置），所以"钉死力度"这件事有闭式解：
T=3 阶梯上每一格 α 配一个 T=6 那一列的等力度剂量

        α'(α) = α · ‖∇(T²·KL)‖_T=3 / ‖∇(T²·KL)‖_T=6 = α × 1.41276 / 2.26708 = α × 0.6232

于是"同力度、只换温度"这一对比较里，权重通道被**逐格构造对齐**（不再是折算），剩下的差只能来自软标签
的形状。两列的每一格共用同一粒种子的 CE 臂 ⇒ 配对差里 CE 锚自动抵消，连减数都不需要重训。

  问句（只有一句）：**把 KD 项推参数的力度钉成同一个数之后，T=6 那一列还比 T=3 多错吗？**

闸（全部在第一次训练之前，任一红 ⇒ 退出、零落盘、不入库）：
  闸 I（读账复算）：两本在册阶梯账（T=3 / T=6，同 8 条流）每格均值必须从它自己的逐流 gap 现算命中，
      **并且**每格的逐流 gap 必须等于那本账 `arms` 里 `wrong_ce − wrong_kd` 的重排——因为这一发的
      配对差直接拿 `wrong_kd` 当货币（CE 抵消），arms 与 ladder_summary 不一致就是这本账自己打架。
  闸 II（配对前提）：两本账流名单逐位同序、档/宽度/学生族/种子底/闸格/锚点剂量/teacher_acc 全同、
      温度列必须不同、五格 α 必须两本都有。
  闸 III（CE 减数 + 老师身份）：两本账每粒种子的 `wrong_ce` 逐位相同；老师权重现算测试准确率必须
      等于两本账自报值 = 板块常量 0.9798。（沿用第4课i 闸 I 的理由：减数不唯一，差里掺 CE 自己的漂移。）
  闸 IV（**这把新尺子的接线闸**，也是最要紧的一道）：本脚本现算的第 0 批 ‖∇CE‖ / ‖∇(T²KL)‖（T=3、T=6）
      必须与 `compress_report_v4c_mid.json` 的 `gradient_probe.by_T` **且**与
      `compress_report_v4i_mechanism_v2.json` 的 `probe_by_T[*]["batch0"]` 逐位相同；
      并且由同一对梯度重算的顶层配平 α 必须等于第4课c 在册那个数（α=0.7 → **0.436**）。
      为什么这一道归本发独有：α' 全由这两个范数算出来，尺子错一格 = 五个剂量全错，
      而那五格正是本发唯一的新原料。第4课c 在册的 `alpha_to_match_anchor` 恰好就是本发阶梯顶那一格
      的 α' ⇒ **免费拿到一条"新算式复现在册配平数"的对账线**（不是新造的常数）。
  闸 V（跨脚本逐位，各 3 条老流）：本脚本重训 (α=0.3, T=3) 与 (α=0.3, T=6) 两格，`wrong_kd` 必须
      分别逐位命中两本账的对应臂。**这是本发的"恒等闸"**：配对差一头是盘上新训的、另一头是从档案读的，
      没有这道闸就没法说这两头出自同一台训练器。（恒等闸 α=0 不适用——阶梯里没有 0 那一格，
      判据也不吃它；接线证明由这道逐位闸承担，它比 α=0 强：它核的是 KD 分支。）

出口（预登记，跑之前写死，此后只许追加"落地"）：主格 = **α=0.05 那一格**（第4课i 的 α* 住在那里、
倒挂也住在那一列），其余四格是"这个残余是不是只有一格才出得来"的对照。承重口径沿用板块那条：
逐流配对差的 95% CI 不含 0（第二十发已证它与"出界幅度 > 半宽"逐字等价）。符号：d = wrong_kd(T3@α) −
wrong_kd(T6@α') ⇒ **d<0 = 同力度下 T=6 多错**。
  (a) 主格 CI 含 0 **且**五格全含 0 ⇒ 力度解释吃下全部跨列差，第4课i 的出口 (b) 被训练侧**否证**。
      这一支必须当**事故**处理并当场写明矛盾在哪：探针（初始一步、第 0 批）与 600 步训练对"有效剂量"
      给出不同答案 ⇒ 要改的是探针那句"权重通道"，不是这一发的数。
  (b) 主格 CI 不含 0 且 d<0 ⇒ 内容通道**第一次有张数**：报"同力度下 T=6 仍多错 |d| 张（CI、同号数）"。
      与第4课i 那句 share=−5.1 张并排读，但**不许相减当分解**（α 曲线非单调，见自我审计② 那条）。
  (c) 任一格 CI 不含 0 而 d>0（同力度下 T=6 反而更浅）⇒ 力度这一维被**高估**：那个范数不是 600 步的
      有效剂量（第4课c 自我审计④ 那条"起手配平不是全程配平"兑现成读数）⇒ 本发的归因作废方向是**探针**，
      不许读成"内容通道有益"。
  (d) 主格含 0 但其余格有不含 0 的（混合）⇒ 逐格不一致 = "匹配强度"这个概念在剂量上不守恒。
      这一支**不判哪个通道**（三分支里没有"部分承重"这一格位置），只登记成下一版判据的债；
      描述性读数不进判词，是板块老规矩。
零 API 费、纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe -u step4j_matched_strength.py --nseeds 3   # 预演，账验完即删、不入库
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe -u step4j_matched_strength.py --nseeds 8   # 真跑
  # 拆红（都在训练之前退出，零成本）：
  ... --nseeds 8 --probeblind    # 不许 backward，改用教科书假设「T² 补偿精确 ⇒ α'=α」⇒ 只有闸 IV 该红
  ... --nseeds 8 --scramble      # 给 T=6 那本账每格 gap +1 ⇒ 只有闸 I 该红

**价签（逐项列加法，加数全部进和——第十五发那条口径的第二次兑现）**：
  靶子 5 格 × 8 条流 = **40 次**（第4课i 欠的那个数）
  + 闸 V 的 T=3 格 3 条老流 = **3 次**
  + 闸 V 的 T=6 格 3 条老流 = **3 次**
  = 真跑 **46 次**；预演本（3 条流）另计 15 + 3 + 3 = **21 次**，验完即删。
  比菜单上那句"40 次"多 6 次，多出来的全是跨脚本逐位闸，且它们在最贵的一步之前就要么红要么绿。

维度账（本发买的是哪一种"买"）：**覆盖**——同一批 8 条流、同一档、同一副学生、同一个老师，
只把 α 搬到五个从未训过的坐标（0.0312/0.0623/0.1246/0.1870/0.4362）。它**不新增独立性维度**
（七维一维都没换）、**不买分辨率**（流数没动），所以出口 (b) 若兑现只能写成
"机制归因第一次有了直接读数"，**不许写成"内容通道被复证"**。它买的也不是判据版本（板块判据 v5 一字不动，
本发根本不进形状判据——判的是另一条量：同力度跨温度的配对差）。

落地（跑完之后追加，上面预登记原文一个字不动）：
  ① 五道闸全绿，两档拆红各自只红自己那道（**都在第一次训练之前退出，各 0 次训练**，RC 均为 1）：
     `--probeblind` 红在**闸 IV**（改用教科书"T² 补偿精确"推定时，T=6 的 ‖∇(T²KL)‖ 报 1.41276
     而 `v4c_mid`/`v4i_mechanism_v2` 两本基准账记 2.26708）；`--scramble` 红在**闸 I**
     （α=0.05 格均值现算 −72.8 ≠ 档案 −73.8）。
  ② **预演本抓到两个 bug，都崩在任何一次训练之前**（崩≠红，所以它们不算闸的功劳，但也不算白跑）：
     ⑴ 常量名拼错：六处 `RUNS` 应为 `RUNGS` ⇒ `NameError`，死在"顶层配平 α 对账"那一行、第一次 backward 之前。
     ⑵ **这一发最该记的一条**：`g3, g6 = gate_strength_map(...)` 解包了一个**按温度索引的字典** ⇒
        拿到的是键 `3.0/6.0` 而不是那两个范数，于是阶梯顶的等力度剂量算成 **0.350 而不是在册的 0.436**。
        它没有红在任何一道"设计来挡它"的闸上——拦住它的是**那条免费的"新算式复现在册配平数"对账线**
        （闸 IV 第二段）。若第4课c 没在册记过 0.436，五个剂量会全体偏移同一个常数因子，
        而下面每一道闸、每一条 CI、每一个同号数**照样全绿**：错的是剂量本身，没有一处能自证。
        可迁移形式：**能给一条新算式找到"复现某个在册数字"的自由对账线，就别省。**
  ③ 货币方向自纠（在真跑之前现读抓出来的，与第4课i 那条"散文方向反"同类，这次是**算式**方向反）：
     未配平对照原写成 `gap(T3) − gap(T6)`，与本发主表 `d = wrong_kd(T3) − wrong_kd(T6@α')` 符号相反 ⇒
     同一张表两行两种货币，读者会把"−32.9 vs +28.1"读成符号翻转。改成 `hi − lo` 后 α=0.05 那行读出
     **−28.1 张**，与第4课h/第4课i 那颗钉子**逐位相同** ⇒ 这条对照顺带成了第二次跨脚本对账（零成本）。
  ④ 读数（8 条流、46 次训练、产物 `compress_report_v4j_matched_n8.json`、实测 43 秒）：
     等力度比 g3/g6 = **0.62316** ⇒ 五格剂量 0.03116 / 0.06232 / 0.12463 / 0.18695 / 0.43621
     （顶格逐位复现在册 0.436）。**出口走 (b)**：主格 α=0.05（T=6 那一头是 α'=0.0312）
     d = **−32.9 张、CI [−42.3, −23.5]、8/8 同号、二项 p=0.0078** ⇒ **内容通道第一次有张数**：
     把 KD 项推参数的力度钉成同一个数、并且给 T=6 那一头**更低**的 α（0.0312 < 0.05）之后，
     T=6 仍然多错 33 张。α=0.1 那格也不含 0（−19.4 张、7/8、p=0.0703）⇒ 承重格数 2/5；
     α≥0.2 三格全含 0（−1.2 / +2.0 / −3.5 张）⇒ **残余只住在小剂量端**，这与第4课i 那个 α*=0.080
     住在的位置一致（但那一致是并排读出来的，不是本发的判据）。接线：闸 V 6/6 逐位命中两本账
     （794/797、795/786、821/820）。
  ⑤ **一条描述性读数，不进判词、而且它欠一次配对**：问"配平这一步换掉了多少"，点估计是**反号**的——
     未配平 −28.1 张 → 等力度 −32.9 张，按第4课i 的折算该收回 +5.1 张，实测却往深里走了 4.8 张。
     但这两行相减**不是**一个配对统计量（该配的是同一列上 `wrong_kd(T6@0.0312) − wrong_kd(T6@0.05)`），
     本发没为它算 CI，而两条 CI 大幅重叠（[−42.3,−23.5] vs [−42.3,−14.0]）⇒ 这句话只到**未判定**。
     欠账（配方已写死、**下一发零次训练**）：把本档 `matched_ladder["α=0.05"]` 每条记录的 `wrong_kd`
     与 `v4f_mid_n8_t6_v5.json` 的 `arms["α0.05 种子<seed>"]["wrong_kd"]` 按同 8 条流配对，
     报那一格的"配平换掉了多少张"（两头都是 T=6 列、同一批流、闸 V 已证同一台训练器）。
  ⑥ 边界（`limitations` 里已入库，这里只重复最要紧两条）：**起手配平不是全程配平**——‖∇(T²KL)‖ 是
     第 0 批、初始那一步的量，600 步里两项相对尺度会变（出口 (c) 那一支正是为这条边界留的，本次没走到）；
     以及坐标账：本发只买**覆盖**，七维一维未换 ⇒ "内容通道被复证"一个字都不能说。
  ⑦ **成本自纠（价签那句"预演 21 次"少算了一倍）**：预演实跑四趟——前两趟在第一次训练之前就崩/红
     （②那两条，各 0 次），第三趟绿（21 次）；③ 那处货币修正之后**又**完整预演了一趟（再 21 次，
     账验完即删、未入库）。⇒ 整链实付 **88 次 = 预演 42 + 真跑 46**，零 API 费。
     规规矩矩写全的原因还是那条：成本句漏加的数都是事后才发现的。
  ⑧ 债的状态：`compress_report_v4i_mechanism_v3.json` 的 `limitations` 第 4 条（按内容锚现读，序号每升一版就 +1）
     那句"要把它变成测量需要一发训练…40 次训练"从此**已付**。入库不改，那句话原样留在老档里，
     付账凭据是本档与第4课j 这本账。
     **第廿九发 B 还的就是这一句的两处错**：v1 的 `pays` 把"需要一发训练"记成了"要一次训练"
     ——序号对、引文逐字命中 0 条，是**转述冒充引用**（守卫 ⑮ 上岗第一天抓到的，见 `_red_v3.py` ⑮ 段）；
     同时读档指针还写着 v2（v3 已在册）。两处都在本发改指/改逐字，判词一个字没动 ⇒ 产物升版 v2、
     v1 留盘逐字段对账（对账线：除 `pays`、`archives`、`gates.pairing` 三处外，全部读数与 v1 逐位相同——
     第三处也被迫改过：那句写死的"同 8 条流"在 16 条流那本里会直接变成假话，顺手把两本阶梯的账名带上）。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (DATA_TIERS, FLOW_SEEDS, SEEDS, T95, TEACHER_ARCHIVE_ACC, VAL_N,
                           check_flow_constants, logits_of, summarize, train_student)
from step4c_alpha_T_sweep import power_preflight, term_grad_norms
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
TIER, WIDTH, ARCH = "中数据4500", 128, "mlp"
T_LO, T_HI = 3.0, 6.0
RUNGS = (0.05, 0.1, 0.2, 0.3, 0.7)          # T=3 那一列在册阶梯的五格（0 是恒等闸，不是臂）
PRIMARY = 0.05                              # 主格：第4课i 的 α* 与倒挂都住在这附近
GATE_ALPHA = 0.3                            # 闸 V 重训那一格（两本账都有、且是老 3 条流可对账的格）
# 两列阶梯账名**不写死**（第廿九发 C 前置）：以前是 L3/L6 两个模块常量，写死 `_n8` ⇒ 16 条流那一发
# 会拿 8 条流的阶梯当配对另一头。现在走 ladder_names(nseeds)，见下面的 def。
V4C = "compress_report_v4c_mid.json"           # 探针在册基准 + alpha_to_match_anchor
V4I = "compress_report_v4i_mechanism_v3.json"  # 这笔债的原文所在（第廿九发 B：改指**在册最高版**）
DEBT_ANCHOR = "内容通道没有换算仪器"            # 债句的**内容锚**：序号会漂，这半句不会
# 档案版本位（第廿九发 B 起）：产物只能升版，同一坐标重跑不许盖旧账，旧账留盘逐字段对账。
# 按坐标而不是按全局记：8 条流那本在第廿九发被升版（引文改成逐字 + 指针改指 v3），
# 而 16 条流那本（第廿九发 C）是**那个坐标的第一本**，它没有 v1 可 supersede。
REPORT_VER = {3: 1, 8: 2, 16: 1}
# 买流数那一发的"前一档流数"：配对 CI 半宽比要对着它读（第廿七发把这句话判倒之后，
# 开口必打的那一行需要一个 sibling，否则它连"预期是多少"都印不出来）。
SIBLING = {16: 8}


def ladder_names(nseeds: int) -> tuple[str, str]:
    """两列阶梯账名**按流数派生**（第廿九发 C 的代码前置）。

    为什么这条必须进代码而不是留在散文里：写死 `_n8` 的那两行让 16 条流那一发去拿
    8 条流的阶梯当配对另一头 ⇒ 闸 II 比的是两本阶梯之间的一致性（它们当然一致），
    而本发真正的新数（16 条流的 d）与旧数（8 条流的 cells）之间的流数差**不在任何闸的射程里**。
    这与第廿三发 `step4c.v4b_path` 读自己模块全局那笔缺陷是同一个形状。

    `nseeds == 3` 是预演档：盘上没有 3 条流的 T=6 列阶梯（那一列是第廿一发之后才补的），
    所以它照旧借 8 条流那两本走一遍接线，**不入库、不产读数**——这里把它打印出来而不是让它静默。
    """
    n = 8 if nseeds == 3 else nseeds
    if n != nseeds:
        print(f"[读档] 预演档 nseeds={nseeds} 借 {n} 条流那两本阶梯走接线（不入库、不产读数；"
              f"入库账都带自己的 _n 位）")
    suf = "" if n == 3 else f"_n{n}"
    return f"compress_report_v4f_mid{suf}_v5.json", f"compress_report_v4f_mid{suf}_t6_v5.json"


def debt_index(v4i: dict, where: str) -> int:
    """那句债在 `where` 这本账的 limitations 里排第几（返回 1 起的序号）。

    为什么按内容而不是按序号：`limitations` 每一版都在**头部**插一条变更说明（v1 6 条 → v2 7 条
    → v3 8 条），所以同一句话的序号每升一版就 +1。第廿三发把 v3 入库之后，本脚本原先写死的
    "第 3 条"就只对 v2 成立了，而守卫当时没有任何一档会因这件事红（第二十六发把它记成了新欠之一）。
    命中 0 条 = 那句话被改写或指错了账本，命中 >1 条 = 锚点不唯一——两种情况都没有资格报序号，
    一律在**第一次训练之前**退出（贵的失败要提前到花钱之前，第4课f 第十发那条老规矩）。
    """
    hits = [i for i, s in enumerate(v4i.get("limitations", [])) if DEBT_ANCHOR in str(s)]
    if len(hits) != 1:
        raise SystemExit(f"{where} 的 limitations 里按『{DEBT_ANCHOR}』命中 {len(hits)} 条（要恰好 1 条）："
                         f"0 条 = 那句话被改写或这本不是债源，>1 条 = 锚点不唯一；"
                         f"两种情况都不许再引用序号，先修锚点再来付账")
    return hits[0] + 1


def out_name(nseeds: int) -> Path:
    """流数是本发的判别维度：不带 _n8 会让 3 条流那本预演账撞掉真跑那本。

    版本位（第廿九发 B）按坐标查 `REPORT_VER`，不给全局数字：16 条流那本是那个坐标的第一本，
    它没有 v1 可 supersede，跟着 8 条流那本一起写 v2 会凭空造出一本不存在的旧账。
    """
    suf = "" if nseeds == 3 else f"_n{nseeds}"
    ver = REPORT_VER[nseeds]
    return HERE / f"compress_report_v4j_matched{suf}{'' if ver == 1 else f'_v{ver}'}.json"


def sibling_halfwidth(out: Path, nseeds: int, diffs: dict) -> dict:
    """本发的 d 的 CI 半宽 ÷ **前一档流数**那本同格的半宽（第廿九发 C 的事后读数）。

    为什么要单独一个函数、而且要跟 ex-ante 那一行分开印：`power_preflight` 读的是同坐标 α×T 网格的
    历史比值（别人的尺子），这一行才是本发这个量自己买到的东西。拿别人的比值给自己的发交差，
    等于用结论证结论。

    三种"没测到"分开出声，一种都不许返回空 dict 冒充通过：这一档没登记 sibling（不该走这里）、
    sibling 那本不在盘上、那本在但剂量格对不上。第三种返回 `n_cells: 0` 而不是不算——
    0 格可比是一个**读数**，漏掉它就等于把"我比不了"说成"我比过了"。
    """
    if nseeds not in SIBLING:
        return {}
    prev = SIBLING[nseeds]
    sib = out_name(prev)
    if not sib.exists():
        print(f"\n[事后半宽比·本发的 d] 前一档流数那本 {sib.name} 不在盘上 ⇒ **无从开口**"
              f"（不许拿别的坐标的比值冒充这一格）")
        return {"sibling": sib.name, "n_cells": 0, "why": "前一档那本不在盘上"}
    sb = json.loads(sib.read_text(encoding="utf-8")).get("paired_diff", {})
    hw = lambda s: (s["ci95_imgs"][1] - s["ci95_imgs"][0]) / 2.0
    ratios = {f"α={a:g}": round(hw(diffs[a]["paired"]) / hw(sb[f"α={a:g}"]["paired"]), 3)
              for a in RUNGS
              if sb.get(f"α={a:g}", {}).get("paired", {}).get("ci95_imgs")}
    if not ratios:
        print(f"\n[事后半宽比·本发的 d] {sib.name} 在盘上，可它的 paired_diff 与本发五格"
              f"（{[f'α={a:g}' for a in RUNGS]}）一格都对不上 ⇒ **无从开口**"
              "（宁可不印中位数，也不许拿空表算出一个数）")
        return {"sibling": sib.name, "n_cells": 0,
                "why": "剂量格不是同一套：两本账没有共通的格可比"}
    vals = sorted(ratios.values())
    ge1 = sum(1 for v in vals if v >= 1.0)
    exp = (round(T95[nseeds] / T95[prev] * (prev / nseeds) ** 0.5, 3)
           if nseeds in T95 and prev in T95 else None)
    print(f"\n[事后半宽比·本发的 d] {prev}→{nseeds} 条流（{sib.name} → {out.name}，"
          f"{len(ratios)}/{len(RUNGS)} 格可比）" + "、".join(f"{k}:{v}" for k, v in ratios.items()) +
          f" ⇒ 中位 {statistics.median(vals):.3f}" +
          (f" vs σ̂ 不变时预期 {exp}" if exp is not None else "（在册系数表没有这一档流数 ⇒ 预期算不出）") +
          f"；半宽 ≥ 1（**这一格反而变宽了**）{ge1}/{len(vals)} 格"
          "（第廿七发那句话的全部内容：买流数不保证买到分辨率）")
    return {"sibling": sib.name, "flow_step": f"{prev}→{nseeds}", "ratios": ratios,
            "median": round(statistics.median(vals), 3), "cells_ge_1": ge1, "n_cells": len(vals),
            "expected_if_sigma_constant": exp,
            "meaning": "本发每个等力度格那条配对差 d 的 CI 半宽 ÷ 前一档流数那本同格的半宽（< 1 = 更窄）。"
                       "与 power_preflight_ex_ante 分开：那一行是同坐标 α×T 网格的历史比值（别人的尺子），"
                       "这一行才是本发这个量自己买到的东西"}


# ---------- 读账（三本在册账；本发只新训 T=6 那一列的五个等力度格） ----------
def load_ladder(name: str, scramble: bool = False) -> dict:
    """一本阶梯账 → 每格逐流 gap / wrong_kd / 每粒种子的 CE 错图数。

    为什么把 wrong_kd 也读出来：本发的配对差以错图数为货币（CE 在同一条流内相减自动抵消），
    而 gap 只是它的派生量——两道复算（均值命中 + gap == wrong_ce − wrong_kd）都过才敢说这两个词指同一批数。
    """
    d = json.loads((HERE / name).read_text(encoding="utf-8"))
    seeds = [int(s) for s in d["seeds"]]
    arch = str(d.get("student_arch", "mlp"))
    cells, wrongs, armgap = {}, {}, {}
    for a, v in d["ladder_summary"].items():
        cells[float(a)] = dict(zip(seeds, [int(g) for g in v["gaps_imgs"]]))
    for k, row in d["arms"].items():
        af = float(k.split(" 种子")[0][1:])
        wrongs.setdefault(af, {})[int(row["seed"])] = int(row["wrong_kd"])
        armgap.setdefault(af, {})[int(row["seed"])] = int(row["wrong_ce"]) - int(row["wrong_kd"])
    if scramble:
        cells = {a: {s: g + 1 for s, g in c.items()} for a, c in cells.items()}
    ce: dict[int, set] = {}
    for row in d["arms"].values():
        ce.setdefault(int(row["seed"]), set()).add(int(row["wrong_ce"]))
    for sd, vals in sorted(ce.items()):
        if len(vals) > 1:
            raise SystemExit(f"{name}：种子 {sd} 的 CE 错图数有 {sorted(vals)} 个取值 ⇒ 减数不唯一")
    return {"name": name, "seeds": seeds, "arch": arch, "tier": d["tier"], "width": int(d["width"]),
            "t": float(d["temps_fixed"]), "floor": int(d["seed_floor_imgs"]),
            "teacher_acc": float(d["teacher_acc"]), "probe_alpha": float(d["probe_alpha"]),
            "anchor_alpha": float(d["anchor_alpha"]), "cells": cells, "wrongs": wrongs, "armgap": armgap,
            "means": {float(a): float(v["mean_gap_imgs"]) for a, v in d["ladder_summary"].items()},
            "ce": {sd: next(iter(v)) for sd, v in ce.items()}}


def gate_recalc(b: dict) -> None:
    """闸 I：每格均值现算命中 + 每格逐流 gap == arms 里的 wrong_ce − wrong_kd。"""
    for a, want in sorted(b["means"].items()):
        got = round(sum(b["cells"][a].values()) / len(b["cells"][a]), 1)
        if got != want:
            raise SystemExit(f"{b['name']}：α={a:g} 格均值现算 {got} ≠ 档案记的 {want} ⇒ 读错账")
        for sd in b["seeds"]:
            if b["armgap"][a][sd] != b["cells"][a][sd]:
                raise SystemExit(f"{b['name']}：α={a:g}/种子{sd} 的 ladder gap {b['cells'][a][sd]} ≠ "
                                 f"arms 里的 wrong_ce−wrong_kd {b['armgap'][a][sd]}"
                                 f" ⇒ 这本账的两种记法不是同一批数，本发的货币不成立")


def gate_pairing(lo: dict, hi: dict) -> None:
    """闸 II：两本账只许差一个温度列，且五格 α 都在。"""
    if lo["seeds"] != hi["seeds"]:
        raise SystemExit("两本阶梯账的流名单不同序/不同成员 ⇒ 逐流配对没有立足点")
    same = {k: (lo[k], hi[k]) for k in ("tier", "width", "arch", "floor", "teacher_acc",
                                        "probe_alpha", "anchor_alpha") if lo[k] != hi[k]}
    if same:
        raise SystemExit(f"这两本不是同一格只换温度列（{same}）⇒ 判别变量没对齐，差值不属于任何假说")
    if lo["t"] == hi["t"]:
        raise SystemExit(f"两本账温度列都是 T={lo['t']:g} ⇒ 跨列差恒为 0")
    for a in RUNGS:
        if a not in lo["cells"] or a not in hi["cells"]:
            raise SystemExit(f"α={a:g} 不是两本账都有 ⇒ 那一格的等力度配对没有原料")


def gate_ce_identity(lo: dict, hi: dict, teacher_acc: float) -> dict:
    """闸 III：老师身份 + 两列 CE 减数逐位相同。"""
    want = {lo["teacher_acc"], hi["teacher_acc"]}
    if len(want) != 1:
        raise SystemExit(f"两本账自报 teacher_acc 不一致（{want}）⇒ 不是同一副老师的账")
    got = round(teacher_acc, 4)
    if got != lo["teacher_acc"] or got != round(TEACHER_ARCHIVE_ACC, 4):
        raise SystemExit(f"老师现算测试准确率 {got:.4f} ≠ 两本账记的 {lo['teacher_acc']} / "
                         f"板块常量 {TEACHER_ARCHIVE_ACC} ⇒ 本课引用的账与盘上这副权重不是同一次实验")
    bad = [s for s in lo["seeds"] if lo["ce"].get(s) != hi["ce"].get(s)]
    if bad:
        raise SystemExit(f"CE 臂在流 {bad} 上两列不等 ⇒ 跨列差里掺着 CE 自己的漂移")
    return {"teacher_acc_recomputed": got, "teacher_acc_archives": lo["teacher_acc"],
            "ce_minus_number_checked": f"{len(lo['seeds'])}/{len(lo['seeds'])}"}


def gate_strength_map(g: dict, v4c: dict, v4i: dict, probeblind: bool) -> dict:
    """闸 IV：等力度映射的原料必须命中两本在册账，且顶层配平 α 复现在册那个数。

    `probeblind` 是它的拆红档：不许 backward、改用教科书假设「T² 补偿精确 ⇒ ‖∇‖ 与 T 无关」，
    于是 α' 恒等于 α、权重通道"按声称"整条消失 ⇒ 这一档红钉的是
    **"映射必须由一次实测的梯度算出，不能由 T² 推定"**（第4课i 的 --t2claim 的同族，
    区别是这里它直接决定五个剂量本身）。"""
    for t in (T_LO, T_HI):
        a = v4c["gradient_probe"]["by_T"][f"{t}"]
        if a["g_ce"] != v4i["probe_by_T"][f"{t}"]["batch0"]["g_ce"] \
                or a["g_kd_unit"] != v4i["probe_by_T"][f"{t}"]["batch0"]["g_kd_unit"]:
            raise SystemExit(f"两本基准账自己的探针就不一致（{V4C} vs {V4I}，T={t:g}）⇒ 谁都不能当尺子")
        if g[t]["g_ce"] != a["g_ce"] or g[t]["g_kd_unit"] != a["g_kd_unit"]:
            raise SystemExit(f"本脚本现算的第 0 批探针（T={t:g}）= {g[t]} ≠ {V4C}/{V4I} 记的 "
                             f"‖∇CE‖{a['g_ce']} / ‖∇(T²KL)‖{a['g_kd_unit']}"
                             f"{'（本档正在演示：用教科书 T² 推定代替一次 backward）' if probeblind else ''}"
                             f" ⇒ 这把尺子与产出那两本账的尺子不是同一把，五个剂量全不发")
    return {t: v4c["gradient_probe"]["by_T"][f"{t}"]["g_kd_unit"] for t in (T_LO, T_HI)}


def alpha_map(g3: float, g6: float) -> dict:
    """α'(α) = α · ‖∇(T²KL)‖_3 / ‖∇(T²KL)‖_6；α 是 loss 里 KL 梯度的纯标量前置，所以这是闭式而非近似。"""
    return {a: a * g3 / g6 for a in RUNGS}


def main(nseeds: int = 8, probeblind: bool = False, scramble: bool = False) -> None:
    mode = "probeblind" if probeblind else "scramble" if scramble else ""
    if probeblind:
        print("[验红] --probeblind：不许 backward，按教科书假设「T² 补偿精确」⇒ α'=α，只有闸 IV 该红")
    if scramble:
        print("[验红] --scramble：给 T=6 那本账每格 gap +1，只有闸 I 该红")
    if nseeds not in FLOW_SEEDS:
        raise SystemExit(f"复现流只有 {sorted(FLOW_SEEDS)} 这几档")
    seeds = FLOW_SEEDS[nseeds]
    check_flow_constants(len(seeds))     # 缺系数的 KeyError 会炸在 46 次训练之后（第4课n=12 那次的教训）
    out = out_name(nseeds)
    if out.exists() and not mode:
        raise SystemExit(f"{out.name} 已入库——重跑会盖掉这本判决档案（要重跑得先改名或删档）")

    L3, L6 = ladder_names(nseeds)
    # 开口必打（第廿八发把第廿七发那条否定读数固化成规矩）：再买流数之前先印"补流数买不买得到分辨率"。
    # 只在**登记了前一档流数**的坐标上打（现读 SIBLING：16→8）；8 条流那本不发这一行，
    # 否则 v2 会比 v1 多一个顶层键，"除 pays/archives 外逐位相同"那条对账线就作废了。
    pp = (power_preflight(WIDTH, TIER, where=f"step4j 买流数（{nseeds} 条流）", arch=ARCH)
          if nseeds in SIBLING else {})
    lo, hi = load_ladder(L3), load_ladder(L6, scramble=scramble)
    gate_recalc(lo)                                              # 闸 I
    gate_recalc(hi)
    gate_pairing(lo, hi)                                         # 闸 II
    v4c = json.loads((HERE / V4C).read_text(encoding="utf-8"))
    v4i = json.loads((HERE / V4I).read_text(encoding="utf-8"))
    debt = debt_index(v4i, V4I)         # 闸 VI：债句按内容命中，序号现读（在第一次训练之前）

    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    t_acc = float((predict(teacher, xte, False) == yte).mean())
    ident = gate_ce_identity(lo, hi, t_acc)                       # 闸 III

    xd, yd = xt_all[:DATA_TIERS[TIER]], torch.tensor(yt_all[:DATA_TIERS[TIER]])
    soft = logits_of(teacher, xd)
    g = {t: term_grad_norms(xd, yd, soft, t, WIDTH, arch=ARCH) for t in (T_LO, T_HI)}
    if probeblind:
        g[T_HI] = {"g_ce": g[T_HI]["g_ce"], "g_kd_unit": g[T_LO]["g_kd_unit"]}
    smap = gate_strength_map(g, v4c, v4i, probeblind)         # 闸 IV（返回按温度索引的字典）
    g3, g6 = smap[T_LO], smap[T_HI]
    AM = alpha_map(g3, g6)
    top_booked = v4c["gradient_probe"]["by_T"][f"{T_HI}"]["alpha_to_match_anchor"]
    if round(AM[RUNGS[-1]], 3) != top_booked:
        raise SystemExit(f"由现算梯度重算的阶梯顶等力度 α 得 {AM[RUNGS[-1]]:.3f}，"
                         f"而 {V4C} 在册的 alpha_to_match_anchor（T={T_HI:g}）是 {top_booked}"
                         f" ⇒ 本发的映射式与第4课c 的配平臂不是同一个算式，五个剂量全不发")

    print(f"[{TIER} × S{WIDTH} × {ARCH} 族] 五格 × {len(seeds)} 条流 @ T={T_HI:g}（等力度）| "
          f"配对另一头读 {L3}（T={T_LO:g}）")
    print(f"[闸 I] 两本阶梯 {len(lo['cells'])}+{len(hi['cells'])} 格均值现算命中、"
          f"gap 与 arms 的 wrong_ce−wrong_kd 逐格逐流相同")
    print(f"[闸 II] 两本账 {len(seeds)} 条流同序、判别变量全同、只温度列不同")
    print(f"[闸 III] 老师身份：现算 {ident['teacher_acc_recomputed']} = 两本账自报 "
          f"{ident['teacher_acc_archives']} = 板块常量 {TEACHER_ARCHIVE_ACC}；"
          f"CE 减数逐位核过 {ident['ce_minus_number_checked']} 条")
    print(f"[闸 IV] 第 0 批探针逐位命中 {V4C} 与 {V4I}：‖∇(T²KL)‖ T=3 {g3:.5f} / T=6 {g6:.5f}"
          f"（‖∇CE‖ {g[T_LO]['g_ce']:.5f}，与 T 无关）⇒ 等力度比 g3/g6 = {g3 / g6:.4f}")
    print(f"[闸 VI] 债句指针按内容命中：『{DEBT_ANCHOR}』在 {V4I} 的 limitations 里现读是第 {debt} 条"
          f"（序号是算出来的、没写死：那一列每升一版就在头部多一条变更说明 ⇒ 写死的序号会静默漂走）")
    print(f"[等力度映射] α'（T=6 上与本发各格同力度）= " +
          "、".join(f"α={a:g}→{AM[a]:.4f}" for a in RUNGS) +
          f"；阶梯顶那一格复现在册配平数 {top_booked}（第4课c 的 match_T6 臂用的就是它）")

    # ---------- 先立预言（此后不许改口） ----------
    print("\n[先立预言（此后不许改口）]")
    print(f"  H_力度（第4课i 的 18% 那句推到极端：跨列差全由 α·‖∇‖ 解释）→ 同力度五格 d≈0，"
          f"CI 应含 0；出口 (a)")
    print(f"  H_内容（第4课i 实际走的出口 (b)：形状本身带伤害）→ 主格 d<0 且 CI 不含 0，"
          f"量级参考：未配平的跨列差是 −28.1 张，权重折算只吃了 −5.1 张 ⇒ 剩下约 −23 张开外")
    print(f"  反向读数（同力度下 T=6 更浅）不是 H_内容 也不是 H_力度：那是探针把'有效剂量'估高了，出口 (c)")

    # ---------- 各臂 ----------
    rows = {a: [] for a in RUNGS}
    gates = []
    print(f"\n[开跑] 每条形如 (T=6, α'=等力度剂量) 一副；CE 锚不重训（配对里自动抵消），"
          f"接线由闸 V 的 {2 * len(SEEDS)} 次逐位对账承担：")
    for seed in seeds:
        line = []
        for a in RUNGS:
            ap = AM[a]
            m, _ = train_student(WIDTH, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=ap, t=T_HI, arch=ARCH)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            rows[a].append({"seed": seed, "alpha_matched": round(ap, 5), "T": T_HI,
                            "wrong_kd": wrong, "wrong_kd_T3": lo["wrongs"][a][seed],
                            "d_imgs": lo["wrongs"][a][seed] - wrong})
            line.append(f"a{a:g}(→{ap:.3f}):{wrong}")
        print(f"  种子 {seed:<5} " + " ".join(line))
        if seed in SEEDS:                       # 只有老 3 条有跨脚本账可核
            for t, ref, book in ((T_LO, lo, L3), (T_HI, hi, L6)):
                m, _ = train_student(WIDTH, xd, yd, soft, "KD", xv, yv, seed=seed,
                                     alpha=GATE_ALPHA, t=t, arch=ARCH)
                p = predict(m, xte, False)
                wrong = int((p != yte).sum())
                want = ref["wrongs"][GATE_ALPHA][seed]
                gates.append({"T": t, "alpha": GATE_ALPHA, "seed": seed, "archive": book,
                              "this_run_wrong_kd": wrong, "archive_wrong_kd": want,
                              "match": bool(wrong == want)})
    bad_gate = [x for x in gates if not x["match"]]
    if not gates:
        raise SystemExit(f"闸 V 一条记录都没有（老 3 条流 × 两列 = {2 * len(SEEDS)} 次）：空表『全过』不是通过")
    print(f"\n[闸 V 跨脚本逐位 (α={GATE_ALPHA:g})] {len(gates) - len(bad_gate)}/{len(gates)} 命中 "
          f"T=3 读 {L3}、T=6 读 {L6}"
          + ("" if not bad_gate else f" ⇒ **不通过：{bad_gate}**——本脚本与那两本账不是同一台训练器，"
                                     f"一头新训一头读档的配对差全部作废"))
    if bad_gate or mode:
        raise SystemExit(f"[{'验红模式 ' + mode if mode else '闸 V 红'}] 走到了这里就不发读数、不入库")

    # ---------- 判决 ----------
    diffs = {}
    print(f"\n[同力度跨温度·逐流配对差 d = 错图数(T3@α) − 错图数(T6@α')] 负 = 同力度下 T=6 多错")
    for a in RUNGS:
        d = [r["d_imgs"] for r in rows[a]]
        s = summarize(d, f"α={a:g} 等力度配对差（T3 − T6）")
        crosses = bool(s["ci95_imgs"][0] < 0 < s["ci95_imgs"][1])
        diffs[a] = {"alpha_matched_T6": round(AM[a], 5), "paired": s, "ci_crosses_zero": crosses}
        print(f"  α={a:<5}（α'={AM[a]:.4f}） d {d} 均值 {s['mean_gap_imgs']:+6.1f} 张 "
              f"CI [{s['ci95_imgs'][0]:+.1f}, {s['ci95_imgs'][1]:+.1f}] σ {s['sd_imgs']} "
              f"同号 {s['same_sign_seeds']} p={s['p_seeds_binomial']} ⇒ "
              + ("CI 含 0：这一格同力度下两列没分开" if crosses else
                 ("CI 不含 0 且 d<0：同力度下 T=6 仍多错" if s["mean_gap_imgs"] < 0 else
                  "CI 不含 0 且 **d>0**：同力度下 T=6 反而更浅 ⇒ 出口 (c)，探针被高估")))
    # 未配平那一头（零成本，从两本在册账现读）：给出"配平这一步换掉了多少"的对照
    # 货币与上面的 d 一致：gap(T6) − gap(T3) = wrong_kd(T3) − wrong_kd(T6)，负 = T=6 多错
    unbal = {}
    for a in RUNGS:
        s = summarize([hi["cells"][a][sd] - lo["cells"][a][sd] for sd in seeds], f"α={a:g} 未配平跨列差")
        unbal[a] = s
        print(f"  对照·未配平（固定 α、只换 T）α={a:<5} 差 {s['mean_gap_imgs']:+6.1f} 张 "
              f"CI [{s['ci95_imgs'][0]:+.1f}, {s['ci95_imgs'][1]:+.1f}] 同号 {s['same_sign_seeds']}")

    # ---------- 事后半宽比（第廿九发 C）：本发这个量自己买到了几分之几的分辨率 ----------
    # 上面那行 ex-ante 是**别人的尺子**（同坐标 α×T 网格的历史比值），这一行是**本发的 d**：
    # 两把尺子分开印，才许说"这一发买到的分辨率是预期的几分之几"。合起来印就等于用结论证结论。
    post_rec = sibling_halfwidth(out, nseeds, diffs)

    prim = diffs[PRIMARY]
    nonnull = [a for a, v in diffs.items() if not v["ci_crosses_zero"]]
    neg = [a for a in nonnull if diffs[a]["paired"]["mean_gap_imgs"] < 0]
    pos = [a for a in nonnull if diffs[a]["paired"]["mean_gap_imgs"] > 0]
    if pos:
        exit_rule = (f"(c) 探针被高估：{pos} 这些格同力度下 T=6 反而更浅（CI 不含 0）。"
                     f"⇒ 第 0 批那个梯度范数不是 600 步训练的有效剂量（第4课c 自我审计④ 那条"
                     f"'起手配平不是全程配平'兑现成读数），本发的归因作废方向是**探针**，"
                     f"不许读成『内容通道有益』，也不许拿它否掉第4课i 的出口 (b)")
    elif prim["ci_crosses_zero"] and not nonnull:
        exit_rule = (f"(a) 五格全部 CI 含 0 ⇒ 力度解释吃下全部跨列差，第4课i 的出口 (b)"
                     f"（大头在内容通道）被训练侧**否证**。这是**事故**不是发现：矛盾点在"
                     f"「初始一步、第 0 批的 ‖∇‖」与「600 步之后的结果」之间，"
                     f"要改的是探针那句『权重通道吃 18%』的换算，不是这一发的数")
    elif not prim["ci_crosses_zero"] and PRIMARY in neg:
        p = prim["paired"]
        exit_rule = (f"(b) 内容通道第一次有张数：把 KD 项力度钉成同一个数之后，主格 α={PRIMARY:g}"
                     f"（T=6 那一头是 α'={prim['alpha_matched_T6']:.4f}）T=6 仍多错 "
                     f"{abs(p['mean_gap_imgs']):.1f} 张（CI [{p['ci95_imgs'][0]:+.1f},"
                     f" {p['ci95_imgs'][1]:+.1f}]、同号 {p['same_sign_seeds']}、二项 p={p['p_seeds_binomial']}）"
                     f"⇒ 第4课i 那句『排除法给的归因』从此有了一次**直接测量**；"
                     f"承重的格数 {len(neg)}/{len(RUNGS)}（其余格见上表，不合并成一个数）")
    else:
        exit_rule = (f"(d) 混合读数：主格 CI 含 0 而 {nonnull} 不含 ⇒ 『匹配强度』这个概念在剂量上不守恒。"
                     f"本发的三分支里没有『部分承重』这一格位置 ⇒ 不判哪个通道，"
                     f"登记成下一版判据的债（描述性读数不进判词是老规矩）")
    print(f"\n[出口（预登记）] {exit_rule}")

    report = {"quantity": "同力度（α·‖∇(T²KL)‖ 相等）下 T=6 列与 T=3 列的逐流配对错图数之差，"
                          "五格剂量各一条（负 = 同力度下 T=6 多错）",
              "pays": f"{V4I} 的 limitations 第 {debt} 条（『把它变成测量需要一发训练…40 次训练』；"
                      f"序号按内容锚『{DEBT_ANCHOR}』现读，不写死）",
              "tier": TIER, "width": WIDTH, "student_arch": ARCH, "seeds": list(seeds),
              "temps": [T_LO, T_HI], "rungs": list(RUNGS), "primary_rung": PRIMARY,
              "strength_map": {"g_kd_unit_T3": g3, "g_kd_unit_T6": g6,
                               "ratio_g3_over_g6": round(g3 / g6, 5),
                               "alpha_matched": {str(a): round(AM[a], 5) for a in RUNGS},
                               "parity_against_v4c_anchor": top_booked},
              "archives": {"T3_ladder": L3, "T6_ladder": L6, "probe_baseline": V4C, "debt_source": V4I},
              "gates": {**ident, "recalc_cells": f"{len(lo['cells'])}+{len(hi['cells'])}",
                        "pairing": f"同 {len(seeds)} 条流同序、判别变量全同、只温度列不同"
                                   f"（T=3 那头 {L3} / T=6 那头 {L6}）",
                        "probe_batch0_bitwise": True,
                        "cross_script": {"n": len(gates), "all_match": True, "detail": gates}},
              "predictions_ex_ante": {"H_strength_d_is_zero": "五格 CI 全含 0",
                                      "H_content_d_negative": "主格 d<0 且 CI 不含 0",
                                      "over_matched_probe_d_positive": "任一格 d>0 且 CI 不含 0"},
              "matched_ladder": {f"α={a:g}": rows[a] for a in RUNGS},
              "paired_diff": {f"α={a:g}": diffs[a] for a in RUNGS},
              "unmatched_cross_column_diff_read_from_books": {f"α={a:g}": unbal[a] for a in RUNGS},
              **({"power_preflight_ex_ante": pp or
                  {"status": "无从开口",
                   "why": f"坐标 {ARCH} × {TIER} × S{WIDTH} 的 α×T 网格账在盘上凑不出相邻流数的两本"
                          "（现读那批同族账的流数版本不足相邻两档）"
                          "⇒ 开口预检这一格是**否定读数**，不是没跑"}}
                 if nseeds in SIBLING else {}),
              **({"paired_hw_ratio_vs_prev_flow": post_rec} if post_rec else {}),
              "exit": exit_rule, "verdict": exit_rule,
              "cost": {"arm_trainings": len(RUNGS) * len(seeds),
                       "cross_script_gate_trainings": len(gates),
                       "ce_trainings": 0,
                       "total": len(RUNGS) * len(seeds) + len(gates),
                       "api_fee": 0,
                       "note": "CE 锚不重训：配对差 d = wrong_kd(T3) − wrong_kd(T6)，同一条流内 CE 抵消"
                               "（与第4课c ③b 那条'CE 锚自动抵消'同一种读法）"},
              "limitations": [
                  f"配平是**起手配平**，不是全程配平：‖∇(T²KL)‖ 在第 0 批（{WIDTH} 张）、初始那一步测，"
                  "训练 600 步里两项的相对尺度会随学生逼近老师而变（与第4课c 自我审计④ 同一条）。"
                  "所以『同力度』指的是初始那一步的系数对齐，不是一个被全程守住的量——"
                  "出口 (c) 那一支正是这条边界的兑现方式，本发不假装它不存在",
                  (f"这一发买的是**覆盖**（{len(RUNGS)} 个从未训过的 α 坐标 × 同一批 {len(seeds)} 条流）："
                   "七维独立性一维都没换、流数没动 ⇒ 出口 (b) 只写成『第一次直接读数』，"
                   "不许写成『内容通道被复证』") if nseeds == 8 else
                  (f"这一发动的**只有流数**这一维（{len(seeds)} 条流 vs 在册 {SIBLING.get(nseeds, 8)} 条）："
                   f"坐标（{TIER} × S{WIDTH} × {ARCH} 族）、温度对、五格剂量、等力度映射、判据、货币全部照旧"
                   f"⇒ 读数只能写成『同一坐标上一条更窄（或更宽）的 CI』，"
                   "不许写成『内容通道被复证』，也不许与 8 条流那本合并报一个数"
                   "（两本共用同一段老流，第廿七发闸 VI 证过嵌套性，但嵌套≠独立）"),
                  "五格**不许合并成一个数**：α 曲线非单调（第4课i 自我审计那句），"
                  "所以没有一个"
                  "『平均同力度残余』是合法的机制量；承重的是主格那一格，其余四格是它的对照组",
                  "跨列差的换算不许做减法分解：|d| 与第4课i 的 share=−5.1 张量的是不同的东西"
                  "（一个是训练结果之差、一个是初始斜率×Δα），本发只把它们并排放，不相减",
                  f"读数只属于**一个坐标**（{TIER} × S{WIDTH} × {ARCH} 族 × 那 {len(seeds)} 条流）："
                  "满数据档/低数据档、S64、Fashion、CNN 学生各自要各自的等力度映射（g3/g6 是按坐标现算的，"
                  "第4课c 在 CNN 族上读到的是另一组数），本发一个字都没碰",
                  "判据版本不涉及：本发不进形状判据 v5（它判的是 α 阶梯的形状），"
                  "这一发的量是同力度跨温度的配对差 ⇒ 不改任何在册判词，也不进守卫的翻转集合参评对象",
                  f"多重比较：{len(RUNGS)} 格 × {len(seeds)} 条流，主格是**跑之前**点名的（α=0.05），"
                  "其余四格按对照读，不许事后从五格里挑最好看的那格当出口"],
              }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{report['cost']['total']} 次训练 = "
          f"靶子 {report['cost']['arm_trainings']} + 闸 V {report['cost']['cross_script_gate_trainings']}，"
          f"10000 张全量测试，纯 CPU，零 API 费）· {exit_rule[:60]}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="把 α·‖∇(T²KL)‖ 钉成常数、只换温度：等力度阶梯配平实验")
    ap.add_argument("--nseeds", type=int, choices=(3, 8, 16), default=8,
                    help="3 = 预演/对账那批老流（借 8 条流那两本阶梯走接线，账验完即删、不入库）；"
                         "8 = 板块共用那批（真跑，第廿八发已入库）；"
                         "16 = 买流数那一发（第廿九发 C：开口先打 power_preflight、事后打本发 d 的半宽比）")
    ap.add_argument("--probeblind", action="store_true",
                    help="拆红：不许 backward，改用教科书假设「T² 补偿精确 ⇒ α'=α」⇒ 只有闸 IV 该红")
    ap.add_argument("--scramble", action="store_true",
                    help="拆红：给 T=6 那本账每格 gap +1 ⇒ 只有闸 I 该红")
    a = ap.parse_args()
    main(nseeds=a.nseeds, probeblind=a.probeblind, scramble=a.scramble)
