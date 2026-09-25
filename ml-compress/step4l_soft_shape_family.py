# -*- coding: utf-8 -*-
"""第4课l（零训练）：那条"跨族偏移"机制上拆给谁——软标签**形状**还是**这份形状 × 这个学生族**。

配方不是这里发明的，是 `step4f_alpha_curve.py` 落地⑤ 写死的那一句：
    「α=0.05 那一端 CNN 减 MLP 在三档分别 +103.6 / +74.4 / +82.7 张……**它只是点估计**：
      逐流配对、CI、同号由第廿五发（`step4l`）现算。」
这一发就是那一发。两头都是在册账，一次训练都不做。

为什么这句问得出答案（第4课i 那条二分在这里换成三读法）：软标签只来自老师，学生架构进不了
老师的 softmax ⇒ **两族读到的形状是同一份**（读法 C 逐位钉这件事）。既然形状相同，两族读数之差
就不可能来自形状本身，只能来自"这份形状作用在不同学生上"——交互。而交互这句话是可检验的：
形状的锐度由 T 控制（第4课i 实测 T=3→T=6 塌缩率 23.7%→0.0%、KL=0.2050 nat），若偏移真是交互，
它应当在 T=1 那几格量不出来、在 T=6 那几格才到位。于是三读法各管一件事：

  读法 A（配对张数，8 条流）：T=6 那一列、三档、共同的 α 格 {0.05, 0.3, 0.7} —— 逐流配对偏移
        `offset_s = gap_cnn,s − gap_mlp,s`，报均值 / CI / 同号 / 二项 p。这一份是"落地⑤ 那句点估计"
        第一次有误差棒。
  读法 B（偏移 vs 温度，3 条流）：第4课c 那六本网格（三档 × 两族）里 α∈{0.3,0.7,1.0} × T∈{1,3,6}
        共 27 格，各算一次同样的配对偏移；另外按流配对算 `offset(T=6) − offset(T=1)` 这条**差之差**。
        它量的是"偏移跟不跟 T 走"，不量张数大小（3 条流与 8 条流是两把尺子，见 limitations）。
  读法 C（形状互锁，现算）：两族六本网格里 `soft_label_audit_by_T` 必须**逐位相同**，且现读老师
        （盘上那副 + 那批图 + 那个温度）复算出的体检必须等于两族在册那一份。这一条不通，本课
        一个字也不写：那说明"两族读的是同一份软标签"这个前提塌了，差异又回到"读的不是同一个东西"。

出口（**跑之前**写死在这里，事后只许追加"落地"。第廿四发落地③ 那条教训——出口集合必须穷尽——
这里靠两个计数拼成不重不漏的四格，而不是列一串"如果…就…"）：
  令 p = 读法 A 那 9 格里配对 CI **不含 0** 的格数（0~9）；
  令 q = 读法 B 的 **T=1 那一列** 9 格里 CI 不含 0 的格数（0~9，只数 T=1：交互说预言的正是"最钝
      那一列量不出来"，把 T=6 那九格也计进来会让同一发既支持又不支持自己）。
  先查读法 C：红 ⇒ 走 (d)。否则：
  (a) q == 0 且 p >= 1 ⇒ **交互成立**：T=1 处跨族差处处测不出、T=6 处测得出，而形状两族同一份
      ⇒ 那条偏移既不是"CNN 天生抗蒸馏"也不是"两族形状不同"，是**这份形状 × 这个学生族**。
  (b) q >= 1 ⇒ **交互说不成立**：偏移与软标签锐度无关 ⇒ 更可能是族本身的容量/优化差；
      此时第4课与第二十一发里那些跨族句子要按"族本征"重写，不许留"交互"字样。
  (c) q == 0 且 p == 0 ⇒ **未判定**：T=6 处那条 +82 量级的点估计连 8 条流也测不动；
      落地⑤ 那句"近似档无关的跨族偏移"就还只是点估计，本发不替它加任何方向词。
  (d) 读法 C 红 ⇒ **前提塌**，本发不产出任何跨族结论，先去查读的是不是同一副老师/同一批图。

仪器（五道闸，全在读档 + 一次老师前向，零训练）：
  闸 I   配对资格：每一对账的流名单必须**逐位同序**且等于登记处 `FLOW_SEEDS[n]`；
  闸 II  逐流恒等式（两条腿）：II-a 每一行 `gap_imgs == wrong_ce − wrong_kd`；II-b 配对用的逐流 offset
         序列必须等于"按种子号各自查两本档案、再逐位相减"的那一条独立重算序列（且之和精确等于
         两族之和之差）。**II-b 才是配对的凭据**：均值对配对顺序不变 ⇒ 把 CNN 那条流换成别条流，
         点估计、CI 宽度、甚至 II-a 全都可以一字不动。
  闸 III 配对两本必须同**档**、同**宽度**，家族位与 `student_arch` 不冲突（MLP 那几本记于该字段上线
         之前 ⇒ 家族只由文件名的 `_cnn` 位定，"没记这一位"这件事现读打印，不静默补默认）；
  闸 IV  形状互锁 = 读法 C（两族逐位相同 + 与现算逐位相同 + 老师精度等于档案常数）；
  闸 V   来源自证：打印的每一个数都带 (档案名, 格名, 流名单)，`summarize` 用的就是登记处那 8/3 条流。

拆红两档（各钉一道闸，且互相要求对方**不**红 ⇒ 两腿分离）：
  --tierblind  把低数据档那本 MLP 阶梯换成中数据档那本（流名单仍逐位相同）⇒ 只有闸 III 该红、闸 I 全绿；
  --unpair     把配对里 CNN 那一侧整体错开一位（种子 42 配种子 7 的读数）⇒ 只有闸 II-b 该红，
               而 II-a、闸 I、闸 III 全绿（这正是"均值不变、配对已坏"那个形状）。

落地（跑完之后追加，上面预登记原文一字未改；读数全部来自 `compress_report_v4l_soft_shape_family.json`）：
  ① 实付：`cost` 三项全 **0**（arm_trainings 0 / total 0 / api_fee 0）——纯读 12 本在册账 + 一次老师前向。
     墙钟不给秒数（老规矩：只留读数不留倍数，而这一档的读数本来就落在档案字段里）。
  ② 读法 A（T=6 那一列 × 8 条流，三档 × α∈{0.05,0.3,0.7} = 9 格逐流配对）：**p = 9/9**。九格 CI 全不含 0，
     且**每格同号 8/8（+）**、二项 p=0.0078。点估计（张）：
       低 +103.6 [78.1,129.2] / +75.5 [47.4,103.6] / +63.0 [38.9,87.1]
       中  +74.4 [46.6,102.2] / +94.8 [75.1,114.4] / +96.1 [74.0,118.3]
       满  +82.6 [59.3,106.0] / +105.8 [81.9,129.6] / +110.9 [82.1,139.7]
     ⇒ 上一发那句点估计（+82.7）第一次有误差棒；按 `limitations` 第 5 条，**本发不据"三档是否同号 / 哪一档最大"
     另立判词**（"近似档无关"是不是真的档无关，8 条流的配对不回答这个问题）。
  ③ 读法 B（v4c 那 27 格 × 3 条流）：**q = 0/9** —— T=1 那一列没有一格 CI 不含 0，点估计跨 −9.0 ~ +17.3，
     其中 **4 格本身是负号**（−0.7 / −2.7 / −5.0 / −9.0）；T=6 那一列 9/9 不含 0（只作描述，不进出口计数）；
     九条按流配制的 `offset(T=6) − offset(T=1)`（每格 `t_slope_paired`）**全部不含 0、全部 3/3 同号**，
     落在 **+63.7 ~ +151.3 张**。⇒ **这条差之差才是出口 (a) 的承重**：q=0 单独什么都兑现不了（它是半宽的功劳，
     不是效应的功劳），预登记 (a) 那句把两者并列写，落地时按实测改口。
  ④ 出口 **(a)「交互成立」兑现**（`exit` 与 `verdict` 逐字同预登记）。形状互锁先通：9/9 格两族体检逐位相同、
     与现读老师（精度 **0.9798** == 档案常数）复算逐位相同 ⇒ 跨族差住在"这份形状 × 这个学生族"上。
     三笔"不许读成"随读数一起立：不是复证/翻案（七维一维都没换）、不替任何一档的**形状判词**说话、
     +82 张不是"CNN 的疗效"而是"同一份软标签在两族上的收益之差"。
  ⑤ 免费的跨脚本对账线：这一发满数据 × α=0.05 报 **+82.6**，step4f 落地⑤ / step4k 报 **+82.7**
     ⇒ 差 0.1 张，来源是双舍入（两端各自舍入再相减 vs 逐流差再取均值）。同一形状盘上先例两处
     （`_mid_n16_t6_v5` 29.2/29.3、第廿四发 29.5/29.4）⇒ **这里不报"第几次兑现"**：那一串序数有两套基数、
     并读会打架，改口记在 `step4f_alpha_curve.py` 落地⑦。**两个数都留在盘上，谁引用谁带口径。**
  ⑥ 四处"写下来的断言"被运行纠正（预登记原文不动，纠正记在这里）：
     (1) 上面第 44 行那句"`--unpair` 下点估计、**CI 宽度**、甚至 II-a 全都可以一字不动"——前半对、后半过头：
         置换保住一次矩与逐行恒等式，**不保逐流差的方差**（实测 σ̂ 33.0 → 25.6、半宽 25.6 → 21.8 跟着配对移动）。
         看得见配对坏掉的仍然是**逐流序列**（II-b），不是 CI 宽度。
     (2) 判决落盘之后，"档案已存在就不许重跑"那道护栏把两档拆红**一起拦死** ⇒ 拆红档在判决之后永远验不了红 =
         **会衰减的门禁不算门禁**；修法是把护栏只管正常模式，而不是把判决改名。
     (3) 报告里手写的计数（"II-a 153 行"、"II-b 12 个格"）与真跑的 **306 行 / 36 次** 差 2~3 倍 ⇒ 全部换成活计数。
         那份手写版没删，改名留在盘上当逐字段对账基线：`_v4l_handcount_bad_v1.json`（`grep 'II-a 153' ` 可点）。
     (4) 我预备写进 `limitations` 的那句"阶梯三对里有一对记 v4 时代判据、判词与 v5 同判（守卫闸① 复算通过）"——
         现读是**两对**（`_low_n8_thin_t6_v4`、`_n8_t6_v4` 都记 `verdict.shape_rule=="v4"`），而"在 v5 之下怎么判"
         **档案里没有那个字段、守卫也没替它复算** ⇒ 那句"复算通过"是把"没测到"写成"通过"。现在 `limitations`
         第 6 条活计 + 明写"只取原始错图数（与判据版本无关），不取判词"。
  ⑦ 两档拆红实测（原件 `_tmp_l_tierblind.log` / `_tmp_l_unpair.log`，两腿分离双向兑现）：
     `--tierblind` rc=1、红 **1** 条、全在闸 III，且闸 I 与 II-b 全绿；
     `--unpair` rc=1、红 **9** 条、全在闸 II-b，且 II-a（306 行）、闸 I、闸 III 全绿。
     两档跑完都逐字节核对过**没有落盘**（判决档案一字未动）。
  ⑧ 没买到的（下一发别把这几条当已有）：档间是否真"档无关"（见 ②）；Fashion / S64 / 第二副老师上这条偏移
     跟不跟 T 走（本课只在 MNIST × 主老师 × S128）；**T=1 那一列在 8 条流尺子下的功效**——读法 B 是 3 条流的账，
     q=0 里"效应为 0"与"3 条流测不动"分不开（要分开得把 v4c 的 T=1 列扩到 8 条流，那是训练预算的事）；
     以及本课自带那 **45 条 CI 不受守卫闸⑪ 覆盖**（⑪ 的入册键是 `paired_step_small_to_anchor`，派生账没有那位；
     本轮手工核过 45/45 中点漂移 ≤ 0.05 张，**但那不叫有闸**，见 `_red_v3.py` 头部【第二十五发】第①笔诚实账）。

════════════════════════════════════════════════════════════════════════════════════════════════
追加预登记（v2 —— 买上面 ⑧ 第 3 笔那笔欠账；写在这段的是**不变量与判据**，事后只许在下面追加"落地"）
════════════════════════════════════════════════════════════════════════════════════════════════

v1 跑完之后盘上多了三本 8 条流的 v4c 网格（`_cnn_low_n8` / `_cnn_n8` / `_low_n8`，训练预算在那一发付，
本脚本仍然一次训练都不做）。于是读法 B 的 T=1 那一列第一次能在**加宽了的同一把尺子**上重测。

对象：v1 那句 **q = 0/9** 里缠在一起的两件事——"T=1 处跨族差确实小"还是"3 条流功效不够"。

先立三条规矩（跑之前写死，事后不许改口）：
  R1 **这一段不重开出口**。出口 (a)/(b)/(c) 的定义、`exit`、`verdict` 三个字段与 v1 逐字相同；
     v2 只新增一个独立键 `section2_power_check`。谁引用 q=0，从此必须同时报 q8 与分母。
  R2 **分母不同就不比大小**。中数据档两族都没有 8 流网格（在册只有 3 流那本）⇒ 段② 只有两档，
     q8 的分母是 **2 档 × 3 个 α = 6 格**，不是 9 格。打印分母，禁止把 q8 与 q 并排写成"0/9 → 0/6"这种
     看起来像下降的数（6 格里全含 0 与 9 格里全含 0 是同一件事，格数少不代表更弱）。
  R3 8 流那九格里的 T=6 列**不并入 p**：p 的账来自 v4f 阶梯（α 格与 T 的取法都不同尺），两本不混。

前置（三条都过才允许下判词；任一不过 ⇒ 走 (u)：只报数、明写"这一段没测到"，不写"通过"）：
  甲 **嵌套性（新闸 VI）**：8 流那两本与同档 3 流那两本，共有的前 3 条流（42/7/2026）在**全部共有格**
     （含 `match_T6`，它不进读法 B）上 `wrong_ce`/`wrong_kd`/`gap_imgs` 逐位相同。
     红 ⇒ 两把尺子不是"同一批流加宽"，而是两批不同的读数 ⇒ 8-vs-3 的横比从根上污染，段② 不可用。
     三态：比对行数 = 0 也算不过（"没测到"不是"通过"）。
  乙 **形状互锁跟着扩**：闸 IV 的读档逐位相同这一腿，把 8 流那两对也纳进来（现算那一腿仍只在
     中数据档那对上做——8 流没有中数据档的账，硬凑就是换尺子）。
  丙 **半宽确实变窄**：6 格各自 `hw8/hw3 < 1`，其中 hw =（CI 上端 − 下端）/2。
     预期（σ̂ 不变时）= T95[8]/T95[3] × √(3/8) = 2.365/4.303 × 0.612 ≈ **0.34**——但"约 0.34"是**预期不是判据**，
     判据只要求 < 1；任一格 ≥ 1 ⇒ (u)。

判定（前置全过才走）：q8 = 那 6 格里配对 CI 不含 0 的格数。
  (u) 前置不过 ⇒ 只报数，本段不产出任何话。
  (i) q8 == 0 ⇒ "3 条流测不动"这一支被**削弱**（不是证伪）：尺子加宽到 8 条流、半宽按甲/丙核过的比例
      缩下来，T=1 那一列仍然处处含 0 ⇒ q=0 偏向"效应确实小"。落地必须同时报出实际缩了多少倍，
      否则"削弱"两个字没有凭据。
  (ii) q8 >= 1 ⇒ "3 条流功效不够"直接坐实（8 条流量得出的东西 3 条流量不出）⇒ v1 那句 q=0 在
      `limitations` 的口径下降级为**功效陈述**；出口 (a) 的承重整个挪到 `t_slope_paired` 那条差之差上
      （v1 落地③ 本来就已经这么写了，所以这里**不动 v1 的判决**，只是把它的承重点坐实）。

拆红加一档（新闸要有专属红，且两腿分离）：
  --nest  把 8 流 CNN 那本 `match_T6` 格里种子 42 的 `wrong_ce` 在内存里 +1（这一格不进读法 A/B，
          也不进 II-a / 闸 I / 闸 III / 闸 IV 读的任何位）⇒ 只有闸 VI 该红。
          若 II-a、闸 I、闸 III 跟着红 ⇒ 说明闸 VI 读到了配对以外的东西，两腿分离没验到。
  （v1 那两档 --tierblind / --unpair 的定义一字不改，v2 之后仍需各自复跑一次。）

落地（跑完之后追加）：
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import (DATA_TIERS, FLOW_SEEDS, T95, TEACHER_ARCHIVE_ACC, logits_of,
                           soft_label_audit, summarize)
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
# v2：只多段② 那几个键（`exit`/`verdict`/读法 A/B 逐字段与 v1 相同）；v1 档案留盘做逐字段对账
OUT = HERE / "compress_report_v4l_soft_shape_family_v2.json"

ALPHAS = (0.05, 0.3, 0.7)                 # 三对阶梯账**共有**的 α 格（0.1/0.2 只有两档有 ⇒ 不顺手扩）
GRID_ALPHAS = (0.3, 0.7, 1.0)
TEMPS = (1.0, 3.0, 6.0)
AUDIT_TIER = "中数据4500"                  # 现算体检只在这一档做（与第4课i 同一坐标，不另起炉灶）

LADDER = {                                 # 读法 A：T=6 那一列 × 8 条流 × S128，两族各一本
    "低数据2000": ("compress_report_v4f_cnn_low_n8_thin_t6_v5.json",
                   "compress_report_v4f_low_n8_thin_t6_v4.json"),
    "中数据4500": ("compress_report_v4f_cnn_mid_n8_t6_v5.json",
                   "compress_report_v4f_mid_n8_t6_v5.json"),
    "满数据8500": ("compress_report_v4f_cnn_n8_thin_t6_v5.json",
                   "compress_report_v4f_n8_t6_v4.json"),
}
GRID = {                                   # 读法 B：第4课c 的 α×T 网格 × 3 条流 × S128
    "低数据2000": ("compress_report_v4c_cnn_low.json", "compress_report_v4c_low.json"),
    "中数据4500": ("compress_report_v4c_cnn_mid.json", "compress_report_v4c_mid.json"),
    "满数据8500": ("compress_report_v4c_cnn.json", "compress_report_v4c.json"),
}
# 段②：同一把 α×T 网格搬到 8 条流的尺子上。**只有两档**——中数据档两族都没有 8 流网格（在册只有 3 流那本），
# 硬凑进来就是拿别的尺子的账冒充同尺，所以 q8 的分母是 2×3=6 格，不是读法 B 那 9 格。
GRID8 = {
    "低数据2000": ("compress_report_v4c_cnn_low_n8.json", "compress_report_v4c_low_n8.json"),
    "满数据8500": ("compress_report_v4c_cnn_n8.json", "compress_report_v4c_n8.json"),
}
AUDIT_KEYS = ("mean_top1", "pct_top1_gt_0.99", "mean_p2", "mean_margin_top1_top2")

red3: list[str] = []                       # 闸 III 的失败（打印 + 计数，最后统一裁决）
red2b: list[str] = []                      # 闸 II-b 的失败（同上）
red6: list[str] = []                       # 闸 VI（共流嵌套性）的失败（段② 的前置·甲）
stats = {"gap_rows": 0, "blocks": 0, "nest_rows": 0}   # 各闸自己的**实跑**次数：报告里的计数不许手写


def read(name: str) -> dict:
    p = HERE / name
    if not p.exists():
        raise SystemExit(f"盘上没有 {name} —— 本课全程只读在册账，缺档就是配方写错了，不去重跑")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["_name"] = name
    return d


def check_family(book: dict, want: str) -> None:
    """家族位：只由文件名的 `_cnn` 定；`student_arch` 若记了就必须与它一致，没记要**打印**出来。

    这是守卫闸⑨ 那处教训的正面用法：可缺省字段读成默认值 = 一批账安静地免检。"""
    booked = book.get("student_arch")
    if booked is not None and booked != want:
        raise SystemExit(f"闸 III {book['_name']}：文件名的家族位说是 {want} 族，`student_arch` 记的是 "
                         f"{booked} ⇒ 名字与臂不是同一族，这一对不能配对")


def arm_row(book: dict, alpha: float, seed: int) -> dict:
    key = f"α{alpha:g} 种子{seed}"
    if key not in book["arms"]:
        raise SystemExit(f"闸 III {book['_name']} 里没有 {key} 这一格（有 "
                         f"{'、'.join(sorted({k.split(' 种子')[0] for k in book['arms']}))}）")
    return book["arms"][key]


def grid_row(book: dict, alpha: float, temp: float, seed: int) -> dict:
    key = f"a{alpha:g}_T{temp:g}"
    cell = book["grid"].get(key)
    if cell is None:
        raise SystemExit(f"闸 III {book['_name']} 的网格里没有 {key} 这一格")
    for r in cell["per_seed"]:
        if int(r["seed"]) == seed:
            return r
    raise SystemExit(f"闸 III {book['_name']} 的 {key} 格里没有种子{seed} 那一行")


def gap_of(row: dict, book: dict, cell: str) -> int:
    """gap = wrong_ce − wrong_kd（正 = KD 比 CE 少错 = 收益）。闸 II-a 在这里把关。"""
    g = int(row["gap_imgs"])
    stats["gap_rows"] += 1
    if g != int(row["wrong_ce"]) - int(row["wrong_kd"]):
        raise SystemExit(f"闸 II-a {book['_name']} 的 {cell}：gap_imgs={g} ≠ wrong_ce−wrong_kd="
                         f"{int(row['wrong_ce']) - int(row['wrong_kd'])} ⇒ 这一行的符号口径不是本课假设的那个")
    return g


def paired(offsets: list[int], label: str) -> dict:
    st = summarize(offsets, label)
    lo, hi = st["ci95_imgs"]
    st["ci_excludes_zero"] = not (lo <= 0 <= hi)
    return st


def half_width(st: dict) -> float:
    """CI 半宽 =（上端 − 下端）/2 —— 段② 前置·丙 比的就是它（'加宽尺子到底有没有换来更窄的区间'）。"""
    lo, hi = st["ci95_imgs"]
    return (hi - lo) / 2.0


def col_hits(read_b: dict, suffix: str) -> int:
    return sum(1 for blk in read_b.values() for k, st in blk["offset_by_cell"].items()
               if k.endswith(suffix) and st["ci_excludes_zero"])


def nest_check(b8: dict, b3: dict, tag: str) -> None:
    """闸 VI（段② 前置·甲）：8 流那本与**同档** 3 流那本，共有的那几条流必须逐位复读同一批数。

    红 ⇒ 两把尺子不是"同一批流加宽"，而是两批不同的读数 ⇒ 段② 那句"8 流仍然测不出"从根上污染。
    `match_T6` 这类不进读法 B 的格也在这里比：闸 VI 要的是'同一条流同一起点'这件事本身。"""
    m3 = {c: {int(r["seed"]): r for r in cell["per_seed"]} for c, cell in b3["grid"].items()}
    for cell, rows in b8["grid"].items():
        shared = m3.get(cell)
        if shared is None:
            continue
        for r in rows["per_seed"]:
            s = int(r["seed"])
            if s not in shared:
                continue
            stats["nest_rows"] += 1
            for k in ("wrong_ce", "wrong_kd", "gap_imgs"):
                if int(r[k]) != int(shared[s][k]):
                    red6.append(f"闸 VI {tag}：{cell} 种子{s} 的 {k} 在 8 流账里记 {r[k]}、同档 3 流账里记 "
                                f"{shared[s][k]} ⇒ 同一条流在两本里读出了不同的数，两把尺子不嵌套")


def pair_block(cb: dict, mb: dict, seeds: list[int], getter, cellname: str, tag: str, unpair: bool):
    """两族按**种子号**各查各的、逐位相减；II-b 的那条独立重算序列同时产出。"""
    off, gc, gm, by_seed = [], [], [], {}
    stats["blocks"] += 1
    for i, s in enumerate(seeds):
        rc, rm = getter(cb, s), getter(mb, s)
        c = gap_of(rc, cb, cellname(s))
        m = gap_of(rm, mb, cellname(s))
        j = (i + 1) % len(seeds) if unpair else i              # --unpair：CNN 那一侧整体错开一位
        shifted = gap_of(getter(cb, seeds[j]), cb, cellname(seeds[j])) if unpair else c
        off.append(shifted - m)
        gc.append(c)
        gm.append(m)
        by_seed[str(s)] = c - m
    independent = [by_seed[str(s)] for s in seeds]
    if off != independent:
        red2b.append(f"闸 II-b {tag}：配对序列与『按种子号各自查档再逐位相减』的重算序列不等"
                     f"（{off} vs {independent}）⇒ 配的对不是这些流，点估计与 CI 全部作废")
    if sum(off) != sum(gc) - sum(gm):
        red2b.append(f"闸 II-b {tag}：配对偏移之和 {sum(off)} ≠ 两族之和之差 {sum(gc) - sum(gm)}")
    return off, gc, gm, by_seed


def read_B(pairs: dict) -> dict:
    """读法 B 的机器：{档: (CNN 本, MLP 本)} → 逐格配对偏移 + 按流配制的 `offset(T6)−offset(T1)`。

    v1 只喂 3 流那六本；v2 用**同一个函数**再喂 8 流那四本 ⇒ 两把尺子的横比不是"两套代码的两个数"。"""
    read_b: dict[str, dict] = {}
    for tier, (cb, mb) in pairs.items():
        seeds = [int(s) for s in cb["seeds"]]
        per_cell, slope = {}, {}
        for a in GRID_ALPHAS:
            for t in TEMPS:
                off, gc, gm, by_seed = pair_block(
                    cb, mb, seeds, lambda b, s: grid_row(b, a, t, s), lambda s: f"a{a:g}_T{t:g}",
                    f"{tier} × α={a:g} × T={t:g}", unpair=False)
                st = paired(off, f"{tier} × α={a:g} × T={t:g}：跨族配对偏移（{len(seeds)} 条流）")
                st["offset_by_seed"] = by_seed
                st["provenance"] = {"cnn": cb["_name"], "mlp": mb["_name"], "cell": f"a{a:g}_T{t:g}",
                                    "flows": seeds, "n": len(seeds)}
                per_cell[f"α={a:g},T={t:g}"] = st
            d = [x - y for x, y in zip(per_cell[f"α={a:g},T=6"]["gaps"],
                                       per_cell[f"α={a:g},T=1"]["gaps"])]
            slope[f"α={a:g}"] = paired(d, f"{tier} × α={a:g}：offset(T=6) − offset(T=1)（按流配对）")
        read_b[tier] = {"offset_by_cell": per_cell, "t_slope_paired": slope}
    return read_b


def print_B(read_b: dict, head: str, denom: int) -> tuple[int, int]:
    q = col_hits(read_b, "T=1")
    t6 = col_hits(read_b, "T=6")
    print(f"\n[读法 {head}]")
    for tier, blk in read_b.items():
        print(f"  {tier}: " + "  ".join(
            f"α{a:g}:" + "/".join(f"T{t:g}={blk['offset_by_cell'][f'α={a:g},T={t:g}']['mean_gap_imgs']:+.1f}"
                                   for t in TEMPS) for a in GRID_ALPHAS))
        for k, st in blk["t_slope_paired"].items():
            print(f"      {k}: offset(T6)−offset(T1) = {st['mean_gap_imgs']:+.1f} 张 CI{st['ci95_imgs']} "
                  f"同号 {st['same_sign_seeds']} → {'不含 0' if st['ci_excludes_zero'] else '含 0'}")
    print(f"  ⇒ 前置·乙：q（T=1 那一列 {denom} 格里 CI 不含 0）= {q}；"
          f"T=6 那一列 {denom} 格里不含 0 = {t6}（只作描述）")
    return q, t6


def main(red: str = "") -> None:
    # 快照在一切动作之前取：红档那条自查比的是"跑前 vs 跑后"，不是"盘上有没有"
    snap = OUT.read_bytes() if OUT.exists() else None
    # 这条护栏只管正常模式：拆红档从头到尾不写盘（下面还有一条自查钉它），拦它就是拦"判决之后再也验不了红"
    if OUT.exists() and not red:
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉这本跨族归因的判决（要重跑得先改名或删档）")
    if red == "--tierblind":
        LADDER["低数据2000"] = (LADDER["低数据2000"][0], LADDER["中数据4500"][1])
        print("[验红] --tierblind：低数据档那本 MLP 阶梯换成中数据档的（流名单仍逐位相同）⇒ 只有闸 III 该红")
    if red == "--unpair":
        print("[验红] --unpair：把配对里 CNN 那一侧整体错开一位 ⇒ 只有闸 II-b 该红（II-a / 闸 I / 闸 III 全绿）")

    books = {t: (read(c), read(m)) for t, (c, m) in LADDER.items()}
    grids = {t: (read(c), read(m)) for t, (c, m) in GRID.items()}
    grids8 = {t: (read(c), read(m)) for t, (c, m) in GRID8.items()}
    if red == "--nest":
        row = next(r for r in grids8["满数据8500"][0]["grid"]["match_T6"]["per_seed"]
                   if int(r["seed"]) == FLOW_SEEDS[3][0])
        row["wrong_ce"] = int(row["wrong_ce"]) + 1
        print(f"[验红] --nest：把 8 流 CNN 那本 `match_T6` 种子{row['seed']} 的 wrong_ce 在内存里 +1"
              f"（这一格不进读法 A/B，也不进 II-a 读的 9 个 α×T 格）⇒ 只有闸 VI 该红")

    # ---- 闸 III：配对两本必须同档同宽（先于取数；红只登记，最后统一裁决）----
    for tag, pairs in (("阶梯(8 流)", books), ("网格(3 流)", grids), ("网格(8 流)", grids8)):
        for tier, (cb, mb) in pairs.items():
            for b, fam in ((cb, "cnn"), (mb, "mlp")):
                check_family(b, fam)
                if int(b["width"]) != 128:
                    raise SystemExit(f"闸 III {b['_name']}：宽度 {b['width']} 不是 S128 ⇒ 与这一发的配方不同尺")
            if cb["tier"] != tier or mb["tier"] != tier:
                red3.append(f"闸 III {tag}·{tier}：两本记的档是 {cb['tier']} / {mb['tier']}，配方点名的是 "
                            f"{tier} ⇒ 拿别的档的账来配对 = 把两把不同数据量的尺子当成一对，这一格作废")
    n_books = 2 * (len(books) + len(grids) + len(grids8))
    all_pairs = (*books.values(), *grids.values(), *grids8.values())
    n_missing_field = sum(1 for ps in all_pairs for b in ps
                          if b.get("student_arch") is None)
    # 判据版本现读（缺字段的单独数出来，不读成默认值）：limitations 里那句"不许混引 v4/v5"要有来源
    rules: dict[str, int] = {}
    for ps in all_pairs:
        for b in ps:
            rv = (b.get("verdict") or {}).get("shape_rule") if isinstance(b.get("verdict"), dict) else None
            rv = rv if rv is not None else "（没记这一位）"
            rules[rv] = rules.get(rv, 0) + 1
    v4_mlp = [mb["_name"] for cb, mb in books.values()
              if isinstance(mb.get("verdict"), dict) and mb["verdict"].get("shape_rule") == "v4"]
    print(f"[闸 III] {n_books} 本全部 S128；档位与配方点名的档逐本核对（红 {len(red3)} 条），"
          f"家族位由文件名定、`student_arch` 若记则必须一致（{n_missing_field} 本没记这一位，"
          f"它们记于该字段上线之前 ⇒ 打印在此，不静默补默认）")
    print(f"[闸 III] 这 {n_books} 本各自记录的 `verdict.shape_rule` 分布："
          f"{'、'.join(f'{k}×{v}' for k, v in sorted(rules.items()))} ⇒ 本发只取原始错图数，不取判词")

    # ---- 闸 I：配对资格（红法本身不许绕过它）----
    flow_sizes = set()
    for tag, pairs in (("阶梯", books), ("网格", grids), ("网格(8 流)", grids8)):
        for tier, (cb, mb) in pairs.items():
            sc_, sm = [int(s) for s in cb["seeds"]], [int(s) for s in mb["seeds"]]
            if sc_ != sm:
                raise SystemExit(f"闸 I {tag}·{tier}：两族流名单不逐位相同（{sc_} vs {sm}）⇒ 不许配对")
            if sc_ != list(FLOW_SEEDS[len(sc_)]):
                raise SystemExit(f"闸 I {tag}·{tier}：那 {len(sc_)} 条流不是登记处 `FLOW_SEEDS[{len(sc_)}]` "
                                 f"那一档 ⇒ 换尺子了")
            flow_sizes.add(len(sc_))
    print(f"[闸 I] {len(books) + len(grids) + len(grids8)} 对账两两同序、同名单，且等于登记处那一档"
          f"（{'、'.join(f'{n} 条流' for n in sorted(flow_sizes))}）⇒ 有配对资格")

    # ---- 闸 VI（段② 前置·甲）：8 流那把尺子必须是 3 流那把的**加宽**，不是另起一炉 ----
    for tier, (cb8, mb8) in grids8.items():
        cb3, mb3 = grids[tier]
        nest_check(cb8, cb3, f"{tier} × CNN")
        nest_check(mb8, mb3, f"{tier} × MLP")
    if stats["nest_rows"] == 0:
        red6.append("闸 VI 一行都没比到（共有格 × 共有流 = 0）⇒ 不是通过，是没测到：段② 的甲前置无凭据")
    print(f"[闸 VI · 共流嵌套性] 8 流两对 vs 同档 3 流两对，共有流 × 共有格共比 {stats['nest_rows']} 行 × "
          f"3 个字段，红 {len(red6)} 条（这一闸要的是'同一条流同一起点'，含不进读法 B 的 `match_T6`）")

    # ---- 读法 A ----
    read_a: dict[str, dict] = {}
    for tier, (cb, mb) in books.items():
        seeds = [int(s) for s in cb["seeds"]]
        cells = {}
        for a in ALPHAS:
            off, gc, gm, by_seed = pair_block(
                cb, mb, seeds, lambda b, s: arm_row(b, a, s), lambda s: f"α{a:g}",
                f"{tier} × α={a:g} × T=6", unpair=(red == "--unpair"))
            st = paired(off, f"{tier} × α={a:g} × T=6：CNN gap − MLP gap（逐流配对，{len(seeds)} 条流）")
            st["mean_cnn"], st["mean_mlp"] = round(sum(gc) / len(gc), 1), round(sum(gm) / len(gm), 1)
            st["offset_by_seed"] = by_seed
            st["provenance"] = {"cnn": cb["_name"], "mlp": mb["_name"], "cell": f"α{a:g}",
                                "flows": seeds, "n": len(seeds)}
            cells[f"α={a:g}"] = st
        read_a[tier] = cells
    p_cells = [c for t in read_a for c in read_a[t].values()]
    p_hits = sum(1 for c in p_cells if c["ci_excludes_zero"])
    print("\n[读法 A · T=6 那一列的跨族配对偏移（正 = CNN 这一族从同一份软标签里多获益）]")
    for tier, cells in read_a.items():
        for a_key, st in cells.items():
            print(f"  {tier} {a_key}: {st['mean_gap_imgs']:+.1f} 张 CI{st['ci95_imgs']} "
                  f"同号 {st['same_sign_seeds']} 二项 p={st['p_seeds_binomial']} σ̂(D4) {st['d4_sigma_hat']} "
                  f"（CNN {st['mean_cnn']:+.1f} / MLP {st['mean_mlp']:+.1f}）→ "
                  f"{'不含 0' if st['ci_excludes_zero'] else '含 0'}")
    print(f"  ⇒ p（{len(p_cells)} 格里 CI 不含 0 的格数）= {p_hits}")

    # ---- 读法 B（3 条流那把尺子，与段② 共用 read_B 这一条代码路径）----
    read_b = read_B(grids)
    q_hits, t6_hits = print_B(read_b, "B · 同一件事在 α×T 网格上（3 条流那把尺子，只答『跟不跟 T 走』）",
                              len(grids) * len(GRID_ALPHAS))

    # ---- 读法 C / 闸 IV：形状互锁 ----
    nine = 0
    for tag, pairs in (("3 流", grids), ("8 流", grids8)):
        for tier, (cb, mb) in pairs.items():
            for t in TEMPS:
                ka = f"a1_T{t:g}"
                if ka not in cb["soft_label_audit_by_T"] or ka not in mb["soft_label_audit_by_T"]:
                    raise SystemExit(f"闸 IV {tag}·{tier} 的体检里没有 {ka} ⇒ 参评对象缺料，不是通过")
                for tag2 in ("原始", "除以T"):
                    if cb["soft_label_audit_by_T"][ka][tag2] != mb["soft_label_audit_by_T"][ka][tag2]:
                        raise SystemExit(f"闸 IV {tag}·{tier} × T={t:g} × {tag2}：两族在册体检**不等** ⇒ 出口 (d)："
                                         f"两族读的根本不是同一份软标签，本课一个字也不写")
                nine += 1
    print(f"\n[闸 IV · 形状互锁] {len(grids) + len(grids8)} 对 × T∈{{1,3,6}} 共 {nine} 格的 "
          f"`soft_label_audit_by_T`：两族逐位相同（⇒ 跨族差不可能来自形状本身）"
          f"（8 流那两对是段② 的前置·乙，v1 只锁了 3 流那三对）")

    teacher = build_cnn()
    teacher.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    teacher.eval()
    xt_all, yt_all, xte, yte = load_split()
    t_acc = float((predict(teacher, xte, False) == yte).mean())
    if abs(t_acc - TEACHER_ARCHIVE_ACC) * 100 >= 0.05:
        raise SystemExit(f"闸 IV 老师现读 {t_acc:.4f} ≠ 档案常数 {TEACHER_ARCHIVE_ACC:.4f} ⇒ 盘上这副老师"
                         f"不是算出那些体检的那一副，出口 (d)")
    n_data = DATA_TIERS[AUDIT_TIER]
    soft = logits_of(teacher, xt_all[:n_data])
    cb, mb = grids[AUDIT_TIER]
    for t in TEMPS:
        live = soft_label_audit(soft, t=t)["除以T"]
        booked = cb["soft_label_audit_by_T"][f"a1_T{t:g}"]["除以T"]
        for k in AUDIT_KEYS:
            if live[k] != booked[k] or live[k] != mb["soft_label_audit_by_T"][f"a1_T{t:g}"]["除以T"][k]:
                raise SystemExit(f"闸 IV 现算 T={t:g} 的 {k}={live[k]} ≠ 两族在册 {booked[k]} "
                                 f"⇒ 现读的不是 {AUDIT_TIER} 那 {n_data} 张图 / 那副老师")
    print(f"[闸 IV] 老师现读精度 {t_acc:.4f} == 档案 {TEACHER_ARCHIVE_ACC:.4f}；{AUDIT_TIER} 那 {n_data} 张图"
          f"现算体检与两族在册逐位相同（T∈{{1,3,6}} × {len(AUDIT_KEYS)} 个量）")

    # ---- 段②：把读法 B 的 T=1 那一列搬到 8 条流的尺子上（与 v1 共用 read_B，不是另写一套）----
    read_b8 = read_B(grids8)
    q8_hits, t86_hits = print_B(read_b8, "B2（段②）· 同一件事、同一把 α×T 网格，尺子换成 8 条流",
                                len(grids8) * len(GRID_ALPHAS))
    denom8 = len(grids8) * len(GRID_ALPHAS)
    hw_ratio: dict[str, float] = {}
    for tier in read_b8:
        for a in GRID_ALPHAS:
            k = f"α={a:g},T=1"
            hw_ratio[f"{tier} × α={a:g}"] = round(
                half_width(read_b8[tier]["offset_by_cell"][k])
                / half_width(read_b[tier]["offset_by_cell"][k]), 3)
    丙 = all(v < 1 for v in hw_ratio.values())
    甲 = (not red6) and stats["nest_rows"] > 0
    med_hw = round(statistics.median(list(hw_ratio.values())), 3)
    print(f"  前置·丙（同一格 8 流半宽 ÷ 3 流半宽，判据只要求 < 1；σ̂ 不变时预期 ≈ "
          f"{T95[8] / T95[3] * (3 / 8) ** 0.5:.3f}）："
          + "  ".join(f"{k.split(' × ')[0]}·{k.split('α=')[-1]}:{v}" for k, v in hw_ratio.items())
          + f" ⇒ {'全 < 1，通过' if 丙 else '有格 ≥ 1，段② 不可用'}")
    print(f"  前置·甲（闸 VI 共流嵌套）{'通过' if 甲 else '不过'}：比了 {stats['nest_rows']} 行、红 {len(red6)} 条；"
          f"前置·乙（闸 IV 含 8 流两对）已在上面钉住")
    if not (甲 and 丙):
        section2 = (f"(u) 前置不过 ⇒ 本段只报数，不产出任何话。甲红 {len(red6)} 条（比了 {stats['nest_rows']} 行）、"
                    f"丙半宽比 {hw_ratio} ⇒ 两把尺子要么不嵌套、要么加宽没换来更窄的区间，"
                    f"'8 流仍然测不出'这句话没有凭据（这一段没测到，不叫通过）")
    elif q8_hits == 0:
        section2 = (f"(i) q8 = 0/{denom8} 且甲乙丙全过 ⇒ 『3 条流测不动』这一支被**削弱**（不是证伪）：尺子加宽到 "
                    f"8 条流、同一格半宽实测缩到 3 流的 {med_hw:g} 倍"
                    f"（中位数，逐格 {hw_ratio}），T=1 那一列仍然处处含 0 ⇒ v1 那句 q=0 更偏'效应确实小'。"
                    f"分母是 {denom8} 格不是 9 格（中数据档没有 8 流网格），与 q 不许比大小")
    else:
        section2 = (f"(ii) q8 = {q8_hits}/{denom8} ≥ 1 ⇒ 『3 条流功效不够』直接坐实：8 条流量得出的东西 "
                    f"3 条流量不出 ⇒ v1 那句 q=0 降级为功效陈述；出口 (a) 的承重整个挪到 t_slope_paired 那条"
                    f"差之差上（v1 落地③ 本已这么写，这里不动它的判决）")
    print(f"  段② 判词：{section2[:150]}…")

    # ---- 拆红档的裁决（先于正常模式的出口判定；红档不写档案）----
    if red == "--tierblind":
        if not red3:
            raise SystemExit("闸 III（拆红模式 --tierblind）一条都没抓到 ⇒ 那道『配对两本必须同档』的闸"
                             "是恒绿的装饰（而闸 I 全绿这件事恰恰说明：光核对流名单拦不住跨档配对）")
        if red2b:
            raise SystemExit(f"--tierblind 只许红闸 III，却连 II-b 也红了 {len(red2b)} 条 ⇒ 两腿分离没验到："
                             f"注入的是**档**，配对顺序一字未动，II-b 跟着红说明它比的是档不是配对")
        print(f"  两腿分离验到：闸 I 在这一档下**全绿**（错档那本的流名单本来就逐位相同），"
              f"红只在闸 III（{len(red3)} 条），II-b 0 条")
    if red == "--unpair":
        if not red2b:
            raise SystemExit("闸 II-b（拆红模式 --unpair）一条都没抓到 ⇒ 那条『按种子号各自查档再逐位相减』"
                             "的重算是恒绿的装饰；注入的是配对顺序，均值一字不动，只有逐流序列看得见它")
        if red3:
            raise SystemExit(f"--unpair 只许红 II-b，却连闸 III 也红了 {len(red3)} 条 ⇒ 两腿分离没验到："
                             f"注入的是**配对顺序**，档/宽/族一字未动，闸 III 跟着红说明它读到了配对")
        print(f"  两腿分离验到：II-a（逐行 gap==wrong_ce−wrong_kd）、闸 I、闸 III 全绿，"
              f"红只在 II-b（{len(red2b)} 条）⇒ 『均值不变、配对已坏』这个形状被钉住了")
    if red == "--nest":
        if not red6:
            raise SystemExit(f"闸 VI（拆红模式 --nest）一条都没抓到 ⇒ 那道『8 流账必须复读同档 3 流账』的闸是"
                             f"恒绿的装饰（比了 {stats['nest_rows']} 行却看不见注入的 +1，说明它没读到那一格）")
        if red3 or red2b:
            raise SystemExit(f"--nest 只许红闸 VI，却连闸 III/II-b 也红了 {len(red3) + len(red2b)} 条"
                             f" ⇒ 两腿分离没验到：注入的是 `match_T6`，那一格不进读法 A/B，"
                             f"别的闸跟着红说明它们读到了段② 之外的东西")
        print(f"  两腿分离验到：II-a（{stats['gap_rows']} 行）、闸 I、闸 III 全绿，红只在闸 VI（{len(red6)} 条）")
    if red:
        now = OUT.read_bytes() if OUT.exists() else None
        if now != snap:
            raise SystemExit(f"{red} 拆红档动了盘上的 {OUT.name}（快照 {len(snap or b'')} 字节 → 现在 "
                             f"{len(now or b'')} 字节）⇒ 判据是『这一趟不许改动档案』，不管它跑前在不在："
                             f"注入的数一旦落盘就被当读数引用")
        print(f"\n[{red}] 该红的闸红了；红档不写档案（已逐字节核过：这一趟没动 {OUT.name}）")
        for f in (red3 + red2b + red6):
            print("   ", f)
        raise SystemExit(1)
    for f in (red3 + red2b + red6):
        print(f"  → **红**：{f}")
    if red3 or red2b or red6:
        raise SystemExit(f"正常模式下闸 II-b/闸 III/闸 VI 红了 {len(red3) + len(red2b) + len(red6)} 条"
                         f" ⇒ 先查账，本发不停在判词上")

    # ---- 出口判定（预登记那四格：(d) 已在闸 IV 处当场退出）----
    exit_rule = (
        f"(a) 交互成立：q={q_hits}（T=1 那一列 9 格处处测不出）、p={p_hits}（T=6 那一列 9 格里 "
        f"{p_hits} 格测得出），而闸 IV 已把两族的软标签形状钉成同一份 ⇒ 那条跨族差既不是『CNN 天生抗蒸馏』、"
        f"也不是『两族读到的形状不同』，是**这份形状 × 这个学生族**的交互"
        if q_hits == 0 and p_hits >= 1 else
        f"(b) 交互说不成立：q={q_hits} ≥ 1 ⇒ 跨族差在最钝的 T=1 那一列也测得出（{q_hits}/9 格不含 0）"
        f"⇒ 偏移与软标签锐度无关，更像族本身的容量/优化差；第4课与第二十一发里那些跨族句子按此重写，"
        f"不许留『交互』字样"
        if q_hits >= 1 else
        f"(c) 未判定：q=0 且 p=0 ⇒ 连 8 条流的配对也测不动 T=6 处那条跨族差，落地⑤ 那句"
        f"『近似档无关的跨族偏移』仍然只是点估计，本发不给它加任何方向词")

    report = {
        "quantity": "跨族偏移（CNN 学生 − MLP 学生，同一份软标签）跟不跟软标签的温度走：先逐流配对，"
                    "再看它在 T=1 / T=3 / T=6 三列上各是多少",
        "pays": "step4f_alpha_curve.py 落地⑤（『逐流配对、CI、同号由第廿五发现算』那句欠账）",
        "sign_convention": "gap_imgs = wrong_ce − wrong_kd（正 = KD 比 CE 少错 = 收益）；"
                           "offset = gap_cnn − gap_mlp（正 = CNN 这一族从同一份软标签里多获益）",
        "gates": {"闸I_配对资格": f"{len(books) + len(grids) + len(grids8)} 对账两两同序且等于登记处"
                                 f"（{'、'.join(f'{n} 条流' for n in sorted(flow_sizes))}）",
                  "闸II_逐流恒等式": f"II-a {stats['gap_rows']} 行 gap==wrong_ce−wrong_kd 逐位；"
                                     f"II-b {stats['blocks']} 次配对的逐流序列 == 按种子号各自查档再逐位相减的重算序列"
                                     f"（红 {len(red2b)} 条）",
                  "闸III_同档同宽家族位": f"{n_books} 本全部 S128、档位等于配方点名（红 {len(red3)} 条）；"
                                          f"{n_missing_field} 本没记 `student_arch`（打印，不补默认）",
                  "闸IV_形状互锁": f"两族 {nine} 格体检逐位相同（{len(grids)} 对 3 流 + {len(grids8)} 对 8 流 × "
                                   f"T∈{{1,3,6}}；段② 的前置·乙）+ 现算（{AUDIT_TIER} × {n_data} 张 × "
                                   f"T∈{{1,3,6}} × {len(AUDIT_KEYS)} 量）逐位相同 + 老师精度 {t_acc:.4f}",
                  "闸VI_共流嵌套性": f"8 流两对 vs 同档 3 流两对：共有流 × 共有格共 {stats['nest_rows']} 行 × "
                                     f"3 字段逐位相同（红 {len(red6)} 条）；0 行算『没测到』不算通过"
                                     f"（段② 的前置·甲）",
                  "闸V_来源": "每个数带 (档案名, 格名, 流名单)，见各格 `provenance` 与 `offset_by_seed`"},
        "reading_A_paired_T6_8flows": read_a,
        "reading_B_offset_vs_T_3flows": read_b,
        "reading_B2_offset_vs_T_8flows": read_b8,
        "section2_power_check": {"buys": "上面 docstring 落地⑧ 第 3 笔：q=0 里『效应为 0』与『3 条流测不动』分不开",
                                 "q8_T1_cells_out_of_zero": q8_hits,
                                 "denominator": f"{denom8} 格（{len(grids8)} 档 × α∈{{0.3,0.7,1.0}} 的 T=1 列）"
                                                f"——不是读法 B 的 9 格，中数据档两族都没有 8 流网格",
                                 "half_width_ratio_8_over_3": hw_ratio,
                                 "half_width_ratio_median": med_hw,
                                 "ex_ante_expected_ratio": round(T95[8] / T95[3] * (3 / 8) ** 0.5, 3),
                                 "precondition_甲_嵌套": f"{'过' if 甲 else '不过'}（比了 {stats['nest_rows']} 行、"
                                                         f"红 {len(red6)} 条）",
                                 "precondition_乙_形状互锁": f"闸 IV 已含 8 流两对（{nine} 格逐位相同）",
                                 "precondition_丙_半宽变窄": f"{'过' if 丙 else '不过'}（逐格 {hw_ratio}）",
                                 "exit": section2, "verdict": section2,
                                 "T6_column_8flows_descriptive": t86_hits,
                                 "not_a_re_verdict": "R1：这一段不重开出口——`exit`/`verdict` 两个顶层键与 v1 逐字"
                                                     "相同，段② 只住在这一个独立键里"},
        "counts": {"p_A_cells_out_of_zero": p_hits, "p_A_cells_total": len(p_cells),
                   "q_B_T1_cells_out_of_zero": q_hits, "q_B_T1_cells_total": len(grids) * len(GRID_ALPHAS),
                   "B_T6_cells_out_of_zero_descriptive": t6_hits,
                   "q8_B2_T1_cells_out_of_zero": q8_hits,
                   "q8_B2_T1_cells_total": denom8,
                   "nested_rows_compared": stats["nest_rows"]},
        "shape_identity": {"两族软标签体检逐位相同": f"{nine}/{(len(grids) + len(grids8)) * len(TEMPS)} 格"
                                                   f"（{len(grids)} 对 3 流 + {len(grids8)} 对 8 流 × T∈{{1,3,6}}）",
                          "老师在这个拟合集上的精度": TEACHER_ARCHIVE_ACC},
        "v2_increment": "v2 相对 v1 只多了段② 那几位（reading_B2_offset_vs_T_8flows、section2_power_check、"
                        "counts 里三枚 q8/嵌套计数、gates 里的闸VI）＋闸 IV 的参评对数（3 对→5 对）与由此带动的"
                        "活计数（II-a 行数 / II-b 次数 / 本数）。`quantity`、`pays`、`sign_convention`、`exit`、"
                        "`verdict`、`reading_A_paired_T6_8flows`、`reading_B_offset_vs_T_3flows` 与 v1 逐字段相同"
                        "⇒ 除这些位之外任何一处不相同，就是这段升版把判决碰着了",
        "exit": exit_rule, "verdict": exit_rule,
        "cost": {"arm_trainings": 0, "total": 0, "api_fee": 0,
                 "note": f"纯读档 + 一次老师前向：{n_books} 本在册账的原始错图数逐流相减，不训练任何学生。"
                         f"但段② 不是免费的：它吃的三本 8 流 v4c 网格（`compress_report_v4c_cnn_n8.json` / "
                         f"`compress_report_v4c_cnn_low_n8.json` / `compress_report_v4c_low_n8.json`）"
                         f"是 step4c 那 88×3 = 264 次训练换来的，训练数记在那一发，不记在本脚本"},
        "limitations": [
            "两把尺子不许横比：读法 A 是 8 条流、读法 B 是 3 条流，σ̂ 与 CI 半宽都不同；"
            "B 只回答『偏移跟不跟 T 走』，不回答『偏移是多少张』——后者只由 A 报",
            "『形状同一份』由闸 IV 钉（读档 + 现算）。它排除的是『两族读的不是同一批软标签』这一支，"
            "不排除『同一份形状对不同架构作用不同』——恰恰后者是本发的读数对象",
            "本发不新增任何独立性维度：七维（档/宽/族/任务/老师/温度列/流数）一维都没换，"
            "换的是**归因**（把点估计买成配对读数）⇒ 既不叫复证也不叫翻案",
            "读法 B 的温度只有三个点且带并列 ⇒ 这里不报相关系数：3 点、带 ties 的 Spearman ρ 在任何"
            "预登记口径下都够不着判定门槛，报了就是给噪声配一个数字",
            "出口 (a) 只数『CI 不含 0 的格数』；不含 0 的格落在哪一档、三档是否同号，全部写在 "
            "`reading_A_paired_T6_8flows` 里由读者自己看，本发不据它另立一句判词",
            f"阶梯三对里有 {len(v4_mlp)} 对的 MLP 本记的是 v4 时代的判据（{'、'.join(v4_mlp)}）：它们的 "
            f"`student_arch` 字段是缺的 ⇒ 家族位这件事只由文件名定；本发只取它们的原始错图数"
            "（`wrong_ce`/`wrong_kd`/`gap_imgs` 与判据版本无关），**不取它们的判词** ⇒ 谁引用这一格的形状判词，"
            "不许把 v4 账与 v5 账混作一版（档案里没有'这本在 v5 之下怎么判'的字段，守卫也没替它复算过）",
            "段② 只把**流数**这一维从 3 加宽到 8，档/宽/族/任务/老师/温度列一维都没换 ⇒ 它加强的是"
            "读法 B 那条结论的**功效**，不是它的外部效度；而且 q8 的分母是 6 格（中数据档两族都没有 8 流"
            "网格）——与 q 的 9 格不许比大小，只许各自报数",
            "(i) 那一支只**削弱**不证伪『3 条流测不动』：8 条流仍可能功效不足（判据只要求半宽比 < 1，"
            "没要求窄到能测出某个预先指定的效应量）⇒ 谁要把 q=0 读成『T=1 处跨族差为 0』，"
            "那是把『更宽的尺子也量不出』读成『量为 0』",
        ],
        "red_modes": {"--tierblind": "把低数据档的 MLP 阶梯换成中数据档那本（流名单仍相同）⇒ 只有闸 III 该红，"
                                     "且闸 I 必须全绿（两腿分离）",
                      "--unpair": "把配对里 CNN 那一侧整体错开一位 ⇒ 只有闸 II-b 该红；II-a、闸 I、闸 III 全绿"
                                  "（实测不变的只有**点估计与两族各自的逐侧均值**——CI 端点会跟着配对移动，"
                                  "所以看得见它的仍然是逐流序列；预登记那句'CI 宽度不动'是断言过头，见落地⑥）",
                      "--nest": "（v2 新增，钉闸 VI）把满数据档 8 流 CNN 那本 `match_T6` 里种子 42 的 "
                                "`wrong_ce` 在内存里 +1 ⇒ 只有闸 VI 该红：那一格既不进读法 A/B 的 9 个 α×T 格"
                                "（所以 II-a 看不见它），也不动档/宽/族/流名单（所以闸 I、III 看不见它）"},
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（0 次训练、零 API 费）· {exit_rule[:90]}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="第4课l（零训练）：跨族偏移拆给形状还是拆给交互"
                                             "（v2 多段②：把读法 B 的 T=1 那一列搬到 8 条流的尺子上）")
    ap.add_argument("--tierblind", action="store_true", help="把低数据档的 MLP 阶梯换成中数据档那本 ⇒ 只红闸 III")
    ap.add_argument("--unpair", action="store_true", help="把配对里 CNN 那一侧错开一位 ⇒ 只红闸 II-b")
    ap.add_argument("--nest", action="store_true",
                    help="给 8 流 CNN 那本 match_T6 的种子42 的 wrong_ce 内存里 +1 ⇒ 只红闸 VI（共流嵌套性）")
    a = ap.parse_args()
    main(red="--tierblind" if a.tierblind else "--unpair" if a.unpair else "--nest" if a.nest else "")
