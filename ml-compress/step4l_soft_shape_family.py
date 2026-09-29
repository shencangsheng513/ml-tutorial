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
（注：第廿八发 D 那一趟"换任务"的预登记与落地，在 v1/v2 两段之后、本 docstring 的最末尾。）

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
         那份手写版没删，改名留在这台机器上（不在库里：它带的是坏形状，`_` 前缀专为躲 `compress_report_v*.json`
         那一族 glob，此件不入库）：`_v4l_handcount_bad_v1.json`。它在盘上只有一处计数句，逐字抄在这里以便不依赖那个文件：
         `"II-a 153 行 gap==wrong_ce−wrong_kd 逐位；II-b 12 个格的配对序列 == 按种子号各自查档再逐位相"`。
     (4) 我预备写进 `limitations` 的那句"阶梯三对里有一对记 v4 时代判据、判词与 v5 同判（守卫闸① 复算通过）"——
         现读是**两对**（`_low_n8_thin_t6_v4`、`_n8_t6_v4` 都记 `verdict.shape_rule=="v4"`），而"在 v5 之下怎么判"
         **档案里没有那个字段、守卫也没替它复算** ⇒ 那句"复算通过"是把"没测到"写成"通过"。现在 `limitations`
         第 6 条活计 + 明写"只取原始错图数（与判据版本无关），不取判词"。
  ⑦ 两档拆红实测（凭据 = 命令形态：`python _red_v3.py --tierblind` 与 `python _red_v3.py --unpair`，
     当时落在 `_tmp_l_tierblind.log` / `_tmp_l_unpair.log`——本机一次性件、`*.log` 被本目录 `.gitignore`
     挡在库外，从第卅发起不再当引用），两腿分离双向兑现：
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
     **第廿八发把这条倒掉的判据固化成开口预检**（零训练）：丙实测没过 ⇒ "补流数 = 买分辨率"在本板块
     第一次有了反例，于是任何**再买流数**的一发，开口都要先把这个比摆出来再决定买不买。实现落在
     `step4c.power_preflight()`（α×T 网格那一族的档案名归它管，别的脚本调它而不是各抄一份），
     `step4c --nseeds >3` 在第一次训练之前调用并同步写进档案的 `power_preflight_ex_ante` 键。
     **两个量不许混**：丙比的是**跨族 offset** 的 CI 半宽（本脚本自己算，六格），预检比的是**每格 gap** 的
     CI 半宽（族内、逐流配对差，九格）——同形状、各自独立，两个都摆出来才叫"丙不是某一处的偶然"。
     预检**不拦跑也不判红**：拦跑等于替作者决定买不买，而这条读数的全部内容就是"买之前知道自己在赌什么"；
     盘上凑不出可比的两本时它明打"预检无从开口"，不许打印 1.0 假装通过。

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
  ① 实付：本脚本仍 **0** 次训练；那三本 8 流网格的 264 次记在第廿七发，不在这里。
  ② 三条前置按甲→乙→丙的顺序现读，**丙倒下**：甲过（120 行 × 3 字段逐位相同、红 0 条，`counts
     .nested_rows_compared` 就是那 120）；乙过（闸 IV 从 3 对 9 格扩到 **5 对 15 格**，8 流那两对也钉住了）；
     丙不过（六格半宽比 **0.602 ~ 1.441**、中位 **0.766**、两格 ≥1，而 σ̂ 不变时的期望是 **0.337**）
     ⇒ 整段落 **(u)：只报数，本段不产出任何话**。
  ③ 读数：q8 = **0/6**（分母写死 = 2 档 × 3 个 α；谁引用 q8 必须连这个 6 一起引用），T=6 那一列
     6/6 不含 0 只作描述（`T6_column_8flows_descriptive`）。**(u) 不等于"8 条流也测不出所以效应为 0"**：
     丙没过意味着这把新尺子**根本没变快**，它测不出是它自己的事。
  ④ R1 兑现（不是声明，是刚比过的）：把 v1 与 v2 两本档案逐字段对照，`quantity` / `pays` /
     `sign_convention` / `exit` / `verdict` / `reading_A_paired_T6_8flows` /
     `reading_B_offset_vs_T_3flows` **七个字段 JSON 归一化后逐字段相同** ⇒ 段② 的新东西只住在
     `section2_power_check` 与 `reading_B2_*` 两个独立键里，出口一格一字未动。
  ⑤ 三档专属拆红实测（两腿分离全部兑现，跑完都逐字节核过没落盘）：`--nest` 红 **1** 条、只在闸 VI；
     `--tierblind` 复跑仍红 **1** 条、只在闸 III；`--unpair` 复跑仍红 **9** 条、只在 II-b。
     活计数同步涨到：闸 I **8 对**、闸 III **16 本**（**7 本没记 `student_arch`**，打印不补默认）、
     II-a **594 行**、II-b **54 次**、闸 IV **5 对 15 格**（老师精度 0.9798 == 档案常数）。
  ⑥ 丙那条倒下的判据没有停在讨论里：第廿八发把它固化成 `step4c.power_preflight()`，任何再买流数的一发
     开口都要先摆出这个比（`[开口预检·买流数]` 那几行），凑不出相邻两本时明打"预检无从开口"。

────── 换任务那一趟的预登记（第廿八发 D，`--dataset fashion`；同样跑之前写死，事后只许追加"落地"）──────

**问的那一句**：上面那一整段只在 MNIST × 老师 A × S128 上量过。这一趟买的是同一句问话的**第二个任务取值**
——Fashion × 两族 × 三档 × 3 条流，只跑读法 B 那条代码路径（同一个 `read_B`，一字未改）。

**结构上少的三样原料**（不是"没测"，是"这一本名下没有那种账"，各自打印"无从开口"并进档案）：
  · 读法 A 吃 α→0 阶梯账 ⇒ Fashion 的 CNN 侧一格都没造过 ⇒ p 的分母是 **0**，不是 p=0；
  · 段② 与前置·甲/丙吃 8 流网格账 ⇒ Fashion 三档全部只有 3 条流（第廿七发 (u) 实测"补流数 = 买分辨率"
    有反例：六格半宽比 0.602~1.441、两格 ≥1 ⇒ 这一发只买 3 条流那一档，不做 88 次/本的补流）；
  · 闸 VI 同理 ⇒ **不记红也不记通过**。空表在这里有两种毒：`all([])` 恒真会让甲/丙"通过"，
    `median([])` 会当场崩 ⇒ 整段走 NA、判词记 None，不许塌成 True 或 0。
  留下真跑的：闸 I / II-a / II-b / 闸 III / 闸 IV（读档 + 现读 Fashion 那副老师）——与 MNIST 同一份代码路径。

**出口四格**（三个计数都在 `task_switch_crosscheck` 里；n_hit = Fashion 那 9 格里配对 CI 不含 0 的格数，
n_agree = 其中与 MNIST 同名格同号的格数，n_mag = 其中"MNIST 该格点估计落在 Fashion 该格 CI 内"的格数——
后者是"量级相当"唯一可核的形式，不引入新数）：
  (u) n_hit == 0 ⇒ **只报数、不产话**：这条偏移在第二个任务上记成"没量过"，不是"量过且没差别"
      （后一句要效应量上界，3 条流给不出）；第廿五发那句边界仍只站在 MNIST 上。
  (b) n_hit ≥ 1 且有格与 MNIST **反号** ⇒ 那句归因在第二个任务上被**否证**，边界反过来写死成 MNIST 特产；
      反号的格逐格点名，不许合并成一句方向词。
  (a) n_hit ≥ 1、全部同号、且 n_mag == n_hit ⇒ 边界从"这一个任务"扩到**两个任务**。
  (m) n_hit ≥ 1、全部同号、但 n_mag < n_hit ⇒ 换任务跟的是**方向**、不跟的是**高度**：
      不许写成"两任务偏移相当"，也不许写成"翻号"。
  （(a)/(b)/(m)/(u) 由 n_hit 是否为 0、n_dis 是否为 0、n_mag 是否等于 n_hit 三个判据拼成 ⇒ 不重不漏；
    读法 C 那一腿（闸 IV）红仍然**当场退出**，同 MNIST 的 (d)。）

**两条不许读成**：⑴ 这一趟**不产生复证**——换任务是第七维的一个新取值，不是同名坐标的两本对照；
⑵ 两个任务的 9 格**不许并成 n=18**，这里没有差之差的 CI，只有两条各自成立的 CI。

落地（跑完之后追加，上面预登记原文一字未改；读数全部来自 `compress_report_v4l_soft_shape_family_fashion_v3.json`）：
  ① 实付 **250 次训练**（不是价签原先写的 234）：读的 12 本账 = 6 本 v4b 底 × 6 + 6 本 v4c 网格 × 33 = 234，
     另 **16 次是预检件的副作用** —— `_preflight_fashion.py` 的 [6] 腿本意"零训练地真跑一次 CLI"，
     但它只枚举了两种盘态，在第三种盘态下真去训练了，产出一本计划外的 `compress_report_v4b_fashion_n8.json`
     （fashion × MLP × 满数据 × 8 条流，2 臂 × 8 流）。**本趟不读那一本** ⇒ 它不进任何判据，但钱是花了的。
     本脚本自身仍 `cost` 三项全 0（0 次训练、零 API 费）。
  ② 三处"结构上少原料"全部做成显式不适用，实测各就其位：读法 A 打"无从开口"、`counts.p_A_cells_total = 0`
     （**分母 0，不是 p=0**）；段② 与前置·甲/丙走 NA、`counts.q8_B2_T1_cells_out_of_zero = None`
     （不是 True 也不是 0）；闸 VI 打"**不适用**"而不是"比了 0 行、红 0 条"。两种毒都绕过了：
     `all([])` 那支由 NA 分支挡住，`median([])` 那支根本没被调用。
  ③ 真跑那四道闸的现读：闸 III **6 本**全 S128、红 0 条、**0 本**缺 `student_arch`（MNIST 那侧是 7 本缺）；
     闸 I **3 对**同序且等于登记处那 3 条流；II-a **162 行**逐位、II-b **27 次**配对、红 0 条；
     闸 IV 两族 **9/9 格**体检逐位相同 + 现算（中数据4500 × 4500 张 × T{1,3,6} × 4 量）逐位相同 +
     老师现读精度 **0.8771 == 档案里那个 0.8771**（与 MNIST 那本的 0.9798 各自成立，两副老师不混）。
  ④ **出口按预登记字面落在 (u)**：T=1 那一列 9 格里配对 CI 不含 0 = **0** 格（q(T=1 列)=0，与 MNIST 的
     q=0 同判）。⇒ 第廿五发那句"跨族偏移住在这份形状 × 这个族"的边界**仍然只站在 MNIST 上**，
     在 Fashion 上这一格记成**"没量过"**（不是"量过且没差别"——后一句要效应量上界，3 条流给不出）。
  ⑤ 三处"写下来的东西"被运行纠正（预登记原文不动，纠正记在这里）：
     (1) **分母口径**：上面那句"n_hit = Fashion 那 9 格"没写清是哪 9 格，而对照表实际吃的是读法 B 整张
         网格 **27 格**（3 档 × 3 α × 3 T）。两种读法给出**不同出口**：27 格 → n_hit=3、同号 3、量级 3 ⇒ **(a)**；
         T=1 那一列 9 格 → n_hit=0 ⇒ **(u)**。判据走预登记字面（T=1 列），27 格那一读法**同格并列报出**
         （`task_switch_crosscheck.whole_grid_descriptive` 与 `exit` 末尾那句），因为两种读法的差全部来自
         T≠1 的那 18 格，而那一块在 MNIST 侧本来就已知不含 0 ⇒ 拿它当分母等于把已知答案的格子算进新证据。
         **代码侧的 `"/9"` 是写死的** ⇒ 现在两个格数都由 `xt_counts()` 现读，且**一次算两种**（谁挑分母都会
         挑到对自己有利的那个，那就都摆出来）。
     (2) **档案顶层形状**：`report` 字面量里 `section2_power_check` 少一个闭合括号，而它由 `red_modes` 末尾
         一个**多出来的** `}` 顶岗 ⇒ 语法合法、`py_compile` 绿、判词照印，但 `counts` 之后 9 个顶层键
         （含 `exit`/`verdict`/`cost`/`limitations`/`task_switch_crosscheck`）全被吞成 `section2_power_check`
         的子键 —— 只有下游按名字读账的人拿到 KeyError。修法：括号搬回该在的位置 + 新增**结构闸**
         `check_report_shape()`（顶层键必须与登记处 `REQUIRED_TOP` 逐字相符，缺键则点名每个键被嵌进了
         哪个子层并**拒写档案**）。两腿实测：红腿吃那本真坏档案（改名留在这台机器上、**不在库里**：
         `_v4l_fashion_v2_swallowedkeys_v1.json`，它不入库，前缀 `_` 是为了不被 `_red_v3.py` 的
         `compress_report_v*.json` glob 扫成一本在册账）⇒ 拦下并点名 9 个键；绿腿 = 本发重跑后打印
         "顶层 20 个键齐"，且这道结构闸从此常驻在每次写盘之前（缺键就拒写，clone 之后照样打得红）。**一条盲区照实报**：MNIST 那一腿今天没有活体绿腿（那本在册，重跑会被
         "已入库"那道护栏先拦下），而键集合与任务分支无关（分支只改值不改键）。
     (3) **出口四格里"量级相当"那条腿在这一发是恒真的**：预登记把 `n_mag` 定义成"MNIST 该格点估计落在
         Fashion 该格 CI 内"的格数，而实测它**每一格都成立**（primary 那 9 格 = 9/9、整张网格 = 27/27），
         机理是 Fashion 那把尺子比 MNIST 的钝：配对 CI 半宽中位 **213.2 张 vs 18.9 张（11.3 倍）**（T=1 列）／
         **196.3 vs 39.6（5.0 倍）**（整张网格）。⇒ 出口 (a) 的第三条判据在这里**没有区分度**，它若真把那句
         边界"扩到两个任务"，扩出去的那一步只由一条永真的腿点亮。本发落在 (u)，这条腿没被选中，但缺陷属于
         预登记本身 ⇒ 处置**不是换一条判据**（那等于事后改出口），是把它的恒真**现读出来贴在判词旁边**：
         `xt_counts()` 每种读法多三个键（`n_mag_all_cells` / `ci_halfwidth_median_imgs` / `mag_leg_degenerate`），
         聚合成 `task_switch_crosscheck.leg_check` 并在出口句与屏幕上同格报出。**档案为此升版 v3**
         （护栏要求人处置 ⇒ 不覆盖 v2；v2 留盘）。逐字段对账实测：`cells` 那 27 格与 `denominators`
         **与 v2 完全相同**，只有上述三键、`leg_check` 一条与 `exit`/`verdict` 那两句话变长 ⇒ 没有一个读数被动过。
  ⑥ 没买到的（下一发别把这几条当已有）：Fashion 上的**读法 A**（要 CNN 的 α→0 薄阶梯，`step4f` 那侧
     `--arch cnn --dataset fashion` 在第一次训练之前就拒，理由已登记）〔**第卅二发 D 买掉这一条的一半**：
     低数据档那一对薄阶梯已造齐（CNN 35 次 + 它自己的 8 条流底 16 次），读法 A 在 Fashion 上从此
     **能开口**，但登记处只有 1 档、两本的共有 α 只有 {0.05, 0.7} ⇒ p 的分母是 **2 格而不是 9 格**；
     中数据/满数据两档的 CNN 薄阶梯仍然没有，那一半继续按"没买到"读；另贴一条更正——上面那句
     "在第一次训练之前就拒"是**当时**的前置状态，那条按任务一刀切的拒绝已在第卅一发撤掉（撤的是
     "整个组合一律拒"，不是"这一档没造过"），原文按老规矩不动，更正只贴在它脚边〕；Fashion 上**段② 那种加宽尺子**
     （三档全 3 条流 ⇒ 本本 q 没有兜底，测不出只能记"没量过"）；T=1 那一列在**第二个任务 × 更宽尺子**下
     的功效（两个维度一起欠着）；以及 `exit` 里那个 (m) 分支——四格在本任务上只实到 (u)，(m) 仍是空的一格。

────── 第卅三发 D+E 落地（菜单 D = 上面⑥ 的"段② 那种加宽尺子"，E = 上面⑥ 的"中数据/满数据两档 CNN 薄阶梯" +
      "T=1 那一列在第二个任务 × 更宽尺子下的功效"；预登记与上面各条原文一字未改，读数来自
      `compress_report_v4l_soft_shape_family_fashion_v5.json`）──────
  ⑦ 实付 **294 次训练**（本脚本自己仍 0 次、零 API 费），分三波开账、价签与实际逐本对上：
     3 本 v4b 底 = 48 次（`compress_report_v4b_fashion_cnn_mid_n8.json`、`compress_report_v4b_fashion_cnn_n8.json`、
     `compress_report_v4b_fashion_low_n8.json`）＋ 2 本 v4f 薄阶梯 = 70 次（各 32 + 恒等闸 3）
     ＋ 2 本 v4c 8 流网格 = 176 次（各 72 KD + 8 CE + 8 配平臂）。整批墙钟 1282 秒、退码 0。
     **波1 三本全部 SKIPPED**（盘上已有同名账 ⇒ 驱动没重复付钱，这是"单实例锁 + 幂等命名"第一次在真实批次里
     兑现，不是设计声明）。
  ⑧ 一条**我自己造成的误判**要留档：中途按 `Get-CimInstance` 的行数数进程，看到两批 ⇒ 当场向你申报
     "批被重复启动、波1 付了两遍钱"。真相是 venv 的转发桩与它 redirect 的真身各占一行，**同一个进程两行记录**
     ⇒ 数进程要按 PPID 链数，不按行数。更正的代价：我为了止损杀掉了当时正在跑的波2 两本薄阶梯（各已完成
     约 2~3 分钟、没有账落盘），随后重跑才入库 ⇒ 这一笔是**实付 > 入库**的小额浪费，两处读数都没受影响，
     但"我报过一次不存在的双跑"这件事本身要记着，不许在落地段里抹平。
  ⑨ 读数（三处第一次）：
     (1) 读法 A 在 Fashion 上**第一次有两档**（登记处从 1 档成 2 档）⇒ p 的分母从 2 格涨到 **4 格**，
         实测 p = **2/4**：满数据档两格配对 CI 不含 0（α=0.05 时 +37.6 张 CI[9.9, 65.3]；α=0.7 时
         +125.1 张 CI[51.5, 198.8]、同号 8/8、二项 p=0.0078），且 MLP 侧这两格的点估计都是**负的**
         （−8.2 / −91.4）⇒ 跨族偏移在第二个任务上第一次不是"无从开口"也不是"没量过"，而是**有格测得出**。
     (2) 段② 在 Fashion 名下**第一次有原料**（8 流网格低数据档那一对，本轮 176 次换的）⇒ 前置甲/乙/丙
         全部有活体读数：甲过（共流嵌套 60 行 × 3 字段、红 0 条）、乙过（闸 IV 把 8 流那两对也钉进形状互锁）、
         **丙也过**（三格半宽比 0.23 / 0.462 / 0.331、中位 **0.331**，对 σ̂ 不变时的期望 **0.337**）
         ⇒ 判词落 **(i)**：MNIST 那一趟"3 条流测不动"这一支在 Fashion 上被**削弱**（尺子确实变快了）。
         注意这与 MNIST 那趟的 (u) **不矛盾也不叠加**：那是同一个判据在两个任务上的两个取值，各自成立。
     (3) 顶层出口**仍按预登记字面落在 (u)**（T=1 那一列 9 格 q = 0）⇒ 本发第一次出现**"(u) 与 p ≥ 1 并存"**
         的形状。处置不是回头改出口（那等于事后挑一把对自己有利的尺子），是把并存本身现读成档案里的一位
         `task_switch_crosscheck.exit_u_vs_reading_A`：『"没量过"的主语永远是 T=1 那一列，不是"这一发
         什么都没量到"；反过来 p 那两格也不把出口抬成 (a)』。
  ⑩ 现读改造的**边界**（这条是本发的纪律而不是读数）：MNIST 那一支的文案保持**字面**可复现（在册那本 v2
     账对应的句子一个字不许因为"现在能数了"而变长），所以 `dataset_applicability`、
     `section2_power_check.denominator`、`cost`、`limitations` 的现读版本**只落在换任务那一支**；
     其中 `denominator` 那句"中数据档两族都没有 8 流网格"在 Fashion 分支改成现读缺档名，在 MNIST 分支
     保留原字面并旁注它为什么是字面。
  ⑪ 逐叶子对账（v4 ↔ v5，现读同一台机器上的两本）：共同叶子 **1538** 个里差异 **26** 处（全部是散文位的
     现读长度/档位名变化）、消失 **0** 处、新增 **508** 位 = 读法 A 满数据档 84 + 段② 低数据档 420 +
     前置·丙 3 + `exit_u_vs_reading_A` 1；顶层 `exit` / `verdict` / `quantity` / `pays` / `sign_convention` /
     `reading_B_offset_vs_T_3flows` **逐一相同**，只有 `counts` 按预期变了（p 的分母 2 → 4、q8 从 None → 0/3）。
     ⇒ 没有一格旧读数被动过。
  ⑫ 新欠（下一发别把这几条当已有）：
     (1) Fashion **中数据档的 MLP 侧**仍缺 8 流底与薄阶梯 ⇒ 读法 A 那一档要**先付 16 次造底、再付 35 次造阶梯
         = 51 次**才能补成三档；这一笔**不在**本轮宣布的 294 次价签内，等一次新的授权。
         〔第卅四发照字面付掉了这一笔；本句原样留着，是因为它是"价签怎么算出来的"的证据，不是还欠着。〕
     (2) 段② 在 Fashion 只有 **1 档**（8 流网格 1 档 vs 3 流网格 3 档）⇒ q8 的分母是 **3 格**，
         与 q 的 9 格**不许比大小**；"T=1 那一列在第二个任务 × 更宽尺子下的功效"这一条只买到 1 档 3 格，
         仍按"没量够"读。
     (3) `exit` 里那个 (m) 分支**仍然是空的一格**：本发 q=0 ⇒ 四格只实到 (u)，跟买了多少尺子无关。
         〔第卅五发把这格从"空的"改口成"沿补流这根轴买不到"，见下面 ⑬；本句原样留着是因为它当时的
         取值确实成立——它只是还不知道另一件事。〕
  ⑬ 出口 (m) 的**可达性现读**（第卅五发，零训练；不改判据、不给任何一格加豁免）：
     (1) 预登记那四格里 (m) 要的读数 = T=1 那一列里"出方向、但与 MNIST 同号、且 MNIST 点估计落在本格 CI
         **之外**"。现读 9 格：**0 格出得了方向**（配对 CI 全部含 0），所以"空的"这一句在盘上到今天仍成立。
     (2) 但"空"不等于"买得到"。按 1/√流数 把每一格的半宽换算成"要几条流才出方向"，再按档取门槛（同一档的
         三格共用一把尺子：α 在同一本网格账里，流数只能整档一起补）⇒
         全表**最早**出方向的一格是 满数据档 × α=1,T=1，约 **29 条流**（今天 3 条），而它与 MNIST **反号**
         ⇒ 沿"整档一起补流"这根轴，先撞到的出口是 **(b)**，不是 (m)。
     (3) 唯一能走到 (m) 的形状是**只**把低数据档补到约 **859 条流**（那一档 α=0.7,T=1 那格：出方向门槛 96 流、
         落 (m) 门槛 859 流）；它之所以在这一档排得上，是因为**这一档唯一反号的那格要 2730 条流才出得了方向**
         ⇒ 补到 859 流时那格还出不了方向、"抢先把出口定成 (b)"这件事还没发生（另外两档正好相反：反号格的
         出方向门槛比同号格低，362 / 29 流就先撞 (b)）。按现读价签"同一档两族每补 1 条流 = 22 次训练"
         ⇒ 约 **18,898 次**（口径钉住：这是"整档按目标流数全额计"的**上界**，没扣已有的 3 条流，扣了是 18,832
         ⇒ 量级结论不受影响）。这一格登记成**死账**（"不再买"），与第廿五发"池子算术不可买"同一处置：
         欠的不是流程，是一次明确的**不买**决定。
     (4) 顺带量出**判据自己的软肋**（这条比价签更要紧）：算 n_mag 那条腿吃的是 MNIST 的**点估计**，
         不吃 MNIST 自己的 CI；而 MNIST T=1 那 9 格的半宽现读 **10.8～39.6 张**，比要分辨的那 **8.7 张**还宽
         ⇒ 尺子加到任何一格刚出方向时，那一格几乎必然仍写着"量级对上"⇒ 判 (a) 而不是 (m)。
         处置是**登记成判据的形状**，不是改判据：事后把这条腿换成"吃 MNIST 的 CI"等于挑一把对自己有利的尺子。
     (5) 换算边界照旧钉住：半宽 ∝ 1/√流数 只在 3→8/12/16 流的**实测范围**里核过（第卅三发丙：实测中位
         0.331 vs σ̂ 不变时的期望 0.337）⇒ 外推到几百倍流数**只用来回答"够不够得着"，不写成预测**。
         凭据：`_tmp_r35_d_probe.py`（本机、不入库）现读那两本 v4l 账（fashion 的 v7 与 mnist 的 v2）逐格算出
         上面所有门槛，读表不读我的转述。
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
import step4c_alpha_T_sweep as sc            # 开口预检的唯一实现（买流数那句 (u) 读数固化在那里，不在这儿重抄）
from step4_distill import (DATA_TIERS, DATASETS, DATASET_SUFFIX, FLOW_SEEDS, T95, TEACHER_PT, dataset_loader,
                           logits_of, soft_label_audit, summarize, teacher_archive_acc)

HERE = Path(__file__).resolve().parent
# 任务位（第廿八发 D）：默认 mnist ⇒ 在册那本 v2 账的名字、读法、判决一字不动。
DATASET = "mnist"
# v2：只多段② 那几个键（`exit`/`verdict`/读法 A/B 逐字段与 v1 相同）；v1 档案留盘做逐字段对账
# 版本位登记处（不自动升：自动挑号等于绕过"已入库"那条要人处置的护栏）。fashion 升 v3 的原因写在
# 本文件 docstring 的 D 落地段：往 `task_switch_crosscheck` 里加了三个"腿自己有没有区分度"的键。
# fashion 再升 v4（第卅二发 D）：`reading_A_paired_T6_8flows` 从空表变成"低数据档一对 × 共有 α 两格"，
# `reading_A_applicability` 那句随之不能再说"没造过"⇒ 内容变了就换号，v3 留盘不覆盖。mnist 那本一个字不动。
# fashion 又升 v5（第卅三发 D+E）：两件事同时变了——读法 A 从一档成两档（满数据档那一对买齐了），
# 段② 与闸 VI 在 Fashion 名下**第一次有原料**（低数据档 8 流网格两族各一本）⇒ 那三处"不适用"的文案
# 全部随之不能再说"没有"。理由细节在 docstring 的 D 落地段。mnist 那本照旧一个字不动。
# fashion 又升 v6（第卅四发 B）：读法 A 从**两档**成**三档**（中数据档那对的 MLP 腿买齐了），而且这一对
# 两族**共有三格** α（{0.05, 0.3, 0.7}——MLP 那本走的是按族分册的账，闸格 α=0.3 独立于锚点），前两档那
# 两对仍只有两格 ⇒ p 的分母从 4 格变成 4+3=7 格。边界照第卅二发那句：分母变大 ≠ 换任务的判决变硬，
# 它只说明能点名的格子多了三格。
# fashion 再升 v7（第卅四发收口，0 次训练）：**不是读数变了，是 `cost` 里那句写死的清单过期了**——
# 那句话写"14 本分三发买（234 + 51 + 294）"，而本发 B 又付了 51 次（中数据档 MLP 侧那两本），
# "报的价必须等于付的价"这条纪律不容一句假账留在档案里 ⇒ 按板块规矩内容变了就换号、v6 留盘不覆盖，
# 并附 v6→v7 逐叶子对账（差异只许落在 `cost` 那两个字段上，读数键一位不动）。
REPORT_VER = {"mnist": "v2", "fashion": "v7"}
OUT = HERE / f"compress_report_v4l_soft_shape_family{DATASET_SUFFIX[DATASET]}_{REPORT_VER[DATASET]}.json"
MNIST_V2 = HERE / "compress_report_v4l_soft_shape_family_v2.json"   # 换任务那趟的对照基准（按内容核身份）

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
# 第廿八发 D：读法 B 搬到第二个任务。六本网格账是那一发的 198 次训练换来的，档位后缀跟着 step4c 的名字规则。
GRID_FASHION = {
    "低数据2000": ("compress_report_v4c_fashion_cnn_low.json", "compress_report_v4c_fashion_low.json"),
    "中数据4500": ("compress_report_v4c_fashion_cnn_mid.json", "compress_report_v4c_fashion_mid.json"),
    "满数据8500": ("compress_report_v4c_fashion_cnn.json", "compress_report_v4c_fashion.json"),
}
# Fashion 名下这两张表**历史上就是空的**，不是忘了填（第卅二发之前，读法 A 的 CNN 侧一格都没造过）：
#   · 读法 A 吃的是 α→0 阶梯账，硬凑只能去借 MNIST 的 CNN 阶梯，那是换任务冒充同任务；
#   · 段② 吃的是 8 流网格，Fashion 两族都只有 3 流那一档（这一发只买 3 条流，理由写在第廿七发 (u)）
#     ⇒ 甲（共流嵌套）与丙（半宽变窄）都没有原料。
# 所以这两处在 Fashion 模式下打印"无从开口"并把键写进档案，**不回落 MNIST**。
# 〔第卅二发：读法 A 那一半的空格按菜单 D 买掉了低数据档；第卅三发 D+E 又买掉满数据档，
#  于是读法 A 在 Fashion 名下是**两档**、不是三档：缺的正是中间那一档，而且缺的不是 CNN 侧
#  （CNN 的中数据薄阶梯本轮已造齐）是 **MLP 侧**——`compress_report_v4b_fashion_mid_n8.json` 那本 8 流底**不在库里**
#  （本轮没付那 16 次，盘上现读就没有这一本），所以先付 16 次造底、再付 35 次造阶梯 = 51 次才能补成三档。这一笔**不在**
#  本轮宣布的 294 次价签里 ⇒ 记成新欠、等一次新的授权，p 的分母按现读算
#  （见 reading_A_applicability 那句），不许按 MNIST 的 9 格读，也不许写成"三档齐"。
#  第卅三发同时把 GRID8_FASHION 从空表变成一档（低数据 × 两族各一本 8 流网格，本轮 176 次换的）
#  ⇒ 段② / 闸 VI / 前置·甲丙 在 Fashion 名下第一次有原料；中数据档两族都没有 8 流网格，
#  满数据档也只有 3 流 ⇒ 这一张表和 MNIST 那张一样**不是**三档齐，q8 分母 = 1×3 = 3 格。
#  〔第卅四发：上面那段里"那本 8 流底不在库里""本轮没付那 16 次""读法 A 在 Fashion 名下是两档、
#   缺的正是中间那一档"三句**已过期**，原文照抄不删——那 16 + 35 = 51 次本轮付了，中数据档那对
#   两族薄阶梯都造齐并进了 `LADDER_FASHION` ⇒ 读法 A 在 Fashion 是**三档**、p 的分母按现读是 7 格。
#   这一笔的连带效果现读得到：闸⑯ 的名字分母里少了一个——那句带"不在库里"声明的引用，
#   从今天起指着的是一本**在册**的账，于是它不再是库外断链候选（名字 46 → 45、命中 99 → 98）。〕
LADDER_FASHION: dict[str, tuple[str, str]] = {
    "低数据2000": ("compress_report_v4f_fashion_cnn_low_n8_thin_t6_v5.json",
                   "compress_report_v4f_fashion_low_n8_thin_t6_v5.json"),
    # 中数据档是第卅四发新补的一对，也是换任务侧**第一对两族走同一条读账路由**的账：两本都用按族分册
    # 的 v4b+v4c（底来自各自那本 8 流 v4b、闸格 α=0.3 独立于锚点 ⇒ `gate_is_anchor=False`）。
    # 下面低数据/满数据那两对的 MLP 腿走的是 step6 那本按档合并的旧账（底 3 条流的 `own_floor_imgs`、
    # 闸格恰好落在锚点上 ⇒ `gate_is_anchor=True`）⇒ 那两对的跨族配对本来就是两把尺子，只有名字对齐。
    "中数据4500": ("compress_report_v4f_fashion_cnn_mid_n8_thin_t6_v5.json",
                   "compress_report_v4f_fashion_mid_n8_thin_t6_v5.json"),
    # MLP 那一本的名字是旧命名规则下的（没有 `_t6` 也没有 `_v5`，温度只能由档案字段核）：
    # 这里登记的是"同一坐标两族各一本"这件事，温度/档/宽/流数全部由闸 III 与闸 I 现读，不由名字读。
    "满数据8500": ("compress_report_v4f_fashion_cnn_n8_thin_t6_v5.json",
                   "compress_report_v4f_fashion_n8_thin.json"),
}
GRID8_FASHION: dict[str, tuple[str, str]] = {
    "低数据2000": ("compress_report_v4c_fashion_cnn_low_n8.json", "compress_report_v4c_fashion_low_n8.json"),
}
AUDIT_KEYS = ("mean_top1", "pct_top1_gt_0.99", "mean_p2", "mean_margin_top1_top2")


def books_for(ds: str) -> tuple[dict, dict, dict]:
    """三张表按任务取，且只在这里取一次（别在每个闸里各写一遍 `if ds == ...`）。"""
    if ds == "fashion":
        return LADDER_FASHION, GRID_FASHION, GRID8_FASHION
    if ds == "mnist":
        return LADDER, GRID, GRID8
    raise SystemExit(f"任务 {ds!r} 在 step4l 没有登记账表 ⇒ 要它得先登记那三张表，"
                     f"而不是让它安静地读 MNIST 的本")


def out_path(ds: str) -> Path:
    """产物名里的任务位 + 版本位（查 `REPORT_VER`）。mnist 是空串（在册那本 v2 的名字就是本板块的身份，
    改名等于断开所有对照）；版本位**不自动升**——自动挑一个没被占用的号，等于绕过"已入库"那条要人处置的护栏。"""
    return HERE / f"compress_report_v4l_soft_shape_family{DATASET_SUFFIX[ds]}_{REPORT_VER[ds]}.json"

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


def read_baseline() -> dict:
    """换任务那一趟的对照基准 = 在册 MNIST 那本 v2 的 `reading_B_offset_vs_T_3flows`。

    身份**按内容**核，不按文件名核：`dataset` 位（老账没记就读成 mnist，并把"没记"数出来）+ 每一格
    `provenance` 里两族的档案名不许带任务位 `_fashion`（那说明基准本身被换成了 Fashion 的账，
    于是"换任务对照"变成自己跟自己比）。覆盖不足**当场退出**——少比几格继续判，报出来的分母就是假的。
    """
    if not MNIST_V2.exists():
        raise SystemExit(f"对照基准 {MNIST_V2.name} 不在盘上：换任务那一趟吃的是 MNIST 在册那 9 格，"
                         f"缺它就没有'换任务'这个读数（不是回落成只报 Fashion 自己的数）")
    d = json.loads(MNIST_V2.read_text(encoding="utf-8"))
    if d.get("dataset", "mnist") != "mnist":
        raise SystemExit(f"{MNIST_V2.name} 记的任务是 {d['dataset']}，不是 mnist ⇒ 拿它当 MNIST 侧基准是接错线")
    rb = d["reading_B_offset_vs_T_3flows"]
    print(f"[基准] 对照读 {MNIST_V2.name}（dataset={'mnist（老账没记这一位，按历史口径读成 mnist）' if 'dataset' not in d else d['dataset']}）"
          f"：{len(rb)} 档 × {len(next(iter(rb.values()))['offset_by_cell'])} 格")
    return rb


def baseline_cell(rb: dict, tier: str, cell: str) -> dict:
    blk = rb.get(tier)
    if blk is None or cell not in blk["offset_by_cell"]:
        raise SystemExit(f"基准那本里没有 {tier} × {cell} ⇒ 换任务的对照表缺格（分母会虚报成少几格），"
                         f"先查那一本的档位/α/T 取法与本次是否同一张网格")
    return blk["offset_by_cell"][cell]


def cross_task(read_b: dict, rb: dict) -> tuple[dict, int, int]:
    """把 Fashion 的逐格 offset 与 MNIST **同名格**并排。这不是配对：两任务的流、图、老师都不同，
    所以这里只有'两条 CI 各摆各的'与'符号是否一致'，没有差之差的 CI。"""
    cmp: dict[str, dict] = {}
    n_hit = n_agree = 0
    for tier, blk in read_b.items():
        for cell, st in blk["offset_by_cell"].items():
            base = baseline_cell(rb, tier, cell)
            lo, hi = st["ci95_imgs"]
            bm = float(base["mean_gap_imgs"])
            item = {"fashion_mean_imgs": st["mean_gap_imgs"], "fashion_ci95_imgs": st["ci95_imgs"],
                    "fashion_out_of_zero": st["ci_excludes_zero"], "fashion_flows": st["provenance"]["flows"],
                    "mnist_mean_imgs": bm, "mnist_ci95_imgs": list(base["ci95_imgs"]),
                    "mnist_out_of_zero": base["ci_excludes_zero"],
                    "same_sign": (st["mean_gap_imgs"] > 0) == (bm > 0) if st["mean_gap_imgs"] and bm else None,
                    # 预登记里"量级相当"的可核形式：基准那格的点估计落在本格 CI 之内（不引入新数）
                    "mnist_point_inside_fashion_ci": lo <= bm <= hi,
                    "provenance": st["provenance"],
                    "baseline_book": MNIST_V2.name}
            cmp[f"{tier} × {cell}"] = item
            if item["fashion_out_of_zero"]:
                n_hit += 1
                if item["same_sign"]:
                    n_agree += 1
    return cmp, n_hit, n_agree


def halfwidths(cmp_table: dict, keys) -> tuple:
    """这一读法全部格子上，两任务各自那条配对 CI 半宽的中位数（张）。"""
    def med(ci) -> float:
        return round(statistics.median([(cmp_table[k][ci][1] - cmp_table[k][ci][0]) / 2 for k in keys]), 1)
    return med("fashion_ci95_imgs"), med("mnist_ci95_imgs")


def mag_leg_note(name: str, d: dict) -> str:
    """『量级相当』这条腿自己有没有区分度——恒真就当场点名，不留给读者事后发现。"""
    if not d["mag_leg_degenerate"]:
        return ""
    hw = d["ci_halfwidth_median_imgs"]
    return (f"但 {name} 那一读法的『量级相当』腿是**恒真的**：{d['n_cells']} 格里 "
            f"{d['n_mag_all_cells']}/{d['n_cells']} 格『MNIST 点估计落在 Fashion 本格 CI 内』"
            f"（半宽中位 Fashion {hw['fashion']} 张 vs MNIST {hw['mnist']} 张，约 "
            f"{round(hw['fashion'] / hw['mnist'], 1) if hw['mnist'] else None} 倍钝）⇒ 它是"
            f"**这个任务上那把尺子更钝**的副产品，不是『两个任务偏移高度相当』的证据；这一读法若被选中，"
            f"出口 (a) 只由一条恒真的腿点亮")


def xt_counts(cmp_table: dict) -> dict:
    """两种读法各自的计数，**一次全算出来**（谁挑分母都会挑到对自己有利的那一个 ⇒ 那就都摆出来）。

    primary = 预登记字面上那"9 格" = T=1 那一列（与 q 同口径，也是本发唯一的新证据）；
    whole   = 对照表实际吃的整张读法 B 网格（3 档 × 3 α × 3 T）。
    除三个判据计数外，每种读法还现读自己的**腿有没有区分度**（`n_mag_all_cells` / 半宽 / `mag_leg_degenerate`）
    ——它们不进判据，只负责让"换任务后高度相当"这类话在账上无处藏身。
    """
    def cnt(keys) -> dict:
        hit = [k for k in keys if cmp_table[k]["fashion_out_of_zero"]]
        mag_all = sum(1 for k in keys if cmp_table[k]["mnist_point_inside_fashion_ci"])
        hw_f, hw_m = (halfwidths(cmp_table, keys) if keys else (None, None))
        return {"n_cells": len(keys), "n_hit": len(hit),
                "n_agree": sum(1 for k in hit if cmp_table[k]["same_sign"]),
                "n_mag": sum(1 for k in hit if cmp_table[k]["mnist_point_inside_fashion_ci"]),
                "hits": hit,
                # ↓ 两条"腿自己有没有区分度"的现读诊断（不参与判据，只负责让读者不必自己发现）
                "n_mag_all_cells": mag_all,
                "ci_halfwidth_median_imgs": {"fashion": hw_f, "mnist": hw_m},
                "mag_leg_degenerate": bool(keys) and mag_all == len(keys)}
    t1 = [k for k in cmp_table if k.endswith(",T=1")]
    return {"T1_column_pre_registered": cnt(t1), "whole_grid_descriptive": cnt(list(cmp_table))}


def exit_fashion(q_hits: int, cmp_table: dict) -> str:
    """换任务那一趟的出口（四格穷尽；判据只吃 `xt_counts()` 里 primary 那三个计数，都在
    `task_switch_crosscheck` 里可核）。

    n_hit = **预登记字面**那 9 格（T=1 那一列）里配对 CI 不含 0 的格数；n_agree / n_mag = 这些格中与 MNIST
    **同名格**同号 / "MNIST 点估计落在本格 CI 内"（预登记里"量级相当"的可核形式）的格数。四格互斥且覆盖全部取值。
    全表（27 格）那一读法的格数与出口在下面 `whole_grid_descriptive` 里并列报出，不替换判据。
    """
    c = xt_counts(cmp_table)
    p, w = c["T1_column_pre_registered"], c["whole_grid_descriptive"]
    hit, n_hit = p["hits"], p["n_hit"]
    n_agree, n_mag = p["n_agree"], p["n_mag"]
    n_dis = n_hit - n_agree
    whole_exit = ("(u)" if w["n_hit"] == 0 else
                  "(b)" if w["n_hit"] - w["n_agree"] >= 1 else
                  "(a)" if w["n_mag"] == w["n_hit"] else "(m)")
    alt = (f"（若把整张读法 B 网格 {w['n_cells']} 格当分母：n_hit={w['n_hit']}、同号 {w['n_agree']}、"
           f"量级对上 {w['n_mag']} ⇒ 出口是 {whole_exit}。两种读法的差全部来自 T≠1 的那 "
           f"{w['n_cells'] - p['n_cells']} 格，而那一块在 MNIST 侧本来就已知不含 0 ⇒ "
           f"判据按预登记字面走 primary，另一读法只作描述，事后不许挑对自己有利的那个。"
           f"{mag_leg_note('whole_grid_descriptive', w)}）")
    if n_hit == 0:
        return (f"(u) 按预登记字面（{p['n_cells']} 格 = T=1 那一列）配对 CI 不含 0 = **0** 格"
                f"（q(T=1 列)={q_hits}），且逐格点估计的符号都没站住 ⇒ 这一发的读数是**只报数、不产话**："
                f"这条跨族偏移在第二个任务上记成**『没量过』**，不是『量过且没差别』（后一句要的是效应量的"
                f"上界，3 条流给不出）。第廿五发那句边界仍然只站在 MNIST 上{alt}")
    if n_dis >= 1:
        bad = "、".join(f"{k}（Fashion {cmp_table[k]['fashion_mean_imgs']:+.1f} vs MNIST "
                        f"{cmp_table[k]['mnist_mean_imgs']:+.1f}）" for k in hit
                        if not cmp_table[k]["same_sign"])
        return (f"(b) Fashion 在 {p['n_cells']} 格里有 {n_hit} 格测得出，其中 **{n_dis} 格与 MNIST 反号**"
                f"（{bad}）⇒ 第廿五发那句『跨族偏移』在第二个任务上被**否证**：它的边界反过来写死成 MNIST 特产。"
                f"逐格点名在此，不许合并成一句方向词{alt}")
    if n_mag == n_hit:
        return (f"(a) Fashion 在预登记那 {p['n_cells']} 格里有 {n_hit} 格测得出、全部与 MNIST 同名格同号，"
                f"且 MNIST 该格点估计都落在 Fashion 该格 CI 内（『量级相当』的可核形式）⇒ 那条『只换学生族就有"
                f"跨族偏移』的句子边界从『这一个任务』扩到**两个任务**（q(T=1 列)={q_hits}，与 MNIST 的 q=0 同判）。"
                f"{mag_leg_note('primary（T=1 那一列）', p)}{alt}")
    weak = "、".join(f"{k}（Fashion {cmp_table[k]['fashion_mean_imgs']:+.1f} "
                     f"CI{cmp_table[k]['fashion_ci95_imgs']} vs MNIST {cmp_table[k]['mnist_mean_imgs']:+.1f}）"
                     for k in hit if not cmp_table[k]["mnist_point_inside_fashion_ci"])
    return (f"(m) Fashion 在预登记那 {p['n_cells']} 格里有 {n_hit} 格测得出、全部与 MNIST 同号，"
            f"但 **{n_hit - n_mag} 格的量级对不上**（MNIST 点估计落在本格 CI 之外：{weak}）⇒ "
            f"换任务跟的是**方向**、不跟的是**高度**：不许写成『两个任务上偏移相当』，也不许写成『翻号』"
            f"（一格都没翻）{alt}")


REQUIRED_TOP = ("dataset", "dataset_applicability", "quantity", "pays", "sign_convention", "gates",
                "reading_A_paired_T6_8flows", "reading_A_applicability", "reading_B_offset_vs_T_3flows",
                "reading_B2_offset_vs_T_8flows", "section2_power_check", "counts", "shape_identity",
                "task_switch_crosscheck", "v2_increment", "exit", "verdict", "cost", "limitations", "red_modes")


def check_report_shape(report: dict) -> None:
    """结构闸：顶层键必须与登记处逐字相符。缺键 = 某个未闭合的字面量把它们吞成了子键。

    这类事故的毒在于它**不报语法错**：`exit` 的判词照样打印、档案照样写出，坏的是名字空间——下游按
    `report["task_switch_crosscheck"]` 读账的人拿到 KeyError，而判词已经进了 README。"""
    miss = [k for k in REQUIRED_TOP if k not in report]
    extra = [k for k in report if k not in REQUIRED_TOP]
    if not miss and not extra:
        print(f"\n[结构闸] 顶层 {len(REQUIRED_TOP)} 个键齐（现读登记处）⇒ 没有键被嵌进子层")
        return
    where = {k: [pk for pk, pv in report.items() if isinstance(pv, dict) and k in pv] for k in miss}
    raise SystemExit(f"档案顶层形状不对：缺 {miss}"
                     + ("".join(f"；'{k}' 其实被嵌进了 {v}" for k, v in where.items() if v) or
                        "；它们不在任何子层里 = 真的没写")
                     + (f"；多出 {extra}" if extra else "")
                     + " ⇒ 拒写：判词已经印出来了，但按名字读这本账的下游会拿到 KeyError")


def arm_row(book: dict, alpha: float, seed: int) -> dict:
    key = f"α{alpha:g} 种子{seed}"
    if key not in book["arms"]:
        raise SystemExit(f"闸 III {book['_name']} 里没有 {key} 这一格（有 "
                         f"{'、'.join(sorted({k.split(' 种子')[0] for k in book['arms']}))}）")
    return book["arms"][key]


def ladder_alphas(cb: dict, mb: dict) -> tuple[float, ...]:
    """这一对阶梯账**真的**共有哪几格 α：两本 `arms` 键名的交集，再与登记的那三格相交。
    MNIST 那三对算出来就是 `ALPHAS` 本身 ⇒ 读数一字不动；Fashion 低数据档的 MLP 那本只有
    {0.05, 0.7}（薄阶梯没跑 α=0.3）⇒ 交集是两格，p 的分母跟着变成 2，而不是"少比一格、照 9 报"。"""
    def cell_alphas(b: dict) -> set:
        return {float(k.split(" 种子")[0][1:]) for k in b["arms"]}
    both = cell_alphas(cb) & cell_alphas(mb)
    return tuple(a for a in ALPHAS if a in both)


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


def main(red: str = "", ds: str = "mnist") -> None:
    global DATASET, OUT
    DATASET = ds
    OUT = out_path(ds)                 # 不在 def 时绑死：换任务要换的是这一本的**名字**，不只是内容
    sc.DATASET = ds                    # 账名规则与开口预检的唯一实现住在 step4c，这里不重抄一份
    lad_t, grid_t, grid8_t = books_for(ds)
    if ds != "mnist" and red:
        raise SystemExit(f"{red} 的注入点写在 MNIST 那三张账表上（阶梯/8 流网格），任务换成 {ds} 之后那些本"
                         f"根本不参评 ⇒ 换任务这一趟不许跑拆红档（要么先给这个任务登记注入点，"
                         f"要么它会安静地红在别处，那不算拆红）")
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

    books = {t: (read(c), read(m)) for t, (c, m) in lad_t.items()}
    grids = {t: (read(c), read(m)) for t, (c, m) in grid_t.items()}
    grids8 = {t: (read(c), read(m)) for t, (c, m) in grid8_t.items()}
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
    # 阶梯配对里 MLP 侧那几本的"字段齐不齐"——现读，不手写：Fashion 名下的 `compress_report_v4f_fashion_n8_thin.json` 与
    # `compress_report_v4f_fashion_low_n8_thin_t6_v5.json` 都记于 `student_arch` 上线之前，而 MNIST 名下三本齐不齐由这里同一句算出来。
    no_arch_ladder_mlp = sorted(mb["_name"] for cb, mb in books.values() if mb.get("student_arch") is None)
    no_rule_ladder_mlp = sorted(mb["_name"] for cb, mb in books.values()
                                if not (isinstance(mb.get("verdict"), dict)
                                        and mb["verdict"].get("shape_rule")))
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
    vi_na = not grids8
    if not vi_na:
        for tier, (cb8, mb8) in grids8.items():
            cb3, mb3 = grids[tier]
            nest_check(cb8, cb3, f"{tier} × CNN")
            nest_check(mb8, mb3, f"{tier} × MLP")
        if stats["nest_rows"] == 0:
            red6.append("闸 VI 一行都没比到（共有格 × 共有流 = 0）⇒ 不是通过，是没测到：段② 的甲前置无凭据")
        print(f"[闸 VI · 共流嵌套性] 8 流两对 vs 同档 3 流两对，共有流 × 共有格共比 {stats['nest_rows']} 行 × "
              f"3 个字段，红 {len(red6)} 条（这一闸要的是'同一条流同一起点'，含不进读法 B 的 `match_T6`）")
    else:
        print(f"[闸 VI · 共流嵌套性] **本任务（{DATASET}）无从开口**：段② 要的 8 流网格账两族都没有"
              f"（这一发只买 3 条流，理由 = 第廿七发 (u)『补流数不保证买到分辨率』）⇒ 甲前置没有原料，"
              f"既不记红也不记通过；段② 那一整段随之不产出任何话")

    # ---- 读法 A ----
    if not books:
        print(f"\n[读法 A · 本任务（{DATASET}）**无从开口**] 它吃的是 α→0 阶梯账（同一坐标两族各一本），"
              f"而登记处在这一名下是**空的** ⇒ 没有可配对的坐标，p 的分母是 0；本课不借别的任务的 CNN 阶梯来凑。"
              f"（历史上 Fashion 名下三档全空；第卅二发 D 买掉低数据档那一对、第卅三发 D 买掉满数据档那一对，"
              f"剩的中数据档**不是**『再买一薄』就齐：CNN 侧已造齐，缺的是 MLP 侧那本 8 流薄阶梯的**底**，"
              f"要先付 16 次造底再付 35 次造阶梯 = 51 次，见 README【第三十三发】）")
    read_a: dict[str, dict] = {}
    for tier, (cb, mb) in books.items():
        seeds = [int(s) for s in cb["seeds"]]
        alphas = ladder_alphas(cb, mb)
        if not alphas:
            print(f"  {tier}：登记了这一对，但两本账**没有共有的 α 格** ⇒ 这一档不进 p 的分母"
                  f"（是『无从开口』，不是『比过且 p=0』）")
        cells = {}
        for a in alphas:
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
    print(f"  ⇒ p（{len(p_cells)} 格里 CI 不含 0 的格数）= {p_hits}"
          + ("" if p_cells else "  ← 分母是 0：见上面那句『无从开口』，这一格不是 p=0"))

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

    teacher_pt = MODELS / TEACHER_PT[DATASET]
    if not teacher_pt.exists():
        raise SystemExit(f"盘上没有 {teacher_pt.name}（任务 {DATASET} 的老师权重）⇒ 闸 IV 的现算那一腿没有原料，"
                         f"本课不许借别的任务的老师来'证明'两族读的是同一份软标签")
    acc_archive = teacher_archive_acc(DATASET)
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(teacher_pt, weights_only=True))
    teacher.eval()
    xt_all, yt_all, xte, yte = dataset_loader(DATASET)()
    t_acc = float((predict(teacher, xte, False) == yte).mean())
    if abs(t_acc - acc_archive) * 100 >= 0.05:
        raise SystemExit(f"闸 IV 老师现读 {t_acc:.4f} ≠ 档案值 {acc_archive:.4f} ⇒ 盘上这副老师"
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
    print(f"[闸 IV] 老师现读精度 {t_acc:.4f} == 档案 {acc_archive:.4f}（任务 {DATASET}）；{AUDIT_TIER} 那 {n_data} 张图"
          f"现算体检与两族在册逐位相同（T∈{{1,3,6}} × {len(AUDIT_KEYS)} 个量）")

    # ---- 段②：把读法 B 的 T=1 那一列搬到 8 条流的尺子上（与 v1 共用 read_B，不是另写一套）----
    if not grids8:
        # 空表在这里有两种毒：`all(v < 1 for v in {})` 恒真 ⇒ 前置·丙会"通过"；
        # `statistics.median([])` 当场崩。两者都不许发生，所以整段走 NA、判词记 None 而不是记 True。
        read_b8, q8_hits, t86_hits, denom8, med_hw, hw_ratio = {}, None, None, 0, None, {}
        丙, 甲 = None, None
        section2 = (f"(不适用) 任务 {DATASET} 名下没有 8 条流的 v4c 网格账 ⇒ 段② 的甲（共流嵌套）与丙（半宽变窄）"
                    f"都没有原料，q8 无从定义。**不适用不等于通过**：这一发只买 3 条流那一档，"
                    f"理由 = 第廿七发 (u)『补流数 = 买分辨率』已有反例（六格半宽比 0.602~1.441、两格 ≥1）")
        print(f"\n[段②] {section2[:130]}…")
    else:
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

    # 第廿八发：把 (u) 固化成开口预检之后，这里顺手摆出**另一把尺**上的同一形状（每格 gap 的 CI 半宽，
    # 族内九格）。它跟上面丙那六格（跨族 offset）是两个量，所以两行都印、各标各的口径。
    # 档的来源也跟着分叉：有 8 流原料时问段② 那两档（与丙同坐标），没有就问本趟真在读的那几档——
    # 预检在 Fashion 上必然报"无从开口"（同坐标只有一本 3 流账），而那句话本身就是要留下的记录。
    print("  [开口预检·第廿八发固化 (u)] 与丙不同量：下面比的是**每格 gap** 的 CI 半宽（族内、九格），"
          "由 step4c.power_preflight 现读盘上同坐标的相邻两本：")
    for _fam in ("mlp", "cnn"):
        for _t in sorted(read_b8 if grids8 else read_b):
            sc.power_preflight(128, _t, where=f"step4l {'段②' if grids8 else '换任务'}·{_fam}", arch=_fam)

    # ---- 换任务那一趟的对照（第七维换取值）：与 MNIST 同名格并排。不配对、不合并分母 ----
    xtask = None
    if DATASET != "mnist":
        cmp_table, _, _ = cross_task(read_b, read_baseline())
        xc = xt_counts(cmp_table)
        p, w = xc["T1_column_pre_registered"], xc["whole_grid_descriptive"]
        xtask = {"buys": "第廿八发 D：把『同数据、同老师、只换学生族』那条跨族 offset 搬到第二个任务上",
                 "baseline_book": MNIST_V2.name, "cells": cmp_table,
                 "denominators": (f"两种读法都算：primary = 预登记字面那 {p['n_cells']} 格（T=1 那一列，与 q 同口径）"
                                  f"⇒ 进判据；whole = 读法 B 整张网格 {w['n_cells']} 格（{len(grids)} 档 × "
                                  f"{len(GRID_ALPHAS)} 个 α × {len(TEMPS)} 个 T）⇒ 只作描述。预登记那句只写了"
                                  f"『那 9 格』没写口径，落地时如实登记为它自己的缺陷（见 docstring 落地段），"
                                  f"两种读法的出口都在 `exit` 那一格里"),
                 "T1_column_pre_registered": {k: v for k, v in p.items() if k != "hits"} | {"cells_out_of_zero": p["hits"]},
                 "whole_grid_descriptive": {k: v for k, v in w.items() if k != "hits"} | {"cells_out_of_zero": w["hits"]},
                 "leg_check": " ".join(x for x in (mag_leg_note("primary（T=1 那一列）", p),
                                                   mag_leg_note("whole（整张网格）", w)) if x)
                                or "两种读法的『量级相当』腿都**不是**恒真（各自有区分度）",
                 "not_paired": "两任务的流、图、老师都不是同一批 ⇒ 这里没有差之差的 CI，只有两条各自成立的 CI；"
                               f"Fashion 的 {w['n_cells']} 格与 MNIST 的 {w['n_cells']} 格不许并成一个计数"}
        # 并存形状那一键（`exit_u_vs_reading_A`）在出口算出来之后才补，见下面 `exit_rule` 那一段的末尾：
        # 它的话头要引用出口实落哪一格，写在字典字面量里就是拿一个还没算出来的数下断言。
        print(f"\n[换任务对照] 按预登记字面（T=1 那一列 {p['n_cells']} 格）配对 CI 不含 0 = {p['n_hit']} 格"
              f"（同号 {p['n_agree']}、量级对上 {p['n_mag']}）；整张网格 {w['n_cells']} 格口径则是 "
              f"{w['n_hit']} 格（同号 {w['n_agree']}、量级 {w['n_mag']}）"
              f"（基准 {MNIST_V2.name}；两本同格子、非配对）")
        print(f"[换任务对照·腿自查] {xtask['leg_check']}")
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
    if DATASET != "mnist":
        exit_rule = exit_fashion(q_hits, cmp_table)
        print(f"\n[出口 · 换任务] {exit_rule[:150]}…")
    else:
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

    if xtask is not None:                        # 并存形状那一键在这里补：出口落点此刻才算得出来
        xtask["exit_u_vs_reading_A"] = (
            f"顶层 `exit` 实落 {exit_rule[:3]}：出口判据按预登记字面只数 T=1 那一列"
            f"（{len(grids) * len(GRID_ALPHAS)} 格，q={q_hits}），而同一趟的读法 A 吃的是 T=6 那一列 × "
            f"8 条流（{len(p_cells)} 格，p={p_hits}）⇒ "
            + (f"两把尺子在这一趟**并存且不同判**（p ≥ 1 而 q = 0）：『(u) 没量过』的主语**永远是 T=1 那一列**，"
               f"不是『这一发什么都没量到』；反过来 p 那 {p_hits} 格也不把出口抬成 (a)——事后改分母等于挑"
               f"对自己有利的那一把（R1：读法 A 与段② 各住在自己的键里）"
               if p_hits and not q_hits else
               f"两把尺子在这一趟没有分叉（p={p_hits}、q={q_hits}），不存在并存形状的处置问题"))

    report = {
        "dataset": DATASET,
        "dataset_applicability": (
            f"任务 = {DATASET}。这一本名下三张表的**现读长度**：α→0 阶梯 {len(books)} 档、3 流 α×T 网格 "
            f"{len(grids)} 档、8 流 α×T 网格 {len(grids8)} 档 ⇒ 读法 A / 段② / 闸 VI 谁不适用由这三张表"
            f"自己说（空表才走『无从开口』那一支），**不由任务名说**；而 MNIST 那本在册账没有 `dataset` "
            f"这一位（记于任务位上线之前），对照它时按历史口径读成 mnist"
            if DATASET != "mnist" else
            "本发在册三张账表（阶梯 / 3 流网格 / 8 流网格）全部落在这一个任务上，无一格跨任务"),
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
                  "闸VI_共流嵌套性": (f"8 流两对 vs 同档 3 流两对：共有流 × 共有格共 {stats['nest_rows']} 行 × "
                                     f"3 字段逐位相同（红 {len(red6)} 条）；0 行算『没测到』不算通过"
                                     f"（段② 的前置·甲）" if not vi_na else
                                     f"**不适用**（任务 {DATASET}）：这一本名下两族都没有 8 流网格账，"
                                     f"甲前置没有原料 ⇒ 不记红也不记通过（写成『比了 0 行、红 0 条』会让读者"
                                     f"以为闸跑过了）"),
                  "闸V_来源": "每个数带 (档案名, 格名, 流名单)，见各格 `provenance` 与 `offset_by_seed`"},
        "reading_A_paired_T6_8flows": read_a,
        "reading_A_applicability": (
            f"参评 {len(read_a)} 档（登记处 {len(books)} 档）× 每档取两本账的**共有** α ⇒ 共 {len(p_cells)} 格，"
            f"α 集合 {sorted({float(k.split('=')[1]) for t in read_a.values() for k in t})}；"
            f"MNIST 那三对算出来是 9 格（登记处三档 × α∈{{0.05,0.3,0.7}}），"
            f"分母不同 ⇒ p 的读数**不许跨任务横比**，只能各自对着自己的分母读"
            if read_a and len(read_a) == len(books) else
            f"**不适用**（任务 {DATASET}）：读法 A 吃的是同一坐标两族各自的 α→0 阶梯账，"
            f"登记处在这一名下是空的（{len(books)} 档）⇒ 上面那张表的意思是『无从开口』，"
            f"不是『测过了且 p=0（即测不出跨族差）』；p 的分母因此是 0，不参与任何计数"),
        "reading_B_offset_vs_T_3flows": read_b,
        "reading_B2_offset_vs_T_8flows": read_b8,
        "section2_power_check": {"buys": "上面 docstring 落地⑧ 第 3 笔：q=0 里『效应为 0』与『3 条流测不动』分不开",
                                 "q8_T1_cells_out_of_zero": q8_hits,
                                 "denominator": (f"{denom8} 格（{len(grids8)} 档 × α∈{{0.3,0.7,1.0}} 的 T=1 列）"
                                                 + ("——不是读法 B 的 9 格，中数据档两族都没有 8 流网格"
                                                    # ↑ MNIST 那一支保持**字面**：在册那本 v2 档案里就是这句话，
                                                    #   改成现读会让盘上那本变成代码复现不出来的东西。
                                                    if DATASET == "mnist" else
                                                    f"——不是读法 B 的 {len(grids) * len(GRID_ALPHAS)} 格，"
                                                    f"缺 {'、'.join(sorted(set(grids) - set(grids8)))} 那几档的"
                                                    f" 8 流网格") if grids8 else
                                                 f"**不适用**（任务 {DATASET}）：0 档有 8 流网格 ⇒ q8 无从定义，"
                                                 f"这里的 0 不是『六格全含 0』那个意思"),
                                 "half_width_ratio_8_over_3": hw_ratio,
                                 "half_width_ratio_median": med_hw,
                                 "ex_ante_expected_ratio": round(T95[8] / T95[3] * (3 / 8) ** 0.5, 3),
                                 "precondition_甲_嵌套": ("不适用（本任务没有 8 流网格账，等于『没有原料』，"
                                                        "不等于『过了』或『不过』）" if vi_na else
                                                        f"{'过' if 甲 else '不过'}（比了 {stats['nest_rows']} 行、"
                                                        f"红 {len(red6)} 条）"),
                                 "precondition_乙_形状互锁": (f"闸 IV 已含 8 流两对（{nine} 格逐位相同）" if grids8 else
                                                            f"不适用：闸 IV 只锁到 3 流那 {nine} 对，"
                                                            f"本任务名下没有 8 流那两对"),
                                 "precondition_丙_半宽变窄": ("不适用（同上，没有可比的相邻两本）" if not grids8 else
                                                            f"{'过' if 丙 else '不过'}（逐格 {hw_ratio}）"),
                                 "exit": section2, "verdict": section2,
                                 "T6_column_8flows_descriptive": t86_hits,
                                 "not_a_re_verdict": ("R1：这一段不重开出口——`exit`/`verdict` 两个顶层键与 v1 逐字"
                                                     "相同，段② 只住在这一个独立键里" if DATASET == "mnist" else
                                                     "R1（换任务口径）：段② 的判词只住在 `section2_power_check` 里，"
                                                     "不回头改顶层 `exit`；顶层那一格在本发是 (a)/(b)/(m)/(u) 四格，"
                                                     "由 `task_switch_crosscheck` 的计数决定"),
        },
        "counts": {"p_A_cells_out_of_zero": p_hits, "p_A_cells_total": len(p_cells),
                   "q_B_T1_cells_out_of_zero": q_hits, "q_B_T1_cells_total": len(grids) * len(GRID_ALPHAS),
                   "B_T6_cells_out_of_zero_descriptive": t6_hits,
                   "q8_B2_T1_cells_out_of_zero": q8_hits,
                   "q8_B2_T1_cells_total": denom8,
                   "nested_rows_compared": stats["nest_rows"]},
        "shape_identity": {"两族软标签体检逐位相同": f"{nine}/{(len(grids) + len(grids8)) * len(TEMPS)} 格"
                                                   f"（{len(grids)} 对 3 流 + {len(grids8)} 对 8 流 × T∈{{1,3,6}}）",
                          "老师在这个拟合集上的精度": acc_archive},
        "task_switch_crosscheck": (xtask if xtask else
                                   "不适用：本趟任务就是对照基准那一侧（`reading_B_offset_vs_T_3flows` 那 9 格"
                                   "是被比的那一份），换任务那一趟读的是另一本档案"),
        "v2_increment": ("v2 相对 v1 只多了段② 那几位（reading_B2_offset_vs_T_8flows、section2_power_check、"
                         "counts 里三枚 q8/嵌套计数、gates 里的闸VI）＋闸 IV 的参评对数（3 对→5 对）与由此带动的"
                         "活计数（II-a 行数 / II-b 次数 / 本数）。`quantity`、`pays`、`sign_convention`、`exit`、"
                         "`verdict`、`reading_A_paired_T6_8flows`、`reading_B_offset_vs_T_3flows` 与 v1 逐字段相同"
                         "⇒ 除这些位之外任何一处不相同，就是这段升版把判决碰着了"
                         if DATASET == "mnist" else
                         "本发（换任务）新增的位：`dataset`、`dataset_applicability`、`task_switch_crosscheck`、"
                         "两处 applicability 文案与 gates 里闸VI 的不适用分支。读法 B 那一格的代码路径与 MNIST "
                         "那本逐字相同（同一个 `read_B`），**没有 v1 可比**——MNIST 侧的 v1→v2 那条声明指的是 "
                         "`compress_report_v4l_soft_shape_family_v2.json`，与本本无关"),
        "exit": exit_rule, "verdict": exit_rule,
        "cost": {"arm_trainings": 0, "total": 0, "api_fee": 0,
                 "note": (f"纯读档 + 一次老师前向：{n_books} 本在册账的原始错图数逐流相减，不训练任何学生。"
                          f"但段② 不是免费的：它吃的三本 8 流 v4c 网格（`compress_report_v4c_cnn_n8.json` / "
                          f"`compress_report_v4c_cnn_low_n8.json` / `compress_report_v4c_low_n8.json`）"
                          f"是 step4c 那 88×3 = 264 次训练换来的，训练数记在那一发，不记在本脚本"
                          if DATASET == "mnist" else
                          f"本趟同样 0 次训练、零 API 费：{n_books} 本 Fashion 在册账（阶梯 {2 * len(books)} 本 + "
                          f"3 流网格 {2 * len(grids)} 本 + 8 流网格 {2 * len(grids8)} 本）逐流相减 + 一次 Fashion "
                          f"老师前向（{teacher_pt.name}）。对照基准 {MNIST_V2.name} 是读档，不重算。"
                          f"**账本不是免费的：{n_books} 本分四发买（第廿八发 D 那 234 次 + 第卅二发 D 那 51 次 + "
                          f"第卅三发 D+E 那 294 次 + 第卅四发 B 那 51 次），训练数记在 step4b/step4c/step4f 那几发，"
                          f"不记在本脚本**"),
                 "paid_elsewhere": (None if DATASET == "mnist" else
                                    "第廿八发 D：6 本 v4b 底 × 6 次 = 36 ＋ 6 本 v4c 网格 × 33 次 = 198 ⇒ 本趟读的"
                                    "那 12 本账付 234 次；**但那一发实付 250 次** —— 多出的 16 次来自预检件"
                                    " `_preflight_fashion.py` 的 [6] 腿：它本意是'零训练地真跑一次 CLI'，"
                                    "却因盘态判窄（只枚举了两种盘态）在第三种盘态下真去训练了，产出一本计划外的 "
                                    "`compress_report_v4b_fashion_n8.json`（fashion × MLP × 满数据档 × 8 条流，"
                                    "2 个臂 × 8 条流 = 16 次）。**到现在仍然不读它**：本轮把满数据档那一对登记进读法 A "
                                    "时现读过两本账自己记的 `floor_from`——MLP 侧那本 `compress_report_"
                                    "v4f_fashion_n8_thin.json` 声明的底是 `compress_report_v6_fashion.json`"
                                    "（第6课那本），CNN 侧那本新阶梯声明的是 `compress_report_v4b_fashion_cnn_n8.json` "
                                    "⇒ 两边都不是那本计划外的 `..._v4b_fashion_n8.json`，它依旧是"
                                    "**不进任何判据的孤儿**（这句是现读字段，不是推测）。"
                                    "第卅二发 D：1 本 v4b 底 16 ＋ 1 本 v4f 薄阶梯 35 = 51（低数据档那一对）；"
                                    "第卅三发 D+E：3 本 v4b 底 48 ＋ 2 本 v4f 薄阶梯 70 ＋ 2 本 v4c 8 流网格 176 "
                                    "= 294；第卅四发 B：1 本 v4b 底 16 ＋ 1 本 v4f 薄阶梯 35 = 51（中数据档 MLP "
                                    "侧那两本 ⇒ 这一档的阶梯走的是新加的 `--from-v4bc` 那条按族分册的读账路由，"
                                    "因为 step6 那本按档合并的旧账只登记满数据／低数据两档，中数据档在那把尺子下"
                                    "没有底可读）")},
        "limitations": [
            "两把尺子不许横比：读法 A 是 8 条流、读法 B 是 3 条流，σ̂ 与 CI 半宽都不同；"
            "B 只回答『偏移跟不跟 T 走』，不回答『偏移是多少张』——后者只由 A 报",
            "『形状同一份』由闸 IV 钉（读档 + 现算）。它排除的是『两族读的不是同一批软标签』这一支，"
            "不排除『同一份形状对不同架构作用不同』——恰恰后者是本发的读数对象",
            ("本发不新增任何独立性维度：七维（档/宽/族/任务/老师/温度列/流数）一维都没换，"
             "换的是**归因**（把点估计买成配对读数）⇒ 既不叫复证也不叫翻案" if DATASET == "mnist" else
             "本发换的正是七维里的**任务**那一维（档/宽/族/老师/温度列/流数都没换）。但它**不叫复证**："
             "复证要的是同名坐标上的两本对照，这里是给同一个读数买第二个任务的取值 ⇒ 一条跨任务边界读数，"
             "第廿五发那句归因既不因此扩界也不因此作废（扩与不扩由 `exit` 那一格说）"),
            "读法 B 的温度只有三个点且带并列 ⇒ 这里不报相关系数：3 点、带 ties 的 Spearman ρ 在任何"
            "预登记口径下都够不着判定门槛，报了就是给噪声配一个数字",
            ("出口 (a) 只数『CI 不含 0 的格数』；不含 0 的格落在哪一档、三档是否同号，全部写在 "
             "`reading_A_paired_T6_8flows` 里由读者自己看，本发不据它另立一句判词" if DATASET == "mnist" else
             "出口那一格只数计数（n_hit / n_agree / n_mag）；不含 0 的格落在哪一档、两任务各是多少张、CI 端点，"
             "全部写在 `task_switch_crosscheck.cells` 里由读者自己看，本发不据它们另立一句判词"),
            ([] if DATASET != "mnist" else
             [f"阶梯三对里有 {len(v4_mlp)} 对的 MLP 本记的是 v4 时代的判据（{'、'.join(v4_mlp)}）：它们的 "
              f"`student_arch` 字段是缺的 ⇒ 家族位这件事只由文件名定；本发只取它们的原始错图数"
              "（`wrong_ce`/`wrong_kd`/`gap_imgs` 与判据版本无关），**不取它们的判词** ⇒ 谁引用这一格的形状判词，"
              "不许把 v4 账与 v5 账混作一版（档案里没有'这本在 v5 之下怎么判'的字段，守卫也没替它复算过）"])
            + ([f"读法 A 在 Fashion 名下登记了 {len(books)} 档，其中 {len(no_arch_ladder_mlp)} 本的 MLP 侧"
                 f"没记 `student_arch`（{'、'.join(no_arch_ladder_mlp)}）、{len(no_rule_ladder_mlp)} 本连 "
                 f"`verdict.shape_rule` 都没记（{'、'.join(no_rule_ladder_mlp) or '无'}）⇒ 上面那句"
                 f"『v4 判据 vs v5 判据』的债**已经落到 Fashion 这一本上了**：家族位这件事在这一档只由文件名定。"
                 f"本发只取原始错图数、不取判词，谁改名谁断链"]
               if DATASET != "mnist" else []),
            *([] if DATASET != "mnist" else [
                "段② 只把**流数**这一维从 3 加宽到 8，档/宽/族/任务/老师/温度列一维都没换 ⇒ 它加强的是"
                "读法 B 那条结论的**功效**，不是它的外部效度；而且 q8 的分母是 6 格（中数据档两族都没有 8 流"
                "网格）——与 q 的 9 格不许比大小，只许各自报数",
                "(i) 那一支只**削弱**不证伪『3 条流测不动』：8 条流仍可能功效不足（判据只要求半宽比 < 1，"
                "没要求窄到能测出某个预先指定的效应量）⇒ 谁要把 q=0 读成『T=1 处跨族差为 0』，"
                "那是把『更宽的尺子也量不出』读成『量为 0』",
            ]),
            *([] if DATASET == "mnist" else [
                "两个任务的格数不许相加：Fashion 的 9 格与 MNIST 的 9 格各配各的流、各配各的老师，"
                "`task_switch_crosscheck` 里只有『两条各自成立的 CI』与『同号与否』，"
                "没有差之差的 CI ⇒ 谁把 n_hit 写成 18 格里的命中数，那是把两把尺子当一把",
                "『量级相当』在本发只有这一个可核形式 = MNIST 该格的**点估计**落在 Fashion 该格的配对 CI 内；"
                "它不等于两任务的 CI 重叠（那是更宽的要求），也不等于斜率相同 ⇒ 出口 (m) 就是为"
                "『方向跟、高度不跟』那一格留的，不许把它读成 (a) 或 (b)",
                f"段② 在 Fashion 名下**只有一档**（{len(grids8)}/{len(grids)}：低数据档 8 流网格是第卅三发 "
                f"D+E 那 176 次换的，中数据与满数据两档仍是 3 流）⇒ q8 的分母 {denom8} 格与 q 的 "
                f"{len(grids) * len(GRID_ALPHAS)} 格不许比大小；而当初不买满三档的理由未变 = 第廿七发 (u) "
                f"实测『补流数 = 买分辨率』有反例（六格半宽比 0.602~1.441、两格 ≥1），88 次/本不保证买到更窄的区间 "
                f"⇒ 读法 B 那三档里没被加宽尺子兜底的两档，测不出仍只能记成『没量过』",
                "MNIST 那 26 本在册账一个数字都没重测：本发只新增 Fashion 名下这 12 本（6 底 + 6 网格）的读数 ⇒ "
                "『跨任务』这句话的对照侧永远指向盘上那本 v2（按内容核身份，见 `read_baseline`），"
                "不是指向一个新算出来的 MNIST 数",
            ]),
        ],
        "red_modes": ({"--tierblind": "把低数据档的 MLP 阶梯换成中数据档那本（流名单仍相同）⇒ 只有闸 III 该红，"
                                      "且闸 I 必须全绿（两腿分离）",
                        "--unpair": "把配对里 CNN 那一侧整体错开一位 ⇒ 只有闸 II-b 该红；II-a、闸 I、闸 III 全绿"
                                    "（实测不变的只有**点估计与两族各自的逐侧均值**——CI 端点会跟着配对移动，"
                                    "所以看得见它的仍然是逐流序列；预登记那句'CI 宽度不动'是断言过头，见落地⑥）",
                        "--nest": "（v2 新增，钉闸 VI）把满数据档 8 流 CNN 那本 `match_T6` 里种子 42 的 "
                                  "`wrong_ce` 在内存里 +1 ⇒ 只有闸 VI 该红：那一格既不进读法 A/B 的 9 个 α×T 格"
                                  "（所以 II-a 看不见它），也不动档/宽/族/流名单（所以闸 I、III 看不见它）"}
                       if DATASET == "mnist" else
                       {"本任务名下不可跑": f"三档注入点（阶梯错档 / 配对错位 / 8 流 `match_T6` +1）全部写在 MNIST "
                                            f"那三张账表上，任务换成 {DATASET} 之后那些本根本不参评 ⇒ 硬跑只会安静地"
                                            f"红在别处，那不算拆红。`main()` 在这一趟遇到 `--tierblind`/`--unpair`/"
                                            f"`--nest` 会当场退出（零训练脚本没有'重跑一遍'的代价，所以拒得起来）；"
                                            f"闸 III / II-a / II-b / 闸 IV 这四道在本本上是**真跑过**的，"
                                            f"它们吃的注入与 MNIST 那本是同一份代码路径"}),
    }
    check_report_shape(report)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（0 次训练、零 API 费）· {exit_rule[:90]}…")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="第4课l（零训练）：跨族偏移拆给形状还是拆给交互"
                                             "（v2 多段②：把读法 B 的 T=1 那一列搬到 8 条流的尺子上；"
                                             "第廿八发 D 加任务位：同一句问话搬到第二个任务）")
    ap.add_argument("--tierblind", action="store_true", help="把低数据档的 MLP 阶梯换成中数据档那本 ⇒ 只红闸 III")
    ap.add_argument("--unpair", action="store_true", help="把配对里 CNN 那一侧错开一位 ⇒ 只红闸 II-b")
    ap.add_argument("--nest", action="store_true",
                    help="给 8 流 CNN 那本 match_T6 的种子42 的 wrong_ce 内存里 +1 ⇒ 只红闸 VI（共流嵌套性）")
    ap.add_argument("--dataset", choices=DATASETS, default=DATASET,
                    help="读哪一组任务的在册账（choices 来自登记处，不在这里另列名单）。"
                         "非 mnist 时三档拆红一律当场退出——注入点写在 MNIST 的账表上，那些本不参评")
    a = ap.parse_args()
    main(red="--tierblind" if a.tierblind else "--unpair" if a.unpair else "--nest" if a.nest else "",
         ds=a.dataset)
