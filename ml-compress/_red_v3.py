# -*- coding: utf-8 -*-
"""v3/v4/v5 形状判据的离线验红（零训练）：把已入库的 v4f 档案按 v1/v2/v3/v4/v5 各算一遍，看翻转集合落在哪。

十三道闸（沿用第四发的规矩：一道闸只钉一件修复，红要知道是哪件修复没装上；
  ⑪ 是第二十发的"口径等价闸"，⑫ 是第二十一发的"档案文本闸"，⑨ 的家族位也在第二十一发接上——换模型族那一发；
  ⑬ 是本轮菜单 C 补上的"入口闸"——它判的是脚本能不能打开，不判读数）：
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
                 **第二十一发两处升级**：① "同配置"现在含**家族位**（`student_arch`）——CNN 那两本与
                 MLP 同档同宽的账不再是"重跑本"，跨族当成重跑会把架构差异报成漂移；
                 ② 那句"已重跑 N 本"的计数换成**守恒账**：`候选 = 逐位核过 + 首跑/无更低版本前账`，
                 两个数由同一个循环产出、不平即红（上一版这里印"已重跑 25 本"，而同一轮里有 19 本
                 被明确打印成"该配置首跑"——那句计数在说谎）。专属拆红档 `--g5drop`。
  ⑥ 扩流/扩格不改老读数 两本同配置账之间，**谁老是新由流集合/阶梯的严格包含决定，不看文件名**
                 （`_n12` 字典序在 `_n8` 前面，按文件名定向会结构性地漏掉"8 条流 ⊆ 12 条流"这一对）：
                 小账的每一格必须在大账里逐位重现。这一条是"3/8/12/16 条流的账能对着读"的**唯一许可证**——
                 本轮所有"换流数后判决变了/没变"的说法都默认了它，此前只是口头默认。
                 实测在册 **27 对**（现读，本轮 +3：`_mid_n8_t6_v5`（第二十发同流重跑本）接到
                 `_mid_n12_t6_v5` 与 `_mid_n16_t6_v5` 两本 ⇒ +2（它与 `_mid_n8_t6_v4` 是**同一批 8 条流**、
                 不构成严格包含 ⇒ 按定义不配对，那条对账线归 ⑤）；
                 第二十一发买到**跨族第一对** `_cnn_mid_t6_v5`（3 流）⊆ `_cnn_mid_n8_t6_v5`（8 流）
                 ⇒ +1（15 格逐位）。前几轮：第十四发两本 16 条流的账入库后 +7、第十五发 S64 × 中数据档
                 入库后 +1 = `_s64_mid_v5`（3 流）⊆ `_s64_mid_n8_v5`（8 流）那 **15 格**逐位相同——
                 S64 那副学生在中数据档的"3 条流与 8 条流能对着读"从此也有盘上凭据了；
                 `_mid_n16_v5` 一本就同时是
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
  ⑨ 底不许跨档/  第十五发查出那处定向缺陷的常驻对账线：`v4b_path` 旧写法只在 `width == 128` 时贴档位后缀
     跨宽/跨族借 （写死的假设"S64 只跑过满数据档"）⇒ S64 × 中数据档的新账会去借**满数据档**那本底来判自己的
                 闸门，而 `load_anchor` 一路在拦的那件事这次拦不到（那本档案里真有一个"满数据8500 种子42"
                 的臂，只是档不对）。这一道不判数字、只判**读档来源**：对每一本 MNIST 的 `compress_report_v4f_*`
                 现读 `floor_from`，要求它恰好等于 `v4b_path(该书 width, 该书 tier, 该书 student_arch)`。
                 （第二十一发之前这里是两个入参。）
                 为什么放在守卫而不是预飞脚本：它要能在每一轮加账后自动重读，而预飞只在开跑前跑一次。
                 实测参评 **23 本**全等（第十五发 +3；第十七发 +1；**第二十一发 +2 且新接一条腿**：
                 `v4b_path` 现在按 `(width, tier, family)` 三维定向，档案名里的家族位（`_cnn`）必须与
                 `student_arch` 一致 ⇒ 闸⑨ 从此既拦"借别的档的底"，也拦"借别的族的底"。
                 配拆红档 `--borrowfloor`（把档位后缀回落成旧写法）⇒ 本轮实测 **3 → 5 条红**
                 （三本 S64 中数据账照旧 + **两本 CNN 中数据账各一条**：旧写法连"这一族"这一维都不认，
                 摘掉后缀后它们一起被点名"底读错了哪本"）。
                 另配 `--famblind`（只摘家族位）⇒ 实测 12 条红：⑨×4（两本 CNN 账各两条腿：
                 名字与架构不同族 + 底回落成 MLP 那本）＋ ⑤×2（跨族账被当成重跑本，判词与训练路径一起红）
                 ＋ ⑥×6（跨族被当成同配置 ⇒ 15 格逐位对账全红）。
  ⑩ 名单与系数只  钉的是"并成一份之后不许再漂回两份"：模块侧逐个**现读**它自己那份 SEEDS 与缓存的
     有这一份     系数（step5.D4、step7.T95），按值要求等于登记处 `FLOW_SEEDS`/`D4`/`T95` 里同一档的
                 数；档案侧要求凡是记了自己名单（`seeds`/`seed_order`/`seed_pools`）的账，名单必须
                 逐位等于登记处那一档。为什么按值不按 import 行：把 `SEEDS = flows(8)` 抄成一份字面量
                 之后，源码里既没有别的文件的名字也没有可疑 import，文本检查全绿，而"同一批随机流"
                 这句话已经悄悄只在 7 条流上成立。实测**模块侧 10 项、档案侧 80 条名单全等**（+6：
                 第二十一发两本 CNN 账各带三种名单字段）；
                 配拆红档 `--flowcopy`（名单漂一条 + 系数还留着 n=3 那档）⇒ 两条腿各红 1 条。
  ⑪ v5 那条腿的 第二十发装上的"口径等价闸"：那条腿的分母写成 CI 半宽，当时只有后果、没有理由。
     口径等价与  四条腿常驻核：① 每条配对步长的 CI 中点离自己的点估计在容差内（= 区间对称，等价的前提）；
     双记账       ② 那条腿的分子与步长的点估计必须是同一个数；③「|出界| > 半宽」与「CI 不含 0」逐本同判；
                 ④ 档案必须自带 `verdict.ci_leg_reading` 双记账、并与守卫复算逐字段核。
                 **参评条数为 0 算红不算通过**（老规矩：没测到不许记成通过）。
                 实测**参评 29 本、其中 3 本自带双记账**；两档专属红 `--ciasym` **2** 条、`--ciunread` **1** 条。
                 ④ 为什么写成"至少一本"：这条腿的口径此前只被守卫自己核过，产出它的那段训练代码没把
                 "我在测什么"写进档案 ⇒ 修复没落到盘上（`--ciunread` 钉的正是这一句）。
  ⑫ 档案文本闸（  第二十一发装上的，两条腿，钉的都是"这本账写给后来人的那句话还成立吗"：
     未渲染占位符   腿 A 全册递归展开每一条字符串字段，读到 `{...}` 形状（且里面不像代码）就是**漏了 f 前缀**
     与种子数对账）  ——本轮真实事故：`step4b_seed_variance.py` 那句 `limitations` 少一个 f，于是五本已入库档案
                   里印着字面量 `{len(seeds)}`，**没有任何一个数字因此变红**，坏掉的只是"该信到什么程度"那句。
                 腿 B 核那句"种子底由 N 个点的极差估"里的 N 是否等于**该本自己的** `len(seeds)`——
                   钉的是"补渲染时把别的账的数填进句子"那一类（腿 A 抓不到：它不留花括号）。
                 实测**全册 80 本 / 展开 2872 条字符串字段 / 8 本带那句，N 与各自 `len(seeds)` 全部相等**。
                 专属拆红两档：`--unformatted`（把带那句的账退回漏 f 形状）红 **8** 条、
                 `--wrongcount`（只把 N 改成 N+1、不留花括号）红 **1** 条，并且后者额外验**两腿分离**
                 （注入错数时腿 A 一条都不许红，否则这一档只是 `--unformatted` 的重复）。
                 **反面要如实说**：本轮"修漏 f 的脚本"自己误伤过三本没坏的账（措辞从"欠估"变成"仍是欠估"）——
                 那一处不留花括号、也不改数 ⇒ 腿 A、腿 B 都抓不到它，这类只改措辞的误伤当前只有
                 `git diff` 加人眼这一道防线。**别把它记成有闸。**
  ⑬ 入口闸（每个  带开关的脚本跑一次 `--help`，退出码必须是 0。钉的是第十八发撞到、此后一直挂着的那笔仪器债：
     带开关的脚本  `step5_second_teacher.py --help` **自那行帮助文本写下起就没通过过**（argparse 把 help 整串
     都必须打得开）做 `%` 插值，帮助文本里写 `50%` 必须写成 `50%%`），而守卫是 import 模块、不走 argparse
                 ⇒ 前十二道闸对"打开就炸"零反应：它不在任何训练路径上，只在"想知道这个脚本有哪些开关"那一步。
                 被测清单**按值现读**（源码里出现 `add_argument` 就算），新加一个带开关的脚本自动进清单。
                 专属拆红 `--helpblind`：生成一个只有 argparse 的**合成**脚本、help 串里放一个裸 %
                 （与那处崩溃同形状）⇒ 实测红 **1** 条，且打印明写"注入体不在真账清单里"——
                 这一档证明的是检测器会对 rc≠0 出声，真账现在全绿由上面那行绿色计数说。
                 **代价如实记在这里**：`--help` 要走一遍 torch 的 import，单次实测 **6.3 秒 / 本**；
                 11 本、4 并发下这一档两次实测 **24.7 / 18.2 秒**（同机、同清单，差在并发争抢），
                 守卫整轮从上一版的 18.2 秒涨到实测 **31~37 秒**。

自我拆红（每条各拆一道闸，跑了不红的模式=那条闸没测到，不许拿它当通过。下面全是**本轮实测**，
不是预期——规矩是实测再写回这张表，写死的计数只对"当前在册的那几本"成立：

**【第二十三发·CNN 第二坐标（低数据2000 × T=6）＋第二十二发（step4j）一并入库】在册 34 本（v4f 名下，+1）、
全册 86 本（+6：`v4b_cnn_low`、`v4b_cnn_low_n8`、`v4c_cnn_low`、`v4f_cnn_low_n8_thin_t6_v5`、
`v4i_mechanism_v2`、`v4j_matched_n8`），正常模式 rc=0 / 0 条红，二十一档全部重跑（逐档现读红数）**：
misfeed **99**、decorative 27、collateral **15**、v4blind 13、v5strict 13、famblind **16**、
unformatted **10**、v4loose **8**、drift 5、borrowfloor **6**、v5blind 4、sibdrift 3、laddersib 2、
flowcopy 2、ciasym 2、oldsib 1、g5drop 1、ciunread 1、wrongcount 1、helpblind 1、destroy 0。
**与下面那张【第二十一发】表相比：涨六档、十四档一位没动、`--destroy` 照旧 0 红（只算灵敏度）。**
先记第二笔诚实账：**那张表到现在又腐烂了两批发**（第二十二发入库的是 `v4j` 与 `v4i_v2`，两本都不在
`compress_report_v4f*` 这条 glob 里 ⇒ 判据侧计数原地，但 ⑫ 的"全册"分母从 80 涨到 86）
⇒ "加账就得整张重测"这条老教训第 N+1 次兑现，而**没重测的表不是旧数据，是假数据**。
逐档归因（现读 grep 失败名单、按本点名，不猜）：
- **涨：misfeed 94→99（+5）——这五条全部点名新那本 `_cnn_low_n8_thin_t6_v5`**，而且它**推翻了下表那条盲区**：
  第二十一发写的是"CNN 账在 `--misfeed` 上零红 ⇒ ① 在 CNN 上不承重"。真相是那句话的适用范围是
  **那两本的判词**（『有阈值』）：反号喂进『有阈值』的账，复算还是『有阈值』；而这一本判『污染型』，
  反号喂进去复算成『有阈值 / 测不动』⇒ ① 当场点它五次名（v1~v4 并排列各一次 + 本发判词一次）。
  **可迁移：一条"某类账不承重"的盲区断言，如果它的证据只是同类账的样本，那它判的是那几本的判词，不是那一族。**
- **涨：famblind 12→16（+4）** —— ⑨ 由 4 条变 6 条、⑤ 由 2 条变 4 条，各 +2：新那本 CNN 账在摘掉家族位后
  同时被"名字与 `student_arch` 不同族"和"底回落成 MLP 那本"两条腿点名，并被当成 MLP 重跑本（⑤ 的判词与
  训练路径各一条）。⑥ 仍 6 条：这本**没有**少流的同族老账可被严格包含（`--nseeds 3` 预演账按预登记删掉了）
  ⇒ 闸⑥ 许可证 **27 → 27 对，一对没多**（"对数与买了哪一维无单调关系"第七次现读，这次落在 +0 上）。
- **涨：unformatted 8→10（+2）** —— 两条都是注入腿：新入库的 `v4b_cnn_low` 与 `v4b_cnn_low_n8` 各带那句
  『种子底由 N 个点』⇒ ⑫ 的"带那句"名单从 8 本涨到 **10 本**，正常模式下这 10 本的 N 与各自 `len(seeds)`
  全等（现读"对不上的有 0 本"）。这一档的红数涨**只说明被测对象变多**，不说明现在有人漏 f。
- **涨：v4loose 7→8（+1）** —— 松腿第一次落在**换族 × 换档**那一侧的界内账上：这本份额 57%、档案记『污染型』、
  "忘掉同号且更大"一装上就复算成『污染型·倒挂』⇒ 那条腿的被测对象维度集合又长一个。
- **涨：collateral 14→15（+1）** / **涨：borrowfloor 5→6（+1）** —— 前者是新那本的 v3 并排列（"第 2 支挪前＋
  忘掉反号"会把它读成反号线）；后者是 ⑨ 那条读档来源腿第一次点到 CNN 低数据档那本（旧写法连档都不认）。
- **十四档没动**（decorative 27、v4blind 13、v5strict 13、v5blind 4、drift 5、sibdrift 3、laddersib 2、
  flowcopy 2、ciasym 2、oldsib 1、g5drop 1、ciunread 1、wrongcount 1、helpblind 1）：本轮没换判据、没动流名单、
  没改系数册 ⇒ 按老规矩这句**不许当通过**，它只说明这几档的被测集合本轮没变大。
**其余各闸现读（全部本轮实测，替代下表同名数字）**：闸① 复算 **34/34** 本通过；闸④ 板块在册『污染型』
**22 → 23 本**逐字保持；闸⑤ 候选 **26 本 = 逐位核过 6 + 首跑/无更低版本前账 20**（新那本进的是**首跑**那一列，
不误伤）；闸⑥ **27 对**；闸⑨ 参评 **23 → 24 本**；闸⑩ 模块侧 10 项、档案侧 **80 → 86 条**名单全等；
闸⑪ 参评 **29 → 30 本**、其中自带双记账的 **3 → 4 本**（新那本写了 `ci_leg_reading`，九项与守卫复算逐字段相同）；
闸⑫ 全册 **86 本 / 展开 3049 条字符串字段 / 10 本带那句 / 0 本对不上**；闸⑬ 现读清单 **11 → 12 本**、
退出码 0 的 12 本。**墙钟要如实报**：同一档（⑬）本轮三次实测 **56.8 秒 / 30.9 秒 / 19.6 秒**（同机、同清单、4 并发；
最后一次整轮 25 秒、rc=0），比下表记的 24.7 / 18.2 秒又宽了一倍 ⇒ 这一档的成本**不是常数**，写死的秒数和写死的计数一样会腐；
本轮只报"三个可核读数"，不报倍数。
闸⑧ 现读：v5 名下判『倒挂』的账 **4 本 / 归并 1 个坐标**（`mlp族 × 中数据4500 × S128 × T=6.0`）——
**与下表同值、没涨**，因为新那本 CNN 界内不进名单 ⇒ 那是"唯一坐标"这句话的第八次现读复核。

（下表【第二十一发】是**过去时**：它记的在册 33 本、misfeed 94、famblind 12、unformatted 8、v4loose 7、
borrowfloor 5、collateral 14、⑥ 27 对、⑨ 23 本、⑩ 80 条、⑪ 29 本/3 本双记账、⑫ 80 本/2872 字段/8 本带那句、
⑬ 11 本 都已被上面那张表替代；留着是因为它的**逐档归因**（尤其那条 misfeed 盲区）是被第二十三发推翻的对象，
删掉它就没有人知道自己错在哪。**其中"misfeed 对 CNN 账零红 ⇒ ① 在 CNN 上不承重"那一句已作废**，
作废的理由写在上面第一条归因里。）
**【第二十一发·换模型族】在册 33 本，正常模式 rc=0 / 0 条红，二十档全部重跑（逐档现读红数）**：
misfeed **94**、decorative **27**、collateral **14**、v4blind **13**、v5strict **13**、famblind **12**（新）、
unformatted **8**（新）、v4loose 7、drift 5、borrowfloor **5**、v5blind 4、sibdrift 3、laddersib 2、
flowcopy 2、ciasym 2、oldsib 1、g5drop **1**（新）、ciunread 1、wrongcount **1**（新）、destroy 0。
**与这张表上一次写死的数字（第十八发）相比：涨五档、新增四档、十一档一位没动。**
先记一条诚实账：**上次那张表到现在已经腐烂了两批发**（第十九发零训练没入库、第二十发入库一本 v5-era
落地本），所以"88 → 94"这六条不是本轮一发造成的，本轮只新增两本 CNN 账——**这正是"写死的拆红表会自己
过期"这条老教训第 N 次兑现：加账就得整张重测，不重测的表不是旧数据，是假数据。**
逐档归因（现读 grep 失败名单，不猜）：
- **涨：decorative 21→27** ——六条里有 **3 条点名 `_cnn_mid_t6_v5`**（①：v2/v3/v4 三个并排列在瞎掉阶梯顶
  之后全复算不出来）。这是本轮唯一一处"CNN 新账直接抬高老档红数"的地方。
- **涨：borrowfloor 3→5** ——多出的两条正是两本 CNN 中数据账：`v4b_path` 一旦回落成旧写法，
  它连"这一族"这一维都不认 ⇒ 闸⑨ 的那条"读档来源"腿第一次跨到族上。
- **涨：misfeed 88→94 / collateral 13→14 / v4blind 12→13** ——**这三档的失败名单里 CNN 两本一次都没被点名**
  （现读：`grep 'cnn_' 验红段` = 0 条）。misfeed 那六条按本点名可归因的是 `_mid_n8_t6_v5` 被 ① 点满五次
  （第二十发的落地本，v5-era 机制照旧），余下的一条我没拆到逐本级 ⇒ **记成"未完全归因"，不许记成已解释**。
  **CNN 两本在 `--misfeed` 上零红是一条要写下来的盲区**：它们档案判词是『有阈值（H1 半活）』，
  把伤害方向反号喂进去复算仍是这一句 ⇒ ① 这条"读错数字当场被抓"的腿**在 CNN 账上不承重**。
  按第十五发那条规矩处理：**没有一档红会因它而红 ⇒ 它不进守卫，只进文档**（见 README 第二十一发）。
- **新增 famblind 12 / g5drop 1 / unformatted 8 / wrongcount 1**：各钉本轮一处修复（家族位、⑤ 的守恒计数、
  ⑫ 的两条腿），四条都有末端 must-catch，恒绿即记失败。`--wrongcount` 那档除 must-catch 外还有一条
  **两腿分离**断言：注入错数时腿 A 若同时红，说明注入的还是花括号形状 ⇒ 这一档只是 `--unformatted` 的重复。
- **十一档没动**（v4loose 7、drift 5、sibdrift 3、v5blind 4、laddersib 2、oldsib 1、flowcopy 2、ciasym 2、
  ciunread 1、v5strict 13、destroy 0）：本轮没换判据、没改流名单、那几档的被测对象集合确实没变大。
  **但这一次"没动"不能当通过**——本轮真新增的两本账只被 ⑤⑥⑨ 三道闸接住，别拿十一档的原地不动说事。
- **闸⑥ 24 → 27 对（跨两批）**：第二十发那本同流重跑本 +2、本轮跨族第一对 +1（15 格逐位）。
  又一次落空的那句预测：预登记时我按"换族这一发大概多 1 对"猜，实测本轮确实 +1，但**上一发的 +2 没预登记**
  ⇒ 许可证对数仍只能现读（第六次）。闸⑨ 参评 20 → **23 本**、闸⑩ 档案侧 74 → **80 条**、
  闸⑪ 参评 27 → **29 本**（双记账仍 3 本：CNN 那两本是 v5-era 但走的是 `_t6_v5` 判词路径，本轮没新增双记账本，
  这句话是现读的、不是推定）、闸④ 在册『污染型』21 → **22 本**逐字保持、闸⑧ 现读 v5 名下『倒挂』
  **4 本 / 1 个坐标**（第 4 本 = 第二十发落地本 `_mid_n8_t6_v5`，同一坐标只差流数）。
  **本轮最硬的一句**：这 1 个坐标现在带**家族位**了——归并键是（族/档/宽/温度列），
  打印出来是 `mlp族 × 中数据4500 × S128 × T=6.0` ⇒ "唯一坐标"这句话从此把"哪一族的几何"写在脸上。

（**上一轮（第十八发·并入流名单）不入库任何新账**，做的是把五份手抄流名单并进登记处 + 加闸⑩。
**十四档全部重跑**：老十三档**一位没动**（misfeed 88、decorative 21、collateral 13、v4blind 12、
v5blind 4、drift 5、v4loose 7、v5strict 3、borrowfloor 3、sibdrift 3、laddersib 2、oldsib 1、
destroy 0，全部逐档现读），新增的 `--flowcopy` **2 条红**（名单腿 + 系数腿各 1）。
一条没动是**预期的**：那一轮一个数字都没重训、一本账都没新增，被判据吃进的那批读数原地不动。
所以那一轮"红数不涨"的含义与上一轮不同——上一轮是"新账进了老闸的名单却不多红"（机制已解释），
那一轮是**老闸根本没有新被测对象**，全部增量都在闸⑩ 自己身上：它一旦被摘掉，`--flowcopy` 那 2 条
当场消失，而其余十三档照旧全绿 ⇒ **那一发买到的东西只有这一档能证明**，别拿十三档的原地不动当通过。）
（**再往前：第十七发**入库两本（28→**30 本**：`compress_report_v4f_n12_t6_v5.json` = MNIST ×
满数据8500 × S128 × T=6 那列补到 12 条流；`compress_report_v4f_fashion_low_n26_thin_t6_v5.json`
= Fashion × 低数据2000 × S128 × T=6 那列补到 26 条流——两本都是**补流数**，判据一个字没换）。
十三档全部重跑：**涨三档**（`--misfeed` 78→**88**、`--v4loose` 5→**7**、`--collateral` 12→**13**），
**十档一位没动**（decorative 21、drift 5、sibdrift 3、oldsib 1、laddersib 2、v4blind 12、v5blind 4、
v5strict 3、borrowfloor 3），`--destroy` 仍 0 红。三条涨的机制各记一条：
① `--misfeed` +10 = **两本新账各被复算闸① 点满五次**（v1/v2/v3/v4 并排列各一 + 本发判词一列），
按本点名实测：两本新账各 5 条，与上一轮那条"每本 v5-era 账最多被点五次名"的机制同值、无新增机制；
② `--v4loose` +2 = **两本新账都第一次在"第四个坐标"上被松腿当场抓到**：它们份额都界内
（满数据那本 99%、Fashion 那本 67%）、档案记『污染型』，"忘掉同号且更大"一装上就复算成
『污染型·倒挂』⇒ 这条腿的被测对象现在覆盖 换温度列 / 换宽度 / 换任务 / 换流数 四个维度各至少一本；
③ `--collateral` +1 = 满数据那本新账档案里的 v3 并排列写着『污染型』，"第 2 支挪前＋忘掉反号"
一装上就被读成反号线 ⇒ 同一条顺序约束多一个被测对象。
**闸⑥ 22 对 → 24 对（+2，正好每本新账各带一对）**：`_fashion_low_n8_thin_t6_v5 ⊆ _fashion_low_n26_thin_t6_v5`
（16 格逐位）与 `_n8_t6_v4 ⊆ _n12_t6_v5`（40 格逐位）。这是那条老教训（"许可证对数与这一发买了哪一维
没有单调关系"，前三次落空见下一段）第一次**没有**落空——但别把它读成规律：两本新账在盘上都各有
一本同配置的少流老账可包含，才凑出 1:1。
闸⑨ 参评 19 → **20 本**（+1：满数据那本新账第一次有了"自己那档 × 那副宽度"的底要对账，
Fashion 那本结构上仍不进参评集合）；闸④ 在册『污染型』19 → **21 本**逐字保持；
闸⑤ 把两本都算**重跑本**（同流同配置有前一版账可指）；闸⑧ 现读 v5 名下『倒挂』仍 **3 本 / 1 个坐标**
（两本新账都界内，不进名单）。
下一轮若加账，这张表必须整张重测重写；沿用上轮的数字=这张表已经在说谎。
（上一发（第十六发）入库一本（27→28 本：`compress_report_v4f_fashion_low_n8_thin_t6_v5.json`
= Fashion × 低数据2000 × S128 × T=6 那本薄阶梯 8 条流首跑账，判『污染型』、份额 93% 界内）。
十三档全部重跑：**涨两档**（`--misfeed` 73→**78**、`--v4loose` 4→**5**），**十一档一位没动**
（decorative 21、collateral 12、drift 5、sibdrift 3、oldsib 1、laddersib 2、v4blind 12、v5blind 4、
v5strict 3、borrowfloor 3），`--destroy` 仍 0 红。两条涨的机制不同，分别记：
① `--misfeed` +5 = **复算闸① 对被点名的次数**，这一本 v5-era 账在并排表里被点满五次
（v1/v2/v3/v4 各一列 + 本发判词一列），与上一轮"每本新账最多被点五次名"那条机制同值；
② `--v4loose` +1 是**这一本第一次让松腿在第二个任务上被抓**：它界内（93%）、档案记『污染型』，
"忘掉同号且更大"一装上就复算成『污染型·倒挂』。上一轮这条机制只量在 S64 那三本（同任务、换宽度），
本轮它跨到了**换任务**那一侧 ⇒ 同一档红现在有两个不同维度的被测对象。
**闸⑥ 的对账一对都没多（仍 22 对）**：这本是它那个配置的首跑账，盘上没有同配置的 3 流 Fashion 薄账
能被它包含 ⇒ **这是"许可证对数与这一发买了哪一维没有单调关系"第四次实测落空**
（前三次：第十一轮三本 `_t6` +0、第十四发只补分辨率 +7、v5 同流重跑本预登记 +1 实测 +3）。
闸⑨ 参评仍 **19 本**（它只评 MNIST 名下的 v4f 账，Fashion 结构上不在参评集合里）；
闸⑤ 把这本算**首跑**（不误伤）；闸⑧ 现读 v5 名下『倒挂』仍 **3 本 / 1 个坐标**（这本界内，不进名单）；
闸④ 在册『污染型』**19 本**逐字保持（这本是第 19 本，它加进名单而判词不动）。
下一轮若加账，这张表必须整张重测重写；沿用上轮的数字=这张表已经在说谎。
（上一发（第十五发）入库**三本**（24→27 本：`compress_report_v4f_s64_mid_v5.json`（3 条流的预演账）、
`_s64_mid_n8_v5.json` 与 `_s64_mid_n8_t6_v5.json`（8 条流的 T=3 / T=6 两本**首跑**账）——
判据一个字没换、七维里只换了**宽度**），拆红从**十二档涨到十三档**（新增 `--borrowfloor`，钉闸⑨），
十三档全部重测。计数变了三条：
**每一本新账一入库就成了复算闸① 的新被测对象**（一本 v5-era 账在 ① 的并排表里最多被点五次名：
v1/v2/v3/v4 并排列各一次＋本发判词一次），所以 `--misfeed` 58→**73**（①×67，三本新账各被点满）、
`--collateral` 9→**12**（① 多出的三条就是这三本：它们档案里的 v3 并排列写着『污染型』，
"第 2 支挪到前面＋忘掉反号条件"一装上就被读成反号线 ⇒ 同一条顺序约束现在多三个被测对象）；
**第三条不是被测对象变多，是这一档第一次抓到了正例**：`--v4loose` 1→**4**。上一轮它只红 ⑦ 一条
（"界内的账被连坐改成倒挂"这件事只能从翻转集合里看）；这三本 S64 中数据账的份额都在界内
（72% / 67% / 79%），所以"忘掉同号且更大"那条松腿**第一次在一本当场就该判『污染型』的新账上被抓**：
① 多出的三条红的内容正是"档案记『污染型』、松腿复算出『污染型·倒挂』"。
**没动的八档**：`--decorative` 21、`--drift` 5、`--sibdrift` 3、`--oldsib` 1、`--laddersib` 2、
`--v4blind` 12、`--v5blind` 4、`--v5strict` 3，`--destroy` 仍 0 红（⑥ 的对账 21 对 → **22 对**，见上面
那一条；闸⑨ 参评 19 本）。**"没动"这一句要跟着一条反面教训**：三本新账一条也没让 `--decorative`
与 `--v5blind` 多红——它们界内、判词五版同判『污染型』，拆掉阶梯顶或拆掉 CI 腿都不改变那一句
⇒ **红数不涨不等于新账没被测到**（它们已经进了 ①×67 与 ④ 的名单），但也别反过来读成"新账无所谓"。
上一轮（第十四发）的那处自纠留在原地：`--drift` 的排序键第一位改成"注入实际红几条"之后，
红数由账册结构定、不由平票裁决。加一本账就重跑全部档位再改这张表，沿用上轮的数字=这张表已经在说谎。）：
  --decorative  v3 瞎掉阶梯顶（s_top=None）→ 实测 21 条红：①×14 ② ③ ⑤×2 ⑦ ⑧×2
                （谁都翻不动=装饰；⑦ 顺带红，因为瞎顶的 v4 连"倒挂"这一格原料都没有，⑧ 跟着红两次——
                 翻转集合变空 ＋ 唯一该站着的倒挂也站不住。① 从 11 涨到 14（实测点名）：多出的三条全在
                 两本新账上——`_mid_n16_t6_v5` 两次（v4 并排列 + 本发判词都写着『倒挂』），
                 `_mid_n16_v5` 一次（它档案的 v4 并排列也写着『倒挂』：份额 105% 在 v4 下越界 ⇒
                 瞎掉顶之后连那一列都复算不出来））
  --destroy     把流级腿放松成"最多一条流反向"→ 实测 0 条红：它给的是"这批在册账的方向判定
                不靠最后一条流"这一条敏感性读数，**不算任何一道闸的验红通过**（承重由另十二档拆）
  --collateral  第 2 支忘记要求反号、且挪到第 1 支前面 → 实测 12 条红：①×9 ② ④ ⑦
                （顺序与反号条件都承重；① 从 6 涨到 9 多出的三条全是第十五发那三本 S64 中数据账
                 （`_s64_mid_v5`、`_s64_mid_n8_v5`、`_s64_mid_n8_t6_v5`）——它们档案里的 v3 并排列写着
                 『污染型』，分支顺序一挪就被读成反号线 ⇒ "第 2 支必须在前 + 必须反号"这条约束
                 现在多三个被测对象，而且第一次跨到 S64 那副学生身上）
  --misfeed     复算时把伤害方向 dirn 取反 → 实测 78 条红：①×72 + ④ + ⑤×2 + ⑦ + ⑧×2（读错数字当场被抓）
                （① 由 67 涨到 72：多出的五条全是本轮那本 Fashion 新账被点名五次）
  --drift       把扩出来的那本某一格的 wrong_kd/gap 挪 +1 → 实测 5 条红：⑥×5
                （扩流/扩格本偷改老读数必须被抓。**注入点按包含关系挑、且挑"扩展出的老账最多"的那一本**
                 （第十五发是 `_mid_n16_v5`，它同时是 5 本老账的扩展：`_mid`/`_mid_n8_v3`/`_mid_n8_v4`/
                 `_mid_n8_v5`/`_mid_n12_v4` ⇒ 一次注入该有 5 条红，**红数由账册结构定，不由字典序定**）。
                 第二类自纠（**同一类漂移第二次出现**）：上一轮定这条规则时把排序键第一位写成了
                 "反向对条数"，可打印的强度是"注入实际红几条"（=`len(hub[nw])`）——两本新账入库后
                 `_mid_n16_v5`（扩展 5 本）与 `_mid_n8_hi_v4`（扩展 4 本）反向对**都是 3 条**，
                 平票被文件名倒序判给小的那本 ⇒ 这一档停在 4 条红，而它本来该红 5。
                 教训：**"挑最强的一组"这句规则必须把"强"定义成这一档自己要打印的那个数**，
                 否则它会在账册增长时又一次悄悄降级，而且降级的方式恰好藏在平票里。
                 现在键是 (len(hub[nw]), 反向对数, 名字)。
                 这里有一处**自纠**：上一版挑的是"文件名排在后面的第一个反向对"，12 条流那本入库的那一发，
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
                 ⑧ 那条红的仍是"旧腿瞎了会连带让新腿没原料"；第十五发 ① 多的两条是 `_mid_n16_t6_v5`
                 与 `_mid_n16_v5`——前者档案判『倒挂』，后者档案 v4 并排列也写着『倒挂』，
                 所以摘腿现在多两个当场可抓的对象）
  --v4loose     倒挂腿忘掉"同号且更大"，只要判到污染型就改口 → 实测 5 条红：①×4 + ⑦，
                ⑦ 红的内容仍是"参评那几本污染型全翻成倒挂"（连坐），①×4 里有四本：第十五发新增的三本 S64 中数据账 + 本轮那本 Fashion 薄阶梯账
                ——它们界内（72% / 67% / 79%）、档案记『污染型』，松腿复算出『污染型·倒挂』当场对不上。
                **④ 抓不到它**——不是 ④ 松，是 ④ 定义在 v2/v3 复算上，结构性地看不见 v4 这条腿。
  --v5blind     v5 瞎掉那条 CI 腿（`ci_leg=False` ⇒ 与 v4 逐字同判）→ 实测 4 条红：①×2 + ⑤ + ⑧
                （装饰端：v5 若与 v4 一字不差，⑧ 的翻转集合就是空的，"换了判据"什么都没换到；
                 ① 与 ⑤ 那条红是 v5 落地本自己的——它的档案判词按 v4 复算不出来，
                 所以这一档同时钉住了"落地本必须真的被 v5 判过"。第十五发多出的那条 ① 是 `_mid_n16_v5`：
                 它按 v4 复算判『倒挂』（份额 105% 越界）而档案记的是 v5 那句『污染型』⇒
                 **v5 那条腿第一次在一本"当场复算不出来"的新账上承重**，不再只是翻旧账的口径）
  --v5strict    把 v5 那条腿的分母从 CI **半宽**换成**全宽** `hi − lo` → 实测 3 条红：① + ⑧×2
                （过头端：全宽之下 `_mid_n8_t6_v4`（29.8 vs 44.7）也跟着翻 ⇒ 翻转集合变五本、
                 该站着的那本站不住；第十五发多的一条 ① 是 `compress_report_v4f_mid_n12_t6_v5.json`
                 这本 12 条流的新账——它出界 28.0 张 < 全宽 28.8 张，同一把"过宽的尺"喂进去，
                 它自己档案里那句"倒挂"也当场复算不出来。
                 **但同一支坐标在 16 条流那本上不翻**：`_mid_n16_t6_v5` 出界 29.2 张 > 全宽 22.7 张
                 ⇒ 全宽这把尺在 8/12/16 条流上依次判 翻 / 翻 / 不翻。这条不翻**不计进红数**，
                 它如实记作：补流把半宽收窄的同时也收窄了"全宽"，过头端因此少了一条当场可抓的红。
                 **这一档钉的是这条腿里唯一没被预言钉死的算术自由度**——
                 不拆它，"半宽还是全宽"就成了"实现怎么写都算通过"）
  --borrowfloor 把 `v4b_path` 的档位后缀回落成"S64 只跑过满数据档"那个旧写法 → 实测 **5 条红**：⑨×5
                （第十五发那处定向缺陷的专属红，**第十五发第一次有红**：旧写法下 S64 × 中数据档的三本账
                 （`_s64_mid_v5`、`_s64_mid_n8_v5`、`_s64_mid_n8_t6_v5`）会去借满数据档那本底，
                 闸⑨ 现读 `floor_from` 与 `v4b_path(该书 width, 该书 tier, 该书族)` 不相等 ⇒ 当场点名"底读错了哪本"。
                 **第二十一发 3 → 5**：两本 CNN 中数据账进了同一个参评集合，旧写法连族都不认 ⇒
                 这条腿的被测对象从此覆盖 换档 / 换宽 / 换族 三种"借别人的底"。）
                 为什么这一档值得单独有：**它不判数字、判读档来源**，所以前十二档全绿也照样拦不住借底；
                 反过来，没有它，那次修复就只是一句注释（老规矩：一个修复配一档专属红才算修完））
  --flowcopy    复刻"手抄一份名单"那两步漂移：把 step7 那份 8 条流的名单最后一条换成登记处没有的
                种子，同时把它缓存的 T95 退回 n=3 那档的数 → 实测 **2 条红**：⑩×2（名单腿 + 系数腿各一）。
                为什么老十三档一条都抓不到它：它们判的是**数字**（判词、阶梯、底、CI），
                这一档判的是**这些数字出自哪一批随机流**。名单漂一条，训练照样跑、CI 照样算得出、
                判词照样自证（档案里记的就是那 8 个漂过的标签）——只有拿登记处当基准按值核才看得见。
                系数腿单独有红是因为第十五发真踩过：补了流、σ̂ 换算系数还留着 1.693，
                每一格的"过本档底"闸集体变松 1.68 倍，而那本书表面上一个字都没错）
  --ciasym      把一本在册账的 CI 上端 +40 张（区间变成不对称）→ 实测 **2 条红**：⑪×2
                （前提腿 + 等价腿）。为什么第二十发从 3 降到 2：那第三条是"全场没有一本自带双记账"
                那句**按设计该消掉**的恒红——判词侧把 `ci_leg_reading` 写进档案之后它就不该再响，
                留着它反而盖住真红。这一档钉的是"对称性是等价的前提，前提坏了必须出声"）
  --ciunread    抹掉档案里的 `verdict.ci_leg_reading` 双记账 → 实测 **1 条红**：⑪
                （"至少一本自带双记账"那句。它钉的不是数字而是**那段代码有没有把自己在测什么写进盘**：
                 没有它，闸⑪ 的口径只活在守卫自己的手算里，下一次改训练路径没人拦）
  --famblind    把每本档案里的 `student_arch` 抹掉 ⇒ 守卫按默认族（mlp）去要求底与"同配置"，
                等价于"改名之前"的状态：CNN 那两本被当成 MLP 那本的同配置重跑
                → 实测 **12 条红**：⑨×4 + ⑤×2 + ⑥×6（第二十一发 CNN 两本入库后才有被测对象）。
                为什么三条闸一起红：跨族一旦被当成同族，同一本账同时犯三种错——
                借了 MLP 那族的底（⑨）、把架构差异当成"重跑动了训练路径"（⑤）、
                把两个族的格子配成"同一批流对着读"（⑥）。**一条缺陷在三种对账里都出声，
                这就是为什么家族位要做成单点（`FAMILY_SUFFIX`）而不是每处判据各写一遍）
  --g5drop      给闸⑤ 加一条"新增却没登记"的退出路径，让一本候选既不被逐位核过也不被算成首跑
                → 实测 **1 条红**：⑤ 的守恒账（`核过 + 首跑 ≠ 候选`，并打印没归属的那本名单）。
                这一档钉的是**守卫自己那句计数**：上一版它印"已重跑 25 本"，而同一轮输出里有 19 本
                被自己打印成"该配置首跑、闸⑤ 不管"——计数与退出路径分家之后，那句自证就成了谎。
                修法是让两个数由同一个循环产出、末尾做守恒；没有这一档，"守恒"也只是句注释）
  --helpblind   生成一个只有 argparse 的合成脚本、它的 help 串里放一个裸 %（= 第十八发那处 `step5 --help`
                崩溃的同一形状），塞进闸⑬ 的被测清单 → 实测 **1 条红**：⑬。
                打印里明写"注入体不在真账清单里"：这一档测的是**检测器会不会对 rc≠0 出声**，
                真账现在能不能打开由同一档上一行那个绿色计数说（两者混成一句就是拿合成注入给真账背书）。
"""
import json
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
    """一发实验的**判别配置**（不含流数、不含判据版本）：这两样换了是另一发，不是重跑。

    家族位（`student_arch`）必须在配置里：第二十一发换的是**学生架构族**，档/宽/α/T/流名单可以
    一格不差地重合（中数据4500 × S128 × T=6 × 同 8 条流），但那两本账量的不是同一把尺子——
    没有这一位，闸⑤ 会把 CNN 那本当成 MLP 那本的重跑本、要求逐位相同，红的是一条不存在的漂移。"""
    return (d.get("dataset", "mnist"), d.get("student_arch", "mlp"), d["tier"], d["width"],
            str(d["probe_alpha"]), str(d.get("temps_fixed")),
            tuple(float(a) for a in d["ladder"] if float(a) != 0.0))


def same_cfg(a, b):
    """同一发配置的**重跑**：判据版本与流数可以不同，配置六项必须逐字相同。"""
    return setup_of(a)[:6] == setup_of(b)[:6]


def ladder_within(a, b):
    """a 的阶梯是 b 的阶梯的子集（b 是"扩了格"的那本）。"""
    return set(setup_of(a)[6]).issubset(set(setup_of(b)[6]))


if MODE == "--famblind":
    print("  （拆红注入）把所有账里的 student_arch 抹掉 ⇒ 守卫看不见『族』这一位，"
          "退回到改名之前的状态：CNN 那本会被当成 MLP 那本的同配置重跑")

for path in sorted(glob.glob("compress_report_v4f*.json")):
    d = json.loads(open(path, encoding="utf-8").read())
    if MODE == "--famblind":
        d.pop("student_arch", None)
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
        inv5.setdefault(setup_of(docs[r[0]])[:6], []).append(r[0])
print(f"  【现读】v5 复算判『倒挂』的账共 {sum(len(v) for v in inv5.values())} 本，"
      f"按配置（族/档/宽/温度列）归并成 {len(inv5)} 个坐标：")
for k, v in inv5.items():
    note = f"（同一坐标的 {len(v)} 本：只差流数，逐位对账线在 ⑥）" if len(v) > 1 else ""
    print(f"        {k[1]}族 × {k[2]} × S{k[3]} × T={k[5]} ⇒ {', '.join(v)}{note}")
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
g5_checked, g5_first = [], []      # ⑤ 真正逐位核过的 / 被 ⑤ 明确算成首跑的（两个数由同一个循环产出）
if not fresh:
    fails.append("闸⑤ 一本 _v3/_v4 重跑账都没有——预言没人验过就等于没落地，别宣布新判据换到了东西")
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
        g5_first.append(p)
        print(f"  {p}：{len(sibs)} 本同配置账、但没有一本流集合相同 → 这是该配置的**首跑**，"
              f"不是重跑本，闸⑤ 不管（它的对账线是恒等闸＋跨脚本闸）")
        continue
    pick_oldest = MODE == "--oldsib"
    earlier = [q for q in same_flow if ver_rank(q) < ver_rank(p)]
    if not earlier and not pick_oldest:
        g5_first.append(p)
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
    if MODE == "--g5drop" and g5_checked:
        print(f"  （拆红注入 --g5drop）{p} 被一条**新增却没登记**的退出路径跳掉 ⇒ 钉『每本候选必须有归属』")
        continue
    g5_checked.append(p)
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

# 计数句必须由循环自己产出：候选表是按**记录版本**筛的，而"这本到底有没有前一版可指"只有循环里知道。
# 第二十一发的现读事故：两本 CNN 账以 v5 名义进候选 ⇒ 老那句"已重跑 25 本"把它们算成重跑本，
# 而同一个闸往下几行又把同一本打印成"该配置首跑、闸⑤ 不管"。同一份输出里两句话互相打脸。
print(f"[闸⑤ 落地自证] 候选 {len(fresh)} 本 = 逐位核过 {len(g5_checked)} 本 + 首跑/无更低版本前账 "
      f"{len(g5_first)} 本（后者名单：{g5_first}）")
if len(g5_checked) + len(g5_first) != len(fresh):
    orphans = [p for p in fresh if p not in g5_checked and p not in g5_first]
    fails.append(f"闸⑤ 候选账不平：核过 {len(g5_checked)} + 首跑 {len(g5_first)} ≠ 候选 {len(fresh)}，"
                 f"没归属的是 {orphans} ⇒ 有一本候选既没被逐位核过也没被算成首跑，"
                 f"那句计数就成了谎（配专属红 --g5drop）")

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

# 闸⑨：底不许跨档/跨宽度借。钉的是第十五发查出的那处定向缺陷——`v4b_path` 旧写法只在
# width == 128 时贴档位后缀（写死的假设"S64 只跑过满数据档"），于是 64×中数据 会去读满数据那本，
# 而那本里真有一个"种子42"的臂 ⇒ `load_anchor` 的档检查拦不到，闸门建在别的档的底上。
print("\n[闸⑨ 每本账的底必须是它自己那档 × 那副宽度的 v4b]")
import step4c_alpha_T_sweep as sc                      # noqa: E402
if MODE == "--borrowfloor":
    def _old_v4b_path(width: int, tier: str):           # 第十五发之前的那版定向
        stem = "compress_report_v4b" if width == 128 else f"compress_report_v4b_s{width}"
        return sc.HERE / (stem + ("" if width != 128 else sc.V4B_SUFFIX.get(tier, "")) + ".json")
    sc.v4b_path = _old_v4b_path
    print("  （拆红注入）把 v4b_path 退回旧写法：宽度非 128 时丢掉档位后缀 ⇒ 中数据档借满数据档的底")
n9 = 0
_arch_saved = sc.ARCH                             # 守卫自己的全局不能决定"该读哪一族的底"
for name, d in sorted(docs.items()):
    if str(name).endswith(".json") and d.get("dataset") == "mnist" and d.get("tier"):
        fam = d.get("student_arch", "mlp")
        sc.ARCH = fam                             # 每本按**自己那一族**算档案名（cross-family 借底 = 借了别的尺子）
        want = sc.v4b_path(int(d["width"]), d["tier"]).name
        got = d.get("floor_from")
        n9 += 1
        if ("_cnn" in str(name)) != (fam == "cnn"):
            fails.append(f"闸⑨ {str(name).split(chr(92))[-1]}：档案名里的家族位与档案内容对不上"
                         f"（名字 {'带' if '_cnn' in str(name) else '不带'} _cnn，记的学生架构是 {fam}）"
                         f"——名字与臂不是同一族，这一本的判决不知道是谁家的")
            print(f"  {name} → **红**：名字与 student_arch 不同族")
        if got != want:
            fails.append(f"闸⑨ {name.split(chr(92))[-1]}：底读的是 {got}，该读 {want}"
                         f"（{fam}族 × tier={d['tier']} × width={d['width']}）——闸门建在别的档/别的族的底上，这一本的判决作废")
            print(f"  {name} → **红**：floor_from={got} ≠ 该档该宽度该族的 {want}")
sc.ARCH = _arch_saved
if n9 == 0:
    fails.append("闸⑨ 没有一本 MNIST 账可评——不是通过，是没测到")
elif MODE == "--borrowfloor" and not any(f.startswith("闸⑨") for f in fails):
    fails.append("闸⑨（拆红模式 --borrowfloor）一条都没抓到：这一档专门钉 S64×中数据 那本账，"
                 "它没入库之前此档等于没测到，不许记成通过")
else:
    print(f"  参评 {n9} 本：每本的 floor_from 都等于自己那档 × 那副宽度 × 自己那一族算出的档案名，"
          f"且档案名的家族位与 student_arch 一致")

# 闸⑩：流名单与系数只有一个出处——**按值**查，不看 import 行、不看注释里写了什么。
# 第十八发 E 把 step4e/4g/5/6b/7 那五份手抄名单并进登记处，这一档钉的是"并完不许再漂回去"：
# 换人换会话时最省事的做法是在自己文件里再抄一份，抄的那一刻两份相同，补流之后就不同了，
# 而"少一条流"这种漂移不会报错，只会安静地跑出一本标着 8 条流、实际只有 7 条对得上的账。
# 第二条腿管系数：第十五发在 step4d 抓到的是"名单补了、σ̂ 换算系数还留着 n=3 那个数"，
# 那一次是读代码读出来的；从这一发起由这一档盯着（系数与名单必须同档，否则当场点名）。
print("\n[闸⑩ 流名单/系数按值等于登记处（模块侧 + 档案侧）]")
import step4_distill as sd                         # noqa: E402
import step4e_marginal_recheck as s4e              # noqa: E402
import step4g_flip_zone as s4g                     # noqa: E402
import step5_second_teacher as s5                  # noqa: E402
import step6b_fashion_dry_teacher as s6b           # noqa: E402
import step7_task_teacher_2x2 as s7                # noqa: E402

if MODE == "--flowcopy":
    s7.SEEDS = s7.SEEDS[:-1] + [s7.SEEDS[-1] + 1]
    s7.T95 = sd.T95[3]
    print("  （拆红注入）复刻当年那两步：step7 那份 8 条流的名单最后一条换成登记处没有的种子，"
          "并把缓存的 T95 退回 n=3 那档的数（抄了名单忘了换系数）⇒ 名单腿与系数腿各该红一条")

LISTS = {"step4e.OLD+NEW": list(s4e.OLD_SEEDS) + list(s4e.NEW_SEEDS),
         "step4g.SEEDS": list(s4g.SEEDS),
         "step4g.SEEDS8": list(s4g.SEEDS8),
         "step4g.SEED_SET[3]": list(s4g.SEED_SET[3]),
         "step4g.SEED_SET[8]": list(s4g.SEED_SET[8]),
         "step5.SEEDS": list(s5.SEEDS),
         "step6b.SEEDS8": list(s6b.SEEDS8),
         "step7.SEEDS": list(s7.SEEDS)}
COEF = {"step5.D4": (s5.D4, sd.D4[len(s5.SEEDS)], len(s5.SEEDS)),
        "step7.T95": (s7.T95, sd.T95[len(s7.SEEDS)], len(s7.SEEDS))}
n10 = 0
for label, lst in sorted(LISTS.items()):
    n10 += 1
    want = list(sd.FLOW_SEEDS.get(len(lst), []))
    if lst != want:
        extra = "登记处没有这一档" if not want else f"该是 {want}"
        fails.append(f"闸⑩ {label}：{len(lst)} 条流却与登记处不同（实际 {lst}，{extra}）"
                     f"⇒ 名单有了第二副本，'同一批随机流'这句话在这本账上不再成立")
        print(f"  {label} → **红**：与登记处逐位不同")
for label, (got, want, n_used) in sorted(COEF.items()):
    n10 += 1
    if abs(got - want) > 1e-9:
        fails.append(f"闸⑩ {label}={got}，而它那份名单有 {n_used} 条流"
                     f"⇒ 该用 {want}：名单与系数不是同档（第十五发 step4d 那个 bug 的形状）")
        print(f"  {label} → **红**：{got} ≠ 该名单长度的 {want}")
# 档案侧：凡是记了自己名单的那本，名单必须是登记处那一档
rec_arch = 0
unreg = []
for path in sorted(glob.glob("compress_report_v*.json")):
    d = json.loads(open(path, encoding="utf-8").read())
    pools = {}
    if isinstance(d.get("seeds"), list):
        pools["seeds"] = d["seeds"]
    if isinstance(d.get("seed_order"), list):
        pools["seed_order"] = d["seed_order"]
    if isinstance(d.get("seed_pools"), dict):
        pools.update({f"seed_pools[{k}]": v for k, v in d["seed_pools"].items()})
    for field, lst in pools.items():
        lst = [int(x) for x in lst]
        rec_arch += 1
        if lst != list(sd.FLOW_SEEDS.get(len(lst), [])):
            unreg.append(f"{path.split(chr(92))[-1]}:{field}")
            fails.append(f"闸⑩ {path} 的 {field}（{len(lst)} 条）不是登记处那一档 ⇒ 这本账的复现单位"
                         "没有名字，跨账配对不许用它")
print(f"  模块侧 {n10} 项（名单 + 缓存系数）按值核过登记处")
if unreg:
    print(f"  档案侧 {rec_arch} 条名单里 {len(unreg)} 条对不上：{unreg}")
elif rec_arch:
    print(f"  档案侧 {rec_arch} 条名单全部等于登记处那一档")
else:
    fails.append("闸⑩ 档案侧一本记了名单的都没读到——不是通过，是没测到")
if MODE == "--flowcopy" and not any(f.startswith("闸⑩") for f in fails):
    fails.append("闸⑩（拆红模式 --flowcopy）注入的两处漂移一条都没抓到：这一档是恒绿的，"
                 "它对『名单漂了』零反应，那么它拦不住任何一次手抄")

# 闸⑪：v5 那条 CI 腿的分母不许是口味（第二十发还的债）。
# 这条腿写的是"出界幅度 > 配对步长 CI 的半宽"，而它**声称**在测的是"这一步自己的 CI 不含 0"。
# 两者只在区间以点估计为中点时等价——那句前提此前只活在注释里，所以它必须当场可核：
#   腿 A 对称性：档案里的 CI 中点离点估计不许超过 sf.CI_CENTER_TOL（= 一位小数的两倍舍入）；
#   腿 B 分子同一性：那条腿比的"出界幅度"必须就是这一步自己的点估计（差只许来自各自舍入，0.25 张）；
#   腿 C 等价性：拿**当前模式下的规则实现**问一遍它判成承重没有，答案必须等于"CI 不含 0"
#     （所以 --v5strict 把分母换成全宽时，红的是这条腿，红的内容是"实现与它声称的命题分家"，
#      而不是上一轮那种"某本账的判词变了"——那是后果，不是理由）；
#   腿 D 双记账：凡档案自带 ci_leg_reading（由训练路径那段代码写的）的，守卫必须能独立复算出同一份，
#     全册一本都没有 ⇒ 这条腿只被守卫自己核过、实现侧没人管 ⇒ 算红，不算通过。
# 参评集合与闸⑧ **不同**：闸⑧ 只评"记录在 v4 名下且阶梯顶在账"的那几本，这一闸评所有记了配对步长的账
# （两闸数的是不同的东西，混在一起会各自都看不出对方漏了什么）。
print("\n[闸⑪ v5 那条 CI 腿 == 「配对步长 CI 不含 0」，且它的前提（区间对称）当场可核]")
if MODE == "--ciasym":
    inj = [p for p in docs if p.endswith(MUST_STAY)]
    for p in inj:
        ci = docs[p]["paired_step_small_to_anchor"]["ci95_imgs"]
        ci[1] = round(float(ci[1]) + 40.0, 1)     # 只动上端：区间不再以点估计为中点
    print(f"  （拆红注入）给 {MUST_STAY} 的 CI 上端 +40 张 ⇒ 中点漂移、且「> 半宽」与「不含 0」分家")
if MODE == "--ciunread":
    stripped = [p for p in docs if (docs[p].get("verdict") or {}).pop("ci_leg_reading", None) is not None]
    print(f"  （拆红注入）抹掉 {len(stripped)} 本账的 ci_leg_reading ⇒ 双记账那条腿退回「只有守卫核得到」")
    if not stripped:
        fails.append("闸⑪（拆红模式 --ciunread）全册一本双记账都抹不到 ⇒ 这一档现在是空转的："
                     "落地本还没入库，『抹掉它该红』这件事此刻没有被测过")
n11 = n11_double = 0
for name, d in sorted(docs.items()):
    ps = d.get("paired_step_small_to_anchor")
    v = d.get("verdict") or {}
    if not isinstance(ps, dict) or ps.get("mean_gap_imgs") is None or not ps.get("ci95_imgs"):
        continue
    lo, hi = (float(x) for x in ps["ci95_imgs"])
    m = float(ps["mean_gap_imgs"])
    half = (hi - lo) / 2
    book = name.split(chr(92))[-1]
    n11 += 1
    off = abs((lo + hi) / 2 - m)
    if off > sf.CI_CENTER_TOL:
        fails.append(f"闸⑪ {book}：CI [{lo:+.1f}, {hi:+.1f}] 的中点离点估计 {m:+.1f} 有 {off:.2f} 张 > 容差 "
                     f"{sf.CI_CENTER_TOL:g} ⇒ 这条腿的分母不再是它的 margin，「> 半宽」与「不含 0」不再等价")
    gap_oob = abs(m) if v.get("oob_gap_imgs") is None else float(v["oob_gap_imgs"])
    if abs(gap_oob - abs(m)) > 0.25:
        fails.append(f"闸⑪ {book}：那条腿比的出界幅度是 {gap_oob} 张，而这一步自己的点估计是 {m} 张"
                     "⇒ 比的不是同一个量（差超过两处一位小数舍入之和 0.25 张）")
    # 必须把模块全局显式喂进去：`full_width` 是 oob_resolution 的**入参**，它自己不读 FULL_WIDTH_MODE，
    # 裸调用会永远拿半宽 ⇒ --v5strict 之下这条腿假绿（"走实现"这句话就成了注释）。
    load_rule = gap_oob > sf.oob_resolution(ps, full_width=bool(sf.FULL_WIDTH_MODE))
    load_ci = bool(lo > 0 or hi < 0)
    if load_rule != load_ci:
        fails.append(f"闸⑪ {book}：规则实现判「{('承重' if load_rule else '不承重')}」而 CI 不含 0 判"
                     f"「{('承重' if load_ci else '不承重')}」⇒ 实现与它声称的命题分家"
                     + ("（这正是把分母换成全宽的后果：板块没有任何一句话要用 |均值| > hi−lo）"
                        if MODE == "--v5strict" else ""))
    dbl = v.get("ci_leg_reading")
    if isinstance(dbl, dict):
        n11_double += 1
        want = sf.ci_leg_reading(ps, gap_oob)
        diff = {k: (dbl.get(k), want.get(k)) for k in set(dbl) | set(want) if dbl.get(k) != want.get(k)}
        if diff:
            fails.append(f"闸⑪ {book}：档案自带的双记账与守卫复算不一致 {diff}"
                         "⇒ 训练路径那段代码与守卫读的不是同一把尺子")
if n11 == 0:
    fails.append("闸⑪ 一本记了配对步长的账都没有——不是通过，是没测到")
elif n11_double == 0:
    fails.append(f"闸⑪ {n11} 本参评账没有一本自带 ci_leg_reading 双记账 ⇒ 这条腿的口径只被守卫自己核过，"
                 "产出它的那段代码没有把『我在测什么』写进档案（修复还没落到盘上）")
else:
    print(f"  参评 {n11} 本（每本一条小→锚配对步长）；其中 {n11_double} 本自带双记账且与守卫复算逐字段相同")
if MODE == "--ciasym" and not any("中点离点估计" in f or "分家" in f for f in fails):
    fails.append("闸⑪（拆红模式 --ciasym）注入的非对称 CI 一条都没抓到：这一档恒绿，"
                 "它拦不住『区间不对称却仍用半宽当 margin』")
if MODE == "--ciunread" and not any("双记账" in f for f in fails):
    fails.append("闸⑪（拆红模式 --ciunread）抹掉双记账之后没红：说明『至少一本自带双记账』那句是句空话")

if MODE == "--famblind" and not (any(f.startswith("闸⑤") for f in fails)
                                 and any("家族位" in f for f in fails)):
    fails.append("闸⑤＋闸⑨（拆红模式 --famblind）两条腿没同时红：既要有『跨族当成重跑』的逐位红，"
                 "又要有『档案名与 student_arch 不同族』的红——少一条就是 CNN 那本还没入库或守卫根本没读族，"
                 "此档等于没测到，不许记成通过")

if MODE == "--g5drop" and not any(f.startswith("闸⑤ 候选账不平") for f in fails):
    fails.append("闸⑤（拆红模式 --g5drop）『候选必须有归属』那条守恒账没红：要么那句计数已经不再被打印"
                 "（那这条拆红档就是恒绿的装饰），要么丢掉的这本被别的退出路径接住了——两种都不是通过，"
                 "此档等于没测到，不许记成通过")

# 闸⑫：档案里的自我限定句子，不许是"没渲染的模板"（第二十一发现的两处自伤）。
# 踩中的东西很具体：`step4b_seed_variance.py` 有一句 limitations 漏了 f 前缀，于是那本账把自己
# 的复现单位打印成了字面量 `{len(seeds)}`。这类 bug 不会让它任何数字变红——训练照样跑、
# CI 照样算、判词照样自证——它只把"这本账知道自己底是欠估"那句话变成一句没人读过的代码文本。
# 所以它不并进判据类闸（⑤–⑪ 判的是数字出自哪批流、比值等价于什么），单立一道**文本闸**，
# 但按第十五发那条规矩**每一腿都得有自己的专属红**（`--unformatted` 钉腿 A、`--wrongcount` 钉腿 B）：
# 没有专属红的腿，就是恒绿的装饰。
#   腿 A 残留：任何字符串字段里出现 `{…}` 且花括号内像代码（`标识符(`、含下标或引号）⇒ 红；
#   腿 B 补渲染补错数：凡 limitations 里那句"种子底由 N 个点的极差估"，N 必须等于该本自己的
#     len(seeds)——钉的是"把数填进句子时填了别的账的数"那一类。**反面要如实说**：本轮真的自伤过一次
#     （补渲染的脚本把三本没坏的账那句措辞从"欠估"改成了"仍是欠估"），那一处**不留花括号、也不改数**，
#     腿 A、腿 B 都抓不到它 ⇒ 这类只改措辞的误伤当前只有 `git diff` 加人眼这一道防线，别把它记成有闸。
#   分母自证：全册 80 本、展开的字符串字段条数必须现读打印；读到 0 条 ⇒ 扫描器本身失效，算红。
print("\n[闸⑫ 档案文本里没有未渲染的占位符，且『种子底由 N 个点』那句的 N == 该本自己的 len(seeds)]")
BRACE = re.compile(r"\{([^{}]*)\}")


def _looks_like_code(inner):
    return bool(re.search(r"[A-Za-z_]\w*\s*[(\[]", inner)) or "'" in inner or '"' in inner or "[" in inner


def _walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


arch_all = {}
for path in sorted(glob.glob("compress_report_v*.json")):
    arch_all[path] = json.loads(open(path, encoding="utf-8").read())
if MODE == "--unformatted":
    hit = 0
    for book, d in sorted(arch_all.items()):
        for i, s in enumerate(d.get("limitations") or []):
            if isinstance(s, str) and "种子底由" in s and "个点" in s:
                d["limitations"][i] = re.sub(r"(种子底由 )\d+( 个点)", r"\1{len(seeds)}\2", s)
                hit += 1
    print(f"  （拆红注入）把 {hit} 本账的种子底那句退回漏 f 的形状 ⇒ 全册应读到 {hit} 处花括号残留")
    if hit == 0:
        fails.append("闸⑫（拆红模式 --unformatted）注入腿找不到任何一本带『种子底由 N 个点』的账 ⇒ "
                     "这一档现在是空转的：腿 A 此刻没有被测过，不许记成通过")
if MODE == "--wrongcount":
    picked = None
    for book, d in sorted(arch_all.items()):
        if not isinstance(d.get("seeds"), list):
            continue
        for i, s in enumerate(d.get("limitations") or []):
            m = isinstance(s, str) and re.search(r"(种子底由 )(\d+)( 个点)", s)
            if m:
                wrong = int(m.group(2)) + 1
                d["limitations"][i] = f"{m.group(1)}{wrong}{m.group(3)}"
                picked = f"{book.split(chr(92))[-1]}（{m.group(2)} → {wrong}，seeds 实为 {len(d['seeds'])} 条）"
                break
        if picked:
            break
    print(f"  （拆红注入）把 {picked or '（没有一本可注入）'} 的那句改成错数 ⇒ 花括号一个都没多，"
          "只有腿 B 看得见")
    if not picked:
        fails.append("闸⑫（拆红模式 --wrongcount）找不到『种子底由 N 个点 + seeds 是列表』的账可注入 ⇒ "
                     "腿 B 此刻是空转的，不许记成通过")
n12_books = n12_str = 0
n12_seed_sent = n12_seed_bad = 0
for book, d in sorted(arch_all.items()):
    name = book.split(chr(92))[-1]
    n12_books += 1
    for field, s in _walk(d):
        n12_str += 1
        for inner in BRACE.findall(s):
            if _looks_like_code(inner):
                fails.append(f"闸⑫ {name}{field}：读到未渲染的占位符 {{{inner}}} ⇒ 这本账的自我限定"
                             "是一条代码文本而不是句子（写它的句子漏了 f 前缀）")
    for s in d.get("limitations") or []:
        if not isinstance(s, str):
            continue
        m = re.search(r"种子底由 (\d+) 个点的极差估", s)
        if m:
            n12_seed_sent += 1
            seeds = d.get("seeds")
            if isinstance(seeds, list) and int(m.group(1)) != len(seeds):
                n12_seed_bad += 1
                fails.append(f"闸⑫ {name}：那句写的是 {m.group(1)} 个点，而这本自己的 seeds 有 "
                             f"{len(seeds)} 条 ⇒ 补渲染补错了本（数字是从别的账抄来的）")
if n12_books == 0 or n12_str == 0:
    fails.append(f"闸⑫ 全册读到 {n12_books} 本 / {n12_str} 条字符串 ⇒ 扫描器根本没展开档案，"
                 "这一档等于没测到，不许记成通过")
else:
    print(f"  分母自证：全册 {n12_books} 本、展开 {n12_str} 条字符串字段；"
          f"其中 {n12_seed_sent} 本带『种子底由 N 个点』那句，N 与各自 len(seeds) 对不上的有 {n12_seed_bad} 本")
if MODE == "--unformatted" and not any(f.startswith("闸⑫") and "占位符" in f for f in fails):
    fails.append("闸⑫（拆红模式 --unformatted）注入了花括号却没读到红：腿 A 恒绿，"
                 "它拦不住下一次漏 f（此档等于没测到，不许记成通过）")
if MODE == "--wrongcount":
    if not any(f.startswith("闸⑫") and "补错了本" in f for f in fails):
        fails.append("闸⑫（拆红模式 --wrongcount）改了数却没读到红：腿 B 恒绿，"
                     "『N 必须等于该本自己的 len(seeds)』那句是空话（此档等于没测到，不许记成通过）")
    if any(f.startswith("闸⑫") and "占位符" in f for f in fails):
        fails.append("闸⑫（拆红模式 --wrongcount）腿 A 也红了：注入进去的还是花括号形状 ⇒ 两腿没分离，"
                     "这一档其实只是 --unformatted 的重复，它没有带来新被测对象")

# 闸⑬：脚本"能不能打开"——每个带开关的脚本跑一次 `--help`，退出码必须是 0。
# 来历：第十八发撞到 `step5_second_teacher.py --help` **自那行帮助文本写下起就没通过过**
#   （argparse 把 help 整串做 `%` 插值，帮助文本里写 `50%` 必须写 `50%%`）。这类崩溃不在任何训练路径上，
#   而守卫是 import 模块、不走 argparse ⇒ 前十二道闸对它零反应：改了判据的崩、读档的崩都拦不到"打开就炸"。
# 这一档判的是**入口**不判读数 ⇒ 它不产生任何新的对账，也不许被读成"某句话被复证"。
# 被测清单按值现读（源码里出现 add_argument 就算），名单不写死：新加一个带开关的脚本自动进清单。
print("\n[闸⑬ 每个带开关的脚本 --help 都能打开（退出码 0）]")
HELP_SCRIPTS = [p for p in sorted(glob.glob("step*.py"))
                if "add_argument" in open(p, encoding="utf-8").read()]
PROBE = "_help_probe_v3red.py"


def _help_rc(script):
    r = subprocess.run([sys.executable, "-u", script, "--help"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    return script, r.returncode, (r.stderr or "")


n13 = 0
_t13 = time.monotonic()
try:
    if MODE == "--helpblind":
        open(PROBE, "w", encoding="utf-8").write(
            "# -*- coding: utf-8 -*-\n"
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--x', help='覆盖率达到 50%、就这一句')\n"
            "p.parse_args()\n")
        print(f"  （拆红注入）生成 {PROBE}：一个只有 argparse 的合成脚本，help 串里放一个裸 % "
              "（第十八发那处崩溃的同一形状）⇒ 它必须打不开")
    targets = HELP_SCRIPTS + ([PROBE] if MODE == "--helpblind" else [])
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_help_rc, targets))
    for script, rc, err in sorted(results):
        n13 += 1
        if rc != 0:
            tail = (err.strip().splitlines() or ["（无 stderr）"])[-1]
            fails.append(f"闸⑬ {script}：--help 退出码 {rc}（不是 0）⇒ 这个脚本的开关清单打不开，"
                         f"报错最后一行：{tail}")
            print(f"  {script} → **红**：rc={rc}")
    if n13 == 0:
        fails.append("闸⑬ 清单里一个带开关的脚本都没读到 ⇒ 扫描器没展开，这一档等于没测到，不许记成通过")
    else:
        green = n13 - sum(1 for f in fails if f.startswith("闸⑬"))
        print(f"  现读清单 {n13} 本（源码含 add_argument），退出码 0 的 {max(green, 0)} 本，"
              f"墙钟 {time.monotonic() - _t13:.1f} 秒（4 并发）")
    if MODE == "--helpblind":
        if not any(f.startswith("闸⑬") and PROBE in f for f in fails):
            fails.append(f"闸⑬（拆红模式 --helpblind）注入的 {PROBE} 一条都没抓到 ⇒ 这一档恒绿，"
                         "它对『帮助文本里有裸 %』零反应，拦不住下一次同类崩溃（此档等于没测到）")
        else:
            print(f"  must-catch 兑现：注入体 {PROBE} 被这一档点名（**它是合成脚本、不在那 {len(HELP_SCRIPTS)} "
                  "本真账里** ⇒ 这一档证明的是检测器会对 rc≠0 出声，不证明真账里现在有人打不开——"
                  "上面那行绿色计数说的才是真账）")
finally:
    if MODE == "--helpblind" and os.path.exists(PROBE):
        os.remove(PROBE)

print(f"\n模式：{MODE or '（正常验红）'}")
if fails:
    print(f"\n>>> 验红 {len(fails)} 条红：")
    for f in fails:
        print("   ", f)
    raise SystemExit(1)
print("\n结论：v3 只买『判据看得见阶梯顶』这一件事，v4 只买『出界的份额话改口成倒挂』这一件事"
      "——其余各本判词逐字不动。")
