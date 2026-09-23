# -*- coding: utf-8 -*-
"""第4课h：两列温度之间的斜率差——**按流配对**，不是两个点估计相减（纯读已入库的账，零训练、零 API 费）

欠账的原文（`面试版判决表.md` 第四节，上一轮新记的那条小账）：反号线在 MNIST 低数据档有两个坐标
（T=3 与 T=6），两列"小剂量→阶梯顶"的配对步长各自是 +48.6 与 +101.5 张 ⇒
"斜率随温度变陡"这句话**至今只是两个各自带 CI 的点估计相减**。同一批 8 条流、同一批图、
同一副学生、同一个老师，所以这两列本可以按流配对——配对了才有它自己的 CI。中数据档那一 pair
（倒挂的两个坐标，步长 7.1 与 29.8）顺手一起算，因为同一条规矩管两档。

四道检查必须先于读数成立（能被拆红的才叫闸，见下面"拆红"）：
  闸 A（**配对前提逐位可核，且必须核满整池**）：两本账 α=0 那一格是 CE 臂，它与温度无关。如果同一条流
      在两列里真是同一次初始化、同一批图，那它的 `ce_wrong` 必须**逐位相同**。这一条把"我凭什么按种子配对"
      从口头假设变成盘上对账。**CE 读数的来源本轮（D）换过格子**：先前读的是恒等闸名单（薄档只在老 3 条流
      上跑恒等闸 ⇒ 低数据那一 pair 一度只核到 3/8，其余 5 条靠"同一构造推定"），而每一臂行本来就带着自己
      那粒种子的 `wrong_ce`，整池在册。换来源之后"推定"这条降级路径**没有理由再存在** ⇒ 覆盖不到整池
      不再"如实报个比例继续跑"，而是当场退出（`gate_ce_pairing` 里的 `short` 那条）。
      顺带三条新的自洽检查，都挂在 `load()` 上：同一粒种子在不同 α 下的 CE 读数必须唯一（否则"CE 与 α
      无关"只是口头的）、臂行与恒等闸两个来源必须报同一个数（同本账自相矛盾）、缺读数不许进配对。
  闸 B（复算对账）：本脚本从 `ladder_summary[α]["gaps_imgs"]` 现列重算的**每一格均值**，以及
      α=小剂量→阶梯顶 那条步长的 均值 / σ / SE / 极差 / d4 折算 σ̂ / 二项 p / **逐流序列**，
      必须与**那本账自己记着的**读数逐位相同。对不上就是读错账（串档、取错 α、拿错档），当场退出——
      不许把错位来的差值当新读数打印出去。
  闸 C（两列必须真的不同 + **两本必须真的同池**）：先核两本账的 `seeds` **逐位相同**（成员与顺序都算），
      再核 `temps_fixed` 不等、其余判别变量全同。两列相同 ⇒ 斜率差恒为 0，那一发什么都没测
      （"空表全过 = 没测到"的同一族）；流池不同 ⇒ 跨列的差只能按"两个均值相减"读，
      按流配对当场退出。**这一条是第十三发新加的对**（8 流那一对 vs 12 流那一对）逼出来的：
      一本 12 条流的账与一本 8 条流的账文件名长得几乎一样，只查"档位/宽度/温度"三样会放它们过关。
      配套的入口检查在 `load()`：每本账的流名单必须**整本等于**板块在册的某一池（`POOLS` 的 8 / 12 / 16），
      对不上就退出——不许"少几条也算同一批"。三池是**前缀关系**（16 那池的前 12 条就是 12 那池、
      12 那池的前 8 条就是 8 那池）⇒ 这道检查能命中的仍然只有三格，截一段下来不算过关。

拆红（每档只钉一道闸，跑不出红的模式＝那道闸没测过）：
  --red-check  把中数据那一对的 T=6 账换成另一本 T=3 账 ⇒ 只有闸 C 该红
  --unlink     只旋转 T=6 那本的**种子标签**（模拟「同名种子其实不是同一条流」）⇒ 红的第一道必须是闸 A。
               这一档由 `probe_gates` 把 A、B 各自的反应**实测打印**出来，不靠注释断言：B 的
               均值/σ/SE/极差/二项 p 都不读键 ⇒ 该照旧绿；但 B 的「逐流序列」那条子检查读键序 ⇒
               它也会红。所以 B 不是 A 的替身，可也不能说 B 完全看不见配对方向——这句话说的是实测。
  --scramble   给 T=6 那本的所有 gap +1（模拟取错档案/单位）⇒ 只有闸 B 该红（A 只看 CE 臂，不碰 KD gap）
  --poolblind  第十三发新加的那一对（12 条流）逼出来的一档，钉流池检查。本轮升到**六笔实测**
               （①②③③b③c④，全绿才算"承重"）：
               ① 拿 12 条流的 T=3 账配 8 条流的 T=6 账 ⇒ 同池检查必须红；② 拆掉这道检查重跑同一对
               ⇒ 它**不炸**、只是把用池换成 8 条（这就是没有闸时"静默换分母"的真实后果，也是①那条红
               承重的证据）；③ 把 12 条流那本的最后一条种子换成池外值（**条数仍是 12、仍在系数册上**）
               ⇒ 在册池成员检查必须红，拦它的不是 `check_flow_constants`（这一区分防止两道闸互相顶包）；
               ③b 关掉成员检查重读同一本换了成员的账 ⇒ 它**过了系数册**、红在下一道（CE 逐流读数凑不齐，
               因为池外那粒种子在这本账里没有臂行）。**这一笔是本轮的更正**：上一版写的是"关掉入口检查
               同一本照样读进来"，那句在 CE 覆盖检查上线后就不成立了——不重跑没人会发现这一支红到了别的闸上。
               ③c 成员一字不差、只把名单**倒序** ⇒ 入口成员检查红；再把成员检查关掉重读 ⇒ CE 覆盖**全绿**
               （12/12 条都有读数）⇒ 成员检查钉的是**顺序**这一维（配对按位置 zip gap），CE 那道管不到它，
               所以 ③b 的实测不能被读成"成员检查冗余"；④ 截成 11 条 ⇒ 入口对流数的检查红。
  --ceblind    第十三发 D（CE 逐流读数改从臂行 `wrong_ce` 取）逼出来的一档，钉上面那三条新检查，也是四步：
               ① 抹掉两粒种子的 CE 读数 ⇒ 入口"缺读数不许进配对"必须红；② 把入口那道拆掉重跑同一对
               ⇒ 闸 A 的整池覆盖检查必须**接着**红（这一条证明两道检查互为备份而不是一道顶一道；
               如果②不红，就说明"覆盖不足当场退出"其实只在入口生效）；③ 让同一粒种子在两个 α 下报
               两个 CE 数 ⇒ "CE 与 α 无关"这条前提的检查红；④ 只挪恒等闸那一个来源 ⇒ 两个来源对账的红。

预登记的出口（跑之前写死，此后不许改口；钉子数字现读自已入库档案）：
  ① 四道检查（流池 + A + B + C）全过才发读数；任一红 ⇒ 退出、不入库。
  ② 斜率差的**点估计由算术钉死**，不许当成新信息：中数据 = 29.8 − 7.1 = **+22.7 张**、
     低数据 = 101.5 − 48.6 = **+52.9 张**。本脚本的点估计是从逐流 gap 现算的，而档案存的两个均值
     各自已圆到 0.1 张 ⇒ 允许**双舍入的 0.1 张**差（`NAIL_TOL`）；出这个界＝本脚本算错，不发读数。
     钉子是"和两本账自报的读数对账"，不是"预测"。这一发唯一的新信息是**它的 CI、同号计数与二项 p**。
     第三对（`中数据4500@12流`）在 `EX_ANTE` 里写的是 **None**：写这条预登记时 T=6 那本 12 条流的账
     还不存在（它本身就是那一发要买的东西）⇒ 那一格只要求"现算的斜率差 = 两本自报步长相减"这条**算术**，
     不许事后往空档里补一个看起来像预言的数字。第四对（`中数据4500@16流`）同理也是 **None**：
     它两本账都是第十四发补流买来的，写预登记时都不存在。
  ③ 出口：(a) CI 不含 0 ⇒ "斜率随温度变陡/变浅"从此是**以流为单位的配对读数**（升格）；
         (b) CI 含 0 ⇒ 这句话仍然只是两个点估计之差，欠账改写成"按 1/√n 还要再一倍流"。
     两档各自判，一升一不升要如实分开报，不许挑好的说。
  ④ 顺手多算的那一列（逐 α 的跨列配对差 T6−T3）不许事后挑：它回答"变陡发生在哪一端"。
     判法写死在 `cellwise_verdict`，只看小剂量格与阶梯顶格这两格的不含 0 模式，四个分支全覆盖：
     小格不含 0 而顶格含 0 ⇒ 变化发生在**小剂量端**、两列的顶差不多；顶格不含 0 而小格含 0 ⇒
     发生在**顶端**；两格都不含 0 ⇒ 整段一起平移（读不出哪一端）；两格都含 0 ⇒ 这一列什么都没判。

这一发买的是什么，要说清：它**本身既不是分辨率**（这个脚本一条流都不加，0 次训练）**也不是独立性**
（没换那七维里的任何一维）**也不是覆盖**（没加一格 α）——它是**第四种买法：同一批流上换一个判别量**
（把"两个点估计相减"改成"以流为单位的配对差"）。这正是上一轮给第4课g 那条债写下的"或者换一个判别量"
的做法第一次落地；同一套配对可以指向那一发（三档 × 两条样本批次），本轮不做。
**例外要单独记（两发都一样）**：第三对用的那本 12 条流的 T=6 账、以及第四对的两本 16 条流账，
都是第4课f 补流买来的（那两发买的是分辨率），本脚本只是把它们接进配对里 ⇒ 于是
"12 流那一对从 (b) 走到 (a)""16 流那一对也走 (a)"这两件事的**付费方都是补流，不是换判别量**；
两笔账各记各的，不许在总结里合成一句"斜率差被做硬了"。更不许因为 8/12/16 三对里有两对都过 (a)
就把它们并排成"三条独立证据"——它们是**同一个坐标的三次粗细读数**。

落地（跑完之后才写，只写实测到的事）：
  ① 头一版三道闸各钉住一次拆红，没有一档是装饰（第四档 `--poolblind` 是本轮新加的，实测在 ⑧）：
     `--red-check` 红在闸 C（"两本账的温度列都是 T=3"）；
     `--unlink` 红在闸 A，而**同一档里实测打印**出闸 B 只红 1 条子检查——「步长的逐流序列」，
     五个格均值与步长的 6 个统计量全绿 ⇒ 「B 不读键序所以看不见配对方向」这句话本轮被推翻了一半：
     B 的均值族确实不读键，但它的序列子检查读，所以 B 与 A 有重叠、不能互相替身；
     `--scramble` 红在闸 B（第一条就是 α=0.05 格均值现算 −72.8 ≠ 档案的 −73.8）。
  ② 预登记②的钉子对上了：中数据现算 +22.6 vs 档案相减 +22.7（双舍入 0.1 之内）、低数据 +52.9 = +52.9。
     于是"点估计不是发现"这句话本轮是**被对账了一次**，不是被引用了一次。
  ③ 两档一升一不升，如实分开报：**低数据** +52.9 张 CI [+33.1, +72.7]、8/8 同号、二项 p=0.0078
     ⇒ 出口 (a)：「加热把 α→0 曲线的斜率变陡」在这一档从此是以流为单位的配对读数；
     **中数据** +22.6 张 CI [−0.4, +45.7]、6/8 ⇒ 出口 (b)：仍然只是两个点估计之差。
     中数据的下界离 0 只差 **0.4 张**——σ 维持 27.5 时 n=12 就把下界抬到约 +5.1（按现 σ 折算，
     不是读数；上两轮补流实测过 σ 会涨）。而 T=3 那一列的 12 条流账 `_mid_n12_v4` **已经在册**，
     缺的只有 T=6 那一列 ⇒ 新落一条有明确买法的分辨率债：`step4f --tier 中数据4500 --t 6 --nseeds 12`
     之后重跑本脚本，一发判干净。
  ④ 逐 α 那一列读出来的东西比"变陡"两个字更有信息，而且**两档不是同一件事**：
     中数据 ⇒ 变化在小剂量端（α=0.05 跨列差 −28.1 张 CI [−42.3, −14.0] 不含 0，阶梯顶 −5.5 张跨 0），
     符号是 **T=6 那一列在最浅的一格少错 28 张** ⇒ "斜率更陡"全部来自加热把小剂量端救回来，
     不是顶上有额外伤害；低数据 ⇒ 变化在顶端（α=0.05 差 −3.1 张跨 0，α=0.3 +50.5、α=0.7 +49.8 都不含 0）
     ⇒ 整个高剂量平台被抬起来。同名的一句话在两档是两种机制，这条不许合并成一句"温度越高斜率越陡"。
  ⑤ 配对前提的覆盖度（本轮 D 之后）：**三对全部满池逐位核过**——8 流中数据 8/8、12 流中数据 12/12、
     低数据 8/8。低数据那一格从前是 3/8，缺的 5 条不是"跑不出来"，而是**检查读错了格子**：CE 读数
     本来就逐条躺在每一臂行的 `wrong_ce` 里，恒等闸名单只是它的 3 条抽样。零重训、零 API 费。
  ⑥ **第十三发（补流之后重跑）新增的第三对**：`中数据4500@12流` = `_mid_n12_v4`（T=3）vs
     `_mid_n12_t6_v5`（T=6），12 条流、CE 臂 12/12 逐位核过（这道闸第一次有了满覆盖的配对前提）。
     斜率差 **+27.3 张 CI [+12.0, +42.6]、10/12 同号、二项 p=0.0386 ⇒ 走出口 (a)**：③ 里那条
     "中数据只差 0.4 张"的分辨率债就此结掉——但**结它的是补流（分辨率），不是这个判别量变干净了**。
  ⑦ 同一坐标的粗细两档并排放：**8 条流 +22.6 张 [−0.4, +45.7]、σ=27.5；12 条流 +27.3 张
     [+12.0, +42.6]、σ=24.1**。点估计自己从 22.6 漂到 27.3（新加的 4 条流把斜率推得更陡），
     而 8 流那一版的预登记钉子 22.7 仍与 8 流那对逐位对得上 ⇒ 两对是**同一坐标的两次读数**，
     不许写成"斜率被复证了一次"。σ 反而缩了 3.4 张——这与上两轮"补流 σ 会涨"的读数不矛盾，
     因为这里配对的是**斜率差**（两端各自的噪声部分相消），不是单端步长。
  ⑧ 五档拆红（`--red-check` / `--unlink` / `--scramble` / `--poolblind` / 新增 `--ceblind`）本轮全部
     重跑，RC 都是 1（红着退出、永不入库）：`--red-check` 红在闸 C、`--unlink` 红在闸 A（且实测打印出
     闸 B 只红「逐流序列」那一条子检查）、`--scramble` 红在闸 B（α=0.05 格均值现算 −72.8 ≠ 档案 −73.8）、
     `--poolblind` 四步全如期（**这一句的 ③b 部分被 ⑪ 更正**）、`--ceblind` 四步全如期。
  ⑨ **D 这一发（CE 读数换来源）落地的两件事**：① 低数据那一 pair 的配对前提从 3/8 变 8/8；
     ② "覆盖不足"从"报个比例继续跑"升级成"当场退出"，于是"其余按同一构造推定"这句话在本脚本里
     从此没有落点（`gate_ce_pairing` 的 `short` 与 `load()` 的 `missing` 各拦一次，`--ceblind` 盯住两道
     不互相顶包）。三对的斜率差读数**一个数字都没动**（见下面的逐字段对账）——这一发改的是读哪一格，
     不是买新信息。
     两条回归核对，**各管一段**：① 改来源**之前**把产物指针临时改成盘上临时文件重跑正常模式，与在册的
     `compress_report_v4h_crosscol_v2.json` **逐字节相同**——它验的是上一轮 `gate_flow_pool` 的抽取
     （它原先住在 `gate_pair` 里）没动任何读数，临时件验完即删；② 改来源**之后**把新档案 v3 与 v2
     逐字段对账，实测到的差只有两处：`pairs` 里**仅**低数据的 `ce_pairing_checked` 3/8 → 8/8，
     `limitations` 里**仅**第 2 条那句"其余按推定"随来源作废而改写；`verdict` 与三对的全部数字键
     （斜率差 / 册内步长 / 逐 α 跨列差 / 钉子 / 两个出口 / 流池）**逐位相同** ⇒ "改来源不改读数"
     这句话现在是被对过账的，不是被声称的。
  ⑩ **第十四发的第四对**：`中数据4500@16流` = `_mid_n16_v5`（T=3）vs `_mid_n16_t6_v5`（T=6），
     16 条流、CE 臂 16/16 逐位核过。斜率差 **+27.3 张 CI [+14.8, +39.8]、13/16 同号、二项 p=0.0213、
     σ=23.4 ⇒ 走出口 (a)**。三件事必须分开说：
       · 点估计与 12 流那对**完全同值**（都是 +27.3）⇒ 这不是"又复证一次"，是同一坐标的第三次读数；
       · CI 半宽从 15.3 收到 12.5 张——这是**补流（分辨率）买来的**，不是判别量变干净了；
       · 逐 α 那一列的判法在两档之间**换了口径**：12 流判"变化在小剂量端"（阶梯顶 α=0.7 的跨列差
         CI [−21.5, +2.5] 跨 0），16 流判"整段一起平移"（同一格 CI [−18.9, −0.5] 不含 0，
         而它离 0 只有 **0.5 张**）⇒ "变陡发生在哪一端"这句话在两档分辨率下**不稳**，
         而且翻口的那一格是擦边翻的。两档各自如实报，不许合并成一句结论，更不许挑 16 流那版说。
  ⑪ **本轮五档拆红重跑抓到的两件事**（这一条是对 ⑧ 的更正，不是补充）：
     ① `--poolblind` 的 ③b 那一步**已经不按上一版的记录走**了。上一版写"关掉入口检查，同一本照样读进来"，
       实测却是：那本换了成员的账过了系数册之后**红在 CE 缺读数**那一道（池外种子在账里没有臂行）。
       RC 仍然是 1，所以只看退出码这件事永远不会暴露。本轮把这次退出**打印成实测**（它顺带证明 ③ 那条红
       不是 `check_flow_constants` 顶的包），并补 ③c：成员一字不差、只把名单**倒序** ⇒ 成员检查红，
       而同一本关掉成员检查后 CE 覆盖**全绿**（12/12）⇒ 成员检查钉的是**顺序**这一维，CE 那道看不见它，
       所以 ③b 的实测不能被读成"成员检查冗余"。
       **教训（与守卫 `--drift` 那一族同源）**：拆红档自己也会过期——每上线一道新闸，必须把**所有**旧拆红档
       重跑并逐笔核对"红在哪一道"，只数退出码等于没测。
     ② v4→v5 的逐字段对账：v4 落盘时 `limitations` 里写死了"三对 / 8 流两对 / 中数据两对"，第四对一进来
       就成了假话 ⇒ v5 把那三处改成从 `out["pairs"]` 现读派生。实测两本档案的差**只在 `limitations`**
       （7 条里 4 条改写），`pairs` 与 `verdict` 全部读数**逐字段相同** ⇒ 这一版升号没动任何读数。
本脚本 0 次训练，产物 `compress_report_v4h_crosscol_v5.json`（第一版叫 `..._crosscol.json`，第二版
`..._v2.json` 是因为第三对加了进来，第三版是第十三发 D 把配对前提的覆盖度从 3/8 提到整池，第四版
`..._v4.json` 加第四对（16 条流），第五版只因为 v4 的 `limitations` 里写死了配对计数——派生物不入库会被
覆盖，所以每换一次闸口强度或口径就换一次名字，而不是重跑盖掉旧判决；旧四本仍在盘上，可以逐字段对账）。
它是**纯读档的派生物**，不进守卫 `_red_v3.py` 的 `compress_report_v4f*` 在册账名单
（那八道闸盯的是训练路径与判词，这里没有训练路径）；它自己的守卫就是上面四道检查＋五档拆红，
这一条边界写在 `ml-compress/README.md` 里。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from step4_distill import check_flow_constants, summarize
from step4f_alpha_curve import SEEDS8, SEEDS12, SEEDS16

HERE = Path(__file__).resolve().parent
OUT = HERE / "compress_report_v4h_crosscol_v5.json"   # v4→v5：只因为 v4 的 limitations 里写死了"三对/两对"
                                                      # 计数，第四对（16 条流）一进来就作废 ⇒ 逐字段对账必须
                                                      # 证明只有 limitations 动过（v4 留盘当基准，不覆盖）
POOLS = {8: list(SEEDS8), 12: list(SEEDS12), 16: list(SEEDS16)}
# 每批的前 8 条就是板块共用的那批；16 那批的前 12 条就是 12 那批（逐位对账线在守卫闸⑥）

# 每对 = 同档 / 同宽 / 同任务 / 同一条流池（顺序也一致），只有温度列不同。布局差异只允许出现在这张表里。
PAIRS = {
    "中数据4500": ("compress_report_v4f_mid_n8_v4.json", "compress_report_v4f_mid_n8_t6_v4.json"),
    "低数据2000": ("compress_report_v4f_low_n8_thin_v3.json", "compress_report_v4f_low_n8_thin_t6_v4.json"),
    # 第十三发补进来的那一对：两本都是 12 条流，T=6 那本是补流买来的（这一对的**账**买了分辨率，
    # 而"斜率差"这个量本身仍是判别量——两件事不许混着报）。
    "中数据4500@12流": ("compress_report_v4f_mid_n12_v4.json", "compress_report_v4f_mid_n12_t6_v5.json"),
    # 第十四发那一对：两本都是 16 条流。它与上面两对是**同一坐标的第三次粗细读数**，
    # 不许并排成"三条独立证据"（预登记写在 step4f 头部第十四发那一节）。
    "中数据4500@16流": ("compress_report_v4f_mid_n16_v5.json", "compress_report_v4f_mid_n16_t6_v5.json"),
}
# 预登记②的算术钉子（抄自两本账各自的 paired_step_small_to_anchor，跑完不许改）。
# None = 写预登记时这本账还不存在 ⇒ 钉子只能是**算术**（现算的逐流差均值必须等于两本自报步长之差），
# 不许事后往这个格子里补一个"看起来像预言"的数字。
EX_ANTE = {"中数据4500": 22.7, "低数据2000": 52.9, "中数据4500@12流": None, "中数据4500@16流": None}
NAIL_TOL = 0.1 + 1e-9        # 预登记②：档案两个均值各自圆到 0.1 ⇒ 两次取整最多差 0.1 张


def excl0(s: dict) -> bool:
    """CI 不含 0。`summarize` 不供这个键，只能由 ci95_imgs 两端现算（键名写错＝静默假绿）。"""
    return bool(s["ci95_imgs"][0] > 0 or s["ci95_imgs"][1] < 0)


def load(name: str, unlink: bool = False, bump: int = 0, blind_pool: bool = False,
         drop_last: int = 0, swap_last: bool = False, ce_drop: int = 0, blind_ce: bool = False,
         split_ce: bool = False, id_shift: bool = False, rev_order: bool = False) -> dict:
    """读一本账：先核名单/档位/温度列，再把逐流序列摊成 {α: {seed: gap}}。

    `unlink` 只旋转**标签**不旋转数值：这正是「同名的种子在两列里其实不是同一条流」那个假说
    会长成的样子——册内读数（均值、册内步长）全都不动，只有跨册配对会散架。
    `bump` 给所有 KD gap 加常数（模拟取错档案/单位）。两者只服务拆红，正常跑时都是默认值。
    `blind_pool` 关掉"整本流名单等于在册某一池"这道入口检查。`drop_last` 把流名单末尾截掉若干条
    （流数不再在册），`swap_last` 只把最后一条换成池外种子（**流数仍在系数册上**，所以拦它的只可能
    是成员检查而不是 `check_flow_constants`——这一区分是这两档拆红不互相顶包的前提）。三者只服务拆红。
    `ce_drop` 抹掉前若干条流的 CE 读数（模拟"这一档只存了几条恒等闸"的薄档）、`split_ce` 让某一粒
    种子在不同 α 下报两个 CE 数（模拟 CE 其实随 α 变）、`id_shift` 把恒等闸那个来源挪 1 张
    （模拟同本账两个来源打架）；`blind_ce` 只关掉"缺读数就退出"那一道，用来证明前一条不是装饰。
    """
    p = HERE / name
    if not p.exists():
        raise SystemExit(f"缺 {name}——两列配对缺一本就没有「跨列」这回事，不许用一本账下一个斜率的判决")
    d = json.loads(p.read_text(encoding="utf-8"))
    seeds = [int(s) for s in d["seeds"]]
    if drop_last:
        seeds = seeds[:-drop_last]                     # 流数不再在册（11 条）
    if swap_last:
        seeds = seeds[:-1] + [seeds[-1] + 1]           # 流数在册、成员有一条不在池里
    if rev_order:
        # 成员一字不差、只有名单顺序变了。跨列配对是按位置 zip gap 的 ⇒ 顺序也算池的一部分。
        # （若同时关掉成员检查，下面的 cells 键序就刻意错位——③c 那一支只用它证明 CE 覆盖看不见顺序。）
        seeds = seeds[::-1]
    if not blind_pool and seeds != POOLS.get(len(seeds)):
        raise SystemExit(f"{name} 的流名单不是板块在册的某一池（{len(seeds)} 条流应等于 "
                         f"{POOLS.get(len(seeds))}）⇒ 跨列的差不能按流配对")
    check_flow_constants(len(seeds))
    cells = {float(a): dict(zip(seeds, [int(g) for g in v["gaps_imgs"]]))
             for a, v in d["ladder_summary"].items()}
    # CE 逐流读数的**来源**是每一臂行自己的 `wrong_ce`（CE 与 α 无关 ⇒ 一行就是一粒种子的读数），
    # 不是恒等闸名单——薄档只在老 3 条流上跑恒等闸，拿它当配对前提就把 8 条/12 条的配对压成 3 条。
    # 本轮（D）改在这里：整池的 CE 读数本来就在册，只是先前取错了格子。
    ce_rows: dict[int, set[int]] = {}
    for row in d["arms"].values():
        ce_rows.setdefault(int(row["seed"]), set()).add(int(row["wrong_ce"]))
    if split_ce:
        sd0 = seeds[0]
        ce_rows[sd0] = set(ce_rows[sd0]) | {next(iter(ce_rows[sd0])) + 1}
    for sd, vals in sorted(ce_rows.items()):
        if len(vals) > 1:
            raise SystemExit(f"{name}：种子 {sd} 的 CE 读数在不同 α 下有 {sorted(vals)} 个取值"
                             f"⇒ CE 臂与 α 无关这件事不成立，逐列配对的立足点没了")
    ce = {sd: next(iter(v)) for sd, v in ce_rows.items()}
    for sd in seeds[:ce_drop]:
        ce.pop(sd)
    missing = [sd for sd in seeds if sd not in ce]
    if missing and not blind_ce:
        raise SystemExit(f"{name}：{len(missing)} 条流（{missing}）没有 CE 读数 ⇒ 逐位配对缺原料，"
                         f"不许按流配对（要么退回不配对，要么先把这几条补跑）")
    id_gate = {int(e["seed"]): int(e["ce_wrong"]) for e in d["gate_alpha_zero_identity"]}
    if id_shift:
        first = min(id_gate)
        id_gate[first] += 1
    for sd, v in id_gate.items():                        # 两个来源必须报同一个数
        if sd in ce and ce[sd] != v:
            raise SystemExit(f"{name}：种子 {sd} 的 CE 读数在臂行里是 {ce[sd]}、在恒等闸里是 {v}"
                             f"⇒ 同本账自相矛盾，先查读的是哪一批数字")
    if unlink:
        rot = seeds[1:] + seeds[:1]
        cells = {a: dict(zip(rot, g.values())) for a, g in cells.items()}
        ce = dict(zip(rot, ce.values()))
    if bump:
        cells = {a: {sd: g + bump for sd, g in c.items()} for a, c in cells.items()}
    return {"name": name, "seeds": seeds, "dataset": d["dataset"], "tier": d["tier"],
            "width": int(d["width"]), "t": float(d["temps_fixed"]), "teacher_acc": float(d["teacher_acc"]),
            "probe_alpha": float(d["probe_alpha"]), "anchor_alpha": float(d["anchor_alpha"]),
            "floor": int(d["seed_floor_imgs"]), "cells": cells, "ce": ce, "id_gate": len(id_gate),
            "seed_cell_means": {float(a): float(v["mean_gap_imgs"]) for a, v in d["ladder_summary"].items()},
            "step": d["paired_step_small_to_anchor"]}


def gate_flow_pool(lo: dict, hi: dict, where: str, blind: bool = False) -> list[int]:
    """配对前提本身：两本账必须吃**同一批流**（成员与顺序逐位相同）。返回这一对实际用的流池。

    `blind=True` 只服务拆红：不拦，返回**交集**并打印被缩掉了几条——这一档要证的是"没有这道闸，
    脚本不会炸，只会把『12 条流那一对』悄悄当成 8 条流的配对发出去"。
    """
    if lo["seeds"] == hi["seeds"]:
        return lo["seeds"]
    shared = [s for s in lo["seeds"] if s in set(hi["seeds"])]
    msg = (f"{where}：两本账的流池不是同一批（{len(lo['seeds'])} 条 vs {len(hi['seeds'])} 条，"
           f"T3 独有 {[s for s in lo['seeds'] if s not in hi['seeds']] or '无'}、"
           f"T6 独有 {[s for s in hi['seeds'] if s not in lo['seeds']] or '无'}）")
    if blind:
        print(f"[实测] 关掉这道闸 ⇒ 不报错，按交集 {len(shared)} 条流继续配对：{msg}"
              f"——它**不炸**，只是悄悄把这一对的流池换成子集，等于换了分母")
        return shared
    raise SystemExit(f"{msg} ⇒ 逐位配对没有立足点，只能退回不配对、报两个均值之差")


def gate_pair(lo: dict, hi: dict, where: str) -> tuple[float, float]:
    """闸 C：两列必须**不同温**、其余判别变量必须**全同**、判据吃的那两格 α 必须都在两本账里。"""
    if lo["t"] == hi["t"]:
        raise SystemExit(f"{where}：两本账的温度列都是 T={lo['t']:g} ⇒ 斜率差恒为 0，这一发什么都没测")
    same = {k: (lo[k], hi[k]) for k in ("dataset", "tier", "width", "teacher_acc", "floor",
                                        "probe_alpha", "anchor_alpha") if lo[k] != hi[k]}
    if same:
        raise SystemExit(f"{where}：这两本不是同一格只换列（{same}）——判别变量没对齐，差值不属于任何假说")
    small, top = min(lo["cells"]), max(lo["cells"])
    if top != lo["anchor_alpha"] or hi["anchor_alpha"] != top or hi["probe_alpha"] != lo["probe_alpha"]:
        raise SystemExit(f"{where}：阶梯顶/锚点剂量两本不一致（{top} vs {hi['anchor_alpha']}）⇒ 尺子不同")
    if small not in hi["cells"] or top not in hi["cells"]:
        raise SystemExit(f"{where}：α={small:g}/{top:g} 不是两本账都有，跨列配对没有原料")
    return small, top


def gate_ce_pairing(lo: dict, hi: dict, where: str) -> str:
    """闸 A：CE 臂与温度无关 ⇒ 整池每一条流在两列里的 `ce_wrong` 必须**逐位相同**。

    本轮（D）把"核到多少报多少"改成硬要求：CE 读数的来源换成臂行的 `wrong_ce` 之后整池都在册，
    于是"按同一构造推定"这条降级路径**没有理由再存在**——覆盖不到整池就是读错了格子，退出。
    """
    both = sorted(set(lo["ce"]) & set(hi["ce"]))
    if not both:
        raise SystemExit(f"{where}：两本账没有共同的 CE 读数 ⇒ 配对前提一次都没核到（空表不算通过）")
    short = [f"{b['name']} 的 CE 读数只有 {len(b['ce'])}/{len(b['seeds'])} 条"
             for b in (lo, hi) if len(b["ce"]) != len(b["seeds"])]
    if short:
        raise SystemExit(f"{where}：CE 读数没覆盖整池（{short}）⇒ 这一对的差值只能退回不配对，"
                         f"不许一边按 8 条流配对、一边说『其余按推定』")
    bad = [s for s in both if lo["ce"][s] != hi["ce"][s]]
    if bad:
        raise SystemExit(f"{where}：CE 臂在流 {bad} 上两列不等 ⇒ 同名的种子不是同一条流，"
                         f"跨列差值不许按流配对（只能退回不配对、报两个均值之差）")
    return f"{len(both)}/{len(hi['seeds'])}"


def step_recalc(b: dict, small: float, top: float) -> dict:
    """册内那条 α=小剂量→阶梯顶 的配对步长，从盘上的逐流 gap 现算（闸 B 与读数共用这一份代码）。"""
    return summarize([b["cells"][top][sd] - b["cells"][small][sd] for sd in b["seeds"]], "α=小→顶 配对步长")


def gate_recalc(b: dict, where: str, collect: bool = False) -> list[str]:
    """闸 B：现列重算必须逐位命中那本账自己记着的读数（每格均值 + 那条小剂量→阶梯顶的步长）。

    `collect=True` 时不退出，把违规攒成名单返回——--unlink 要靠它打印「B 的哪几条子检查动了」。
    """
    bad: list[str] = []

    def check(what: str, got, want) -> None:
        if got != want:
            bad.append(f"{what}：现算 {got} ≠ 档案记的 {want}")

    for a, recorded in b["seed_cell_means"].items():
        check(f"α={a:g} 格均值", round(sum(b["cells"][a].values()) / len(b["cells"][a]), 1),
              round(recorded, 1))
    small, top = min(b["cells"]), max(b["cells"])
    s = step_recalc(b, small, top)
    for k in ("mean_gap_imgs", "sd_imgs", "se_imgs", "range_imgs", "d4_sigma_hat", "p_seeds_binomial"):
        check(f"α={small:g}→{top:g} 步长的 {k}", s[k], b["step"][k])
    check("步长的逐流序列", [int(x) for x in s["gaps"]], [int(x) for x in b["step"]["gaps"]])
    if bad and not collect:
        raise SystemExit(f"{where}：{bad[0]} ⇒ 读错账（取错格子/串档/单位不对），这条读数不发")
    return bad


def cellwise_verdict(diff: dict, small: float, top: float) -> str:
    """预登记④：判法写死——只看小剂量格与阶梯顶格这两格的跨列差不含 0 的模式。"""
    a, b = excl0(diff[small]), excl0(diff[top])
    if a and not b:
        return "跨列的变化发生在**小剂量端**，两列的顶差不多"
    if b and not a:
        return "跨列的变化发生在**阶梯顶那一端**"
    if a and b:
        return "两格都不含 0 ⇒ 整段一起平移（读不出发生在哪一端）"
    return "两格都含 0 ⇒ 这一列什么都没判"


def probe_gates(lo: dict, hi: dict, where: str) -> None:
    """--unlink 专用：把 A、B 各自的反应实测打印出来，再红着退出（这一支永不入库）。

    为什么要替 B 也报一次：「B 看不见配对结构」是我上一版**口述**的断言，而 B 的逐流序列子检查
    其实读键序。只有跑出来才知道 B 到底动了哪几条——不跑就是把一句没验过的话留在仓库里当证据。
    """
    msg_a = ""
    try:
        gate_ce_pairing(lo, hi, where)
    except SystemExit as e:
        msg_a = str(e)
    v_b = gate_recalc(hi, where, collect=True)
    print(f"[实测] 闸 A：{'红（如期）' if msg_a else '不红 ← 这一档没测到，配对前提是装饰'}")
    if msg_a:
        print(f"        {msg_a}")
    print(f"[实测] 闸 B：{len(v_b)} 条子检查红（名单如下；一条都没红＝B 不读键序，那它就不能替 A 作证）")
    for v in v_b:
        print(f"        {v}")
    if not msg_a:
        raise SystemExit(f"{where}：--unlink 没能拆红闸 A")
    raise SystemExit(f"{where}：--unlink 验红结束（红的是闸 A），不发读数、不入库")


def probe_pool() -> None:
    """--poolblind 专用：逐笔实测本轮新加的两道流池检查，红着退出、永不入库。

    ③b / ③c 是第十四发改的：原先那句"关掉入口检查，同一本截断账照样读进来"在 CE 覆盖检查上线之后
    **不再成立**（池外种子在那本账里没有臂行），于是不重跑就没人知道这一支红在了别的闸上。
    为什么必须实测：这两道检查拦的是**静默换分母**——没有它们脚本不会炸，只是把"12 条流那一对"
    悄悄配成 8 条流的子集，读数照样打印出去。跑得绿的闸不叫闸，所以"在位时该红"与"拆掉后不该炸"
    两头都要各跑一次。
    """
    n12, n8 = "compress_report_v4f_mid_n12_v4.json", "compress_report_v4f_mid_n8_t6_v4.json"
    where = f"流池拆红（{n12} vs {n8}）"
    lo, hi = load(n12), load(n8)
    if lo["t"] == hi["t"]:
        raise SystemExit(f"{where}：这两本不是不同列（都 T={lo['t']:g}）——换错了账，这一档没测到同池检查")
    try:
        gate_flow_pool(lo, hi, where)
    except SystemExit as e:
        print(f"[实测①] 同池检查在位：红（如期）——{e}")
    else:
        raise SystemExit(f"{where}：12 条流配 8 条流居然过关 ⇒ 同池检查是装饰")
    shared = gate_flow_pool(lo, hi, where, blind=True)
    if len(shared) != len(SEEDS8):
        raise SystemExit(f"{where}：拆掉闸后用池是 {len(shared)} 条，既不是 12 也不是 8"
                         f"⇒ 换分母的方式与预期不符，①那条红的含义要重写")
    print(f"[实测②] 拆掉同池检查：不红、也不炸，只是把用池换成 {len(shared)} 条 ⇒ ① 那条红承重")
    try:
        load(n12, swap_last=True)
    except SystemExit as e:
        print(f"[实测③] 在册池成员检查（换掉最后一条 ⇒ 条数仍是 12）：红（如期）——{e}")
    else:
        raise SystemExit(f"{where}：换掉一条成员仍被认成在册池 ⇒ 入口的成员检查是装饰")
    try:
        load(n12, swap_last=True, blind_pool=True)
    except SystemExit as e:
        print(f"[实测③b] 关掉入口成员检查：这本换了成员的账**过了系数册**（12 条仍在册）却红在下一道 ⇒ "
              f"{str(e).splitlines()[0]}\n        "
              f"两件事同时测到：③ 那条红不是 `check_flow_constants` 顶的包；池外种子还有第二道兜底"
              f"（它在{len(SEEDS12)}本账里没有臂行，CE 逐流读数凑不齐）")
    else:
        raise SystemExit(f"{where}：关掉成员检查后池外种子一路读到底 ⇒ ③ 与 CE 两道都没承重")
    try:
        load(n12, rev_order=True)
    except SystemExit as e:
        print(f"[实测③c] 成员一字不差、只把名单倒序：入口成员检查红（如期）——{str(e).splitlines()[0]}"
              f" ⇒ 配对是按位置 zip gap 的，顺序也算池的一部分")
    else:
        raise SystemExit(f"{where}：只换名单顺序被认成同一池 ⇒ 成员检查漏了顺序这一维")
    b_rev = load(n12, rev_order=True, blind_pool=True)
    print(f"[实测③c-补] 同一本倒序账关掉成员检查后 CE 覆盖**全绿**（{len(b_rev['ce'])}/"
          f"{len(b_rev['seeds'])} 条都有读数）⇒ CE 那道看不见顺序，成员检查不是它的替身")
    try:
        load(n12, drop_last=1)
    except SystemExit as e:
        print(f"[实测④] 截短一档（11 条）：红（如期）——{str(e).splitlines()[0]}")
    else:
        raise SystemExit(f"{where}：11 条流被认成在册池 ⇒ 入口对流数的检查是装饰")
    raise SystemExit(f"{where}：①②③③b③c④ 全部如期 ⇒ 流池检查与 CE 覆盖检查各管一段"
                     f"（成员/顺序 vs 原料齐不齐），互不顶包；不发读数、不入库")


def probe_ce() -> None:
    """--ceblind 专用：实测本轮（CE 读数改从臂行取）新增的三条检查，红着退出、永不入库。

    这一发改的是**读哪一格**：薄档只存 3 条恒等闸，所以配对前提一直只核到 3/8；而每一臂行本来就
    带着自己那粒种子的 `wrong_ce`，整池都在册。改完之后"核不到的流按同一构造推定"这句话失去理由，
    于是配了三条新检查——检查没有拆红档就等于没有，所以两头各跑一次。
    """
    n8, nt6 = "compress_report_v4f_mid_n8_v4.json", "compress_report_v4f_mid_n8_t6_v4.json"
    where = f"CE 读数拆红（{n8} vs {nt6}）"
    try:
        load(n8, ce_drop=2)
    except SystemExit as e:
        print(f"[实测①] 缺读数的账不许按流配对：红（如期）——{e}")
    else:
        raise SystemExit(f"{where}：抹掉 2 条 CE 读数仍被当成满覆盖 ⇒ ①是装饰")
    lo_short = load(n8, ce_drop=2, blind_ce=True)
    hi = load(nt6)
    try:
        gate_ce_pairing(lo_short, hi, where)
    except SystemExit as e:
        print(f"[实测②] 关掉入口那道之后闸 A 仍拦住覆盖不足：红（如期）——{e}")
    else:
        raise SystemExit(f"{where}：入口检查拆掉、闸 A 也没拦住 ⇒ 这两道互为备份但一起失灵")
    try:
        load(n8, split_ce=True)
    except SystemExit as e:
        print(f"[实测③] 同一粒种子在不同 α 下报两个 CE 数：红（如期）——{e}")
    else:
        raise SystemExit(f"{where}：CE 与 α 无关这条前提没有检查 ⇒ 「CE 臂与温度无关」是口头的")
    try:
        load(n8, id_shift=True)
    except SystemExit as e:
        print(f"[实测④] 臂行与恒等闸两个来源打架：红（如期）——{e}")
    else:
        raise SystemExit(f"{where}：同本账自相矛盾没人管 ⇒ ④是装饰")
    raise SystemExit(f"{where}：四步全部如期 ⇒ 换来源之后新增的三条检查都承重；不发读数、不入库")


def main(mode: str = "") -> None:
    unlink, scramble = mode == "unlink", mode == "scramble"
    if mode == "poolblind":
        print("[验红] --poolblind：拿 12 条流那本配 8 条流那本，实测两道流池检查各拦一次（永不入库）")
        probe_pool()
    if mode == "ceblind":
        print("[验红] --ceblind：实测 CE 读数换来源之后新增的三条检查（永不入库）")
        probe_ce()
    if mode == "red-check":
        # 验红：把中数据那一对里的 T=6 账换成另一本 T=3 账——档位对、宽度对、流对，只有温度列不对。
        # 若闸 C 恒绿，「两列」这两个字就是装饰。跑通即证明它会红着退出（不写档案）。
        PAIRS["中数据4500"] = (PAIRS["中数据4500"][0], "compress_report_v4f_mid_n8_v3.json")
        print("[验红] --red-check：中数据那一对的第二本换成 T=3 的 `_n8_v3`，闸 C 必须拦下")
    if unlink:
        print("[验红] --unlink：只旋 T=6 那本的种子标签 ⇒ 红的第一道必须是闸 A；闸 B 动了哪几条一并打印")
    if scramble:
        print("[验红] --scramble：只给 T=6 那本的 gap +1 ⇒ 只有闸 B 该红（闸 A 只看 CE 臂，该照旧绿）")
    if OUT.exists() and not mode:
        # 验红那一支永不入库（它必须走到读数断言），所以不能被这道闸先挡掉。
        raise SystemExit(f"{OUT.name} 已入库——重跑会盖掉这本跨列配对的判决（要重跑得先改名或删档）")

    out = {"quantity": "同一条流池上，T=6 那一列减 T=3 那一列（单位：张/10000，正=T=6 更往多错走）"
                       "· 每对用的是哪一池见 pairs[*].flow_pool",
           "seed_pools": {str(k): v for k, v in POOLS.items()}, "mode": mode or "（正常跑）",
           "prediction_ex_ante_imgs": dict(EX_ANTE), "pairs": {}}
    for tier, (n_lo, n_hi) in PAIRS.items():
        # 两列只有一本被动：两本同向旋转等于没旋，两本同加常数等于差里没动。
        lo = load(n_lo)
        hi = load(n_hi, unlink=unlink, bump=1 if scramble else 0)
        where = f"{tier}（{n_lo} vs {n_hi}）"
        pool = gate_flow_pool(lo, hi, where)            # 配对前提：两本吃同一批流（成员与顺序全同）
        small, top = gate_pair(lo, hi, where)           # 闸 C：不同温 + 其余判别变量全同
        if unlink:
            probe_gates(lo, hi, where)                  # 只有这一档替 A/B 各报一次实测再退出
        cov = gate_ce_pairing(lo, hi, where)            # 闸 A：配对前提
        gate_recalc(lo, where)                          # 闸 B：两本各对一次账
        gate_recalc(hi, where)
        if scramble:
            raise SystemExit(f"{where}：--scramble 走完三道闸居然没红 ⇒ 闸 B 是装饰")
        steps = {tag: step_recalc(b, small, top) for tag, b in (("T3", lo), ("T6", hi))}
        diff = {a: summarize([hi["cells"][a][sd] - lo["cells"][a][sd] for sd in pool], f"α={a:g} 跨列差")
                for a in sorted(set(lo["cells"]) & set(hi["cells"]))}
        slope = summarize([(hi["cells"][top][sd] - hi["cells"][small][sd])
                           - (lo["cells"][top][sd] - lo["cells"][small][sd]) for sd in pool],
                          "斜率差（T6 步长 − T3 步长，按流配对）")
        nail = round(steps["T6"]["mean_gap_imgs"] - steps["T3"]["mean_gap_imgs"], 1)
        if EX_ANTE[tier] is not None and abs(nail - EX_ANTE[tier]) > 1e-9:
            raise SystemExit(f"{where}：两本账自报步长相减得 {nail}，与预登记②写的 {EX_ANTE[tier]} 不符"
                             f"⇒ 读错账（串档/取错 α/拿错流池），这条读数不发")
        if abs(slope["mean_gap_imgs"] - nail) > NAIL_TOL:
            raise SystemExit(f"{where}：斜率差的点估计 {slope['mean_gap_imgs']} 对不上预登记②——档案自报步长"
                             f"相减得 {nail}、钉子写的是 {EX_ANTE[tier]}、双舍入容差 {NAIL_TOL:g}"
                             f" ⇒ 本脚本算错了，这条读数不发（钉子是与账对账，不是预测）")
        exit_slope = "(a) 升格成以流为单位的配对读数" if excl0(slope) else "(b) 仍只是两个点估计之差"
        exit_cell = cellwise_verdict(diff, small, top)
        out["pairs"][tier] = {"archives": {"T3": n_lo, "T6": n_hi}, "flow_pool": len(pool),
                              "criterion_alphas": [small, top],
                              "ce_pairing_checked": cov, "steps_within_column": steps,
                              "cross_column_cellwise": {f"α={a:g}": s for a, s in diff.items()},
                              "slope_diff_paired": slope, "ex_ante_nail_imgs": nail,
                              "exit_slope": exit_slope, "exit_cellwise": exit_cell}
        print(f"\n[{tier}] {len(pool)} 条流 · 配对前提：CE 臂 {cov} 条逐位核过（闸 A 要求整池，缺一条就退出）"
              f" · 判据吃的两格 α={small:g}/{top:g}")
        for tag in ("T3", "T6"):
            st = steps[tag]
            print(f"   册内步长 {tag}：{st['mean_gap_imgs']:+.1f} 张 CI [{st['ci95_imgs'][0]:+.1f},"
                  f" {st['ci95_imgs'][1]:+.1f}] 同号 {st['same_sign_seeds']}（档案原读数，闸 B 已逐位对过）")
        print(f"   斜率差（按流配对）{slope['mean_gap_imgs']:+.1f} 张（预登记钉子 {nail:+.1f}）"
              f" CI [{slope['ci95_imgs'][0]:+.1f}, {slope['ci95_imgs'][1]:+.1f}] 同号 {slope['same_sign_seeds']}"
              f" 二项 p={slope['p_seeds_binomial']} sd={slope['sd_imgs']} 极差 {slope['range_imgs']} 张 ⇒ {exit_slope}")
        for a, s in diff.items():
            print(f"   α={a:<5g} 跨列差 {s['mean_gap_imgs']:+7.1f} 张 CI [{s['ci95_imgs'][0]:+.1f},"
                  f" {s['ci95_imgs'][1]:+.1f}] 同号 {s['same_sign_seeds']}"
                  f" {'不含 0' if excl0(s) else '跨 0'}")
        print(f"   逐 α 那一列按预登记④的判法 ⇒ {exit_cell}")

    out["verdict"] = ("；".join(f"{t} 的斜率差走出口 {v['exit_slope']}" for t, v in out["pairs"].items())
                      + " ——两档各自判，一升一不升如实分开报")
    mid = [t for t in out["pairs"] if t.startswith("中数据")]
    paid = [f"{t}（{out['pairs'][t]['archives']['T6']}）" for t in out["pairs"]
            if "_n12_" in out["pairs"][t]["archives"]["T6"] or "_n16_" in out["pairs"][t]["archives"]["T6"]]
    cover = "、".join(f"{t} {v['ce_pairing_checked']}" for t, v in out["pairs"].items())
    out["limitations"] = [
        "这一发**读的是判别量**：把「两个点估计相减」换成「逐流配对差」，它本身不换那七维里的任何一维，"
        "所以斜率差再干净也**永不构成复证**。唯一的例外在流池上："
        f"{'、'.join(paid)} 这几对的 T=6 那本账是 step4f 补流买来的（买的是分辨率）⇒ 那些格里"
        "「分辨率」与「判别量」两件事各买各的，不许混着报",
        "配对前提的覆盖度（每对的 `ce_pairing_checked`）本轮起是**整池**，现读：" + cover +
        "。来源是每一臂行自己的 `wrong_ce`（CE 与 α 无关），不再是恒等闸名单——薄档只存 "
        "3 条恒等闸时那句「其余按同一构造推定」随来源一起作废：现在覆盖不到整池是**退出**，不是报个比例",
        f"中数据那 {len(mid)} 对（{'、'.join(mid)}）是**同一个坐标的粗细几档读数**：粗那档的流池是细那档的前缀"
        "（守卫闸⑥ 已把这些格钉成逐位相同）⇒ 几对的斜率差不算几次独立复证，只能并排看收敛",
        "低数据那一对的两本账跨判据版本（T=3 是 `_v3`、T=6 是 `_v4`）：配的是**训练路径上的逐流 gap**，"
        "判据版本只改派生判词不改 gap，所以可比；但这里的「同一把尺子」=同一个训练器，不是同一个判据",
        "闸 A 只核到 CE 臂那一格：两列的 CE 读数逐位相同，恰恰因为 CE 与 T 无关。它核到「同一次初始化、"
        "同一批图」，核不到 KL 那一侧——这是配对前提的下界，不是全证",
        "斜率差的点估计由算术钉死（见 prediction_ex_ante_imgs 与每对的 ex_ante_nail_imgs），"
        "所以它不是发现；发现只能是 CI / 同号计数 / 二项 p",
        f"{len(out['pairs'])} 对读数全部来自 MNIST × S128 这几个坐标、且共用同一个老师与同一批图 ⇒ "
        "「加热把斜率变陡」跨不跨任务、跨不跨档、跨不跨宽度，本脚本一个字都没说",
    ]
    if mode:
        raise SystemExit(f"[验红模式 {mode}] 居然走到了发读数这一步 ⇒ 至少一道闸没拦住，不发不入库")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {OUT.name}（纯读档：0 次训练，零 API 费）· {out['verdict']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"两列温度之间的斜率差：纯读 {len(PAIRS)} 对"
                                             f"（{len(set(sum(PAIRS.values(), ())))} 本）已入库的账，按流配对")
    ap.add_argument("--red-check", action="store_true", help="把 T=6 那本换成另一本 T=3 账，验闸 C 会红")
    ap.add_argument("--unlink", action="store_true", help="旋转 T=6 的种子标签，验闸 A（配对前提）会红")
    ap.add_argument("--scramble", action="store_true", help="给 T=6 的 gap 全 +1，验闸 B（复算对账）会红")
    ap.add_argument("--poolblind", action="store_true",
                    help="拿 12 条流那本配 8 条流那本，验流池检查会红（拆掉后不炸只换分母；名单倒序也算换池）")
    ap.add_argument("--ceblind", action="store_true",
                    help="实测 CE 读数（换自臂行）相关的三条检查：缺读数/两个 CE 数/两个来源打架")
    a = ap.parse_args()
    main(mode="red-check" if a.red_check else "unlink" if a.unlink else "scramble" if a.scramble
         else "poolblind" if a.poolblind else "ceblind" if a.ceblind else "")
