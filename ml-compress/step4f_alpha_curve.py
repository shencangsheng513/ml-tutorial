# -*- coding: utf-8 -*-
"""第4课c 欠的最后一发：把 α 从 0.3 往下探到 0，看伤害是"剂量"还是"污染"。

第4课c 扫了 α∈{0.3, 0.7, 1.0}（T=3 那一行实测 −72.0 / −75.0 / −76.7 张），判词是
"α 轴平、斜率 −4.7 张 = 底的 0.11x"。但**平在 0.3~1.0 之间，不等于平在 0~0.3 之间**——
稀释说（H1：伤害 ∝ 给了软标签多少权重）没有被那场扫参杀掉，它只是退到了最后一块地方：

  H1 的残城：老师项是**按份额**掺进来的毒，α=0.05 只该坏 5 张（−75 × 0.05/0.7），
            α=0.3 才坏 32 张。那么从 0 走到 0.3 这一段必须是**陡的**。
  污染说（H1 已死）：softmax 目标分布一旦参与，学生学到的就是那副分布的**形状**，
            份额只决定它多快生效。α=0.05 就该看到接近 −72 的伤害，而 0.3→1.0 加不了多少。

两说的分歧点只有一个数字：**α=0.05 那一格离 0 近，还是离 −72 近。**
（H1 预测 −5.4 张，污染说预测 −70 张开外，相差 13 倍——这是本板块最便宜的一次判决。）

两件免费仪器，训练前立好：
  · **α=0 恒等闸**：loss 在 α=0 时按定义就是纯 CE。它必须与 CE 臂**逐位相同**
    （错图数一样、与 CE 的不同预测数=0）。这不是一个臂，是 α 这根线**接没接对**的证明——
    接错了下面五格全是废数。
  · **α=0.3 跨脚本闸**：第4课c 已经跑过这一格并把逐种子读数存进 v4c 档案，
    本课重跑一遍，与档案逐位对账。对不上说明新增的这条路径动了 train_student 的调用方式。

判据（训练前写死，此后不许改口）：小 α 那一头每格的均值 gap 都低于种子底（41 张），
所以**不能用"过不过底"当判决**——那等于宣布这一发没判决。用逐图配对：同一副权重、同一批数据，
两臂只有目标函数不同，翻坏/翻正的不对称比"重训抖动"敏感得多（第4课的老道理）。
  · 若 α≤0.1 两格配对检验不出方向（p>0.05 且坏≈正）而 α=0.2/0.3 出方向 → 有阈值，H1 半活；
  · 若 α=0.05 就同号且合并 p 显著 → 稀释说最后一块地没了，伤害是**污染型**。
零 API 费，纯 CPU。跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4f_alpha_curve.py
  # 换档（档案驱动：底与 α=0.3 基准都从那一档自己的账里读）：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4f_alpha_curve.py --tier 中数据4500

第二发（`--dataset fashion`）问的是这条判决的**适用范围**："伤害是污染型不是剂量型"目前只站在
MNIST 的两副学生上（S128 79% / S64 63%），任务那一维一次都没换过。换任务要连三样东西一起换，
而且**一样都不许抄**：
  · **温度列**：MNIST 这条阶梯铺在 T=3（锚点温度、也是第4课c 判定的伤害列）上，可 Fashion 那本账
    只有 T∈{1,6}。所以阶梯所在那一列由 `max(v6.temps)` 现读 = **T=6**（伤害最大那一端，
    a1_T6 = −141.0 张），不是把 3.0 抄过来——抄过来会得到一格不存在的锚点、以及一个对不上的闸。
  · **锚点剂量与跨脚本闸**：Fashion 没跑过 α=0.3，那本账里最小的 α 是 0.7 ⇒ 阶梯的最高一档
    （=锚点=闸格）取 `min(v6.alphas)` = **0.7**，它同时是 H1 直线的那把尺子和逐位对账的基准。
    于是这一发比 MNIST 那一发多一格（阶梯 0.0/0.05/0.1/0.2/0.3/**0.7**），而"锚点格就是闸格"
    这件事必须写在报告里：它意味着这一发没有独立于锚点的第二条对账线。
  · **底**：读 Fashion 该档自己的 own floor（满数据档 112 张），不借 MNIST 的 41/37。
α=0 恒等闸跨任务照样成立（它测的是接线，不是世界）。产物 `compress_report_v4f_fashion.json`。
跑法：
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 /c/Users/shenh/miniconda3/python.exe step4f_alpha_curve.py --dataset fashion --tier 满数据8500

第三发（`--dataset fashion --tier 满数据8500 --nseeds 8 --thin --t 1`）补的是第二发自己留下的那条边界：
上一发的阶梯铺在 `max(v6.temps)` = T=6 上，而 v6 那本账里 α=0.7/T=1 那格读数是 **+44.7 张（正号：
KD 比 CE 少犯错）**。所以"有阈值"当时只是 **T=6 那一列**的形状，不能往别的温度列搬。把阶梯搬到 T=1，
两说的分歧点照旧只有一个数字，只是符号反了：
  · 污染说 → α=0.05 就该吃到 +44.7 量级的增益（这一列的阶梯是平的）；
  · H1 直线 → α=0.05 只有 +3.2 张（这一列的阶梯是陡的）。
`--t` 换的是**列**不是档，所以档案名带 `_t1`，且 `check_t` 只允许开在这一本账已有的温度列上：
拿一本没有那一列的账铺阶梯，锚点格与闸格都会凭空造出来（最坏的一种绿）。产物
`compress_report_v4f_fashion_n8_thin_t1.json`。

第四发（`SHAPE_RULE = "v2"`）还的是**自己欠的一条口径债**，判据写在重跑之前：
v1 的形状判据只吃**逐图合并 p**（`p_pooled`）。于是 T=1 那一列出了一种分裂读数、脚本照样印了"有阈值"：
α=0.7 那格合并 p=0.030（显著）却只有 5/8 条流与本格均值同号，配对步长 CI [−35.0,+71.8] 跨 0。
病根在检验单位：合并 p 是在**同一批 10000 张测试图**上数翻牌，8 条流共用的正是那一批图——它再小也只
证明"这条流内部稳"，不证明"换个随机源还稳"，而本课从头到尾的复现单位是**流**（第4课e 立的规矩）。
  · v2：一格算"**有方向**" ⟺ 逐图合并 p < 0.05 **且** 全部流同号（任何一条流反向即判没方向）。
    小剂量格有方向 → 污染型；只有锚点格有方向 → 有阈值；两端都没有 → 测不动。
  · 判据先对着**六本已入库档案**离线验红（零训练）：v1 复算必须逐本等于档案里那句判词（证明喂的是
    同一批数字），v2 必须**只**翻掉 T=1 那一本，另五本判词一字不动。谁都翻不动的判据是摆设，
    连坐翻掉板块最硬结论的判据是把债转成损失——两个方向都算验红失败。
  · **验红第一版真的红了**：当时把流级那条腿写成"流级 p < 0.05"，结果六本翻了五本。原因不是判据太严，
    是我算错了它的松紧——n=3 时全同号的精确二项 p 只有 0.25（要到 6/6 才 0.031，8/8 是 0.0078，
    而 7/8 回到 0.0703）。固定 p 阈值等于宣布"3 条流的实验一律不配拥有方向"，那是把三条『污染型』
    判词一起没收。改成计数形式（k == n）之后：n=8 上两版等价，n=3 上退化成 v1 本来就在用的"没有反例"。
    这一步不算事后改判词——改的是一条**内部不自洽**的判据（我在预登记里就把 3/3 写成了 p<0.05），
    而不自洽是验红抓出来的，不是读数难看抓出来的。留在这里的代价是明写的：v2 的流级腿在 3 条流上
    只有"无反例"那么松，想让它真有意义，得把老档案扩到 8 条流（那是**补流**，不是再换一次判据）。
  · 产物名带 `_v2`：判据版本跟 --tier / --width / --t 一样是判别变量，必须落进档案名，否则这一本的判决
    会无声盖掉 v1 那一本（而 v1 那一本记的正是这笔债的现场）。

第五发（`--tier 低数据2000 --nseeds 8 --thin`）把阶梯搬到**数据量**这一维——形状判词此前只跨过任务、
架构、温度列，从没跨过档。这一发交出的两样东西都不是"形状答案"：

  ① **一个仪器 bug，被自己的闸红着抓住**：薄阶梯原先写成"只留最小剂量格 + 锚点格"，于是把跨脚本闸格
    （MNIST 的 α=0.3）一起砍掉了 ⇒ 那发跑完两条闸里只剩恒等闸，"空表全过"那道硬闸当场红着退出。
    修法：薄化只许砍**不参与配对差**的中间格，闸格永不砍（Fashion 那两本恰好闸格=锚点格，去重后元组
    不变 ⇒ 已入库的薄账仍可逐位复跑）。
  ② **判据的视野缺陷**：形状判据吃的是（最小剂量格, 闸格）两格，而 MNIST 的闸格是 α=0.3、阶梯顶是
    α=0.7。这一发阶梯顶实测 8/8 同号、+29.8 张（按 v2 两条腿它**有方向**），判据却压根没看它，于是
    印出"测不动"。所以这一本的判词记的是**仪器的视野**，不是这一列的可读性。
  读数本身很干净：α=0.05 = −18.9 张（7/8 同号、逐图 p=1e-06）、α=0.3 = +9.6、α=0.7 = +29.8，
  配对步长 α=0.05→0.7 = +48.6 张、CI [+33.6,+63.6]、8/8。**小剂量端与锚点端反号**：污染说预言两格
  同量级、H1 过原点直线预言两格同号且按比例，两说都不合；留下的形状是"带负截距的上升线"。
  这一句是**描述性**读数，不在预登记的三分支里 ⇒ 不改判词，只记成 v3 的债：v3 必须吃阶梯顶，
  并给"两格反号"一个正式分支（否则每一发都得靠人现场解释"测不动其实读得出来"）。
  顺带修掉一处自欺：『第一格买到锚点伤害的 x%』用的是 abs/abs，两格反号时会把"方向相反"印成
  "买到了 89%"——现在同号才报份额，反号直接报反号。

第六发（还账时读回自己刚写的两本档案，抓出**同一量两个源**）：v2 那两本的同一页上出现过
18.3 与 18.4（都是 α=0.05→0.7 那一步），低数据档那一本的两段之和是 48.7 而整段配对步长是 48.6。
病根三处，都不在训练路径上：
  · 分段步长是"两格各自已四舍五入的均值再相减"，而配对步长是"同批种子上先相减再平均"——
    两个都合法，却不该同时出现在同一页上指同一件事。改：分段也走配对（同种子差），于是
    lo+hi 恒等于整段配对步长（已拿两本档案的 gaps 离线核过：18.4/18.4、28.5+20.1=48.6）。
  · "判决的分量"与份额拿 `anchor_mean` 当锚点高度，而它是**借来的 3 条流**档案读数（T=1 列 +44.7、
    低数据档 +21.3），同一句话里的小剂量格却是本格的 8 条流——薄阶梯把锚点格搬进本次训练之后，
    本格就有锚点读数（+17.1 / +29.8），不必再借。改：份额与反号那句一律读本格的阶梯顶；
    `anchor_mean` 退回它正当的用途＝训练前写死的 H1 先验（h1_line 与"污染说预测"那一行仍吃它，
    那是预测，不是读数）。代价：两本档案各重跑一次（gaps/配对步长/两条闸/arms 已逐位证明不变
    ⇒ 重跑只改派生量），旧档留 `_prev_*.json` 到新档对完账为止。
  · 同一句话还有第三处错位：它拿**闸格**（s_probe）当"锚点格"。Fashion 薄账上两者恰好同格所以没露馅，
    MNIST 的闸格是 α=0.3 ⇒ 反号分支印成"α=0.05 买到锚点伤害的 197%"。改：份额与同号一律对
    `summary[ANC]`（本格的阶梯顶），低数据档那本随之从 197% 修成 63%。

第七发（`SHAPE_RULE = "v3"`）还的是第五发自己记下的那笔债，判据写在重跑之前。v2 有两条视野缺陷，
都不是"腿太松"，是**压根没看那一格**：
  ① 判据只吃（最小剂量格, 闸格）两格。MNIST 的闸格是 α=0.3、阶梯顶是 α=0.7，于是低数据档那一发
    阶梯顶实测 8/8 同号、+29.8 张（按 v2 两条腿它有方向）却进不了判决，印出来是"测不动"——
    那句判词记的是仪器的视野，不是那一列的可读性。
  ② "两格反号"没有正式分支。α=0.05 = −18.9（逐图 p=1e-06）与阶梯顶 +29.8 反号，污染说预言两格
    同量级、H1 过原点直线预言两格同号且按剂量比例，两说都不合；可 v2 的三分支里没有"反向"这一格位置，
    只能靠人在判词外面补一段描述性读数。描述性读数不进判词 = 下一轮没人能复算它。
  · v3 的分支顺序（写死，此后不许改口；"有方向"的定义**逐字不动**＝逐图合并 p<0.05 且全部流同号）：
      1 污染型：最小剂量格自己就有方向、且方向与锚点同侧（=v2 的第一支，一字未改）
      2 **反号线**（新）：最小剂量格逐图决定性（合并 p<0.05）且与**阶梯顶**反号，同时阶梯顶有方向
        ⇒ 判"反号线"，读数是配对步长那条 CI；这一支排在"有阈值"之前，因为"之前测不出"这句话
        在"之前反向地显著"面前是错的。
      3 有阈值：小剂量端没方向，而闸格**或阶梯顶**有方向（v2 只看闸格）
      4 测不动：三格都没过两关，且连"逐图决定性反号"这条最弱的形状证据都凑不齐
    阶梯顶与闸格同格时（Fashion 薄账）第 2、3 支自动退回 v2 的行为——这不是巧合，是要求：
    **v3 不许在 v2 本来就读得清的地方改口**，它只买"判据看得见阶梯顶"这一件事。
  · 离线验红（零训练，八本已入库档案）：v1/v2 复算必须各自等于档案里那句判词（v2 那两本按 v2 记，
    v1-era 六本按 v1 记；已知例外只有 T=1 那一本，它已被 _v2 副本改判），v3 相对 v2 的**翻转集合
    必须恰好等于**低数据档那一本（测不动 → 反号线），其余七本判词逐字不变。全翻是破坏，不翻是装饰。

第八发（`SHAPE_RULE = "v4"`）还的是第七发自己点名的那笔**定义域债**，判据同样写在重跑之前。
「α=0.05 就已经买到锚点伤害的 x%」这句份额话在 v3 的一本文里出界了：中数据档补到 8 条流那本
（`compress_report_v4f_mid_n8_v3.json`）现读 share = **118%**——第一格 −45.6 张、阶梯顶 −38.5 张，
最小剂量的伤害**比顶还大**。份额是 abs/abs，出 100% 那句话就自指失效（"买到了 118% 的伤害"
不是任何可读的量），可判词照旧印成『污染型』、句子照旧报份额。上一轮的处置是"在判词外面补一句人写的
注解"——而第七发刚立的规矩是：**描述性读数不进判词，下一轮没人能复算它**。所以该改口的是那句话，
不是那个数；改的方式是给它一个正式分支。
  · v4 相对 v3 **只多一条分支**，其余四支的条件与顺序逐字不动（"有方向"仍＝逐图合并 p<0.05 且全流同号）：
      1b **倒挂**：v3 的第 1 支（污染型）成立，**且**阶梯顶在账里、与第一格**同号**、
        |第一格均值| > |阶梯顶均值|（即份额 > 100%）⇒ 判『污染型·倒挂（最小剂量比阶梯顶还伤）』。
        读数改说"最小剂量这一格比顶还伤 x 张"，并把两端钉在一起的**配对步长 CI** 端出来当承重：
        CI 不含 0 ⇒ 倒挂本身测得干净；CI 跨 0 ⇒ 判词只到"点估计倒挂、这一步还没测干净"，不许写成已定。
        反号（=v3 的第 2 支）与它互斥，所以这条只嵌在第 1 支内部，**不动分支顺序**。
  · 预登记的翻转集合（现读全部在册档案的 share 与两格均值算出来，此后不许改口）：
      翻：**只有** `compress_report_v4f_mid_n8_v3.json`（118%，唯一 >100 的一本）；
      不翻：`v4f`(79) / `v4f_mid`(97) / `v4f_n8_v3`(82) / `v4f_s64`(63) / `v4f_s64_n8_v3`(62) 五本
        污染型全在界内（97% 那本看着离界最近，见下面自纠 ①），另外八本（3 本有阈值 + 4 本测不动 +
        1 本反号线）根本进不到第 1 支。全翻是破坏，一本都不翻是装饰。
  · 验红落在守卫脚本 `_red_v3.py` 新增的**闸⑦**（v4 相对 v3 的翻转集合恰好等于预言那一本，
    其余各本 v4 判词＝v3 判词），并配两个拆红模式：`--v4blind`（v4 瞎掉倒挂腿 → ⑦ 红，证明这条腿承重）、
    `--v4loose`（忘掉同号与大小条件、只要份额>0 就判倒挂 → 预期 ④⑦ 连坐红）。
  · **跑完后的两条自纠**（预登记不许被静默改写，所以红真值记在这里）：
      ① 近邻对照挑错了本。`compress_report_v4f_mid.json` 份额 97% 看着离界最近，但它是 3 条流的账、
        **阶梯顶不在账里** ⇒ 倒挂腿对它恒等于 v3，它测不到这条腿。参评账（顶在账，14 本里 11 本）
        里在册『污染型』的只有 3 本：118%（翻）、82%、62%（后两本保持原判词）⇒ 真正的近邻对照是
        **82% 那本**（`compress_report_v4f_n8_v3`），离界 18 个点而不是 3 个点。
      ② `--v4loose` 实测**只红 ⑦，不红 ④**（预期落空）。不是 ④ 松：④ 的定义是"按 v2/v3 复算后
        在册污染型不许变"，它压根没读 v4 这条腿 ⇒ **v4 的连坐只有 ⑦ 一道防线**。这一条要记着：
        以后再加判据版本，旧的"不连坐"闸不会自动获得对新版本的判决权，必须显式接进去。
      ③ 落地实测（同流 45+3 次重跑，零 API 费）：判词 = 『污染型·倒挂』、14 个训练路径字段逐位不动，
        守卫 `--v4blind` / `--v4loose` 各红 1 条（都在 ⑦）。
  · 换流数＝换实验：v4 不回头改任何已入库判词，只对 118% 那一发做同流重跑本（`_v4` 后缀），
    训练路径 14 个字段必须逐位不变才算"翻案是判据换来的"。
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import torch

from step1_quantize import MODELS, build_cnn, predict
from step4_distill import DATA_TIERS, TEACHER_ARCHIVE_ACC, VAL_N, logits_of, sign_test_p, train_student
from step4e_marginal_recheck import OLD_SEEDS, summarize          # 复现单位是流：CI 用同一套系数表
from step4c_alpha_T_sweep import ALPHA_ANCHOR, ANCHOR_T, TIER, V4C_SUFFIX, key, load_anchor, out_path, v4b_path
from step6_fashion_replication import TEACHER_PT, load_fashion
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
LADDER = (0.0, 0.05, 0.1, 0.2, 0.3)               # 0 是恒等闸不是臂；0.3 是跨脚本闸
SEEDS = (42, 7, 2026)                              # 与第4课b/c 同一批随机流，才能与档案逐位对
SEEDS8 = SEEDS + (11, 202, 3003, 40004, 55555)     # 第4课e 那五条：老 3 条只对账，新 5 条才是复现
PROBE_ALPHA = 0.3                                   # 已在 v4c 档案里的那一格
P_SIG = 0.05
SHAPE_RULE = "v4"        # 形状判据版本：v1 只看逐图合并 p，v2 加流级同号腿，v3 让判据吃阶梯顶＋给反号一个正式分支，
                         # v4 在第 1 支内部给"份额>100%"那个定义域出界一个正式分支（倒挂）
V6 = HERE / "compress_report_v6_fashion.json"
RULES_WITH_TOP = ("v3", "v4")     # 把阶梯顶交给判据的版本：v2 及更早压根没看那一格

DATASETS = {
    "mnist": {"loader": load_split, "teacher_pt": MODELS / "mnist_cnn_fp32.pt"},
    "fashion": {"loader": load_fashion, "teacher_pt": TEACHER_PT},
}


def flow_sign_p(k: int, n: int) -> float:
    """n 条流里 k 条与本格均值同号 → 精确二项双侧 p（H0：各占一半）。

    这条曲线的形状值得记：3/3 只有 p=0.25，6/6 才到 0.031，8/8 是 0.0078 而 7/8 已经回到 0.0703。
    也就是说『流级符号检验显著』这把尺子**在 n=3 的账上根本不存在**。所以 v2 的流级腿写成计数
    （k == n，不许任何一条流反向），p 只作为透明度报进档案——见文档串第四发那段验红。"""
    m = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n)


def flow_leg(s: dict, n: int) -> dict:
    """一格的『方向』由两条腿给出：逐图合并 p（同一批 10000 张图上数翻牌）+ 同号流数（复现单位）。"""
    k = max(sum(1 for g in s["gaps_imgs"] if g > 0), sum(1 for g in s["gaps_imgs"] if g < 0))
    return {"same_dir_flows": f"{k}/{len(s['gaps_imgs'])}", "p_pooled": s["p_pooled"],
            "p_flows": round(flow_sign_p(k, n), 6),
            "has_direction": bool(s["p_pooled"] < P_SIG and k == n)}


def shape_verdict(s_small: dict, s_probe: dict, n: int, dirn: float,
                  a_small: float, a_probe: float, anc_a: float,
                  s_top: dict | None = None, pair_step: dict | None = None, *,
                  inverted_leg: bool = False) -> tuple:
    """形状判据 v3/v4（同一台机器，v4 只多一条腿；两版都预登记于重跑之前）：一格**有方向** = 逐图合并 p < 0.05 **且** 全流同号。

    `s_top=None` 时逐字退化为 v2 的三分支——离线验红就靠这条把已入库档案的 v2 判词复算一遍。
    传进 `s_top`（本发的阶梯顶 α={anc_a}）就是 v3：多一支「反号线」，并让「有阈值」也能由阶梯顶撑起。
    分支顺序写死：① 污染型（条件与 v2 一字未改）② 反号线 ③ 有阈值 ④ 测不动。
    阶梯顶与闸格同格时（Fashion 薄账）②③ 自动退回 v2 的行为——**v3 不许在 v2 本来就读得清的地方改口**。

    `inverted_leg=True` 才是 v4：只在第 1 支内部再问一句"第一格与阶梯顶同号、而第一格更伤吗"，
    是则改判『污染型·倒挂』。**默认关**——离线验红要用它把 v3 的复算保持逐字不变（否则闸① 会把
    "喂同一批数字"这件事复算成另一句话）。为什么只嵌在第 1 支里：反号（第 2 支）与"同号且更大"
    互斥，两支不可能同时亮，所以顺序不动、其余三支的条件也不动。

    为什么「反号线」要排在「有阈值」之前、又要两条件同时才判：v2 的三分支里没有"反向"的位置，
    于是低数据档那一发只能印『测不动』再补一段人写的描述性读数——描述性读数不进判词，下一轮没人能复算它。
    而"有阈值"说的是"起点之前测不出"，读数说的是"起点之前在**另一侧**"，那是两回事。
    门槛刻意收紧（小剂量格逐图决定性 **且** 阶梯顶自己有方向），差一条就回到『测不动』，不硬给形状。

    为什么 v1 那一关不够：合并 p 是在同一批 10000 张测试图上数翻牌，n 条流共用的就是那一批图——
    它再小也只证明『这条流内部稳』，不证明『换个随机源还稳』。复现单位是流，所以**方向这一票由流派**。
    代价当场可查：v1 会说『有阈值』的读数里，凡是靠少数几条流撑起来的，v2 一律降级为『测不动』。

    为什么流级这条腿是**计数**（k == n）而不是『流级 p < 0.05』：验红第一版按后者实现，六本档案里
    翻掉了五本——因为 n=3 时全同号的 p 只有 0.25，那把尺子在 3 条流的账上压根不存在，固定 p 阈值
    等于没收板块三条『污染型』判词的判决权（那是把债转成损失，不是还债）。改成计数之后两版在
    n=8 上等价（8/8 才过），在 n=3 上退化成 v1 原本就在用的『没有反例』——判据的松紧跟着**流数**变，
    但不再跟着一个对 n=3 无定义的阈值变。p 仍然算、仍然进档案，就是为了让人看见这条腿有多松。

    返回 (shape, reading, legs)，legs 里是两格各自的两个 p 与同号流数，直接进档案。"""
    lo, hi = flow_leg(s_small, n), flow_leg(s_probe, n)
    harm_small = lo["has_direction"] and dirn * s_small["mean_gap_imgs"] > 0
    lo["alpha"], hi["alpha"] = a_small, a_probe
    top = hi if s_top is None else dict(flow_leg(s_top, n), alpha=anc_a)
    reverse = bool(s_top is not None) and top["has_direction"] and lo["p_pooled"] < P_SIG \
        and s_small["mean_gap_imgs"] * s_top["mean_gap_imgs"] < 0
    inverted = bool(inverted_leg and s_top is not None and harm_small) and \
        s_small["mean_gap_imgs"] * s_top["mean_gap_imgs"] > 0 and \
        abs(s_small["mean_gap_imgs"]) > abs(s_top["mean_gap_imgs"])
    if hi["p_pooled"] < P_SIG:
        v1_note = (f"特别注意 α={a_probe:g} 那一格：v1 只看逐图合并 p 会把它读成『有方向』"
                   f"（{hi['p_pooled']:.3g} < 0.05），但它只有 {hi['same_dir_flows']} 条流同号"
                   f" ⇒ 那一票是图给的，不是流给的")
    else:
        v1_note = (f"两格的逐图合并 p 本来就没有全过（{lo['p_pooled']:.3g} / {hi['p_pooled']:.3g}）："
                   f"这一发 v1 与 {SHAPE_RULE} 会同判『测不动』，判据版本不是这里的不确定来源，"
                   f"不确定来自**流数与效应大小**")
    if a_probe != anc_a and s_top is None:
        v1_note += (f"。另记一笔债：判据吃的是 α={a_small:g} 与 α={a_probe:g} 这两格，"
                    f"而这一发的阶梯顶是 α={anc_a:g}——它有没有方向、跟这两格同不同向，判据一概没看")
    if harm_small and inverted:
        gap = abs(s_small["mean_gap_imgs"] - s_top["mean_gap_imgs"])
        ps = ("" if pair_step is None else
              f"。把两端钉在一起的**配对步长** α={a_small:g}→{anc_a:g} 是 {pair_step['mean_gap_imgs']:+.1f} 张"
              f"（CI [{pair_step['ci95_imgs'][0]:+.1f}, {pair_step['ci95_imgs'][1]:+.1f}]、"
              f"{pair_step['same_sign_seeds']} 条流同号）⇒ "
              + ("CI 不含 0：倒挂这一步测得干净"
                 if pair_step["ci95_imgs"][0] * pair_step["ci95_imgs"][1] > 0 else
                 "CI 跨 0：**点估计倒挂、这一步本身还没测干净**，别写成已定"))
        shape = "污染型·倒挂（最小剂量比阶梯顶还伤）"
        reading = (f"α={a_small:g} 已经把方向摆出来（{lo['same_dir_flows']} 条流同号、流级 p={lo['p_flows']:.4g}、"
                   f"逐图合并 p={lo['p_pooled']:.4g}、均值 {s_small['mean_gap_imgs']:+.1f} 张）——到这里还是 v3 那句污染型。"
                   f"但**份额这句话出界了**：它是锚点剂量的 {a_small / anc_a:.0%}，却比阶梯顶 α={anc_a:g}"
                   f"（{s_top['mean_gap_imgs']:+.1f} 张）还多伤 {gap:.1f} 张，也就是『买到锚点伤害的 "
                   f"{round(abs(s_small['mean_gap_imgs']) / abs(s_top['mean_gap_imgs']) * 100)}%』——"
                   f"一个 >100% 的份额不是任何可读的量。改口不改数：这一发读的是**倒挂**，"
                   f"剂量再加回去反而没那么伤（H1 过原点直线预言『按剂量比例』在这里直接反了，"
                   f"污染说预言『两格同量级』至少量级还对）{ps}")
    elif harm_small:
        shape = "污染型（有伤害不必有剂量）"
        reading = (f"α={a_small:g} 已经把方向摆出来（{lo['same_dir_flows']} 条流同号、流级 p={lo['p_flows']:.4g}、"
                   f"逐图合并 p={lo['p_pooled']:.4g}、均值 {s_small['mean_gap_imgs']:+.1f} 张），"
                   f"而它是锚点剂量的 {a_small / anc_a:.0%}——H1 的『按份额掺毒』在最后一块地上也没了："
                   f"软标签一旦进目标函数，坏的是它**在不在**，不是它占多少")
    elif reverse:
        shape = "反号线（小剂量端与阶梯顶反向：两说都不合）"
        ps = ("" if pair_step is None else
              f"；把两端钉在一起的**配对步长** α={a_small:g}→{anc_a:g} 是 {pair_step['mean_gap_imgs']:+.1f} 张"
              f"（CI [{pair_step['ci95_imgs'][0]:+.1f}, {pair_step['ci95_imgs'][1]:+.1f}]、"
              f"{pair_step['same_sign_seeds']} 条流同号、二项 p={pair_step['p_seeds_binomial']}）")
        mid = ("" if a_probe == anc_a else
               f"（闸格 α={a_probe:g} 均值 {s_probe['mean_gap_imgs']:+.1f} 张夹在中间、流级 {hi['same_dir_flows']}"
               f" 没方向，所以这一发读的不是『小剂量有害、大剂量有益』，而是整条线在往上走且截距为负）")
        reading = (f"最小剂量端自己就是**反向**的：α={a_small:g} 均值 {s_small['mean_gap_imgs']:+.1f} 张"
                   f"（逐图合并 p={lo['p_pooled']:.4g} 决定性、流级 {lo['same_dir_flows']} p={lo['p_flows']:.3g}），"
                   f"而阶梯顶 α={anc_a:g} 两关同过：{s_top['mean_gap_imgs']:+.1f} 张"
                   f"（合并 p={top['p_pooled']:.4g}、流级 {top['same_dir_flows']} p={top['p_flows']:.3g}）"
                   f"{mid}{ps}。⇒ 污染说预言两格同量级（不合：两端差 "
                   f"{abs(s_top['mean_gap_imgs'] - s_small['mean_gap_imgs']):.1f} 张且反号）、"
                   f"H1 过原点直线预言两格同号且按剂量比例（也不合：符号相反）——数据支持的是一条"
                   f"**带负截距的上升线**：剂量小时 KD 比 CE 更差，剂量大了才翻正并继续涨。"
                   f"这一支不判谁赢，只把『份额线』和『饱和线』一起否掉。")
    elif hi["has_direction"] or top["has_direction"]:
        carrier, which = (hi, "闸格") if hi["has_direction"] else (top, "阶梯顶")
        shape = "有阈值（H1 半活）"
        reading = (f"小剂量端（α≤{a_small:g}）两关都出不了方向（合并 p={lo['p_pooled']:.3g}、"
                   f"流级 {lo['same_dir_flows']} p={lo['p_flows']:.3g}），到 α={carrier['alpha']:g}"
                   f"（{which}）两关同过"
                   f"（合并 p={carrier['p_pooled']:.3g}、流级 {carrier['same_dir_flows']} p={carrier['p_flows']:.3g}）"
                   f"→ 存在一个 0~{carrier['alpha']:g} 之间的起点，之前测不出、之后基本饱和；"
                   f"说『伤害随 α 线性增长』仍然没有证据")
    else:
        shape = "测不动（这一发没判决）"
        reading = (f"两端都过不了两关（α={a_small:g}：合并 p={lo['p_pooled']:.3g} / 流级 "
                   f"{lo['same_dir_flows']} p={lo['p_flows']:.3g}；α={a_probe:g}：合并 p={hi['p_pooled']:.3g} "
                   f"/ 流级 {hi['same_dir_flows']} p={hi['p_flows']:.3g}"
                   + ("" if s_top is None or a_probe == anc_a else
                      f"；阶梯顶 α={anc_a:g}：合并 p={top['p_pooled']:.3g} / 流级 {top['same_dir_flows']} "
                      f"p={top['p_flows']:.3g}")
                   + "）→ 这一列的剂量响应判不了，别硬解。" + v1_note)
    legs = {"small": lo, "anchor": hi} if s_top is None else {"small": lo, "anchor": hi, "top": top}
    return shape, reading, legs


def shape_verdict_v1(s_small: dict, s_probe: dict, dirn: float) -> str:
    """v1 判据（只吃逐图合并 p，锚点那一腿不看流级）——留着是为了能把两版判词对着**同一批数字**并排算。

    离线验红靠它：新判据若谁都翻不动就是摆设，若连坐翻掉板块最硬的几条就是把债转成损失。"""
    n, k = (int(x) for x in s_small["same_dir_seeds"].split("/"))
    if s_small["p_pooled"] < P_SIG and k == n and dirn * s_small["mean_gap_imgs"] > 0:
        return "污染型"
    if s_small["p_pooled"] >= P_SIG and s_probe["p_pooled"] < P_SIG:
        return "有阈值"
    return "测不动"


def v4c_name(width: int, tier: str) -> Path:
    return out_path(width, tier)


def name_suffix(tier: str, width: int) -> str:
    """产物名跟着每一个判别变量走；后缀规则只有一个来源（step4c 的 V4C_SUFFIX）。"""
    if tier != TIER and tier not in V4C_SUFFIX:
        raise SystemExit(f"档 {tier} 没登记档案名后缀（V4C_SUFFIX 只有 {list(V4C_SUFFIX)}）："
                         f"回落到默认名会和 {TIER} 那本撞名，无声盖掉它的判决")
    return V4C_SUFFIX.get(tier, "") + ("" if width == 128 else f"_s{width}")


def check_t(t: float, grid: dict, where: str) -> float:
    """阶梯只能铺在这一本账**已有**的温度列上：没有那一列就没有锚点格，
    也就没有尺子更没有闸——换列等于换一本账。"""
    cols = sorted({float(k.split("_T")[1]) for k in grid})
    if t not in cols:
        raise SystemExit(f"{where} 这本账只有温度列 {cols}，没有 T={t:g}："
                         f"先把那一列跑进账里再来铺阶梯")
    return t


def setup(dataset: str, width: int, tier: str, extra: str = "",
          t_col: float | None = None) -> dict:
    """六样读数（温度列 / 锚点剂量 / 闸格 / 该档的底 / 锚点伤害 / 产物名）全从**那一本账**现读。

    为什么必须现读而不是各写一套：Fashion 那本账没有 T=3、也没有 α=0.3——把 MNIST 的两个常量
    抄过来，会得到一个不存在的锚点格和一格对不上的闸（而"对不上的闸"是最坏的一种绿）。
    账本字段布局的差异只允许出现在这一个函数里。"""
    if dataset == "mnist":
        v4b, v4c = v4b_path(width, tier), v4c_name(width, tier)
        missing = [p.name for p in (v4b, v4c) if not p.exists()]
        if missing:
            raise SystemExit(f"缺 {'、'.join(missing)}——底和 α={PROBE_ALPHA} 基准全从这两本读，"
                             f"先跑 step4b_seed_variance.py 与 step4c_alpha_T_sweep.py（带上同一组 --tier）")
        arch = json.loads(v4c.read_text(encoding="utf-8"))
        if int(arch["width"]) != width or arch["tier"] != tier:
            raise SystemExit(f"{v4c.name} 是 {arch['tier']}/S{arch['width']} 的表，不是 {tier}/S{width}")
        anc_a, gate_a = ALPHA_ANCHOR, PROBE_ALPHA
        # 阶梯必须**真的训到锚点剂量**：v3 的判据能吃阶梯顶，而上一轮刚把"份额/反号"那两句话
        # 改成读本格现读的阶梯顶（不借 3 条流的档案读数）。少这一格，那两句话与 v3 的第三格都没有原料。
        lad = LADDER + (anc_a,)
        t = ANCHOR_T if t_col is None else t_col
        floor, floor_kind, floor_from = load_anchor(v4b, width, tier)["floor"], "v4b 的 KD 臂跨种子极差", v4b.name
        grid, anchor_from = arch["grid"], v4c.name
        t = check_t(t, grid, v4c.name)
        out = HERE / v4c.name.replace("v4c", "v4f").replace(".json", f"{extra}.json")
        rows_of = lambda k: grid[k]["per_seed"]                        # noqa: E731
        need = [key(anc_a, t), key(gate_a, t)]
    else:
        if not V6.exists():
            raise SystemExit(f"缺 {V6.name}——换任务的阶梯只能对着那一本任务自己的账铺，先跑 step6")
        d = json.loads(V6.read_text(encoding="utf-8"))
        if d.get("dataset") != "FashionMNIST" or int(d["width"]) != width or tier not in d["per_tier"]:
            raise SystemExit(f"{V6.name} 是 {d.get('dataset')}/S{d.get('width')}"
                             f"（档位 {list(d.get('per_tier', {}))}），装不下 {dataset}/S{width}/{tier}")
        pt, lad_t = d["per_tier"][tier], [float(x) for x in d["temps"]]
        t = max(lad_t) if t_col is None else t_col   # 默认铺在伤害最大那一端；--t 换列
        anc_a = min(float(x) for x in d["alphas"])  # 那本账里最小的 α 就是锚点剂量（0.7）
        gate_a, lad = anc_a, LADDER + (anc_a,)      # 没有 α=0.3 可对账 ⇒ 闸落在锚点格上
        floor, floor_kind, floor_from = int(pt["own_floor_imgs"]), "该档自己的底（第4课g 规矩）", V6.name
        grid, anchor_from = pt["grid"], f"{V6.name} 的 {tier}"
        t = check_t(t, grid, V6.name)
        rows_of = lambda k: pt["raw_rows"][k]                          # noqa: E731
        out = HERE / f"compress_report_v4f_{dataset}{name_suffix(tier, width)}{extra}.json"
        need = [key(anc_a, t), key(gate_a, t)]
    miss = [k for k in dict.fromkeys(need) if k not in grid]
    if miss:
        raise SystemExit(f"{anchor_from} 里没有 {miss} 这几格：这一列温度/这个剂量那一本账没跑过，"
                         f"阶梯的尺子和闸就没有基准（有 {sorted(grid)}）")
    peers = {tt: grid[key(anc_a, tt)]["mean_gap_imgs"] for tt in
             sorted({float(k.split("_T")[1]) for k in grid} - {t})
             if key(anc_a, tt) in grid}
    gate = rows_of(key(gate_a, t))
    return {"t": t, "anchor_alpha": anc_a, "gate_alpha": gate_a, "ladder": lad,
            "teacher_acc": (float(d["teacher"]["test_acc"]) if dataset == "fashion" else TEACHER_ARCHIVE_ACC),
            "anchor_mean": float(grid[key(anc_a, t)]["mean_gap_imgs"]),
            "gate_rows": gate, "anchor_rows": rows_of(key(anc_a, t)), "gate_mean": round(sum(r["gap_imgs"] for r in gate) / len(gate), 1),
            "floor": floor, "floor_kind": floor_kind, "floor_from": floor_from, "anchor_from": anchor_from,
            "peers": peers, "out": out,
            "gate_is_anchor": bool(gate_a == anc_a), "grid_keys": sorted(grid)}


def main(dataset: str = "mnist", width: int = 128, tier: str = TIER,
         nseeds: int = 3, thin: bool = False, t_col: float | None = None) -> None:
    ds = DATASETS[dataset]
    if nseeds not in (3, 8):
        raise SystemExit(f"复现流只有 3（对账那批）与 8（加第4课e 那五条）两档")
    seeds = SEEDS if nseeds == 3 else SEEDS8
    extra = ("" if nseeds == 3 else "_n8") + ("_thin" if thin else "") + \
            ("" if t_col is None else f"_t{t_col:g}") + f"_{SHAPE_RULE}"
    if not ds["teacher_pt"].exists():
        raise SystemExit(f"缺 {ds['teacher_pt'].name}——这一课的老师权重只能复用，别现造")
    L = setup(dataset, width, tier, extra, t_col)
    T, LAD, PROBE, ANC = L["t"], L["ladder"], L["gate_alpha"], L["anchor_alpha"]
    if thin:
        # 薄阶梯：只留判决真正吃的那几格（最小剂量 + 锚点，外加**跨脚本闸格**），恒等闸只在老 3 条流上跑。
        # 为什么可以砍：这一发要判的是"从小剂量到锚点这一步陡不陡"，中间格不参与那个配对差。
        # 为什么闸格不在可砍之列：闸格是"这一发与立锚点那课是同一台训练器"的唯一独立证明，
        #   砍了它，MNIST 那几档（闸格 α=0.3 ≠ 锚点 α=0.7）就变成一格对账都没有的实验。
        #   Fashion 那发闸格恰好就是锚点格，去重后元组不变 ⇒ 已入库的两本 Fashion 薄账仍可逐位复跑。
        LAD = tuple(dict.fromkeys((0.0, LAD[1], PROBE, ANC)))
        print(f"[薄阶梯] 只跑 {LAD}：两格剂量差 = 污染说与 H1 的全部分歧所在，"
              f"闸格 α={PROBE:g} 是独立于锚点的对账线所以保留")
    floor, anchor_mean = L["floor"], L["anchor_mean"]
    probe = {"mean_gap_imgs": L["gate_mean"], "per_seed": L["gate_rows"]}
    # 跨脚本对账的格子跟着**判据吃哪几格**走：v3 能让阶梯顶承载判决，那一格就必须有自己的独立对账线
    # （否则"最硬的那一格"恰恰是唯一没跟上游对过的）。闸格与顶同格时（Fashion 薄账）去重成一条。
    gate_refs = {a: rr for a, rr in ((PROBE, L["gate_rows"]), (ANC, L["anchor_rows"]))}
    out = L["out"]
    if out.exists():
        raise SystemExit(f"{out.name} 已入库——重跑会盖掉那本账。要重跑先删或改名")

    xt_all, yt_all, xte, yte = ds["loader"]()
    cut = len(xt_all) - VAL_N
    xv, yv = xt_all[cut:], torch.tensor(yt_all[cut:])
    teacher = build_cnn()
    teacher.load_state_dict(torch.load(ds["teacher_pt"], weights_only=True))
    teacher.eval()
    xd, yd = xt_all[:DATA_TIERS[tier]], torch.tensor(yt_all[:DATA_TIERS[tier]])
    soft = logits_of(teacher, xd)

    h1_line = {a: round(a / ANC * anchor_mean) for a in LAD if a > 0}
    print(f"[{dataset} · {tier} · S{width}] 训练 {DATA_TIERS[tier]} 张 | 测试 {len(yte)} | "
          f"α 阶梯 {LAD} @ T={T:g} | {len(seeds)} 条流")
    print(f"[仪器] 底（{L['floor_from']} 的{L['floor_kind']}）= {floor} 张；"
          f"锚点格 α={ANC:g}/T={T:g} 三种子均值 {anchor_mean:+.1f} 张（{L['anchor_from']}）")
    print(f"[仪器] α={PROBE:g} 档案读数（跨脚本对账基准）：均值 {probe['mean_gap_imgs']:+.1f} 张，"
          f"逐种子 {[r['gap_imgs'] for r in probe['per_seed']]}")
    if L["gate_is_anchor"]:
        print(f"[仪器] **闸格就是锚点格**（α={PROBE:g}=锚点剂量）：这一发只剩恒等闸是独立于锚点的对账线")
    print("\n[先立预言（此后不许改口）]")
    print(f"  H1 残城（伤害 ∝ α，直线过原点）→ " +
          "、".join(f"α={a:g} {h1_line[a]:+.1f} 张" for a in LAD if a > 0))
    print(f"  污染说（份额只决定多快生效）→ α={LAD[1]:g} 就有 {anchor_mean:+.1f} 张量级的效应（方向随锚点符号），"
          f"且阶梯小剂量各格彼此差不超过一个底（{floor} 张）")
    print(f"  判别线：α={LAD[1]:g} 实测距 0 更近（|gap| < {abs(h1_line[LAD[1]]) * 2:.0f} 张）→ H1 活；"
          f"距 {anchor_mean:+.1f} 更近（|gap| > {floor}）→ H1 死在最后一块地上")

    ce_pred, rows = {}, {a: [] for a in LAD if a > 0}
    id_gate, probe_gate = [], []
    print("\n[开跑] 每个种子一副 CE + 阶梯各一副（同起点同批次流）：")
    for seed in seeds:
        mce, _ = train_student(width, xd, yd, soft, "CE", xv, yv, seed=seed)
        pe = predict(mce, xte, False)
        ce_pred[seed] = pe
        ce_wrong = int((pe != yte).sum())
        line = [f"  种子 {seed:<5} CE 错 {ce_wrong} 张"]
        for a in LAD:
            if a == 0.0 and thin and seed not in SEEDS:
                continue                     # 薄阶梯：接线证明在老 3 条上跑过就够（它不随种子变）
            if a == 0.0:                       # 恒等闸：这一格不许当臂用，它测的是接线
                m0, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=0.0, t=T)
                p0 = predict(m0, xte, False)
                n_diff = int((p0 != pe).sum())
                id_gate.append({"seed": seed, "wrong": int((p0 != yte).sum()),
                                "ce_wrong": ce_wrong, "diff_vs_ce_imgs": n_diff,
                                "bit_identical": bool(n_diff == 0 and int((p0 != yte).sum()) == ce_wrong)})
                line.append(f"a0(恒等):差{n_diff}")
                continue
            m, _ = train_student(width, xd, yd, soft, "KD", xv, yv, seed=seed, alpha=a, t=T)
            p = predict(m, xte, False)
            wrong = int((p != yte).sum())
            broken = int(((pe == yte) & (p != yte)).sum())
            fixed = int(((pe != yte) & (p == yte)).sum())
            rows[a].append({"seed": seed, "alpha": a, "wrong_kd": wrong, "wrong_ce": ce_wrong,
                            "gap_imgs": ce_wrong - wrong, "broken": broken, "fixed": fixed,
                            "diff_vs_ce_imgs": int((p != pe).sum()),
                            "p_sign": round(sign_test_p(broken, fixed), 6)})
            line.append(f"a{a:g}:{ce_wrong - wrong:+d}(差{int((p != pe).sum())})")
            for ga, gref in gate_refs.items():
                if a != ga or seed not in SEEDS:
                    continue                   # 只有老 3 条有跨脚本账可核
                r = rows[a][-1]
                ref = gref[SEEDS.index(seed)]
                probe_gate.append({"alpha": a, "seed": seed,
                                   "match": bool(r["wrong_kd"] == ref["wrong_kd"] and r["broken"] == ref["broken"]
                                                 and r["fixed"] == ref["fixed"]),
                                   "this_run": {k: r[k] for k in ("wrong_kd", "broken", "fixed")},
                                   "archive": {k: ref[k] for k in ("wrong_kd", "broken", "fixed")}})
        print(" ".join(line))

    bad_id = [g for g in id_gate if not g["bit_identical"]]
    src = "v4c" if dataset == "mnist" else "v6"
    print(f"\n[恒等闸 α=0 必须逐位等于 CE] {f'{len(id_gate)}/{len(id_gate)} 通过：α 这根线接对了，阶梯上的数才是剂量的数' if not bad_id else 'FAIL：' + str(bad_id)}")
    if bad_id:
        print("  α=0 与 CE 不同 ⇒ loss 里 α 的位置不是教科书那个位置（比如 (1−α) 也乘在别处、"
              "或 KL 项被无条件加了进来）。**下面所有阶梯读数作废**，先修接线再谈剂量。")
        raise SystemExit(1)
    bad_probe = [g for g in probe_gate if not g["match"]]
    if not probe_gate:
        raise SystemExit(f"跨脚本闸一条记录都没有（阶梯 {list(rows)} 里没有对账格 "
                         f"{sorted(gate_refs)}）：空表『全过』不是通过，是没测到")
    for ga in sorted(gate_refs):
        sub = [g for g in probe_gate if g["alpha"] == ga]
        role = ("闸格＝阶梯顶" if ga == PROBE == ANC else
                ("阶梯顶" if ga == ANC else "") + ("闸格" if ga == PROBE else ""))
        print(f"[跨脚本闸 α={ga:g}（{role}）] "
              + (f"{len(sub)}/{len(sub)} 与 {src} 档案逐位一致" if sub else "这一格不在本发阶梯里（没测到）"))
    print(f"  → {'本课与那一课是同一台训练器，判据吃的每一格都有独立对账线' if not bad_probe else 'FAIL：' + str(bad_probe)}")
    if bad_probe:
        print(f"  对不上 ⇒ 本课与 {src} 不是同一台仪器，锚点伤害就没有基准，阶梯读数只能重跑")
        raise SystemExit(1)

    dirn = -1.0 if anchor_mean < 0 else 1.0      # 伤害方向跟着锚点符号：T=1 那列锚点可能是正号
    print(f"\n[阶梯] 每格 {len(seeds)} 条随机流（10000 张全量测试；差=与同种子 CE 预测不同的图数，是**分辨率**不是疗效）：")
    summary = {}
    for a, rr in rows.items():
        gaps = [r["gap_imgs"] for r in rr]
        tb, tf = sum(r["broken"] for r in rr), sum(r["fixed"] for r in rr)
        pooled = round(sign_test_p(tb, tf), 6)
        n_pos = sum(1 for g in gaps if g > 0)
        n_neg = sum(1 for g in gaps if g < 0)
        k_same = max(n_pos, n_neg)
        summary[a] = {"gaps_imgs": gaps, "mean_gap_imgs": round(sum(gaps) / len(gaps), 1),
                      "vs_floor": round(abs(sum(gaps) / len(gaps)) / floor, 2),
                      "mean_diff_vs_ce_imgs": round(sum(r["diff_vs_ce_imgs"] for r in rr) / len(rr), 1),
                      "pooled_flips": {"broken": tb, "fixed": tf}, "p_pooled": pooled,
                      "same_dir_seeds": f"{k_same}/{len(gaps)}",
                      "p_flows": round(flow_sign_p(k_same, len(gaps)), 6),
                      "h1_pred_imgs": h1_line[a],
                      "beats_h1_line": bool(dirn * (sum(gaps) / len(gaps)) > 2 * abs(h1_line[a]))}
        s = summary[a]
        print(f"  α={a:<5} gap {gaps} 张 均值 {s['mean_gap_imgs']:+6.1f}（底 {floor} 的 {s['vs_floor']}x）| "
              f"合并翻坏 {tb}/翻正 {tf} p={pooled:.4g} | 流同号 {s['same_dir_seeds']} p={s['p_flows']:.3g} | "
              f"与 CE 预测不同 {s['mean_diff_vs_ce_imgs']:.0f} 张 | "
              f"H1 该格预测 {h1_line[a]:+.1f} → {'**比直线坏得多：污染型**' if s['beats_h1_line'] else '与直线同量级：H1 撑住'}")

    # ---------- 配对步长（第4课e 那把尺）：同一条流里 锚点格 − 最小剂量格 ----------
    # 污染说预测这一步 ≈ 0（两格都早饱和），H1 直线预测它 ≈ 锚点伤害的绝大部分。
    # 老 3 条流那发只有逐图配对（合并 p）：同一批 10000 张图上重复数牌，复现单位是图不是流。
    small = min(a for a in rows)
    by_seed = {a: {r["seed"]: r["gap_imgs"] for r in rows[a]} for a in (small, ANC)}
    step = [by_seed[ANC][sd] - by_seed[small][sd] for sd in seeds]
    pair_step = summarize(step, f"α={small:g}→{ANC:g} 配对步长")
    crosses = bool(pair_step["ci95_imgs"][0] < 0 < pair_step["ci95_imgs"][1])
    print(f"\n[配对步长·以流为单位] α={small:g}→{ANC:g}（剂量 ×{ANC / small:.0f}）"
          f"每条形如（锚点格 − 小剂量格）：{step}")
    print(f"  均值 {pair_step['mean_gap_imgs']:+.1f} 张，95% CI [{pair_step['ci95_imgs'][0]:+.1f}, "
          f"{pair_step['ci95_imgs'][1]:+.1f}]，σ {pair_step['sd_imgs']} 张，同号 {pair_step['same_sign_seeds']} 条"
          f"（二项 p={pair_step['p_seeds_binomial']}）")
    print(f"  污染说预测这一步 ≈ 0，H1 直线预测 {h1_line[ANC] - h1_line[small]:+.1f} 张 → "
          + ("CI 跨 0：这一段掉多少说不清，两说都没判决" if crosses else
             ("CI 不含 0 且为负：这一段真的在掉 ⇒ 阶梯不是平到锚点的，污染说在这一列站不住"
              if pair_step["mean_gap_imgs"] < 0 else
              "CI 不含 0 且为正：小剂量格比锚点格更差，先查仪器再谈形状")))

    # ---------- 判决：形状由数字挑，不由人挑（判据 = SHAPE_RULE，见文档串第四发与第七发） ----------
    s_small, s_probe = summary[small], summary[PROBE]
    s_top = summary[ANC] if SHAPE_RULE in RULES_WITH_TOP else None   # v3 起把阶梯顶交给判据
    shape, reading, legs = shape_verdict(s_small, s_probe, len(seeds), dirn, small, PROBE, ANC,
                                         s_top, pair_step, inverted_leg=(SHAPE_RULE == "v4"))
    v1_shape = shape_verdict_v1(s_small, s_probe, dirn)
    v2_shape = shape_verdict(s_small, s_probe, len(seeds), dirn, small, PROBE, ANC)[0]
    v3_shape = shape_verdict(s_small, s_probe, len(seeds), dirn, small, PROBE, ANC, s_top)[0]
    print(f"\n[两关明细] 复现单位是流：合并 p 数的是同一批 10000 张图上的翻牌，流级腿数的是 {len(seeds)} 条流")
    for nm, lg in (("小剂量", legs["small"]), ("闸格", legs["anchor"])):
        print(f"  {nm} α={lg['alpha']:g}  逐图合并 p={lg['p_pooled']:.4g} | 流同号 {lg['same_dir_flows']} "
              f"p={lg['p_flows']:.4g} → {'两关同过：这一格有方向' if lg['has_direction'] else '缺一条腿：这一格没方向'}")
    if PROBE != ANC:
        top = flow_leg(summary[ANC], len(seeds))
        top["alpha"] = ANC
        legs["top"] = top
        print(f"  阶梯顶 α={ANC:g}（判据 {SHAPE_RULE} {'**吃了**这一格' if s_top is not None else '没吃这一格'}）"
              f"逐图合并 p={top['p_pooled']:.4g} | "
              f"流同号 {top['same_dir_flows']} p={top['p_flows']:.4g} → "
              f"{'这一格其实有方向' if top['has_direction'] else '这一格也没方向'}"
              + ("" if s_top is not None else
                 ("  ⇒ 上面那句『测不动』是**判据看不见阶梯顶**造成的，不是这一列真的读不出来：这笔债记进 v3"
                  if top["has_direction"] else "")))
    SHORT = lambda s: s.split("（")[0]                                    # noqa: E731
    prev_nm = "v3" if SHAPE_RULE == "v4" else "v2"   # 换到东西要相对**前一版**算，不能相对最老那版领功
    prev = v3_shape if SHAPE_RULE == "v4" else v2_shape
    print(f"[判据版本] 同一批数字交给四版判词：v1（只看逐图合并 p）{SHORT(v1_shape)} ｜ "
          f"v2（加流级同号腿）{SHORT(v2_shape)} ｜ v3（吃阶梯顶＋反号线）{SHORT(v3_shape)} ｜ "
          f"**{SHAPE_RULE}（本发采用）**{SHORT(shape)} → "
          + ("四版同判，这一发判据版本不换东西（也无损失）"
             if len({SHORT(v1_shape), SHORT(v2_shape), SHORT(v3_shape), SHORT(shape)}) == 1 else
             (f"**{SHAPE_RULE} 相对 {prev_nm} 换到了东西**："
              + ("这一发的份额出了 100% 的定义域，改判倒挂——读数改说『比顶还伤几张』加配对步长 CI"
                 if SHAPE_RULE == "v4" else "判据这一发看得见阶梯顶（多出来的那格方向或反号进了判词）")
              if SHORT(prev) != SHORT(shape) else
              f"更早的版本换到了东西，而 {SHAPE_RULE} 与 {prev_nm} 同判（这一发新那条腿不改变结论）")))
    print(f"\n[判决] {shape}：{reading}")
    # 分段也用以流为单位的**配对**差（同一批种子上相减），于是两段之和恰好等于上面那个整段配对步长。
    # 早先一版拿两格各自四舍五入过的均值相减，同一页上同一量出现 18.3 / 18.4 两个数。
    by_seed = {a: {r["seed"]: r["gap_imgs"] for r in rows[a]} for a in dict.fromkeys((small, PROBE, ANC))}
    lo_step = round(sum(by_seed[PROBE][sd] - by_seed[small][sd] for sd in seeds) / len(seeds), 1)
    hi_step = None                                   # 闸格=锚点格时上面没有第二段，档案里就该是 null
    # 这句话比的是"第一格 vs 锚点剂量那一格"，不是"vs 闸格"——MNIST 的闸格是 α=0.3，拿它当锚点会印出"买到 197%"
    s_anc = summary[ANC]
    same_sign_cells = s_small["mean_gap_imgs"] * s_anc["mean_gap_imgs"] > 0
    share = round(abs(s_small["mean_gap_imgs"]) / abs(s_anc["mean_gap_imgs"]) * 100)
    ci_lo, ci_hi = pair_step["ci95_imgs"]
    out_of_domain = same_sign_cells and share > 100      # 份额是 abs/abs：>100% 时这句话不再量任何可读的东西
    print(f"[判决的分量] 阶梯第一格 α={small:g}（锚点剂量的 {small / ANC:.0%}）"
          + (f"就已经买到锚点伤害的 **{share}%**：{s_small['mean_gap_imgs']:+.1f} / {s_anc['mean_gap_imgs']:+.1f} 张"
             if same_sign_cells and not out_of_domain else
             f"比阶梯顶 α={ANC:g} **还伤**：{s_small['mean_gap_imgs']:+.1f} vs {s_anc['mean_gap_imgs']:+.1f} 张。"
             f"『买到锚点伤害的百分之几』这句话在这里出界（现算 {share}%>100%，份额是 abs 比 abs，"
             f"出界后它不量任何可读的东西）⇒ 改口不改数：这一格比顶多伤 "
             f"{abs(s_small['mean_gap_imgs'] - s_anc['mean_gap_imgs']):.1f} 张，"
             f"把两端钉在一起的配对步长 α={small:g}→{ANC:g} 走 {pair_step['mean_gap_imgs']:+.1f} 张"
             f"（CI [{ci_lo:+.1f}, {ci_hi:+.1f}]，{'不含 0：倒挂这一步测得干净' if ci_lo * ci_hi > 0 else '跨 0：点估计倒挂、这一步本身还没测干净'}）"
             if out_of_domain else
             f"与锚点格**反号**：{s_small['mean_gap_imgs']:+.1f} vs 锚点 {s_anc['mean_gap_imgs']:+.1f} 张——"
             f"『买到锚点伤害的百分之几』这句话在这里不适用（份额是 abs 比 abs，反号时它会把『方向相反』"
             f"读成『买到了 {share}%』）。改看配对步长：α={small:g}→{ANC:g} 走 {pair_step['mean_gap_imgs']:+.1f} 张"
             f"（CI [{ci_lo:+.1f}, {ci_hi:+.1f}]）"))
    if L["gate_is_anchor"]:
        print(f"[判决的边界] α={small:g}→{PROBE:g}（剂量 ×{PROBE / small:.0f}）再走 {lo_step:+.1f} 张；"
              f"往上没有第二格可看——闸格就是锚点格，所以这一发判的是『0~{PROBE:g} 这一段陡不陡』，"
              f"判不了『{PROBE:g} 以上还涨不涨』（那一问留在 MNIST 的 α={PROBE_ALPHA:g}→{ALPHA_ANCHOR:g} 上）")
    else:
        hi_step = round(sum(by_seed[ANC][sd] - by_seed[PROBE][sd] for sd in seeds) / len(seeds), 1)
        sat = bool(lo_step * hi_step > 0 and abs(hi_step) < abs(lo_step) / 2)
        print(f"[判决的边界] 剂量往上再加：α={small:g}→{PROBE:g}（剂量 ×{PROBE / small:.0f}）"
              f"走 {lo_step:+.1f} 张；α={PROBE:g}→{ANC:g}（剂量 ×{ANC / PROBE:.2f}）"
              f"走 {hi_step:+.1f} 张（阶梯顶本格现读的 {summary[ANC]['mean_gap_imgs']:+.1f} 张）"
              + (" → 后一段同号且不足前一段的一半：**曲线在第一个读数之后基本就平了**（饱和的形状）" if sat else
                 " → 两段没有『同号且后段不足前段一半』，所以这一列**读不出饱和**；"
                 + ("小剂量端还反着号，这条线压根不是份额曲线" if lo_step * hi_step <= 0 or
                    s_small["mean_gap_imgs"] * summary[ANC]["mean_gap_imgs"] < 0 else "它更像一条还在走的线")))

    under = [a for a, s in summary.items() if abs(s["mean_gap_imgs"]) <= floor]
    print(f"\n[自我审计① 判决单位是配对，不是底] "
          + (f"α∈{under} 各格的 |gap| 在底（{floor} 张）以下——这是**设计使然**："
             "H1 自己就预测它们小。所以本课的判决单位是逐图配对（同一副权重内比翻牌），"
             "不是跨重训的极差。反过来也成立：谁要是拿『不到底』来否掉这些格，"
             "也就是在否掉自己用来立锚点的那套配对检验。" if under else
             f"这一轮每一格都过了底（{floor} 张）——连 α={small:g} 那格都过了，"
             "所以本课不需要拿配对检验去救小读数；判决的力气全在第一格的**方向**上（同号 + 合并 p），"
             "不在它离底多远。"))
    print("[自我审计② 恒等闸只证明接线，不证明灵敏度] α=0 逐位等于 CE 说的是『α 确实乘在该乘的地方』；"
          "它不给仪器定分辨率。分辨率由每格与 CE 不同的预测图数给出（上面『差 xxx 张』那一列）——"
          "那一列如果本身就是几十张，任何『这一格与 CE 打平』的说法都只是没看见。")
    print(f"[自我审计③ 阶梯是 {len(seeds)} 条随机流 × 每流一次] 与第4课b/c 同病：每种子只训一次，"
          "跨种子的极差没有二次底。"
          + ("这一发已按第4课e 的办法扩到 8 条流：极差仍是粗估，但判决改吃配对步长的 CI。"
             if nseeds == 8 else "要把它当正式判决用，得先照第4课e 的办法扩到 8 条流。"))
    peers = {tt: m for tt, m in L["peers"].items() if tt != T}
    print(f"[自我审计④ 只在 T={T:g} 上扫 α] 第4课c 判过 T 才是主效应、α 是平的，"
          f"所以本阶梯是**在 T={T:g} 这一列上**看剂量。"
          + ("同一本档案里 " + "、".join(f"α={ANC:g}/T={tt:g} 那格 {m:+.1f} 张"
                                        for tt, m in peers.items())
             + " 本课没测——阶梯的形状结论不许往这些列搬。" if peers else
             f"这本账只有 T={T:g} 一列，无其它温度可对照。"))
    print(f"[自我审计⑤ 对账基准跟着档案走] 本课的 α={PROBE:g} 格与 {src} 同种子同数据同起点，"
          "所以能逐位对账；换了 --dataset / --tier / --width 之后这条闸连着换档案——"
          "拿 A 档的账对 B 档的数就是自欺。")
    if dataset != "mnist":
        print(f"[自我审计⑥ 换任务同时换了三样] 任务的难易、老师的干湿（Fashion 满档老师 "
              f"{L['teacher_acc']}，MNIST 那副 {TEACHER_ARCHIVE_ACC}）、以及那本账里可用的温度列。"
              "所以这一发判的是『污染型伤害在第二个任务上还在不在』，不是『污染型伤害与 MNIST 无关』。")

    report = {"dataset": dataset, "tier": tier, "width": width, "temps_fixed": T, "ladder": list(LAD),
              "seeds": list(seeds), "probe_alpha": PROBE, "anchor_alpha": ANC,
              "paired_step_small_to_anchor": pair_step,
              "t_column_override": t_col, "thin_ladder": bool(thin), "identity_gate_seeds": len(id_gate),
              "gate_is_anchor": L["gate_is_anchor"], "teacher_acc": L["teacher_acc"],
              "floor_from": L["floor_from"], "floor_kind": L["floor_kind"], "seed_floor_imgs": floor,
              "anchor_mean_from": L["anchor_from"], "anchor_mean_gap_imgs": anchor_mean,
              "predictions_ex_ante": {"H1_line_through_origin": h1_line,
                                      "contamination_pred": f"α={LAD[1]:g} 即接近 {anchor_mean:+.1f} 张"},
              "gate_alpha_zero_identity": id_gate, "gate_probe_vs_archive": probe_gate,
              "ladder_summary": {str(a): summary[a] for a in summary},
              "arms": {f"α{a:g} 种子{r['seed']}": r for a, rr in rows.items() for r in rr},
              "verdict": {"shape": shape, "reading": reading,
                          "shape_rule": SHAPE_RULE,
                          "verdict_under_v1_rule": v1_shape,
                          "verdict_under_v2_rule": v2_shape,
                          "verdict_under_v3_rule": v3_shape,
                          "criterion_cells": [small, PROBE] + ([] if L["gate_is_anchor"] else [ANC]),
                          "two_leg_detail": legs,
                          "first_cell_alpha": small,
                          "first_cell_same_sign_as_anchor": bool(same_sign_cells),
                          "first_cell_share_of_anchor_damage_pct": share,
                          "step_low_imgs": lo_step, "step_low_dose_x": round(PROBE / small, 2),
                          "step_high_imgs": hi_step,
                          "step_high_dose_x": (None if L["gate_is_anchor"] else round(ANC / PROBE, 2))},
              "limitations": [f"形状判据 {SHAPE_RULE} 的流级腿是计数（全部流同号），它的松紧跟着流数走："
                              f"n={len(seeds)} 时全同号对应 p={flow_sign_p(len(seeds), len(seeds)):.4g}"
                              f"（反向一条即 {flow_sign_p(len(seeds) - 1, len(seeds)):.4g}）——"
                              "3 条流的账上它只等于『没有反例』，并不构成 0.05 意义上的显著。"
                              "要收紧只能补流，不能再换判据",
                              f"判据 {SHAPE_RULE} 吃的是 {('最小剂量格 + 闸格 + 阶梯顶' if SHAPE_RULE in RULES_WITH_TOP and not L['gate_is_anchor'] else '最小剂量格 + 闸格')}"
                              "：『测不动』现在说的是这几格**都**没过两关，而『反号线』只报形状"
                              "（负截距 + 还在涨），不判污染说与 H1 谁赢；"
                              + ("同一支里新加的『倒挂』也只改口不改数——它把出界的份额话换成张数与配对步长 CI，"
                                 "至于『剂量加回去反而没那么伤』是什么机制，这一发没有读数能判；"
                                 if SHAPE_RULE == "v4" else "")
                              + "小剂量端那条流级腿仍是计数，松紧跟着流数走",
                              "每种子每格只训一次：跨种子极差没有二次底" if nseeds == 3 else
                              f"{len(seeds)} 条流各训一次：这是把分辨率做精，新增的不是独立性",
                              f"只扫 T={T:g} 一列，其余温度上的剂量响应未测",
                              "小 α 各格均值本就低于底（H1 的预测就是这个量级），"
                              "判决依赖逐图配对检验而非重训抖动",
                              f"只 {tier}+S{width}：跨档/跨架构要各自带 --tier/--width 重跑，"
                              "档案不同不能混读"]
              + ([] if dataset == "mnist" else
                 ["换任务同时换了任务难度/老师干湿/那本账里可用的温度列：判的是『第二个任务上还在』，"
                  "不是『与 MNIST 无关』",
                  f"闸格就是锚点格（{src} 账里没有 α=0.3）：只剩恒等闸一条独立于锚点的对账线",
                  f"底是 {src} 该档自己的 own floor（{floor} 张），比 MNIST 两本账借来的 41/37 张宽"])}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 {out.name}（{len(seeds) + len(seeds) * len([a for a in LAD if a > 0])}"
          f" + 恒等闸 {len(id_gate)} 次训练，10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="α→0 剂量曲线（默认 MNIST S128+满数据，T=3 一列）")
    ap.add_argument("--dataset", choices=tuple(DATASETS), default="mnist",
                    help="换任务：fashion 时温度列/锚点剂量/底全部从 compress_report_v6_fashion.json 现读")
    ap.add_argument("--width", type=int, choices=(128, 64), default=128)
    ap.add_argument("--nseeds", type=int, choices=(3, 8), default=3,
                    help="8 = 加第4课e 那五条随机流（老 3 条仍只当逐位对账的锚）；档案名加 _n8")
    ap.add_argument("--thin", action="store_true",
                    help="薄阶梯：只跑最小剂量格与锚点格两格（配对步长只吃这两格），"
                         "恒等闸只在老 3 条流上跑；档案名加 _thin")
    ap.add_argument("--t", type=float, choices=(1.0, 3.0, 6.0), default=None,
                    help="把阶梯铺在哪一列温度上；默认 MNIST=T=3、Fashion=该账最大温度列。"
                         "显式给值时档案名加 _t<值>（换列=换一本账，不能与默认那本撞名）")
    ap.add_argument("--tier", choices=tuple(DATA_TIERS), default=TIER,
                    help="数据档位；底与跨脚本闸基准都从那一档（那一本任务）自己的账里读，缺格就退出")
    args = ap.parse_args()
    main(args.dataset, args.width, args.tier, nseeds=args.nseeds, thin=args.thin, t_col=args.t)
