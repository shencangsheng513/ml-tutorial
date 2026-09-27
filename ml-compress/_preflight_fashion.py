# -*- coding: utf-8 -*-
"""第廿八发 D 的零训练预检（我的临时件，收工时按圈码逐项确认后再删）。

要证的六件事，一次训练都不烧：
  [1] 名字逐字相同：同一坐标（任务 × 族 × 档 × 宽 × 流数）在 step4b 写盘那处与 step4c 读数那处
      必须算出同一个名字（两处各拼一遍 = 第二把尺子，第廿三发在流数位上摔过一次）。
  [2] **不撞名回归**：盘上每一本 v4b/v4c 账，由它**自己记的内容**（任务/族/宽/档/条流名单长度）
      反算出来的名字，必须逐字命中盘上那个文件 ⇒ 加"任务位"这件事没有把任何一本历史账改名，
      而新任务的账也必须由同一条规则命名（分两桶计数：MNIST 侧那一桶才是"不改老账名"的凭据）。
      凭据是内容不是文件名：拿文件名反推文件名的检查是循环论证。
  [3] 反借账闸双向：load_anchor 读到任务位不符的档案必须拒（fashion 问 mnist 的账、mnist 问
      fashion 的账都要红），且**第三向**要成立（身份全对时必须放行）——否则那条闸是恒红摆设。
  [4] committed() 对 Fashion 必须返回 None：ARCHIVE 里那些数是 MNIST 的，"逐位复现"这条线
      在 Fashion 上**结构上不存在**，不许安静地拿 MNIST 的锚点冒充。
  [5] 任务登记处是唯一来源：teacher 权重名、档案精度（从 v6 现读）、装载器可取；
      未登记的任务名必须硬退出；且 **step4_distill / step4c 的顶层 import 不许把 step6 拉进来**
      （循环 import 的根因）。这一条**各起一个干净子进程问**：本进程先 import 过 step4b，而
      step4b→step4e→step6 是既有顶层链，拿本进程的 sys.modules 判归属会把那条链算到登记处头上
      （第一版就红在这里——红的是预检自己，不是被测对象）。
  [6] 活体闸 + **launch 前先判盘态**：`--dataset fashion --tier 满数据8500 --nseeds 8` 必须**不被
      MLP 的合并档案闸拦下**，而是走到某一道训练前的拒跑（缺那本 8 流基准底 / 目标本已入库）。
      两种合法退出都要点名，第三种盘态（3 流底在、8 流目标本不在）= step4b 会真去训练 16 次 ⇒
      **不 launch，记一条红说"这一腿此刻不再零训练"**。
      为什么挪到 launch 之前：第一版只断言"退 0 就是它在烧训练"，那条红是**跑完之后**才叫的——
      本回合底入库后复跑预检，真的烧掉 16 次并写出 `compress_report_v4b_fashion_n8.json`，
      而训练不能倒带。"两种盘态都成立"这句话我自己写在下面，摔的恰恰是没把第三种盘态算进去的这两条。

写这一份时按 `_preflight_d.py` 的教训办：断言一律写成"入库前 / 入库后两种盘态都成立"的形式，
钉的是不变量而不是"跑预检的那一刻"（那份第一版就在链的下一段变成两条假红）。
—— 实测补一句：**"两种"是乐观的**。[2] 与 [6] 在底入库之后各自坏了，而坏的方向相反：
[2] 把**名字本来正确**的新账判成 10 条假红，[6] 安静地**去训练了**。前者是断言写窄了，
后者是断言写宽了（该拦的没拦）。教训：盘态相关的腿要枚举"这个坐标的每一步产物在/不在"的**全四格**，
不是"入库前/入库后"两格。
"""
import json
import subprocess
import sys
from pathlib import Path

import step4b_seed_variance as sb
import step4c_alpha_T_sweep as sc
from step1_quantize import MODELS
from step4_distill import DATASET_SUFFIX, DATASETS, TEACHER_PT, teacher_archive_acc

HERE = Path(__file__).resolve().parent
FAKE = HERE / "_tmp_pf_fake_book.json"           # 名字不带 compress_report ⇒ 不进 ⑫ 的全册扫描面
FULL, MID, LOW = "满数据8500", "中数据4500", "低数据2000"
COORDS = [(fam, tier) for fam in ("mlp", "cnn") for tier in (FULL, MID, LOW)]
fails: list[str] = []


def _doc(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _ident(d: dict) -> tuple:
    """一本账**自己记的**身份：(任务, 族, 宽, 档名位, 流数)。缺字段的按老账口径补，并数出来。"""
    ds = d.get("dataset", "mnist")
    fam = d.get("student_arch", "mlp")
    # 档名位只有三种真情况：单档（记 tier 或 tier_arg）、第4课那两档共本（tier_arg 缺席或写着"默认"）、
    # 没有这一位（最老那本合并账连 tier_arg 都没记）。第三种按"合并"读，不靠字符串猜。
    tier = d.get("tier") or d.get("tier_arg") or ""
    if str(tier).startswith("默认"):
        tier = ""
    return ds, fam, int(d["width"]), tier, len(d["seeds"])


# ---------- [1] 同一坐标两处同名 ----------
print("[1] Fashion 六本（2 族 × 3 档）：step4b 要写的名字 vs step4c 要读的名字")
for fam, tier in COORDS:
    sb.DATASET, sb.ARCH, sb.TIER, sb.WIDTH, sb.NSEEDS = "fashion", fam, tier, 128, 3
    sc.DATASET, sc.ARCH, sc.NSEEDS = "fashion", fam, 3
    written = sb.book_name(3).name
    try:
        read = sc.v4b_path(128, tier, 3).name
    except SystemExit as e:
        read = f"（拒跑：{str(e).splitlines()[0][:40]}…）"
    cell = sc.out_path(128, tier).name
    print(f"  {fam:<4} × {tier:<9} → 写 {written:<42} 读 {read:<42} 网格本 {cell}")
    if read != written:
        fails.append(f"[1] {fam} × {tier}：写盘名 {written} ≠ 读数名 {read} ⇒ 造出来的底没人读得到")

# ---------- [2] 不撞名回归（凭内容反算名字） ----------
print("\n[2] 盘上全部 v4b/v4c 账：按自己记的身份（含任务位）反算名字，必须逐字命中"
      "（加任务位不改**老账**的名字）")
missing_field = 0
n_hit = 0
n_hit_mnist = n_hit_new = 0
for p in sorted(HERE.glob("compress_report_v4[bc]*.json")):
    d = _doc(p)
    ds, fam, width, tier, n = _ident(d)
    if "dataset" not in d or "student_arch" not in d:
        missing_field += 1
    if p.name.startswith("compress_report_v4b"):
        sb.DATASET, sb.ARCH, sb.TIER, sb.WIDTH = ds, fam, tier or "", 128 if width == 128 else width
        got = sb.book_name(n).name
    else:
        sc.DATASET, sc.ARCH, sc.NSEEDS = ds, fam, n
        try:
            got = sc.out_path(width, tier or FULL).name
        except SystemExit as e:
            got = f"（拒跑：{str(e).splitlines()[0][:36]}…）"
    if got != p.name:
        fails.append(f"[2] {p.name}（自记 {ds}/{fam}/S{width}/{tier or '合并'}/{n} 流）反算出 {got} ⇒ "
                     f"{'名字规则改了历史账' if ds == 'mnist' else '新任务账的名字与登记处算出的不是同一个'}")
    else:
        n_hit += 1
        if ds == "mnist":
            n_hit_mnist += 1
        else:
            n_hit_new += 1
print(f"  反算命中 {n_hit} 本 / 扫描 {len(list(HERE.glob('compress_report_v4[bc]*.json')))} 本"
      f"（其中 {missing_field} 本没记 dataset 或 student_arch，按老账口径读成 mnist/mlp 并数在这里）")
print(f"  分两桶看：MNIST 侧历史账 {n_hit_mnist} 本（这一桶才是『加任务位不改名』的那半条判据）、"
      f"新任务账 {n_hit_new} 本。原先这里写死『非 mnist 即红』，前提是本发新造的账还没入库 ⇒ "
      f"入库后它把**名字本来正确**的账判成 {n_hit_new} 条假红（第一版就在这条上红过，"
      f"红的是预检自己）：断言要钉不变量（名字 = f(自记内容)），不能钉跑预检那一刻的盘态")

# ---------- [3] 反借账闸双向 + 第三向放行 ----------
print("\n[3] load_anchor 的任务位：fashion↔mnist 互借必须都拒，身份全对必须放行（否则是恒红摆设）")
src = _doc(HERE / "compress_report_v4b.json")
arm_key = f"{FULL} 种子{sc.SEEDS[0]}"
for want_ds, doc_ds, label in (("fashion", "mnist", "fashion 问 MNIST 的底"),
                               ("mnist", "fashion", "mnist 问 Fashion 的底")):
    d = dict(src, dataset=doc_ds)
    FAKE.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    sc.DATASET, sc.ARCH, sc.NSEEDS = want_ds, "mlp", 3
    try:
        sc.load_anchor(FAKE, 128, FULL, 3)
        fails.append(f"[3] {label} 没被拒（{want_ds} 读到了 {doc_ds} 的账）⇒ 任务位不在闸里")
    except SystemExit as e:
        print(f"  [OK] {label} 训练前拒跑：{str(e).splitlines()[0][:72]}…")
FAKE.write_text(json.dumps(dict(src, dataset="fashion"), ensure_ascii=False), encoding="utf-8")
sc.DATASET, sc.ARCH, sc.NSEEDS = "fashion", "mlp", 3
try:
    got = sc.load_anchor(FAKE, 128, FULL, 3)
    print(f"  [OK] 第三向放行：身份全对时读到锚点 {got['kd']['acc']} / 底 {got['floor']} 张"
          f"（⇒ 上面那两条红不是恒红）")
except SystemExit as e:
    fails.append(f"[3] 身份全对也被拒 ⇒ 那条闸是恒红摆设：{str(e).splitlines()[0][:60]}…")
FAKE.unlink(missing_ok=True)

# ---------- [4] Fashion 没有第4课锚点 ----------
print("\n[4] committed()：Fashion 侧必须无值可对（不许拿 MNIST 的数当锚点），MNIST 侧必须仍有值")
for ds, fam, tier, want in (("fashion", "mlp", FULL, None), ("fashion", "mlp", LOW, None),
                            ("mnist", "mlp", FULL, "有值"), ("mnist", "cnn", FULL, None)):
    sb.DATASET, sb.ARCH, sb.TIER, sb.WIDTH = ds, fam, tier, 128
    got = sb.committed(tier)
    ok = (got is None) if want is None else (got is not None)
    print(f"  {ds:<7} × {fam:<4} × {tier:<9} → {'None' if got is None else got}  "
          + ("[OK]" if ok else "[期望不符]"))
    if not ok:
        fails.append(f"[4] {ds} × {fam} × {tier} 的 committed() 返回 {got}，期望 {want}")

# ---------- [5] 登记处是唯一来源 ----------
print("\n[5] 任务登记处：权重名 / 档案精度 / 装载器 / 未登记任务硬退出 / 循环 import")
for ds in DATASETS:
    pt = MODELS / TEACHER_PT[ds]
    acc = teacher_archive_acc(ds)
    src_name = {"mnist": "TEACHER_ARCHIVE_ACC（模块常数，第4课那本）",
                "fashion": "compress_report_v6_fashion.json 现读"}[ds]
    print(f"  {ds:<7} 后缀 {DATASET_SUFFIX[ds]!r:<11} 权重 {pt.name:<24} 在盘={pt.exists()} "
          f"档案精度={acc} ← {src_name}")
    if not pt.exists():
        fails.append(f"[5] {ds} 的老师权重 {pt.name} 不在盘上 ⇒ 这一本的读数没有来源")
try:
    teacher_archive_acc("cifar")
    fails.append("[5] 未登记的任务名没被拒 ⇒ 登记处会安静回落")
except SystemExit as e:
    print(f"  [OK] 未登记任务硬退出：{str(e).splitlines()[0][:70]}…")
def _steps_of(mod: str) -> tuple[list[str], bool]:
    """干净子进程里只 import 这一个模块，看它把哪些 step* 拉了进来、其中有没有 step6。
    归属必须逐模块问：本进程先 import 过 step4b，而 step4b 顶层就 import step4e、step4e 顶层 import step6
    ⇒ 拿"当前进程的 sys.modules"判成环，会把那条既有链算到登记处头上（第一版就红在这里）。"""
    code = (f"import {mod}; import sys; "
            f"print('|'.join(sorted(m for m in sys.modules if m.startswith('step'))))")
    r = subprocess.run([sys.executable, "-u", "-c", code], cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    mods = [m for m in (r.stdout or "").strip().split("|") if m]
    return mods, "step6_fashion_replication" in mods

print(f"  [循环] 各起干净子进程问一次（只有前两个是不变量）：")
for mod, must_be_clean in (("step4_distill", True), ("step4c_alpha_T_sweep", True),
                           ("step4b_seed_variance", False)):
    mods, dirty = _steps_of(mod)
    pulled = [m for m in mods if m.startswith("step6")]
    print(f"    import {mod:<26} → 拉到 step6={dirty}"
          + ("" if not dirty else f"（来源 {'、'.join(pulled)}）")
          + ("　← 登记处若在此处顶层 import step6，换新任务就成环" if must_be_clean
             else "　← 既有链 step4b→step4e→step6，只出声不判红"))
    if must_be_clean and dirty:
        fails.append(f"[5] {mod} 一被 import 就把 step6 拉进来 ⇒ 顶层 import 成环，延迟 import 没生效")
from step4_distill import dataset_loader
for ds in DATASETS:
    fn = dataset_loader(ds)
    print(f"  [OK] dataset_loader({ds!r}) → {fn.__module__}.{fn.__name__}（没真装载，装载在 CLI 那一侧）")
print(f"  问完 fashion 之后，本进程里有 step6 了={('step6_fashion_replication' in sys.modules)}"
      f"（延迟 import 的兑现处就在这一行：先前那次 _steps_of(step4b) 也拉过它，但那是别的进程）")

# ---------- [6] 活体闸：真跑 CLI，但**先证它在这一刻到不了训练** ----------
print("\n[6] 真跑 CLI（训练前退出）：Fashion × MLP × 满数据 × --nseeds 8 不该被 MLP 的合并档案闸拦下")
# 这一腿吃的坐标在入库前后是**两种盘态**，而只有一种是零训练的：
#   目标 8 流本已在盘上   → 必然死在 step4b 的 [已入库] 闸（训练之前）⇒ 安全，可以 launch
#   同坐标 3 流底也不在   → 必然死在 [缺那本 8 流基准底]（训练之前）⇒ 安全，可以 launch
#   底在、目标本不在      → 它会**真去训练 16 次**（8 流 × 两臂）⇒ 这一腿不再零训练，不许 launch
# 本回合就摔在第三态：底入库后复跑预检，它烧了 16 次并写出 `compress_report_v4b_fashion_n8.json`，
# 而那句"退 0 = 它真去训练了"的红是**跑完之后**才叫的——训练不能倒带。所以现在把判定挪到 launch 之前。
sb.DATASET, sb.ARCH, sb.TIER, sb.WIDTH = "fashion", "mlp", FULL, 128
tgt8, tgt3 = sb.book_name(8), sb.book_name(3)
will_train = (not tgt8.exists()) and tgt3.exists()
print(f"  [盘态] 目标本 {tgt8.name} 在盘={tgt8.exists()} ／ 同坐标 3 流底 {tgt3.name} 在盘={tgt3.exists()} "
      f"⇒ {'**第三态：这一腿会真烧训练，不 launch**' if will_train else '训练前必退，可以 launch'}")
if will_train:
    fails.append("[6] 当前盘态下这一腿不再是零训练（3 流底已入库、8 流目标本还没有 ⇒ step4b 会跑完 16 次训练）"
                 "⇒ 拒跑是对的，但**这不等于通过**：要么先把目标本造出来（那是 16 次的钱，不在这发的价签里），"
                 "要么给 step4b 加一个『打印计划后退出』的入口，这一腿才有活体形状")
else:
    r = subprocess.run([sys.executable, "-u", "step4b_seed_variance.py", "--dataset", "fashion",
                        "--arch", "mlp", "--tier", FULL, "--nseeds", "8"],
                       cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
    msg = (r.stdout or "") + (r.stderr or "")
    tail = [ln for ln in msg.strip().splitlines() if ln.strip()][-1:] or ["（零输出）"]
    merged = "属于合并档案" in msg
    n8 = "当跨脚本基准" in msg and "不在盘上" in msg
    committed = "已入库" in msg
    print(f"  rc={r.returncode}  被合并档案闸拦下={merged}  报的是缺那一本底/基准={n8}  报的是已入库={committed}")
    print(f"  末行：{tail[0][:110]}")
    if r.returncode == 0:
        fails.append("[6] 退 0 = 它真去训练了：上面那句盘态判定错了（或 step4b 的训练前闸改了位置），"
                     "这条要查，因为本回合已经为这个形状烧过 16 次")
    elif merged:
        fails.append("[6] Fashion 的单档跑法被 MLP 的合并档案闸拦下 ⇒ 任务位没接进那道闸的历史口径")
    elif not (n8 or committed):
        fails.append("[6] 拦下了但报的不是『缺那一本底/基准』也不是『已入库』⇒ 拒跑原因接错了线，"
                     "红得没有信息量（两种合法的训练前退出都得点名）")

r2 = subprocess.run([sys.executable, "-u", "step4b_seed_variance.py", "--arch", "mlp", "--tier", FULL],
                    cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")
m2 = (r2.stdout or "") + (r2.stderr or "")
print(f"  [对照] MNIST × MLP × 满数据 仍被合并档案闸拦下：rc={r2.returncode} 命中={'属于合并档案' in m2}")
if r2.returncode == 0 or "属于合并档案" not in m2:
    fails.append("[6] 对照腿红：MLP 两档共本那道闸不再生效 ⇒ 新登记的空后缀炸了历史")

print("\n" + ("[预检] 全绿 ⇒ 在当前盘态下：名字规则逐字命中、反借账闸三向成立、登记处唯一来源、"
              "两条活体腿都在训练之前退出。每段开跑前复跑本件（36 次底已到账，网格那一趟在路上）"
              if not fails else "[预检] 有红，先别训练："))
for f in fails:
    print("  ✗ " + f)
raise SystemExit(1 if fails else 0)
