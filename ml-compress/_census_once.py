# 四桶清点（入库件·第四十一发 D；判据与第卅六／卅八／卅九／四十发同源）
#
# 只做**只读盘点**：全程一个文件都不删。删除必须用户逐项确认，且执行前要用同一件重跑一次
# （桶每发都过期——第卅三发量到两条桶迁移，第卅六发量到 47→83）。
#
# 与上一发那件一次性件（本机、不入库）的差别只有一处，而那一处是本发要还的债：
# 「哪些件必须留着」不再由我写在正文里的话决定，而是**现读**同目录下那张保留登记处
# （见 KEEP_PATH）。登记衰减、条目不在盘、受保护件掉进删除面、名单里出现清点面之外的名字、
# 整张表变成恒真冗余——这几种坏法各有只有它拦得住的输入，都由本件的自证逐条验。
#
# 用法：
#   python _census_once.py              盘点并公示四桶 + 保护名单 + 族对账，并跑常驻自证
#   python _census_once.py --selftest   额外跑七条注入腿（逐项摘名单／加一条假条目／坏写法／同行歧义
#                                       ／摘一族／塞一个假族／盘面与在册双双喂空）
# 退出码：0 = 盘点跑通且全部票绿；1 = 有票红（红条逐条印出）。
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEEP_PATH = os.path.join(HERE, "_census_keep.txt")
SELF = os.path.basename(os.path.abspath(__file__))
LABEL = {HERE: "./", ROOT: "../"}

# 我造的那些件的前缀族。这一栏**仍是手写的**——它挡的是"别人的件"，而且删除候选面必须永远由手写
# 的那一栏圈定（形状自动收 = 把爆炸半径交给命名习惯）。第四十四发起它不再是一笔无票的失明：
# 下面的 family_check 把"盘上现读的族 ⊆ 在册的族"做成常驻两向票，登记处落后一族就当场点名。
MINE = ("_tmp_r28", "_tmp_r29", "_tmp_r30", "_tmp_r31", "_tmp_r32", "_tmp_r33", "_tmp_r34", "_tmp_r35",
        "_tmp_r36", "_tmp_r37", "_tmp_r38", "_tmp_r39", "_tmp_r40", "_tmp_r41", "_tmp_r43", "_tmp_r44",
        "_tmp_r45", "_tmp_r46", "_tmp_r47",
        "_red_batch_r43", "_red_batch_r44", "_red_batch_r45", "_red_batch_r46", "_red_batch_r47",
        "_tmp_red2", "_tmp_oldrule_check",
        "_preflight", "_v4l_", "_tmp_b_preflight", "_tmp_e_rot", "_tmp_l_audit", "_tmp_patch_table",
        "_tmp_c_probe")
NOT_MINE = ("ml-decision", "step0_diagnostic", "multihop_ab_v4_replica50", "models")
EXPLICIT_NOT_MINE = {"_preflight_c.py", "_tmp_b_preflight.py", "_tmp_e_rot.py", "_tmp_l_audit.py",
                     "_tmp_patch_table.py", "_v4l_fashion_v2_no_legcheck_v1.json",
                     "_v4l_fashion_v2_pricetag234_v1.json", "_v4l_fashion_v2_swallowedkeys_v1.json",
                     "_v4l_handcount_bad_v1.json"}

# 族＝前缀 + 发号（例：本机一次性件、不在库里的 `_tmp_r43_batch2.log` 与 `_red_batch_r43_stop/` 同属
# `_tmp_r43`／`_red_batch_r43`——clone 到别人机器上查无原件，所以它们只做形状举例，不当凭据引）。
# 只有这两个前缀参与对账：`_tmp_red2` 那类老 literal 族尾随的不是数字，不参与发号生命周期。
FAMILY_PREFIXES = ("_tmp_r", "_red_batch_r")
FAMILY_FLOOR = 3              # 盘上现读的族数少到个位 ⇒ 判"面塌"（第卅九发那一轮在册族 13 个）


def _family_of(name):
    for p in FAMILY_PREFIXES:
        if name.startswith(p):
            digits = ""
            for ch in name[len(p):]:
                if not ch.isdigit():
                    break
                digits += ch
            if digits:
                return p + digits
    return None


def families_on_disk(bases=(HERE, ROOT)):
    out = set()
    for base in bases:
        for name in os.listdir(base):
            if name == SELF or any(k in name for k in NOT_MINE):
                continue
            fam = _family_of(name)
            if fam:
                out.add(fam)
    return out


def families_in_registry(mine=MINE):
    return {f for f in (_family_of(x) for x in mine) if f}


def family_check(disk=None, reg=None):
    """常驻两向票。反向（面外新族）钉的是第 42／43 发那笔 ㈧⑴：登记处停在 r41，于是本轮 53 件
    对四个桶完全隐形，而清点件照样回一张"🟢 0 项、一个文件都没删"的绿票。
    正向（僵尸族）钉相反那一型：族清完了条目还留着，下一发照名单排除会静默少排一件。
    两向都不许由被检对象自己算分母；面塌（现读族数为零）单独出一条，免得"空集合减空集合"印成通过。"""
    disk = families_on_disk() if disk is None else set(disk)
    reg = families_in_registry() if reg is None else set(reg)
    fails = []
    outside = sorted(disk - reg)
    zombie = sorted(reg - disk)
    if len(disk) < FAMILY_FLOOR:
        fails.append(f"族对账（面塌）：盘上现读到的族只有 {len(disk)} 个（下限 {FAMILY_FLOOR}）⇒ "
                     "这一格不是『没有新族』，是**压根没扫到盘**（cwd 不对、列目录失败、或前缀全换了）；"
                     "空盘面与干净盘面在这行字上必须长得不一样")
    if outside:
        fails.append(f"族对账（面外新族）：盘上有 {'、'.join(outside)} 这些族（前缀 + 发号，是我的形状），"
                     "而 MINE 那一栏不含它们 ⇒ 四个桶对它们完全失明：既不进删除候选，也不受登记处保护。"
                     "把族补进 MINE 并当场量它，不许只在这里念一句")
    if zombie:
        fails.append(f"族对账（僵尸族）：MINE 登记着 {'、'.join(zombie)}，而盘上一件都没有 ⇒ "
                     "这一族谁也不保（它只让下一发照名单动手时多排一次空），撤条目或把原件请回来")
    return fails, {"disk": len(disk), "reg": len(reg), "outside": outside, "zombie": zombie}


def _git(args, cwd):
    r = subprocess.run(["git", "-c", "core.quotepath=false"] + args, cwd=cwd, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"清点要用 git 现读在册名单，而 {args} 在 {cwd} 回退码 {r.returncode}"
                         f"（stderr {r.stderr[:80]!r}）⇒ 分母读不到就不许报桶数")
    return [x for x in r.stdout.splitlines() if x]


def tracked_bodies():
    """引用面＝**在册**的 .py/.md：模块内的按模块相对名，仓根的加 [根] 前缀（与上一发同源）。
    草稿引草稿不算引用面——第卅发那条坑（把 pathspec 当过滤用）就是分母坏，不是正文坏。
    仓根那份名单**也含本模块的文件**（以 `ml-compress/xxx` 的形式），所以按绝对路径去重，
    否则同一本正文会被读两遍、引用面那一列的分母虚高（上一发的一次性件就是这副样子）。"""
    bodies, seen = {}, set()
    for p in _git(["ls-files", "--", "."], HERE):
        fp = os.path.join(HERE, p)
        if p.endswith((".py", ".md")) and os.path.isfile(fp):
            bodies[p] = open(fp, encoding="utf-8", errors="replace").read()
            seen.add(os.path.normcase(os.path.abspath(fp)))
    for p in _git(["ls-files", "--full-name", ":/"], ROOT):
        fp = os.path.join(ROOT, p)
        if not p.endswith((".py", ".md")) or not os.path.isfile(fp):
            continue
        if os.path.normcase(os.path.abspath(fp)) in seen:
            continue
        seen.add(os.path.normcase(os.path.abspath(fp)))
        bodies["[根]" + p] = open(fp, encoding="utf-8", errors="replace").read()
    return bodies


def parse_keep(lines):
    """逐行解析保留登记处 → (条目, 坏写法)。条目＝(标签, 全路径, 名字)；口径①之外的写法一律**拒读**。
    剥前缀这一步是承重的：../ 有三个字符，只剥两个就给名字带出一个多余的 `/`，于是拼出来的全路径
    在盘上必然不存在（报两条假『登记衰减』），而 keep_map 的键与清点面上的名字也对不上（保护整个静默
    失效）。所以剥完还含 `/` 的行按坏写法拒读，而不是收下再少排一件。"""
    entries, bad = [], []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("./"):
            base, rel = HERE, s[2:]
        elif s.startswith("../"):
            base, rel = ROOT, s[3:]
        else:
            bad.append(s)
            continue
        name = rel.rstrip("/")
        if "/" in name:
            bad.append(f"{s}｜剥掉前缀之后名字里仍含 /（../ 与 `./` 之外一律不认）")
            continue
        entries.append((LABEL[base], os.path.join(base, name), name))
    return entries, bad


def read_keep(path=KEEP_PATH):
    """现读保留登记处（口径见 `parse_keep`）。文件读不到就直接短路：保护名单是本件的全部前提。"""
    if not os.path.isfile(path):
        raise SystemExit(f"保留登记处读不到（{path}）⇒ 保护名单是本件的全部前提，缺它不许报删除面")
    with open(path, encoding="utf-8") as f:
        return parse_keep(f.read().splitlines())


def candidates():
    rows = []
    for base in (HERE, ROOT):
        for name in sorted(os.listdir(base)):
            if name == SELF:
                continue
            if not any(name.startswith(k) or name == k for k in MINE):
                continue
            if any(k in name for k in NOT_MINE):
                continue
            rows.append((LABEL[base], os.path.join(base, name), name))
    return rows


def measure(full):
    if os.path.isdir(full):
        files = [os.path.join(r, f) for r, _, fs in os.walk(full) for f in fs]
        return f"目录（{len(files)} 个文件）", sum(os.path.getsize(p) for p in files)
    return "文件", os.path.getsize(full)


def classify(label, full, name, cited, keep_map):
    if name in EXPLICIT_NOT_MINE:
        return "⛔不是我的", ""
    if (label, name) in keep_map:
        return "🟠登记处保留", ("（正文也点名 ⇒ 双保）" if cited else "（只有登记处撑着 ⇒ 这一项承重）")
    if cited:
        return "🟡被正文引用", ""
    if os.path.isdir(full):
        return "🔴墙钟原件（目录）", ""
    return "🟢零引用", ""


def census(keep_map, bodies):
    order = {"🟢零引用": 0, "🟠登记处保留": 1, "🟡被正文引用": 2, "🔴墙钟原件（目录）": 3, "⛔不是我的": 4}
    rows = []
    for label, full, name in candidates():
        cited = [k for k, b in bodies.items() if name in b]
        bkt, note = classify(label, full, name, cited, keep_map)
        kind, size = measure(full)
        rows.append((bkt, label + name, kind, size, cited, note, label, name))
    rows.sort(key=lambda r: (order[r[0]], r[1]))
    return rows


def report(rows):
    out = []
    for bkt, rel, kind, size, cited, note, _l, _n in rows:
        out.append(f"{bkt}  {rel:48s} {kind:16s} {size:>10,d} B"
                   + (f"  ← {', '.join(cited[:3])}" if cited else "") + note)
    cnt = lambda p: len([r for r in rows if r[0].startswith(p)])  # noqa: E731
    g = [r for r in rows if r[0] == "🟢零引用"]
    out.append(f"\n合计 {len(rows)} 项；🟢 {cnt('🟢')} 项 / {sum(r[3] for r in g):,} 字节；🟠 {cnt('🟠')} 项；"
               f"🟡 {cnt('🟡')} 项；🔴 {cnt('🔴')} 项；⛔ {cnt('⛔')} 项")
    return "\n".join(out)


def votes(entries, bad, rows, bodies, verbose=True):
    """常驻三条：口径①拒读／登记衰减／锁步（名单点名却掉进删除面）＋清点面之外＋恒真冗余。"""
    fails = []
    if bad:
        fails.append(f"登记处有 {len(bad)} 行写法不合口径（{bad[:3]}）⇒ 本件对这样的行**拒读**，"
                     "不是宽容收下然后少排一件")
    face = {(r[6], r[7]) for r in rows}
    for label, full, name in entries:
        if not os.path.exists(full):
            fails.append(f"登记衰减：名单里 {label}{name} 这一项在盘上不存在 ⇒ 它替一件已经不存在的东作保，"
                         "而下一次照名单执行的排除会静默少排一件（撤条目或把原件请回来）")
        elif (label, name) not in face:
            fails.append(f"名单里的 {label}{name} 在盘上、却不在清点面上（MINE 那一栏是手写的，前缀不含它）"
                         "⇒ 这一项谁也不保：要么把前缀补进 MINE 并当场量它，要么撤条目")
    green = {r[1] for r in rows if r[0] == "🟢零引用"}
    for label, full, name in entries:
        if label + name in green:
            fails.append(f"锁步断了：名单点名 {label}{name}，它却还落进 🟢 删除面 ⇒ 保护没生效，"
                         "照这份清单动手就会吃掉它")
    keep_rows = [r for r in rows if r[0] == "🟠登记处保留"]
    only = [r for r in keep_rows if "只有登记处撑着" in r[5]]
    if entries and not only:
        _in_face = [(label, name) for label, _f, name in entries if (label, name) in face]
        if len(_in_face) != len(entries):
            fails.append(f"登记处现读 {len(entries)} 项，其中 {len(entries) - len(_in_face)} 项**根本没落到清点面上**"
                         f"（{'、'.join(l + n for l, _f, n in entries if (l, n) not in face)[:90]}）⇒ 这一格本件不下"
                         f"『恒真冗余』的判断：名单没生效（键对不上、前缀剥错、或 MINE 不含它）与名单是冗余"
                         f"是两种相反的坏法，不许打在同一行字上（第 112／113 条那一族）——先按上面"
                         f"『登记衰减／不在清点面』那几条修取物路径，再回来看这一条")
        else:
            fails.append(f"登记处现读 {len(entries)} 项，全部同时被在册正文点着名 ⇒ 这一发这张表是恒真的冗余"
                         "（正文早把它挡在 🟡 了）。要么把本轮新造的锚件在**决定留它的那一刻**登进来，"
                         "要么撤掉整张表；留一张恒真的表就是「第 115 条」那一型")
    if verbose:
        print(f"\n[自证·常驻] 名单现读 {len(entries)} 项（坏写法 {len(bad)} 行）；落进 🟠 的 {len(keep_rows)} 项，"
              f"其中只有登记处撑着的 {len(only)} 项（承重）；掉进 🟢 的 {len([r for r in keep_rows if r[1] in green])}"
              " 项；引用面 " + str(len(bodies)) + " 本")
    return fails


def selftest(entries, rows, bodies):
    fails = []
    for label, full, name in entries:
        slim = {(l, nm) for l, _f, nm in entries if (l, nm) != (label, name)}
        r2 = census(slim, bodies)
        row = next((r for r in r2 if r[1] == label + name), None)
        if row is None:
            fails.append(f"注入1（摘掉 {label}{name}）之后这件从清点面上消失了 ⇒ 桶的算法跟着名单抖，"
                         "读数不能信（先查 candidates 的遍历有没有被这条改动影响）")
        elif row[0] == "🟠登记处保留":
            fails.append(f"注入1（摘掉 {label}{name}）之后它仍受登记处保护 ⇒ 名单里有重复行或别的行把它顶住了，"
                         "这一项不承重")
        else:
            print(f"  注入1（摘项）·承重凭据：摘掉 {label}{name} ⇒ 它从 🟠 落到 {row[0]}"
                  f"（正文引用 {len(row[4])} 处）⇒ 保护确实由名单给出，不是措辞")
    fake = ("../", os.path.join(ROOT, "_census_injected_fake.txt"), "_census_injected_fake.txt")
    f2 = votes(entries + [fake], [], rows, bodies, verbose=False)
    if not any("登记衰减" in x and fake[2] in x for x in f2):
        fails.append(f"注入2（假条目）往名单里加一个盘上不存在的名字 {fake[2]}，却没点亮登记衰减那条红"
                     "⇒ 那条判据是摆设，名单真坏了也不会出声")
    else:
        print("  注入2（假条目）·有牙凭据：往名单里加一条盘上不存在的名字 ⇒ 当场报登记衰减（这条红由注入产生，"
              "不落进上面的申报）")
    # 注入3（坏写法承重）：口径①那条"拒读"是不是有牙，只有喂三种写法才知道——无前缀、正确 ../、
    # 以及 ..//（少剥一个字符就会长成这样，也是本件第 ① 版真实踩过的坑：条目名带出多余的 /，
    # 拼出的路径必然不在盘上 ⇒ 两条假『登记衰减』，而 keep_map 的键对不上 ⇒ 保护静默失效、全程不红）。
    _i3_e, _i3_b = parse_keep(["_tmp_r40_red_citedlink.txt", "../_tmp_r40_red_citedlink.txt",
                               "..//_tmp_r40_red_citedlink.txt"])
    if len(_i3_e) != 1 or len(_i3_b) != 2 or "/" in (_i3_e[0][2] if _i3_e else "/"):
        fails.append(f"注入3（坏写法）喂三行（无前缀／正确 ../／..//）⇒ 现读 成条目 {len(_i3_e)} 行、"
                     f"拒读 {len(_i3_b)} 行（要 1 与 2，且条目名不含 /；现读条目名 "
                     f"{[e[2] for e in _i3_e]}）⇒ 『口径① 拒读』这条判据没牙或对前缀的剥法错了，"
                     f"这样的名单会把保护变成一句空话而不自知")
    else:
        print(f"  注入3（坏写法）·有牙凭据：三行写法只有正前缀那一行成条目（{_i3_e[0][2]}），"
              f"无前缀与 ..// 两行都被拒读 ⇒ 少剥／多剥一个字符都会出声，不会静默少排一件")
    # 注入4（两种坏法不许同行）：只喂一条**不在清点面上**的条目，并把 🟠 行从 rows 里摘掉（制造
    # "没有承重项"的表面）。此时该报的是"名单没生效"，不该报"恒真冗余"——后者会让人以为判据变严了。
    _rows_nk = [r for r in rows if r[0] != "🟠登记处保留"]
    _f4 = votes([fake], [], _rows_nk, bodies, verbose=False)
    _f4_redundant = [x for x in _f4 if "恒真的冗余" in x]
    _f4_notface = [x for x in _f4 if "根本没落到清点面上" in x]
    if _f4_redundant or not _f4_notface:
        fails.append(f"注入4（同行歧义）只喂一条不在清点面上的条目 ⇒ 现读 报冗余 {len(_f4_redundant)} 条、"
                     f"报没生效 {len(_f4_notface)} 条（要 0 与 ≥1）⇒ 『名单没生效』仍会借『恒真冗余』那句话"
                     f"出口，而这两件事一个是保护是空的、一个是判据变严了，读法完全相反")
    else:
        print("  注入4（同行歧义）·有牙凭据：只喂一条不在清点面上的条目 ⇒ 报的是『名单没生效』而不是"
              "『恒真冗余』（这条红由注入产生，不落进上面的申报）")
    # 注入5／6／7（族对账三腿）：这三条判据各自的反面凭据都是"喂进去它不红"，所以判法统一取
    # **相对常驻基线新增的红**——基线此刻可能正红着（MINE 落后盘面），拿绝对条数判就会把注入腿
    # 读成误报，或者反过来把基线的那条红算成注入的牙。
    _disk7, _reg7 = families_on_disk(), families_in_registry()
    _base7, _ = family_check(_disk7, _reg7)
    _both7 = sorted(_disk7 & _reg7)
    if not _both7:
        fails.append("注入5（族·承重）盘上现读族与在册族**没有交集** ⇒ 这一腿压根跑不到：摘谁都摘不动在册"
                     "那一份，面外新族那条红的承重性没有凭据。先把 MINE 与盘面对齐（看上面族对账那两条红）"
                     "再回来看这一条")
    for fam in _both7:
        _new5 = [x for x in family_check(_disk7, _reg7 - {fam})[0] if x not in _base7]
        if len(_new5) != 1 or "面外新族" not in _new5[0] or fam not in _new5[0]:
            fails.append(f"注入5（族·承重）从在册里摘掉在盘上的 {fam} ⇒ 相对基线新增的红是 {_new5}"
                         f"（要恰好 1 条、且那条点名 {fam}）⇒ 在册少写一族要么不出声，要么出的是别的账，"
                         f"读不出这一族的债归谁")
        else:
            print(f"  注入5（族·承重）·凭据：从在册里摘掉在盘上的 {fam} ⇒ 只新增一条『面外新族』红并点名它"
                  f"（这条红由注入产生，不落进上面的申报）⇒ 那一族确由在册那一栏单独承重")
    _fake7 = "_tmp_r99"
    _new6 = [x for x in family_check(_disk7, _reg7 | {_fake7})[0] if x not in _base7]
    if len(_new6) != 1 or "僵尸族" not in _new6[0] or _fake7 not in _new6[0]:
        fails.append(f"注入6（族·有牙）往在册里塞一个盘上没有的族 {_fake7} ⇒ 相对基线新增的红是 {_new6}"
                     f"（要恰好 1 条『僵尸族』且点名 {_fake7}）⇒ 撤干净的那一栏还赖在名单里时不会出声，"
                     f"下一发照名单排除就多排一次空")
    else:
        print(f"  注入6（族·有牙）·凭据：往在册里塞一个盘上没有的族 {_fake7} ⇒ 只新增一条『僵尸族』红并点名它"
              "（这条红由注入产生，不落进上面的申报）")
    _new7 = [x for x in family_check(set(), set())[0] if x not in _base7]
    if not any("面塌" in x for x in _new7):
        fails.append(f"注入7（面塌）把盘面与在册双双喂空 ⇒ 相对基线新增的红是 {_new7}（要含一条『面塌』）"
                     "⇒ 空集合减空集合被印成通过：扫不到盘（cwd 不对、列目录失败、前缀全换了）时这张票"
                     "会跟着一起失明，而这正是本轮要治的那一型")
    else:
        print("  注入7（面塌）·有牙凭据：盘面与在册双双喂空 ⇒ 新增一条『面塌』红而不是 0 红"
              "（空盘面与干净盘面在这行字上长得不一样）")
    return fails


def main():
    inject = "--selftest" in sys.argv[1:]
    entries, bad = read_keep()
    bodies = tracked_bodies()
    keep_map = {(label, name) for label, _f, name in entries}
    rows = census(keep_map, bodies)
    print(report(rows))
    fails = votes(entries, bad, rows, bodies)
    fam_fails, fam_info = family_check()
    fails += fam_fails
    print(f"\n[自证·族对账] 盘上现读族 {fam_info['disk']} 个／在册族 {fam_info['reg']} 个；"
          f"面外新族 {len(fam_info['outside'])} 个"
          f"{'（' + '、'.join(fam_info['outside']) + '）' if fam_info['outside'] else ''}；"
          f"僵尸族 {len(fam_info['zombie'])} 个"
          f"{'（' + '、'.join(fam_info['zombie']) + '）' if fam_info['zombie'] else ''}")
    if inject:
        fails += selftest(entries, rows, bodies)
    if fails:
        print(f"\n>>> 清点自证 {len(fails)} 条红：")
        for f in fails:
            print("   ", f)
        raise SystemExit(1)
    print("\n结论：只读盘点，**一个文件都没删**；删除面＝上面那个 🟢 桶，逐项确认之前不许动手。")


if __name__ == "__main__":
    main()
