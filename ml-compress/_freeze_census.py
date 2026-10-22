# -*- coding: utf-8 -*-
# 冻结清点件（第四十三发 B·还第四十二发新欠⑻）。
#
# 为什么需要它：一票冻结从前只回答"变了多少件"，不回答"谁动的"。第卅二发的现场是：批中在册 markdown 正文两处
# 变了，那一句"不是一票冻结"是我逐件去对内容与钟点才得出的——**归因住在我脑子里，不住在件里**，
# 于是它每一发都要重新人肉做一遍，而下一发只要我忘了对，就会把"第三方写过"读成"冻结成立"。
# 这个件把归因做成代码：快照给全跟踪面每件取 size@mtime@sha1，比对给三列（我名下／非我／申报未动）
# 再加两列它自己会坏的地方（申报了但不在跟踪面、读不到摘要），判词只由这五列算，不由人补。
#
# 两格必须分开（这是本件唯一容易被写坏的地方）：
#   declared=None ⇒ 本轮**没有申报**，只报"变了多少件"，永远不许印冻结成立；
#   declared=[]   ⇒ 申报"本轮我没写在跟踪面上的件"，此时任何变更都归"非我"。
# 把 None 当空名单用，就等于"没人申报 ⇒ 别人动的全算我名下"，那是第 145 条那一型（读失败与读干净同行）。
#
# 用法（本目录）：
#   PYTHONUTF8=1 python -u _freeze_census.py snap --out _red_batch_logs/freeze_before.txt
#   PYTHONUTF8=1 python -u _freeze_census.py diff --before ..._before.txt --after ..._after.txt --mine=
# 整轮由 _red_batch.py 在批前批后各调一次，把归属那一行印进回执。
import argparse
import hashlib
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FIELD_SEP = "\t"
UNREAD = "unread"
NAME_CAP = 12          # 名单超过这么长就只印前若干件＋"另 k 件"，分母照旧在最前面


def repo_root():
    """仓根由 git 现读，不写死。写死它等于赌板块永远挂在同一个目录下。"""
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("读不到仓根（git rev-parse --show-toplevel 退码 " + str(r.returncode)
                         + "）⇒ 扫描面无从确定，这一票不许按『没有变更』记")
    return r.stdout.strip().replace("\\", "/")


def norm(p):
    q = str(p).replace("\\", "/")
    while q.startswith("./"):
        q = q[2:]
    return q.strip("/")


def tracked_face(root=None):
    """全仓跟踪面（仓根相对名）。空名单必须出声：它会让比对回"零变更"，也就是最坏的那种假绿。"""
    r = subprocess.run(["git", "-c", "core.quotepath=false", "ls-files", "--full-name"],
                       cwd=root or repo_root(), capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("读跟踪面失败（git ls-files 退码 " + str(r.returncode) + "）⇒ 同上，不许当成『面是空的』")
    face = sorted({norm(x) for x in r.stdout.splitlines() if x.strip()})
    if not face:
        raise SystemExit("跟踪面读到 0 件 ⇒ 是尺子瞎了（不在 git 工作树里？pathspec 坐标系错？），"
                         "不许回落成空批（第 145 条那一型：读失败与读干净打同一行字）")
    return face


def digest(abs_path):
    """一件的三票：(字节数, 修改钟点整秒, sha1)。读不到回 (UNREAD, 异常类名, 空串)——
    三列一起留空会跟另一侧的空串配成"未变"，那才是这件最坏的坏法；所以不可核有自己的第一个字段。"""
    try:
        with open(abs_path, "rb") as f:
            body = f.read()
        st = os.stat(abs_path)
        return (str(len(body)), str(int(st.st_mtime)), hashlib.sha1(body).hexdigest())
    except OSError as e:
        return (UNREAD, type(e).__name__, "")


def snap_map(root=None, face=None):
    root = root or repo_root()
    face = tracked_face(root) if face is None else face
    return {norm(p): digest(os.path.join(root, norm(p))) for p in face}


def write_snap(path, checked_at=None, root=None):
    """落一份可 diff 的原件：两行头（钟点＋分母）→ 逐件一行 → 末行 RC=0。
    末行 RC 是这件的"我是怎么结束的"凭据（第 174 条）：外层重定向被截时，头两行与 RC 行都不完整，
    读的那一侧当场拒绝，而不是拿半份快照比出个"零变更"。"""
    root = root or repo_root()
    m = snap_map(root)
    lines = ["# snap checked_at=" + (checked_at or time.strftime("%Y-%m-%d %H:%M:%S"))
             + " root=" + root,
             "# tracked=" + str(len(m))]
    for p in sorted(m):
        lines.append(FIELD_SEP.join([p] + list(m[p])))
    lines.append("# RC=0")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return m


def read_snap(path):
    """回 (map, meta)。四样任一不齐就 raise：宁可这一发没有冻结票，也不要一张读残缺了的票。"""
    if not os.path.isfile(path):
        raise SystemExit("快照件不存在：" + path + " ⇒ 这一发没有冻结票（不是『票过了』）")
    with open(path, encoding="utf-8") as f:
        raw = f.read().splitlines()
    head = [x for x in raw if x.startswith("# snap checked_at=")]
    den = [x for x in raw if x.startswith("# tracked=")]
    if len(head) != 1:
        raise SystemExit("快照件 " + path + " 的钟点头行有 " + str(len(head)) + " 行（要 1）⇒ 头被盖或写了两遍")
    if len(den) != 1:
        raise SystemExit("快照件 " + path + " 的分母行有 " + str(len(den)) + " 行（要 1）⇒ 没有分母的快照不许比")
    if not raw or raw[-1].strip() != "# RC=0":
        raise SystemExit("快照件 " + path + " 末行不是 # RC=0 ⇒ 那件没跑到收尾，它的内容不可引")
    body = [x for x in raw if x and not x.startswith("#")]
    m = {}
    for ln in body:
        parts = ln.split(FIELD_SEP)
        if len(parts) != 4:
            raise SystemExit("快照件 " + path + " 有一行不是 4 段（档名+三票）：" + ln[:60]
                             + " ⇒ 行被截或档名里混进了制表符，比对会把它读成『不存在的件』")
        m[norm(parts[0])] = (parts[1], parts[2], parts[3])
    want = int(den[0].split("=", 1)[1])
    if want != len(m):
        raise SystemExit("快照件 " + path + " 的分母写 " + str(want) + "，正文实际 " + str(len(m))
                         + " 件 ⇒ 分母与正文不符，这一份不可引")
    return m, {"checked_at": head[0].split("checked_at=", 1)[1].split(" root=", 1)[0],
               "tracked": want}


def _unread(v):
    """这一件在这一侧到底取到摘要没有：只有第一列回 UNREAD 才算不可核（None 是"这一侧没这件"，
    那是件数差，由 before_n/after_n 自己说，不许混进不可核那一列）。"""
    return bool(v) and v[0] == UNREAD


def attribute(before, after, declared):
    """三列归因 + 两列自曝。纯函数：不碰盘、不调 git，所以守卫的拆红档能把合成夹具直接喂进来。

    返回键：
      mine／other／untouched／offface／unread —— 五列都是**名单**（不是计数），判词按名单算；
      changed／before_n／after_n —— 分母；
      attributable —— declared 是不是真被申报过（None ⇒ False）；
      frozen —— 只在 attributable 且五列全空且两个分母相等时才 True。
    """
    named = None if declared is None else sorted({norm(x) for x in declared if norm(x)})
    face = sorted(set(before) | set(after))
    changed = [p for p in face if before.get(p) != after.get(p)]
    # 只在一侧出现的件（新增/删除）也算变更：上面那行对 None!=值 已经成立，这里不另开列，
    # 但件数差由 before_n/after_n 两列自己说出来，免得"面缩了"被读成"没变"。
    mine = [p for p in changed if named and p in set(named)]
    other = [p for p in changed if p not in mine]
    untouched = [] if named is None else [p for p in named if p in face and p not in changed]
    offface = [] if named is None else [p for p in named if p not in face]
    unread = [p for p in face if _unread(before.get(p)) or _unread(after.get(p))]
    frozen = (named is not None and not changed and not untouched and not offface
              and not unread and len(before) == len(after))
    return {"mine": sorted(mine), "other": sorted(other), "untouched": sorted(untouched),
            "offface": sorted(offface), "unread": sorted(unread),
            "changed": len(changed), "before_n": len(before), "after_n": len(after),
            "declared_n": 0 if named is None else len(named), "attributable": named is not None,
            "frozen": frozen}


def cap(names):
    return "、".join(names[:NAME_CAP]) + ("｜另 " + str(len(names) - NAME_CAP) + " 件"
                                        if len(names) > NAME_CAP else "")


def verdict_line(res):
    """一句自带五个分母与三列名单的判词。**这句就是给回执抄的那一行**，所以它不许依赖上下文。"""
    if not res["attributable"]:
        return ("归属：未申报（declared=None）⇒ 变了 " + str(res["changed"]) + " 件，只报数不归因，"
                "本行不是冻结票")
    if res["frozen"]:
        return ("归属：我名下 0／非我 0／申报未动 0／申报不在面 0／不可核 0"
                "（跟踪面 " + str(res["before_n"]) + "→" + str(res["after_n"]) + " 件；"
                "申报 " + str(res["declared_n"]) + " 件）⇒ 一票冻结成立")
    why = []
    if res["mine"]:
        why.append("我名下 " + str(len(res["mine"])) + " 件＝" + cap(res["mine"]))
    if res["other"]:
        why.append("非我 " + str(len(res["other"])) + " 件＝" + cap(res["other"]))
    if res["untouched"]:
        why.append("申报未动 " + str(len(res["untouched"])) + " 件＝" + cap(res["untouched"]))
    if res["offface"]:
        why.append("申报不在跟踪面 " + str(len(res["offface"])) + " 件＝" + cap(res["offface"]))
    if res["unread"]:
        why.append("不可核 " + str(len(res["unread"])) + " 件＝" + cap(res["unread"]))
    return ("归属：不是一票冻结（" + str(res["changed"]) + " 件变了，面 "
            + str(res["before_n"]) + "→" + str(res["after_n"]) + "；申报 "
            + str(res["declared_n"]) + " 件）：" + "｜".join(why))


def declare_precheck(declared, before_map, window_start=None):
    """起跑**前**的申报对照（第四十六发 E·结第四十五发新欠⑿）。

    病：`--mine` 的语义原先只由批后那句判词反着教。判据（`frozen` 里那条 `not untouched`）要求
    "申报了而本轮没再写"那一列是空的，所以只要我把窗口之前就已落盘的件报进名单，那一轮就注定
    交不出冻结票——可这句话要等跑完二十几分钟才印出来。原因在开跑之前就能算：批前快照里那件的
    mtime 早于窗口起点 ⇒ 它不是这一轮写的。本函数把它提前印成一行。

    纯函数：只吃 (申报名单, 批前快照 map, 窗口起点秒)，不碰盘、不调 git ⇒ --freezemine 那族拆红腿
    能拿合成夹具直接喂进来。declared=None（未申报）与 []（申报空名单）照旧分两格，同 attribute。

    判词为什么住在这一件里而不是跑批件里：起跑那一行由 `_red_batch.py` 直接印 res["pre_line"]，
    守卫的拆红腿对账的是**同一个字符串**；格式若搬到跑批件，守卫就只能拿自己重写的一份去比，
    那是第 115 条那一型（自己对自己恒真）。

    回的列：attributable／declared_n／on_face／off_face／unread／stale／fresh／checkable／pre_line。
    """
    named = None if declared is None else sorted({norm(x) for x in declared if norm(x)})
    checkable = window_start is not None
    on_face, off_face, unread, stale, fresh = [], [], [], [], []
    if named is not None:
        for p in named:
            v = before_map.get(p)
            if v is None:
                off_face.append(p)
                continue
            if _unread(v):
                unread.append(p)      # 钟点那一列此时装的是异常类名，不可判就不许算进 stale
                continue
            (stale if int(v[1]) < int(window_start) else fresh).append(p)
    res = {"attributable": named is not None, "declared_n": 0 if named is None else len(named),
           "on_face": sorted(on_face), "off_face": sorted(off_face), "unread": sorted(unread),
           "stale": sorted(stale), "fresh": sorted(fresh), "checkable": checkable,
           "window_start": window_start}
    head = "申报对照："
    if named is None:
        res["pre_line"] = (head + "未申报（declared=None）⇒ 这一轮本来就交不出冻结票，"
                           "所以本行不报过期件数——『没人申报』与『申报了零件』是两件事（㈧⑻ 那一族的同形）")
    elif not named:
        res["pre_line"] = (head + "申报 0 件（空名单＝申报『我没动跟踪面上的任何一件』）⇒ 没有要对照的对象，"
                           "批后那句归属里任何变更都归『非我』并逐件点名")
    elif not checkable:
        res["pre_line"] = (head + "申报 " + str(len(named)) + " 件，但没给窗口起点 ⇒ mtime 早不早**不可判**，"
                           "这一行不是『0 件过期』")
    else:
        parts = []
        if stale:
            parts.append("窗口之前就已落盘 " + str(len(stale)) + " 件＝" + cap(stale)
                         + "（若本轮不再写它们，批后它们会落进『申报未动』那一列，而判据要求那一列空"
                         " ⇒ 『一票冻结』那句当场作废：要么别把它们报进来，要么接受归属写成『不是冻结』）")
        if fresh:
            parts.append("窗口之内已写 " + str(len(fresh)) + " 件＝" + cap(fresh)
                         + "（这一类是真申报，批后应落在『我名下』那一列）")
        if off_face:
            parts.append("申报不在跟踪面 " + str(len(off_face)) + " 件＝" + cap(off_face)
                         + "（名字写错了：写成板块相对名或盘上没这路径，批后归『申报不在面』）")
        if unread:
            parts.append("摘要取不到 " + str(len(unread)) + " 件＝" + cap(unread)
                         + "（钟点那一列装的是异常类名 ⇒ 过期与否对它们不可判）")
        res["pre_line"] = (head + "申报 " + str(len(named)) + " 件｜窗口起点 "
                           + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(window_start))
                           + "｜" + "｜".join(parts)
                           + "｜限度：『窗口之前就已落盘』是**预测**不是判词——第三方若在批中改写它，"
                             "批后归因会把它转到『非我』那一列，那一型由 腿8 管，不由这一行管")
    return res


def main():
    ap = argparse.ArgumentParser(description="全跟踪面冻结快照与两向归因")
    ap.add_argument("cmd", choices=["snap", "diff"])
    ap.add_argument("--out", default=None, help="snap：快照件落盘路径")
    ap.add_argument("--before", default=None, help="diff：批前快照件")
    ap.add_argument("--after", default=None, help="diff：批后快照件")
    ap.add_argument("--mine", default=None,
                    help="diff：本轮我写过的那些件（仓根相对名，逗号分隔）。**不传本项 = 未申报**，"
                         "与传空串（= 申报『我没动跟踪面上的件』）是两件事，见文件头")
    a = ap.parse_args()
    if a.cmd == "snap":
        if not a.out:
            raise SystemExit("snap 要 --out：没有落盘路径的快照不是证据")
        m = write_snap(a.out)
        print("快照 " + a.out + "｜跟踪面 " + str(len(m)) + " 件｜钟点 "
              + time.strftime("%Y-%m-%d %H:%M:%S"))
        return 0
    before, mb = read_snap(a.before or _die("diff 要 --before"))
    after, ma = read_snap(a.after or _die("diff 要 --after"))
    declared = None if a.mine is None else [x.strip() for x in a.mine.replace("，", ",").split(",") if x.strip()]
    res = attribute(before, after, declared)
    print("快照钟点：批前 " + mb["checked_at"] + "｜批后 " + ma["checked_at"])
    print(verdict_line(res))
    if a.mine is None:
        print("      限度：未申报时这一行只报件数。它既不是『冻结成立』，也不是『别人动了』——"
              "后者需要有人先把自己写过的那件报上来。")
    return 0 if res["attributable"] else 3


def _die(msg):
    raise SystemExit(msg)


if __name__ == "__main__":
    sys.exit(main())
