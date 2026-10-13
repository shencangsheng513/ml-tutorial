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
