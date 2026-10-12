"""第三件 · open() 之后，with 替你做的两件事（把字刷到盘上 + 松开文件）。

你只做一件事：双击跑起来，输入一个数字（写到第几行的时候炸），回车。
同一份内容会按七种情形各跑一遍（五种死法 + 两种松手状态），全在你自己的临时目录里，跑完当场删干净。

为什么值得看一眼：第二件里我说「写完立刻读回来比」，那是因为我用了 with。
这一件把「不用 with 会怎样」演给你看 —— 不是靠我说，是屏幕上那几个字节数自己说话。

事先说清楚：这个程序不联网（import 的模块就文件头那五个，subprocess 只启动你自己
硬盘上那几个小 python），也不读写你的打卡本 —— 第 1 节记下打卡本此刻的字节数和摘要，
第 7 节再量一次，两个一样才算这一件跑完。
"""

import hashlib
import io
import os
import subprocess
import sys
import tempfile

LINE = "x" * 800 + "\n"
MIN_K, MAX_K = 1, 30
BAR = "=" * 60
HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "write_demo_child.py")
PREFIX = "r48_lesson03_open_with"
APP_DIR_NAME = "agent-lesson-streak"
STREAK_FILE = "streak.json"

ROWS = [
    ("R1", "plain", "raise", None, "不用 with", "程序自己抛了个异常（最常见的死法）"),
    ("R2", "plain", "exithard", None, "不用 with", "进程被硬杀（os._exit ≈ 任务管理器结束进程 / 断电）"),
    ("R3", "with", "raise", None, "用 with", "同样抛异常，但外面包着 with"),
    ("R4", "with", "exithard", None, "用 with", "包了 with 也照样硬杀"),
    ("R5", "plain", "exithard", "flush", "不用 with，但死前 flush()", "硬杀之前自己刷一次盘"),
]


def streak_path():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, APP_DIR_NAME, STREAK_FILE)


def repo_root_under(dirpath):
    """这个目录（含它的上层）里有没有 .git；有就回那个仓库根，没有回 None。"""
    cur = os.path.abspath(dirpath)
    while True:
        marker = os.path.join(cur, ".git")
        if os.path.isdir(marker) or os.path.isfile(marker):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def stat2(path):
    """(字节数, sha 前 12)。不存在回 (None, None)；读不动回 (-1, 错误类名)。"""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except Exception as exc:
        return -1, type(exc).__name__
    return len(blob), hashlib.sha256(blob).hexdigest()[:12]


def disk_size(path):
    return os.path.getsize(path)


def delete_attempt(path):
    """真删。回 (删掉了没, 说明) —— 删不掉时把错误类名和 WinError 号一起打出来。"""
    try:
        os.remove(path)
        return True, "删掉了"
    except Exception as exc:
        win = getattr(exc, "winerror", None)
        return False, "%s%s" % (type(exc).__name__, ("（WinError %s）" % win) if win else "")


def rmdir_attempt(path):
    """删目录（必须是空的）。os.remove 删不了目录，所以单开一个。"""
    try:
        os.rmdir(path)
        return True, "目录删掉了"
    except Exception as exc:
        win = getattr(exc, "winerror", None)
        return False, "%s%s" % (type(exc).__name__, ("（WinError %s）" % win) if win else "")


def count_prefix(base):
    """这个临时目录里，名字以 %s 开头的东西还剩几个（每次跑都该归 0）。"""
    try:
        return len([n for n in os.listdir(base) if n.startswith(PREFIX)])
    except Exception:
        return -1


def ask_line():
    """回 (行号或 None, 档位, 原样输入)。五档各说各的话。"""
    sys.stdout.write("写到第几行的时候炸？（%d..%d，一行 %d 个字符；直接回车=不跑演示）："
                     % (MIN_K, MAX_K, len(LINE)))
    sys.stdout.flush()
    try:
        raw = sys.stdin.readline()
    except Exception:
        return None, "no-tty", ""
    if raw == "":
        return None, "no-tty", raw
    s = raw.strip()
    if s == "":
        return None, "enter", raw
    try:
        v = int(s)
    except ValueError:
        return None, "bad", raw
    if v < MIN_K or v > MAX_K:
        return min(max(v, MIN_K), MAX_K), "clamp", raw
    return v, "typed", raw


def mode_sentence(mode, k, raw):
    if mode == "typed":
        return "这一档：你输入了 %d —— 后面每一行演示都写到第 %d 行才停（炸或者睡着），一共 %d 行可写。" % (k, k, MAX_K)
    if mode == "enter":
        return "这一档：你只按了回车 —— 今天不跑演示，临时目录压根不建，一个文件都不会出现。"
    if mode == "bad":
        return "这一档：「%s」不是整数 —— 同样不跑演示，也不建目录。" % raw.strip()[:20]
    if mode == "clamp":
        return "这一档：你输入的数字超出 %d..%d，夹到 %d —— 上限就是演示一共几行，量出来的，不是我猜的。" % (MIN_K, MAX_K, k)
    return "这一档：键盘这一档不存在（没有 stdin）—— 不跑演示，连临时目录都省了；这一档存在只为让你看到「跑」和「不跑」打的是不同的行。"


def run_child(path, k, how, death, flush_flag, py):
    """起一个小 python 去写、去死。回 (磁盘字节数, 进程退码, 它临终报的异常类名或 '-')。"""
    argv = [py, "-X", "utf8", CHILD, path, str(k), how, death]
    if flush_flag:
        argv.append(flush_flag)
    r = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    err_name = "-"
    if r.returncode != 0 and r.stderr.strip():
        last = r.stderr.strip().splitlines()[-1]
        err_name = (last.split(":")[0] if ":" in last else last)[:24]
    return disk_size(path), r.returncode, err_name


def loss_note(total, size):
    if size == total:
        return "一个字没少"
    return "少了 %d 字节" % (total - size)


def run_sleeper(path, k, how, py):
    """起一个『写完还活着、句柄攥在手里不放』的小进程，趁它睡着的时候由我（别的程序）去删那个文件。
    回 (删掉了没, 说明, 它有没有报到 READY, 删那一刻磁盘上几字节)。"""
    p = subprocess.Popen([py, "-X", "utf8", CHILD, path, str(k), how, "sleep"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace")
    try:
        ready = p.stdout.readline().strip() == "READY"
    except Exception:
        ready = False
    size_now = disk_size(path) if os.path.isfile(path) else None
    deleted, note = delete_attempt(path)
    p.terminate()
    p.wait()
    return deleted, note, ready, size_now


def main():
    py = sys.executable
    target = streak_path()
    before = stat2(target)

    print(BAR)
    print("第三件 · open() 之后 with 干了什么　　python：%s" % py)
    print(BAR)
    print()

    print("[1/7] 在哪儿演示，以及四条保证")
    if not os.path.isfile(CHILD):
        print("      [STOP] 演示用的那个小文件不在：%s" % CHILD)
        print("             这两个 .py 得待在一起，别只搬一个。")
        return 2
    base = os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir()
    inside = repo_root_under(base)
    if inside is not None:
        print("      [STOP] 临时目录（%s）在 git 仓库里面：%s" % (base, inside))
        print("             这一件的演示文件不能写进仓库 —— 远端是公开的。换掉 TEMP 再跑（正常机器不用管）。")
        return 2
    print("      演示目录：%s" % base)
    print("      保证一：仓库外 —— 这句话是程序自己从这个路径往上逐层查 .git 查出来的，不是写死的。")
    print("      保证二：跑完当场删干净，第 7 节去查那个目录还在不在。")
    print("      保证三：你的打卡本 %s" % target)
    print("              这一件不读它的内容、也不写它，只量它的字节数和摘要：跑前 %s 字节 / %s，跑后再量一次。"
          % (before[0], before[1] if before[1] else "-"))
    print("      保证四：不联网（import 的模块就 hashlib/io/os/subprocess/sys/tempfile，没一个能出机器）。")
    print()

    k, mode, raw = ask_line()
    print("[2/7] 你输入的那一个数字")
    print("      %s" % mode_sentence(mode, k, raw))
    print()

    if mode in ("enter", "bad", "no-tty"):
        print("[3/7] 校准（一行在盘上占几字节）")
        print("      没跑演示，不建目录，也就没东西可校准。")
        print("[4/7] 反面（不用 with）")
        print("      没跑。")
        print("[5/7] 正面（用 with）")
        print("      没跑。")
        print("[6/7] 七行对照（写字组 R1..R5 + 松手组 R6..R7）")
        print("      没跑 —— 所以屏幕上不该出现 R1..R7 那七行。")
        print()
        print("[7/7] 收尾清点")
        n = count_prefix(base)
        print("      临时目录里以 %s 开头的东西剩几个：%d -> %s" % (PREFIX, n, "OK" if n == 0 else "FAIL"))
        after = stat2(target)
        ok_streak = before == after
        print("      打卡本：跑前 %s 字节 / %s → 跑后 %s 字节 / %s -> %s"
              % (before[0], before[1], after[0], after[1], "OK" if ok_streak else "FAIL（这一件动它了）"))
        print()
        print(BAR)
        if n != 0 or not ok_streak:
            print("结论：[FAIL] 说好什么都不建，结果盘上留了东西（见 [7/7]）。")
            return 1
        print("结论：今天没跑演示，盘上什么都没多出来。双击它，输入一个数字就看到了。")
        return 0

    d = tempfile.mkdtemp(prefix=PREFIX, dir=base)

    print("[3/7] 先量一把尺：一行字在盘上占几字节")
    cal = os.path.join(d, "cal.txt")
    with open(cal, "w", encoding="utf-8") as cf:
        cf.write(LINE)
    line_bytes = disk_size(cal)
    print("      一行 %d 个字符，在盘上占 %d 字节%s" % (len(LINE), line_bytes,
          "（多出来的 %d 个字节是 Windows 文本模式把 \\n 换成 \\r\\n，不是写重了）" % (line_bytes - len(LINE))
          if line_bytes > len(LINE) else ""))
    print("      你选了 %d 行 = 期望 %d 字节" % (k, k * line_bytes))
    print("      缓冲区大小（io.DEFAULT_BUFFER_SIZE）= %d 字节 —— 『为什么写了却还没落盘』就是卡在它上面"
          % io.DEFAULT_BUFFER_SIZE)
    delete_attempt(cal)
    print()

    total = k * line_bytes

    print("[4/7] 反面：不用 with，写到第 %d 行炸（就在你这个窗口里跑）" % k)
    p_plain = os.path.join(d, "plain_inproc.txt")
    f = open(p_plain, "w", encoding="utf-8")
    for _ in range(k):
        f.write(LINE)
    try:
        raise RuntimeError("第 %d 行写完之后炸了" % k)
    except RuntimeError as exc:
        print("      炸在这一句：%s" % exc)
    while_size = disk_size(p_plain)
    pending = total - while_size
    print("      此刻磁盘上 %d 字节；你写了 %d 字节，差 %d 字节还压在缓冲区里（我没让它刷，它就没出去）"
          % (while_size, total, pending))
    locked, lock_note = delete_attempt(p_plain)
    print("      期望 句柄开着的时候删不掉   实测 %s -> %s"
          % (lock_note, "OK" if locked is False else "FAIL（它居然删掉了，Windows 这一版没锁？）"))
    print("      现在我自己 close() 一下 —— with 退出时替你做的正是这一下：")
    f.close()
    after_close = disk_size(p_plain)
    ok_close_full = after_close == total
    print("      期望 关句柄之后磁盘上 %d 字节（压着那 %d 字节全出来了）   实测 %d -> %s"
          % (total, pending, after_close, "OK" if ok_close_full else "FAIL"))
    ok_deleted, note_deleted = delete_attempt(p_plain)
    print("      期望 关了就能删   实测 %s -> %s" % (note_deleted, "OK" if ok_deleted else "FAIL"))
    print()

    print("[5/7] 正面：用 with，同一第 %d 行炸" % k)
    p_with = os.path.join(d, "with_inproc.txt")
    try:
        with open(p_with, "w", encoding="utf-8") as g:
            for _ in range(k):
                g.write(LINE)
            raise RuntimeError("第 %d 行写完之后炸了（这次外面包着 with）" % k)
    except RuntimeError as exc:
        print("      炸在这一句：%s" % exc)
    ok_closed = g.closed is True
    print("      期望 句柄已经关了 g.closed = True   实测 %s -> %s" % (g.closed, "OK" if ok_closed else "FAIL"))
    with_size = disk_size(p_with)
    ok_with_full = with_size == total
    print("      期望 磁盘上 %d 字节   实测 %d -> %s" % (total, with_size, "OK" if ok_with_full else "FAIL"))
    ok_del2, note_del2 = delete_attempt(p_with)
    print("      期望 删得掉   实测 %s -> %s" % (note_del2, "OK" if ok_del2 else "FAIL"))
    print("      这一节我没写过一句 close()，也没写过 flush() —— 它是 with 在退出那一刻替你做的，异常都没挡住它。")
    print()

    print("[6/7] 两组并排跑（每一种都是一个真会死 / 真活着的小进程，都写到第 %d 行、期望 %d 字节）" % (k, total))
    print("      写字组：这一组比的是『怎么死』，with 不是变量（抛异常时 Python 有机会替你把句柄收尾）")
    table = {}
    for tag, how, death, flush_flag, label, why in ROWS:
        p = os.path.join(d, "%s.txt" % tag)
        size, rc, err_name = run_child(p, k, how, death, flush_flag, py)
        table[tag] = size
        print("      %s  %-4s  %-22s 磁盘 %6d 字节  %s"
              % (tag, "硬死" if death == "exithard" else "抛异常", label, size, loss_note(total, size)))
        print("         退码 %d，它临终打的：%s；　%s" % (rc, err_name, why))
    print()
    ok_two_vote = while_size == table["R2"]
    print("      第一票对第二票：[4/7] 量到『句柄开着时磁盘上有 %d 字节』，R2（硬死，另一个进程）量到 %d 字节"
          % (while_size, table["R2"]))
    print("      期望 两个数一样（同一件事，一个在本窗口量、一个在死掉的进程里量）   实测 %s -> %s"
          % ("一样" if ok_two_vote else "不一样", "OK" if ok_two_vote else "FAIL"))
    ok_r1 = table["R1"] == total
    print("      期望 R1（不用 with，但抛异常）一个字没少：%s -> %s" % (loss_note(total, table["R1"]), "OK" if ok_r1 else "FAIL"))
    print("      这一条不是说不用 with 也行 —— 它靠的是 Python 死前有机会收尾，硬死就没这机会（看 R2）")
    ok_r2 = table["R2"] == while_size and (table["R2"] < total) == (pending > 0)
    print("      期望 R2 少的正是压着的那 %d 字节   实测 %s -> %s"
          % (pending, loss_note(total, table["R2"]), "OK" if ok_r2 else "FAIL"))
    ok_r3 = table["R3"] == total
    print("      期望 R3（with + 抛异常）一个字没少   实测 %s -> %s" % (loss_note(total, table["R3"]), "OK" if ok_r3 else "FAIL"))
    ok_r4 = table["R4"] == table["R2"]
    print("      期望 R4（with + 硬死）跟 R2 一样少 —— with 不是护身符，硬死时关句柄那一步压根没跑到   实测 %s -> %s"
          % ("和 R2 一样" if ok_r4 else loss_note(total, table["R4"]), "OK" if ok_r4 else "FAIL"))
    ok_r5 = table["R5"] == total
    print("      期望 R5（不用 with，但死前自己 flush()）一个字没少 —— 真正决定落盘的是 flush/close 这一下，with 只是保证它一定发生   实测 %s -> %s"
          % (loss_note(total, table["R5"]), "OK" if ok_r5 else "FAIL"))
    print()
    print("      松手组：这一组比的才是『有没有 with』—— 小进程写完还活着，我（另一个程序）去删它")
    p6 = os.path.join(d, "R6.txt")
    del6, note6, ready6, size6 = run_sleeper(p6, k, "plain", py)
    ok6 = (del6 is False) and ready6 and size6 == while_size
    print("      R6  攥着不放  不用 with      我去删：%s   它报到 READY=%s   删之前磁盘 %s 字节 -> %s"
          % (note6, ready6, size6, "OK" if ok6 else "FAIL"))
    p7 = os.path.join(d, "R7.txt")
    del7, note7, ready7, size7 = run_sleeper(p7, k, "with", py)
    ok7 = (del7 is True) and ready7 and size7 == total
    print("      R7  已经松手  用 with        我去删：%s   它报到 READY=%s   删之前磁盘 %s 字节 -> %s"
          % (note7, ready7, size7, "OK" if ok7 else "FAIL"))
    print("      这两臂同一台机器、同一个目录、同一个第 %d 行，唯一差别就是有没有 with：%s"
          % (k, "攥着那份我删不动，松手那份我删掉了" if (ok6 and ok7) else "跟期望不一样，见上面两行的 FAIL"))
    print("      R6 那份我删失败之后没再管它 —— 是它被 terminate（硬死）之后才松的手，第 7 节清点会把它删掉。")
    print()

    print("[7/7] 收尾清点")
    left = sorted(os.listdir(d))
    for name in left:
        delete_attempt(os.path.join(d, name))
    still = sorted(os.listdir(d)) if os.path.isdir(d) else []
    note_rmdir = rmdir_attempt(d)[1]
    ok_removed = (not still) and (not os.path.isdir(d))
    n = count_prefix(base)
    print("      这个目录里曾有 %d 个文件：%s" % (len(left), ", ".join(left)))
    print("      期望 文件删光、目录也没了、临时目录里以 %s 开头的东西归 0   实测 剩 %d 个 / 目录在 %s（%s）/ 同名前缀 %d 个 -> %s"
          % (PREFIX, len(still), os.path.isdir(d), note_rmdir, n, "OK" if (ok_removed and n == 0) else "FAIL"))
    after = stat2(target)
    ok_streak = before == after
    print("      打卡本：跑前 %s 字节 / %s → 跑后 %s 字节 / %s -> %s"
          % (before[0], before[1], after[0], after[1], "OK" if ok_streak else "FAIL（说好不动它）"))
    print()
    print(BAR)

    checks = [ok_close_full, locked is False, ok_deleted, ok_with_full, ok_closed, ok_del2,
              ok_two_vote, ok_r1, ok_r2, ok_r3, ok_r4, ok_r5, ok6, ok7,
              ok_removed, n == 0, ok_streak]
    if all(checks):
        print("结论：[OK] 七行各得其所（写字组 5 行比怎么死，松手组 2 行比有没有 with）。一句话记法：")
        print("      with 干的就是『无论如何替你 close() 一下』，")
        print("      而 close() 是两件事 —— 把缓冲区里那 %d 字节推上盘 + 松手让 Windows 允许你删它。" % pending)
        if while_size == 0:
            print("      你这一档：R2 是 0 字节（一个字都没来得及出去）。再双击一次把数字换大（奔着 %d 那一头），"
                  "缓冲区满过一轮之后，先出去的那半截会留在盘上，R2 就不再是 0。" % MAX_K)
        else:
            print("      你这一档：磁盘上已经留了 %d 字节（缓冲区满过一轮，先出去的那半截收不回来了），"
                  "还压着 %d 字节。把数字换回 1 再跑，看 R2 变成 0。" % (while_size, pending))
        return 0
    print("结论：[FAIL] 有几格跟期望不一样。把上面带 FAIL 的那几行原样贴回来（都是临时文件名，没有秘密）。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
