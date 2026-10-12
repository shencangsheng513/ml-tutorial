"""第二件 · 学习打卡本（真的会存盘）。

你只做一件事：双击跑起来，在窗口里输入今天学了多少分钟，回车。
这一次屏幕上的数字是你自己的记录，不是我造的样本。

有一条硬规矩写在代码里：记录本必须落在 git 仓库外面。这个仓库的远端是公开的，
学习记录落进去等于把它发出去。所以这一格不是"建议"，是默认拒绝。
"""

import datetime
import hashlib
import json
import os
import sys

APP_DIR_NAME = "agent-lesson-streak"
FILE_NAME = "streak.json"
MIN_M, MAX_M = 1, 600

BAR = "=" * 60


def data_path():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, APP_DIR_NAME, FILE_NAME)


def repo_root_of(path):
    """这个路径往上走，第一个带 .git 的目录就是仓库根；没有就回 None。"""
    cur = os.path.dirname(os.path.abspath(path))
    while True:
        marker = os.path.join(cur, ".git")
        if os.path.isdir(marker) or os.path.isfile(marker):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def sha12(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12], None
    except Exception as exc:
        return None, type(exc).__name__


def load_days(path):
    """返回 (days 字典 或 None, 状态, 说明)。状态：missing / ok / corrupt。"""
    if not os.path.isfile(path):
        return None, "missing", "这个文件还不存在（第一次跑会建）"
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        return None, "corrupt", "%s：%s" % (type(exc).__name__, str(exc)[:60])
    if not isinstance(obj, dict) or not isinstance(obj.get("days"), dict):
        return None, "corrupt", "里面没有 days 这一格（形状不认识）"
    days = {}
    for k, v in obj["days"].items():
        if not isinstance(v, int) or isinstance(v, bool):
            return None, "corrupt", "days 里有个值不是整数（%s）" % k
        days[k] = v
    return days, "ok", "读到 %d 天的记录" % len(days)


def ask_minutes():
    """返回 (分钟数或 None, 档位, 原样输入)。五档各打各的话。"""
    sys.stdout.write("今天学了多少分钟？（%d..%d，直接回车=今天不卡）：" % (MIN_M, MAX_M))
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
    if v < MIN_M or v > MAX_M:
        return min(max(v, MIN_M), MAX_M), "clamp", raw
    return v, "typed", raw


def mode_sentence(mode, minutes, raw):
    if mode == "typed":
        return "这一档：你输入了 %d，今天按 %d 分钟记。" % (minutes, minutes)
    if mode == "enter":
        return "这一档：你只按了回车，等于今天不卡 —— 不会写文件，盘上那个记录本一个字节都不会变。"
    if mode == "bad":
        return "这一档：「%s」不是整数，所以今天不卡 —— 也不会写文件。" % raw.strip()[:20]
    if mode == "clamp":
        return "这一档：你输入的数字超出我定的区间，夹到 %d —— 区间是我验过的，不是我猜的。" % minutes
    return "这一档：键盘这一档不存在（没有 stdin）—— 不写文件。这一档存在只为了让你看到「不写盘」和「写盘」打的是不同的行。"


def consecutive_days(days, today):
    """从今天往回数连续有记录的天数。今天没记录 ⇒ 0（屏幕上会另说一句"昨天那串还剩几天"）。"""
    n = 0
    cur = today
    while True:
        if days.get(cur.isoformat(), 0) <= 0:
            break
        n += 1
        cur = cur - datetime.timedelta(days=1)
    return n


def trailing_days(days, yesterday):
    n = 0
    cur = yesterday
    while True:
        if days.get(cur.isoformat(), 0) <= 0:
            break
        n += 1
        cur = cur - datetime.timedelta(days=1)
    return n


def first_gap(days, today, lookback=40):
    """今天往前第一个"没有记录"的日子；都满就回 None。"""
    for i in range(0, lookback):
        d = (today - datetime.timedelta(days=i)).isoformat()
        if days.get(d, 0) <= 0:
            return d
    return None


def save_days(path, days):
    obj = {"version": 1, "app": APP_DIR_NAME, "days": days,
           "updated_at": datetime.datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)
    return obj


def main():
    today = datetime.date.today()
    path = data_path()

    print(BAR)
    print("第二件 · 学习打卡本（会存盘）　运行日期：%s" % today.isoformat())
    print(BAR)
    print()

    print("[1/5] 记录本在哪")
    inside = repo_root_of(path)
    exists = os.path.isfile(path)
    if exists:
        size = os.path.getsize(path)
        s12, err = sha12(path)
        print("      路径：%s" % path)
        print("      已存在：%d 字节  sha 前 12：%s" % (size, s12 if err is None else "读失败(" + err + ")"))
    else:
        size, s12 = 0, None
        print("      路径：%s" % path)
        print("      还没有这个文件（第一次打卡会建它）")
    if inside is not None:
        print("      [STOP] 这个路径在 git 仓库里面：%s" % inside)
        print("             仓库远端是公开的，学习记录不能落在这儿。今天不写盘。")
        print("             要改位置就设环境变量 LOCALAPPDATA（正常不用管，默认就在仓库外）")
        return 2
    print("      在仓库外：%s（这条是程序自己往上走逐层查 .git 查出来的，不是写死的）" % (inside is None))
    print("      这个文件只在你自己这台机器上：程序不联网、不上传，屏幕上这些字不会离开你的电脑。")
    print()

    days, state, note = load_days(path)
    print("[2/5] 读回来的历史")
    if state == "corrupt":
        print("      [STOP] 记录本读不出可信内容：%s" % note)
        print("             路径：%s" % path)
        print("             我不重建、不覆盖 —— 免得把可能还在的内容当成空本子写掉。你把它贴给我看一眼。")
        return 2
    print("      %s" % note)
    if state == "ok":
        recent = sorted(days.keys())[-5:]
        for d in recent:
            print("        %s  %d 分钟" % (d, days[d]))
    print()

    minutes, mode, raw = ask_minutes()
    print("[3/5] 你这次输入")
    print("      %s" % mode_sentence(mode, minutes, raw))
    print()

    if mode in ("enter", "bad", "no-tty"):
        base = dict(days) if state == "ok" else {}
        shown = consecutive_days(base, today)
        print("[4/5] 连续天数怎么算（今天没卡，所以只报现状）")
        print("      今天 %s 有记录：%s" % (today.isoformat(), today.isoformat() in base))
        y = (today - datetime.timedelta(days=1)).isoformat()
        print("      昨天 %s 有记录：%s" % (y, y in base))
        print("      昨天那一串还有 %d 天（今天不卡的话，明天不卡就断在这儿）" % trailing_days(base, today - datetime.timedelta(days=1)))
        print("      连续天数（按「今天必须有记录」这个口径）= %d" % shown)
        print()
        print("[5/5] 落盘")
        print("      这一档不写文件。盘上那个记录本没动过：字节数 %d → %d，sha 前 12 %s → %s"
              % (size, (os.path.getsize(path) if exists else 0), s12, (sha12(path)[0] if exists else None)))
        print()
        print(BAR)
        print("结论：今天没记，什么都没写。想记就重跑一次，输入一个数字。")
        return 0

    base = dict(days) if state == "ok" else {}
    key = today.isoformat()
    had = key in base
    before_val = base.get(key, 0)
    after_val = before_val + minutes
    base[key] = after_val

    print("[4/5] 连续天数怎么算")
    print("      今天 %s 原来有记录：%s（%d 分钟）" % (key, had, before_val))
    print("      今天合计 = 原来的 %d + 这次的 %d = %d 分钟" % (before_val, minutes, after_val))
    want_streak = consecutive_days(base, today)
    print("      从今天往前逐日查，连续有记录的天数 = %d（口径：今天必须有记录才从 1 起算）" % want_streak)
    gap = first_gap(base, today)
    print("      最近一次断档：%s" % (gap if gap else "近 40 天里没断过"))
    print()

    obj = save_days(path, base)
    back, back_state, back_note = load_days(path)
    print("[5/5] 落盘复核（写完立刻读回来比，不靠我说「应该写进去了」）")
    ok_same = back_state == "ok" and back.get(key) == after_val and len(back) == len(base)
    print("      期望 今天那格 = %d   实测 %s -> %s"
          % (after_val, back.get(key) if isinstance(back, dict) else "读不到", "OK" if ok_same else "FAIL"))
    print("      期望 天数 %d 格   实测 %d 格 -> %s"
          % (len(base), len(back) if isinstance(back, dict) else -1,
             "OK" if isinstance(back, dict) and len(back) == len(base) else "FAIL"))
    new_size = os.path.getsize(path)
    new_s12, _ = sha12(path)
    print("      文件：%s" % path)
    print("      字节数 %d → %d   sha 前 12 %s → %s" % (size, new_size, s12, new_s12))
    print()
    print(BAR)
    if ok_same:
        print("结论：[OK] 这一天的记录已经在你盘上了。连续 %d 天。" % want_streak)
    else:
        print("结论：[FAIL] 写进去的和读回来的不一样，把上面 [5/5] 那几行原样贴回来（路径里没有秘密，可以贴）。")
    print("明天再双击一次，输入一个数字就行。")
    return 0 if ok_same else 1


if __name__ == "__main__":
    sys.exit(main())
