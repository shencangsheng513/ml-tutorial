"""第一课 · 只改不写。

这一版程序我已经写完。你只做一件事：双击跑起来，在窗口里**输入一个数字**，回车。
然后看屏幕上哪几行变了、哪几行没变。全程只用标准库，不写文件、不连网。

只讲一格：**函数的默认值会在多次调用之间被共享**（`def f(x, acc=[])` 那个坑）。
载体是"学习打卡本"——同一份数据用反面写法和正面写法各跑一遍，差异直接打在屏幕上。
"""

import sys

# 这一行是"不按键盘时"的默认值。想改可以直接改这里，也可以双击之后在窗口里输入数字。
REPEAT = 2

MIN_R, MAX_R = 1, 12

# 内联样本数据（假的，为了每次跑结果都一样）。这不是你的真实学习记录。
SAMPLE_DAYS = [
    ("周一", 25),
    ("周二", 40),
    ("周三", 15),
]

BAR = "=" * 60


def add_day_shared(entry, bucket=[]):
    """反面写法：默认值 `bucket=[]` 只在 def 那一行被执行时创建一次，之后每次调用拿到同一个列表。"""
    bucket.append(entry)
    return bucket


def add_day_fresh(entry, bucket=None):
    """正面写法：默认值用 None 当哨兵，函数体里当场新建列表。"""
    if bucket is None:
        bucket = []
    bucket.append(entry)
    return bucket


def collect_shared(days, repeat):
    """调用处一个参数都不多写——坑就藏在这一行：没人往里传列表。"""
    per_day = {}
    for day, minutes in days:
        last = []
        for _ in range(repeat):
            last = add_day_shared((day, minutes))
        per_day[day] = last
    return per_day


def collect_fresh(days, repeat):
    """每天自己准备一个空格子，当天反复往里加。"""
    per_day = {}
    for day, minutes in days:
        bucket = None
        for _ in range(repeat):
            bucket = add_day_fresh((day, minutes), bucket)
        per_day[day] = bucket
    return per_day


def head(row):
    if not row:
        return "（空）"
    return "，".join("%s %s 分钟" % (d, m) for d, m in row[:2])


def ask_repeat(default):
    """读一个数字。返回 (用的值, 这一趟属于哪一档)。每一档的措辞互不相同，不许哑。

    typed   = 他真输了区间内的数
    enter   = 他只按了回车（等于没改）
    bad     = 输入的不是整数
    clamp   = 是整数但不在 1..12，按 MAX_R 跑
    no-tty  = 键盘这一档根本不存在（比如被管道喂住），按默认跑
    """
    sys.stdout.write("请输入一个数字（%d..%d），直接按回车就用默认的 %d：" % (MIN_R, MAX_R, default))
    sys.stdout.flush()
    try:
        raw = sys.stdin.readline()
    except Exception:
        return default, "no-tty", ""
    if raw == "":
        return default, "no-tty", ""
    text = raw.strip()
    if text == "":
        return default, "enter", text
    try:
        value = int(text)
    except ValueError:
        return default, "bad", text
    if not MIN_R <= value <= MAX_R:
        return MAX_R, "clamp", text
    return value, "typed", text


def section_sample(days):
    print("[1/4] 样本数据（内联假数据，不是你的真实学习记录）")
    total = sum(m for _, m in days)
    print("      " + " / ".join("%s %d 分钟" % (d, m) for d, m in days) + " —— 合计 %d 分钟" % total)
    print("      这一节跟你输入的数字没关系：换任何一个数，这三行都不该动。")
    print()


def section_shared(days, repeat):
    per_day = collect_shared(days, repeat)
    lens = [len(per_day[d]) for d, _ in days]
    same = per_day[days[0][0]] is per_day[days[-1][0]]
    print("[2/4] 反面写法  def add_day(entry, bucket=[])")
    print("      每天打卡 %d 遍，跑完看每天那一格里装了什么：" % repeat)
    for day, _ in days:
        print("        %s那格 = %d 条   头两条 = %s" % (day, len(per_day[day]), head(per_day[day])))
    print("      [FLAG] 三格其实是同一个对象（is=%s）：%s那格里混进了别天的记录" % (same, days[0][0]))
    print("      后果：算『最活跃那天有几条』会得到 %d，真相是 %d —— 虚高 %d 倍（= 天数）"
          % (max(lens), repeat, len(days)))
    print()
    return max(lens)


def section_fresh(days, repeat):
    per_day = collect_fresh(days, repeat)
    lens = [len(per_day[d]) for d, _ in days]
    same = per_day[days[0][0]] is per_day[days[-1][0]]
    print("[3/4] 正面写法  def add_day(entry, bucket=None):  if bucket is None: bucket = []")
    print("      同一份数据再跑一遍：")
    for day, _ in days:
        print("        %s那格 = %d 条   头两条 = %s" % (day, len(per_day[day]), head(per_day[day])))
    print("      [OK] 三格互不相同（is=%s）：每格只装当天" % same)
    print("      为什么：默认值只在 def 那一行被执行时求值一次；None 是每次调用当场新建一个列表。")
    print()
    return max(lens)


def section_check(days, repeat, shared_len, fresh_len, mode, typed):
    want_shared = len(days) * repeat
    want_fresh = repeat
    ok_shared = shared_len == want_shared
    ok_fresh = fresh_len == want_fresh
    print("[4/4] 程序自己核对实测与公式（不靠你猜有没有跑对）")
    print("      反面那格 期望 %d（= %d 天 × 数字）  实测 %d -> %s"
          % (want_shared, len(days), shared_len, "OK" if ok_shared else "FAIL"))
    print("      正面那格 期望 %d（= 那个数字）        实测 %d -> %s"
          % (want_fresh, fresh_len, "OK" if ok_fresh else "FAIL"))
    print()
    if mode == "typed":
        print("      这一档：你输入了 %s，程序按 %s 跑的。" % (typed, repeat))
        print("      再输入一个别的数（比如 1 或 %d），上面两行的『实测』会跟着变，"
              % MAX_R)
        print("      但 [1/4] 那三行永远不动 —— 那才是这一格要你看的东西。")
    elif mode == "enter":
        print("      [note] 这一档：你只按了回车，等于没改，所以用的还是默认的 %d。" % repeat)
        print("      [note] 想看到那一格动：再跑一次，这次输入 4。")
    elif mode == "bad":
        print("      [note] 这一档：你输入的是 %r，不是整数，所以退回默认的 %d。" % (typed, repeat))
        print("      [note] 只打数字，别打汉字和标点。")
    elif mode == "clamp":
        print("      [note] 这一档：你输入 %s，超出我验过的区间 %d..%d，这趟按 %d 跑。"
              % (typed, MIN_R, MAX_R, repeat))
    elif mode == "no-tty":
        print("      [note] 这一档：键盘这一档不存在（窗口是被程序喂住的），所以按默认的 %d 跑。" % repeat)
    else:
        print("      [FAIL] 输入档位不认识：%r —— 这是程序的洞，把这行原样贴回来。" % mode)
        ok_shared = ok_fresh = False
    return ok_shared and ok_fresh


def main():
    print(BAR)
    print("第一课 · 可变默认参数 —— 载体：学习打卡本（你只输入一个数字）")
    print(BAR)
    if not isinstance(REPEAT, int) or isinstance(REPEAT, bool):
        print("[STOP] 文件里那行 REPEAT 必须是整数，别带引号、别写小数。现在它是 %r（类型 %s）"
              % (REPEAT, type(REPEAT).__name__))
        return 2
    if not MIN_R <= REPEAT <= MAX_R:
        print("[STOP] 文件里那行 REPEAT 我只在 %d..%d 验过输出，现在写的是 %d。"
              % (MIN_R, MAX_R, REPEAT))
        return 2
    repeat, mode, typed = ask_repeat(REPEAT)
    print()
    print("      本次用的数字 = %d（来源：%s）" % (repeat, mode))
    print()
    section_sample(SAMPLE_DAYS)
    shared_len = section_shared(SAMPLE_DAYS, repeat)
    fresh_len = section_fresh(SAMPLE_DAYS, repeat)
    ok = section_check(SAMPLE_DAYS, repeat, shared_len, fresh_len, mode, typed)
    print()
    print(BAR)
    if ok:
        print("结论：[OK] 实测和公式对得上。这只说明这一格讲完了，不代表要你会写。")
    else:
        print("结论：[FAIL] 有一格对不上，回到 [4/4] 看哪一行写着 FAIL。")
    print("下一格还是这个形式：我写整段，你换一个数字，看输出。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
