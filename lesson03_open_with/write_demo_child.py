"""演示用的"写几行就死 / 写完不松手"的小程序。你不双击它，是 open_with.py 用子进程反复调用它。

为什么要有这么一个小文件：有些死法（进程被硬杀）只能在一个真的会死的进程里演，
在当前这个窗口里是演不出来的 —— 它一死，屏幕上的字也没人打了。
而"别的程序占着这个文件"那一档，要有一个还活着的别人 —— 就是它。

用法（由 open_with.py 传参，你不用管）：
    python write_demo_child.py <写到哪个文件> <写几行> <with|plain> <raise|exithard|sleep> [flush]
"""

import os
import sys
import time

LINE = "x" * 800 + "\n"


def die(death):
    """两种死法：raise = 常见死法（Python 有机会收尾）；exithard = 硬死（一点机会都不给）。"""
    if death == "raise":
        raise RuntimeError("这一行是故意炸的")
    os._exit(9)


def write_and_die(path, k, how, death, do_flush):
    """plain 那一支全程不 close；with 那一支由 with 在块退出时 close。
    两种死法（raise / exithard）都必须在块里发生 —— 挪到块外面，with 就没差别了。"""
    if how == "with":
        with open(path, "w", encoding="utf-8") as f:
            for _ in range(k):
                f.write(LINE)
            if do_flush:
                f.flush()
            if death != "sleep":
                die(death)
        # 只有 sleep 那一档走得到这里：块已经正常退出，句柄由 with 关掉了，而进程还活着
        notify_and_sleep()
    else:
        f = open(path, "w", encoding="utf-8")
        for _ in range(k):
            f.write(LINE)
        if do_flush:
            f.flush()
        if death == "sleep":
            notify_and_sleep()
            return
        die(death)


def notify_and_sleep():
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    time.sleep(30)


def main():
    if len(sys.argv) < 5:
        sys.stderr.write("need args: path lines with|plain raise|exithard|sleep [flush]\n")
        return 2
    write_and_die(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4],
                  len(sys.argv) > 5 and sys.argv[5] == "flush")
    return 0


if __name__ == "__main__":
    sys.exit(main())
