# -*- coding: utf-8 -*-
"""诊断卷的开工把手。

它不改你的答案文件，只负责两件事：把"该用哪条命令"从记忆里拿掉，
以及把那一题的自检原样跑出来。所以你可以整晚开着它，不会碰坏卷面。
"""
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = r"D:/ML_Projects/ml-tutorial/.venv/Scripts/python.exe"
CONDA_PY = r"C:/Users/shenh/miniconda3/python.exe"

# q2 要 pandas，教程 venv 里没装，miniconda 里有（step0_diagnostic/README.md 那条例外）。
QUESTIONS = [
    ("q1", "q1_basics.py", VENV_PY, "函数/字符串/dict/默认参数陷阱", "10 分钟"),
    ("q2", "q2_data.py", CONDA_PY, "pandas 清洗脏行情 + 分组收益", "10 分钟"),
    ("q3", "q3_exceptions.py", VENV_PY, "窄异常分类（禁用 except Exception）", "10 分钟"),
    ("q4", "q4_oop.py", VENV_PY, "过程式持仓改 dataclass + 类", "10 分钟"),
]
SLOW = {"q2": "这一发要现起 pandas：本机实测三连发 341 / 358 / 484 毫秒，屏上停半秒是正常的，不是卡死。"}


def plan():
    print("阶段 0 诊断卷 · 四道题，总限时 45 分钟")
    print()
    print("  编号  文件                    内容                                  限时")
    for key, fname, _py, desc, mins in QUESTIONS:
        print("  %-4s  %-22s  %-34s  %s" % (key, fname, desc, mins))
    print("  q5    q5_oral.md              5 道口答（把回答打字发给导师即可）      5~10 分钟")
    print()
    print("纪律（这一条比分数重要）：")
    print("  1. 禁止让 AI 代写任何一题的答案。这张卷子只测水位，代写会把你的路线图引偏。")
    print("  2. 单题超时就空着往下走——空着本身就是诊断信号。")
    print("  3. 不要改测试去迁就实现。测试错了就是你的实现错了。")
    print()
    print("怎么用：")
    print("  在下面的提示符输入 q1 回车 = 跑第一题的自检（不做任何保存动作，也不改你的文件）。")
    print("  改答案请直接用编辑器打开上面那个文件，只动 TODO 区；改完回到本窗口再输一次题号看 [OK]/[FAIL] 变没变。")
    print("  想结束就输入 q 回车，或者直接关掉这个窗口。")
    print()
    print("全部做完后，对导师说一句「诊断做完」。")
    return 0


def run_one(key, fname, py):
    target = os.path.join(HERE, fname)
    if not os.path.exists(target):
        print("[FAIL] 题目文件不在：%s" % target)
        return 1
    if not os.path.exists(py):
        print("[FAIL] 解释器不在：%s" % py)
        print("       这一题没法跑。别去改题目，先把这个解释器补回来（告诉导师这一行）。")
        return 1
    if key in SLOW:
        print(SLOW[key])
    argv = [py, target]
    print("$ " + " ".join('"%s"' % a for a in argv))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(argv, env=env)
    print()
    print("（上面那段是 %s 自己打的判分，最后一行是 %s: 通过数/总数）" % (fname, key.upper()))
    return r.returncode


def oral():
    p = os.path.join(HERE, "q5_oral.md")
    if not os.path.exists(p):
        print("[FAIL] 口答题不在：%s" % p)
        return 1
    print(io.open(p, encoding="utf-8").read())
    print("（答完直接把文字发给导师，不必写进文件。）")
    return 0


def dispatch(key):
    key = key.strip().lower()
    if key in ("q5", "5", "oral"):
        return oral()
    for k, fname, py, _d, _m in QUESTIONS:
        if key == k:
            return run_one(k, fname, py)
    print("[FAIL] 不认识的题号：%r（可输入 q1 q2 q3 q4 q5，或直接关窗口）" % key)
    return 2


def interact():
    """plan() 说了"在本窗口输入 q1 回车"，这一支就是那句话的实现。"""
    while True:
        try:
            line = input("\n要跑哪一题（q1..q5，回车重看说明，q 结束）> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        key = line.strip().lower()
        if not key:
            plan()
        elif key in ("q", "quit", "exit"):
            return 0
        else:
            dispatch(key)


def main(argv):
    if len(argv) < 2:
        plan()
        return interact()
    return dispatch(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
