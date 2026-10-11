# -*- coding: utf-8 -*-
# 跑批驱动（第四十二发 A·还第四十一发新欠⑻）。
#
# 为什么需要它：前三发的"整轮重跑"是**人敲 38 次命令**，回执靠我把每份日志的末几行抄进守卫头部。
# 那一套有两个洞：
#   ① "stderr 各 0 字节"这句话没有自带是谁量的。第卅九发之后实测过：本模块的 docstring 里有非法转义，
#      3.12 起会发 SyntaxWarning 打到 stderr，3.10/3.11 一声不吭 ⇒ 同一句"0 字节"在两版解释器上是
#      两个相反的事实，而旧流程里这句话会被下一发原样抄下去。
#   ② 只看退码等于只看"跑没跑崩"。第卅九发补跑链实测过：裸档名被静默丢弃后跑成正常验红、退码仍 0。
# 这个驱动把两件事都变成断言：每份日志必须**自带口径行**，且全批的口径行逐字相同；
# 每份的『模式：』行必须等于传进去的那个串。断言不住在散文里，住在 `--selftest` 的合成样本上。
#
# 用法（本目录）：
#   PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -u _red_batch.py --selftest   # 零守卫运行，只验这台驱动有牙
#   PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -u _red_batch.py --jobs 2     # 整轮：normal + MODES 现读到的每一档
#   PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python -u _red_batch.py --only --drift --floorcut   # 复跑个别档
# 日志与回执落在 --out 指的那个目录（默认 `_red_batch_logs`，可再生产物，不进册）。
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GUARD = "_red_v3.py"

# 三行读数全靠现读，不写死档名也不写死档数：
# ① 档名从守卫的 MODES 登记处读（写死就等于"加一档而批不跑它"这种恒绿装饰又活了）；
# ② 口径行与模式行从每份日志自己读。
_MODES_BLOCK = re.compile(r"^MODES = \((.*?)\)\s*\n", re.S | re.M)
_QUOTED = re.compile(r'"([^"]+)"')
_CALIBER = re.compile(r"^口径：(.+)$", re.M)
_MODE_PRINT = re.compile(r"^模式：(.+)$", re.M)
_RED_COUNT = re.compile(r">>> 验红 (\d+) 条红")
_NORMAL_PRINT = "（正常验红）"      # 守卫把 normal 当"没有档名"打印，这一串由它自己那行决定


def read_modes(src_text):
    """从守卫源码现读档名名单。读不到必须**出声退出**，不许回落成空批：
    空批的退码是 0，也就是说"登记处被改写所以读不到"会跑成"这一轮所有档都通过了"。"""
    m = _MODES_BLOCK.search(src_text)
    if not m:
        raise SystemExit("读不到 MODES 登记处（形状不是「MODES = ( ... )」）⇒ 是尺子瞎了，不是没有档")
    modes = _QUOTED.findall(m.group(1))
    if not modes:
        raise SystemExit("MODES 登记处读到了空名单 ⇒ 同上，不许当成『本轮无拆红档』")
    return modes


def expected_mode_print(mode):
    return _NORMAL_PRINT if mode == "normal" else mode


def check_one(mode, rc, out, err_bytes):
    """一份日志的各条腿（腿号由红条自己带，数目不写在话里），返回 `([(腿号, 话), ...], 红数, 口径行或 None)`。
    纯函数：`--selftest` 喂合成样本，真批喂 subprocess 的产物，两支共用一把口径。"""
    reds = []
    caliber = _CALIBER.findall(out)
    mp = _MODE_PRINT.findall(out)

    if rc not in (0, 1):
        reds.append(("腿1", "退码 " + str(rc) + " 不在 {0,1} ⇒ 这份根本没跑到判决（崩、或被 argparse 拦在门外）"))
    if len(caliber) != 1:
        reds.append(("腿4", "『口径：』行有 " + str(len(caliber)) + " 行（应为 1 行）⇒ 这份没带解释器口径"
                     "，它的『stderr 若干字节』跨版本不可比"))
    if len(mp) != 1:
        reds.append(("腿2", "『模式：』行有 " + str(len(mp)) + " 行（应为 1 行）⇒ 末段没跑到，"
                     "后面的红数与结论都不许引"))
    # 只剥**尾部**：守卫是子进程，stdout 写进管道时行尾带 CRLF，捕获串里那一行末尾多一个 \r——
    # 那是行信封不是内容。真错的两型各有自己的格子点名：整行缺失（上面那条）、写的是另一个档名
    # （下面那条，剥尾巴不会把 "--drift" 剥成 "--floorcut"，所以它照样红）。
    elif mp[0].rstrip() != expected_mode_print(mode):
        reds.append(("腿2", "传入 " + repr(mode) + "，日志那行写的是 " + repr(mp[0])
                     + " ⇒ 参数被静默丢弃或派发没到位"
                     "（第四十发新欠⑻那一型：退码 0 而跑的其实是别的那一档；两串看着同值时读 repr）"))
    if len(err_bytes):
        reds.append(("腿3", "stderr " + str(len(err_bytes)) + " 字节（应为 0）："
                     + err_bytes.decode("utf-8", "replace").strip()[:120]))

    n = None
    if rc == 1:
        c = _RED_COUNT.findall(out)
        if len(c) != 1:
            reds.append(("腿6", "退码 1 但『>>> 验红 N 条红』有 " + str(len(c)) + " 行 ⇒ 它不是被判据红出去的"
                         "（SystemExit 的报错串也算退码 1，那种红数不可引）"))
        elif int(c[0]) == 0:
            reds.append(("腿6", "退码 1 而红数写 0 ⇒ 互证不成立"))
        else:
            n = int(c[0])
    else:
        c = _RED_COUNT.findall(out)
        if c:
            reds.append(("腿6", "退码 0 却印了红数 " + c[0] + " ⇒ 有红却没出去"))
        n = 0
    return reds, n, (caliber[0] if caliber else None)


def check_batch(rows):
    """批级那条腿（腿5）：全批的口径行必须逐字相同。
    为什么单份合格还不够：把三份日志的红数抄进同一行头部文字，就等于默认它们是同一把尺子量的；
    这一票比的正件事——若有人中途换了 python 或加了 -W，这里先红，红在"抄表"之前。"""
    seen = {}
    for r in rows:
        if r["caliber"] is not None:
            seen.setdefault(r["caliber"], []).append(r["mode"])
    if len(seen) > 1:
        return [("腿5", "全批口径行有 " + str(len(seen)) + " 种不同的读法："
                 + " ｜ ".join(c[:60] + "×" + str(len(v)) for c, v in sorted(seen.items()))
                 + " ⇒ 这些红数不在同一把尺子上，逐档红数那一行不许照抄")]
    if not seen:
        return [("腿5", "没有任何一份带口径行 ⇒ 批级对账无从开口（不是通过）")]
    return []


def _bare(mode):
    """档名的"裸形"：只剥**存在的那两个横杠**。
    第四十二发那笔事故就是把这一步写成了 `mode[2:]`（无条件剥两个字符）⇒ 回执上 `normal` 印成了 `rmal`，
    与守卫自己那档 `--namecut`（名字被截）打的同一族红，只是发生在我自己的跑批件里。"""
    return mode[2:] if mode.startswith("--") else mode


def tally_line(rows, modes):
    """派生『逐档红数现读』那一行，并自带两个分母（段数、档名对不上登记处的段数）。
    返回 (那一行的正文, 红条列表)。红条由 main 并进批级红，所以"段数少了"或"档名被截"都会红在抄表之前。"""
    tally = "｜".join(_bare(r["mode"]) + " " + str(r["red_count"]) for r in rows)
    toks = tally.split("｜")
    registry = set(_bare(m) for m in modes)
    unknown = [t for t in toks if t.split(" ")[0] not in registry]
    reds = []
    if len(toks) != len(rows):
        reds.append(("腿7", "逐档红数那行有 " + str(len(toks)) + " 段，而批次有 " + str(len(rows))
                     + " 份 ⇒ 段数与份数不等，这一行不能当逐档红数的抄表来源"))
    if unknown:
        reds.append(("腿7", "逐档红数那行有 " + str(len(unknown)) + " 段的档名不在登记处（裸形对不上）："
                     + "、".join(unknown[:6]) + " ⇒ 档名被改写或被截短（`--namecut` 那一型住在我自己这一行里）"))
    header = ("逐档红数现读（" + str(len(toks)) + " 段 == " + str(len(rows)) + " 份；"
              "档名对不上登记处 " + str(len(unknown)) + " 段）：")
    return header + tally, reds


def run_one(mode, out_dir):
    t0 = time.time()
    res = subprocess.run([sys.executable, "-u", GUARD, mode], cwd=HERE,
                         capture_output=True)
    dt = time.time() - t0
    out = res.stdout.decode("utf-8", "replace")
    err = res.stderr
    with open(os.path.join(out_dir, mode + ".txt"), "w", encoding="utf-8", newline="") as f:
        f.write(out)
    with open(os.path.join(out_dir, mode + ".err"), "wb") as f:
        f.write(err)
    reds, n, caliber = check_one(mode, res.returncode, out, err)
    return {"mode": mode, "rc": res.returncode, "reds": reds, "red_count": n,
            "caliber": caliber, "out_bytes": len(out.encode("utf-8")),
            "err_bytes": len(err), "secs": round(dt, 1)}


def selftest():
    """各条腿各自都要有红票，另配一票证它不过度点名；外加一条"整批口径不一致"的批级票。
    票数不在这里写死（由下面那行结论现读），因为**合成样本的形状本身就是被测对象**：这一支此前的
    样本全是 LF，而真批的日志来自子进程管道、行尾是 CRLF ⇒ 腿2 在真批上假红过两次（第四十二发），
    LF 的夹具永远走不到那一型。现在 腿2 有四票：裸档名／整行缺失／CRLF 同值算绿／CRLF 而档名真错算红。
    为什么要有这一支：守卫的红要在真档案上才算数，而这个驱动的红只需要**它自己能被骗到**——
    把合成样本喂进同一个 `check_one`，就是问"注入这种日志时它点名了吗"。全绿说明它没牙。
    """
    ok_out = ("口径：" + "python=3.13.13｜-W=（无）｜SyntaxWarning 动作=default\n\n"
              + "模式：" + _NORMAL_PRINT + "\n\n结论：ok\n")
    cases = []
    leg_votes = {}

    def note(want_legs):
        if want_legs:
            for lg in want_legs:
                leg_votes[lg] = leg_votes.get(lg, 0) + 1
        else:
            leg_votes["（绿票不点名）"] = leg_votes.get("（绿票不点名）", 0) + 1

    def add(name, mode, rc, out, err, want_legs, batch=None):
        reds, n, cal = check_one(mode, rc, out, err_bytes=err)
        got = sorted({leg for leg, _ in reds})
        note(want_legs)
        cases.append((name, got == sorted(want_legs), "点名 " + (",".join(got) or "无")
                      + "｜期望 " + (",".join(sorted(want_legs)) or "无")
                      + "｜红数 " + str(n)))
        if batch is not None:
            rows = [{"mode": "a", "caliber": cal}, {"mode": "b", "caliber": batch}]
            got2 = sorted({leg for leg, _ in check_batch(rows)})
            cases.append((name + "（批级腿5）", got2 == ["腿5"],
                          "点名 " + (",".join(got2) or "无") + "｜期望 腿5"))

    add("干净样本", "normal", 0, ok_out, b"", [])
    add("腿1 退码 2", "normal", 2, ok_out, b"", ["腿1"])
    add("腿2 裸档名被丢弃", "--drift", 0, ok_out.replace(_NORMAL_PRINT, _NORMAL_PRINT), b"", ["腿2"])
    add("腿2 模式行缺失", "normal", 0, ok_out.replace("模式：" + _NORMAL_PRINT + "\n", ""), b"", ["腿2"])
    # 第四十二发真批的两条 腿2 是**假红**：守卫是子进程，stdout 写进管道时行尾带 CRLF ⇒ 捕获串那一行
    # 末尾多一个 \r，而旧比较不剥行尾。合成样本此前全是 LF ⇒ 自测永远走不到这一型。
    # 补两条而不是一条：只补"CRLF 算绿"会把 腿2 剥哑，所以另有一票钉住"CRLF 而档名真错"照样出声。
    crlf_out = ok_out.replace("\n", "\r\n")
    add("腿2 CRLF 行尾不算错（真批那一型的反面凭据）", "--drift", 0,
        crlf_out.replace(_NORMAL_PRINT, "--drift"), b"", [])
    add("腿2 CRLF 而档名真错 ⇒ 仍红（剥行尾没把腿剥哑）", "--drift", 0, crlf_out, b"", ["腿2"])
    add("腿3 stderr 非零", "normal", 0, ok_out, b"x" * 246, ["腿3"])
    add("腿4 口径行缺失", "normal", 0, ok_out.replace("口径：python=3.13.13｜-W=（无）｜SyntaxWarning 动作=default\n", ""),
        b"", ["腿4"])
    add("腿6 退码1而无红数行", "normal", 1, ok_out, b"", ["腿6"])
    add("腿6 退码1红数写0", "normal", 1, ok_out.replace("模式：", ">>> 验红 0 条红：\n模式："), b"", ["腿6"])
    add("腿6 退码0却印了红数", "normal", 0, ok_out.replace(_NORMAL_PRINT, _NORMAL_PRINT + "\n\n>>> 验红 5 条红："),
        b"", ["腿6"])
    # 批级那一票单独跑：两份都各自合格，但口径不同 ⇒ 只有批级腿看得见。
    r1 = check_one("normal", 0, ok_out, b"")
    alt = ok_out.replace("python=3.13.13", "python=3.10.13")
    r2 = check_one("normal", 0, alt, b"")
    same = check_batch([{"mode": "x", "caliber": r1[2]}, {"mode": "y", "caliber": r1[2]}])
    diff = check_batch([{"mode": "x", "caliber": r1[2]}, {"mode": "y", "caliber": r2[2]}])
    none = check_batch([{"mode": "x", "caliber": None}])
    cases.append(("腿5 两份口径相同 ⇒ 绿", note([]) or same == [], "点名 " + str([l for l, _ in same])))
    cases.append(("腿5 两份口径不同 ⇒ 红", note(sorted(l for l, _ in diff)) or sorted(l for l, _ in diff) == ["腿5"],
                  "点名 " + str([l for l, _ in diff])))
    cases.append(("腿5 全无口径 ⇒ 红（不是通过）", note(sorted(l for l, _ in none)) or sorted(l for l, _ in none) == ["腿5"],
                  "点名 " + str([l for l, _ in none])))

    # 档名现读这一支也要有牙：登记处读不到时必须出声，不许回落成空批。
    try:
        read_modes("MODES = ()\n")
        cases.append(("MODES 空名单 ⇒ 出声退出", False, "它回落成了空批"))
    except SystemExit as e:
        cases.append(("MODES 空名单 ⇒ 出声退出", True, str(e)[:40]))
    try:
        read_modes("没有登记处\n")
        cases.append(("MODES 读不到 ⇒ 出声退出", False, "它没退出"))
    except SystemExit as e:
        cases.append(("MODES 读不到 ⇒ 出声退出", True, str(e)[:40]))
    got = read_modes('MODES = ("--a",\n         "--b")\n')
    cases.append(("MODES 逐档现读", got == ["--a", "--b"], "读到 " + str(got)))

    # —— 逐档红数那一行自己（本发把 normal 印成 rmal，所以它从"我盯一下"变成"这三票钉着"）——
    cases.append(("_bare 只剥存在的那两个横杠", _bare("normal") == "normal" and _bare("--drift") == "drift",
                  "normal→" + _bare("normal") + "｜--drift→" + _bare("--drift")))
    _tl, _tr = tally_line([{"mode": "normal", "red_count": 0}, {"mode": "--drift", "red_count": 8}],
                          ["normal", "--drift"])
    cases.append(("逐档红数行：normal 原样在册、红数逐段带上",
                  _tr == [] and _tl.endswith("：normal 0｜drift 8") and "2 段 == 2 份" in _tl,
                  _tl[:52]))
    _tl2, _tr2 = tally_line([{"mode": "rmal", "red_count": 0}], ["normal"])
    cases.append(("逐档红数行：档名被截 ⇒ 腿7 红（旧写法今天就红在这一票上）",
                  len(_tr2) == 1 and _tr2[0][0] == "腿7" and "rmal" in _tr2[0][1],
                  _tr2[0][1][:46] if _tr2 else "没红"))
    _tl3, _tr3 = tally_line([{"mode": "normal", "red_count": "3｜9"}], ["normal"])
    cases.append(("逐档红数行：段数 ≠ 份数 ⇒ 腿7 红（分母自己会说出来）",
                  any(l == "腿7" and "段" in m for l, m in _tr3),
                  "红 " + str(len(_tr3)) + " 条"))

    bad = [c for c in cases if not c[1]]
    print("[驱动自测] 共 " + str(len(cases)) + " 项：")
    for name, ok, detail in cases:
        print("   " + ("OK  " if ok else "FAIL") + " " + name + "｜" + detail)
    if bad:
        print(">>> 驱动自测 " + str(len(bad)) + " 项不合格 ⇒ 这个驱动说『合格』不算数")
        raise SystemExit(1)
    _tally_cases = sum(1 for c in cases if c[0].startswith("逐档红数行") or c[0].startswith("_bare"))
    print("结论：" + str(len(cases)) + " 项全 OK｜各腿票数（按**期望票**计，红腿点名／绿票不点名）："
          + "｜".join(k + "×" + str(v) for k, v in sorted(leg_votes.items()))
          + "｜MODES 与逐档红数行那 " + str(_tally_cases) + " 项不进腿票数（它们验的不是单份日志，"
            "而是登记处读法与回执那一行自己）。"
             "票数不写死：本发给 腿2 补了两票 CRLF，写死的那句当场就过期了。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="守卫跑批驱动（回执自带解释器口径）")
    ap.add_argument("--jobs", type=int, default=2, help="并发档数；2 是历史口径，改它只改墙钟不改判据")
    ap.add_argument("--out", default="_red_batch_logs", help="日志与回执落盘目录（可再生产物）")
    ap.add_argument("--only", default=None, metavar="MODE,MODE",
                    help="只跑点名的档（补跑链用），写成 `--only=--drift,--floorcut` 一整串。"
                         "为什么不用 `--only --drift`：argparse 会把 `--drift` 当成自己的选项吃掉，"
                         "回一句 expected one argument——传进去的档名以两个横杠开头这件事没法用参数形状绕开")
    ap.add_argument("--selftest", action="store_true",
                    help="零守卫运行：拿合成日志验这各条腿各自点名，跑不到真批")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    registered = read_modes(open(os.path.join(HERE, GUARD), encoding="utf-8").read())
    if args.only:
        modes = [x.strip() for x in args.only.replace("，", ",").split(",") if x.strip()]
        unknown = [m for m in modes if m not in registered]
        if unknown:
            raise SystemExit("点名的档不在 MODES 登记处：" + str(sorted(unknown))
                             + "｜登记处在册 " + str(len(registered)) + " 档 ⇒ 不跑它，也不许让它占回执一行"
                             "（否则『本轮跑了 N 份』里的 N 包含一份根本没跑的）")
    else:
        modes = ["normal"] + registered
    dups = [m for m in set(modes) if modes.count(m) > 1]
    if dups:
        raise SystemExit("点名的档里有重复：" + str(sorted(dups)) + " ⇒ 同一档跑两遍会把自己的日志盖掉，"
                         "回执上的份数就不是跑过的份数")
    out_dir = os.path.join(HERE, args.out)
    os.makedirs(out_dir, exist_ok=True)
    scope = ("整轮：normal + MODES 全部" if not args.only else "点名补跑：" + str(len(registered)) + " 档在册")
    print("[跑批] " + str(len(modes)) + " 份（" + scope + "）｜jobs=" + str(args.jobs)
          + "｜解释器 " + sys.version.split()[0])
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rows = list(ex.map(lambda m: run_one(m, out_dir), modes))
    wall = time.time() - t0

    tally, tally_reds = tally_line(rows, modes)
    calib_reds = check_batch(rows)
    batch_reds = calib_reds + tally_reds
    all_reds = [(r["mode"], leg, msg) for r in rows for leg, msg in r["reds"]] \
               + [("", leg, msg) for leg, msg in batch_reds]

    for r in rows:
        print("  " + r["mode"] + "｜rc=" + str(r["rc"]) + "｜红 " + str(r["red_count"])
              + "｜stdout " + str(r["out_bytes"]) + "B｜stderr " + str(r["err_bytes"]) + "B｜"
              + str(r["secs"]) + "s")
    print("\n口径（全批 " + str(len(modes)) + " 份逐字相同 = " + str(not batch_reds) + "）："
          + (rows[0]["caliber"] or "（读不到）"))
    print("机时：并发 " + str(args.jobs) + "，末份落盘于开跑后 " + str(round(wall)) + " 秒（只报读数，不报倍数）")
    print(tally)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"caliber": rows[0]["caliber"], "wall_secs": round(wall), "jobs": args.jobs,
                   "n_runs": len(modes), "rows": rows, "batch_reds": batch_reds}, f,
                  ensure_ascii=False, indent=1)
    if all_reds:
        print(">>> 本轮不合格 " + str(len(all_reds)) + " 条：")
        for m, leg, msg in all_reds:
            print("   [" + (m or "批级") + "][" + leg + "] " + msg)
        raise SystemExit(1)
    print("结论：每份退码与红数互证、模式行回读一致、stderr 各 0 字节、全批口径逐字相同。")


if __name__ == "__main__":
    main()
