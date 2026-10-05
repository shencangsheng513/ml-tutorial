# -*- coding: utf-8 -*-
"""在桌面放一个「双击即开阶段 0 诊断卷」的快捷方式。

为什么不把 .bat 复制到桌面：桌面多一份副本，以后改了原件，副本还在说旧的话——
你双击的和我验的就不是同一个东西了（`_plans/make_desktop_shortcut.py` 那条同一理由）。
中文快捷方式名不能拼进命令行（码页会写成乱码），所以整段脚本编成 UTF-16LE 的
-EncodedCommand 递进去，中文只在这一处出现。
"""
import base64
import os
import subprocess
import sys

BAT = r"D:\ML_Projects\ml-tutorial\step0_diagnostic\start_diagnostic.bat"
WORKDIR = r"D:\ML_Projects\ml-tutorial\step0_diagnostic"
NAME = "阶段0诊断卷"
WORD = "诊断"


def build_ps(lnk):
    return "\n".join([
        "$sh = New-Object -ComObject WScript.Shell",
        "$s = $sh.CreateShortcut('%s')" % lnk,
        "$s.TargetPath = '%s'" % BAT,
        "$s.WorkingDirectory = '%s'" % WORKDIR,
        "$s.Description = 'stage 0 diagnostic paper - four questions, 45 minutes'",
        "$s.IconLocation = '%SystemRoot%\\System32\\shell32.dll,41'",
        "$s.Save()",
        "$c = $sh.CreateShortcut('%s')" % lnk,
        "Write-Output ('TARGET=' + $c.TargetPath)",
        "Write-Output ('WORKDIR=' + $c.WorkingDirectory)",
    ])


def main():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop):
        print("[FAIL] 找不到桌面目录 %s" % desktop)
        return 1
    if not os.path.exists(BAT):
        print("[FAIL] 原件不在 %s，先别建快捷方式" % BAT)
        return 1
    dup = [e for e in os.listdir(desktop) if e.endswith(".lnk") and WORD in e]
    if dup and dup != [NAME + ".lnk"]:
        print("[FAIL] 桌面上已有「%s」类快捷方式：%s（不覆盖，先告诉导师）" % (WORD, dup))
        return 1
    lnk = os.path.join(desktop, NAME + ".lnk")
    enc = base64.b64encode(build_ps(lnk).encode("utf-16-le")).decode("ascii")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                        "-EncodedCommand", enc], capture_output=True)
    out = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")
    print(out.strip())
    if r.returncode != 0:
        print("[FAIL] powershell 退码 %d（桌面没动过）" % r.returncode)
        return 1
    got = [l for l in out.splitlines() if l.startswith("TARGET=")]
    if not got or got[0] != "TARGET=" + BAT:
        print("[FAIL] 回读指向 %s，应为 %s" % (got, BAT))
        return 1
    listing = [e for e in os.listdir(desktop) if e.endswith(".lnk") and WORD in e]
    if listing != [NAME + ".lnk"]:
        print("[FAIL] 桌面上「%s」类快捷方式不止一个：%s" % (WORD, listing))
        return 1
    print("[OK] 桌面已放好：%s.lnk -> %s" % (NAME, BAT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
