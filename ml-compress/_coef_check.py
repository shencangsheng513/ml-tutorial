"""自证 D4 / T95 两张系数表：先把已在册的 n=3/5/8/12 复算一遍对上，再给 n=16 出数。
方法：D4 = E[极差]/σ = 2n∫xΦ(x)^(n-1)φ(x)dx（Simpson）；T95 = 双侧 95% t 分位数，
由正则化不完全贝塔函数的连分式（Numerical Recipes betacf）+ 二分反解。不查表、不凭记忆。"""
import math


def phi(x):
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d2(n, lo=-12.0, hi=12.0, steps=20000):
    h = (hi - lo) / steps
    tot = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        w = 1.0 if i in (0, steps) else (4.0 if i % 2 else 2.0)
        tot += w * x * (Phi(x) ** (n - 1)) * phi(x)
    return 2.0 * n * (h / 3.0) * tot


def betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = c * d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """正则化不完全贝塔 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * betacf(b, a, 1.0 - x) / b


def t95(nu, conf=0.95):
    """双侧 conf 的分位数：I_x(nu/2, 1/2) = 1-conf，x = nu/(nu+t^2)。"""
    target = 1.0 - conf
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if betai(nu / 2.0, 0.5, mid) < target:
            lo = mid
        else:
            hi = mid
    x = (lo + hi) / 2.0
    return math.sqrt(nu * (1.0 - x) / x)


if __name__ == "__main__":
    print("表里在册的值 → 复算值（先把在册的 n=3/5/8/12/16 全对齐，才敢用同一把积分去要新档 n=26）")
    for n, d_old, t_old in ((3, 1.693, 4.303), (5, 2.326, 2.776), (8, 2.847, 2.365),
                            (12, 3.258, 2.201), (16, 3.532, 2.131)):
        gd, gt = d2(n), t95(n - 1)
        print(f"  n={n:<3} 在册 D4={d_old:.3f} 复算 {gd:.4f}（差 {gd - d_old:+.4f}） | "
              f"在册 T95={t_old:.3f} 复算 {gt:.4f}（差 {gt - t_old:+.4f}）")
    print(f"  n=26  复算 D4={d2(26):.4f} | T95(df=25)={t95(25):.4f}"
          f"   ← 第十八发（Fashion 那坐标补到 26 条流）要写进 step4_distill 的两把系数")
