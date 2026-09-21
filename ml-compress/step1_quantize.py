# -*- coding: utf-8 -*-
"""压缩第1课：量化——把 fp32 换成 fp16/int8，到底省了什么、又偷偷付出了什么。
开工前立三个问题，本课每个臂都为其中一问服务：
  Q1 精度掉多少？（量化是有损压缩：连续值被塞进 256 个格子）
  Q2 体积真的小了吗？小几倍？（理论 int8 = 1/4，实际取决于"哪些层真被换了"）
  Q3 变快还是变慢？（int8 要算子买账，CPU 上常见"只省体积不省时间"）
臂设计：
  A fp32 基线                 参照物
  B fp16（model.half()）       一行代码的"穷人量化"
  C int8 动态量化·只换 Linear   torch 官方默认口径
  D int8 动态量化·点名 Conv2d   预期落空：动态量化不支持卷积，应与 C 完全相同
  E int8 静态量化（QuantStub 校准）唯一能吃到卷积的一臂
两条预埋的翻车，都是这一课的真正内容：
  ① C==D：本模型约 96% 的乘法在卷积里，而 torch 的 quantize_dynamic 根本不碰 Conv2d。
     "调 API 没报错"不等于"真的量化了"，所以每臂强制打印"换了几层 + 残余浮点参数占比"。
  ② C 和 D 是同一份权重，跑出的延迟却差一倍——这就是本实验的**噪声底**，
     所以延迟测 3 轮取中位数，且任何小于这个噪声的"提速"都不许当成结论。
延迟口径：锁单线程（多线程把固定开销摊平，会系统性低估量化的收益），batch=1 测中位延迟、
batch=128 测吞吐——两个都要，延迟看体验、吞吐看账单，量化对这两者的影响方向经常相反。
用法：先 python train_baseline.py 拿到基线权重，再 python step1_quantize.py
"""
import copy
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.ao.quantization as tqa
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA = HERE.parent / "data"
OUT = HERE / "compress_report_v1.json"
ARCHIVE_ACC = 0.9798          # ml-vision 第5课档案值
COMPUTE = ("Conv1d", "Conv2d", "Linear", "LSTM", "GRU")
CALIB_N, WARMUP, TIMED, ROUNDS = 1280, 50, 1500, 4


def build_cnn() -> nn.Module:
    """与 train_baseline.py / ml-vision 第5课同一副架构（动一行就等于换实验对象）。"""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))


def load_test() -> tuple[torch.Tensor, np.ndarray]:
    ds = MNIST(str(DATA), train=False, download=True)   # 已缓存则零网络
    x = torch.tensor(ds.data.numpy().astype(np.float32) / 255.0).unsqueeze(1)
    return x, np.array(ds.targets)


def load_calib(n: int) -> torch.Tensor:
    """静态量化的观察者只能看训练图。用测试图校准=把考卷混进拟合过程，
    量出来的精度就带水分了——数据泄漏最隐蔽的一种，因为它不需要多写一行代码就会发生。"""
    ds = MNIST(str(DATA), train=True, download=True)
    return torch.tensor(ds.data.numpy()[:n].astype(np.float32) / 255.0).unsqueeze(1)


class QuantWrapper(nn.Module):
    """静态量化必须亲手告诉它"数据从哪儿进、到哪儿出"：入口插 QuantStub、出口插 DeQuantStub。
    不包这层就会撞上 NotImplementedError: quantized::conv2d.new（本课第一版就是这么死的）。"""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.q = tqa.QuantStub()
        self.m = inner
        self.dq = tqa.DeQuantStub()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.dq(self.m(self.q(t)))


def payload_kb(model: nn.Module) -> float:
    """磁盘口径才算数：state_dict 序列化后的字节数（含 scale/zero_point 这些量化元数据）。
    只数 numel 是"理论体积"，不是用户下载到的大小。"""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024


def compute_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if type(m).__name__ in COMPUTE]


def quant_audit(model: nn.Module, ref: list[nn.Module]) -> tuple[int, int, float]:
    """本课的测谎仪：算力层被换了几层 / 共几层 / 被换的层覆盖了多少参数量。
    两个坑都踩过才写下来的：①动态量化后的 Linear 不满足 isinstance(m, nn.Linear)，
    按 isinstance 数会漏层，所以按**类名**匹配；②量化权重塞进了 _packed_params，
    既不在 model.parameters() 也不在 state_dict 的浮点张量里，所以"残余浮点"不能自己算分子分母，
    必须拿 fp32 参照模型的逐层参数量当尺子（否则 C 臂会谎报"残浮点 100%"）。"""
    layers = compute_layers(model)
    assert len(layers) == len(ref), f"量化前后层数对不上：{len(layers)} vs {len(ref)}"
    q = sum("quantized" in type(m).__module__ for m in layers)
    ref_n = [sum(p.numel() for p in r.parameters()) for r in ref]
    covered = sum(n for n, m in zip(ref_n, layers, strict=True)
                  if "quantized" in type(m).__module__)
    return q, len(layers), covered / (sum(ref_n) or 1)


def predict(model: nn.Module, x: torch.Tensor, half: bool) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            xb = x[i:i + 256].half() if half else x[i:i + 256]
            out.append(model(xb).argmax(1).numpy())
    return np.concatenate(out)


def timing(model: nn.Module, x: torch.Tensor, half: bool) -> dict:
    torch.set_num_threads(1)
    b128 = x[:128].half() if half else x[:128]
    model.eval()
    p50s, thr = [], []
    with torch.no_grad():
        for _ in range(ROUNDS):
            for _ in range(WARMUP):
                model(x[:1].half() if half else x[:1])
            lats = []
            for i in range(TIMED):
                t = x[i:i + 1].half() if half else x[i:i + 1]
                t0 = time.perf_counter()
                model(t)
                lats.append((time.perf_counter() - t0) * 1000)
            p50s.append((statistics.median(lats), sorted(lats)[int(TIMED * 0.95)]))
            t0 = time.perf_counter()
            for _ in range(8):
                model(b128)
            thr.append(8 * 128 / (time.perf_counter() - t0))
    kept = p50s[1:]                                # 第一轮冷（页缓存/线程池还没热），丢掉
    meds = [v[0] for v in kept]
    return {"ms_p50": round(statistics.median(meds), 3),
            "ms_p95": round(statistics.median(v[1] for v in kept), 3),
            "ms_rounds": [round(v, 3) for v in meds],
            "ms_spread_pct": round((max(meds) - min(meds)) / statistics.median(meds) * 100, 1),
            "throughput": round(statistics.median(thr[1:]), 1)}


def scale_error_demo(model: nn.Module) -> dict:
    """量化误差从哪来：拿最大的那层 Linear 权重现场量一遍再反量，看残差。"""
    lin = [m for m in model.modules() if type(m).__name__ == "Linear"]
    w = max(lin, key=lambda m: m.weight.numel()).weight.detach()
    scale = w.abs().max() / 127.0
    recon = torch.round(w / scale).clamp(-127, 127) * scale
    return {"layer_params": int(w.numel()), "scale": round(float(scale), 8),
            "max_abs_err": round(float((w - recon).abs().max()), 6),
            "mean_abs_err": round(float((w - recon).abs().mean()), 6),
            "rel_l2_err_pct": round(float((w - recon).norm() / w.norm() * 100), 3)}


def static_int8(trained: nn.Module, x: torch.Tensor) -> nn.Module:
    """E 臂：观察者看数据分布 → convert 成 int8 算子。校准只喂 1280 张训练图，不碰测试集。
    注意传进来的必须是**已训练**的模型——这里踩过一次坑：图省事写了 build_cnn()，
    于是量化了一个随机初始化的网络，精度 8.4% 还自我感觉"量化掉得真多"。"""
    st = QuantWrapper(copy.deepcopy(trained).eval())
    st.qconfig = tqa.get_default_qconfig(torch.backends.quantized.engine)
    prep = tqa.prepare(st)
    with torch.no_grad():
        for i in range(0, CALIB_N, 64):
            prep(x[i:i + 64])
    return tqa.convert(prep)


def main() -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        sys.exit("缺基线权重，先跑 python train_baseline.py")
    base = build_cnn()
    base.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    x, y = load_test()
    calib = load_calib(CALIB_N)
    ref = compute_layers(base)
    print(f"测试集 {len(y)} 张 | 校准 {CALIB_N} 张（取自训练划分）| torch {torch.__version__}"
          f" | 量化后端 {torch.backends.quantized.engine}"
          "（这台机器的构建不含 fbgemm，所以 int8 卷积只能走 oneDNN 路径）")

    dem = scale_error_demo(base)
    print(f"\n[误差从哪来] 最大 Linear 共 {dem['layer_params']} 个权重，一个 scale={dem['scale']} 摊 256 格："
          f"单值最大误差 {dem['max_abs_err']}、相对 L2 误差 {dem['rel_l2_err_pct']}%")

    arms: dict[str, tuple[nn.Module, bool]] = {
        "A fp32基线": (base, False),
        "B fp16": (copy.deepcopy(base).half(), True),
        "C int8动态(Linear)": (tqa.quantize_dynamic(copy.deepcopy(base), {nn.Linear}, dtype=torch.qint8), False),
        "D int8动态(点名Conv)": (tqa.quantize_dynamic(copy.deepcopy(base), {nn.Conv2d, nn.Linear},
                                                     dtype=torch.qint8), False),
    }
    try:
        arms["E int8静态(含卷积)"] = (static_int8(base, calib), False)
    except Exception as e:
        print(f"[E 臂失败] {type(e).__name__}: {str(e).splitlines()[0][:120]}  ← 记进报告，不当没发生过")

    rows: dict[str, dict] = {}
    base_pred = predict(base, x, False)
    base_kb = payload_kb(base)
    for name, (mdl, half) in arms.items():
        pred = predict(mdl, x, half)
        acc = float(accuracy_score(y, pred))
        q, tot, cover = quant_audit(mdl, ref)
        rows[name] = {"kb": round(payload_kb(mdl), 1), "acc": round(acc, 4),
                      "wrong": int(len(y) - round(acc * len(y))),
                      "flip_broken": int(((base_pred == y) & (pred != y)).sum()),
                      "flip_fixed": int(((base_pred != y) & (pred == y)).sum()),
                      "layers_quantized": f"{q}/{tot}", "param_coverage": round(cover, 3),
                      **timing(mdl, x, half)}
        r = rows[name]
        print(f"{name:18s} {r['kb']:6.1f} KB（{base_kb/r['kb']:.1f}x）| 精度 {r['acc']:.4f}"
              f" | 错 {r['wrong']:3d} 张（翻坏 {r['flip_broken']}/翻正 {r['flip_fixed']}）"
              f"| 换层 {r['layers_quantized']} 覆盖参数 {r['param_coverage']:.0%}"
              f"| {r['ms_p50']:.2f} ms/张（p95 {r['ms_p95']:.2f}，{len(r['ms_rounds'])} 轮离散 {r['ms_spread_pct']:.0f}%）"
              f"| {r['throughput']:.0f} 张/秒")

    a = rows["A fp32基线"]
    same = rows["C int8动态(Linear)"]["kb"] == rows["D int8动态(点名Conv)"]["kb"] \
        and rows["C int8动态(Linear)"]["acc"] == rows["D int8动态(点名Conv)"]["acc"]
    print(f"\n[基线自检] A 精度 {a['acc']:.4f} vs 档案 {ARCHIVE_ACC:.4f}"
          f"（差 {abs(a['acc'] - ARCHIVE_ACC) * 100:.2f}pp）")
    print(f"[翻车①C==D] 点名 Conv2d 有没有被量化：{'没有——两臂逐字节同尺寸同精度，动态量化不支持卷积' if same else '有差异，需查'}")
    print(f"[翻车②噪声底] 同一份权重的 C/D 两臂 ms_p50 = {rows['C int8动态(Linear)']['ms_p50']}"
          f" vs {rows['D int8动态(点名Conv)']['ms_p50']} ms → "
          f"小于这个差倍的'提速'都不能算结论")
    print("判决口径：体积按 A 归一、速度看 ms_p50 归一；换层不满 4/4 的臂，"
          "它的压缩比不能算作 int8 的整体成绩。")
    report = {"archive_acc": ARCHIVE_ACC, "test_n": int(len(y)), "torch": torch.__version__,
              "engine": torch.backends.quantized.engine, "single_thread": True,
              "quant_error_demo": dem, "arms": rows,
              "size_ratio_vs_A": {k: round(a["kb"] / v["kb"], 2) for k, v in rows.items()},
              "c_equals_d": bool(same)}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已存 compress_report_v1.json（全量 10000 张测试，零 API 费）")


if __name__ == "__main__":
    main()
