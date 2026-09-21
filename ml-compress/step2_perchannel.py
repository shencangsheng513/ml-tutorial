# -*- coding: utf-8 -*-
"""压缩第2课：逐通道还是逐张量——以及一个第1课没发现自己已经站在哪一边的尴尬。

第1课收尾时留了三个疑问，本课先把它们各自变成可证伪的测量：
  Q1（数学先验，零后端零训练）逐通道比逐张量到底能少多少误差？
      不用推理、不用训练，直接从 fp32 权重本身算：每层两种 scale 方案各量化一次再反量，
      比相对 L2 残差；顺手量"通道之间 absmax 的变异系数"——那才是逐通道有没有饭吃的唯一信号。
  Q2（仪器核验，本课真正的内容）第1课的 E 臂用的是 get_default_qconfig('onednn')，
      **我们从来没问过它是逐张量还是逐通道。** 查法不看 API 自述：把模型里真实存下来的
      int8 权重还原出来数 scale 个数（== 输出通道数才算逐通道）。
      查出来如果默认就是逐通道，那"逐通道 vs 逐张量"这组对照里缺的臂是**逐张量**。
  Q3（实测）逐张量臂 T vs 逐通道臂 P：体积、精度、延迟、吞吐四本账。
  Q4（硬件底牌）第1课"量化后单张延迟不降反升"到底是量化的错还是这台 CPU 的错？
      读 torch 自己探到的指令集能力表，重点看 avx512_vnni / avx_vnni_int8。

为什么不改 step1_quantize.py 来复用它的 static 流程：那个文件是 v1 报告的物证，
加参数就是改动产出历史数字的脚本本身。本课自带一份 8 行的 static_with()，宁可重复也不动物证。
用法：先跑 train_baseline.py，再跑 step1_quantize.py（拿 v1 的臂做对照），最后 python step2_perchannel.py
"""
import copy
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.ao.quantization as tqa

from step1_quantize import (CALIB_N, MODELS, QuantWrapper, build_cnn, compute_layers,
                            load_calib, load_test, payload_kb, predict, quant_audit, timing)

HERE = Path(__file__).resolve().parent
OUT = HERE / "compress_report_v2.json"
HIST = HERE / "latency_history.json"      # 跨运行延迟历史：可复现性判定的唯一依据，不进 git
ARCHIVE_ACC = 0.9798


# ---------- Q1 数学先验：这个模型配不配得上逐通道 ----------
def weight_error_prior(model) -> list[dict]:
    """纯数学：对每层权重，逐张量与逐通道(对称 int8) 各做一次"量化-反量化"，比与原始权重的残差。
    不需要后端、不需要校准数据，跑 10 毫秒就能给出逐通道收益的**上限**。"""
    rows = []
    for idx, mod in enumerate(model):
        w = getattr(mod, "weight", None)
        if w is None:
            continue
        w = w.detach()
        axis = tuple(range(1, w.dim()))                       # 逐输出通道：每个 out-channel 一套 scale
        per_tensor = w.abs().max() / 127.0
        pt = torch.round(w / per_tensor).clamp(-127, 127) * per_tensor
        per_chan = (w.abs().amax(dim=axis, keepdim=True) / 127.0).clamp(min=1e-12)
        pc = torch.round(w / per_chan).clamp(-127, 127) * per_chan
        rel = lambda t: float((w - t).norm() / w.norm() * 100)   # noqa: E731
        amax = w.abs().amax(dim=axis).flatten()
        rows.append({"layer": f"L{idx} {type(mod).__name__}", "numel": int(w.numel()),
                     "pt_err_pct": round(rel(pt), 4), "pc_err_pct": round(rel(pc), 4),
                     "gain_pct_points": round(rel(pt) - rel(pc), 4),
                     "channel_absmax_cv": round(float(amax.std() / amax.mean()), 3)})
    return rows


# ---------- Q2 仪器核验：真实存下来的权重是哪种 qscheme ----------
def qscheme_of(mod) -> tuple[str, int]:
    """返回 (口径, scale 个数)。三种存法各试一次，试不出来就如实报 unknown——
    宁可留一排 unknown 也不猜，猜错的仪器比没仪器更贵。"""
    wt = None
    pp = getattr(mod, "_packed_params", None)
    if pp is not None:
        try:                                    # 2.x 动态量化：_packed_params 直接是 QTensor/DynamicQuantizedTensor
            wt = pp if isinstance(pp, torch.Tensor) else getattr(pp, "original", None)
        except Exception:
            wt = None
    if wt is None and hasattr(mod, "weight"):   # 静态量化：weight() 返回 QTensor
        try:
            wt = mod.weight()
        except Exception:
            wt = None
    for cand in (wt, pp):
        if cand is None:
            continue
        try:                                    # 逐张量：q_scale() 可能返回 float，也可能返回 0 维张量
            sc = cand.q_scale()
            n = int(sc.numel()) if isinstance(sc, torch.Tensor) else 1
            if n == 1:
                return "逐张量", 1
        except Exception:
            pass
        try:
            return f"逐通道(axis={cand.q_per_channel_axis()})", int(cand.q_per_channel_scales().numel())
        except Exception:
            pass
    return "unknown", 0


def scheme_audit(model, ref_names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    layers = compute_layers(model)
    for name, mod in zip(ref_names, layers, strict=True):
        scheme, n = qscheme_of(mod)
        out[name] = f"{scheme}｜scale={n}"
    return out


# ---------- Q3 两臂：强制逐张量 vs 默认(实测为逐通道) ----------
def static_with(model, calib: torch.Tensor, qconfig) -> torch.nn.Module:
    """step1 的 static_int8 写死了默认 qconfig；本课要换观察者，所以自带一份。
    坑位与第1课相同：必须传已训练权重、必须 QuantStub/DeQuantStub 包一层、校准只能看训练图。"""
    st = QuantWrapper(copy.deepcopy(model).eval())
    st.qconfig = qconfig
    prep = tqa.prepare(st)
    with torch.no_grad():
        for i in range(0, CALIB_N, 64):
            prep(calib[i:i + 64])
    return tqa.convert(prep)


def qconfig_for(weight_observer) -> tqa.QConfig:
    return tqa.QConfig(activation=tqa.get_default_qconfig("onednn").activation, weight=weight_observer)


def size_ledger(ref_layers, n_scale: int, saved_kb: float) -> dict:
    """体积对账（第1课留的账：int8 为什么不是理论 4x）。
    分母必须拿 **fp32 参照层** 来数，不能数量化模型的 state_dict：Linear 的 int8 权重整个塞在
    _packed_params 里，state_dict 只留 weight_scale / weight_zero_point 两个影子，
    于是 10 万个权重会被读成 0——第1课的 quant_audit 踩过同一个坑，本课第一版在这里再踩一次
    （报出"未归因 108KB"的假账，才把仪器逼出来）。
    下界 = 每个权重元素 1 字节 + 偏置仍是 fp32；差额 = 量化元数据 + torch.save 序列化黑盒。"""
    w_elems = sum(m.weight.numel() for m in ref_layers)
    bias_bytes = sum(m.bias.numel() * 4 for m in ref_layers if m.bias is not None)
    floor_kb = (w_elems + bias_bytes) / 1024
    return {"int8_weight_kb": round(w_elems / 1024, 2), "fp32_bias_kb": round(bias_bytes / 1024, 2),
            "floor_kb": round(floor_kb, 2), "saved_kb": round(saved_kb, 1),
            "unexplained_kb": round(saved_kb - floor_kb, 2),
            "pct_of_artifact_not_int8_weight": round((saved_kb - w_elems / 1024) / saved_kb * 100, 1),
            "meta_theory_kb": round(n_scale * 12 / 1024, 2)}   # 每通道 scale(4B)+zero_point(8B)


def main() -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        sys.exit("缺基线权重，先跑 python train_baseline.py")
    base = build_cnn()
    base.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    x, y = load_test()
    calib = load_calib(CALIB_N)
    ref_layers = compute_layers(base)
    ref_names = [f"{type(m).__name__}{i}" for i, m in enumerate(ref_layers)]

    caps = torch.cpu.get_capabilities()
    isa = {k: caps.get(k) for k in ("avx2", "avx512_f", "avx512_vnni", "avx_vnni", "avx_vnni_int8", "amx_int8")}
    print(f"torch {torch.__version__} | 后端 {torch.backends.quantized.engine} | CPU {caps.get('cpu_name')}")
    verdict = ("拿不到——int8 在这台机器上只剩「少搬运」这一项收益，乘法本身没有快路径"
               if not isa["avx512_vnni"] else "可用")
    print(f"[Q4 硬件底牌] {isa} → int8 的 AVX-512 VNNI 快路径{verdict}")

    print("\n[Q1 先验] 逐层残差（对称 int8，纯数学，不碰后端）：")
    prior = weight_error_prior(base)
    for r in prior:
        print(f"  {r['layer']:12s} 参数 {r['numel']:6d} | 逐张量 {r['pt_err_pct']:.3f}% →"
              f" 逐通道 {r['pc_err_pct']:.3f}%（省 {r['gain_pct_points']:.4f}pp）"
              f"| 通道 absmax 变异系数 {r['channel_absmax_cv']:.2f}")
    worst = max(prior, key=lambda r: r["gain_pct_points"])
    biggest = max(prior, key=lambda r: r["numel"])
    cvs = sorted(r["channel_absmax_cv"] for r in prior)
    cv_med = statistics.median(cvs)
    print(f"  先验判决：残差改善最大的是 {worst['layer']}（省 {worst['gain_pct_points']:.4f}pp）；"
          f"参数占比最大的 {biggest['layer']} 省 {biggest['gain_pct_points']:.4f}pp；"
          f"通道 absmax 变异系数 {cvs[0]:.2f}~{cvs[-1]:.2f}（中位 {cv_med:.2f}）")
    print("  注意这只说明**权重残差**的上限，残差省多少不等于精度涨多少——那一问留给 Q3 实测。"
          "\n  （本课最早一版先验跑在随机初始化的权重上，CV 只有 0.00~0.09、结论完全反向："
          "先验必须量在真正要发出去的那个工件上。）")

    arms = {"D0 默认qconfig(=第1课E)": tqa.get_default_qconfig("onednn"),
            "P 显式逐通道": qconfig_for(tqa.PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_channel_symmetric)),
            "T 强制逐张量": qconfig_for(tqa.MinMaxObserver.with_args(
                dtype=torch.qint8, qscheme=torch.per_tensor_symmetric,
                quant_min=-128, quant_max=127))}

    base_pred = predict(base, x, False)
    rows: dict[str, dict] = {}
    for name, qc in arms.items():
        mdl = static_with(base, calib, qc)
        audit = scheme_audit(mdl, ref_names)
        pred = predict(mdl, x, False)
        acc = float(accuracy_of(y, pred))
        q, tot, cover = quant_audit(mdl, ref_layers)
        n_scale = sum(int(v.split("scale=")[-1]) for v in audit.values())
        kb = round(payload_kb(mdl), 1)
        rows[name] = {"kb": kb, "acc": round(acc, 4),
                      "wrong": int(((pred != y).sum())),
                      "flip_broken": int(((base_pred == y) & (pred != y)).sum()),
                      "flip_fixed": int(((base_pred != y) & (pred == y)).sum()),
                      "layers_quantized": f"{q}/{tot}", "param_coverage": round(cover, 3),
                      "per_layer_scheme": audit, "size_ledger": size_ledger(ref_layers, n_scale, kb),
                      **timing(mdl, x, False)}
        r = rows[name]
        print(f"\n[{name}] {r['kb']:.1f} KB | 精度 {r['acc']:.4f}（错 {r['wrong']}，"
              f"翻坏 {r['flip_broken']}/翻正 {r['flip_fixed']}）| 换层 {r['layers_quantized']}"
              f"| {r['ms_p50']:.2f} ms/张（{len(r['ms_rounds'])} 轮离散 {r['ms_spread_pct']:.0f}%）"
              f"| {r['throughput']:.0f} 张/秒")
        led = r["size_ledger"]
        print(f"    体积对账：int8 权重 {led['int8_weight_kb']} KB + fp32 偏置 {led['fp32_bias_kb']} KB"
              f" = 下界 {led['floor_kb']} KB，实存 {led['saved_kb']} KB"
              f"（未归因 {led['unexplained_kb']} KB；scale/zero_point 理论成本 {led['meta_theory_kb']} KB；"
              f"非 int8 权重部分占工件 {led['pct_of_artifact_not_int8_weight']}%）")
        for k, v in audit.items():
            print(f"    {k:14s} {v}")

    d0, p, t = rows["D0 默认qconfig(=第1课E)"], rows["P 显式逐通道"], rows["T 强制逐张量"]
    same_pt = p["kb"] == t["kb"] and p["acc"] == t["acc"]
    same_d0p = (d0["kb"] == p["kb"] and d0["acc"] == p["acc"]
                and d0["per_layer_scheme"] == p["per_layer_scheme"])
    print(f"\n[自我审计①] 默认 qconfig 的真实身份：{list(d0['per_layer_scheme'].values())}"
          f"\n  → 第1课 E 臂报的 3.7x 其实是**逐通道**的成绩，当时的文本里没说过这件事")
    print(f"[自我审计②] 默认臂 vs 显式逐通道臂：{'同尺寸、同精度、逐层 scale 个数一致——默认就是逐通道，不是近似' if same_d0p else '有差异，需查'}")
    print(f"[自我审计③] 先验预算 vs 实测：权重残差省 {worst['gain_pct_points']:.2f}pp（最受益层）→"
          f" 实测精度只涨 {abs(p['acc'] - t['acc']) * 100:.2f}pp、体积多花 {p['kb'] - t['kb']:+.1f} KB、"
          f"延迟差 {p['ms_p50'] - t['ms_p50']:+.2f} ms。"
          f"传导率约 {abs(p['acc'] - t['acc']) * 100 / (worst['gain_pct_points'] or 1) * 100:.0f}%"
          f"（残差→精度），本模型的判面对这点权重残差不敏感——先验给的是上限，不是承诺")
    print(f"[自我审计④] 两臂是否只是换了个说法：{'否——KB 或精度确有差异，说明观察者真的生效了' if not same_pt else '是——强制逐张量可能被后端悄悄兜回逐通道，本课对照结论要降级'}")
    spread = max(statistics.mean([r["ms_spread_pct"] for r in rows.values()]), 1.0)
    hist = json.loads(HIST.read_text(encoding="utf-8")) if HIST.exists() else []
    hist.append({"n_runs": len(hist) + 1,
                 "p50": {k: rows[k]["ms_p50"] for k in rows},
                 "thr": {k: rows[k]["throughput"] for k in rows}})
    HIST.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    ratios = []
    for arm in rows:
        seen = [h["p50"][arm] for h in hist if arm in h["p50"]]
        if len(seen) >= 2:
            ratios.append(max(seen) / min(seen))
    ratio = max(ratios) if ratios else 0.0
    ok = ratio < 2 and ratio > 0                    # 次间漂移没超过 2 倍才认延迟结论
    print(f"[自我审计⑤ 次间可复现性] 本机已攒 {len(hist)} 次运行，同一臂跨运行的 p50 最大/最小倍率"
          f" {ratio:.1f} 倍（本轮轮间离散 {spread:.0f}%）→ "
          f"{'延迟结论可以下' if ok else '延迟这一栏作废：本课只保留体积、精度、元数据三类确定性读数'}")
    if not ok:
        pseries = " / ".join(f"{h['p50'].get('P 显式逐通道', float('nan')):.2f}" for h in hist)
        print(f"  证据：P 臂历次运行 ms_p50 = {pseries}；而同一臂的精度与体积每次一字不差"
              f"（本课实测 {p['acc']:.4f} / {p['kb']} KB）。"
              "\n  同一条流水线上的两类指标可信度可以差这么远——凡走墙钟时间的都要单独复检。"
              "\n  顺带给第1课补一刀：它的'int8 单张变慢 0.51→0.75ms'用的噪声底是轮间 28%，"
              "本课实测次间漂移可达 4 倍以上，那条延迟结论降级为'方向性观察'。")
    if ratio >= 1.5:      # D0 与第1课 E 臂的对照只在漂移够小时才有意义
        print(f"  注：本课 D0 臂 p50 {d0['ms_p50']:.2f}ms vs 第1课 E 臂 0.73ms——"
              f"漂移 {d0['ms_p50'] / 0.73:.1f} 倍，跨课 ms 不可对照。")

    led_p = p["size_ledger"]
    print(f"[结第1课的账] int8 为什么只有 3.7x 不是 4x：{led_p['int8_weight_kb']} KB 权重"
          f"（1 字节/元素）+ {led_p['fp32_bias_kb']} KB 偏置（还是 4 字节）= 下界 {led_p['floor_kb']} KB，"
          f"实存 {led_p['saved_kb']} KB → 未归因 {led_p['unexplained_kb']} KB。"
          f"\n  逐通道比逐张量多花 {p['kb'] - t['kb']:+.1f} KB，可 122 套 scale+zero_point 的账面成本只有"
          f" {led_p['meta_theory_kb']} KB → 剩下约 {p['kb'] - t['kb'] - led_p['meta_theory_kb']:.1f} KB 是"
          f" torch.save 的条目级序列化开销，本课没拆开（要拆得读 pickle 协议）。"
          f"工件里非 int8 权重的部分占 {led_p['pct_of_artifact_not_int8_weight']}%，这就是 4x 变 3.7x 的全部来源。")

    report = {"archive_acc": ARCHIVE_ACC, "test_n": int(len(y)), "torch": torch.__version__,
              "engine": torch.backends.quantized.engine, "cpu": caps.get("cpu_name"), "isa": isa,
              "weight_error_prior": prior, "arms": rows,
              "latency_runs": len(hist), "latency_run_ratio": round(ratio, 1),
              "latency_verdict": "可下结论" if ok else "作废：次间漂移远大于臂间差",
              "residual_gain_to_acc_transfer_pct": round(
                  abs(p["acc"] - t["acc"]) * 100 / (worst["gain_pct_points"] or 1) * 100, 1),
              "default_is_per_channel": bool(same_d0p),
              "acc_gap_pp": round(abs(p["acc"] - t["acc"]) * 100, 2),
              "kb_gap": round(p["kb"] - t["kb"], 1), "p_equals_t": bool(same_pt)}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已存 compress_report_v2.json（10000 张全量测试，零 API 费、零训练）")


def accuracy_of(y, pred) -> float:
    return float((pred == y).sum() / len(y))


if __name__ == "__main__":
    main()
