# -*- coding: utf-8 -*-
"""压缩第3课：结构化剪枝——专门去兑第1课自己开的那张支票。

第1课判词原文：「要降低单张延迟 → 量化不是这一课的答案，那是剪枝/蒸馏的活。」
开出支票就得兑现，所以本课只问一件事：**剪枝到底能不能把单张延迟打下来**。
但按本仓库的规矩，先把它拆成可证伪的四问，而且顺序不能反——先算确定性账，再碰墙钟：

  Q1 先验（零后端、零训练）：剪 conv2 的 r% 输出通道，理论上去掉多少参数、多少乘加？
      第1课的教训是「体积跟参数分布走、速度跟算力分布走，这模型两者正好相反」，
      所以两笔账必须分开列。剪枝选点选在算力那一侧（conv2 占整网九成乘加），
      这正是它被寄予厚望的原因；先验若显示它同时砍两侧，那是可加的好消息，得当场核。
  Q2 仪器核验：torch.nn.utils.prune 挂上去的到底是「剪掉了」还是「盖了张遮布」？
      数 state_dict 的真实字节 + 看拓扑有没有跟着变，不看 API 有没有抛异常。
      （第1课 D 臂那个坑会在这儿换个样子重演。）
  Q3 精度：直接剪 vs 剪后微调，各掉几个点？翻牌数逐臂打印（老纪律：只看平均分会把噪声当疗效）。
  Q4 延迟：第2课实测同一臂跨运行 p50 能漂 4.3 倍，那本课怎么敢下延迟结论？
      → 换测量法：**同进程内交错配对**测（干扰会同时打到同一轮里的所有臂，比值不动），
        并且塞一份「和基线一模一样的复本」当对照臂，先量这个仪器自己的分辨率；
        噪声底压不住就把延迟栏作废，不许嘴硬。
  Q5 组合：剪完再 int8 静态量化，两笔收益可加吗？能加才叫流水线，不能加说明在抢同一块肉。

一处自己埋的雷先说在前面：剪 conv2 通道要连带改 fc1 的输入宽度（32×7×7=1568 那一维），
索引算错不会报错、只会悄悄给出一个「精度看着还行」的错模型。所以 main 的第一件事是
**全保留对照**：keep = 全部通道重切一遍，权重必须与基线逐位相等、预测必须逐张一致——
先校准手术刀，再往活的模型上落。

用法：先 train_baseline.py（拿权重），再
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python step3_prune.py
    # 那两个环境变量不是讲究：自校准那行要印 ✓/✗，Windows 默认 GBK 控制台会在算完之后、
    # 落盘之前崩掉。纯 CPU，MNIST 已缓存，零 API 费。
"""
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as tqa
import torch.nn.utils.prune as prune

from step1_quantize import MODELS, build_cnn, payload_kb, predict
from step2_perchannel import static_with
from train_baseline import load_split

HERE = Path(__file__).resolve().parent
OUT = HERE / "compress_report_v3.json"
HIST = HERE / "latency_history_v3.json"          # 跨运行历史：延迟可复现性的唯一依据，不进 git
ARCHIVE_ACC = 0.9798
SEED, FT_EPOCHS, FT_LR, FT_BS, VAL_N = 42, 8, 1e-4, 128, 1500
C2, FC1 = 3, 7                                    # build_cnn() 里 conv2 / fc1 的下标
LAT_ROUNDS, LAT_N, LAT_GATE_PCT = 8, 800, 5.0     # 交错轮数 / 每轮张数 / 对照臂噪声底门槛
# 采样量不是装饰：best 口径靠「够多样本里总有一次没被干扰」，300 张那一轮对照臂直接漂到 11%


# ---------- Q1 先验：两笔账分开算 ----------
def compute_layers2(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """统一命名，让「乘加账」和「参数账」用同一套 key，免得两张表对不上。
    QuantWrapper 不可迭代，先拆内层：静态量化只换算子、不改拓扑，
    所以按 fp32 内层的形状数乘加才是同一口径（第一版没拆，量化臂直接 TypeError 崩在这里）。"""
    n = {"Conv2d": 0, "Linear": 0}
    out = []
    for mod in getattr(model, "m", model):
        t = type(mod).__name__
        if t in n:
            n[t] += 1
            out.append((f"{'conv' if t == 'Conv2d' else 'fc'}{n[t]}", mod))
    return out


def macs_by_layer(model: nn.Module, side: int = 28) -> dict:
    """解析式乘加数（确定性，跨运行一字不差）。卷积 out×in×k²×H_out×W_out，全连接 in×out。
    只算这两类——本课的对象就是它们。pool/relu 的搬运不计，第1课已证它不是瓶颈。"""
    hw = {"conv1": side * side, "conv2": (side // 2) ** 2}
    out = {}
    for name, mod in compute_layers2(model):
        if type(mod).__name__ == "Conv2d":     # 按类名，不按 isinstance：量化版 Conv2d 不满足 isinstance
            out[name] = mod.out_channels * mod.in_channels * mod.kernel_size[0] ** 2 * hw[name]
        else:
            out[name] = mod.in_features * mod.out_features
    return out


def params_by_layer(model: nn.Module) -> dict:
    return {name: sum(p.numel() for p in mod.parameters()) for name, mod in compute_layers2(model)}


# ---------- 手术刀：按重要性剪 conv2 的输出通道，连带改 fc1 输入宽度 ----------
def channel_importance(model: nn.Module) -> torch.Tensor:
    """NetU 那套老办法：每个输出通道的权重 L1 范数当重要性。剪之前先看它的分布——
    32 个通道若一样重要（变异系数小），那「剪谁」根本不重要、只有「剪几个」重要；
    CV 大才说明选择性剪枝有饭吃。这是第2课「先验定上限」的同一条路子。"""
    w = model[C2].weight.detach()
    return w.flatten(1).abs().sum(1)


def rebuild(src: nn.Module, keep: torch.Tensor) -> nn.Module:
    """真删：新建一副窄的 Sequential，把要留的权重搬过去。
    conv2 剪输出通道 c ⇒ fc1 的第 c*spatial..(c+1)*spatial-1 **列**一起删（Flatten 的 c 在最外层）。
    偏置挂在输出侧，fc1 的偏置一行都不该少——这里最容易写错的就是拿 weight 的行去剪。"""
    c1, c2, fc1, fc2 = src[0], src[C2], src[FC1], src[9]
    spatial = fc1.in_features // c2.out_channels
    k = int(keep.numel())
    m = nn.Sequential(
        nn.Conv2d(c1.in_channels, c1.out_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(c1.out_channels, k, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(k * spatial, fc1.out_features), nn.ReLU(),
        nn.Linear(fc2.in_features, fc2.out_features))
    for tgt, s in ((m[0], c1), (m[9], fc2)):
        tgt.weight.data = s.weight.data.clone()
        tgt.bias.data = s.bias.data.clone()
    m[C2].weight.data = c2.weight.data.index_select(0, keep)
    m[C2].bias.data = c2.bias.data.index_select(0, keep)
    cols = torch.cat([torch.arange(int(c) * spatial, (int(c) + 1) * spatial) for c in keep])
    m[FC1].weight.data = fc1.weight.data.index_select(1, cols)
    m[FC1].bias.data = fc1.bias.data.clone()
    return m


def keep_indices(imp: torch.Tensor, ratio: float) -> torch.Tensor:
    """按重要性留 (1-ratio) 个通道，索引升序返回（升序便于列切片，也便于打印「留了谁、弃了谁」）。"""
    k = max(1, round(imp.numel() * (1 - ratio)))
    return imp.topk(k).indices.sort().values


# ---------- Q2 假剪枝对照：只挂 mask 会怎样，remove 又只做到哪一步 ----------
def mask_only_demo() -> dict:
    """Q2 的对照臂：`torch.nn.utils.prune` 到底做了什么。
    本课开场时的预期是「remove 之后 conv2 的 out_channels 会缩到 16、fc1 维度对不上、前向该抛异常」——
    实测**两条都不成立**：remove 只是把 0 烧进 weight，形状全程没动，前向照常跑，体积回到起点。
    所以这一家 API 是**稀疏化/正则化**工具，不是"删维度"的结构化剪枝工具；
    想要真的把通道拿掉，得自己改拓扑（本课的 rebuild()），或者用 torch.ao.pruning 的依赖图剪枝。
    这个落差不是我为难自己——它正是第1课 D 臂那个坑的同类：**API 做了你以为它做的事了吗**。"""
    m = build_cnn()
    m.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    before = payload_kb(m)
    macs0 = sum(macs_by_layer(m).values())
    keys0 = sorted(m.state_dict().keys())
    prune.ln_structured(m[C2], "weight", amount=0.5, n=1, dim=0)
    masked = payload_kb(m)
    keys1 = sorted(m.state_dict().keys())
    masked_macs = sum(macs_by_layer(m).values())
    with torch.no_grad():
        m.eval()
        t0 = time.perf_counter()
        for _ in range(200):
            m(torch.zeros(1, 1, 28, 28))
        masked_ms = (time.perf_counter() - t0) / 200 * 1000
    prune.remove(m[C2], "weight")
    removed = payload_kb(m)
    err = "（没抛异常）"
    try:
        with torch.no_grad():
            m(torch.zeros(1, 1, 28, 28))
    except Exception as e:
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    return {"kb_before": round(before, 1), "kb_masked": round(masked, 1),
            "kb_after_remove": round(removed, 1),
            "new_state_keys": [k for k in keys1 if k not in keys0],
            # 形状全程没变 → 乘加一分不少。这比墙钟更有决定性，而且是现算的，不是我预期的常数
            "masked_macs_pct_of_base": round(masked_macs / macs0 * 100, 1),
            "masked_ms_unpaired": round(masked_ms, 3),   # 未锁线程、未交错的旁证，不与 Q4 同口径
            "conv2_out_after_remove": m[C2].out_channels, "fc1_in_after_remove": m[FC1].in_features,
            "forward_after_remove": err}


# ---------- Q3 微调 ----------
def fine_tune(m: nn.Module, xt, yt, xv, yv, epochs=FT_EPOCHS) -> tuple[nn.Module, list]:
    """剪完的网络是个「被砍了一刀的老模型」，要的是恢复不是重造，所以 lr 比基线小一个数量级。
    验证图从训练划分另切 1500 张、不参与拟合；测试集只在最后读一次——
    拿测试集挑 epoch 就是把考卷混进拟合（第1课坑 #3 的变体）。"""
    opt = torch.optim.Adam(m.parameters(), lr=FT_LR)
    lossf = nn.CrossEntropyLoss()
    hist, n = [], len(xt)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, FT_BS):
            kidx = perm[i:i + FT_BS]
            opt.zero_grad()
            loss = lossf(m(xt[kidx]), torch.tensor(yt[kidx]))
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(kidx)
        m.eval()
        with torch.no_grad():
            va = float((m(xv).argmax(1) == torch.tensor(yv)).float().mean())
        hist.append({"epoch": ep + 1, "train_loss": round(tot / n, 4), "val_acc": round(va, 4)})
    return m, hist


# ---------- Q4 交错配对计时 + 仪器自校准 ----------
def paired_timing(items: list[tuple[str, nn.Module]], x: torch.Tensor) -> tuple[dict, float]:
    """每轮把所有臂各测一遍、并轮转换顺序（消掉「越靠后越热」的系统偏置）。
    同进程内交错比「一臂跑完再跑下一臂」更抗外部干扰：干扰同时打到同一轮的所有臂，比值不动。
    返回 (每臂读数, 对照噪声底%)。噪声底 = 两份一模一样的权重之间的相对差，
    它就是这台机器上这个估计器自己的分辨率；臂间差小于它一律不算数。"""
    torch.set_num_threads(1)
    store = {k: {"p50": [], "min": [], "thr": []} for k, _ in items}
    b128 = x[:128]
    with torch.no_grad():
        for r in range(LAT_ROUNDS):
            order = items[r % len(items):] + items[:r % len(items)]
            for name, mdl in order:
                mdl.eval()
                for _ in range(20):
                    mdl(x[:1])
                lats = []
                for i in range(LAT_N):
                    t = x[i:i + 1]
                    t0 = time.perf_counter()
                    mdl(t)
                    lats.append((time.perf_counter() - t0) * 1000)
                store[name]["p50"].append(statistics.median(lats))
                store[name]["min"].append(min(lats))
                for _ in range(3):                                   # 批量的热身
                    mdl(b128)
                t0 = time.perf_counter()
                for _ in range(6):
                    mdl(b128)
                store[name]["thr"].append(6 * 128 / (time.perf_counter() - t0))
    ref = items[0][0]
    ctrl = [k for k, _ in items if "对照" in k]
    floors: dict[str, float] = {"p50": 0.0, "best": 0.0, "thr": 0.0}
    if len(ctrl) == 1:
        # 三个统计量分别取"丢掉第一轮后"的代表值：best 用最小、p50 与吞吐用中位数。
        # 噪声底必须**分栏报**：第一轮实测 best 只差 1.4%、p50 差 11%，
        # 合成一个数（取最大）会把唯一还有分辨率的那栏一起枪毙。
        reps = {"p50": ("p50", statistics.median), "best": ("min", min),
                "thr": ("thr", statistics.median)}
        for f, (key, pick) in reps.items():
            a, b = pick(store[ref][key][1:]), pick(store[ctrl[0]][key][1:])
            floors[f] = round(abs(a - b) / a * 100, 2)
    rows = {}
    for name, _ in items:
        kept = store[name]["p50"][1:]              # 第一轮冷（页缓存/线程池未热），丢掉
        rows[name] = {"ms_p50": round(statistics.median(kept), 3),
                      "ms_best": round(min(store[name]["min"][1:]), 3),
                      "thr_batch128": round(statistics.median(store[name]["thr"][1:]), 1),
                      "p50_rounds": [round(v, 3) for v in kept],
                      "best_rounds": [round(v, 3) for v in store[name]["min"][1:]],
                      "round_spread_pct": round((max(kept) - min(kept))
                                                / statistics.median(kept) * 100, 1)}
    return rows, floors


def append_history(rows: dict, floors: dict, fit: dict) -> tuple[int, float, float, dict]:
    hist = json.loads(HIST.read_text(encoding="utf-8")) if HIST.exists() else []
    cfg = [LAT_ROUNDS, LAT_N]
    hist.append({"n_runs": len(hist) + 1, "cfg": cfg, "ctrl_floor_pct": floors,
                 "p50": {k: v["ms_p50"] for k, v in rows.items()},
                 "best": {k: v["ms_best"] for k, v in rows.items()},
                 "thr": {k: v["thr_batch128"] for k, v in rows.items()},
                 "fit": {"fixed_overhead_ms": fit["fixed_overhead_ms"],
                         "compute_share_pct": fit["compute_share_pct"]}})
    HIST.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")
    same = [h for h in hist if h.get("cfg") == cfg]        # 不同采样配置不是同一个仪器，不混着比

    def max_ratio(col: str) -> float:
        rs = [max(s) / min(s) for s in
              ([h[col][arm] for h in same if arm in h.get(col, {})] for arm in rows) if len(s) >= 2]
        return round(max(rs), 2) if rs else 0.0

    # 上限是加法量（百分比），跨运行用 pp 跨度报，别用倍率——26% 与 34% 差 8pp，不是 1.3 倍
    # 早期同配置的运行没写 fit（那一栏是本轮才加的），只比有 fit 的：凑数会把仪器说小。
    with_fit = [h for h in same if "fit" in h]
    shares = [h["fit"]["compute_share_pct"] for h in with_fit]
    oh = [h["fit"]["fixed_overhead_ms"] for h in with_fit]
    spread = ({"n": len(with_fit), "share_min": min(shares), "share_max": max(shares),
               "share_span_pp": round(max(shares) - min(shares), 1),
               "overhead_min_ms": min(oh), "overhead_max_ms": max(oh)} if len(with_fit) >= 2 else {})
    return len(same), max_ratio("p50"), max_ratio("best"), spread


# ---------- 主流程 ----------
def main() -> None:
    if not (MODELS / "mnist_cnn_fp32.pt").exists():
        sys.exit("缺基线权重，先跑 python train_baseline.py")
    torch.manual_seed(SEED)
    xt_all, yt_all, xte, yte = load_split()
    cut = len(xt_all) - VAL_N
    xt, xv, yt, yv = xt_all[:cut], xt_all[cut:], yt_all[:cut], yt_all[cut:]
    base = build_cnn()
    base.load_state_dict(torch.load(MODELS / "mnist_cnn_fp32.pt", weights_only=True))
    base.eval()
    print(f"训练 {len(xt)} 拟合 / {len(xv)} 验证（都来自训练划分）| 测试 {len(yte)} 只读一次"
          f" | torch {torch.__version__} | 后端 {torch.backends.quantized.engine}")

    # --- 手术刀校准：全保留必须逐位等于基线 ---
    twin = rebuild(base, torch.arange(32))
    same_w = all(torch.equal(a, b) for a, b in
                 zip(base.state_dict().values(), twin.state_dict().values(), strict=True))
    base_pred = predict(base, xte, False)
    agree = int((predict(twin, xte, False) == base_pred).sum())
    print(f"\n[手术刀校准] keep=全部 32 通道重切一遍：权重逐位相等={same_w} | 预测逐张一致 {agree}/{len(yte)}")
    assert same_w and agree == len(yte), "手术刀没校准：先修 rebuild()，否则后面所有剪枝臂的数字都是废的"

    # --- Q1 先验 ---
    imp = channel_importance(base)
    cv = float(imp.std() / imp.mean())
    m_base, p_base = macs_by_layer(base), params_by_layer(base)
    tot_m, tot_p = sum(m_base.values()), sum(p_base.values())
    print(f"\n[Q1 先验] conv2 通道重要性 L1 {imp.min():.2f}~{imp.max():.2f}，变异系数 {cv:.2f} → "
          f"{'通道之间明显不等，选择性剪枝有饭吃' if cv > 0.2 else '通道几乎一样重要：剪谁不重要，剪几个才重要'}")
    print(f"  算力账 {m_base} → conv2 占 {m_base['conv2'] / tot_m:.0%}"
          f" | 参数账 {p_base} → fc1 占 {p_base['fc1'] / tot_p:.0%}（第1课那两个相反的方向）")
    prior = {}
    for ratio in (0.2, 0.5):
        keep = keep_indices(imp, ratio)
        pm = macs_by_layer(rebuild(base, keep))
        pp = params_by_layer(rebuild(base, keep))
        prior[ratio] = {"keep": int(keep.numel()),
                        "macs_pct": round(sum(pm.values()) / tot_m * 100, 1),
                        "param_pct": round(sum(pp.values()) / tot_p * 100, 1)}
        print(f"  名义剪 {ratio:.0%} → 实留 {prior[ratio]['keep']}/32 通道："
              f"乘加降到 {prior[ratio]['macs_pct']}%（延迟先验）、参数降到 {prior[ratio]['param_pct']}%"
              f"（体积先验）→ 这次两笔账同向，与第1课「压 fc 只动体积」正好相反")

    # --- Q2 假剪枝 ---
    md = mask_only_demo()
    print(f"\n[Q2 仪器核验] 只挂 mask 不 remove：{md['kb_before']} KB → {md['kb_masked']} KB"
          f"（{'不降反升' if md['kb_masked'] > md['kb_before'] else '确实变小'}），"
          f"新增 state_dict 条目 {md['new_state_keys']}")
    print(f"  决定性证据是算力账：挂 mask 后乘加仍占基线 {md['masked_macs_pct_of_base']}%"
          f" → 一个通道都没真的消失。（旁证：未锁线程的粗测 {md['masked_ms_unpaired']} ms/张，"
          f"与 Q4 的配对法不同口径，只看不比。）")
    print(f"  prune.remove 之后 → {md['kb_after_remove']} KB，conv2 输出 {md['conv2_out_after_remove']} 通道，"
          f"但 fc1 输入仍是 {md['fc1_in_after_remove']}，前向：{md['forward_after_remove']}"
          f"\n  → 「remove」只把遮布烧进权重，**不会替你改拓扑**。这就是本课自己写 rebuild() 的理由。")

    # --- Q3/Q5 各臂 ---
    arms: dict[str, dict] = {}
    models: list[tuple[str, nn.Module]] = [("A0 fp32基线", base), ("A0复本(对照)", copy.deepcopy(base))]
    base_kb = md["kb_before"]

    def record(tag: str, mdl: nn.Module, extra: dict | None = None,
               param_ref: nn.Module | None = None) -> dict:
        pred = predict(mdl, xte, False)
        acc = float((pred == yte).mean())
        mm = macs_by_layer(mdl)
        n_par = sum(params_by_layer(mdl).values())
        note = ""
        if n_par < 1000:      # 量化权重藏在 _packed_params 里，parameters() 根本看不见它
            if param_ref is not None:
                note = f"参数读数 {n_par} 是残缺的，改用 fp32 参照层的 {sum(params_by_layer(param_ref).values())}"
                n_par = sum(params_by_layer(param_ref).values())
            else:
                note = f"参数读数 {n_par} 可疑且无参照，不采信"
        row = {"kb": round(payload_kb(mdl), 1), "size_vs_base": round(base_kb / payload_kb(mdl), 2),
               "acc": round(acc, 4), "wrong": int((pred != yte).sum()),
               "flip_broken": int(((base_pred == yte) & (pred != yte)).sum()),
               "flip_fixed": int(((base_pred != yte) & (pred == yte)).sum()),
               "macs_pct": round(sum(mm.values()) / tot_m * 100, 1),
               "n_params": n_par, **({"param_count_note": note} if note else {}),
               **dict(extra or {})}
        arms[tag] = row
        print(f"[{tag:16s}] {row['kb']:6.1f} KB（{row['size_vs_base']:.2f}x）"
              f" | 乘加 {row['macs_pct']:5.1f}% 基线 | 参数 {row['n_params'] / 1e3:.1f}k"
              f" | 精度 {row['acc']:.4f}（错 {row['wrong']}，翻坏 {row['flip_broken']}/翻正 {row['flip_fixed']}）")
        return row

    record("A0 fp32基线", base)
    for ratio in (0.2, 0.5):
        keep = keep_indices(imp, ratio)
        raw = rebuild(base, keep)
        dropped = [c for c in range(32) if c not in set(keep.tolist())]
        record(f"R{int(ratio * 100)} 剪未微调", raw,
               {"nominal_ratio": ratio, "keep_ch": int(keep.numel()),
                "actual_ratio": round(1 - keep.numel() / 32, 3),
                "kept": keep.tolist(), "dropped": dropped,
                "prior_macs_pct": prior[ratio]["macs_pct"],
                "prior_param_pct": prior[ratio]["param_pct"]})
        ft, hist = fine_tune(copy.deepcopy(raw), xt, yt, xv, yv)
        record(f"R{int(ratio * 100)}F 剪+微调", ft,
               {"val_curve": [(h["epoch"], h["val_acc"]) for h in hist]})
        models += [(f"R{int(ratio * 100)}F", ft), (f"R{int(ratio * 100)}", raw)]
        if ratio == 0.2 and arms["R20 剪未微调"]["acc"] > arms["A0 fp32基线"]["acc"]:
            # 精度比基线还高 → 先别报疗效。同尺寸随机剪 26 通道当对照，把"挑得准"和"少即是多"分开。
            # 注意：rebuild() 里 nn.Conv2d(...) 的默认初始化要消耗**全局**随机流，
            # 不包一层 RNG 状态就会换掉后面 fine_tune 的 randperm 顺序——本课实测：
            # 加上这条对照之后 R50F 的翻牌数从 40/42 变成 44/46（精度仍是 0.9800，纯属运气）。
            rng_state = torch.get_rng_state()
            g = torch.Generator().manual_seed(SEED)
            rnd = torch.randperm(32, generator=g)[:keep.numel()].sort().values
            rr = record("R20 随机选通道(对照)", rebuild(base, rnd),
                        {"kept": rnd.tolist(), "why": "同尺寸、不看重要性：专门用来证伪「L1 选点有效」"})
            torch.set_rng_state(rng_state)
            d_sel = arms["R20 剪未微调"]["acc"] - rr["acc"]
            d_reg = rr["acc"] - arms["A0 fp32基线"]["acc"]
            tot_up = (arms["R20 剪未微调"]["acc"] - arms["A0 fp32基线"]["acc"]) * 100
            shrink_read = ("随机剪同尺寸就掉 {:.2f}pp → 涨幅不是「参数变少」本身带来的，"
                           "是**挑得准**带来的").format(abs(d_reg) * 100) if d_reg < 0 else \
                (f"随机剪同尺寸也涨 {d_reg * 100:+.2f}pp → 主要红利是「少参数=正则化」，"
                 f"选点只多 {d_sel * 100:+.2f}pp")
            print(f"  [好到可疑的鉴别] 净涨 {tot_up:+.2f}pp（{arms['R20 剪未微调']['flip_fixed'] - arms['R20 剪未微调']['flip_broken']:+d} 张）拆开看："
                  f"{shrink_read}；按 L1 挑比随机挑多 {d_sel * 100:+.2f}pp"
                  f"（{int(round(d_sel * len(yte)))} 张，{abs(d_sel) * len(yte) / 3:.0f} 倍于 ±3 张的噪声底）")
            print(f"     边界：这是**单种子单次**的对照，效应量够大所以方向可信，"
                  f"但 {tot_up:+.2f}pp 这种量级不许写成「剪枝能涨精度」——要写成「剪 20% 未观察到精度损失」。")
            prior["selection_control"] = {"random_acc": rr["acc"], "selection_gain_pp": round(d_sel * 100, 2),
                                          "shrink_gain_pp": round(d_reg * 100, 2)}
        if ratio == 0.5:
            q = static_with(ft, xt_all[:1280], tqa.get_default_qconfig(torch.backends.quantized.engine))
            record("R50FQ 再int8量化", q, {"quantized": True}, param_ref=ft)
            models.append(("R50FQ", q))
            b, c = arms["R50F 剪+微调"], arms["R50FQ 再int8量化"]
            print(f"  [Q5 组合可加性] 体积链：{base_kb} → {b['kb']} KB（剪枝 {base_kb / b['kb']:.2f}x）"
                  f" → {c['kb']} KB（在剪枝模型上再量化又 {b['kb'] / c['kb']:.2f}x）"
                  f"，合计 {base_kb / c['kb']:.2f}x")
            print(f"     乘加：剪枝把算力压到 {c['macs_pct']}%，int8 不改乘加（它改的是每一下乘法的价格）"
                  f" → 两笔收益不同轴，可加是合理的；"
                  f"\n     精度 {b['acc']:.4f} → {c['acc']:.4f}（{(c['acc'] - b['acc']) * 100:+.2f}pp，"
                  f"翻坏 {c['flip_broken']}/翻正 {c['flip_fixed']}，看翻牌别看平均分）")

    # --- Q4 延迟：交错配对 + 对照臂 + 固定开销拟合 ---
    print("\n[Q4 延迟] 交错配对计时中…")
    lrows, floors = paired_timing(models, xte)
    a0 = lrows["A0 fp32基线"]
    for name, v in lrows.items():
        print(f"  {name:16s} p50 {v['ms_p50']:.3f} ms（{a0['ms_p50'] / v['ms_p50']:.2f}x）"
              f" | best {v['ms_best']:.3f} ms（{a0['ms_best'] / v['ms_best']:.2f}x）"
              f" | 吞吐 {v['thr_batch128']:.0f} 张/秒（{v['thr_batch128'] / a0['thr_batch128']:.2f}x）"
              f" | p50 轮间离散 {v['round_spread_pct']:.0f}%")
    # 固定开销拟合：把"剪枝为什么没让单张变快"从解释变成数字
    # （两套 key 不同名是故意的：arms 按实验条件命名，lrows 按计时标签命名）
    # 必须先于 append_history：拟合要写进历史，才能算出"上限"这个读数自己的跨运行离散。
    fit_pts = [(arms[ak]["macs_pct"] / 100, lrows[lk]["ms_best"])
               for ak, lk in (("A0 fp32基线", "A0 fp32基线"),
                              ("R20 剪未微调", "R20"), ("R50 剪未微调", "R50"))]
    mx = statistics.mean(p[0] for p in fit_pts)
    my = statistics.mean(p[1] for p in fit_pts)
    slope = sum((x - mx) * (y - my) for x, y in fit_pts) / sum((x - mx) ** 2 for x, _ in fit_pts)
    icept = my - slope * mx
    resid = max(abs(y - (icept + slope * x)) for x, y in fit_pts)
    ceiling = (a0["ms_best"] - icept) / a0["ms_best"] * 100
    fit = {"points": [[round(x, 3), round(y, 3)] for x, y in fit_pts],
           "fixed_overhead_ms": round(icept, 3), "per_macs_ms": round(slope, 3),
           "max_abs_residual_ms": round(resid, 4), "compute_share_pct": round(ceiling, 1)}

    n_runs, run_p50, run_best, spread = append_history(lrows, floors, fit)
    ok = {k: v < LAT_GATE_PCT for k, v in floors.items()}
    print("  [仪器自校准] 对照臂（同一份权重的两份）跨臂相对差："
          + "｜".join(f"{k} {floors[k]:.1f}%{'✓' if ok[k] else '✗'}" for k in floors)
          + f"（门槛 {LAT_GATE_PCT:.0f}%）→ 逐栏判："
          + ("可用：" + "、".join(k for k, v in ok.items() if v) if any(ok.values()) else "三栏全废，")
          + ("；作废：" + "、".join(k for k, v in ok.items() if not v) if not all(ok.values()) else ""))
    print(f"  [跨运行] 同一采样配置（{LAT_ROUNDS} 轮 × {LAT_N} 张）本机已攒 {n_runs} 次："
          f"跨运行 p50 最大倍率 {run_p50 or '—'}、best 最大倍率 {run_best or '—'}"
          + ("（首次运行，暂无对照）" if not run_p50 else
             "——第2课那个 4.3x 是「一臂跑完再跑下一臂」量的；配对法有没有治好，看 best 这一栏"))
    print(f"  [确定性读数：延迟里有多少是'乘法数量'能管的] 三点拟合 ms_best = "
          f"{icept:.3f} + {slope:.3f}×乘加占比（最大残差 {resid:.3f} ms）"
          f"\n     → 固定开销 {icept:.3f} ms 占基线的 {100 - ceiling:.0f}%。"
          f"任何只砍乘加的手段（剪枝、int8、换算法）在这台机器上把单张延迟压到 {icept:.3f} ms 就到顶了，"
          f"上限 {ceiling:.0f}%——这才是第1课'量化没让单张变快'的真正解释，不用怪后端。"
          + ("" if not spread else
             f"\n     它自己的可复现性：同配置已攒 {spread['n']} 次，固定开销 "
             f"{spread['overhead_min_ms']:.3f}~{spread['overhead_max_ms']:.3f} ms，"
             f"上限 {spread['share_min']:.0f}%~{spread['share_max']:.0f}%"
             f"（跨度 {spread['share_span_pp']:.1f}pp）。这个读数走的是拟合不是墙钟，"
             f"所以判它稳不稳要看 pp 跨度，不看倍率。"))

    # --- 判词与自我审计 ---
    r20, r20f = arms["R20 剪未微调"], arms["R20F 剪+微调"]
    r50, r50f = arms["R50 剪未微调"], arms["R50F 剪+微调"]
    noise_pp = 3 / len(yte) * 100          # ±3 张是本课反复用到的噪声底（与自我审计④同一口径）
    print(f"\n[自我审计① 基线对账] A0 精度 {arms['A0 fp32基线']['acc']:.4f} vs 档案 {ARCHIVE_ACC:.4f}"
          f"（差 {abs(arms['A0 fp32基线']['acc'] - ARCHIVE_ACC) * 100:.2f}pp）")
    for tag, raw, ftt in (("20%", r20, r20f), ("50%", r50, r50f)):
        net = (ftt["acc"] - arms["A0 fp32基线"]["acc"]) * 100
        if abs(net) <= noise_pp:
            name = "与基线分不开（噪声底内，如实报'未观察到损失'，不许报'净赚'）"
        else:
            name = "净赚，不叫代价" if net > 0 else "这才是剪枝本身的代价"
        print(f"[自我审计② {tag} 的账要分两段] 同一刀：未微调 {raw['acc']:.4f} → 微调后 {ftt['acc']:.4f}"
              f"（微调救回 {(ftt['acc'] - raw['acc']) * 100:+.2f}pp）；"
              f"再和基线比是 {net:+.2f}pp（{name}）。"
              "两段不分开报，就会把「没训练够」算成压缩的伤害。")
    print(f"[自我审计③ 先验 vs 实测] 名义 20%/50% → 实际 {r20['actual_ratio']:.1%}/{r50['actual_ratio']:.1%}"
          f"（32 通道取整，粒度不是 bug）；乘加先验 {r20['prior_macs_pct']}%/{r50['prior_macs_pct']}%"
          f" vs 实测 {r20['macs_pct']}%/{r50['macs_pct']}%（解析式与实现同源于形状，若不一致就是形状写错）")
    print("[自我审计④ 好到可疑先查仪器] 逐臂翻牌数已打印；R20 未微调那种 +0.1pp 量级只值十几张，"
          "而 ±3 张就算噪声底——所以它进的是「未观察到损失」，不是「涨精度」")
    stat = "ms_best" if ok["best"] else "ms_p50"
    passed = [k for k, v in ok.items() if v]
    which = (f"{stat}（{'、'.join(passed)} 栏过了自校准）" if passed
             else f"{stat}（注意：三栏对照臂噪声都没压住，下面的数只作方向）")
    a = a0[stat]
    print(f"[判词 Q4 兑现第1课的支票？] 用 {which} 口径："
          f"单张 基线 {a:.3f} → R50 {lrows['R50'][stat]:.3f} ms（{a / lrows['R50'][stat]:.2f}x）"
          f"、R50F {lrows['R50F'][stat]:.3f} ms（{a / lrows['R50F'][stat]:.2f}x）"
          f"，而 R50 只剩 {arms['R50 剪未微调']['macs_pct']}% 乘加")
    print(f"     批量 128 吞吐：基线 {a0['thr_batch128']:.0f} → R50 {lrows['R50']['thr_batch128']:.0f}"
          f"（{lrows['R50']['thr_batch128'] / a0['thr_batch128']:.2f}x）"
          f"、R50FQ {lrows['R50FQ']['thr_batch128']:.0f} 张/秒"
          f"（{lrows['R50FQ']['thr_batch128'] / a0['thr_batch128']:.2f}x）")
    print(f"     第1课说「要降单张延迟，剪枝才是答案」——本课读数是：{'兑现' if a / lrows['R50'][stat] > 1.2 else '没兑现'}。"
          f"乘加砍掉 {100 - arms['R50 剪未微调']['macs_pct']:.0f}%，单张只快 {100 * (1 - lrows['R50'][stat] / a):.0f}%，"
          f"因为延迟的 {100 - ceiling:.0f}% 是与乘加无关的固定开销（见上面的拟合）")

    report = {"archive_acc": ARCHIVE_ACC, "test_n": int(len(yte)), "train_fit_n": len(xt),
              "train_val_n": VAL_N, "torch": torch.__version__,
              "engine": torch.backends.quantized.engine,
              "scissors_calibration": {"weights_equal": bool(same_w), "pred_agree": agree},
              "channel_importance_cv": round(cv, 3), "macs_by_layer_base": m_base,
              "params_by_layer_base": p_base,
              "prior": {str(k): v for k, v in prior.items()},
              "mask_only_demo": md, "arms": arms, "latency_paired": lrows,
              "ctrl_floor_pct": floors, "latency_gate_pct": LAT_GATE_PCT, "latency_fit": fit,
              "latency_verdict": {k: ("可下结论" if v else "作废：对照臂噪声压不住")
                                  for k, v in ok.items()},
              "latency_runs": n_runs, "latency_run_ratio_p50": run_p50,
              "latency_run_ratio_best": run_best,
              "latency_fit_runs": spread}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n已存 compress_report_v3.json（10000 张全量测试，纯 CPU，零 API 费）")


if __name__ == "__main__":
    main()
