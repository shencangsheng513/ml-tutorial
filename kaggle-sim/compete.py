# -*- coding: utf-8 -*-
"""
Kaggle 实战模拟： Titanic 沉顶生存预测 + 公榜/私榜陷阱体验
真实 Kaggle 规则：测试集切成公榜(约占一半,提交立刻出分)+私榜(比赛结束才见)。
排行榜只按公榜排名 → 有人疯狂"试探公榜"(把提交当验证集反复改),公榜分暴涨,
比赛结束私榜一出,原形毕露直接坠机。本课把这套机制完整搬到本地跑一遍：
  我们训练用 train 池；把 355 张测试牌分成 public/private 两半。
  三轮提交：① 裸模型 ② 特征工程 ③ 作弊(直接拿 public 的标签训练)
  看公榜私榜如何打出两个平行宇宙。
数据：web.stanford.edu 的 titanic.csv（887行教学版，无 Cabin/Embarked 列，
     字段和官方有差异——真实比赛第一步永远是 df.columns 认字段！）
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

rng = np.random.RandomState(42)
df = pd.read_csv("raw_titanic.csv")
print("== 第0步：认字段（真实比赛第一步） ==")
print("列:", list(df.columns))
print("缺失值:\n", df.isna().sum()[lambda s: s > 0].to_string() or "  (无)")
print(f"生存率基线: {df.Survived.mean():.3f} → 全猜'死'就有 {1-df.Survived.mean():.3f} 分\n")

# ---- 比赛场地搭建：train池 / public榜 / private榜 ----
train_pool, holdout = train_test_split(df, test_size=0.4, random_state=42, stratify=df.Survived)
public, private = train_test_split(holdout, test_size=0.5, random_state=7, stratify=holdout.Survived)
print(f"train池 {len(train_pool)} | 公榜 {len(public)} | 私榜 {len(private)}\n")


def features(d):
    X = pd.DataFrame()
    X["pclass"] = d.Pclass
    X["female"] = (d.Sex == "female").astype(int)
    X["age"] = d.Age.fillna(d.Age.median()) if d.Age.isna().any() else d.Age
    X["fam_size"] = d["Siblings/Spouses Aboard"] + d["Parents/Children Aboard"] + 1
    X["log_fare"] = np.log1p(d.Fare)
    return X


Xtr, ytr = features(train_pool), train_pool.Survived
Xpu, ypu = features(public), public.Survived
Xpr, ypr = features(private), private.Survived


def submit(name, model, extra="", cols=None, Xp=None, Xq=None):
    """按 Kaggle 流程：对 public 和 private 分别出分，并排看"""
    Xp = Xpu if Xp is None else Xp
    Xq = Xpr if Xq is None else Xq
    f = (lambda X: X[cols]) if cols else (lambda X: X)
    s_pub = accuracy_score(ypu, model.predict(f(Xp)))
    s_pri = accuracy_score(ypr, model.predict(f(Xq)))
    print(f"{name:38s} 公榜 {s_pub:.4f}  私榜 {s_pri:.4f}  {extra}")
    return s_pub, s_pri


# ============================================================
# 第1轮：裸模型（不调参不特征，先占个坑——Kaggle 第一天都这么干）
# ============================================================
print("== 三轮提交 ==")
m1 = RandomForestClassifier(random_state=42)
m1.fit(Xtr, ytr)
submit("① 裸RF(默认参数)", m1, "← 有性别就够了？女性优先是真的")

# 只看性别的地板分
sex_only = pd.DataFrame({"female": Xtr.female})
m_sex = RandomForestClassifier(random_state=42).fit(sex_only, ytr)
submit("   (对照：只喂 female 一列)", m_sex, "← 特征的地板，任何模型得分减它就是特征工程的功劳", cols=["female"])

# ============================================================
# 第2轮：正经特征工程 + 交叉验证（Name 里挖 Title！）
# ============================================================


def features2(d):
    X = features(d)
    # Name 里藏着一个人口学金矿：称谓。Mr/Mrs/Miss 之外还有 Master(小男孩)、
    # Officer/Rev(职业)、Lady/Don(阶层)。Kaggle 上这是 Titanic 第一课的招牌操作
    t = d.Name.str.extract(r",\s*([A-Za-z]+)\.")[0]
    X["title_Master"] = (t == "Master").astype(int)
    X["title_Officer"] = t.isin(["Col", "Major", "Dr", "Rev", "Capt"]).astype(int)
    X["title_Royalty"] = t.isin(["Lady", "Sir", "Don", "Jonkheer", "Countess"]).astype(int)
    X["child"] = (X["age"] < 12).astype(int)
    X["alone"] = (X["fam_size"] == 1).astype(int)
    return X


Xtr2, Xpu2, Xpr2 = features2(train_pool), features2(public), features2(private)
m2 = RandomForestClassifier(400, min_samples_leaf=3, random_state=42)
cv = cross_val_score(m2, Xtr2, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=1), scoring="accuracy")
m2.fit(Xtr2, ytr)
submit("② 特征工程+调参的RF", m2, f"← train池CV={cv.mean():.3f}±{cv.std():.3f}（我们的内部排行榜）", Xp=Xpu2, Xq=Xpr2)

# ============================================================
# 第3轮A：作弊——把公榜的标签也塞进训练集（拿到测试标签，直接取消资格级）
# ============================================================
X3tr = pd.concat([Xtr2, Xpu2])
y3tr = pd.concat([ytr, ypu])
m3 = RandomForestClassifier(400, min_samples_leaf=3, random_state=42).fit(X3tr, y3tr)
s_pub, s_pri = submit("③A 用公榜标签训练(违禁)", m3, "← 注意私榜竟然也涨了", Xp=Xpu2, Xq=Xpr2)
print(f"   为什么还能涨？公榜标签是'真答案'——这就是Kaggle禁止触碰测试标签的原因：")
print(f"   它确实'有效'，有效到不罚没人会守规矩。公榜 {s_pub:.3f} 里约 {(s_pub-0.85)*100:.0f}% 是偷来的。")

# ============================================================
# 第3轮B：合法的死亡螺旋—— submit 登山（每提交一次就微调一点预测）
# ============================================================
pred_pub = np.array(m2.predict(Xpu2), dtype=int)
best = accuracy_score(ypu, pred_pub)
n_flips = 0
for _ in range(400):                       # 模拟400次提交
    i = rng.randint(len(pred_pub))
    trial = pred_pub.copy(); trial[i] ^= 1  # 随机改一个人的生死预测
    s = accuracy_score(ypu, trial)
    if s > best:
        pred_pub, best, n_flips = trial, s, n_flips + 1
print(f"\n③B submit登山：{n_flips} 次'幸运'改动后 公榜 {best:.4f}")
pred_pr = np.array(m2.predict(Xpr2), dtype=int)   # 私榜预测完全没动
print(f"   但私榜还是 {accuracy_score(ypr, pred_pr):.4f} —— 一分钱没学到，公榜分数全是背答案")
print("   真实比赛里这就是'每天改改提交刷公榜'的选手：公榜一路绿灯，结案坠机")

# ============================================================
# 收尾：生成 Kaggle 格式的提交文件（PassengerId,Survived）
# ============================================================
sub = pd.DataFrame({"PassengerId": np.arange(1, len(private) + 1),
                    "Survived": m2.predict(Xpr2).astype(int)})
sub.to_csv("submission_round2.csv", index=False)
print("\n已生成 submission_round2.csv（真实比赛要上传的就是这种文件）")
print("首3行:\n", sub.head(3).to_string(index=False))
print("""
== 本课复盘 ==
1. 公榜≈一半测试集 → 只有 ~178 个样本的'验收集'，反复试探必然过拟合它。
2. 防御武器就是 train 池上的交叉验证——内部排行榜才是真话。
3. Kaggle 高阶玩法'public leaderboard 反推标签'真实存在且被官方禁止（多账号提交）。
4. 永远留一份自己知道、绝不参与任何调参的 holdout，对应我们每课的最后一步。""")
