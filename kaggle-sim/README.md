# kaggle-sim · Kaggle 比赛机制模拟器

不注册账号也能把公榜/私榜陷阱亲手跑一遍（Titanic 887 行，本地搭榜）。

**三轮提交**
- ① 裸RF 0.768 → 彩蛋：只喂 female 一列 0.831 反超（调不好的模型不如一列好特征）
- ② 特征工程（Name 挖称谓）+ 5 折 CV 0.812 → 公榜 0.853 私榜 0.826（CV 预言成功）
- ③A 用公榜标签训练（违禁）：私榜也涨——这就是必须禁止的原因
- ③B submit 登山：400 次提交换来公榜 0.972，私榜纹丝不动——涨幅全是背答案

**运行**：`python compete.py`（秒级）

## 打真赛清单（就绪待发）
1. kaggle.com 注册 → Account 页生成 API token → 存 `C:\Users\shenh\.kaggle\kaggle.json`
2. `pip install kaggle` 后 `kaggle competitions download -c titanic`
3. 纪律：测试集只在结案时看一次；日常信任全用 train 池 CV；每天提交≤2 次
4. 推荐首赛：titanic → 之后再打 house-prices（正好对上项目1的表格技能）
