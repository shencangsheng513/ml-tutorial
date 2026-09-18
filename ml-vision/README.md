# ml-vision · 手写数字识别（5 课 + 上线）

从 8×8 玩具图到 28×28 真 MNIST：CNN 的边界与正名。

**课程轨迹**
1. EDA：像素矩阵、死角落、t-SNE 定性预言（只许怀疑不许定量）
2. 经典模型：SVM 98.0%；t-SNE 预言 0/6 落空的诚实对质
3. MLP vs CNN：8×8 居中图上 CNN 没赢（0.980 vs 0.982）——先验不值钱；
   增强三臂证伪：±1px 平移让 200 张图从 94.2% 砸到 85.8%（亲手制造 train/serve skew）
4. 上线：像素格子+拍照上传双模式 App（端口 8504），预处理单测 0.9622 过账
5. MNIST 正名之战：真战场 CNN 应甩开 MLP——`step4_mnist.py`（见 mnist_showdown.png）

**运行**：`python src/train_cnn.py` 存档模型→`launch_vision_app.bat` 起 App
