# ml-car-price · 加州房价端到端回归

Pipeline 化的回归项目：EDA→清洗→特征工程→调参→误差分析→Streamlit 上线。

**亮点**
- 先切分再处理，所有统计量只在训练折 fit（结构防泄漏）
- 3 个地理特征（距海岸线/两城市距离）RMSE 0.4437→0.4229，收益超过一切调参
- 增益重要性 vs 置换重要性打架，用误差分片裁决：地理比收入重要
- 上线 App 直接吃 8 个原始字段，消除 train/serve skew

**运行**：`python src/train.py` 重训；App 见 `app.py`
