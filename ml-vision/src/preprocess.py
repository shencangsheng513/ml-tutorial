# -*- coding: utf-8 -*-
"""画布 → 模型输入的预处理（训练存档与 App 共用这一份，杜绝 train/serve skew）"""
import numpy as np
from PIL import Image, ImageOps


def canvas_to_input(img_rgba) -> np.ndarray | None:
    """RGBA 画布（白底黑笔）→ 8×8 float32 [0,1] 白字黑底；没画东西返回 None"""
    g = ImageOps.invert(img_rgba.convert("L"))      # 变白字黑底
    mask = np.array(g) > 40
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    g = g.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))
    side = max(g.size)                              # 补成正方形 = 等效居中
    sq = Image.new("L", (side, side), 0)
    off = ((side - g.size[0]) // 2, (side - g.size[1]) // 2)
    sq.paste(g, off)
    return np.asarray(sq.resize((8, 8), Image.BOX), dtype=np.float32) / 255.0
