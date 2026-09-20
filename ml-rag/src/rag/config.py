"""共享配置：凭证走 ~/.bailian 官方存储或环境变量，仓库零明文。"""
import json
import os
from pathlib import Path


def load_credentials() -> tuple[str, str]:
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("LLM_BASE_URL")          # 显式给了就以它为准，排障换端点才有效
    if not key:
        cfg = Path.home() / ".bailian" / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text(encoding="utf-8"))
            key = c.get("api_key")
            if base is None and c.get("base_url"):
                base = c["base_url"].rstrip("/") + "/compatible-mode/v1"
    if base is None:
        base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not key:
        raise SystemExit("缺少凭证：bl auth login --api-key sk-... 或设环境变量")
    return key, base


CHAT_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")
EMBED_MODEL = os.environ.get("LLM_EMBED_MODEL", "qwen3.7-text-embedding")
ROOT = Path(__file__).resolve().parents[3]          # ml-tutorial/
KB_GLOBS = ["*.md", "*/README.md", "ml-text/*.md", "ml-vision/*.md", "ml-timeseries/*.md",
            "ml-car-price/*.md", "kaggle-sim/*.md", "mlflow-lab/*.md"]
INDEX_DIR = Path(__file__).resolve().parents[2] / "index"
