import os
from dotenv import load_dotenv, find_dotenv

# Nạp .env chắc chắn từ thư mục gốc dự án, cho phép override biến môi trường
load_dotenv(find_dotenv(usecwd=True), override=True)


class Settings:

    # Đường dẫn model đơn (giữ tương thích ngược)
    model_path: str = os.getenv("MODEL_PATH", "train1.pkl")

    # Danh sách nhiều model: hỗ trợ MODEL_PATHS (JSON hoặc CSV) và MODEL_PATH2 (tuỳ chọn)
    _raw_paths: str | None = os.getenv("MODEL_PATHS")
    _path2: str | None = os.getenv("MODEL_PATH2")
    if _raw_paths:
        try:
            import json as _json
            parsed = _json.loads(_raw_paths)
            if isinstance(parsed, list):
                model_paths = [str(p) for p in parsed if str(p).strip()]
            else:
                model_paths = [p.strip() for p in _raw_paths.split(",") if p.strip()]
        except Exception:
            model_paths = [p.strip() for p in _raw_paths.split(",") if p.strip()]
    else:
        model_paths = [model_path]
        if _path2 and _path2.strip():
            model_paths.append(_path2.strip())

    model_feature_order: str | None = os.getenv("MODEL_FEATURE_ORDER")
    model_classes: str | None = os.getenv("MODEL_CLASSES")

    # ✅ Thay OpenAI bằng Gemini
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    # ✅ Tavily API
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")

    # PostgreSQL / Redis / Pinecone
    postgres_url: str | None = os.getenv("POSTGRES_URL")
    redis_url: str | None = os.getenv("REDIS_URL")
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")
    pinecone_index: str | None = os.getenv("PINECONE_INDEX")


settings = Settings()
