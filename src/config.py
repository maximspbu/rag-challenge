import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")


BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
PDF_DIR: Path = Path(os.getenv("PDF_DIR", str(DATA_DIR / "pdfs")))


SPLITS_PATH: Path = Path(os.getenv("SPLITS_PATH", str(DATA_DIR / "new_splits.pickle")))
INDEX_PATH: Path = Path(os.getenv("INDEX_PATH", str(DATA_DIR / "my_faiss_index")))
QUESTIONS_PATH: Path = Path(os.getenv("QUESTIONS_PATH", str(DATA_DIR / "questions.json")))
OUTPUT_FILE: Path = Path(os.getenv("OUTPUT_FILE", "sample_answer.json"))


OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL: str = "BAAI/bge-reranker-base"


TEAM_EMAIL: str | None = os.getenv("TEAM_EMAIL")
SUBMISSION_NAME: str | None = os.getenv("SUBMISSION_NAME")
SUBMISSION_URL: str | None = os.getenv("SUBMISSION_URL")
