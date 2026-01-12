import argparse
import sys
from src.utils import setup_system_dependencies, pull_ollama_model, ensure_directories
from src.config import OLLAMA_MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Financial RAG System CLI")

    parser.add_argument("--setup", action="store_true", help="Install system deps and pull Ollama model")
    parser.add_argument("--ingest", action="store_true", help="Run PDF ingestion (Docling + Metadata)")
    parser.add_argument("--index", action="store_true", help="Build FAISS index from ingested splits")
    parser.add_argument("--run", action="store_true", help="Run the RAG inference (Submission generation)")

    args = parser.parse_args()

    if not any(vars(args).values()):
        print("No arguments provided. Defaulting to --run.")
        args.run = True

    ensure_directories()

    if args.setup:
        print("=== Setup Phase ===")
        setup_system_dependencies()
        pull_ollama_model(OLLAMA_MODEL)

    if args.ingest:
        print("=== Ingestion Phase ===")
        from src.ingestion import run_ingestion

        run_ingestion()

    if args.index:
        print("=== Indexing Phase ===")
        from src.indexing import build_vector_index

        build_vector_index()

    if args.run:
        print("=== Inference Phase ===")
        from src.generation import run_inference

        from .inference_runner import run_pipeline

        run_pipeline()


if __name__ == "__main__":
    main()
