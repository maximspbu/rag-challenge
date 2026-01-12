import gc
import torch
import subprocess
import os
from pathlib import Path


def setup_system_dependencies() -> None:
    """
    Installs necessary system packages (simulating apt install commands).
    """
    try:
        print("Updating apt and installing pciutils...")
        subprocess.run(["sudo", "apt", "update"], check=False)
        subprocess.run(["sudo", "apt", "install", "-y", "pciutils"], check=False)
    except Exception as e:
        print(f"Warning: Could not install system dependencies: {e}")


def pull_ollama_model(model_name: str) -> None:
    """
    Pulls the specified model using Ollama CLI.
    """
    print(f"Pulling Ollama model: {model_name}...")
    subprocess.run(["ollama", "pull", model_name], check=True)


def cleanup_memory() -> None:
    """
    Aggressively cleans up GPU memory and garbage collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("Memory cleaned.")


def ensure_directories() -> None:
    """Creates necessary data directories."""
    from src.config import DATA_DIR, PDF_DIR, INDEX_PATH

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_PATH, exist_ok=True)
