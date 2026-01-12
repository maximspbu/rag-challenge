from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import OLLAMA_MODEL, EMBEDDING_MODEL


def get_llm(model_name: str = OLLAMA_MODEL, temperature: float = 0, num_ctx: int = 16384) -> BaseChatModel:
    return ChatOllama(model=model_name, temperature=temperature, num_ctx=num_ctx, num_gpu=-1)


def get_embeddings() -> Embeddings:
    """
    Initializes HuggingFace Embeddings model.

    Returns:
        Embeddings: Configured HuggingFaceEmbeddings instance running on CPU (usually)
                    or GPU depending on internal torch settings, configured for remote code.
    """
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 32}

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
