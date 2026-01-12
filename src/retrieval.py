import pickle
from typing import List, Tuple, Optional, Any

from rapidfuzz import process, fuzz
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.compressors import BaseDocumentCompressor
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from src.config import INDEX_PATH, SPLITS_PATH, RERANKER_MODEL
from src.models import get_embeddings


def load_resources() -> Tuple[VectorStore, List[Document], List[str]]:
    """
    Loads the Vector Store, Document Splits, and extracts known company names.

    Returns:
        Tuple[VectorStore, List[Document], List[str]]:
            - Loaded FAISS vector store.
            - List of Document objects for BM25.
            - List of unique company names found in metadata.
    """
    print("Loading resources...")
    embeddings = get_embeddings()

    vector_db = FAISS.load_local(
        folder_path=str(INDEX_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    with open(SPLITS_PATH, "rb") as f:
        new_splits: List[Document] = pickle.load(f)

    unique_companies = list(
        set([str(i.metadata.get("company_name", "")) for i in new_splits if i.metadata.get("company_name")])
    )

    return vector_db, new_splits, unique_companies


def get_company_match(target_company: str, known_companies: List[str]) -> Optional[str]:
    """
    Attempts to find a matching company name in the database using exact and fuzzy matching.

    Args:
        target_company (str): The company name extracted from the user's query.
        known_companies (List[str]): List of all company names existing in the database.

    Returns:
        Optional[str]: The normalized company name if a match is found (score > 60),
                       otherwise None.
    """
    if not target_company:
        return None

    companies_lower = {c.lower(): c for c in known_companies}
    if target_company.lower() in companies_lower:
        print(f"Exact match: target_company='{companies_lower[target_company.lower()]}'")
        return companies_lower[target_company.lower()]

    match = process.extractOne(target_company, known_companies, scorer=fuzz.token_sort_ratio)
    if match and match[1] > 60:
        print(f"Fuzzy match: '{target_company}' -> '{match[0]}' (Score: {match[1]})")
        return match[0]

    print(f"Company '{target_company}' not found in database.")
    return None


def create_retriever_pipeline(
    vector_db: VectorStore, documents: List[Document]
) -> Tuple[BaseRetriever, VectorStore, BaseDocumentCompressor]:
    """
    Initializes components for the retrieval pipeline.

    Args:
        vector_db (VectorStore): The loaded FAISS index.
        documents (List[Document]): The raw documents for BM25 initialization.

    Returns:
        Tuple[BaseRetriever, VectorStore, BaseDocumentCompressor]:
            - BM25 Retriever instance.
            - Vector Store instance (to be instantiated as retriever later with filters).
            - CrossEncoder Reranker compressor.
    """

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 50

    reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL, model_kwargs={"device": "cpu"})
    compressor = CrossEncoderReranker(model=reranker, top_n=50)

    return bm25_retriever, vector_db, compressor
