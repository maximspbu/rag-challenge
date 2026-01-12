from typing import Dict, Any, Callable, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.compressors import BaseDocumentCompressor

from src.models import get_llm
from src.schemas import Answer, ReformulatedQuery, SearchQuery
from src.retrieval import get_company_match

llm = get_llm()


reformulate_query_prompt = """You are an expert in Financial Information Retrieval.
Rewrite the user's raw question into a targeted, keyword-rich semantic query.
1. STRIP Company Names.
2. REMOVE Constraints (e.g. "return 'N/A'").
3. REMOVE Temporal Noise.
4. EXPAND Terminology (e.g. "let go" -> "redundancy, severance").
Output ONLY the rewritten query string.
User Input:
"""
reformulator = llm.with_structured_output(ReformulatedQuery)


def reformulate_query(query: str) -> ReformulatedQuery:
    """
    Rewrites the raw user query to be more suitable for vector search.
    """
    prompt = reformulate_query_prompt + "\n" + query
    res = reformulator.invoke(prompt)
    return res


query_analyzer = llm.with_structured_output(SearchQuery)


def build_retrieve_fn(
    vector_db: VectorStore, bm25_retriever: BaseRetriever, compressor: BaseDocumentCompressor, known_companies: List[str]
) -> Callable[[Dict[str, Any]], str]:
    """
    Builds a retrieval function that encapsulates the logic for company extraction,
    ensemble retrieval (BM25+Vector), filtering, and reranking.

    Args:
        vector_db (VectorStore): The vector store.
        bm25_retriever (BaseRetriever): The keyword retriever.
        compressor (BaseDocumentCompressor): The reranker.
        known_companies (List[str]): List of valid companies for filtering.

    Returns:
        Callable[[Dict[str, Any]], str]: A function taking input dict with 'question'
                                         and returning a serialized string of contexts.
    """

    def retrieve(inputs: Dict[str, Any]) -> str:
        user_question = inputs["question"]

        analysis = query_analyzer.invoke(f"Extract company name and refine query: {user_question}")
        target_company = analysis.extracted_company
        best_match_name = get_company_match(target_company, known_companies)

        vect_kwargs: Dict[str, Any] = {"k": 50, "fetch_k": 1000}
        if best_match_name:
            vect_kwargs["filter"] = {"company_name": best_match_name}

        vector_retriever = vector_db.as_retriever(search_type="mmr", search_kwargs=vect_kwargs)

        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.3, 0.7])

        refined_query = reformulate_query(user_question).reformulated_query
        print(f"Refined Query: {refined_query}")

        docs: List[Document] = ensemble_retriever.invoke(refined_query)

        if best_match_name:
            docs = [d for d in docs if d.metadata.get("company_name") == best_match_name]

        compressed_docs = compressor.compress_documents(documents=docs, query=refined_query)

        if compressed_docs:
            compressed_docs = compressed_docs[:10]

        serialized = "\n\n".join(
            (
                f"--- DOCUMENT START ---\n"
                f"Filename: {doc.metadata.get('source', 'unknown')}\n"
                f"Page Index: {doc.metadata.get('page_index', 0)}\n"
                f"Metadata: {doc.metadata}\n"
                f"Content: {doc.page_content}\n"
                f"--- DOCUMENT END ---\n"
            )
            for doc in compressed_docs
        )
        return serialized

    return retrieve


system_prompt = """You are a precise financial analyst extracting data from annual reports.
Your task is to answer the user's question using ONLY the provided context documents.

RULES:
1. Extract value exactly. 'kind'='number' -> float. 'kind'='boolean' -> true/false.
2. If info is NOT in Context, return value="N/A".
3. Populate 'references' list strictly from Context headers.
4. JSON Output strictly adhering to schema.
"""


def create_rag_chain(retrieve_fn: Callable[[Dict[str, Any]], str]) -> Runnable:
    """
    Constructs the final LCEL chain for RAG.

    Args:
        retrieve_fn (Callable): The function responsible for retrieving context.

    Returns:
        Runnable: A LangChain runnable that accepts {'question': str, 'kind': str}
                  and outputs an Answer object.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "Question: {question}\nOutput Kind: {kind}\n\nCONTEXT:\n{context}")]
    )

    llm_structured = llm.with_structured_output(Answer)

    chain = (
        {"context": retrieve_fn, "question": lambda x: x["question"], "kind": lambda x: x["kind"]}
        | prompt_template
        | llm_structured
    )
    return chain
