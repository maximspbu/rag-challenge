import os
import pickle
from typing import List, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

from pydantic import BaseModel, Field

from src.config import PDF_DIR, SPLITS_PATH, OLLAMA_MODEL
from src.models import get_llm
from src.utils import cleanup_memory


# --- Metadata Extraction Schema ---
class FileMetaData(BaseModel):
    company_name: str = Field(
        description="The exact name of the company this annual report belongs to, if there is no name, return 'Unknown'"
    )


def extract_metadata_from_doc(doc_content: str, llm_model) -> FileMetaData:
    """
    Uses LLM to extract company name from the document snippet.
    """
    metadata_extractor = llm_model.with_structured_output(FileMetaData)
    snippet = doc_content[:2000]
    prompt = (
        f"Analyze the following text from an annual report cover page and extract the Company Name.\n\nTEXT:\n{snippet}"
    )
    try:
        res = metadata_extractor.invoke(prompt)
        return res
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return FileMetaData(company_name="Unknown")


def run_ingestion() -> None:
    """
    Main function to process PDFs:
    1. Configures Docling with OCR and Table Structure.
    2. Loads and chunks PDFs.
    3. Extracts Company Names via LLM.
    4. Saves the splits to pickle.
    """
    print("Starting Ingestion Process...")

    # 1. Setup Docling Converter
    accelerator_options = AcceleratorOptions(num_threads=4)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    chunker = HybridChunker(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=1000,
    )

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )

    # 2. Identify Files
    if not os.path.exists(PDF_DIR):
        print(f"Error: PDF directory {PDF_DIR} does not exist.")
        return

    file_paths = sorted([os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])

    print(f"Found {len(file_paths)} PDFs to process.")

    # 3. Load and Chunk
    loader = DoclingLoader(
        file_paths,
        converter=converter,
        chunker=chunker,
    )

    print("Lazy loading documents...")
    docs = list(loader.lazy_load())  # Convert iterator to list to process

    print(f"Generated {len(docs)} chunks. extracting metadata...")

    # 4. Enrich Metadata (Company Name)
    llm = get_llm(model_name=OLLAMA_MODEL)

    # We create a cache for source -> company_name to avoid re-querying for every chunk of the same file
    source_to_company: dict[str, str] = {}

    processed_docs: List[Document] = []

    for doc in tqdm(docs, desc="Enriching Metadata"):
        # Normalize metadata
        try:
            # Docling specific metadata access
            dl_meta = doc.metadata.get("dl_meta", {})
            origin = dl_meta.get("origin", {})
            filename = origin.get("filename", "unknown")

            # Page index logic from original script
            try:
                page_no = dl_meta["doc_items"][0]["prov"][0]["page_no"]
                page_index = page_no - 1
            except:
                page_index = 0

            doc.metadata["filename"] = filename
            doc.metadata["source"] = filename
            doc.metadata["page_index"] = page_index

            # Company Name Extraction
            if filename not in source_to_company:
                meta = extract_metadata_from_doc(doc.page_content, llm)
                source_to_company[filename] = meta.company_name

            doc.metadata["company_name"] = source_to_company[filename]

            processed_docs.append(doc)

        except Exception as e:
            print(f"Error processing doc chunk: {e}")
            continue

    # 5. Save Splits
    print(f"Saving {len(processed_docs)} splits to {SPLITS_PATH}...")
    with open(SPLITS_PATH, "wb") as f:
        pickle.dump(processed_docs, f)

    cleanup_memory()
    print("Ingestion complete.")
