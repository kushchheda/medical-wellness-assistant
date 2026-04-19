"""
src/helper.py

Utility functions for the Medical Wellness Assistant RAG pipeline.

Covers:
  - Loading and parsing PDF documents from a directory
  - Splitting documents into overlapping text chunks
  - Downloading and initializing HuggingFace sentence embeddings
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Default path to the folder containing medical PDF documents
DATA_PATH = Path(__file__).resolve().parent.parent / "medical_data"

# Embedding model — produces 384-dimensional dense vectors
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_files(data_path: Path = DATA_PATH) -> list:
    """
    Load all PDF files from the specified directory.

    Args:
        data_path: Path to the directory containing PDF files.
                   Defaults to the project-level 'medical_data/' folder.

    Returns:
        A list of LangChain Document objects extracted from the PDFs.

    Raises:
        FileNotFoundError: If the specified data path does not exist.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    logger.info(f"Loading PDFs from: {data_path}")
    loader = DirectoryLoader(
        str(data_path),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document pages.")
    return documents


def split_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 20) -> list:
    """
    Split a list of documents into smaller text chunks for embedding.

    Args:
        documents:     List of LangChain Document objects to split.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks
                       to preserve context across boundaries.

    Returns:
        A list of smaller LangChain Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} text chunks.")
    return chunks


def download_hugging_face_embeddings(
    model_name: str = EMBEDDING_MODEL_NAME
) -> HuggingFaceEmbeddings:
    """
    Load and return a HuggingFace sentence-transformer embedding model.

    The default model (all-MiniLM-L6-v2) produces 384-dimensional vectors
    and is well-suited for semantic similarity tasks.

    Args:
        model_name: HuggingFace model identifier. Defaults to
                    'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        An initialized HuggingFaceEmbeddings instance.
    """
    logger.info(f"Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
