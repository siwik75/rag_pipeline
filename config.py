"""Configurazione centralizzata della pipeline RAG."""
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass  # python-dotenv non installato: si usano solo le env var di shell

# --- Percorsi ---
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT.parent                      # cartella con i PDF
EXTRACTED_DIR = PROJECT_ROOT / "extracted"         # archivio lossless .md
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"   # ChromaDB persistente
MANIFEST_PATH = PROJECT_ROOT / "ingest_manifest.json"

# --- Vector store ---
COLLECTION_NAME = "trading_knowledge"

# --- Embedding (locale via fastembed/ONNX, niente torch) ---
# Multilingue: query in italiano su testi in inglese.
# Per massima qualita': "intfloat/multilingual-e5-large" (~2.2 GB).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# I modelli E5 richiedono prefissi per passage/query.
IS_E5 = "e5" in EMBEDDING_MODEL.lower()
PASSAGE_PREFIX = "passage: " if IS_E5 else ""
QUERY_PREFIX = "query: " if IS_E5 else ""

# --- Chunking ---
CHUNK_SIZE = 1200        # caratteri
CHUNK_OVERLAP = 250      # caratteri

# --- OCR ---
OCR_LANGS = "eng+ita"    # lingue tesseract
MIN_CHARS_PER_PAGE = 40  # sotto questa soglia la pagina si considera scansionata

# --- Query / LLM ---
TOP_K = 6
CLAUDE_MODEL = "claude-sonnet-5"
