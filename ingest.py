"""Ingestione lossless dei PDF in ChromaDB.

Per ogni PDF:
1. Estrae testo pagina per pagina (PyMuPDF) + tabelle in Markdown.
2. Se una pagina ha poco testo, tenta OCR (pytesseract, se installato).
3. Salva l'intero testo estratto in extracted/<nome>.md (archivio lossless).
4. Divide in chunk con overlap, preservando i confini di pagina nei metadati.
5. Genera embedding locali e li inserisce in ChromaDB (persistente).

Idempotente: i file gia' ingeriti (stesso hash SHA-256) vengono saltati.
Uso: python ingest.py [--force]
"""
import argparse
import hashlib
import json
import sys
import unicodedata

import fitz  # PyMuPDF
import chromadb
from fastembed import TextEmbedding

import config

try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "\n".join(line.rstrip() for line in text.splitlines())


def ocr_page(page) -> str:
    if not OCR_AVAILABLE:
        return ""
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(img, lang=config.OCR_LANGS)
    except Exception as e:
        print(f"    [warn] OCR fallito: {e}")
        return ""


def tables_to_markdown(page) -> str:
    """Estrae le tabelle della pagina come Markdown (perdita zero di struttura)."""
    out = []
    try:
        for t in page.find_tables():
            md = t.to_markdown()
            if md.strip():
                out.append(md.strip())
    except Exception:
        pass
    return "\n\n".join(out)


def extract_pdf(path):
    """Ritorna lista di (numero_pagina, testo) e il testo completo."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = clean(page.get_text("text"))
        if len(text.strip()) < config.MIN_CHARS_PER_PAGE:
            ocr_text = clean(ocr_page(page))
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
                print(f"    pagina {i}: usato OCR")
        tables = tables_to_markdown(page)
        if tables:
            text = f"{text}\n\n[TABELLE PAGINA {i}]\n{tables}"
        pages.append((i, text))
    doc.close()
    return pages


def chunk_pages(pages, source):
    """Chunk con overlap che rispetta i paragrafi; metadati con range di pagine."""
    # Costruisce una sequenza di (paragrafo, pagina)
    paras = []
    for pageno, text in pages:
        for p in text.split("\n\n"):
            p = p.strip()
            if p:
                paras.append((p, pageno))

    chunks = []
    buf, buf_pages = "", []
    for p, pageno in paras:
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) > config.CHUNK_SIZE and buf:
            chunks.append((buf, min(buf_pages), max(buf_pages)))
            # overlap: riparte dalla coda del chunk precedente
            tail = buf[-config.CHUNK_OVERLAP:]
            buf = f"{tail}\n\n{p}"
            buf_pages = [buf_pages[-1], pageno]
        else:
            buf = candidate
            buf_pages.append(pageno)
    if buf.strip():
        chunks.append((buf, min(buf_pages), max(buf_pages)))

    return [
        {
            "id": f"{source}::chunk_{idx}",
            "text": text,
            "metadata": {
                "source": source,
                "page_start": ps,
                "page_end": pe,
                "chunk_index": idx,
            },
        }
        for idx, (text, ps, pe) in enumerate(chunks)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-ingerisce tutto")
    args = ap.parse_args()

    config.EXTRACTED_DIR.mkdir(exist_ok=True)
    manifest = {}
    if config.MANIFEST_PATH.exists() and not args.force:
        manifest = json.loads(config.MANIFEST_PATH.read_text())

    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nessun PDF trovato in {config.PDF_DIR}")

    print(f"Modello embedding: {config.EMBEDDING_MODEL}")
    model = TextEmbedding(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    for pdf in pdfs:
        digest = sha256(pdf)
        if manifest.get(pdf.name) == digest:
            print(f"[skip] {pdf.name} (gia' ingerito)")
            continue

        print(f"[ingest] {pdf.name}")
        pages = extract_pdf(pdf)
        total_chars = sum(len(t) for _, t in pages)
        print(f"    {len(pages)} pagine, {total_chars} caratteri estratti")

        # Archivio lossless
        md_path = config.EXTRACTED_DIR / f"{pdf.stem}.md"
        md_path.write_text(
            f"# {pdf.name}\n\n"
            + "\n\n".join(f"## Pagina {n}\n\n{t}" for n, t in pages),
            encoding="utf-8",
        )

        chunks = chunk_pages(pages, pdf.name)
        print(f"    {len(chunks)} chunk")

        # Rimuove eventuali chunk precedenti dello stesso file
        collection.delete(where={"source": pdf.name})

        texts = [config.PASSAGE_PREFIX + c["text"] for c in chunks]
        embeddings = [e.tolist() for e in model.embed(texts, batch_size=32)]
        collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

        manifest[pdf.name] = digest
        config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

    print(f"\nCompletato. Totale chunk in collezione: {collection.count()}")
    print(f"Vector store: {config.VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
