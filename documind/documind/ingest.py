"""Load PDFs from data/, split them into chunks, store them in Chroma."""
import hashlib
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from documind import config

COLLECTION_NAME = "documents"


def get_collection():
    """Open (or create) the Chroma collection saved on disk."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # measure similarity by angle
    )


def load_pdf(path: Path) -> list[dict]:
    """Return one dict per page that contains text."""
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # skip blank or image-only pages
            pages.append({"source": path.name, "page": page_number, "text": text})
    return pages  # <- same indentation as "for": runs after ALL pages are read


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split each page into overlapping chunks, keeping source and page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = []
    for page in pages:
        for index, piece in enumerate(splitter.split_text(page["text"])):
            raw_id = f"{page['source']}-{page['page']}-{index}"
            chunks.append(
                {
                    "id": hashlib.md5(raw_id.encode()).hexdigest(),
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
    return chunks  # <- same indentation as "for page": runs after ALL pages


def ingest(data_dir: Path = config.DATA_DIR) -> int:
    """Index every PDF in data_dir. Returns the number of chunks stored."""
    pdfs = sorted(Path(data_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    collection = get_collection()
    total = 0
    for pdf in pdfs:
        chunks = chunk_pages(load_pdf(pdf))
        if not chunks:
            print(f"Skipped {pdf.name}: no extractable text (scanned PDF?)")
            continue

        # Remove old chunks of this file first, so edited PDFs never leave stale data.
        collection.delete(where={"source": pdf.name})
        collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        )
        print(f"Indexed {pdf.name}: {len(chunks)} chunks")
        total += len(chunks)
    return total


if __name__ == "__main__":
    count = ingest()
    print(f"Done. {count} chunks stored in {config.CHROMA_DIR}")