import tempfile
from pathlib import Path
from typing import Any, Dict

from langchain_core.documents import Document

from backend.services.indexing_service import VectorDBIngestor
from backend.services.metadata_service import normalize_metadata
from backend.services.document_service import register_document
from backend.services.parsed_reports_merging import PageTextPreparation
from backend.services.pdf_parsing import PDFParser
from backend.services.tables_serialization import TableSerializer
from backend.services.text_splitter import TextSplitter


def ingest_pdf(filename: str, file_bytes: bytes, metadata: Dict[str, Any]) -> Dict[str, int]:
    if not file_bytes:
        raise ValueError("Uploaded PDF is empty")
    filename = Path(filename).name
    if not filename or filename.lower().endswith(".pdf") is False:
        raise ValueError("filename must be a PDF filename")

    metadata = normalize_metadata(metadata)
    # Keep transient files inside the application workspace; some hosted
    # environments deny access to the system temp directory.
    with tempfile.TemporaryDirectory(prefix=".rag_ingest_", dir=str(Path.cwd())) as temp:
        root = Path(temp)
        source = root / filename
        source.write_bytes(file_bytes)
        parsed_dir = root / "parsed"
        prepared_dir = root / "prepared"

        parsed_path = parsed_dir / f"{source.stem}.json"
        try:
            parser = PDFParser(output_dir=parsed_dir)
            parser.parse_and_export([source])
        except RuntimeError as exc:
            if "docling" not in str(exc).lower():
                raise
            # Minimal parser fallback for deployments that only install PyMuPDF.
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            try:
                pages = [
                    {"page": index, "content": [{"type": "text", "text": page.get_text() or ""}]}
                    for index, page in enumerate(doc, 1)
                ]
            finally:
                doc.close()
            parsed_path.parent.mkdir(parents=True, exist_ok=True)
            parsed_path.write_text(__import__("json").dumps({
                "metainfo": {"sha1_name": source.stem, "pages_amount": len(pages)},
                "content": pages,
                "tables": [],
                "pictures": [],
            }), encoding="utf-8")

        # Preserve parser tables, optionally enrich them with LLM serialization, then
        # render page text using the enriched table representation.
        raw = __import__("json").loads(parsed_path.read_text(encoding="utf-8"))
        if raw.get("tables"):
            raw = TableSerializer().serialize_tables(raw)
        preparation = PageTextPreparation(use_serialized_tables=True, serialized_tables_instead_of_markdown=False)
        raw["content"] = preparation.process_report(raw)
        splitter = TextSplitter()
        raw = splitter._split_report(raw)

        documents = []
        for chunk in raw.get("content", {}).get("chunks", []):
            chunk_meta = {
                **metadata,
                "source": filename,
                "page": chunk.get("page"),
                "chunk_id": str(chunk.get("id")),
                "chunk_type": chunk.get("chunk_type", chunk.get("type", "text")),
                "document_id": source.stem,
            }
            documents.append(Document(page_content=chunk.get("text", ""), metadata=chunk_meta))

        if not documents:
            raise ValueError("The PDF did not contain indexable text")
        vector = VectorDBIngestor()
        # Replace an older version of the same file to keep re-upload idempotent.
        vector.delete_document(filename)
        total_chunks = vector.add_documents(documents)
        register_document(filename, total_chunks)
        return {
            "total_chunks": total_chunks,
            "total_pages": len(raw.get("content", {}).get("pages", [])),
            "total_tables": len(raw.get("tables", [])),
        }
