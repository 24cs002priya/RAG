import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.models.schemas import QAResponse, QuestionRequest, UploadResponse
from backend.services.document_service import delete_document, get_all_documents
from backend.services.ingestion_service import ingest_pdf
from backend.services.rag_service import answer_question

router = APIRouter(prefix="/rag", tags=["rag"])
logger = logging.getLogger(__name__)


@router.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    try:
        return get_all_documents()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/documents/{filename}")
async def delete_document_endpoint(filename: str):
    try:
        deleted = delete_document(filename)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Successfully deleted {filename}"}


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    document_type: Optional[str] = None,
    industry: Optional[str] = None,
    client: Optional[str] = None,
    project: Optional[str] = None,
    year: Optional[int] = None,
    department: Optional[str] = None,
) -> UploadResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    metadata = {
        "document_type": document_type,
        "industry": industry,
        "client": client,
        "project": project,
        "year": year,
        "department": department,
    }
    try:
        processing = ingest_pdf(filename, await file.read(), metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF upload/indexing failed for %s", filename)
        raise HTTPException(status_code=500, detail="PDF indexing failed") from exc

    return UploadResponse(
        filename=filename,
        message="PDF uploaded and indexed successfully",
        **processing,
    )


@router.post("/ask", response_model=QAResponse)
async def ask_question_endpoint(request: QuestionRequest) -> QAResponse:
    try:
        result = answer_question(request.query, filters=request.filters, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QAResponse(**result)
