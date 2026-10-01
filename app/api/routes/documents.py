"""Document routes: upload, list, delete, raw search preview."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_document_service
from app.services.document_service import DocumentService
from app.api.schemas import (
    DocumentInfo,
    IngestResult,
    IngestTextRequest,
    SearchPreviewRequest,
)
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger("api.documents")

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_SUFFIXES = {".md", ".txt", ".csv", ".json", ".rst", ".html", ".pdf"}


@router.post("/upload", response_model=IngestResult)
def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> IngestResult:
    """Upload and ingest a document into the RAG knowledge base.

    Supported: ``.md .txt .csv .json .rst .html .pdf``.
    Re-uploading the same file replaces its chunks (idempotent).
    """
    settings = get_settings()
    file_name = file.filename or "upload.txt"
    suffix = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )

    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds max_upload_bytes")

    try:
        result = service.ingest_upload(file_name, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return IngestResult(**result)


@router.post("/text", response_model=IngestResult)
def ingest_text_route(request: IngestTextRequest, service: DocumentService = Depends(get_document_service)) -> IngestResult:
    """Ingest raw text directly under a given source name."""
    result = service.ingest_text(request.text, request.source)
    return IngestResult(**result)


@router.get("", response_model=list[DocumentInfo])
def list_documents(service: DocumentService = Depends(get_document_service)) -> list[DocumentInfo]:
    """List all ingested documents with chunk counts."""
    return [DocumentInfo(**doc) for doc in service.list_documents()]


@router.delete("/{source}")
def delete_document(source: str, service: DocumentService = Depends(get_document_service)) -> dict:
    """Delete one document (all its chunks) from the knowledge base."""
    deleted = service.delete_document(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source}' not found")
    return {"source": source, "chunks_deleted": deleted}


@router.post("/search")
def search_preview(request: SearchPreviewRequest, service: DocumentService = Depends(get_document_service)) -> dict:
    """Debug endpoint: raw semantic search over the knowledge base."""
    hits = service.search_preview(request.query, top_k=request.top_k)
    return {"query": request.query, "results": hits}