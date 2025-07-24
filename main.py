import os
import uuid
from pathlib import Path
from typing import List

from celery.result import AsyncResult
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from celery_worker import merge_pdfs_task, split_pdf_task, pdf_to_word_task, word_to_pdf_task, compress_pdf_task
from utils import save_upload_file, cleanup_directory

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration and Setup ---

# Directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and uploads
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Setup templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --- Helper Functions ---

def create_task_directory() -> Path:
    """Creates a unique directory for each task."""
    task_id = str(uuid.uuid4())
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


# --- Core API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", status_code=200)
def health_check():
    """Endpoint to check if the service is running."""
    return {"status": "ok"}

@app.get("/status/{task_id}", response_class=JSONResponse)
async def task_status(task_id: str):
    """Polls for the status of a Celery task."""
    task = AsyncResult(task_id)
    response = {"state": task.state}
    if task.successful():
        response["result"] = task.result
    elif task.failed():
        # Provide a generic error message for security
        response["error"] = "Task failed during processing."
    return JSONResponse(response)


# --- PDF/Word Processing Endpoints ---

@app.get("/merge", response_class=HTMLResponse)
async def merge_page(request: Request):
    return templates.TemplateResponse("merge.html", {"request": request})

@app.post("/merge")
async def merge_pdfs_endpoint(request: Request, files: List[UploadFile] = File(...)):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Please upload at least two files to merge.")

    task_dir = create_task_directory()
    saved_files = []
    try:
        for file in files:
            saved_path = await save_upload_file(file, task_dir)
            saved_files.append(str(saved_path))

        output_filename = "merged.pdf"
        output_path = str(task_dir / output_filename)

        task = merge_pdfs_task.delay(saved_files, output_path)
        
        # Return the processing page with the task ID
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id})

    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e


@app.get("/split", response_class=HTMLResponse)
async def split_page(request: Request):
    return templates.TemplateResponse("split.html", {"request": request})

@app.post("/split")
async def split_pdf_endpoint(request: Request, file: UploadFile = File(...), ranges: str = Form(None)):
    task_dir = create_task_directory()
    try:
        saved_path = await save_upload_file(file, task_dir)
        output_dir = task_dir / "split_output"
        
        task = split_pdf_task.delay(str(saved_path), ranges, str(output_dir))
        
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id})
    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e


@app.get("/pdf-to-word", response_class=HTMLResponse)
async def pdf_to_word_page(request: Request):
    return templates.TemplateResponse("pdf_to_word.html", {"request": request})

@app.post("/pdf-to-word")
async def pdf_to_word_endpoint(request: Request, file: UploadFile = File(...)):
    task_dir = create_task_directory()
    try:
        saved_path = await save_upload_file(file, task_dir)
        output_filename = f"{saved_path.stem}.docx"
        output_path = str(task_dir / output_filename)

        task = pdf_to_word_task.delay(str(saved_path), output_path)
        
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id})
    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e


@app.get("/word-to-pdf", response_class=HTMLResponse)
async def word_to_pdf_page(request: Request):
    return templates.TemplateResponse("word_to_pdf.html", {"request": request})

@app.post("/word-to-pdf")
async def word_to_pdf_endpoint(request: Request, file: UploadFile = File(...)):
    task_dir = create_task_directory()
    try:
        saved_path = await save_upload_file(file, task_dir)
        output_filename = f"{saved_path.stem}.pdf"
        output_path = str(task_dir / output_filename)

        task = word_to_pdf_task.delay(str(saved_path), output_path)
        
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id})
    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e


@app.get("/compress", response_class=HTMLResponse)
async def compress_page(request: Request):
    return templates.TemplateResponse("compress.html", {"request": request})

@app.post("/compress")
async def compress_document_endpoint(request: Request, file: UploadFile = File(...)):
    task_dir = create_task_directory()
    try:
        saved_path = await save_upload_file(file, task_dir)
        output_filename = f"compressed_{saved_path.name}"
        output_path = str(task_dir / output_filename)

        if file.content_type == "application/pdf":
            task = compress_pdf_task.delay(str(saved_path), output_path)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Assuming compress_word_task exists and is implemented correctly
            # from celery_worker import compress_word_task
            # task = compress_word_task.delay(str(saved_path), output_path)
            raise HTTPException(status_code=501, detail="Word compression not implemented yet.")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for compression.")
            
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id})
    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e


# --- File Preview and Download ---

@app.get("/preview/{task_id}/{filename}", response_class=HTMLResponse)
async def preview_page(request: Request, task_id: str, filename: str):
    """Shows a preview of the processed file with a download link."""
    file_url = f"/uploads/{task_id}/{filename}"
    download_url = f"/download/{task_id}/{filename}"
    return templates.TemplateResponse(
        "preview.html",
        {"request": request, "file_path": file_url, "download_url": download_url, "filename": filename}
    )

@app.get("/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    """Provides the processed file for download."""
    file_path = UPLOAD_DIR / task_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    
    return FileResponse(
        path=str(file_path),
        media_type='application/octet-stream',
        filename=filename
    )

# --- Main Entry Point ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)