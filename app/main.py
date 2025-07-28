import uuid
from pathlib import Path
from typing import List
import openai
from celery.result import AsyncResult
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from app.celery_worker import (
    merge_pdfs_task, split_pdf_task, pdf_to_word_task,
    word_to_pdf_task, compress_pdf_task, summarize_pdf_task
)
from app.utils import save_upload_file, cleanup_directory
from app.config import settings

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration and Setup ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
openai.api_key = settings.openai_api_key

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    job_id: str
    message: str
    history: List[dict]
    model: str

# --- Helper Functions ---
def create_task_directory() -> Path:
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
    return {"status": "ok"}

@app.get("/status/{task_id}", response_class=JSONResponse)
async def task_status(task_id: str):
    task = AsyncResult(task_id)
    response = {"state": task.state}
    if task.successful():
        response["result"] = task.result
    elif task.failed():
        response["error"] = "Task failed during processing."
    return JSONResponse(response)

# --- AI Chat Endpoints ---
@app.get("/summarize", response_class=HTMLResponse)
async def summarize_page(request: Request):
    return templates.TemplateResponse("summarize.html", {"request": request})

@app.post("/summarize")
async def summarize_endpoint(request: Request, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    task_dir = create_task_directory()
    try:
        saved_path = await save_upload_file(file, task_dir)
        task = summarize_pdf_task.delay(str(saved_path), settings.openai_models[0])
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id, "redirect_url": f"/chat_page/{task.id}"})
    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e

@app.get("/chat_page/{task_id}", response_class=HTMLResponse)
async def chat_page(request: Request, task_id: str):
    task = AsyncResult(task_id)
    if not task.successful():
        # This is not a failure, the task is just not ready.
        # We can return the processing page again, or a specific "waiting" page.
        # For simplicity, we'll let the frontend handle polling.
        # A more robust solution might involve a different template.
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id, "redirect_url": f"/chat_page/{task.id}"})

    result = task.result
    job_id = result.get("job_id")
    filename = result.get("filename")
    pdf_url = f"/uploads/{job_id}/{filename}"

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "job_id": job_id,
        "summary": result.get("summary"),
        "pdf_url": pdf_url,
        "openai_models": settings.openai_models
    })

@app.post("/chat", response_class=JSONResponse)
async def chat_endpoint(chat_request: ChatRequest):
    context_path = UPLOAD_DIR / chat_request.job_id / "context.txt"
    if not context_path.exists():
        raise HTTPException(status_code=404, detail="Chat context not found.")
    
    pdf_text = context_path.read_text(encoding="utf-8")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. The user has provided a PDF document. Your task is to answer questions based on its content."},
        {"role": "user", "content": f"Here is the content of the PDF:\n\n{pdf_text}"}
    ]
    messages.extend(chat_request.history)
    messages.append({"role": "user", "content": chat_request.message})

    try:
        response = openai.chat.completions.create(
            model=chat_request.model,
            messages=messages,
            max_tokens=500
        )
        reply = response.choices[0].message.content
        
        updated_history = chat_request.history + [{"role": "user", "content": chat_request.message}, {"role": "assistant", "content": reply}]
        
        return {"reply": reply, "history": updated_history}
    except Exception as e:
        # Use repr(e) to get a developer-friendly, safe representation of the exception.
        error_detail = f"Failed to get response from OpenAI: {repr(e)}"
        raise HTTPException(status_code=500, detail=error_detail)


# --- Existing PDF/Word Processing Endpoints ---
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
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for compression.")
        return templates.TemplateResponse("processing.html", {"request": request, "task_id": task.id})
    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e

# --- File Preview and Download ---
@app.get("/preview/{task_id}/{filename}", response_class=HTMLResponse)
async def preview_page(request: Request, task_id: str, filename: str):
    file_url = f"/uploads/{task_id}/{filename}"
    download_url = f"/download/{task_id}/{filename}"
    return templates.TemplateResponse(
        "preview.html",
        {"request": request, "file_path": file_url, "download_url": download_url, "filename": filename}
    )

@app.get("/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    file_path = UPLOAD_DIR / task_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    
    return FileResponse(
        path=str(file_path),
        media_type='application/octet-stream',
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)