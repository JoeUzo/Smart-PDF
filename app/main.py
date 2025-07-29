import uuid
import logging
import json
from pathlib import Path
from typing import List, AsyncGenerator
import openai
from openai import AsyncOpenAI
import aiofiles
from celery.result import AsyncResult
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from app.celery_worker import (
    merge_pdfs_task, split_pdf_task, pdf_to_word_task,
    word_to_pdf_task, compress_pdf_task, summarize_pdf_task
)
from app.utils import (
    save_upload_file, cleanup_directory, load_ai_prompts_json, setup_logging,
    EmailSchema, send_mail
)
from app.config import settings


# --- App and Logging Setup ---
app = FastAPI()

# --- Configuration and Setup ---
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
LOG_DIR = BASE_DIR / settings.log_dir

# Setup logging
logger = setup_logging(LOG_DIR)

# Load prompts from JSON file
AI_PROMPT = load_ai_prompts_json()

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
client = AsyncOpenAI(api_key=settings.openai_api_key)

# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    job_id: str
    message: str
    history: List[Message]
    model: str

class SummarizeResponse(BaseModel):
    job_id: str
    summary: str
    filename: str

class TaskStatusResponse(BaseModel):
    state: str
    result: SummarizeResponse | str | None = None
    error: str | None = None

# --- Helper Functions ---
def create_task_directory() -> Path:
    """Creates a unique directory for a new task."""
    task_id = str(uuid.uuid4())
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir

def get_vector_store(job_id: str) -> FAISS:
    """Loads a FAISS index from disk for a given job_id."""
    index_path = UPLOAD_DIR / job_id / "faiss_index"
    if not index_path.exists():
        logger.error(f"FAISS index not found for job_id: {job_id}")
        raise HTTPException(status_code=404, detail="Vector index not found. Please upload the document again.")
    
    try:
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Failed to load FAISS index for job_id {job_id}: {e!r}")
        raise HTTPException(status_code=500, detail="Failed to load document context.")

# --- Core API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("contact_us.html", {"request": request})

@app.post("/contact", response_class=HTMLResponse)
async def contact_form(request: Request, name: str = Form(...), email: str = Form(...), phone: str = Form(...), message: str = Form(...)):
    email_data = EmailSchema(name=name, email=email, phone=phone, message=message)
    try:
        send_mail(email_data)
        return templates.TemplateResponse("contact_us.html", {"request": request, "success": True})
    except ValueError as e:
        logger.error(f"Email sending error: {e}")
        return templates.TemplateResponse("contact_us.html", {"request": request, "error": "Could not send email. Please check server configuration."})
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return templates.TemplateResponse("contact_us.html", {"request": request, "error": "An unexpected error occurred."})


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse("privacy_policy.html", {"request": request})

@app.get("/terms", response_class=HTMLResponse)
async def terms_page(request: Request):
    return templates.TemplateResponse("terms_of_service.html", {"request": request})

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}

@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def task_status(task_id: str):
    """Polls for the status of a Celery task."""
    task = AsyncResult(task_id)
    response = {"state": task.state}
    if task.successful():
        response["result"] = task.result
    elif task.failed():
        logger.error(f"Task {task_id} failed with error: {task.info!r}")
        response["error"] = "Task failed during processing. Please try again."
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
        # The task directory name IS the job_id for this workflow
        job_id = task_dir.name
        saved_path = await save_upload_file(file, task_dir)
        
        # The Celery task ID is different from the job_id (directory name)
        task = summarize_pdf_task.delay(str(saved_path))
        
        # We pass the job_id (directory name) to the processing page
        return templates.TemplateResponse(
            "processing.html", 
            {"request": request, "task_id": task.id, "job_id": job_id, "redirect_url": f"/chat_page/{job_id}"}
        )
    except HTTPException as e:
        cleanup_directory(task_dir)
        raise e
    except Exception as e:
        logger.error(f"Error in summarize_endpoint: {e!r}", exc_info=True)
        cleanup_directory(task_dir)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file upload.")

@app.get("/chat_page/{job_id}", response_class=HTMLResponse)
async def chat_page(request: Request, job_id: str):
    """
    Serves the chat page. This endpoint is now hit directly with the job_id.
    It needs to fetch the initial summary from the corresponding task result.
    This requires a way to map job_id back to a task_id if we want to show processing.
    For simplicity, we assume the frontend polls the /status endpoint with the task_id
    and redirects here only on success.
    """
    # To get the summary, we need the task result. We don't have the task_id here.
    # This is a change in flow. The frontend should get the result from /status/{task_id}
    # and then pass the summary data to the chat page, or the chat page can fetch it.
    # Let's assume the chat page makes a call to a new endpoint to get summary details.
    
    # Let's create a placeholder endpoint for that.
    details_path = UPLOAD_DIR / job_id / "details.json"
    if not details_path.exists():
        # This could mean the task is still running or failed.
        # The user should be on a polling page.
        # We can redirect them back to a generic processing page if they land here too early.
        # This is a UX challenge. For now, we'll assume the happy path.
        # A robust solution would involve the frontend handling this state.
        raise HTTPException(404, "Chat session details not found. The processing might still be in progress or has failed.")

    async with aiofiles.open(details_path, "r") as f:
        content = await f.read()
        result = SummarizeResponse.model_validate_json(content)

    pdf_url = f"/uploads/{job_id}/{result.filename}"

    return templates.TemplateResponse("chat.html", {
        "request": request,
        "job_id": job_id,
        "summary": result.summary,
        "pdf_url": pdf_url,
        "openai_models": settings.openai_models
    })


async def stream_chat_responses(vector_store: FAISS, chat_request: ChatRequest) -> AsyncGenerator[str, None]:
    """Generator for streaming chat responses using RAG."""
    # 1. Find relevant document chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Top 5 chunks
    relevant_docs = await retriever.ainvoke(chat_request.message)
    
    context = "\n---\n".join([doc.page_content for doc in relevant_docs])
    
    # 2. Build the prompt
    system_prompt = AI_PROMPT["chat_prompt"]["system"]
    user_prompt = AI_PROMPT["chat_prompt"]["user_template"].format(context=context)
    generation_params = AI_PROMPT["chat_prompt"].get("generation_parameters", {})
    
    prompt_messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_prompt),
    ]
    prompt_messages.extend(chat_request.history)
    prompt_messages.append(Message(role="user", content=chat_request.message))

    # 3. Stream response from OpenAI
    try:
        response_stream = await client.chat.completions.create(
            model=chat_request.model,
            messages=[msg.model_dump() for msg in prompt_messages],
            **generation_params
        )
        async for chunk in response_stream:
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"
    except openai.OpenAIError as e:
        logger.error(f"OpenAI streaming error for job {chat_request.job_id}: {e!r}")
        yield f"data: [ERROR] An error occurred with the AI service. Please try again.\n\n"
    except Exception as e:
        logger.error(f"Unexpected streaming error for job {chat_request.job_id}: {e!r}", exc_info=True)
        yield f"data: [ERROR] An unexpected server error occurred.\n\n"


@app.get("/chat")
async def chat_endpoint(
    job_id: str,
    message: str,
    model: str,
    history: str, # JSON string
    vector_store: FAISS = Depends(get_vector_store)
):
    """Handles streaming chat responses using RAG and Server-Sent Events."""
    try:
        history_list = json.loads(history)
        chat_history = [Message(**msg) for msg in history_list]
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid history format.")

    chat_request = ChatRequest(
        job_id=job_id,
        message=message,
        history=chat_history,
        model=model
    )
    
    return StreamingResponse(
        stream_chat_responses(vector_store, chat_request),
        media_type="text/event-stream"
    )


# --- Existing PDF/Word Processing Endpoints (Unchanged but with async file writes) ---
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