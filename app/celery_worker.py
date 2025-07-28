import os
import shutil
import zipfile
import logging
import time
import json
from pathlib import Path
import openai
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader, PdfWriter
from pdf2docx import Converter
from docx2pdf import convert as docx_to_pdf_convert
import pikepdf
import subprocess
import platform

from app.config import settings
from app.ocr import extract_text_with_ocr

load_dotenv()

# --- Celery Configuration ---
celery_app = Celery(
    "tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.beat_schedule = {
    'cleanup-old-uploads-daily': {
        'task': 'app.celery_worker.cleanup_old_directories_task',
        'schedule': crontab(hour=3, minute=0),  # Runs daily at 3:00 AM
    },
}

logger = logging.getLogger(__name__)
openai.api_key = settings.openai_api_key

# --- Helper Functions ---

def get_relative_path(full_path: Path) -> str:
    """Returns the path relative to the 'uploads' directory as a POSIX path."""
    return full_path.relative_to(full_path.parent.parent).as_posix()

def cleanup_directory(dir_path: Path):
    """Shared cleanup utility."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)

# --- Celery Tasks ---

@celery_app.task(
    bind=True,
    autoretry_for=(openai.OpenAIError,),
    retry_backoff=True,
    max_retries=3,
    task_reject_on_worker_lost=True
)
def summarize_pdf_task(self, file_path: str, model: str):
    """
    Extracts text from a PDF (with OCR fallback), creates a vector index,
    and generates an initial summary.
    """
    job_id = self.request.id
    file_path = Path(file_path)
    upload_dir = file_path.parent
    faiss_index_path = upload_dir / "faiss_index"

    try:
        # 1. Extract text from PDF using the new OCR-aware function
        full_text = extract_text_with_ocr(file_path, job_id)

        if not full_text.strip():
    # return a graceful error payload instead of blowing up
            return {
                "summary": None,
                "job_id": upload_dir.name,
                "filename": file_path.name,
                "error": "No text could be extracted from the PDF"
            }


        # 2. Store full text for chat context (optional, as we use RAG)
        context_path = upload_dir / "context.txt"
        context_path.write_text(full_text, encoding="utf-8")

        # 3. Create token-aware chunks
        logger.info(f"[{job_id}] Splitting text into chunks.")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            chunk_size=500,
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(full_text)
        logger.info(f"[{job_id}] Created {len(chunks)} text chunks.")

        if not chunks:
            raise ValueError("Text was extracted, but splitting into chunks failed.")

        # 4. Generate embeddings and create FAISS index
        logger.info(f"[{job_id}] Generating embeddings and creating FAISS index.")
        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

        # 5. Save FAISS index to disk
        vector_store.save_local(str(faiss_index_path))
        logger.info(f"[{job_id}] FAISS index saved to {faiss_index_path}")

        # 6. Generate a concise overview summary from the first few chunks
        summary_prompt_text = "\n".join(chunks[:4])
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": f"Please provide a concise summary of the following document based on this initial content:\n\n{summary_prompt_text}"}
            ],
            max_tokens=250
        )
        summary = response.choices[0].message.content

        # 7. Save result to a JSON file for the frontend to fetch
        result = {
            "summary": summary,
            "job_id": upload_dir.name,
            "filename": file_path.name,
        }
        details_path = upload_dir / "details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(result, f)

        return result
    except openai.OpenAIError as e:
        logger.error(f"[{job_id}] OpenAI API error in summarize_pdf_task: {e!r}")
        raise  # Trigger Celery's autoretry
    except Exception as e:
        logger.error(
            f"[{job_id}] Unrecoverable error in summarize_pdf_task for file {file_path.name}: {e!r}",
            exc_info=True
        )
        cleanup_directory(upload_dir)
        raise  # Mark the task as failed

@celery_app.task
def cleanup_old_directories_task():
    """
    Periodically cleans up upload directories older than a specified TTL.
    """
    upload_dir = Path(__file__).resolve().parent.parent / "uploads"
    ttl_seconds = settings.upload_dir_ttl_hours * 3600
    now = time.time()
    
    logger.info("Running scheduled cleanup of old upload directories...")
    for subdir in upload_dir.iterdir():
        if subdir.is_dir():
            try:
                dir_mtime = subdir.stat().st_mtime
                if (now - dir_mtime) > ttl_seconds:
                    logger.info(f"Removing old directory: {subdir.name}")
                    cleanup_directory(subdir)
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error during cleanup of directory {subdir.name}: {e!r}", exc_info=True)
    logger.info("Cleanup task finished.")


@celery_app.task(bind=True)
def merge_pdfs_task(self, file_paths: list[str], output_path: str):
    output_path = Path(output_path)
    writer = PdfWriter()
    try:
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if file_path.exists():
                writer.append(str(file_path))
        
        writer.write(str(output_path))
        writer.close()
        
        for file_path_str in file_paths:
            Path(file_path_str).unlink()
            
        return get_relative_path(output_path)
    except Exception as e:
        logger.error(f"Error merging PDFs for task {self.request.id}: {e}", exc_info=True)
        cleanup_directory(output_path.parent)
        raise

@celery_app.task(bind=True)
def split_pdf_task(self, file_path: str, ranges: str | None, output_dir: str):
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        reader = PdfReader(file_path)
        if ranges:
            page_groups = ranges.split(',')
            for i, group in enumerate(page_groups):
                writer = PdfWriter()
                pages = group.split('-')
                start = int(pages[0])
                end = int(pages[1]) if len(pages) > 1 and pages[1] else start
                for page_num in range(start - 1, end):
                    if 0 <= page_num < len(reader.pages):
                        writer.add_page(reader.pages[page_num])
                
                split_filename = f"split_{i+1}.pdf"
                with open(output_dir / split_filename, "wb") as f:
                    writer.write(f)
        else:
            for i, page in enumerate(reader.pages):
                writer = PdfWriter()
                writer.add_page(page)
                with open(output_dir / f"page_{i+1}.pdf", "wb") as f:
                    writer.write(f)

        zip_filename = "split_pages.zip"
        zip_path = file_path.parent / zip_filename
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for f in output_dir.glob('*.pdf'):
                zipf.write(f, f.name)
        
        shutil.rmtree(output_dir)
        file_path.unlink()
        
        return get_relative_path(zip_path)
    except Exception as e:
        logger.error(f"Error splitting PDF for task {self.request.id}: {e}", exc_info=True)
        cleanup_directory(file_path.parent)
        raise

@celery_app.task(bind=True)
def pdf_to_word_task(self, file_path: str, output_path: str):
    file_path = Path(file_path)
    output_path = Path(output_path)
    try:
        cv = Converter(str(file_path))
        cv.convert(str(output_path), start=0, end=None)
        cv.close()
        file_path.unlink()
        return get_relative_path(output_path)
    except Exception as e:
        logger.error(f"Error converting PDF to Word for task {self.request.id}: {e}", exc_info=True)
        cleanup_directory(file_path.parent)
        raise

@celery_app.task(bind=True)
def word_to_pdf_task(self, file_path: str, output_path: str):
    file_path = Path(file_path)
    output_path = Path(output_path)
    try:
        docx_to_pdf_convert(str(file_path), str(output_path))
        file_path.unlink()
        return get_relative_path(output_path)
    except Exception as e:
        logger.error(f"Error converting Word to PDF for task {self.request.id}: {e}", exc_info=True)
        cleanup_directory(file_path.parent)
        raise

@celery_app.task(bind=True)
def compress_pdf_task(self, file_path: str, output_path: str):
    file_path = Path(file_path)
    output_path = Path(output_path)

    try:
        with pikepdf.open(file_path) as pdf:
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
                linearize=True
            )

        original_size = file_path.stat().st_size
        compressed_size = output_path.stat().st_size

        logger.info(f"Original PDF size: {original_size / 1024:.2f} KB")
        logger.info(f"Compressed PDF size: {compressed_size / 1024:.2f} KB")

        if compressed_size >= original_size:
            logger.info("PikePDF compression not effective, trying Ghostscript...")
            ghostscript_compress(file_path, output_path)

        file_path.unlink(missing_ok=True)
        return get_relative_path(output_path)

    except Exception as e:
        logger.error(f"Error compressing PDF for task {self.request.id}: {e}", exc_info=True)
        cleanup_directory(file_path.parent)
        raise

def ghostscript_compress(input_path: Path, output_path: Path):
    if platform.system() == "Windows":
        gs_executable = shutil.which("gswin64c") or shutil.which("gswin32c")
    else:
        gs_executable = shutil.which("gs")

    if not gs_executable:
        raise FileNotFoundError("Ghostscript executable not found. Install it or update PATH.")

    gs_command = [
        gs_executable,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/screen",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_path}",
        str(input_path)
    ]
    subprocess.run(gs_command, check=True)