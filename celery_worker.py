import os
import shutil
import zipfile
import logging
from pathlib import Path
import openai
from celery import Celery
from dotenv import load_dotenv
from PIL import Image
import pikepdf
from pypdf import PdfReader, PdfWriter
from pdf2docx import Converter
from docx2pdf import convert as docx_to_pdf_convert
from config import settings
import subprocess
import platform

load_dotenv()

# --- Celery Configuration ---
celery_app = Celery(
    "tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

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

@celery_app.task(bind=True)
def summarize_pdf_task(self, file_path: str, model: str):
    file_path = Path(file_path)
    try:
        # 1. Extract text from PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # 2. Store full text for chat context
        context_path = file_path.parent / "context.txt"
        context_path.write_text(text,  encoding="utf-8")

        # 3. Chunk text for summary
        # A simple way to chunk is to take the beginning of the text.
        # A more advanced approach would be to use token-aware chunking.
        summary_prompt_text = text[:4000] # Limit to approx. 1000 tokens for the prompt

        # 4. Call OpenAI for a concise overview
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
                {"role": "user", "content": f"Please provide a concise summary of the following document:\n\n{summary_prompt_text}"}
            ],
            max_tokens=150
        )
        summary = response.choices[0].message.content

        return {
            "summary": summary,
            "job_id": file_path.parent.name,
            "filename": file_path.name
        }
    except Exception as e:
        logger.error(f"Error in summarize_pdf_task for {file_path.name}: {e}", exc_info=True)
        cleanup_directory(file_path.parent)
        raise



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
        
        # Clean up the original uploaded files
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
            # Logic for splitting based on custom ranges
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
            # Logic for splitting all pages into individual files
            for i, page in enumerate(reader.pages):
                writer = PdfWriter()
                writer.add_page(page)
                with open(output_dir / f"page_{i+1}.pdf", "wb") as f:
                    writer.write(f)

        # Zip the results
        zip_filename = "split_pages.zip"
        zip_path = file_path.parent / zip_filename
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for f in output_dir.glob('*.pdf'):
                zipf.write(f, f.name)
        
        shutil.rmtree(output_dir) # Clean up the intermediate split files
        file_path.unlink() # Clean up original upload
        
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
        # Step 1: Compress using pikepdf
        with pikepdf.open(file_path) as pdf:
            pdf.save(
                output_path,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
                linearize=True
            )

        # Step 2: Check if size reduction happened
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
        return
    else:
        gs_executable = shutil.which("gs")  # Linux/Mac/Docker

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
