import os
import shutil
import zipfile
import logging
from pathlib import Path
from celery import Celery
from dotenv import load_dotenv
from PIL import Image
import pikepdf
from pypdf import PdfWriter, PdfReader
from pdf2docx import Converter
from docx2pdf import convert as docx_to_pdf_convert

load_dotenv()

# --- Celery Configuration ---
celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_relative_path(full_path: Path) -> str:
    """Returns the path relative to the 'uploads' directory as a POSIX path."""
    return full_path.relative_to(full_path.parent.parent).as_posix()

# --- Celery Tasks ---

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
        with pikepdf.open(file_path) as pdf:
            pdf.save(output_path, compress_streams=True, recompress_flate=True)
        file_path.unlink()
        return get_relative_path(output_path)
    except Exception as e:
        logger.error(f"Error compressing PDF for task {self.request.id}: {e}", exc_info=True)
        cleanup_directory(file_path.parent)
        raise

def cleanup_directory(dir_path: Path):
    """Shared cleanup utility."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)