import logging
from pathlib import Path
import tempfile
import shutil

from pypdf import PdfReader
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
from PIL import Image
import pytesseract

from app.config import settings
from app.utils import setup_logging

# --- Logging Setup ---
LOG_DIR = Path(__file__).resolve().parent.parent / settings.log_dir
logger = setup_logging(LOG_DIR)

MIN_TEXT_LENGTH_THRESHOLD = 100  # If digital text is less than this, OCR will be triggered.

def is_tool_installed(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

def extract_text_with_ocr(file_path: Path, job_id: str) -> str:
    """
    Extracts text from a PDF, falling back to OCR if necessary, with enhanced
    logging and debugging capabilities.
    """
    logger.info(f"[{job_id}] Starting text extraction from {file_path.name}")
    upload_dir = file_path.parent

    # 1. Attempt to extract text directly and gather PDF metadata
    digital_text = ""
    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            logger.warning(f"[{job_id}] PDF is encrypted. Text extraction may fail.")
        
        logger.info(f"[{job_id}] PDF has {len(reader.pages)} pages.")
        digital_text = "".join(page.extract_text() or "" for page in reader.pages)
        logger.info(f"[{job_id}] Initial digital text extraction found {len(digital_text.strip())} characters.")
    except Exception as e:
        logger.error(f"[{job_id}] Error reading PDF with pypdf: {e!r}")

    # 2. Check if digital text is sufficient
    if len(digital_text.strip()) > MIN_TEXT_LENGTH_THRESHOLD:
        logger.info(f"[{job_id}] Digital text is sufficient. Skipping OCR.")
        return digital_text

    # 3. Check for dependencies and settings before attempting OCR
    if not settings.ocr_enabled:
        logger.warning(f"[{job_id}] Digital text is minimal, but OCR is disabled. Returning as-is.")
        return digital_text

    if not is_tool_installed("tesseract"):
        logger.error(f"[{job_id}] Tesseract-OCR is not installed or not in PATH. Cannot perform OCR.")
        return digital_text

    logger.info(f"[{job_id}] Digital text is insufficient. Falling back to OCR.")
    ocr_text = ""
    
    # Use a persistent directory for debugging if enabled
    if settings.ocr_save_debug_images:
        debug_dir = upload_dir / "ocr_debug_images"
        debug_dir.mkdir(exist_ok=True)
        image_output_folder = debug_dir
        logger.warning(f"[{job_id}] OCR debug mode is enabled. Saving intermediate images to {debug_dir}")
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix=f"ocr_{job_id}_")
        image_output_folder = Path(temp_dir.name)

    try:
        # Convert PDF pages to images
        images = convert_from_path(
            file_path,
            dpi=settings.ocr_dpi,
            output_folder=image_output_folder,
            fmt="jpeg",
            thread_count=4,
        )

        if not images:
            logger.warning(f"[{job_id}] pdf2image converted 0 pages. Check PDF content and Poppler installation.")
            return digital_text

        logger.info(f"[{job_id}] Converted {len(images)} pages to images for OCR.")

        # Perform OCR on each image object directly
        for i, image_obj in enumerate(images):
            page_num = i + 1
            try:
                # Pass the PIL.Image object directly to pytesseract
                page_text = pytesseract.image_to_string(
                    image_obj,
                    lang=settings.ocr_lang,
                    config=settings.ocr_config
                )
                ocr_text += page_text + "\n"
                logger.info(f"[{job_id}] OCR Page {page_num}: Extracted {len(page_text.strip())} characters.")
            except pytesseract.TesseractError as e:
                logger.error(f"[{job_id}] Tesseract error on page {page_num}: {e!r}")
            except Exception as e:
                logger.error(f"[{job_id}] Error processing image for page {page_num}: {e!r}")
            finally:
                # Close the image object to free resources
                image_obj.close()

    except (PDFInfoNotInstalledError, PDFPageCountError) as e:
        logger.error(f"[{job_id}] Failed to convert PDF to images. Poppler might be missing or the PDF is corrupted. Error: {e!r}")
        return digital_text
    except Exception as e:
        logger.error(f"[{job_id}] An unexpected error occurred during PDF to image conversion: {e!r}")
        return digital_text
    finally:
        # Clean up the temp directory if we created one
        if not settings.ocr_save_debug_images and 'temp_dir' in locals():
            temp_dir.cleanup()

    if not ocr_text.strip():
        logger.warning(f"[{job_id}] OCR process ran but did not produce any text. The PDF might be blank or unreadable.")

    full_text = digital_text.strip() + "\n" + ocr_text.strip()
    logger.info(f"[{job_id}] Final combined text length is {len(full_text.strip())} characters.")
    return full_text.strip()
