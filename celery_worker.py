from celery import Celery
import os
from dotenv import load_dotenv
import logging
import shutil
import zipfile
from PIL import Image

load_dotenv()

celery_app = Celery(
  "tasks",
  broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
  backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

logger = logging.getLogger(__name__)

@celery_app.task
def merge_pdfs_task(file_paths, output_path):
    from pypdf import PdfWriter
    logger.info(f"Merging PDFs: {file_paths} into {output_path}")
    writer = PdfWriter()
    try:
        for file_path in file_paths:
            logger.info(f"Appending {file_path}")
            writer.append(file_path)
        writer.write(output_path)
        writer.close()
        logger.info("Merge successful")
        return output_path
    except Exception as e:
        logger.error(f"Error merging PDFs: {e}", exc_info=True)
        # Reraise the exception to mark the task as failed
        raise

@celery_app.task
def split_pdf_task(file_path, ranges, output_dir):
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if ranges:
        page_groups = ranges.split(',')
        for i, group in enumerate(page_groups):
            writer = PdfWriter()
            pages = group.split('-')
            if len(pages) == 2:
                start, end = int(pages[0]), int(pages[1]) if pages[1] else len(reader.pages)
                for page_num in range(start - 1, end):
                    writer.add_page(reader.pages[page_num])
            else:
                writer.add_page(reader.pages[int(pages[0]) - 1])
            
            output_filename = f"split_{i+1}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "wb") as f:
                writer.write(f)
    else:
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            output_filename = f"page_{i+1}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, "wb") as f:
                writer.write(f)

    zip_filename = "split_pages.zip"
    zip_path = os.path.join(os.path.dirname(output_dir), zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for f in files:
                zipf.write(os.path.join(root, f), f)
    
    shutil.rmtree(output_dir)
    return zip_path

@celery_app.task
def pdf_to_word_task(file_path, output_path):
    from pdf2docx import Converter
    cv = Converter(file_path)
    cv.convert(output_path, start=0, end=None)
    cv.close()
    return output_path

@celery_app.task
def word_to_pdf_task(file_path, output_path):
    from docx2pdf import convert
    convert(file_path, output_path)
    return output_path

@celery_app.task
def compress_pdf_task(file_path, output_path):
    import pikepdf
    pdf = pikepdf.open(file_path)
    pdf.save(output_path, compress_streams=True, recompress_flate=True)
    pdf.close()
    return output_path

@celery_app.task
def compress_word_task(file_path, output_path):
    temp_dir = "temp_word_unzip"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    media_dir = os.path.join(temp_dir, "word", "media")
    if os.path.exists(media_dir):
        for filename in os.listdir(media_dir):
            image_path = os.path.join(media_dir, filename)
            try:
                img = Image.open(image_path)
                img.save(image_path, optimize=True, quality=85)
            except Exception as e:
                logger.error(f"Could not compress image {filename}: {e}")

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path_in_zip = os.path.relpath(os.path.join(root, file), temp_dir)
                zip_ref.write(os.path.join(root, file), file_path_in_zip)

    shutil.rmtree(temp_dir)
    return output_path
