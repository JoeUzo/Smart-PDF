from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

@celery_app.task
def merge_pdfs_task(file_paths, output_path):
    from pypdf import PdfMerger
    merger = PdfMerger()
    for file_path in file_paths:
        merger.append(file_path)
    merger.write(output_path)
    merger.close()
    return output_path

@celery_app.task
def split_pdf_task(file_path, ranges, output_dir):
    import os
    import shutil
    import zipfile
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
    pdf.save(output_path, compress_streams=True, linearize=True)
    pdf.close()
    return output_path
