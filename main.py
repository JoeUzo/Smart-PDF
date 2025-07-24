import os
import shutil
import zipfile
from typing import List
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiofiles
from celery_worker import merge_pdfs_task, split_pdf_task, pdf_to_word_task, word_to_pdf_task, compress_pdf_task

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/merge", response_class=HTMLResponse)
async def merge_page(request: Request):
    return templates.TemplateResponse("merge.html", {"request": request})

@app.post("/merge")
async def merge_pdfs(request: Request, files: List[UploadFile] = File(...)):
    output_filename = "merged.pdf"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        file_paths.append(file_path)
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

    task = merge_pdfs_task.delay(file_paths, output_path)
    result_path = task.get()

    # Clean up uploaded files
    for file_path in file_paths:
        os.remove(file_path)

    return FileResponse(result_path, media_type='application/pdf', filename=output_filename)

@app.get("/split", response_class=HTMLResponse)
async def split_page(request: Request):
    return templates.TemplateResponse("split.html", {"request": request})

@app.post("/split")
async def split_pdf(request: Request, file: UploadFile = File(...), ranges: str = Form(None)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    output_dir = os.path.join(UPLOAD_DIR, "split_output")
    task = split_pdf_task.delay(file_path, ranges, output_dir)
    result_path = task.get()
    
    os.remove(file_path)

    return FileResponse(result_path, media_type='application/zip', filename=os.path.basename(result_path))


@app.get("/pdf-to-word", response_class=HTMLResponse)
async def pdf_to_word_page(request: Request):
    return templates.TemplateResponse("pdf_to_word.html", {"request": request})

@app.post("/pdf-to-word")
async def pdf_to_word(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    output_filename = f"{os.path.splitext(file.filename)[0]}.docx"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    task = pdf_to_word_task.delay(file_path, output_path)
    result_path = task.get()

    os.remove(file_path)

    return FileResponse(result_path, media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document', filename=output_filename)

@app.get("/word-to-pdf", response_class=HTMLResponse)
async def word_to_pdf_page(request: Request):
    return templates.TemplateResponse("word_to_pdf.html", {"request": request})

@app.post("/word-to-pdf")
async def word_to_pdf(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    output_filename = f"{os.path.splitext(file.filename)[0]}.pdf"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    task = word_to_pdf_task.delay(file_path, output_path)
    result_path = task.get()

    os.remove(file_path)

    return FileResponse(result_path, media_type='application/pdf', filename=output_filename)

@app.get("/compress", response_class=HTMLResponse)
async def compress_page(request: Request):
    return templates.TemplateResponse("compress.html", {"request": request})

@app.post("/compress")
async def compress_document(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    output_filename = f"compressed_{file.filename}"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    if file.content_type == "application/pdf":
        task = compress_pdf_task.delay(file_path, output_path)
        result_path = task.get()
    else:
        # For Word documents, just copy the file for now
        shutil.copy(file_path, output_path)
        result_path = output_path

    os.remove(file_path)

    return FileResponse(result_path, media_type=file.content_.type, filename=output_filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

