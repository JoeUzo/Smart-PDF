import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List
import aiofiles
from fastapi import UploadFile, HTTPException

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
# Maximum file size (e.g., 100 MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

def secure_filename(filename: str) -> str:
    """Sanitizes a filename to prevent path traversal attacks."""
    return Path(filename).name

async def save_upload_file(file: UploadFile, upload_dir: Path) -> Path:
    """
    Saves an uploaded file to a specified directory with a unique name,
    performing security checks.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    # Sanitize filename
    sanitized_filename = secure_filename(file.filename)
    file_extension = Path(sanitized_filename).suffix.lower()

    # Validate file type
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{file_extension}' is not allowed.")

    # Create a unique filename to prevent collisions
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = upload_dir / unique_filename

    # Ensure the upload directory exists
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save the file chunk by chunk to handle large files and check size
    file_size = 0
    async with aiofiles.open(file_path, 'wb') as out_file:
        while content := await file.read(1024 * 1024):  # Read in 1MB chunks
            file_size += len(content)
            if file_size > MAX_FILE_SIZE:
                file_path.unlink()  # Clean up partial file
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds the limit of {MAX_FILE_SIZE / 1024 / 1024} MB."
                )
            await out_file.write(content)

    return file_path

def cleanup_directory(dir_path: Path):
    """Recursively removes a directory and its contents."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)

def load_ai_prompts_json():
    """extracts AI prompts from a JSON file."""
    prompts_path = Path(__file__).resolve().parent.parent/ "config" / "ai_prompts.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        ai_prompt = json.load(f)
    
    return ai_prompt