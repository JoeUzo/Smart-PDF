import os
import json
import uuid
import shutil
import logging
import smtplib
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import List
import aiofiles
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel, EmailStr
from app.config import Settings

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
# Maximum file size (e.g., 100 MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

class EmailSchema(BaseModel):
    name: str
    email: EmailStr
    phone: str
    message: str

def send_mail(data: EmailSchema):
    settings = Settings()
    if not settings.email or not settings.email_key:
        raise ValueError("EMAIL or EMAIL_KEY is not set in the environment.")

    message = f"Name: {data.name}\nEmail: {data.email}\n" \
              f"Phone No.: {data.phone}\nMessage: {data.message}"

    try:
        with smtplib.SMTP("smtp.gmail.com", port=587, timeout=60) as connection:
            connection.starttls()
            connection.login(user=settings.email, password=settings.email_key)
            connection.sendmail(
                from_addr=settings.email,
                to_addrs=settings.email,
                msg=f"subject: Smart PDF\n\n{message}".encode('utf-8')
            )
    except smtplib.SMTPConnectError as e:
        print(f"Failed to connect to the SMTP server. Error: {e}")
        print("Make sure the network allows outbound connection to port 587.")
    except smtplib.SMTPAuthenticationError as e:
        print(f"Authentication error: {e}")
        print("Ensure EMAIL and EMAIL_KEY are correct and App Passwords are enabled.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Check your environment or try again later.")


# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
# Maximum file size (e.g., 100 MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

def setup_logging(log_dir: Path):
    """Configures a rotating file logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Use a rotating file handler
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Get the root logger and add the file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if this function is called multiple times
    if not any(isinstance(h, TimedRotatingFileHandler) for h in root_logger.handlers):
        root_logger.addHandler(file_handler)

    # Also, keep logging to the console
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(stream_handler)
    
    return logging.getLogger(__name__)

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