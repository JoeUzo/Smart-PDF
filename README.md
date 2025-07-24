# Smart PDF

Smart PDF is a web application that provides a set of tools to manage your PDF and Word documents.

## Features

- **Merge PDFs**: Combine multiple PDF files into one.
- **Split PDF**: Extract pages or split a PDF into multiple files.
- **PDF to Word**: Convert a PDF file to a Word document.
- **Word to PDF**: Convert a Word document to a PDF file.
- **Compress Document**: Reduce the file size of a PDF or Word document.

## Tech Stack

- **Backend**: Python 3.10+ with FastAPI
- **Server**: Uvicorn for development, Gunicorn with Uvicorn workers in production
- **PDF and Word processing**: pypdf, pdf2docx, docx2pdf, pikepdf
- **Asynchronous task queue**: Celery with Redis
- **Templating**: Jinja2
- **Frontend**: Bootstrap 5
- **Containerisation**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## Project Layout

```
.
├── .github/workflows/main.yml
├── .gitignore
├── Dockerfile
├── README.md
├── celery_worker.py
├── docker-compose.yml
├── main.py
├── requirements.txt
├── static/
│   └── css/
│       └── styles.css
├── templates/
│   ├── base.html
│   ├── compress.html
│   ├── index.html
│   ├── merge.html
│   ├── pdf_to_word.html
│   ├── split.html
│   └── word_to_pdf.html
└── uploads/
```

## Setup and Usage

### Local Development (without Docker)

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd smart-pdf
    ```

2.  **Create a virtual environment and activate it**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start Redis**:
    Make sure you have Redis installed and running on `localhost:6379`.

5.  **Start the Celery worker**:
    ```bash
    celery -A celery_worker.celery_app worker --loglevel=info
    ```

6.  **Start the FastAPI application**:
    ```bash
    uvicorn main:app --reload
    ```

    The application will be available at `http://localhost:8000`.

### Docker

1.  **Build and run the containers**:
    ```bash
    docker-compose up --build
    ```

    The application will be available at `http://localhost:8000`.

### CI/CD

The project includes a basic CI/CD workflow using GitHub Actions. The workflow is defined in `.github/workflows/main.yml`. It automatically builds the Docker image on every push and pull request to the `main` branch.
