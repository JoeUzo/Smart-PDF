# Smart PDF

Smart PDF is a web application that provides a set of tools to manage your PDF and Word documents.

## Features

- **Merge PDFs**: Combine multiple PDF files into one.
- **Split PDF**: Extract pages or split a PDF into multiple files.
- **PDF to Word**: Convert a PDF file to a Word document.
- **Word to PDF**: Convert a Word document to a PDF file.
- **Compress Document**: Reduce the file size of a PDF or Word document.
- **Chat with PDF**: Summarize and chat with your PDF documents using AI.

## Tech Stack

- **Backend**: Python 3.10+ with FastAPI
- **AI Integration**: OpenAI API, `openai` Python package
- **Server**: Uvicorn for development, Gunicorn with Uvicorn workers in production
- **PDF and Word processing**: pypdf, pdf2docx, docx2pdf, pikepdf, pdfminer.six
- **Asynchronous task queue**: Celery with Redis
- **Templating**: Jinja2
- **Frontend**: Bootstrap 5, JavaScript (fetch)
- **Containerisation**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## Project Layout

```
.
├── .github/workflows/main.yml
├── .gitignore
├── .env
├── Dockerfile
├── README.md
├── celery_worker.py
├── config.py
├── docker-compose.yml
├── main.py
├── requirements.txt
├── static/
│   └── css/
│       └── styles.css
├── templates/
│   ├── base.html
│   ├── chat.html
│   ├── compress.html
│   ├── index.html
│   ├── merge.html
│   ├── pdf_to_word.html
│   ├── processing.html
│   ├── summarize.html
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

4.  **Set up your environment variables**:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

5.  **Start Redis**:
    Make sure you have Redis installed and running on `localhost:6379`.

6.  **Start the Celery worker**:
    ```bash
    celery -A celery_worker.celery_app worker --loglevel=info
    ```

7.  **Start the FastAPI application**:
    ```bash
    uvicorn main:app --reload
    ```

    The application will be available at `http://localhost:8000`.

### Docker

1.  **Set up your environment variables**:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

2.  **Build and run the containers**:
    ```bash
    docker-compose up --build
    ```

    The application will be available at `http://localhost:8000`.

### CI/CD

The project includes a basic CI/CD workflow using GitHub Actions. The workflow is defined in `.github/workflows/main.yml`. It automatically builds the Docker image on every push and pull request to the `main` branch.