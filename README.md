# Smart PDF

Smart PDF is a versatile, all-in-one web application designed to streamline your document management workflow. It offers a comprehensive suite of tools for processing PDF and Word documents, from basic manipulations to advanced AI-powered analysis.

## Key Features

- **Merge PDFs**: Combine multiple PDF files into a single, organized document.
- **Split PDF**: Extract specific pages or ranges from a PDF, or split it into multiple files.
- **Compress PDF**: Reduce the file size of your PDFs for easier sharing and storage.
- **Convert Documents**:
    - **PDF to Word**: Seamlessly convert PDF files into editable Word documents.
    - **Word to PDF**: Transform Word documents into professional-quality PDFs.
- **AI-Powered Summarization & Chat**:
    - **Summarize**: Get a quick, AI-generated summary of your PDF content.
    - **Chat with PDF**: Interact with your documents through a conversational AI interface to find information and gain insights.
- **Contact Form**: A fully functional contact form to send feedback or inquiries directly to the administrator.

## Tech Stack

- **Backend**: Python 3.10+ with FastAPI
- **AI Integration**: OpenAI API, LangChain, FAISS for vector storage
- **Asynchronous Task Queue**: Celery with Redis for background processing
- **Frontend**: Tailwind CSS, JavaScript
- **Templating**: Jinja2
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions for automated builds

## Project Layout

```
.
├── .github/workflows/main.yml
├── app/
│   ├── __init__.py
│   ├── celery_worker.py
│   ├── config.py
│   ├── main.py
│   ├── ocr.py
│   └── utils.py
├── config/
│   └── ai_prompts.json
├── static/
│   └── css/
│       ├── input.css
│       └── styles.css
├── templates/
│   ├── base.html
│   ├── chat.html
│   ├── compress.html
│   ├── contact_us.html
│   ├── index.html
│   ├── merge.html
│   ├── pdf_to_word.html
│   ├── processing.html
│   ├── summarize.html
│   ├── split.html
│   └── word_to_pdf.html
├── .dockerignore
├── .gitignore
├── .env
├── docker-compose.yml
├── Dockerfile
├── package.json
├── requirements.txt
└── setup.py
```

## Setup and Usage

### Prerequisites

- Python 3.10+
- Node.js and npm
- Docker and Docker Compose (for containerized setup)
- Redis

### Local Development

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/JoeUzo/Smart-PDF.git
    cd Smart-PDF
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install frontend dependencies**:
    ```bash
    npm install
    ```

5.  **Build the CSS**:
    ```bash
    npm run build
    ```
    For development with automatic rebuilding, use `npm run watch`.

6.  **Set up environment variables**:
    Create a `.env` file in the root directory and add your configuration:
    ```env
    OPENAI_API_KEY="your-openai-api-key"
    EMAIL="your-email@example.com"
    EMAIL_KEY="your-email-app-password"
    ```

7.  **Start the services**:
    - **Redis**: Ensure your Redis server is running.
    - **Celery Worker**:
      ```bash
      celery -A app.celery_worker.celery_app worker --loglevel=info
      ```
    - **FastAPI Application**:
      ```bash
      uvicorn app.main:app --reload --port 8001
      ```

    The application will be available at `http://localhost:8001`.

### Docker Deployment

1.  **Set up environment variables**:
    Create a `.env` file as described above.

2.  **Build and run the containers**:
    ```bash
    docker-compose up --build
    ```

    The application will be available at `http://localhost:8000`.

## API Endpoints

- **`GET /`**: Home page.
- **`GET /contact`**: Displays the contact form.
- **`POST /contact`**: Submits the contact form and sends an email.
- **`POST /merge`**: Merges multiple PDF files.
- **`POST /split`**: Splits a PDF based on specified ranges.
- **`POST /compress`**: Compresses a PDF file.
- **`POST /pdf-to-word`**: Converts a PDF to a DOCX file.
- **`POST /word-to-pdf`**: Converts a DOCX file to a PDF.
- **`POST /summarize`**: Creates a vector store and summary for a PDF.
- **`GET /chat`**: Handles chat interactions with the vectorized PDF.
- **`GET /status/{task_id}`**: Checks the status of a background task.
- **`GET /download/{task_id}/{filename}`**: Downloads a processed file.
