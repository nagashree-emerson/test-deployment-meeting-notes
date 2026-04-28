# Meeting Notes Summarizer Agent

A professional agent that processes raw meeting transcripts, chat exports, or notes and generates a structured, concise, and actionable summary. It extracts key sections, action items, decisions, and attendees, formats the output as an email-ready summary, and supports follow-up questions—all with strong privacy and compliance guarantees.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

**Direct execution:**
```
python code/agent.py
```

**As a FastAPI server:**
```
uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`
- `AGENT_ID`
- `PROJECT_NAME`
- `PROJECT_ID`
- `SERVICE_NAME`
- `SERVICE_VERSION`

**General**
- `ENVIRONMENT`

**Azure Key Vault**
- `USE_KEY_VAULT`
- `KEY_VAULT_URI`
- `AZURE_USE_DEFAULT_CREDENTIAL`

**Azure Authentication**
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `AZURE_OPENAI_ENDPOINT`

**API Keys / Secrets**
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_CONTENT_SAFETY_KEY`
- `OBS_AZURE_SQL_PASSWORD`
- `AGENT_ENCRYPTION_KEY`

**Service Endpoints**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_CONTENT_SAFETY_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`

**Observability DB**
- `OBS_DATABASE_TYPE`
- `OBS_AZURE_SQL_SERVER`
- `OBS_AZURE_SQL_DATABASE`
- `OBS_AZURE_SQL_PORT`
- `OBS_AZURE_SQL_USERNAME`
- `OBS_AZURE_SQL_SCHEMA`
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**Agent-Specific**
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`
- `VALIDATION_CONFIG_PATH`
- `LLM_MODELS`
- `VERSION`
- `CONTENT_SAFETY_ENABLED`
- `CONTENT_SAFETY_SEVERITY_THRESHOLD`

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

### **POST** `/summarize`
- **Description:** Summarize meeting from transcript text.
- **Request body:**
  ```
  {
    "transcript_text": "string (required)",
    "summary_length": "one-liner | paragraph | full (optional, default: full)",
    "user_email": "string (required)"
  }
  ```
- **Response:**
  ```
  {
    "success": true|false,
    "summary": "string|null",
    "structured_summary": { ... }|null,
    "error": "string|null",
    "tips": "string|null"
  }
  ```

### **POST** `/summarize_file`
- **Description:** Summarize meeting from uploaded file (.txt, .docx, .pdf).
- **Request body:** (multipart/form-data)
  - `file`: File upload (required)
  - `summary_length`: one-liner | paragraph | full (optional, default: full)
  - `user_email`: string (required)
- **Response:**
  ```
  {
    "success": true|false,
    "summary": "string|null",
    "structured_summary": { ... }|null,
    "error": "string|null",
    "tips": "string|null"
  }
  ```

### **POST** `/followup`
- **Description:** Answer follow-up question about meeting content.
- **Request body:**
  ```
  {
    "transcript_text": "string (required)",
    "question": "string (required)"
  }
  ```
- **Response:**
  ```
  {
    "success": true|false,
    "answer": "string|null",
    "error": "string|null",
    "tips": "string|null"
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t Meeting Notes Summarizer Agent -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name Meeting Notes Summarizer Agent Meeting Notes Summarizer Agent
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs Meeting Notes Summarizer Agent
```

### 7. Stop the container:
```
docker stop Meeting Notes Summarizer Agent
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Meeting Notes Summarizer Agent** — Instantly transforms raw meeting transcripts into structured, actionable summaries with privacy and compliance built in.
