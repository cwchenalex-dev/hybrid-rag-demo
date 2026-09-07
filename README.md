# Hybrid RAG Knowledge Base Q&A

A lightweight knowledge-base Q&A demo that combines BM25 and vector retrieval using Reciprocal Rank Fusion (RRF), with grounded LLM generation, source citations, and abstention for unsupported questions.

## Overview

The application answers questions based on a small Markdown knowledge base.

Documents are parsed into sections and indexed through two complementary retrieval approaches:

- **BM25** for lexical keyword matching
- **Vector Search** for semantic similarity

Retrieved results are filtered and combined using **Reciprocal Rank Fusion (RRF)**. The top-ranked context is then provided to an LLM to generate an answer with source citations.

If relevant context cannot be found or the retrieved context is insufficient, the system returns a fallback response instead of generating an unsupported answer.

## Architecture

```text
Markdown Documents
        |
        v
Parsing / Chunking
        |
        +-------------------+
        |                   |
        v                   v
      BM25             Vector Search
        |                   |
        v                   v
 Relevance Filter     Relevance Filter
        |                   |
        +---------+---------+
                  |
                  v
             RRF Fusion
                  |
                  v
            Top-K Context
                  |
                  v
                 LLM
                  |
                  v
          Answer + Sources
```

## Features

- Markdown knowledge-base ingestion
- BM25 lexical retrieval
- FAISS vector retrieval
- Relevance filtering
- Reciprocal Rank Fusion (RRF)
- Grounded LLM generation
- Source citation validation
- Abstention for unsupported questions
- FastAPI REST API
- Lightweight web interface
- Docker support
- Basic automated tests

## Project Structure

```text
hybrid-rag-demo/
├── app/
│   ├── static/
│   │   └── index.html
│   ├── indexer.py
│   ├── main.py
│   ├── retrieval.py
│   ├── routes.py
│   └── schemas.py
├── docs/
│   ├── account_help.md
│   ├── refund_policy.md
│   └── shipping_faq.md
├── tests/
│   ├── test_api.py
│   └── test_retrieval.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements-dev.txt
└── requirements.txt
```

## Quick Start

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file based on `.env.example`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

RAG_MIN_VECTOR_RELEVANCE=0.35
RAG_MIN_BM25_SCORE=0.50
```

Do not commit the `.env` file.

### 4. Start the application

```bash
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Build the knowledge-base index before sending questions.

## API

### Health Check

```http
GET /health
```

### Build Index

```http
POST /index
```

Parses the Markdown documents and builds the BM25 and FAISS indexes.

### Ask a Question

```http
POST /chat
Content-Type: application/json
```

Example request:

```json
{
  "query": "How long does standard shipping take?"
}
```

The response contains the generated answer and retrieved sources.

## Docker

Build and start the application:

```bash
docker compose build
docker compose up -d
```

Check the API:

```bash
curl http://127.0.0.1:8000/health
```

Build the index:

```bash
curl -X POST http://127.0.0.1:8000/index
```

Stop the application:

```bash
docker compose down
```

The generated index is stored under the local `.kb/` directory and is excluded from Git.

## Testing

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run the tests:

```bash
python -m pytest -v
```

The test suite intentionally focuses on lightweight, deterministic checks such as API availability and basic retrieval behavior without requiring external LLM calls.
