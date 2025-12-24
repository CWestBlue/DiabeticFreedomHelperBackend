# Diabetic AI API

AI-powered diabetic data API built with FastAPI, LangGraph, and MongoDB.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Runtime | Python 3.12+ |
| Package Manager | uv |
| Web Framework | FastAPI |
| AI Orchestration | LangGraph |
| Database | MongoDB (Motor async driver) |
| Streaming | SSE (Server-Sent Events) |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- MongoDB instance

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Run development server
uv run uvicorn diabetic_api.main:app --reload
```

### Environment Variables

Create a `.env` file with:

```env
# MongoDB
MONGO_URI=mongodb://localhost:27017
DB_NAME=diabetic_db

# LLM Provider: "openai" or "gemini"
LLM_PROVIDER=gemini

# OpenAI (if using)
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o

# Google Gemini (if using)
GOOGLE_API_KEY=your-google-api-key
GEMINI_MODEL=gemini-2.5-flash

# Optional
LLM_TEMPERATURE=0.0
DEBUG=false
```

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` |
| `DB_NAME` | Database name | `diabetic_db` |
| `LLM_PROVIDER` | LLM provider (`openai` or `gemini`) | `gemini` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `LLM_TEMPERATURE` | LLM temperature (0.0-1.0) | `0.0` |
| `DEBUG` | Enable debug mode | `false` |

## Deployment

### Docker (Local/Dev)

```bash
# With bundled MongoDB
docker-compose up --build

# Development with hot reload
docker-compose -f docker-compose.dev.yml up
```

### TrueNAS / Production

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your settings
#    - Set MONGO_URI to your MongoDB (external or bundled)
#    - Add your GOOGLE_API_KEY or OPENAI_API_KEY

# 3a. With bundled MongoDB (simpler)
docker-compose up -d --build

# 3b. With external MongoDB (recommended for TrueNAS)
docker-compose -f docker-compose.yml -f docker-compose.truenas.yml up -d --build
```

#### TrueNAS External MongoDB Example

If you have MongoDB running elsewhere on your TrueNAS:

```env
MONGO_URI=mongodb://192.168.1.100:27017
# or
MONGO_URI=mongodb://truenas.local:27017
```

#### Resource Limits

The TrueNAS config sets:
- Memory limit: 1GB
- Memory reservation: 256MB

Adjust in `docker-compose.truenas.yml` if needed.

## Project Structure

```
backend/
├── src/diabetic_api/
│   ├── main.py              # FastAPI app entry
│   ├── core/                # Config, settings
│   ├── api/                 # Routes, dependencies
│   ├── services/            # Business logic
│   ├── db/                  # MongoDB, repositories, UoW
│   ├── agents/              # LangGraph agents
│   ├── models/              # Pydantic schemas
│   └── utils/               # Helpers
└── tests/                   # Test suite
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message, receive SSE stream |
| GET | `/sessions` | List chat sessions |
| POST | `/sessions` | Create new session |
| GET | `/sessions/{id}` | Get session with messages |
| GET | `/dashboard` | Get dashboard metrics |
| POST | `/upload` | Upload CSV data file |

## Architecture

```
Routes → Services → UnitOfWork → Repositories → MongoDB
                 ↘
                  LangGraph Agents (Router → QueryGen → Research)
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

