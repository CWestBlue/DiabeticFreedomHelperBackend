# Diabetic AI API

AI-powered diabetic data API built with FastAPI, LangGraph, and MongoDB.

## Features

- Chat with AI about diabetes management
- CareLink data sync and analysis
- Food scanning with nutritional estimation
- Dashboard metrics and insights

## Quick Start

```bash
# Install dependencies
uv sync

# Copy environment file
cp .env.example .env

# Run development server
uv run uvicorn diabetic_api.main:app --reload
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
| POST | `/food/scan` | Scan food image for nutritional analysis |

## Environment Variables

See `.env.example` for all available configuration options.
