# HALLUCINATION_GUARD Backend Setup Guide

## Project Structure

```
backend/
├── main.py                    # FastAPI entry point
├── config.py                  # Configuration management
├── database.py                # Database setup and session management
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (local)
├── .gitignore                # Git ignore file
│
├── models/                   # SQLAlchemy ORM models
│   ├── __init__.py
│   ├── job.py               # Job model
│   ├── stage.py             # Stage model (1-8)
│   └── result.py            # Result model
│
├── schemas/                  # Pydantic input/output schemas
│   ├── __init__.py
│   ├── job_schemas.py       # Job request/response schemas
│   └── event_schemas.py     # WebSocket event schemas
│
├── services/                 # Business logic services
│   ├── __init__.py
│   └── job_manager.py       # Job lifecycle management
│
├── utils/                    # Utility modules
│   ├── __init__.py
│   └── logging_config.py    # Logging configuration
│
├── routes/                   # API route handlers (PHASE 3-6)
├── workers/                  # Celery task workers (PHASE 4-5)
└── tests/                    # Unit and integration tests
```

## Installation & Setup

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Create Virtual Environment
```bash
# On Windows (recommended Python 3.12)
py -3.12 -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3.12 -m venv venv
source venv/bin/activate
```

> Python 3.14 is not compatible with the pinned packages in this project, so using Python 3.12 is recommended.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If you want the optional ML/RAG stack for advanced features, install it separately:
```bash
pip install -r requirements-ml.txt
```

### 4. Configure Environment
The `.env` file is already configured with default values. Modify as needed:

```bash
# Database (default: SQLite for development)
DATABASE_URL=sqlite:///./hallucination_guard.db

# For Production (PostgreSQL):
# DATABASE_URL=postgresql://user:password@localhost:5432/hallucination_guard

# Redis for job queue
REDIS_URL=redis://localhost:6379/0

# Development mode (adds realistic delays for UI testing)
DEV_MODE=True
STAGE_DELAY_MIN_MS=500
STAGE_DELAY_MAX_MS=1500

# API configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:8000,http://localhost:5173

# LLM Models
LLM_MODEL=meta-llama/Llama-2-7b-chat-hf
VLM_MODEL=Qwen/Qwen-VL-Chat
```

### 5. Initialize Database
```bash
python -c "from database import init_db; init_db()"
```

## Running the Backend

### Start FastAPI Server
```bash
# With auto-reload (development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Interactive API**: `http://localhost:8000/redoc` (ReDoc)

### Start Redis (for job queue)
```bash
# Make sure Redis is installed
redis-server

# Or with Docker
docker run -d -p 6379:6379 redis:alpine
```

### Start Celery Worker (PHASE 5)
```bash
# After Celery service is implemented
celery -A workers.tasks worker --loglevel=info
```

## PHASE 1 Completion ✅

**What's been created:**

- ✅ Database models (Job, Stage, Result)
- ✅ Pydantic schemas for validation
- ✅ FastAPI entry point with basic endpoints
- ✅ Job manager service
- ✅ Database initialization
- ✅ Configuration management
- ✅ Logging setup
- ✅ Environment configuration

**Available API Endpoints (PHASE 1):**

```
GET    /                       # Health check
POST   /api/analyze            # Submit analysis job
GET    /api/job/{job_id}       # Get job status
GET    /api/result/{job_id}    # Get final result
DELETE /api/job/{job_id}       # Cancel job
```

## Next Phases

- **PHASE 2**: Implement 8-stage pipeline orchestration
- **PHASE 3**: Add WebSocket real-time progress updates
- **PHASE 4**: Implement job queue system (Redis/Celery)
- **PHASE 5**: Async pipeline execution
- **PHASE 6**: Complete API endpoints

## Testing

```bash
# Run tests (when available in PHASE 9)
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## Development Tips

1. **Debug Mode**: Set `DEBUG=True` in `.env` for detailed logging
2. **Dev Delays**: Set `DEV_MODE=True` to add realistic 0.5-1.5s delays per stage
3. **Check Logs**: Logs are saved to `logs/hallucination_guard_YYYYMMDD.log`
4. **Database Reset**: `python -c "from database import drop_db; drop_db()"` (use cautiously!)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 already in use | `netstat -ano \| findstr :8000` (Windows) or `lsof -i :8000` (macOS/Linux) then kill the process |
| Database lock error | Delete `hallucination_guard.db` and reinit: `python -c "from database import init_db; init_db()"` |
| Redis connection refused | Make sure Redis is running: `redis-server` or Docker container |
| Import errors | Ensure virtual environment is activated and packages installed: `pip install -r requirements.txt` |

## Current Status

**PHASE 1 Status**: ✅ **COMPLETE**

- Database infrastructure ready
- Basic API endpoints functional
- Configuration system in place
- Ready for PHASE 2 (Pipeline stages)

To proceed to PHASE 2, run: `python main.py` and verify all endpoints at `http://localhost:8000/docs`
