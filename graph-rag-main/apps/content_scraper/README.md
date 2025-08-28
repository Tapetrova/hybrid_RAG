# Content Scraper Service

A high-performance web content scraping service built with FastAPI, Celery, and Playwright.

## Features

- 🚀 **Asynchronous Processing**: Uses Celery for distributed task processing
- 🌐 **Web Scraping**: Playwright-based scraping with JavaScript rendering support
- 🔍 **Google Search Integration**: Serper.dev API integration for search results
- 💾 **Smart Caching**: Redis-based caching to avoid redundant scraping
- 📊 **Database Storage**: PostgreSQL for persistent storage
- 🎯 **RESTful API**: FastAPI-based endpoints for easy integration
- 🐳 **Docker Support**: Complete Docker and Docker Compose setup
- 📈 **Monitoring**: Flower dashboard for Celery task monitoring

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │────▶│    Redis    │◀────│   Celery    │
│     API     │     │   (Queue)   │     │   Worker    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                            │
                      ┌─────────────┐
                      │ PostgreSQL  │
                      │  Database   │
                      └─────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Serper.dev API key

### Using Docker Compose

1. Clone the repository and navigate to the content scraper directory:
```bash
cd apps/content_scraper
```

2. Copy the example environment file and configure:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start all services:
```bash
docker-compose up -d
```

4. Check service health:
```bash
python healthcheck.py
```

5. Access the services:
- API: http://localhost:8099
- API Docs: http://localhost:8099/docs
- Flower Dashboard: http://localhost:5555

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

2. Set up PostgreSQL and Redis:
```bash
# Using Docker for dependencies only
docker-compose up -d postgres redis
```

3. Run database migrations:
```bash
python -c "from libs.python.databases.database import db; import asyncio; asyncio.run(db.gino.create_all())"
```

4. Start the API server:
```bash
uvicorn apps.content_scraper.app:app --reload --port 8099
```

5. Start Celery worker:
```bash
celery -A apps.content_scraper.celery_worker worker --loglevel=info
```

6. (Optional) Start Flower:
```bash
celery -A apps.content_scraper.celery_worker flower
```

## API Endpoints

### Search Content
```http
POST /search_content
Content-Type: application/json

{
  "user_id": "user_123",
  "session_id": "session_456",
  "dialog_id": "dialog_789",
  "query": "python web scraping",
  "config": {
    // Configuration object
  }
}
```

### Get Content
```http
POST /get_content
Content-Type: application/json

{
  "query": "python web scraping"
}
```

### Get Content by URLs
```http
POST /data/get_content_source_by_urls
Content-Type: application/json

{
  "urls": [
    "https://example.com/page1",
    "https://example.com/page2"
  ]
}
```

### Health Check
```http
GET /health
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SERPER_API_KEY` | Serper.dev API key (required) | - |
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql://...` |
| `REDIS_HOST` | Redis hostname | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_PASSWORD_LLM` | Redis password | - |
| `PLAYWRIGHT_TIMEOUT_MS` | Playwright timeout in ms | `180000` |
| `API_PORT` | API server port | `8099` |

See `.env.example` for complete configuration options.

## Troubleshooting

### Common Issues

1. **Playwright installation fails**:
   ```bash
   playwright install-deps
   playwright install chromium
   ```

2. **Redis connection error**:
   - Check Redis is running: `docker-compose ps redis`
   - Verify password in `.env` matches Docker Compose config

3. **Database connection error**:
   - Ensure PostgreSQL is running: `docker-compose ps postgres`
   - Check database credentials in `.env`

4. **Serper API errors**:
   - Verify `SERPER_API_KEY` is set correctly
   - Check API quota at https://serper.dev

### Logs

View logs for debugging:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f celery_worker
```

## Development

### Code Structure

```
content_scraper/
├── app.py              # FastAPI application
├── celery_worker.py    # Celery tasks
├── models.py           # Pydantic models
├── services.py         # Business logic
├── config.py           # Configuration management
├── scrapper.py         # Scraping logic
├── routers/
│   └── data.py        # API routes
└── utils/
    ├── serper.py      # Google search integration
    └── ...            # Other utilities
```

### Testing

Run tests:
```bash
pytest tests/ -v
```

### Linting

```bash
# Format code
black .
isort .

# Type checking
mypy .

# Linting
flake8 .
```

## Performance

- **Caching**: Content is cached for 15 minutes by default
- **Concurrent Scraping**: Celery workers can process multiple URLs in parallel
- **Rate Limiting**: Configurable rate limiting for API endpoints
- **Resource Management**: Playwright browsers are properly closed after use

## Security

- API key authentication for Serper.dev
- PostgreSQL connection uses SSL in production
- Redis password protection
- Input validation on all endpoints
- No sensitive data in logs

## License

See main project LICENSE file.