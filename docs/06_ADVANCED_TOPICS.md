# Advanced Topics and Production Deployment

## Table of Contents

1. [Threading vs Async](#threading-vs-async)
2. [API Key Security](#api-key-security)
3. [Error Recovery Patterns](#error-recovery-patterns)
4. [Performance Optimization](#performance-optimization)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Testing Strategies](#testing-strategies)
8. [Scaling Considerations](#scaling-considerations)

---

## Threading vs Async

### Threading (Current Implementation)

RelevanceSearch uses threading for web scraping:

```python
import threading

def scrape_url_thread(thread_id, urls, results, lock):
    """Thread worker function"""
    url = urls[thread_id]
    content = scrape_url(url)

    # Thread-safe write
    with lock:
        results.append(content)

# Launch threads
threads = []
results = []
lock = threading.Lock()

for i in range(len(urls)):
    thread = threading.Thread(
        target=scrape_url_thread,
        args=(i, urls, results, lock)
    )
    threads.append(thread)
    thread.start()

# Wait for all threads
for thread in threads:
    thread.join()
```

**When to use Threading:**
- ✓ I/O-bound tasks (network requests, file I/O)
- ✓ Simple to understand and implement
- ✓ Works well with existing libraries
- ✗ Not ideal for CPU-bound tasks (GIL limitation)
- ✗ Harder to debug than async

### Async/Await Alternative

```python
import asyncio
import aiohttp

async def scrape_url_async(session, url):
    """Async scraping function"""
    async with session.get(url) as response:
        html = await response.text()
        return parse_html(html)

async def scrape_all_urls(urls):
    """Scrape multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_url_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

# Usage
urls = ["https://example1.com", "https://example2.com", ...]
results = asyncio.run(scrape_all_urls(urls))
```

**When to use Async:**
- ✓ Many concurrent I/O operations (1000+ requests)
- ✓ More efficient memory usage
- ✓ Better performance at scale
- ✗ Steeper learning curve
- ✗ Requires async-compatible libraries (aiohttp, not requests)

### Comparison

| Aspect | Threading | Async |
|--------|-----------|-------|
| **Concurrency** | 10-100 tasks | 1000+ tasks |
| **Memory** | ~8MB per thread | ~100KB per task |
| **Complexity** | Moderate | High |
| **Debugging** | Hard | Harder |
| **Libraries** | Most work | Need async versions |
| **Performance** | Good | Excellent |

**Recommendation for RelevanceSearch:**
- **Current scale (10 URLs)**: Threading is perfect ✓
- **If scaling to 100+ URLs**: Consider async migration

---

## API Key Security

### Never Hardcode Keys

```python
# ❌ NEVER do this
API_KEY = "sk-or-v1-abc123..."

# ✓ Use environment variables
import os
API_KEY = os.getenv("OPENROUTER_API_KEY")

# ✓ Use config files (excluded from git)
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    API_KEY = config['api_key']
```

### .gitignore for Sensitive Files

```gitignore
# Add to .gitignore
config.yaml
.env
secrets/
*.key
api_keys.txt
```

### Environment Variables

```bash
# .env file (never commit this)
OPENROUTER_API_KEY=sk-or-v1-...
SERPER_API_KEY=abc123...
GITEE_API_KEY=xyz789...
```

```python
# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment")
```

### Temporary Config Files (RelevanceSearch Approach)

```python
# Read user input
openrouter_key = st.text_input("API Key", type="password")

# Create temporary config
import tempfile
import yaml

config = {
    "openrouter_api_key": openrouter_key,
    "serper_api_key": serper_key
}

# Write to temp file
temp_file = tempfile.NamedTemporaryFile(
    mode='w',
    delete=False,
    suffix='.yaml'
)
yaml.dump(config, temp_file)
temp_file.close()

# Use config
# (Config is automatically deleted when Python exits)
```

### Key Rotation

```python
class APIKeyManager:
    def __init__(self, keys):
        self.keys = keys  # List of API keys
        self.current_index = 0

    def get_key(self):
        """Get current API key"""
        return self.keys[self.current_index]

    def rotate_key(self):
        """Switch to next key (for rate limit management)"""
        self.current_index = (self.current_index + 1) % len(self.keys)
        logger.info(f"Rotated to key #{self.current_index}")

# Usage
keys = [os.getenv("KEY1"), os.getenv("KEY2"), os.getenv("KEY3")]
manager = APIKeyManager(keys)

# If rate limited, rotate
try:
    response = api_call(manager.get_key())
except RateLimitError:
    manager.rotate_key()
    response = api_call(manager.get_key())
```

---

## Error Recovery Patterns

### Retry with Exponential Backoff

```python
import time
import random

def retry_with_backoff(func, max_retries=3, base_delay=1):
    """
    Retry function with exponential backoff

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds

    Returns:
        Function result or raises last exception
    """
    for attempt in range(max_retries):
        try:
            return func()

        except Exception as e:
            if attempt == max_retries - 1:
                # Last attempt, give up
                raise

            # Calculate delay with jitter
            delay = base_delay * (2 ** attempt)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter

            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {total_delay:.2f}s..."
            )
            time.sleep(total_delay)

# Usage
result = retry_with_backoff(
    lambda: requests.get("https://api.example.com/data"),
    max_retries=5,
    base_delay=2
)
```

**Backoff Timeline:**
```
Attempt 1: Fails → Wait 2s
Attempt 2: Fails → Wait 4s
Attempt 3: Fails → Wait 8s
Attempt 4: Fails → Wait 16s
Attempt 5: Succeeds!
```

### Circuit Breaker Pattern

Prevents cascading failures when a service is down:

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func):
        """Execute function with circuit breaker"""

        # Check if circuit is open
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"

        try:
            result = func()

            # Success: reset circuit
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0

            return result

        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()

            # Open circuit if threshold exceeded
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.error("Circuit breaker opened!")

            raise

# Usage
serper_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

def search_with_breaker(query):
    return serper_breaker.call(lambda: serper_client.search(query))
```

**States:**
- **CLOSED**: Normal operation
- **OPEN**: Service unavailable, fail fast (don't even try)
- **HALF_OPEN**: Testing if service recovered

### Fallback Strategies

```python
def search_with_fallback(query):
    """Try multiple search services"""

    # Primary: Serper
    try:
        return serper_search(query)
    except Exception as e:
        logger.warning(f"Serper failed: {e}")

    # Fallback 1: Bing
    try:
        return bing_search(query)
    except Exception as e:
        logger.warning(f"Bing failed: {e}")

    # Fallback 2: DuckDuckGo
    try:
        return duckduckgo_search(query)
    except Exception as e:
        logger.error(f"All search services failed: {e}")

    # Last resort: return cached results or error
    return get_cached_results(query) or []
```

---

## Performance Optimization

### 1. Caching Strategies

#### In-Memory Cache

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def embed_text(text):
    """Cache embeddings (up to 128 items)"""
    return embedding_model.encode(text)

# Same input → Instant return (no API call)
emb1 = embed_text("quantum physics")  # API call
emb2 = embed_text("quantum physics")  # Cached! ⚡
```

#### Redis Cache

```python
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379)

def get_embedding_cached(text):
    """Get embedding from Redis cache or compute"""
    cache_key = f"embedding:{hash(text)}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Not in cache, compute
    embedding = embed_text(text)

    # Store in cache (expire after 1 hour)
    redis_client.setex(
        cache_key,
        3600,
        json.dumps(embedding.tolist())
    )

    return embedding
```

### 2. Database Optimization

#### Persistent ChromaDB

```python
import chromadb

# Instead of in-memory (ephemeral)
client = chromadb.Client()

# Use persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Now embeddings persist across restarts
collection = client.get_or_create_collection("search_results")
```

#### Batch Operations

```python
# ❌ Slow: Individual inserts
for doc in documents:
    collection.add(documents=[doc], ids=[doc.id])

# ✓ Fast: Batch insert
collection.add(
    documents=[doc.content for doc in documents],
    ids=[doc.id for doc in documents],
    metadatas=[doc.metadata for doc in documents]
)
```

### 3. Request Optimization

#### Connection Pooling

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Create session with connection pooling
session = requests.Session()

# Configure retries
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,  # Connection pool size
    pool_maxsize=10
)

session.mount("http://", adapter)
session.mount("https://", adapter)

# Reuse session for multiple requests
response1 = session.get(url1)  # Creates connection
response2 = session.get(url2)  # Reuses connection ⚡
```

#### Batch API Calls

```python
# ❌ Slow: 100 individual API calls
embeddings = []
for text in texts:  # 100 texts
    emb = api.embed(text)  # 100ms per call → 10 seconds
    embeddings.append(emb)

# ✓ Fast: Single batch API call
embeddings = api.embed_batch(texts)  # 1 second total ⚡
```

### 4. Code Profiling

```python
import cProfile
import pstats

def profile_function(func):
    """Profile a function's performance"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 slowest functions

    return result

# Usage
result = profile_function(lambda: search_and_answer(query))
```

**Example Output:**
```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001   10.234   10.234 app.py:150(search_and_answer)
       10    0.052    0.005    8.123    0.812 web_crawler.py:45(scrape_url)
        1    1.234    1.234    1.234    1.234 retrieval.py:78(embed_documents)
```

---

## Production Deployment

### Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run application
CMD ["streamlit", "run", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  searchgpt:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - SERPER_API_KEY=${SERPER_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

**Build and run:**
```bash
docker-compose up -d
```

### Cloud Deployment

#### Streamlit Community Cloud (Free)

```bash
# 1. Create requirements.txt
# 2. Push to GitHub
# 3. Go to https://streamlit.io/cloud
# 4. Connect repository
# 5. Add secrets (API keys) in dashboard
# 6. Deploy!
```

**secrets.toml** (in Streamlit Cloud):
```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
SERPER_API_KEY = "abc123..."
```

**Access in code:**
```python
import streamlit as st

api_key = st.secrets["OPENROUTER_API_KEY"]
```

#### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.11 searchgpt

# Create environment
eb create searchgpt-env

# Deploy
eb deploy

# Open in browser
eb open
```

#### Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Create runtime.txt
echo "python-3.11.0" > runtime.txt

# Deploy
heroku create searchgpt-app
git push heroku main
heroku open
```

### Health Checks

```python
# Add health check endpoint (for load balancers)

@st.cache_data
def health_check():
    """Check if services are operational"""
    checks = {}

    # Check OpenRouter
    try:
        requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        checks['openrouter'] = "OK"
    except:
        checks['openrouter'] = "FAIL"

    # Check Serper
    try:
        requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": serper_key},
            json={"q": "test"},
            timeout=5
        )
        checks['serper'] = "OK"
    except:
        checks['serper'] = "FAIL"

    return checks

# Display in sidebar
if st.sidebar.checkbox("Show Health Status"):
    health = health_check()
    for service, status in health.items():
        if status == "OK":
            st.sidebar.success(f"✅ {service}: {status}")
        else:
            st.sidebar.error(f"❌ {service}: {status}")
```

---

## Monitoring and Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing"""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger("searchgpt")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Search started", extra={"query": query, "user_id": user_id})
```

**Output:**
```json
{
  "timestamp": "2025-11-28T01:23:45.678Z",
  "level": "INFO",
  "logger": "searchgpt",
  "message": "Search started",
  "query": "quantum computing",
  "user_id": "user123"
}
```

### Application Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
search_requests = Counter(
    'search_requests_total',
    'Total search requests',
    ['model', 'status']
)

search_duration = Histogram(
    'search_duration_seconds',
    'Search request duration'
)

embedding_duration = Histogram(
    'embedding_duration_seconds',
    'Embedding generation duration'
)

# Start metrics server (on different port)
start_http_server(8000)

# Instrument code
@search_duration.time()
def perform_search(query):
    try:
        result = search(query)
        search_requests.labels(model='grok', status='success').inc()
        return result
    except Exception as e:
        search_requests.labels(model='grok', status='error').inc()
        raise
```

**Access metrics:**
```bash
curl http://localhost:8000/metrics
```

### Error Tracking (Sentry)

```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://...@sentry.io/...",
    traces_sample_rate=1.0,
    environment="production"
)

# Automatically captures exceptions
try:
    result = search(query)
except Exception as e:
    # Sentry automatically captures this
    raise
```

---

## Testing Strategies

### Unit Tests

```python
import pytest
from web_crawler import WebScraper

def test_scrape_url():
    """Test URL scraping"""
    scraper = WebScraper()
    content = scraper.scrape_url("https://example.com")

    assert len(content) > 0
    assert "Example Domain" in content

def test_scrape_invalid_url():
    """Test handling of invalid URLs"""
    scraper = WebScraper()

    with pytest.raises(requests.exceptions.RequestException):
        scraper.scrape_url("https://invalid-url-12345.com")

def test_extract_pdf():
    """Test PDF extraction"""
    scraper = WebScraper()
    content = scraper.extract_pdf_content(pdf_response)

    assert len(content) > 100
    assert "quantum" in content.lower()
```

### Integration Tests

```python
def test_full_search_pipeline():
    """Test complete search flow"""

    query = "What is machine learning?"

    # Step 1: Search
    fetcher = WebContentFetcher(query)
    contents, response = fetcher.fetch()

    assert len(contents) > 0
    assert response['count'] > 0

    # Step 2: Embeddings
    retriever = EmbeddingRetriever()
    docs = retriever.retrieve_embeddings(
        contents,
        response['links'],
        query
    )

    assert len(docs) > 0

    # Step 3: Answer generation
    answer_generator = GPTAnswer()
    answer = answer_generator.generate_answer(query, docs, response)

    assert len(answer) > 100
    assert "[1]" in answer  # Check citations
```

### Mocking External APIs

```python
from unittest.mock import Mock, patch

@patch('serper_service.requests.post')
def test_serper_search(mock_post):
    """Test Serper search with mocked API"""

    # Mock API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "organic": [
            {"title": "Test", "link": "https://test.com", "snippet": "Test snippet"}
        ]
    }
    mock_post.return_value = mock_response

    # Test
    client = SerperClient()
    result = client.serper("test query")

    assert len(result['organic']) == 1
    assert result['organic'][0]['title'] == "Test"
```

### Performance Tests

```python
import time

def test_search_performance():
    """Ensure search completes within time limit"""

    query = "quantum computing"

    start = time.time()
    result = perform_search(query)
    duration = time.time() - start

    assert duration < 15.0  # Must complete in under 15 seconds
    assert len(result) > 0

def test_concurrent_searches():
    """Test handling multiple concurrent searches"""
    import concurrent.futures

    queries = ["query1", "query2", "query3"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(perform_search, q) for q in queries]
        results = [f.result() for f in futures]

    assert len(results) == 3
    assert all(len(r) > 0 for r in results)
```

---

## Scaling Considerations

### Horizontal Scaling

```python
# Use load balancer (NGINX, AWS ALB) with multiple instances

# Instance 1: Running on port 8501
# Instance 2: Running on port 8502
# Instance 3: Running on port 8503

# Load balancer distributes traffic
```

```nginx
# nginx.conf
upstream streamlit {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    location / {
        proxy_pass http://streamlit;
    }
}
```

### Rate Limiting

```python
from functools import wraps
import time

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()

            # Remove old calls
            self.calls = [c for c in self.calls if c > now - self.period]

            # Check limit
            if len(self.calls) >= self.max_calls:
                raise Exception(f"Rate limit exceeded: {self.max_calls} calls per {self.period}s")

            # Record call
            self.calls.append(now)

            return func(*args, **kwargs)

        return wrapper

# Usage: Max 10 calls per minute
@RateLimiter(max_calls=10, period=60)
def search_with_limit(query):
    return search(query)
```

### Queue-Based Processing

```python
from celery import Celery
import redis

app = Celery('searchgpt', broker='redis://localhost:6379')

@app.task
def process_search_async(query):
    """Process search in background"""
    result = perform_search(query)
    return result

# In Streamlit app
if st.button("Search"):
    # Submit to queue
    task = process_search_async.delay(query)

    # Poll for result
    with st.spinner("Processing..."):
        while not task.ready():
            time.sleep(0.5)

    result = task.get()
    st.write(result)
```

---

## Summary

### Production Checklist

- [ ] API keys stored securely (not in code)
- [ ] Error handling with retries
- [ ] Logging configured (structured JSON)
- [ ] Metrics collection (Prometheus/Sentry)
- [ ] Unit and integration tests
- [ ] Performance profiling done
- [ ] Caching implemented
- [ ] Docker containerization
- [ ] Health checks added
- [ ] Rate limiting configured
- [ ] Monitoring dashboard set up

### Performance Targets

| Metric | Target |
|--------|--------|
| Search latency | < 15 seconds |
| Embedding generation | < 2 seconds |
| LLM response time | < 10 seconds |
| Uptime | > 99% |
| Error rate | < 1% |

### Next Steps

Congratulations! You've completed the RelevanceSearch learning series. You now understand:

✓ RAG system architecture
✓ Vector embeddings and similarity search
✓ LLM integration and streaming
✓ Building interactive UIs with Streamlit
✓ Production deployment and scaling

**Continue learning:**
- Build your own RAG variations
- Experiment with different embedding models
- Try alternative LLMs
- Contribute to the RelevanceSearch project
- Share your knowledge with others!

Happy building! 🚀
