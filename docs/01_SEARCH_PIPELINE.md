# Step 1 & 2: Search and Web Scraping Pipeline

## Overview

This document explains how RelevanceSearch discovers and extracts information from the web - the first two steps of the RAG pipeline.

```
User Query → Serper API → URLs → Web Crawler → Text Content
```

## Component 1: Serper Service (serper_service.py)

### What is Serper?

Serper.dev provides a simple API to access Google Search results programmatically. Instead of scraping Google (which violates their terms of service), we use Serper as an authorized intermediary.

### Code Walkthrough

#### 1. Initialization

```python
class SerperClient:
    def __init__(self):
        # Load API key from config file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        self.url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": config["serper_api_key"],
            "Content-Type": "application/json"
        }
```

**Key Points:**
- Reads API key from `config.yaml` (never hardcoded)
- Sets up HTTP headers for API authentication

#### 2. Performing Search

```python
def serper(self, query: str):
    # Configure search parameters
    serper_settings = {"q": query}

    # Detect Chinese and adjust settings
    if self._contains_chinese(query):
        serper_settings.update({"gl": "cn", "hl": "zh-cn"})

    # Make API request
    response = requests.post(
        self.url,
        headers=self.headers,
        data=json.dumps(serper_settings),
        timeout=30
    )

    return response.json()
```

**What's Happening:**
1. Creates search query parameters
2. Detects language (Chinese gets `gl=cn` for China-specific results)
3. Sends POST request to Serper API
4. Returns JSON response

**Example Response:**
```json
{
  "searchParameters": {
    "q": "quantum computing",
    "type": "search"
  },
  "organic": [
    {
      "title": "Quantum Computing - Wikipedia",
      "link": "https://en.wikipedia.org/wiki/Quantum_computing",
      "snippet": "Quantum computing is a type of computation that harnesses..."
    },
    {
      "title": "What is Quantum Computing? | IBM",
      "link": "https://www.ibm.com/quantum-computing",
      "snippet": "Quantum computers use quantum bits, or qubits..."
    }
    // ... up to 10 results
  ]
}
```

#### 3. Extracting Components

```python
def extract_components(self, serper_response: dict):
    titles, links, snippets = [], [], []

    # Extract from organic results
    for item in serper_response.get("organic", []):
        titles.append(item.get("title", ""))
        links.append(item.get("link", ""))
        snippets.append(item.get("snippet", ""))

    # Detect language
    query = serper_response.get("searchParameters", {}).get("q", "")
    language = "zh-cn" if self._contains_chinese(query) else "en-us"

    return {
        'query': query,
        'language': language,
        'count': len(links),
        'titles': titles,
        'links': links,
        'snippets': snippets
    }
```

**Output Structure:**
```python
{
    'query': 'quantum computing',
    'language': 'en-us',
    'count': 10,
    'titles': ['Title 1', 'Title 2', ...],
    'links': ['https://...', 'https://...', ...],
    'snippets': ['Snippet 1', 'Snippet 2', ...]
}
```

### Language Detection

```python
def _contains_chinese(self, query: str):
    # Unicode range for Chinese characters: U+4E00 to U+9FFF
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(pattern.search(query))
```

**Why This Matters:**
- Chinese queries get results from Chinese search index
- Affects language of AI's final answer
- Ensures culturally relevant results

---

## Component 2: Web Crawler (web_crawler.py)

### Purpose

Takes a URL and extracts readable text content, handling both HTML pages and PDF documents.

### Architecture

```
URL → HTTP Request → Response
                     ↓
            ┌────────┴────────┐
            │                 │
          HTML              PDF
            │                 │
       BeautifulSoup      PyPDF2
            │                 │
            └────────┬────────┘
                     ↓
                  Clean Text
```

### Code Walkthrough

#### 1. User Agent Setup

```python
class WebScraper:
    def __init__(self, user_agent='macOS'):
        self.headers = self._get_headers(user_agent)

    def _get_headers(self, user_agent):
        if user_agent == 'macOS':
            return {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...',
                'sec-ch-ua': '"Google Chrome";v="115"',
                ...
            }
```

**Why User Agents?**
- Websites block requests without proper user agents
- Makes our bot look like a real browser
- Reduces chances of being blocked

#### 2. Fetching HTML

```python
def get_webpage_html(self, url):
    response = requests.Response()  # Empty default

    try:
        response = requests.get(
            url,
            headers=self.headers,
            timeout=15  # 15-second timeout
        )
        response.encoding = "utf-8"
    except requests.exceptions.Timeout:
        # Return empty response on timeout
        return response

    return response
```

**Error Handling:**
- 15-second timeout prevents hanging on slow sites
- Returns empty response instead of crashing
- UTF-8 encoding ensures proper text handling

#### 3. Parsing HTML

```python
def convert_html_to_soup(self, html):
    html_string = html.text
    return BeautifulSoup(html_string, "lxml")
```

**BeautifulSoup:**
- Parses HTML into navigable tree structure
- Uses `lxml` parser (fast and lenient)
- Handles malformed HTML gracefully

#### 4. Extracting Main Content

```python
def extract_main_content(self, html_soup, rule=0):
    main_content = []

    # Define which HTML tags to extract
    tag_rule = re.compile("^(h[1-6]|p|div)" if rule == 1 else "^(h[1-6]|p)")

    # Find all matching tags
    for tag in html_soup.find_all(tag_rule):
        tag_text = tag.get_text().strip()

        # Only keep substantial content (>10 words)
        if tag_text and len(tag_text.split()) > 10:
            main_content.append(tag_text)

    return "\n".join(main_content).strip()
```

**Two Rule Levels:**
- **Rule 0** (default): Extract `<h1>` through `<h6>` and `<p>` tags
  - Headers and paragraphs (main content)
- **Rule 1** (fallback): Also include `<div>` tags
  - Used when Rule 0 produces too little content

**Quality Filter:**
- Requires >10 words per section
- Filters out navigation, ads, footers
- Keeps only substantial content

#### 5. PDF Handling

```python
def extract_pdf_content(self, pdf_response):
    if not PDF_SUPPORT:
        logger.warning("PyPDF2 not available")
        return ""

    try:
        # Convert HTTP response to file-like object
        pdf_file = io.BytesIO(pdf_response.content)
        pdf_reader = PdfReader(pdf_file)

        text_content = []

        # Extract text from each page
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)

        full_text = "\n".join(text_content).strip()
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""
```

**PDF Processing:**
1. Loads PDF into memory (BytesIO)
2. Creates PyPDF2 reader object
3. Iterates through all pages
4. Extracts text from each page
5. Combines into single string

**Graceful Degradation:**
- If PyPDF2 not installed → skip PDFs
- If extraction fails → return empty string
- Logs errors for debugging

#### 6. Main Scraping Logic

```python
def scrape_url(self, url, rule=0):
    webpage_html = self.get_webpage_html(url)

    # Check if PDF (by extension or Content-Type header)
    is_pdf = (
        url.endswith(".pdf") or
        webpage_html.headers.get('Content-Type', '').startswith('application/pdf')
    )

    if is_pdf:
        if PDF_SUPPORT:
            return self.extract_pdf_content(webpage_html)
        else:
            return ""  # Skip PDFs if library unavailable

    # Handle HTML
    soup = self.convert_html_to_soup(webpage_html)
    main_content = self.extract_main_content(soup, rule)
    return main_content
```

**Decision Tree:**
```
URL
 │
 ├── Ends with .pdf? ──YES──> Extract PDF
 │        │
 │       NO
 │        │
 └── Content-Type: application/pdf? ──YES──> Extract PDF
          │
         NO
          │
          └──> Parse HTML
```

---

## Component 3: Multi-threaded Fetcher (fetch_web_content.py)

### Why Threading?

**Sequential Approach** (slow):
```
URL1 (3s) → URL2 (3s) → URL3 (3s) = 9 seconds total
```

**Threaded Approach** (fast):
```
URL1 (3s) ┐
URL2 (3s) ├─> All finish in ~3 seconds
URL3 (3s) ┘
```

### Code Walkthrough

#### 1. Initialization

```python
class WebContentFetcher:
    def __init__(self, query):
        self.query = query
        self.web_contents = []      # Stores fetched content
        self.error_urls = []        # Stores failed URLs
        self.web_contents_lock = threading.Lock()  # Thread-safe list access
        self.error_urls_lock = threading.Lock()
```

**Thread Safety:**
- Multiple threads can't modify lists simultaneously
- Locks prevent race conditions
- Each list has its own lock

#### 2. Thread Worker Function

```python
def _web_crawler_thread(self, thread_id: int, urls: list):
    try:
        url = urls[thread_id]  # Each thread gets one URL
        scraper = WebScraper()
        content = scraper.scrape_url(url, rule=0)

        # Retry with extended rules if content too short
        if 0 < len(content) < 800:
            content = scraper.scrape_url(url, rule=1)

        # Only save substantial content
        if len(content) > 300:
            with self.web_contents_lock:  # Lock before modifying
                self.web_contents.append({
                    "url": url,
                    "content": content
                })

    except Exception as e:
        with self.error_urls_lock:
            self.error_urls.append(url)
        logger.error(f"Thread {thread_id}: Error crawling {url}: {e}")
```

**Adaptive Scraping:**
- First try: Conservative rules (h1-h6, p tags)
- If content < 800 chars: Retry with div tags included
- Minimum threshold: 300 characters
- Anything shorter is discarded

#### 3. Launching Threads

```python
def _crawl_threads_launcher(self, url_list):
    threads = []

    # Create one thread per URL
    for i in range(len(url_list)):
        thread = threading.Thread(
            target=self._web_crawler_thread,
            args=(i, url_list)
        )
        threads.append(thread)
        thread.start()  # Start immediately

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
```

**Execution Flow:**
```
Main Thread
    │
    ├──> Thread 0 (URL 0) ──┐
    ├──> Thread 1 (URL 1) ──┤
    ├──> Thread 2 (URL 2) ──┤
    ...                      ├──> All threads working in parallel
    └──> Thread 9 (URL 9) ──┘
         │
         │  (join() waits for all to finish)
         │
    Main Thread continues
```

#### 4. Main Fetch Method

```python
def fetch(self):
    # Step 1: Get search results
    serper_client = SerperClient()
    serper_results = serper_client.serper(self.query)
    serper_response = serper_client.extract_components(serper_results)

    # Step 2: Extract URL list
    url_list = serper_response["links"]

    # Step 3: Crawl all URLs in parallel
    self._crawl_threads_launcher(url_list)

    # Step 4: Reorder results to match original URL order
    ordered_contents = [
        next((item['content'] for item in self.web_contents if item['url'] == url), '')
        for url in url_list
    ]

    return ordered_contents, serper_response
```

**Why Reorder?**
- Threads finish in unpredictable order
- Search results have ranking (1st result is most relevant)
- Reordering preserves this relevance ranking

---

## Complete Pipeline Example

Let's trace a query through the entire search pipeline:

### Input
```python
query = "What is machine learning?"
```

### Step 1: Serper Search

```python
serper_client = SerperClient()
results = serper_client.serper("What is machine learning?")
components = serper_client.extract_components(results)

# Output:
{
    'query': 'What is machine learning?',
    'language': 'en-us',
    'count': 10,
    'links': [
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://www.ibm.com/topics/machine-learning',
        'https://www.coursera.org/articles/what-is-machine-learning',
        ...
    ],
    'titles': [...],
    'snippets': [...]
}
```

### Step 2: Parallel Web Scraping

```python
fetcher = WebContentFetcher("What is machine learning?")
web_contents, serper_response = fetcher.fetch()

# Output (simplified):
web_contents = [
    "Machine learning (ML) is a field of study in artificial intelligence...",
    "Machine learning is a branch of AI that enables computers to learn...",
    "IBM defines machine learning as a subset of artificial intelligence...",
    ...  # 10 text extracts total
]
```

### Timeline Analysis

**Without Threading:**
```
URL 1: 2.3s
URL 2: 3.1s
URL 3: 1.8s
...
Total: ~25 seconds
```

**With Threading:**
```
All URLs: max(2.3s, 3.1s, 1.8s, ...) = ~3.1 seconds
```

**Speed Improvement: ~8x faster!**

---

## Best Practices Demonstrated

### 1. **Error Handling**
- Try-except blocks prevent crashes
- Timeouts prevent hanging
- Graceful degradation (skip on failure)

### 2. **Logging**
- Info level: Normal operations
- Warning level: Non-critical issues
- Error level: Failures with stack traces

### 3. **Quality Control**
- Minimum content length (300 chars)
- Word count filter (>10 words)
- Adaptive rules (retry if needed)

### 4. **Performance**
- Threading for I/O-bound operations
- Connection timeouts
- Efficient data structures

### 5. **Robustness**
- User agent rotation
- Content type detection
- Multiple scraping strategies

---

## Common Issues and Solutions

### Issue 1: Getting Blocked by Websites

**Symptoms:**
- 403 Forbidden errors
- Empty responses
- CAPTCHAs

**Solutions:**
```python
# Rotate user agents
scrapers = [
    WebScraper(user_agent='macOS'),
    WebScraper(user_agent='Windows')
]

# Add delays
import time
time.sleep(random.uniform(1, 3))  # Random delay between requests

# Respect robots.txt
from urllib.robotparser import RobotFileParser
rp = RobotFileParser()
rp.set_url("https://example.com/robots.txt")
rp.read()
if rp.can_fetch("*", url):
    # OK to scrape
```

### Issue 2: Extracting Too Little Content

**Symptoms:**
- Short text extracts
- Missing main content

**Solutions:**
```python
# Try multiple extraction rules
content = scraper.scrape_url(url, rule=0)
if len(content) < 500:
    content = scraper.scrape_url(url, rule=1)

# Target specific CSS classes
soup.find_all("div", class_="article-content")

# Remove unwanted elements
for tag in soup(['script', 'style', 'nav', 'footer']):
    tag.decompose()
```

### Issue 3: PDF Extraction Failures

**Symptoms:**
- Empty text from PDFs
- Unicode errors

**Solutions:**
```python
# Handle encoding issues
try:
    text = page.extract_text()
except UnicodeDecodeError:
    text = page.extract_text(encoding='latin-1')

# Try alternative libraries
from pdfminer.high_level import extract_text
text = extract_text(pdf_path)

# OCR for image-based PDFs
import pytesseract
from pdf2image import convert_from_path
images = convert_from_path(pdf_path)
text = pytesseract.image_to_string(images[0])
```

---

## Next Steps

Now that you understand how we get content from the web, proceed to:
- **02_RAG_SYSTEM.md** to learn how we process this content
- **03_EMBEDDINGS.md** to understand vector representations

You should be able to:
1. Explain how Serper API works
2. Describe web scraping with BeautifulSoup
3. Understand why we use threading
4. Implement basic error handling
