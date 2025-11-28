# SearchGPT - Complete System Overview

## Introduction

SearchGPT is a **Retrieval-Augmented Generation (RAG)** system that combines web search, content extraction, vector embeddings, and large language models to provide accurate, well-sourced answers to user queries.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances AI responses by:
1. **Retrieving** relevant information from external sources
2. **Augmenting** the AI's context with this retrieved information
3. **Generating** answers based on both the AI's knowledge AND the retrieved facts

This prevents the AI from "hallucinating" (making up false information) by grounding its responses in real, verified sources.

## High-Level Architecture

```
┌─────────────┐
│    User     │
│   (Query)   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────────────────────────┐
│              STREAMLIT WEB INTERFACE                     │
│  - Receives user query                                   │
│  - Displays progress indicators                          │
│  - Streams AI-generated answers                          │
└──────┬──────────────────────────────────────────────────┘
       │
       v
┌──────────────────────────────────────────────────────────┐
│                  RAG PIPELINE                             │
│                                                           │
│  Step 1: SEARCH                                           │
│  ┌─────────────────────┐                                 │
│  │  Serper API         │ ──> Google search results       │
│  │  (Google Search)    │                                 │
│  └─────────────────────┘                                 │
│           │                                               │
│           v                                               │
│  Step 2: WEB SCRAPING                                     │
│  ┌─────────────────────┐                                 │
│  │  Web Crawler        │ ──> Extract text from URLs      │
│  │  (BeautifulSoup)    │     (including PDFs)            │
│  └─────────────────────┘                                 │
│           │                                               │
│           v                                               │
│  Step 3: EMBEDDING & RETRIEVAL                            │
│  ┌─────────────────────┐                                 │
│  │  Vector Database    │ ──> Find most relevant chunks   │
│  │  (ChromaDB)         │                                 │
│  └─────────────────────┘                                 │
│           │                                               │
│           v                                               │
│  Step 4: ANSWER GENERATION                                │
│  ┌─────────────────────┐                                 │
│  │  LLM Service        │ ──> Stream AI answer            │
│  │  (OpenRouter API)   │                                 │
│  └─────────────────────┘                                 │
└──────┬────────────────────────────────────────────────────┘
       │
       v
┌──────────────────┐
│  Final Answer    │
│  + References    │
└──────────────────┘
```

## Core Components

### 1. **app.py** - Web Interface (Frontend)
- Built with Streamlit framework
- Handles user input and API key management
- Orchestrates the entire pipeline
- Displays real-time progress and results

### 2. **serper_service.py** - Search Service
- Interfaces with Google Search via Serper API
- Returns top search results (titles, URLs, snippets)
- Detects language (English/Chinese)

### 3. **web_crawler.py** - Content Extraction
- Scrapes HTML content from URLs
- Extracts main text from web pages
- Handles PDF files using PyPDF2

### 4. **fetch_web_content.py** - Multi-threaded Fetcher
- Manages parallel web scraping
- Uses threading to fetch multiple URLs simultaneously
- Filters out low-quality content

### 5. **retrieval.py** - Vector Search
- Creates embeddings (vector representations) of text
- Stores vectors in ChromaDB
- Retrieves most relevant chunks for the query

### 6. **llm_answer.py** - Answer Generation
- Formats retrieved content as context
- Calls LLM (via OpenRouter API) with context
- Handles response streaming

### 7. **llm_service.py** - LLM API Client
- Manages OpenRouter API connections
- Provides embeddings and chat completions
- Handles different AI models (Grok, GPT-OSS, etc.)

## Data Flow Example

**User Query**: "What is quantum computing?"

```
1. SEARCH (serper_service.py)
   Input:  "What is quantum computing?"
   Output: [
     {title: "Quantum Computing - Wikipedia", url: "...", snippet: "..."},
     {title: "IBM Quantum", url: "...", snippet: "..."},
     ...10 total results
   ]

2. WEB SCRAPING (web_crawler.py + fetch_web_content.py)
   Input:  10 URLs
   Process: Parallel scraping using 10 threads
   Output: [
     "Quantum computing is a type of computation...",
     "IBM's quantum computer uses superconducting qubits...",
     ...text from each URL
   ]

3. EMBEDDING & RETRIEVAL (retrieval.py)
   Input:  Scraped text + original query
   Process:
     a) Split text into 500-character chunks
     b) Convert each chunk to 384-dimensional vector
     c) Store in ChromaDB vector database
     d) Search for chunks most similar to query
   Output: Top 10 most relevant text chunks

4. ANSWER GENERATION (llm_answer.py + llm_service.py)
   Input:  Relevant chunks + query
   Process:
     Prompt = "Web search result: {chunks}\n
               Instructions: You are a researcher.
               Using the provided web search results,
               write a comprehensive reply to: {query}"
     Call OpenRouter API with Grok model
   Output: "Quantum computing is a revolutionary approach
           to computation that harnesses quantum mechanics...
           [1] According to Wikipedia... [2] IBM explains..."
```

## Key Technologies

### Backend
- **Python 3.11+**: Programming language
- **Requests**: HTTP library for API calls
- **BeautifulSoup4**: HTML parsing
- **PyPDF2**: PDF text extraction
- **ChromaDB**: Vector database for embeddings
- **NumPy**: Numerical computations

### Frontend
- **Streamlit**: Web application framework
- **streamlit-lottie**: Animated loading indicators

### APIs
- **Serper API**: Google search results
- **OpenRouter API**: Access to multiple LLM models
  - Grok (x-ai/grok-4.1-fast:free)
  - GPT-OSS (openai/gpt-oss-20b:free)
- **Gitee AI** (optional): BGE-M3 Chinese embeddings

### Embeddings
- **OpenRouter embeddings**: text-embedding-3-small (default)
- **BGE-M3 via Gitee**: Bilingual embeddings (optional)

## Why This Architecture?

### 1. **Separation of Concerns**
Each component has a single, well-defined responsibility:
- Search finds URLs
- Scraper extracts text
- Embedder creates vectors
- LLM generates answers

### 2. **Modularity**
Components can be swapped easily:
- Change search provider (Serper → Bing, etc.)
- Change embedding model (OpenAI → BGE-M3)
- Change LLM (Grok → GPT-4, Claude, etc.)

### 3. **Performance**
- **Threading**: Multiple URLs scraped in parallel
- **Streaming**: AI answers appear word-by-word
- **Caching**: Vector database stores embeddings

### 4. **Scalability**
- Can handle 10+ URLs per search
- Vector database scales to millions of documents
- Stateless design allows horizontal scaling

## Learning Path

To understand this project deeply, study the documentation in this order:

1. **00_OVERVIEW.md** (this file) - Big picture
2. **01_SEARCH_PIPELINE.md** - How search and scraping work
3. **02_RAG_SYSTEM.md** - Deep dive into RAG architecture
4. **03_EMBEDDINGS.md** - Understanding vector representations
5. **04_LLM_INTEGRATION.md** - How LLMs generate answers
6. **05_STREAMLIT_UI.md** - Building the web interface
7. **06_ADVANCED_TOPICS.md** - Threading, streaming, error handling

## Quick Start for Learners

### 1. Read the Code in This Order:
```
app.py (main orchestration)
  ↓
serper_service.py (understand search)
  ↓
web_crawler.py (understand scraping)
  ↓
retrieval.py (understand embeddings)
  ↓
llm_answer.py (understand answer generation)
```

### 2. Run Each Component Individually:
Each file has a `if __name__ == "__main__":` section at the bottom that demonstrates how to use it standalone.

### 3. Experiment:
- Change the search query
- Modify the prompt template
- Try different embedding models
- Adjust chunk sizes

### 4. Build Your Own:
Try building simplified versions:
- Simple search script (just Serper API)
- Basic scraper (just BeautifulSoup)
- Minimal RAG (embeddings + ChromaDB)
- Simple LLM client (just OpenRouter)

## Common Questions

### Q: Why use RAG instead of just asking the LLM directly?
**A**: LLMs have:
- Knowledge cutoff dates (can't know recent events)
- Tendency to hallucinate (make up facts)
- Limited context window (can't remember everything)

RAG solves these by grounding answers in real, current sources.

### Q: What are embeddings?
**A**: Embeddings convert text into numbers (vectors) that capture semantic meaning. Similar text has similar vectors, enabling similarity search.

### Q: Why ChromaDB instead of traditional databases?
**A**: Traditional databases search exact matches. Vector databases find *semantically similar* content, even if words are different.

Example:
- Query: "How do computers think?"
- Traditional DB: No results (exact phrase not found)
- Vector DB: Returns docs about "artificial intelligence", "machine learning", "neural networks" (semantically related)

### Q: Why multi-threading for web scraping?
**A**: Scraping 10 URLs sequentially takes 10× longer than scraping them in parallel. Threading lets us fetch all URLs simultaneously.

### Q: What's the role of the config.yaml file?
**A**: Stores configuration like:
- API keys (temporarily, during runtime only)
- Model selection
- Prompt templates
- Embedding model choice

## Next Steps

Now that you understand the overview, proceed to:
- **01_SEARCH_PIPELINE.md** to learn how we find and extract web content
- **02_RAG_SYSTEM.md** to understand the core RAG architecture in depth

Happy learning! 🚀
