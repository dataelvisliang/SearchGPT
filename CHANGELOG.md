# Changelog

All notable changes to Relevance Search will be documented in this file.

## [2.3.0] - 2025-11-30

### New Features

#### 🤖 Updated Free Model Selection
- **Refreshed model lineup with latest free models**
  - Removed: `x-ai/grok-4.1-fast:free`, `openai/gpt-oss-20b:free`, `z-ai/glm-4.5-air:free`
  - Added: `qwen/qwen3-4b:free` - Alibaba's compact Qwen3 4B model
  - Added: `amazon/nova-2-lite-v1:free` - Amazon's Nova 2 Lite model (new default)
  - Kept: `nvidia/nemotron-nano-9b-v2:free`
  - Kept: `alibaba/tongyi-deepresearch-30b-a3b:free`

### UI Improvements
- **Renamed "Source References" to "Search Results"**
  - Clarifies that this section shows all web sources found
  - Distinguishes from AI-generated citations in the answer
  - Better user understanding of what sources were available

### Technical Details
- Updated model dropdown to 4 curated free models
- Serper API configured with explicit num=10 parameter (free tier limit)
- Retrieval uses TOP_K=10 for LLM context
- Fixed UI bug where sections displayed on model switch
- Fixed UI bug where sections didn't display after search completion

### Benefits
- More focused model selection with proven performers
- All models remain completely free
- Clearer distinction between searched sources and AI citations

## [2.2.2] - 2025-11-30

### Documentation
- Removed premium model references
- Updated README.md with new model listings
- Updated README_STREAMLIT.md with new free models
- All models available in the Streamlit UI dropdown

## [2.2.1] - 2025-11-29

### Performance Improvements

#### ⚡ Batch Embedding Processing
- **Dramatically reduced API calls and embedding time**
  - Implemented batch processing for document embeddings (20 texts per API call)
  - Reduced 93 individual API calls → ~5 batch calls (18x fewer requests!)
  - Significantly faster embedding generation
  - Lower API costs and reduced rate limiting issues
  - Better logging shows batch progress

### Bug Fixes

#### 🔄 Improved API Timeout Handling
- **Enhanced retry logic for embedding API calls**
  - Added automatic retry mechanism (up to 3 attempts) for OpenRouter embedding timeouts
  - Implemented exponential backoff strategy (1s, 2s, 4s delays between retries)
  - Increased timeout progressively on each retry (60s, 90s, 120s)
  - Better error logging and debugging information
  - Graceful handling of ReadTimeout exceptions

### Technical Details
- Added `OpenRouterEmbeddings._get_batch_embedding()` method for batch processing
- Updated `OpenRouterEmbeddings.embed_documents()` to use batching
- Configurable batch size (default: 20 texts per request)
- Updated `_get_embedding()` method with retry logic
- Separate handling for timeout vs other request exceptions
- Detailed logging for each retry attempt and batch progress
- Prevents cascading failures during high API load

## [2.2.0] - 2025-11-28

### New Features

#### ⏱️ Enhanced Pipeline Timing & Metrics
- **Step-by-step timing information**
  - Shows time taken for each pipeline step (search, scrape, embed, generate)
  - Embedded API call tracking with detailed metrics
  - Individual API call timing for embed_documents and embed_query
  - Total embedding time calculation

#### 📡 Embedding API Call Details
- **Complete visibility into embedding operations**
  - Tracks number of API calls to OpenRouter/Gitee
  - Shows chunks processed per API call
  - Displays time spent on each embedding API call
  - Differentiates between document embedding and query embedding
  - Provider-specific metrics (OpenRouter text-embedding-3-small or Gitee BGE-M3)

#### 🧹 Text Cleaning & Formatting
- **Cleaned formatted context for better readability**
  - Removes multiple consecutive empty lines (max 1 empty line between sections)
  - Consolidates multiple spaces to single spaces
  - Trims leading/trailing whitespace on each line
  - Results in cleaner prompts sent to LLM

#### 🔍 Comprehensive Pipeline Tracing
- **Complete visibility into RAG pipeline execution**
  - Added expandable "Pipeline Trace" panel showing all steps
  - Captures and displays data from every stage of processing
  - Helps with debugging, learning, and optimization
  - Stored in session state for persistence

#### 📊 Trace Data Captured

**Step 1: Web Search**
- Query submitted by user
- Number of results found from Serper API
- All URLs, titles, and snippets returned
- Complete search results metadata

**Step 2: Content Scraping**
- Total pages attempted vs successful scrapes
- Failed scrape detection with status indicators
- Content lengths for each URL
- Preview of scraped content (first 200 characters)
- Clear success (✅) / failure (❌) indicators

**Step 3: Embedding & Retrieval**
- Number of relevant chunks retrieved
- **Similarity scores** for each chunk (color-coded)
  - Green (>0.7): High relevance
  - Orange (>0.5): Medium relevance
  - Red (<0.5): Low relevance
- Source URL tracking for each chunk
- Full content preview of retrieved chunks
- Chunk length in characters

**Step 4: Answer Generation** (Most Detailed)
- AI profile/persona used (researcher, expert, etc.)
- **Chunks sent to LLM** with similarity scores
- **Formatted context** showing [1], [2] citation format
- **Complete prompt** sent to LLM including:
  - Template instructions
  - All context chunks with citations
  - Query text
  - Language specification
  - Profile/persona assignment
- Model name and temperature setting
- Generated answer length
- Time taken to generate answer

#### 🎨 UI Enhancements

**Trace Display Features**
- Collapsible expander: "📊 View Detailed Pipeline Execution Trace"
- Organized by pipeline steps with clear section headers
- Scrollable text areas for long content
- Color-coded similarity scores for quick assessment
- Summary statistics panel with metrics
- Separated "Chunks Retrieved" vs "Chunks Sent to LLM"

**Similarity Score Visualization**
- Calculated as: `similarity = 1 - distance` (ChromaDB L2 distance)
- Displayed with 4 decimal precision (e.g., 0.8542)
- Color coding for instant quality assessment
- Shows in both retrieval and generation sections

#### 🐛 Bug Fixes

**text_utils.py**
- Fixed `ValueError: empty separator` error
- Added check to skip empty string ("") in separators list
- Prevents crash when recursive text splitter reaches empty separator
- Falls back to character-count splitting if no separators work

**app.py**
- Fixed trace_data dictionary overwriting issue
- Changed from dict replacement to `.update()` method
- Proper initialization of generation trace data
- Ensures all trace data persists correctly

**retrieval.py**
- Added similarity score calculation to ChromaDB results
- Stores scores in document metadata
- Converts ChromaDB distance to similarity score
- Rounded to 4 decimal places for readability

### Technical Improvements

#### 🔧 Trace Data Structure
```python
trace_data = {
    'query': str,
    'search_results': {
        'count': int,
        'urls': list,
        'titles': list,
        'snippets': list
    },
    'scraped_content': {
        'total_pages': int,
        'successful_scrapes': int,
        'failed_scrapes': int,
        'content_lengths': list,
        'content_previews': list[dict]
    },
    'retrieval': {
        'num_relevant_docs': int,
        'relevant_docs': list[dict with similarity_score]
    },
    'generation': {
        'model': str,
        'temperature': float,
        'profile': str,
        'chunks_sent': list[dict],
        'formatted_context': str,
        'prompt': str,
        'answer_length': int,
        'answer_preview': str,
        'time_taken': float
    }
}
```

#### 🎯 Session State Management
- `st.session_state.trace_data` stores complete trace
- Persists across Streamlit reruns
- Cleared when "Clear Results" button clicked
- Enables trace viewing after search completes

### Use Cases

#### Debugging
- **Verify retrieval quality**: Check similarity scores of chunks
- **Inspect exact prompts**: See what was sent to LLM
- **Diagnose failures**: Identify which URLs failed to scrape
- **Check context assembly**: Verify chunks were formatted correctly

#### Learning & Education
- **Understand RAG pipelines**: See complete data flow
- **Study prompt engineering**: Learn from actual prompts
- **Visualize embeddings**: See similarity scores in action
- **Learn citation formatting**: See [1], [2] mapping to sources

#### Optimization
- **Identify low-quality chunks**: Find low similarity scores
- **Improve prompts**: Iterate on prompt templates
- **Tune retrieval**: Adjust chunk sizes, overlap, top-k
- **Measure performance**: Track time and token usage per step

### Files Modified

- `app.py` - Added complete tracing system with UI display
- `src/retrieval.py` - Added similarity score calculation and metadata
- `src/text_utils.py` - Fixed empty separator bug

---

## [2.1.0] - 2025-11-27

### New Features

#### 📄 PDF Reading Capability
- **Added PDF content extraction**
  - Integrated PyPDF2 library for PDF parsing
  - `web_crawler.py` now detects and extracts text from PDF files
  - Automatically detects PDFs by URL extension (.pdf) or Content-Type header
  - Extracts text from all pages and returns as concatenated string
  - Graceful fallback if PyPDF2 not available
  - Added comprehensive logging for PDF extraction process

#### 🇨🇳 Gitee AI Integration
- **Added Gitee AI API key input**
  - New input field in sidebar for Gitee AI API key
  - Supports BGE-M3 embeddings via ai.gitee.com
  - Status indicator showing which embedding service is active
  - Falls back to OpenRouter embeddings if Gitee key not provided
  - API keys stored in temporary session file only (not saved to config.yaml)

#### 🤖 Streamlined Model Selection
- **Reduced model options to core free models**
  - Kept only `x-ai/grok-4.1-fast:free` (primary)
  - Kept only `openai/gpt-oss-20b:free` (alternative)
  - Removed other model options for simplified user experience

### UI/UX Improvements

#### 📊 Detailed Progress Indicators
- **4-step progress tracking with real-time updates**
  - Step 1: "🌐 Searching the web..." (shows result count)
  - Step 2: "📄 Fetching content from X pages..." (shows successful fetch count)
  - Step 3: "🔍 Creating embeddings and searching..." (shows relevant section count)
  - Step 4: "🤖 Generating AI answer (streaming)..."
  - Each step displays success message with statistics before proceeding
  - All status messages aligned with search bar width using centered columns

#### ⚡ Real-time Streaming Output
- **Implemented word-by-word streaming from OpenRouter**
  - Direct integration with OpenRouter's Server-Sent Events (SSE) API
  - Removed batch response display in favor of incremental updates
  - Answer appears progressively as LLM generates it
  - Better user experience with immediate feedback
  - Uses `response.iter_lines()` for real-time chunk processing

#### 🎨 Markdown Rendering
- **Enhanced answer display with full markdown support**
  - Changed from HTML rendering to `st.markdown()`
  - Properly renders bold, italic, lists, code blocks, and links
  - Better formatting for structured AI responses
  - Preserves markdown in citations and references

#### 📐 Consistent Layout Width
- **All interface elements now match search bar width**
  - Status messages in centered columns `[1, 6, 1]`
  - Error messages and expandable error details
  - References section (Quick Links and Detailed Sources tabs)
  - Download buttons (TXT and JSON export)
  - Loading animation positioned with status messages
  - Creates cohesive, professional appearance

#### 🎭 Removed Subtitle
- **Cleaner header design**
  - Removed "AI-Powered Search Engine with OpenRouter" subtitle
  - Header now shows only main title
  - More minimalist interface

### Technical Improvements

#### 🔧 Dependencies
- **Updated requirements.txt**
  - Added `PyPDF2>=3.0.0` for PDF support
  - Updated `openai>=1.0.0` for latest SDK compatibility

#### 🔐 Security
- **API key handling improvements**
  - Removed all hardcoded API keys from config.yaml
  - All API keys now stored as empty strings in default config
  - Users must provide keys via UI (not committed to git)

### Bug Fixes
- **Fixed indentation errors in references section**
  - Corrected indentation within `with tab2:` block
  - References now display properly in centered layout

### Files Modified

- `app.py` - Added Gitee input, streaming output, 4-step progress, markdown rendering, centered layout
- `src/web_crawler.py` - Added PDF detection and extraction methods
- `src/config/config.yaml` - Added gitee_api_key field, cleared hardcoded keys
- `requirements.txt` - Added PyPDF2 dependency

---

## [2.0.0] - 2025-11-26

### Major Changes

#### 🔄 API Migration
- **Migrated from OpenAI API to OpenRouter API**
  - Now supports multiple AI models including free options
  - Added `llm_service.py` with `OpenRouterService` class
  - Added `OpenRouterEmbeddings` class for vector embeddings
  - Default model: `x-ai/grok-4.1-fast:free` (GPT-4 equivalent)
  - Alternative free model: `openai/gpt-oss-20b:free` (GPT-3.5 equivalent)

#### 🎨 New Streamlit Web Interface
- **Added beautiful Streamlit UI** (`app.py`)
  - Interactive web interface with Lottie animations
  - API key input directly in the UI
  - Model selection dropdown
  - Customizable AI profiles (Researcher, Technical Expert, etc.)
  - Output format customization
  - Real-time progress indicators
  - Export results as TXT or JSON
  - Error handling with expandable debug info

#### 🗑️ Dependencies Cleanup
- **Removed LangChain dependency**
  - Reduced package size and complexity
  - Faster installation and startup
  - Created `text_utils.py` with lightweight replacements:
    - `RecursiveTextSplitter` (replaces LangChain's text splitter)
    - `PromptTemplate` (replaces LangChain's prompt template)
    - `Document` class (replaces LangChain's Document)
  - Updated `requirements.txt` (removed `langchain` and `tiktoken`)

#### 📦 Database Updates
- **Modern ChromaDB integration**
  - Updated to use `chromadb.PersistentClient` (new API)
  - Removed deprecated `chromadb.Client(Settings())` usage
  - Fallback support for both LangChain wrapper and direct ChromaDB usage

### Enhancements

#### 🔍 Improved Search
- **Fixed Serper API pagination issue**
  - Removed hardcoded `page: 2` parameter
  - Now correctly retrieves page 1 results first
  - Better handling of search results

#### 🎯 Better Embeddings
- **Upgraded embedding model**
  - Changed from `text-embedding-ada-002` to `text-embedding-3-small`
  - More efficient and cost-effective
  - Better performance

#### 🐛 Bug Fixes
- **Fixed IndexError in document retrieval**
  - Added dynamic document count handling
  - Changed from fixed `TOP_K=10` to `min(len(docs), TOP_K)`
  - Handles cases with fewer than 10 documents gracefully

#### 📝 Comprehensive Logging
- **Added detailed logging throughout**
  - `serper_service.py`: API requests, responses, result counts
  - `fetch_web_content.py`: Thread-level scraping progress
  - `retrieval.py`: Embedding generation, ChromaDB operations
  - `llm_service.py`: API calls, response status
  - `app.py`: User queries, error diagnostics
  - All logs include timestamps and log levels

#### 🎨 UI/UX Improvements
- **Enhanced error messages**
  - Specific error types displayed
  - Expandable debug information
  - Troubleshooting suggestions
  - Full error tracebacks for debugging

#### 📊 Better User Feedback
- **Real-time status updates**
  - Search progress indicators
  - Loading animations
  - Success animations
  - Statistics display (search time, reference count)

### Configuration Changes

#### ⚙️ Updated Config File
- **Changed API key names** in `config.yaml`:
  - `openai_api_key` → `openrouter_api_key`
  - Added comment for available free models
  - Updated default model to `x-ai/grok-4.1-fast:free`

### Files Added

- `app.py` - Streamlit web interface
- `src/llm_service.py` - OpenRouter API integration
- `src/text_utils.py` - LangChain replacement utilities
- `README_STREAMLIT.md` - Streamlit-specific documentation
- `CHANGELOG.md` - This file

### Files Modified

- `src/llm_answer.py` - Updated to use OpenRouter, added dynamic document handling
- `src/retrieval.py` - ChromaDB modernization, embedding model upgrade
- `src/serper_service.py` - Added logging, fixed pagination
- `src/fetch_web_content.py` - Added comprehensive logging
- `requirements.txt` - Removed LangChain, added Streamlit
- `README.md` - Complete rewrite with new features
- `src/config/config.yaml` - API key name changes

### Technical Details

#### Dependency Changes
**Removed:**
- `langchain==0.0.340`
- `openai==1.3.4`
- `tiktoken==0.7.0`

**Added:**
- `streamlit>=1.28.0`
- `streamlit-lottie>=0.0.5`

**Kept:**
- `beautifulsoup4==4.12.2`
- `chromadb==0.4.18`
- `lxml==4.9.3`
- `PyYAML==6.0.1`
- `Requests==2.31.0`

### Credits

This version builds upon the excellent foundation of [Wilson-ZheLin/SearchGPT](https://github.com/Wilson-ZheLin/SearchGPT).

---

## [1.0.0] - Original Release

Original SearchGPT implementation by Wilson-ZheLin using OpenAI API and LangChain.
