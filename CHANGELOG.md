# Changelog

All notable changes to this project will be documented in this file.

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
  - API key saved to config.yaml when provided

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
