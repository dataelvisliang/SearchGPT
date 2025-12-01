# Building Interactive UI with Streamlit

## What is Streamlit?

Streamlit is a Python framework for creating web applications with minimal code. Perfect for data science and AI projects.

**Traditional Web App:**
```python
# Requires: HTML, CSS, JavaScript, Flask/Django, frontend-backend integration
# 500+ lines of code for simple UI
```

**Streamlit:**
```python
import streamlit as st

st.title("My App")
name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")

# 4 lines of code!
```

## Relevance Search UI Architecture

### Component Structure

```
app.py
├── Configuration (page config, CSS)
├── Header (title, description)
├── Sidebar
│   ├── API Key Inputs
│   ├── Model Selection
│   ├── Profile Selection
│   └── Output Format
├── Main Content
│   ├── Search Input
│   ├── Search Button
│   ├── Progress Indicators
│   ├── Answer Display (streaming)
│   └── References Section
└── Download Options
```

## Page Configuration

### Initial Setup

```python
import streamlit as st

st.set_page_config(
    page_title="Relevance Search",
    page_icon="🔍",
    layout="wide",                 # Use full width
    initial_sidebar_state="expanded"  # Show sidebar by default
)
```

**Options Explained:**
- `page_title`: Browser tab title
- `page_icon`: Emoji or image URL
- `layout`: "centered" (default) or "wide"
- `initial_sidebar_state`: "expanded" or "collapsed"

### Custom CSS

```python
st.markdown("""
    <style>
    /* Main content area */
    .main {
        padding: 2rem;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }

    /* Spinning loader */
    .spinner {
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #4CAF50;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Breathing dots animation */
    .breathing-dots::after {
        content: '...';
        animation: breathe 1.5s ease-in-out infinite;
    }

    @keyframes breathe {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)
```

## Sidebar Components

### API Key Inputs

```python
with st.sidebar:
    st.markdown("### 🔑 API Keys")

    # OpenRouter API Key
    openrouter_api_key = st.text_input(
        "OpenRouter API Key",
        type="password",              # Hides input
        placeholder="sk-or-v1-...",
        help="Get your key at https://openrouter.ai/keys"
    )

    # Serper API Key
    serper_api_key = st.text_input(
        "Serper API Key",
        type="password",
        placeholder="Your Serper API key",
        help="Get your key at https://serper.dev"
    )

    # Gitee API Key (Optional)
    gitee_api_key = st.text_input(
        "Gitee AI API Key (Optional)",
        type="password",
        placeholder="Your Gitee AI API key",
        help="For BGE-M3 embeddings. Get at https://ai.gitee.com/"
    )

    # Status indicator
    if gitee_api_key:
        st.success("✅ Using Gitee AI BGE-M3 embeddings")
    else:
        st.info("💡 Using OpenRouter text-embedding-3-small")
```

**Input Types:**
- `text_input`: Single line text
- `text_area`: Multi-line text
- `number_input`: Numbers only
- `password`: Hidden characters
- `date_input`: Date picker
- `time_input`: Time picker

### Selection Widgets

```python
# Dropdown (single selection)
model_name = st.selectbox(
    "LLM Model",
    [
        "x-ai/grok-4.1-fast:free",
        "openai/gpt-oss-20b:free"
    ],
    help="Free models via OpenRouter"
)

# Radio buttons
profile = st.radio(
    "AI Profile",
    [
        "Conscientious Researcher",
        "Technical Expert",
        "Simple Explainer"
    ]
)

# Multi-select
topics = st.multiselect(
    "Filter Topics",
    ["Science", "Technology", "Business", "Health"]
)

# Slider
max_results = st.slider(
    "Max Results",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)
```

### Conditional Display

```python
# Show advanced options only if checkbox is checked
show_advanced = st.checkbox("Show Advanced Options")

if show_advanced:
    st.slider("Temperature", 0.0, 1.0, 0.7)
    st.number_input("Max Tokens", 100, 4000, 2048)
```

## Main Content Area

### Centered Columns Layout

```python
# Create 3 columns: [empty, content, empty]
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # All content goes in middle column
    st.title("🔍 Relevance Search")
    query = st.text_input("Ask me anything...")
```

**Column Ratios:**
- `[1, 6, 1]`: Content takes 75% width, centered
- `[1, 2, 1]`: Content takes 50% width
- `[2, 3, 2]`: Content takes 42% width

### Search Input

```python
# Search input
query = st.text_input(
    "",
    placeholder="Ask me anything and press Enter...",
    key="search_query",
    label_visibility="collapsed"  # Hide label
)

# Alternative: Text area for longer queries
query = st.text_area(
    "Your Question",
    placeholder="Enter your question here...",
    height=100
)
```

### Buttons

```python
# Basic button
if st.button("🔍 Search"):
    # Execute search
    perform_search(query)

# Button with custom key (for multiple buttons)
col1, col2 = st.columns(2)
with col1:
    if st.button("Search", key="search_btn"):
        search()

with col2:
    if st.button("Clear", key="clear_btn"):
        clear()

# Download button
st.download_button(
    label="📥 Download Answer (TXT)",
    data=answer_text,
    file_name="answer.txt",
    mime="text/plain"
)
```

## State Management

### Session State

Streamlit reruns the entire script on every interaction. Use `st.session_state` to persist data.

```python
# Initialize state
if 'answer' not in st.session_state:
    st.session_state.answer = None

if 'references' not in st.session_state:
    st.session_state.references = None

# Update state
if search_button:
    st.session_state.answer = get_answer(query)
    st.session_state.references = get_references()

# Read state
if st.session_state.answer:
    st.markdown(st.session_state.answer)
```

### Example: Counter

```python
# Initialize
if 'count' not in st.session_state:
    st.session_state.count = 0

# Increment button
if st.button("Increment"):
    st.session_state.count += 1

# Display
st.write(f"Count: {st.session_state.count}")
```

**How it works:**
1. User clicks "Increment"
2. Script reruns from top
3. `count` persists in session_state
4. Incremented value is displayed

## Progress Indicators

### 4-Step Progress Display

```python
# Create centered columns
col1_status, col2_status, col3_status = st.columns([1, 6, 1])

with col2_status:
    # Placeholder for dynamic updates
    status_placeholder = st.empty()

    # Step 1: Search
    status_placeholder.info("🌐 **Step 1/4:** Searching the web...")
    results = search_web(query)
    status_placeholder.success(f"✅ Found {len(results)} results")
    time.sleep(0.5)

    # Step 2: Scraping
    status_placeholder.info(f"📄 **Step 2/4:** Fetching content...")
    contents = scrape_urls(results)
    status_placeholder.success(f"✅ Fetched {len(contents)} pages")
    time.sleep(0.5)

    # Step 3: Embeddings
    status_placeholder.info("🔍 **Step 3/4:** Creating embeddings...")
    chunks = create_embeddings(contents)
    status_placeholder.success(f"✅ Found {len(chunks)} relevant chunks")
    time.sleep(0.5)

    # Step 4: Generation
    status_placeholder.info("🤖 **Step 4/4:** Generating answer...")
    # (Streaming happens here)
```

### Loading Animations

#### Lottie Animations

```python
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    """Load Lottie animation from URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
LOTTIE_LOADING = "https://assets5.lottiefiles.com/packages/lf20_loading.json"
lottie_loading = load_lottieurl(LOTTIE_LOADING)

# Display animation
loading_placeholder = st.empty()
with loading_placeholder:
    st_lottie(lottie_loading, height=200, key="loading")

# Clear when done
loading_placeholder.empty()
```

#### Built-in Spinner

```python
with st.spinner("Searching the web..."):
    results = search_web(query)
    # Spinner shows until this block completes
```

#### Progress Bar

```python
progress_bar = st.progress(0)

for i, url in enumerate(urls):
    scrape_url(url)
    progress_bar.progress((i + 1) / len(urls))

progress_bar.empty()  # Remove when done
```

#### Custom Spinner

```python
# Show custom spinning icon
st.markdown("""
    <div class="generating-container">
        <div class="spinner"></div>
        <div class="generating-text">Generating your answers<span class="breathing-dots"></span></div>
    </div>
""", unsafe_allow_html=True)
```

## Streaming Answer Display

### Real-Time Updates

```python
# Create placeholder
answer_placeholder = st.empty()

# Show generating animation
answer_placeholder.markdown("""
    <div class="generating-container">
        <div class="spinner"></div>
        <div>Generating your answers<span class="breathing-dots"></span></div>
    </div>
""", unsafe_allow_html=True)

# Stream LLM response
full_answer = ""
for chunk in stream_llm_response(query):
    full_answer += chunk
    # Update placeholder with accumulated text
    answer_placeholder.markdown(full_answer)

# Answer is now complete
```

### Why This Works

```python
# st.empty() creates a container that can be updated
placeholder = st.empty()

# First update
placeholder.text("Loading...")

# Second update (replaces previous)
placeholder.text("Almost done...")

# Final update
placeholder.success("Complete!")
```

## Display Components

### Markdown

```python
# Basic markdown
st.markdown("## This is a heading")
st.markdown("**Bold** and *italic* text")

# With HTML/CSS (unsafe_allow_html=True)
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
        <h3>Custom Styled Box</h3>
        <p>This uses HTML and CSS</p>
    </div>
""", unsafe_allow_html=True)

# Code block
st.markdown("""
```python
def hello():
    print("Hello, world!")
```
""")
```

### Tabs

```python
tab1, tab2, tab3 = st.tabs(["📋 Quick Links", "🔗 Detailed Sources", "📊 Stats"])

with tab1:
    st.write("Quick reference links")
    for link in links:
        st.markdown(f"[{link['title']}]({link['url']})")

with tab2:
    st.write("Detailed source information")
    for i, source in enumerate(sources):
        st.markdown(f"**[{i+1}] {source['title']}**")
        st.markdown(f"{source['snippet']}")

with tab3:
    st.metric("Search Time", f"{search_time:.2f}s")
    st.metric("Results Found", len(results))
```

### Expanders

```python
# Collapsible section
with st.expander("🔍 Error Details"):
    st.write("**Error Type:**", type(error).__name__)
    st.write("**Error Message:**", str(error))
    st.code(traceback.format_exc(), language="python")
```

### Metrics

```python
col1, col2, col3 = st.columns(3)

col1.metric(
    label="Search Time",
    value=f"{time:.2f}s",
    delta="-0.5s"  # Shows as improvement (green)
)

col2.metric(
    label="Results",
    value=len(results),
    delta=f"+{new_results}"
)

col3.metric(
    label="Quality Score",
    value="95%",
    delta="+5%"
)
```

### Messages

```python
# Info box (blue)
st.info("💡 Using OpenRouter embeddings")

# Success box (green)
st.success("✅ Search completed successfully!")

# Warning box (yellow)
st.warning("⚠️ API key not found in config")

# Error box (red)
st.error("❌ An error occurred")
```

## References Section

### Tabbed Display

```python
if st.session_state.references:
    col1_ref, col2_ref, col3_ref = st.columns([1, 6, 1])

    with col2_ref:
        st.markdown("---")
        st.markdown("## 📚 Source References")

        # Create tabs
        tab1, tab2 = st.tabs(["📋 Quick Links", "🔗 Detailed Sources"])

        with tab1:
            # Quick links as bullets
            links = st.session_state.references['links']
            titles = st.session_state.references['titles']

            for i, (link, title) in enumerate(zip(links, titles), 1):
                st.markdown(f"{i}. [{title}]({link})")

        with tab2:
            # Detailed source cards
            snippets = st.session_state.references['snippets']

            for i, (link, title, snippet) in enumerate(
                zip(links, titles, snippets), 1
            ):
                st.markdown(f"""
                    <div class="reference-card">
                        <strong>[{i}] {title}</strong><br>
                        <a href="{link}" target="_blank">{link}</a><br>
                        <em>{snippet}</em>
                    </div>
                """, unsafe_allow_html=True)
```

## Download Options

### Text File

```python
# Prepare content
download_content = f"""
Query: {query}
Generated: {datetime.now()}

Answer:
{st.session_state.answer}

References:
{format_references(st.session_state.references)}
"""

# Download button
st.download_button(
    label="📥 Download as TXT",
    data=download_content,
    file_name=f"search_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)
```

### JSON File

```python
import json

# Prepare JSON
result_json = {
    "query": query,
    "answer": st.session_state.answer,
    "references": st.session_state.references,
    "timestamp": datetime.now().isoformat()
}

# Download button
st.download_button(
    label="📥 Download as JSON",
    data=json.dumps(result_json, indent=2, ensure_ascii=False),
    file_name=f"search_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    mime="application/json"
)
```

## Error Handling in UI

### Try-Except with User-Friendly Messages

```python
try:
    # Perform search
    results = search(query)

except Exception as e:
    # Log technical error
    logger.error(f"Search failed: {e}", exc_info=True)

    # Show user-friendly message
    col1_error, col2_error, col3_error = st.columns([1, 6, 1])
    with col2_error:
        st.error(f"❌ An error occurred: {str(e)}")

        # Expandable technical details
        with st.expander("🔍 Error Details"):
            st.write("**Error Type:**", type(e).__name__)
            st.write("**Error Message:**", str(e))

            import traceback
            st.code(traceback.format_exc(), language="python")

            st.write("**Troubleshooting:**")
            st.write("1. Check that your API keys are valid")
            st.write("2. Ensure you have internet connectivity")
            st.write("3. Try a different search query")
```

## Performance Optimization

### Caching

```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_config():
    """Load configuration (cached)"""
    with open('config.yaml') as f:
        return yaml.safe_load(f)

@st.cache_resource  # Cache resource (don't serialize)
def get_embedding_model():
    """Load embedding model (cached)"""
    return SentenceTransformer('all-MiniLM-L6-v2')

# Usage
config = load_config()  # Only loads once per hour
model = get_embedding_model()  # Only loads once per session
```

**When to use:**
- `@st.cache_data`: For data (dicts, lists, DataFrames)
- `@st.cache_resource`: For models, database connections

### Lazy Loading

```python
# Don't load until needed
if 'embedding_model' not in st.session_state:
    if user_clicks_search:
        st.session_state.embedding_model = load_model()

# Use model
if 'embedding_model' in st.session_state:
    embeddings = st.session_state.embedding_model.encode(text)
```

## Responsive Design

### Mobile-Friendly Layouts

```python
# Detect screen size (not directly possible, but adapt layout)

# Desktop: Side-by-side
col1, col2 = st.columns(2)
with col1:
    st.image("image1.png")
with col2:
    st.write("Description")

# Mobile-friendly alternative: Stack vertically
st.image("image1.png")
st.write("Description")
```

### Collapsible Sections for Mobile

```python
# Use expanders for dense content
with st.expander("Advanced Options"):
    # Many options here
    # Collapsed by default, saving screen space
    pass
```

## Best Practices

### 1. Use Placeholders for Dynamic Content

```python
# Good: Reusable placeholder
placeholder = st.empty()
placeholder.info("Loading...")
# ...do work...
placeholder.success("Done!")

# Bad: Multiple static elements
st.info("Loading...")
# ...do work...
st.success("Done!")  # Now you have both messages!
```

### 2. Minimize Reruns

```python
# Bad: Button in main flow (reruns whole script)
if st.button("Process"):
    expensive_computation()

# Better: Use session state flag
if 'processed' not in st.session_state:
    st.session_state.processed = False

if st.button("Process"):
    st.session_state.processed = True

if st.session_state.processed:
    result = expensive_computation()
    st.write(result)
```

### 3. Clear Visual Hierarchy

```python
# Main heading
st.title("🔍 Relevance Search")

# Section headings
st.markdown("## 📊 Search Results")

# Subsections
st.markdown("### Top Results")

# Use horizontal rules to separate
st.markdown("---")
```

## Running the App

### Local Development

```bash
# From project directory
streamlit run app.py

# With custom port
streamlit run app.py --server.port 8080

# With auto-reload on file changes (default)
streamlit run app.py
```

### Deployment

```bash
# Streamlit Community Cloud (free)
# 1. Push to GitHub
# 2. Go to https://streamlit.io/cloud
# 3. Connect repository
# 4. Deploy!

# Heroku
heroku create my-searchgpt-app
git push heroku main

# Docker
# See 06_ADVANCED_TOPICS.md for Docker setup
```

## Summary

### Key Concepts

- **Components**: Buttons, text inputs, selectboxes, etc.
- **Layout**: Columns, sidebar, tabs, expanders
- **State**: st.session_state for persistence
- **Placeholders**: st.empty() for dynamic updates
- **Caching**: @st.cache_data and @st.cache_resource

### Common Patterns

```python
# Input → Process → Output
query = st.text_input("Query")
if st.button("Search"):
    result = process(query)
    st.write(result)

# Progress indication
with st.spinner("Processing..."):
    result = long_operation()

# Streaming updates
placeholder = st.empty()
for chunk in stream:
    text += chunk
    placeholder.write(text)
```

### Next Steps

- **06_ADVANCED_TOPICS.md**: Production deployment, security, optimization

You should now understand:
- ✓ Streamlit components and layout
- ✓ State management patterns
- ✓ Real-time UI updates
- ✓ Error handling in UI
- ✓ Performance optimization
