import streamlit as st
import sys
import os
import time
import json
import logging
from streamlit_lottie import st_lottie
import requests

# Configure logging to capture all logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fetch_web_content import WebContentFetcher
from retrieval import EmbeddingRetriever
from llm_answer import GPTAnswer
from locate_reference import ReferenceLocator

# Page configuration
st.set_page_config(
    page_title="SearchGPT - AI-Powered Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
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
    .search-box {
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .reference-card {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 14px;
    }
    .stat-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Generating answer animation */
    .generating-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        gap: 12px;
    }

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

    .generating-text {
        font-size: 16px;
        color: #666;
        font-weight: 500;
    }

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

# Function to load Lottie animations
def load_lottieurl(url: str):
    """Load Lottie animation from URL"""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Lottie animation URLs
LOTTIE_SEARCH = "https://assets5.lottiefiles.com/packages/lf20_yw2vtlgx.json"  # Search animation
LOTTIE_LOADING = "https://assets2.lottiefiles.com/packages/lf20_a2chheio.json"  # Loading animation
LOTTIE_SUCCESS = "https://assets9.lottiefiles.com/packages/lf20_s2lryxtd.json"  # Success animation

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'answer' not in st.session_state:
    st.session_state.answer = None
if 'references' not in st.session_state:
    st.session_state.references = None
if 'search_time' not in st.session_state:
    st.session_state.search_time = 0

# Header
st.title("🔍 SearchGPT")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Load Lottie animation
    lottie_search = load_lottieurl(LOTTIE_SEARCH)
    if lottie_search:
        st_lottie(lottie_search, height=200, key="search_animation")

    st.markdown("---")

    # API Keys Section
    st.markdown("### 🔑 API Keys")

    # Check if config file exists and has keys
    config_path = os.path.join(os.path.dirname(__file__), 'src', 'config', 'config.yaml')
    config_has_keys = False

    try:
        import yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            if config.get('openrouter_api_key') and config.get('serper_api_key'):
                config_has_keys = True
    except:
        pass

    # OpenRouter API Key
    openrouter_api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
        help="Get your free API key at https://openrouter.ai/",
        value=""
    )

    if not openrouter_api_key and not config_has_keys:
        st.info("💡 Get a free API key at [OpenRouter](https://openrouter.ai/)")

    # Serper API Key
    serper_api_key = st.text_input(
        "Serper API Key",
        type="password",
        placeholder="Your Serper API key",
        help="Get your API key at https://serper.dev/",
        value=""
    )

    if not serper_api_key and not config_has_keys:
        st.info("💡 Get your API key at [Serper](https://serper.dev/)")

    # Gitee API Key
    gitee_api_key = st.text_input(
        "Gitee AI API Key (Optional - for BGE-M3 embeddings)",
        type="password",
        placeholder="Your Gitee AI API key",
        help="Get your API key at https://ai.gitee.com/. If not provided, OpenRouter embeddings will be used.",
        value=""
    )

    if gitee_api_key:
        st.success("✅ Using Gitee AI BGE-M3 embeddings")
    else:
        st.info("💡 Using OpenRouter text-embedding-3-small (default)")

    st.markdown("---")

    # Model Selection
    st.markdown("### 🤖 Model Selection")

    model_name = st.selectbox(
        "LLM Model",
        [
            "x-ai/grok-4.1-fast:free",
            "openai/gpt-oss-20b:free"
        ],
        help="Free models via OpenRouter"
    )

    st.markdown("---")

    # Settings
    st.markdown("### 📝 Output Settings")

    output_format = st.text_area(
        "Output Format (optional)",
        placeholder="e.g., 'Provide a summary in bullet points'",
        help="Specify how you want the answer formatted"
    )

    profile = st.selectbox(
        "AI Profile",
        ["Conscientious Researcher", "Technical Expert", "Business Analyst",
         "Journalist", "Educator", "Custom"],
        help="Choose the perspective for the AI's response"
    )

    if profile == "Custom":
        profile = st.text_input("Enter custom profile:", "")
    elif profile == "Conscientious Researcher":
        profile = ""  # Default value

    st.markdown("---")

    # Display statistics if search has been performed
    if st.session_state.answer:
        st.markdown("### 📊 Search Statistics")
        st.metric("Search Time", f"{st.session_state.search_time:.2f}s")
        if st.session_state.references:
            st.metric("References Found", len(st.session_state.references.get('links', [])))

# Main content
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Search input
    query = st.text_input(
        "",
        placeholder="🔎 Enter your search query...",
        key="search_input",
        label_visibility="collapsed"
    )

    # Search button
    search_button = st.button("🚀 Search", use_container_width=True)

# Perform search
if search_button and query:
    # Validate API keys
    if not openrouter_api_key and not config_has_keys:
        st.error("❌ Please enter your OpenRouter API key in the sidebar.")
        st.stop()

    if not serper_api_key and not config_has_keys:
        st.error("❌ Please enter your Serper API key in the sidebar.")
        st.stop()

    try:
        # Temporarily update config file with API keys for this search session
        config_updated = False
        original_config = None

        if openrouter_api_key or serper_api_key or gitee_api_key:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), 'src', 'config', 'config.yaml')

            # Read and backup original config
            with open(config_path, 'r') as file:
                original_config = file.read()
                config = yaml.safe_load(original_config)

            # Update config with user-provided keys
            if openrouter_api_key:
                config['openrouter_api_key'] = openrouter_api_key
            if serper_api_key:
                config['serper_api_key'] = serper_api_key
            if gitee_api_key:
                config['gitee_api_key'] = gitee_api_key
            if model_name:
                config['model_name'] = model_name

            # Write updated config temporarily
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
            config_updated = True

        # Show loading animation and status in centered columns
        start_time = time.time()

        # Create centered columns for loading animation and status
        col1_status, col2_status, col3_status = st.columns([1, 6, 1])

        with col2_status:
            # Loading animation placeholder
            loading_placeholder = st.empty()
            lottie_loading = load_lottieurl(LOTTIE_LOADING)
            if lottie_loading:
                with loading_placeholder:
                    st_lottie(lottie_loading, height=200, key="loading")

            # Status message placeholder
            status_placeholder = st.empty()

        # Initialize trace data storage
        trace_data = {
            'query': query,
            'search_results': {},
            'scraped_content': {},
            'chunks': {},
            'embeddings': {},
            'retrieval': {},
            'generation': {}
        }

        # Step 1: Fetch web content
        with col2_status:
            status_placeholder.info("🌐 **Step 1/4:** Searching the web...")
        logger.info(f"User query: {query}")

        web_contents_fetcher = WebContentFetcher(query)
        web_contents, serper_response = web_contents_fetcher.fetch()

        # Trace: Search results
        trace_data['search_results'] = {
            'count': serper_response.get('count', 0) if serper_response else 0,
            'urls': serper_response.get('links', []) if serper_response else [],
            'titles': serper_response.get('titles', []) if serper_response else [],
            'snippets': serper_response.get('snippets', []) if serper_response else []
        }

        # Show how many URLs were found
        if serper_response:
            with col2_status:
                status_placeholder.success(f"✅ Found {serper_response.get('count', 0)} results")
                time.sleep(0.5)

            logger.info(f"Fetch completed: web_contents={len(web_contents) if web_contents else 0}, serper_response={'OK' if serper_response else 'None'}")

            if not serper_response:
                st.error("❌ Search service error. Please check your Serper API key and try again.")
                with st.expander("🔍 Debug Info"):
                    st.write("No response from Serper API. This could mean:")
                    st.write("- Invalid Serper API key")
                    st.write("- Network connectivity issues")
                    st.write("- Serper API rate limit reached")
                st.stop()

            if not web_contents or all(not content for content in web_contents):
                st.error("❌ Could not fetch content from any search results.")
                with st.expander("🔍 Debug Info"):
                    st.write(f"Found {serper_response.get('count', 0)} search results, but could not scrape content from any URLs.")
                    st.write("Possible reasons:")
                    st.write("- Websites are blocking automated access")
                    st.write("- Network timeout or connectivity issues")
                    st.write("URLs found:")
                    for i, link in enumerate(serper_response.get('links', [])[:5], 1):
                        st.write(f"{i}. {link}")
                st.stop()

            # Trace: Scraped content
            trace_data['scraped_content'] = {
                'total_pages': len(web_contents),
                'successful_scrapes': sum(1 for c in web_contents if c),
                'failed_scrapes': sum(1 for c in web_contents if not c),
                'content_lengths': [len(c) if c else 0 for c in web_contents],
                'content_previews': [
                    {
                        'url': serper_response['links'][i],
                        'length': len(c) if c else 0,
                        'preview': c[:200] + '...' if c and len(c) > 200 else (c if c else '[FAILED]')
                    }
                    for i, c in enumerate(web_contents)
                ]
            }

            # Step 2: Fetch content from URLs
            with col2_status:
                status_placeholder.info(f"📄 **Step 2/4:** Fetching content from {sum(1 for c in web_contents if c)} pages...")
            time.sleep(0.5)

            # Step 3: Create embeddings and retrieve relevant documents
            with col2_status:
                status_placeholder.info("🔍 **Step 3/4:** Creating embeddings and searching...")
            retriever = EmbeddingRetriever()
            relevant_docs_list = retriever.retrieve_embeddings(
                web_contents,
                serper_response['links'],
                query
            )

            # Trace: Chunks and embeddings with similarity scores
            trace_data['retrieval'] = {
                'num_relevant_docs': len(relevant_docs_list),
                'relevant_docs': [
                    {
                        'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                        'full_length': len(doc.page_content),
                        'source': doc.metadata.get('url', 'Unknown'),
                        'similarity_score': doc.metadata.get('similarity_score', 'N/A')
                    }
                    for doc in relevant_docs_list
                ]
            }

            with col2_status:
                status_placeholder.success(f"✅ Found {len(relevant_docs_list)} relevant sections")
                time.sleep(0.5)

            # Step 4: Generate answer with streaming
            with col2_status:
                status_placeholder.info("🤖 **Step 4/4:** Generating AI answer (streaming)...")

            content_processor = GPTAnswer()
            formatted_relevant_docs = content_processor._format_reference(
                relevant_docs_list,
                serper_response['links']
            )

            # Clear status and loading animation for streaming output
            status_placeholder.empty()
            if lottie_loading:
                loading_placeholder.empty()

            # Display answer with streaming
            col1_answer, col2_answer, col3_answer = st.columns([1, 6, 1])
            with col2_answer:
                answer_placeholder = st.empty()
                # Show generating animation initially
                answer_placeholder.markdown("""
                    <div class="generating-container">
                        <div class="spinner"></div>
                        <div class="generating-text">Generating your answers<span class="breathing-dots"></span></div>
                    </div>
                """, unsafe_allow_html=True)

            # Get streaming answer from LLM
            from llm_service import OpenRouterService
            llm_service = OpenRouterService(
                api_key=content_processor.api_key,
                model_name=content_processor.model_name
            )

            # Build the prompt
            template = content_processor.config["template"]
            from text_utils import PromptTemplate
            prompt_template = PromptTemplate(
                input_variables=["profile", "context_str", "language", "query", "format"],
                template=template
            )
            profile_text = "conscientious researcher" if not profile else profile
            summary_prompt = prompt_template.format(
                context_str=formatted_relevant_docs,
                language=serper_response['language'],
                query=query,
                format=output_format if output_format else "",
                profile=profile_text
            )

            # Trace: Initialize generation data and store prompt/chunks
            trace_data['generation'] = {
                'model': content_processor.model_name,
                'temperature': 0.0,
                'profile': profile_text,
                'prompt': summary_prompt,
                'formatted_context': formatted_relevant_docs,
                'chunks_sent': [
                    {
                        'index': i + 1,
                        'content': doc.page_content,
                        'source': doc.metadata.get('url', 'Unknown'),
                        'similarity_score': doc.metadata.get('similarity_score', 'N/A')
                    }
                    for i, doc in enumerate(relevant_docs_list)
                ]
            }

            # Stream the response
            messages = [{"role": "user", "content": summary_prompt}]
            full_answer = ""

            # Use streaming with word-by-word display
            import requests as req
            headers = {
                "Authorization": f"Bearer {content_processor.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "SearchGPT Application"
            }
            payload = {
                "model": content_processor.model_name,
                "messages": messages,
                "temperature": 0.0,
                "stream": True
            }

            response = req.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_answer += content
                                    # Update display with markdown rendering
                                    answer_placeholder.markdown(full_answer)
                        except json.JSONDecodeError:
                            continue

            answer = full_answer
            end_time = time.time()

            # Trace: Update generation info (don't overwrite, update existing data)
            trace_data['generation'].update({
                'answer_length': len(full_answer),
                'answer_preview': full_answer[:300] + '...' if len(full_answer) > 300 else full_answer,
                'time_taken': end_time - start_time
            })

            # Store results in session state
            st.session_state.answer = answer
            st.session_state.references = serper_response
            st.session_state.search_time = end_time - start_time
            st.session_state.trace_data = trace_data  # Store trace data

            # Clear loading animation
            if lottie_loading:
                loading_placeholder.empty()
            status_placeholder.empty()

            # Show success animation briefly
            success_placeholder = st.empty()
            lottie_success = load_lottieurl(LOTTIE_SUCCESS)
            if lottie_success:
                with success_placeholder:
                    st_lottie(lottie_success, height=150, key="success")
                time.sleep(1)
                success_placeholder.empty()

    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)

        # Display error in centered columns matching search bar width
        col1_error, col2_error, col3_error = st.columns([1, 6, 1])
        with col2_error:
            st.error(f"❌ An error occurred: {str(e)}")

            with st.expander("🔍 Error Details"):
                st.write("**Error Type:**", type(e).__name__)
                st.write("**Error Message:**", str(e))

                # Show detailed traceback
                import traceback
                st.code(traceback.format_exc(), language="python")

                st.write("**Troubleshooting:**")
                st.write("1. Check that your API keys are valid")
                st.write("2. Ensure you have internet connectivity")
                st.write("3. Try a different search query")
                st.write("4. Check the console/terminal for detailed logs")
    finally:
        # Restore original config file if it was modified
        if config_updated and original_config:
            config_path = os.path.join(os.path.dirname(__file__), 'src', 'config', 'config.yaml')
            with open(config_path, 'w') as file:
                file.write(original_config)

# Display results (only if not streaming - streaming already displayed above)
if st.session_state.answer and not search_button:
    st.markdown("---")

    # Answer section with markdown rendering
    col1_result, col2_result, col3_result = st.columns([1, 6, 1])
    with col2_result:
        st.markdown("## 💡 AI-Generated Answer")
        st.markdown(st.session_state.answer)

    # References section in centered columns
    if st.session_state.references:
        col1_ref, col2_ref, col3_ref = st.columns([1, 6, 1])
        with col2_ref:
            st.markdown("---")
            st.markdown("## 📚 Source References")

            # Create tabs for different views
            tab1, tab2 = st.tabs(["📋 Quick Links", "🔗 Detailed Sources"])

            with tab1:
                # Display quick links
                links = st.session_state.references.get('links', [])
                titles = st.session_state.references.get('titles', [])
                snippets = st.session_state.references.get('snippets', [])

                for i, (link, title, snippet) in enumerate(zip(links, titles, snippets), 1):
                    with st.expander(f"[{i}] {title}", expanded=False):
                        st.markdown(f"**URL:** [{link}]({link})")
                        st.markdown(f"**Snippet:** {snippet}")

            with tab2:
                # Display detailed sources
                for i, (link, title, snippet) in enumerate(zip(links, titles, snippets), 1):
                    st.markdown(f"""
                        <div class="reference-card">
                            <strong>[{i}] {title}</strong><br>
                            <a href="{link}" target="_blank">{link}</a><br>
                            <em>{snippet}</em>
                        </div>
                    """, unsafe_allow_html=True)

    # Download options in centered columns
    col1_dl, col2_dl, col3_dl = st.columns([1, 6, 1])
    with col2_dl:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Download answer as text
            st.download_button(
                label="📄 Download Answer (TXT)",
                data=st.session_state.answer,
                file_name=f"searchgpt_answer_{int(time.time())}.txt",
                mime="text/plain"
            )

        with col2:
            # Download as JSON
            json_data = {
                "query": query if 'query' in locals() else "",
                "answer": st.session_state.answer,
                "references": st.session_state.references,
                "search_time": st.session_state.search_time
            }
            st.download_button(
                label="📦 Download Full Results (JSON)",
                data=json.dumps(json_data, indent=2),
                file_name=f"searchgpt_results_{int(time.time())}.json",
                mime="application/json"
            )

        with col3:
            # Clear results
            if st.button("🗑️ Clear Results"):
                st.session_state.answer = None
                st.session_state.references = None
                st.session_state.search_time = 0
                st.session_state.trace_data = None
                st.rerun()

# Pipeline Trace Section
if st.session_state.answer and st.session_state.get('trace_data'):
    col1_trace, col2_trace, col3_trace = st.columns([1, 6, 1])

    with col2_trace:
        st.markdown("---")
        st.markdown("## 🔍 Pipeline Trace")

        with st.expander("📊 View Detailed Pipeline Execution Trace", expanded=False):
            trace = st.session_state.trace_data

            # Step 1: Search Results
            st.markdown("### 🌐 Step 1: Web Search")
            st.write(f"**Query:** {trace['query']}")
            st.write(f"**Results Found:** {trace['search_results']['count']}")

            if trace['search_results']['urls']:
                st.markdown("**Search Results:**")
                for i, (url, title, snippet) in enumerate(zip(
                    trace['search_results']['urls'][:10],
                    trace['search_results']['titles'][:10],
                    trace['search_results']['snippets'][:10]
                ), 1):
                    with st.container():
                        st.markdown(f"**{i}. {title}**")
                        st.caption(url)
                        st.text(snippet[:150] + '...' if len(snippet) > 150 else snippet)
                        st.markdown("")

            # Step 2: Scraped Content
            st.markdown("---")
            st.markdown("### 📄 Step 2: Content Scraping")
            st.write(f"**Total Pages:** {trace['scraped_content']['total_pages']}")
            st.write(f"**Successful:** {trace['scraped_content']['successful_scrapes']}")
            st.write(f"**Failed:** {trace['scraped_content']['failed_scrapes']}")

            st.markdown("**Content Lengths:**")
            for i, preview in enumerate(trace['scraped_content']['content_previews'][:10], 1):
                status = "✅" if preview['length'] > 0 else "❌"
                st.markdown(f"{status} **{i}. {preview['url']}**")
                st.write(f"Length: {preview['length']} characters")
                if preview['length'] > 0:
                    with st.container():
                        st.text_area(
                            f"Preview {i}",
                            value=preview['preview'],
                            height=100,
                            key=f"preview_{i}",
                            label_visibility="collapsed"
                        )
                st.markdown("")

            # Step 3: Retrieval
            st.markdown("---")
            st.markdown("### 🔍 Step 3: Embedding & Retrieval")
            st.write(f"**Relevant Documents Retrieved:** {trace['retrieval']['num_relevant_docs']}")

            st.markdown("**Retrieved Chunks (in order of relevance):**")
            for i, doc in enumerate(trace['retrieval']['relevant_docs'], 1):
                with st.container():
                    # Show similarity score with color coding
                    score = doc['similarity_score']
                    if score != 'N/A' and score is not None:
                        score_color = "green" if score > 0.7 else ("orange" if score > 0.5 else "red")
                        st.markdown(f"**Chunk {i}** - Similarity Score: <span style='color: {score_color}; font-weight: bold;'>{score:.4f}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Chunk {i}**")

                    st.caption(f"Source: {doc['source']}")
                    st.write(f"Length: {doc['full_length']} characters")
                    st.text_area(
                        f"Chunk {i} content",
                        value=doc['content'],
                        height=120,
                        key=f"chunk_{i}",
                        label_visibility="collapsed"
                    )
                    st.markdown("")

            # Step 4: Generation
            st.markdown("---")
            st.markdown("### 🤖 Step 4: Answer Generation")
            st.write(f"**Model:** {trace['generation']['model']}")
            st.write(f"**Profile:** {trace['generation'].get('profile', 'N/A')}")
            st.write(f"**Temperature:** {trace['generation']['temperature']}")
            st.write(f"**Answer Length:** {trace['generation']['answer_length']} characters")
            st.write(f"**Time Taken:** {trace['generation']['time_taken']:.2f} seconds")

            # Show chunks sent to LLM
            st.markdown("**Chunks Sent to LLM:**")
            if trace['generation'].get('chunks_sent'):
                st.write(f"Total chunks: {len(trace['generation']['chunks_sent'])}")
                for chunk_info in trace['generation']['chunks_sent']:
                    with st.container():
                        score = chunk_info.get('similarity_score', 'N/A')
                        if score != 'N/A' and score is not None:
                            score_color = "green" if score > 0.7 else ("orange" if score > 0.5 else "red")
                            st.markdown(f"**Chunk {chunk_info['index']}** - Similarity: <span style='color: {score_color}; font-weight: bold;'>{score:.4f}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**Chunk {chunk_info['index']}**")

                        st.caption(f"Source: {chunk_info['source']}")
                        st.text_area(
                            f"LLM Chunk {chunk_info['index']}",
                            value=chunk_info['content'],
                            height=100,
                            key=f"llm_chunk_{chunk_info['index']}",
                            label_visibility="collapsed"
                        )
                        st.markdown("")

            # Show formatted context
            st.markdown("**Formatted Context (as sent to LLM):**")
            if trace['generation'].get('formatted_context'):
                st.text_area(
                    "Formatted Context",
                    value=trace['generation']['formatted_context'],
                    height=200,
                    key="formatted_context",
                    label_visibility="collapsed"
                )

            # Show full prompt
            st.markdown("**Complete Prompt Sent to LLM:**")
            if trace['generation'].get('prompt'):
                st.text_area(
                    "Full Prompt",
                    value=trace['generation']['prompt'],
                    height=300,
                    key="full_prompt",
                    label_visibility="collapsed"
                )

            st.markdown("**Generated Answer:**")
            st.text_area(
                "Generated Answer",
                value=trace['generation']['answer_preview'],
                height=150,
                key="answer_preview",
                label_visibility="collapsed"
            )

            # Summary Statistics
            st.markdown("---")
            st.markdown("### 📈 Summary Statistics")

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("Search Results", trace['search_results']['count'])

            with col_stat2:
                st.metric("Pages Scraped", trace['scraped_content']['successful_scrapes'])

            with col_stat3:
                st.metric("Chunks Retrieved", trace['retrieval']['num_relevant_docs'])

            with col_stat4:
                st.metric("Total Time", f"{trace['generation']['time_taken']:.1f}s")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Powered by OpenRouter AI • Built with Streamlit</p>
        <p>🔍 SearchGPT - Your AI-Powered Research Assistant</p>
    </div>
""", unsafe_allow_html=True)

# python -m streamlit run app.py