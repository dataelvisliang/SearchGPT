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
st.markdown("### AI-Powered Search Engine with OpenRouter")

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

    st.markdown("---")

    # Model Selection
    st.markdown("### 🤖 Model Selection")

    model_name = st.selectbox(
        "LLM Model",
        [
            "x-ai/grok-4.1-fast:free",
            "openai/gpt-oss-20b:free",
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-sonnet"
        ],
        help="Free models end with ':free'. Others require credits."
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
        # Update config with API keys if provided
        if openrouter_api_key or serper_api_key:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), 'src', 'config', 'config.yaml')

            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Temporarily update config with user-provided keys
            if openrouter_api_key:
                config['openrouter_api_key'] = openrouter_api_key
            if serper_api_key:
                config['serper_api_key'] = serper_api_key
            if model_name:
                config['model_name'] = model_name

            # Write temporary config
            with open(config_path, 'w') as file:
                yaml.dump(config, file)

        # Show loading animation
        with st.spinner(""):
            lottie_loading = load_lottieurl(LOTTIE_LOADING)
            if lottie_loading:
                loading_placeholder = st.empty()
                with loading_placeholder:
                    st_lottie(lottie_loading, height=300, key="loading")

            start_time = time.time()

            # Status updates
            status_placeholder = st.empty()

            # Step 1: Fetch web content
            status_placeholder.info("🌐 Fetching web content...")
            logger.info(f"User query: {query}")

            web_contents_fetcher = WebContentFetcher(query)
            web_contents, serper_response = web_contents_fetcher.fetch()

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

            # Step 2: Retrieve relevant documents
            status_placeholder.info("🔍 Analyzing and retrieving relevant documents...")
            retriever = EmbeddingRetriever()
            relevant_docs_list = retriever.retrieve_embeddings(
                web_contents,
                serper_response['links'],
                query
            )

            # Step 3: Generate answer
            status_placeholder.info("🤖 Generating AI-powered answer...")
            content_processor = GPTAnswer()
            formatted_relevant_docs = content_processor._format_reference(
                relevant_docs_list,
                serper_response['links']
            )

            # Get answer from LLM
            ai_message_obj = content_processor.get_answer(
                query,
                formatted_relevant_docs,
                serper_response['language'],
                output_format if output_format else "",
                profile
            )

            answer = ai_message_obj.content
            end_time = time.time()

            # Store results in session state
            st.session_state.answer = answer
            st.session_state.references = serper_response
            st.session_state.search_time = end_time - start_time

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

# Display results
if st.session_state.answer:
    st.markdown("---")

    # Answer section
    st.markdown("## 💡 AI-Generated Answer")
    st.markdown(f"""
        <div class="result-card">
            {st.session_state.answer.replace(chr(10), '<br>')}
        </div>
    """, unsafe_allow_html=True)

    # References section
    if st.session_state.references:
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

    # Download options
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
            "query": query,
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
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Powered by OpenRouter AI • Built with Streamlit</p>
        <p>🔍 SearchGPT - Your AI-Powered Research Assistant</p>
    </div>
""", unsafe_allow_html=True)

# python -m streamlit run app.py