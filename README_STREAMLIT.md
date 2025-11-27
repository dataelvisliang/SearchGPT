# SearchGPT Streamlit Interface

A beautiful AI-powered search engine interface built with Streamlit and integrated with OpenRouter API.

## Features

- 🎨 Modern, responsive UI with Lottie animations
- 🔑 API key input directly in the interface (no need to edit config files)
- 🤖 Multiple AI model selection (including free models)
- 📊 Real-time search statistics
- 📚 Interactive reference display with expandable cards
- 💾 Download results as TXT or JSON
- 🎯 Customizable output format and AI profiles

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. The app will open in your browser at `http://localhost:8501`

3. In the sidebar, enter your API keys:
   - **OpenRouter API Key**: Get a free key at [https://openrouter.ai/](https://openrouter.ai/)
   - **Serper API Key**: Get your key at [https://serper.dev/](https://serper.dev/)

4. (Optional) Configure additional settings:
   - Select AI model (free models: `x-ai/grok-4.1-fast:free` or `openai/gpt-oss-20b:free`)
   - Set output format
   - Choose AI profile (Researcher, Technical Expert, etc.)

5. Enter your search query and click "🚀 Search"

## Interface Sections

### 🔑 API Keys
- Enter your OpenRouter and Serper API keys
- Keys can also be saved in `src/config/config.yaml` for convenience

### 🤖 Model Selection
- Choose from various AI models
- Free models are marked with `:free` suffix
- Premium models require OpenRouter credits

### 📝 Output Settings
- **Output Format**: Specify how you want the answer formatted (e.g., "bullet points", "detailed report")
- **AI Profile**: Choose the perspective for the AI's response

### 💡 Search Results
- AI-generated comprehensive answer
- Source references with expandable details
- Quick links and detailed source tabs

### 📊 Statistics
- Search time
- Number of references found

### 💾 Download Options
- Download answer as TXT
- Download full results as JSON (includes query, answer, references, and metadata)

## Available Free Models

- **x-ai/grok-4.1-fast:free** - GPT-4 equivalent (recommended)
- **openai/gpt-oss-20b:free** - GPT-3.5 equivalent

## Tips

- Use specific queries for better results
- Try different AI profiles for varied perspectives
- Export results for later reference
- Free models have rate limits but work great for most searches

## Troubleshooting

### API Key Errors
- Make sure your API keys are valid and active
- Free tier limits may apply

### No Results Found
- Try rephrasing your query
- Check your Serper API key is valid

### Dependencies Issues
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

## Screenshots

The interface includes:
- 🎬 Animated Lottie search icon
- ⏳ Loading animations during search
- ✅ Success animations on completion
- 📱 Responsive design for all screen sizes

## Contributing

Feel free to customize the interface by modifying `app.py`:
- Add new Lottie animations
- Customize CSS styling
- Add new AI profiles
- Enhance the reference display

## License

MIT License - feel free to use and modify as needed.
