# LLM Integration and Answer Generation

## Overview

This document covers how Relevance Search integrates with Large Language Models (LLMs) to generate comprehensive answers from retrieved content.

```
Retrieved Chunks → Format Context → LLM API → Streaming Response → User
```

## LLM Service Architecture

### OpenRouter: Multi-Model Gateway

**What is OpenRouter?**
- Unified API for 100+ AI models
- Single API key for GPT-4, Claude, Grok, etc.
- Built-in rate limiting and fallbacks
- Cost tracking and analytics

**Why Use OpenRouter?**
```python
# Without OpenRouter (manage multiple APIs)
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq

openai_client = OpenAI(api_key=openai_key)
anthropic_client = Anthropic(api_key=anthropic_key)
groq_client = Groq(api_key=groq_key)

# With OpenRouter (single API)
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
)

# Access any model with same interface
response = client.chat.completions.create(
    model="x-ai/grok-4.1-fast:free",  # or "anthropic/claude-3", etc.
    messages=[...]
)
```

### LLM Service Implementation

```python
# src/llm_service.py

class OpenRouterService:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1"

    def chat_completion(self, messages, temperature=0.7, stream=False):
        """
        Generate chat completion

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            temperature: Randomness (0.0 = deterministic, 1.0 = creative)
            stream: Whether to stream response word-by-word

        Returns:
            Full response text or streaming generator
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",  # Required by OpenRouter
            "X-Title": "Relevance Search Application"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }

        if stream:
            return self._stream_response(payload, headers)
        else:
            return self._complete_response(payload, headers)

    def _complete_response(self, payload, headers):
        """Get complete response at once"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()
        result = response.json()

        return result['choices'][0]['message']['content']

    def _stream_response(self, payload, headers):
        """Stream response word-by-word using SSE"""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True,  # Enable streaming
            timeout=120
        )

        response.raise_for_status()

        # Yield chunks as they arrive
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')

                # OpenRouter uses Server-Sent Events (SSE) format
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix

                    if data == '[DONE]':
                        break

                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
```

## Available Models in Relevance Search

### 1. Grok 4.1 Fast (Default)

```python
model = "x-ai/grok-4.1-fast:free"
```

**Characteristics:**
- **Speed**: Very fast (1-2 seconds for typical response)
- **Quality**: GPT-4 level reasoning
- **Cost**: FREE on OpenRouter
- **Context**: 128K tokens
- **Best for**: General search queries, technical questions

**Example:**
```python
query = "What is quantum entanglement?"
# Response time: ~1.5 seconds
# Quality: Excellent, accurate citations
```

### 2. GPT-OSS 20B

```python
model = "openai/gpt-oss-20b:free"
```

**Characteristics:**
- **Speed**: Medium (2-4 seconds)
- **Quality**: GPT-3.5 level
- **Cost**: FREE
- **Context**: 8K tokens
- **Best for**: Simple queries, summaries

**Example:**
```python
query = "Summarize Python benefits"
# Response time: ~2.5 seconds
# Quality: Good for straightforward topics
```

### Model Selection Logic

```python
# In app.py
model_name = st.selectbox(
    "LLM Model",
    [
        "x-ai/grok-4.1-fast:free",  # Default
        "openai/gpt-oss-20b:free"
    ],
    help="Free models via OpenRouter"
)
```

**When to use which:**
- **Grok**: Complex queries, need citations, technical topics
- **GPT-OSS**: Simple questions, budget constraints, speed priority

## Prompt Engineering for RAG

### Anatomy of a RAG Prompt

```python
template = """
Web search result:
{context_str}

Instructions: You are a {profile}.
Using the provided web search results, write a comprehensive and detailed
reply to the given query.

Make sure to cite results using [number] notation after the reference.

At the end of the answer, list the corresponding references with indexes.
Each reference should include:
- The URL
- A relevant quote from that source (exactly as it appears in the search results)

Example format:
[1] https://example.com
    "Exact quote from the source that supports the information cited."

Answer in language: {language}
Query: {query}
Output Format: {format}
"""
```

### Component Breakdown

#### 1. Context Section

```python
Web search result:
[1] https://physics.org/quantum-mechanics
Quantum entanglement occurs when two particles become connected...

[2] https://wikipedia.org/Quantum_entanglement
Einstein famously called entanglement "spooky action at a distance"...

[3] https://ibm.com/quantum-computing
Modern quantum computers leverage entanglement for computation...
```

**Purpose:**
- Provides factual grounding
- Prevents hallucination
- Enables citation

**Format:**
- Numbered references [1], [2], [3]
- URL first (for attribution)
- Content excerpt (actual information)

#### 2. Role Assignment

```python
"You are a {profile}."

# Options in Relevance Search:
profiles = [
    "conscientious researcher",    # Default: balanced, thorough
    "technical expert",            # Technical jargon, precision
    "simple explainer",            # ELI5 style
    "professional writer"          # Polished, formal
]
```

**Why This Matters:**

```python
# Same query, different roles:

query = "What is a neural network?"

# Role: Technical expert
"A neural network is a directed acyclic graph of parameterized
 nonlinear transformations, trained via backpropagation using
 gradient descent optimization..."

# Role: Simple explainer
"A neural network is like a brain made of math! It has layers of
 connected nodes that learn patterns from examples..."
```

#### 3. Citation Instructions

```python
"Make sure to cite results using [number] notation after the reference."
```

**Example Output:**

```
Quantum entanglement is a phenomenon where two particles become
correlated [1]. Einstein called this "spooky action at a distance" [2].
Modern quantum computers use entanglement for parallel processing [3].

References:
[1] https://physics.org/quantum-mechanics
    "Quantum entanglement occurs when two particles become connected"
[2] https://wikipedia.org/Quantum_entanglement
    "Einstein famously called entanglement 'spooky action at a distance'"
[3] https://ibm.com/quantum-computing
    "Modern quantum computers leverage entanglement for computation"
```

**Why Citations:**
- Enables fact-checking
- Builds trust
- Prevents hallucination (LLM must use provided sources)

#### 4. Language Specification

```python
"Answer in language: {language}"

# Detected from query
language = "zh-cn" if contains_chinese(query) else "en-us"
```

**Example:**

```python
Query (Chinese): "什么是量子纠缠?"
Language: "zh-cn"

Response:
"量子纠缠是一种现象，当两个粒子相互连接时... [1]"

Query (English): "What is quantum entanglement?"
Language: "en-us"

Response:
"Quantum entanglement is a phenomenon where two particles... [1]"
```

#### 5. Output Format

```python
# User can specify format
formats = [
    "",                    # Default (no specific format)
    "bullet points",       # Concise lists
    "detailed paragraphs", # Long-form
    "step-by-step guide"   # Instructional
]
```

**Example with Format:**

```python
Query: "How to make coffee?"
Format: "step-by-step guide"

Response:
"**How to Make Coffee: Step-by-Step Guide**

1. **Boil water**: Heat water to 195-205°F [1]
2. **Grind beans**: Use medium grind for drip coffee [2]
3. **Add grounds**: 1-2 tablespoons per 6oz water [1]
4. **Brew**: Pour hot water over grounds [2]
5. **Enjoy**: Serve within 30 minutes for best flavor [3]

References:
[1] https://coffee.org/brewing-guide
..."
```

## Streaming Implementation

### Why Stream?

**Without Streaming:**
```
User waits 10 seconds → Entire answer appears at once
```
❌ Poor UX, feels slow

**With Streaming:**
```
User waits 0.5s → Words appear progressively → Total 10 seconds
```
✓ Feels responsive, engaging

### Server-Sent Events (SSE)

OpenRouter uses SSE for streaming:

```
Client → Server: "stream": true
Server → Client: (continuous stream)

data: {"choices":[{"delta":{"content":"Quantum"}}]}
data: {"choices":[{"delta":{"content":" entanglement"}}]}
data: {"choices":[{"delta":{"content":" is"}}]}
data: {"choices":[{"delta":{"content":" a"}}]}
...
data: [DONE]
```

### Implementation in app.py

```python
# In app.py (streaming section)

import requests as req

# Setup
headers = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Relevance Search Application"
}

payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.0,
    "stream": True  # Enable streaming
}

# Make streaming request
response = req.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=payload,
    stream=True,  # Important!
    timeout=120
)

# Display placeholder
answer_placeholder = st.empty()
full_answer = ""

# Process stream
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')

        if line.startswith('data: '):
            data = line[6:]  # Remove "data: " prefix

            if data == '[DONE]':
                break

            try:
                chunk = json.loads(data)

                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        full_answer += content
                        # Update UI in real-time
                        answer_placeholder.markdown(full_answer)

            except json.JSONDecodeError:
                continue  # Skip malformed chunks
```

### Streamlit Integration

**Challenge**: Streamlit reruns entire script on interaction.

**Solution**: Use `st.empty()` placeholder and update it.

```python
# Create placeholder
answer_placeholder = st.empty()

# Update as chunks arrive
for chunk in stream:
    full_answer += chunk
    answer_placeholder.markdown(full_answer)  # Real-time update!
```

**Visual Effect:**

```
Time 0.0s: [empty]
Time 0.5s: "Quantum"
Time 1.0s: "Quantum entanglement"
Time 1.5s: "Quantum entanglement is a"
Time 2.0s: "Quantum entanglement is a phenomenon"
...
Time 10s: [complete answer]
```

## Token Management

### What are Tokens?

Tokens are pieces of words:

```python
"Quantum computing" → ["Quantum", " computing"] = 2 tokens
"AI" → ["AI"] = 1 token
"machine learning" → ["machine", " learning"] = 2 tokens
```

**Rule of thumb**: 1 token ≈ 4 characters (English)

### Token Limits

| Model | Context Window | Max Output |
|-------|----------------|------------|
| Grok 4.1 Fast | 128,000 tokens | 4,096 tokens |
| GPT-OSS 20B | 8,192 tokens | 2,048 tokens |

**Context window** = Input + Output combined

### Managing Context Length

```python
def truncate_context(chunks, max_tokens=6000):
    """Ensure context fits within model limit"""
    total_tokens = 0
    truncated_chunks = []

    for chunk in chunks:
        chunk_tokens = len(chunk) // 4  # Rough estimate

        if total_tokens + chunk_tokens > max_tokens:
            break  # Stop adding chunks

        truncated_chunks.append(chunk)
        total_tokens += chunk_tokens

    return truncated_chunks

# Usage
top_chunks = retrieve_chunks(query, k=20)  # Get 20 candidates
fitted_chunks = truncate_context(top_chunks, max_tokens=6000)
# Might only keep 12-15 chunks
```

### Token Counting (Accurate)

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    """Accurately count tokens for a model"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example
text = "Quantum computing uses qubits for parallel computation"
tokens = count_tokens(text)  # Returns: 10
```

## Temperature and Sampling

### Temperature Parameter

Controls randomness in LLM output:

```python
temperature = 0.0  # Deterministic (same input → same output)
temperature = 0.5  # Balanced
temperature = 1.0  # Creative (varied outputs)
temperature = 2.0  # Very random (often incoherent)
```

**For RAG Systems:**

```python
# Factual Q&A (Relevance Search)
temperature = 0.0  # We want consistent, accurate answers

# Creative writing
temperature = 0.8  # We want variety

# Brainstorming
temperature = 1.2  # We want diverse ideas
```

**Example:**

```python
Query: "What is 2+2?"

# Temperature 0.0 (always same)
"2+2 equals 4."

# Temperature 1.0 (varied)
"The sum of 2 and 2 is 4."
"2 plus 2 makes 4."
"Adding 2 and 2 gives you 4."
```

### Other Sampling Parameters

```python
payload = {
    "model": "x-ai/grok-4.1-fast:free",
    "messages": messages,
    "temperature": 0.7,
    "top_p": 0.9,        # Nucleus sampling
    "max_tokens": 2048,  # Max output length
    "stop": ["\n\n\n"],  # Stop at triple newline
}
```

**Top-p (Nucleus Sampling):**
- Consider only tokens with cumulative probability ≥ p
- `top_p=1.0`: Consider all tokens
- `top_p=0.9`: Consider top 90% probable tokens
- `top_p=0.1`: Very focused, conservative

**For RAG:**
```python
temperature = 0.0
top_p = 1.0  # Don't need nucleus sampling if temp=0
```

## Error Handling

### Common API Errors

#### 1. Rate Limiting

```python
try:
    response = llm_service.chat_completion(messages)
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        # Rate limit hit
        logger.warning("Rate limit reached, retrying in 5s...")
        time.sleep(5)
        response = llm_service.chat_completion(messages)  # Retry
    else:
        raise
```

#### 2. Timeout

```python
try:
    response = requests.post(
        url,
        json=payload,
        timeout=30  # 30 second timeout
    )
except requests.exceptions.Timeout:
    logger.error("LLM request timed out")
    # Fallback: return summary of chunks without LLM
    return "\n\n".join(chunks[:3])
```

#### 3. Invalid API Key

```python
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        raise ValueError("Invalid OpenRouter API key")
    elif e.response.status_code == 403:
        raise ValueError("API key doesn't have access to this model")
    else:
        raise
```

### Retry Logic with Exponential Backoff

```python
import time

def llm_with_retry(payload, max_retries=3):
    """Retry LLM call with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                # Last attempt, give up
                raise

            # Exponential backoff: 2^attempt seconds
            wait_time = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

# Usage
result = llm_with_retry(payload, max_retries=3)
```

## Cost Optimization

### Free Tier Models

Relevance Search only uses **free models**:

```python
FREE_MODELS = [
    "x-ai/grok-4.1-fast:free",      # $0.00 per million tokens
    "openai/gpt-oss-20b:free"       # $0.00 per million tokens
]
```

**Paid alternatives (for reference):**
```python
PAID_MODELS = {
    "openai/gpt-4": "$30 per million tokens",
    "anthropic/claude-3-opus": "$15 per million tokens",
    "google/gemini-pro": "$0.50 per million tokens"
}
```

### Reducing Token Usage

```python
# 1. Limit retrieved chunks
chunks = retrieve_chunks(query, k=10)  # Not k=50

# 2. Truncate long chunks
max_chunk_length = 500  # characters
chunks = [chunk[:max_chunk_length] for chunk in chunks]

# 3. Use concise prompts
# Don't: "Please kindly provide a comprehensive and exhaustive..."
# Do: "Write a detailed reply to:"

# 4. Set max_tokens for output
payload = {
    "max_tokens": 1024  # Limit response length
}
```

## Advanced Techniques

### 1. Chain-of-Thought Prompting

For complex queries, ask LLM to think step-by-step:

```python
prompt = f"""
Web search results:
{context}

Think step-by-step:
1. What are the key facts in the search results?
2. How do they relate to the query?
3. What's the most accurate answer?

Query: {query}
Answer:
"""
```

### 2. Self-Consistency

Generate multiple answers and pick the most consistent:

```python
# Generate 3 answers with temperature=0.7
answers = []
for _ in range(3):
    answer = llm_service.chat_completion(
        messages,
        temperature=0.7
    )
    answers.append(answer)

# Pick most common answer (simplified)
from collections import Counter
most_common = Counter(answers).most_common(1)[0][0]
```

### 3. Critique and Refine

Let LLM critique its own answer:

```python
# Step 1: Generate initial answer
initial_answer = llm_service.chat_completion(messages)

# Step 2: Critique
critique_prompt = f"""
Original answer:
{initial_answer}

Source material:
{context}

Critique: Is this answer accurate? What's missing?
"""

critique = llm_service.chat_completion([
    {"role": "user", "content": critique_prompt}
])

# Step 3: Refine based on critique
refine_prompt = f"""
Original answer: {initial_answer}
Critique: {critique}
Sources: {context}

Provide an improved answer:
"""

final_answer = llm_service.chat_completion([
    {"role": "user", "content": refine_prompt}
])
```

## Evaluation

### Answer Quality Metrics

```python
def evaluate_answer(answer, query, sources):
    """Evaluate answer quality"""

    metrics = {}

    # 1. Faithfulness (does answer match sources?)
    metrics['faithfulness'] = check_faithfulness(answer, sources)

    # 2. Relevance (does answer address query?)
    metrics['relevance'] = check_relevance(answer, query)

    # 3. Citation coverage (are claims cited?)
    metrics['citation_coverage'] = check_citations(answer)

    # 4. Completeness (all aspects covered?)
    metrics['completeness'] = check_completeness(answer, query)

    return metrics
```

### Example Test Cases

```python
test_cases = [
    {
        "query": "What is photosynthesis?",
        "expected_keywords": ["light", "energy", "plants", "chlorophyll"],
        "expected_citations": True
    },
    {
        "query": "How do vaccines work?",
        "expected_keywords": ["immune", "antibodies", "virus"],
        "expected_citations": True
    }
]

for test in test_cases:
    answer = generate_answer(test['query'])

    # Check keywords present
    for keyword in test['expected_keywords']:
        assert keyword.lower() in answer.lower()

    # Check citations present
    if test['expected_citations']:
        assert '[1]' in answer  # At least one citation

    print(f"✓ Test passed: {test['query']}")
```

## Summary

### Key Takeaways

1. **OpenRouter** provides unified access to multiple LLM models
2. **Grok 4.1 Fast** is the default (free, fast, high-quality)
3. **Streaming** provides better UX with real-time responses
4. **Prompt engineering** ensures accurate, cited answers
5. **Temperature=0.0** for consistent, factual RAG responses
6. **Error handling** with retries ensures reliability

### Best Practices

✓ Use streaming for better UX
✓ Set temperature=0.0 for factual answers
✓ Include clear citation instructions
✓ Handle errors gracefully with retries
✓ Monitor token usage to stay within limits
✓ Test with diverse queries

### Next Steps

- **05_STREAMLIT_UI.md**: Build user interfaces for LLM interactions
- **06_ADVANCED_TOPICS.md**: Production deployment and optimization

You should now understand:
- ✓ How LLM APIs work (OpenRouter)
- ✓ Streaming implementation with SSE
- ✓ Prompt engineering for RAG
- ✓ Token management and cost optimization
- ✓ Error handling patterns
