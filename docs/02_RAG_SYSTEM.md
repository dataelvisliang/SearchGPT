# Understanding RAG (Retrieval-Augmented Generation)

## What is RAG?

RAG combines the best of two worlds:
1. **Retrieval**: Finding relevant information from a knowledge base
2. **Generation**: Using an AI to create natural language answers

Think of it like an open-book exam:
- **Without RAG**: The AI only knows what it memorized during training (closed book)
- **With RAG**: The AI can look up current information before answering (open book)

## The RAG Problem

### Traditional LLM Limitations

```python
# Without RAG
user: "What's the latest iPhone model in 2025?"
llm:  "I don't have information beyond my training cutoff in 2024..."

# With RAG
user: "What's the latest iPhone model in 2025?"
system: [Searches web, finds Apple's 2025 announcement]
llm:    "Based on Apple's recent announcement, the iPhone 17 Pro Max
         was released in September 2025 with..."
```

### Why Not Just Search?

Traditional search returns **links**. RAG provides **answers**.

```
Traditional Search:
Query: "How does photosynthesis work?"
Output:
  1. biology.com/photosynthesis
  2. khan academy.org/science/...
  3. wikipedia.org/photosynthesis

RAG System:
Query: "How does photosynthesis work?"
Output:
  "Photosynthesis is the process by which plants convert light
   energy into chemical energy. It occurs in two stages:

   1. Light-dependent reactions: Chlorophyll absorbs light [1]
   2. Calvin cycle: CO2 is converted to glucose [2]

   Sources:
   [1] biology.com/photosynthesis
   [2] wikipedia.org/photosynthesis"
```

## RAG Architecture in SearchGPT

### High-Level Flow

```
┌─────────────────────────────────────────────────────┐
│                   RAG PIPELINE                       │
└─────────────────────────────────────────────────────┘

Step 1: RETRIEVAL PHASE
┌──────────────┐
│ User Query   │ "What is quantum entanglement?"
└───────┬──────┘
        │
        v
┌────────────────────────┐
│   Search & Scrape      │ (Already covered in 01_SEARCH_PIPELINE.md)
│  - Serper API          │
│  - Web Crawler         │
└───────┬────────────────┘
        │
        v
    Raw Text (10 articles worth of content)
        │
        v
┌────────────────────────┐
│   Text Chunking        │ Split into 500-char pieces
└───────┬────────────────┘
        │
        v
    Text Chunks (50-100 chunks)
        │
        v
┌────────────────────────┐
│   Embed Chunks         │ Convert to vectors
│   (text → numbers)     │
└───────┬────────────────┘
        │
        v
    Vector Embeddings (384-dimensional vectors)
        │
        v
┌────────────────────────┐
│   Store in Vector DB   │ ChromaDB
└───────┬────────────────┘
        │
        v
┌────────────────────────┐
│   Similarity Search    │ Find most relevant chunks
└───────┬────────────────┘
        │
        v
    Top K Relevant Chunks (Top 10)

Step 2: GENERATION PHASE
        │
        v
┌────────────────────────┐
│   Format Context       │ Combine chunks into prompt
└───────┬────────────────┘
        │
        v
┌────────────────────────┐
│   LLM Generation       │ OpenRouter API (Grok)
│   (Context + Query)    │
└───────┬────────────────┘
        │
        v
┌────────────────────────┐
│   Streamed Answer      │ Word-by-word display
└────────────────────────┘
```

## Component: Retrieval System (retrieval.py)

### Purpose

Transform text into searchable vectors and retrieve the most relevant chunks for a query.

### Text Chunking

#### Why Chunk?

**Problem**: Web articles are long (5,000-50,000 characters)
**Solution**: Break into manageable pieces (500 characters each)

**Benefits:**
1. **Precision**: Find exact relevant paragraphs, not entire articles
2. **Context Window**: LLMs have limited input size (typically 4,096-8,192 tokens)
3. **Performance**: Smaller chunks = faster similarity search

#### The RecursiveTextSplitter

```python
class RecursiveTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]
```

**How It Works:**

```
Original Text:
"Quantum entanglement is a phenomenon where two particles become
connected. When one particle is measured, the other instantly
reflects the change.

This happens regardless of distance. Einstein called it 'spooky
action at a distance.' Modern quantum computers use entanglement
for computation."

Step 1: Try splitting by paragraph (\n\n)
Chunk 1: "Quantum entanglement is a phenomenon where two particles
          become connected. When one particle is measured, the other
          instantly reflects the change."  [150 chars ✓]

Chunk 2: "This happens regardless of distance. Einstein called it
          'spooky action at a distance.' Modern quantum computers
          use entanglement for computation." [165 chars ✓]

Step 2: Check for overlap
Chunk 1: "...instantly reflects the change."
Chunk 2 overlap: "This happens regardless..."
         (includes last 50 chars of Chunk 1)
```

**Code:**

```python
def split_text(self, text: str) -> List[str]:
    chunks = []

    # Try each separator in order
    for separator in self.separators:
        if separator in text:
            parts = text.split(separator)
            # Process each part recursively
            for part in parts:
                if len(part) <= self.chunk_size:
                    chunks.append(part)
                else:
                    # Part too large, try next separator
                    chunks.extend(self.split_text(part))
            break

    return chunks
```

**Overlap Benefits:**

```
Without Overlap:
Chunk 1: "...quantum computers use"
Chunk 2: "entanglement for computation..."
❌ Context lost: What uses entanglement?

With Overlap:
Chunk 1: "...quantum computers use entanglement"
Chunk 2: "use entanglement for computation..."
✓ Context preserved
```

### Vector Embeddings

#### What are Embeddings?

Embeddings convert text into numbers that capture meaning.

```python
# Text → Embedding (simplified)
"quantum computing" → [0.23, -0.15, 0.87, ..., 0.42]  # 384 numbers
"quantum mechanics" → [0.25, -0.13, 0.85, ..., 0.40]  # Similar!
"pizza recipe"      → [-0.67, 0.92, -0.31, ..., 0.11] # Different!
```

**Why 384 dimensions?**
- Each dimension captures a different aspect of meaning
- Dimension 1 might encode "science-relatedness"
- Dimension 2 might encode "abstractness vs concreteness"
- Dimension 100 might encode "positive vs negative sentiment"

#### Creating Embeddings

```python
class EmbeddingRetriever:
    def __init__(self):
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Choose embedding function
        if self.config.get('gitee_api_key'):
            self.embedding_function = self._get_gitee_embeddings()
        else:
            self.embedding_function = self._get_openrouter_embeddings()
```

**Two Embedding Options:**

1. **OpenRouter (default)**: `text-embedding-3-small`
   - 384 dimensions
   - Fast and efficient
   - English-optimized

2. **Gitee BGE-M3**: Bilingual embeddings
   - 384 dimensions
   - Excellent for Chinese/English mixed queries
   - Better for technical terms

#### Similarity Search

**Cosine Similarity:**

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    # Measure angle between vectors
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Example
query_vec = [0.5, 0.8, 0.1]
chunk1_vec = [0.6, 0.7, 0.2]  # Similar direction
chunk2_vec = [-0.3, -0.9, 0.1]  # Opposite direction

similarity1 = cosine_similarity(query_vec, chunk1_vec)  # 0.95 (very similar)
similarity2 = cosine_similarity(query_vec, chunk2_vec)  # -0.42 (dissimilar)
```

**Visual Representation:**

```
2D Simplified Example:

      Query: "quantum computing"
         │
         │  ← Small angle = High similarity
         │ /
         │/__ Chunk: "quantum algorithms"
         │\
         │ \
         │  ← Large angle = Low similarity
         │   \
         │____\_ Chunk: "cooking recipes"
```

### ChromaDB Integration

#### What is ChromaDB?

A vector database optimized for similarity search.

```python
# Traditional SQL Database
"SELECT * FROM articles WHERE title = 'quantum computing'"
→ Exact match only

# Vector Database (ChromaDB)
"SELECT * FROM articles ORDER BY similarity('quantum computing') LIMIT 10"
→ Returns top 10 semantically similar chunks
```

#### Code Walkthrough

```python
def retrieve_embeddings(self, web_contents, links, query):
    # Step 1: Split text into chunks
    all_documents = []
    all_metadatas = []

    for i, content in enumerate(web_contents):
        if not content:
            continue

        # Split into chunks
        text_splitter = RecursiveTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(content)

        # Store with metadata
        for chunk in chunks:
            all_documents.append(chunk)
            all_metadatas.append({
                "source": links[i],
                "chunk_index": len(all_documents)
            })

    logger.info(f"Created {len(all_documents)} text chunks")
```

**Metadata Purpose:**
- Track which URL each chunk came from
- Provide citations in final answer
- Enable source verification

```python
    # Step 2: Create ChromaDB collection
    import chromadb
    client = chromadb.Client()

    collection = client.create_collection(
        name=f"search_results_{int(time.time())}",
        embedding_function=self.embedding_function,
        metadata={"description": "Web search results"}
    )
```

**Collection = Database Table**
- Each search creates a new collection
- Temporary (in-memory only)
- Auto-generates embeddings when adding documents

```python
    # Step 3: Add documents to collection
    collection.add(
        documents=all_documents,
        metadatas=all_metadatas,
        ids=[f"doc_{i}" for i in range(len(all_documents))]
    )
```

**What Happens Internally:**
1. ChromaDB calls embedding_function on each document
2. Stores: (text, vector, metadata, id) tuple
3. Builds HNSW index for fast similarity search

```python
    # Step 4: Query collection
    results = collection.query(
        query_texts=[query],
        n_results=min(len(all_documents), 10)  # Top 10
    )

    # Extract relevant documents
    relevant_docs = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        relevant_docs.append({
            'content': doc,
            'source': metadata['source']
        })

    return relevant_docs
```

**Query Result:**

```python
[
    {
        'content': "Quantum entanglement occurs when particles...",
        'source': "https://physics.org/quantum-mechanics"
    },
    {
        'content': "Einstein's spooky action at a distance refers...",
        'source': "https://wikipedia.org/Quantum_entanglement"
    },
    # ...8 more chunks
]
```

### Retrieval Quality

#### Factors Affecting Quality

1. **Chunk Size**
   - Too small (50 chars): Lacks context
   - Too large (2000 chars): Too generic
   - Sweet spot: 500 chars

2. **Overlap**
   - No overlap: Context breaks at boundaries
   - Too much overlap (200 chars): Redundancy
   - Sweet spot: 50 chars (10%)

3. **Number of Results (K)**
   - Too few (K=3): Might miss important info
   - Too many (K=50): Adds noise, exceeds context window
   - Sweet spot: K=10

4. **Embedding Model Quality**
   - Better model = better semantic understanding
   - BGE-M3 outperforms many models on benchmarks
   - OpenAI's embedding-3-small is very good

#### Example Retrieval

**Query:** "How do neural networks learn?"

**Chunk Ranking:**

```
Rank 1 (Similarity: 0.92):
"Neural networks learn through backpropagation, adjusting weights
based on error gradients. Each layer contributes to the final
output, and errors propagate backward to update parameters."
Source: https://deeplearning.ai/how-networks-learn

Rank 2 (Similarity: 0.88):
"The learning process involves forward pass (prediction) and
backward pass (error correction). Gradient descent optimizes
the loss function by iteratively updating weights."
Source: https://machinelearning.org/neural-nets-101

Rank 3 (Similarity: 0.85):
"Training neural networks requires labeled data. The model makes
predictions, compares them to true labels, and adjusts its internal
parameters to reduce error."
Source: https://pytorch.org/tutorials/beginner
```

**Why This Ranking?**
- All three directly answer "how do neural networks learn"
- Rank 1 mentions "backpropagation" (key term)
- Rank 2 mentions "learning process" explicitly
- Rank 3 mentions "training" (related concept)

---

## Prompt Engineering for RAG

### Context Formatting

The retrieved chunks are formatted into a structured prompt:

```python
def _format_reference(self, relevant_docs_list, links):
    formatted_docs = []

    for i, doc in enumerate(relevant_docs_list):
        source_url = doc.get('source', 'Unknown')
        content = doc.get('content', '')

        # Format: [index] URL\nContent
        formatted_docs.append(
            f"[{i+1}] {source_url}\n{content}"
        )

    return "\n\n".join(formatted_docs)
```

**Output:**

```
[1] https://physics.org/quantum-mechanics
Quantum entanglement occurs when particles become interconnected
such that the quantum state of one cannot be described independently...

[2] https://wikipedia.org/Quantum_entanglement
Einstein famously referred to entanglement as "spooky action at a
distance" because measurements on one particle instantly affect...

[3] https://ibm.com/quantum-computing
Modern quantum computers leverage entanglement to achieve quantum
parallelism, allowing them to process multiple states simultaneously...
```

### Complete Prompt Template

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

**Filled Example:**

```
Web search result:
[1] https://physics.org/quantum-mechanics
Quantum entanglement occurs when...

[2] https://wikipedia.org/Quantum_entanglement
Einstein famously referred to...

Instructions: You are a conscientious researcher.
Using the provided web search results, write a comprehensive and detailed
reply to the given query.

[Citation instructions...]

Answer in language: en-us
Query: What is quantum entanglement?
Output Format:
```

### Why This Prompt Works

1. **Clear Context**: "Web search result:" signals external information
2. **Role Assignment**: "You are a researcher" sets tone
3. **Explicit Instructions**: "cite using [number]" ensures citations
4. **Format Example**: Shows exactly how to structure answer
5. **Language Specification**: Ensures answer in user's language
6. **Query Repetition**: Reminds LLM of the question

---

## RAG vs Fine-Tuning

### When to Use RAG

✅ **Use RAG when:**
- Information changes frequently (news, stock prices)
- Need to cite sources
- Working with private/proprietary data
- Want to control what LLM can access

### When to Use Fine-Tuning

✅ **Use Fine-Tuning when:**
- Adapting writing style/tone
- Teaching domain-specific patterns
- Information is static
- No need for citations

### Comparison Table

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Data Updates** | Real-time | Requires retraining |
| **Cost** | Low (just API calls) | High (GPU training) |
| **Transparency** | High (see sources) | Low (black box) |
| **Accuracy** | High (grounded in facts) | Variable |
| **Latency** | Medium (retrieval + generation) | Low (just generation) |
| **Use Case** | Search, Q&A, research | Chatbots, style adaptation |

---

## Advanced RAG Techniques

### 1. Hybrid Search

Combine vector search with keyword search:

```python
# Vector search: semantic similarity
vector_results = collection.query(query, n_results=20)

# Keyword search: exact matches
keyword_results = [
    doc for doc in all_documents
    if query.lower() in doc.lower()
]

# Merge results (de-duplicate)
combined_results = merge_and_rank(vector_results, keyword_results)
```

### 2. Re-ranking

Use a cross-encoder to re-rank retrieved results:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Score each (query, chunk) pair
scores = model.predict([
    (query, chunk) for chunk in retrieved_chunks
])

# Re-rank by cross-encoder scores
reranked_chunks = [chunk for _, chunk in sorted(
    zip(scores, retrieved_chunks),
    reverse=True
)]
```

### 3. Query Expansion

Generate multiple variations of the query:

```python
# Original query
query = "neural network training"

# Expanded queries
expanded = [
    "neural network training",
    "how to train neural networks",
    "backpropagation algorithm",
    "deep learning optimization"
]

# Retrieve for each query
all_results = []
for q in expanded:
    results = collection.query(q, n_results=5)
    all_results.extend(results)

# De-duplicate and rank
final_results = deduplicate(all_results)
```

---

## Evaluation Metrics

### How to Measure RAG Quality

1. **Retrieval Accuracy**
   ```python
   # Did we retrieve relevant documents?
   precision_at_k = relevant_in_top_k / k
   recall = relevant_in_top_k / total_relevant
   ```

2. **Answer Quality**
   - **Faithfulness**: Does answer accurately reflect sources?
   - **Relevance**: Does answer address the query?
   - **Completeness**: Are all aspects covered?

3. **Citation Quality**
   - **Accuracy**: Are citations correct?
   - **Coverage**: Is each claim cited?

### Example Evaluation

```python
Query: "What is photosynthesis?"

Retrieved Chunks:
✓ "Photosynthesis converts light to chemical energy..." (Relevant)
✓ "Chloroplasts are the site of photosynthesis..." (Relevant)
✗ "Plant cells have cell walls made of cellulose..." (Not relevant)

Precision@3 = 2/3 = 67%

Generated Answer:
"Photosynthesis is the process by which plants convert light energy
into chemical energy [1]. This occurs in chloroplasts, specialized
organelles in plant cells [2]."

Citation Accuracy: 100% (both claims cited correctly)
Faithfulness: 100% (no hallucinations)
Relevance: 100% (directly answers query)
```

---

## Next Steps

Now that you understand RAG, proceed to:
- **03_EMBEDDINGS.md** for deep dive into vector representations
- **04_LLM_INTEGRATION.md** to learn about the generation phase

You should now be able to:
1. Explain what RAG is and why it's useful
2. Understand text chunking strategies
3. Describe how vector similarity works
4. Implement basic retrieval with ChromaDB
5. Design effective RAG prompts
