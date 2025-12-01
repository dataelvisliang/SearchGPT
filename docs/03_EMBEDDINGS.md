# Deep Dive: Vector Embeddings

## What Are Embeddings?

Embeddings are **numerical representations of text** that capture semantic meaning. Think of them as coordinates in a high-dimensional space where similar meanings cluster together.

### Simple Analogy

Imagine organizing books in a library:

**Traditional Method (Keywords):**
```
"Python Programming" → Shelf: P
"Machine Learning"   → Shelf: M
"Deep Learning"      → Shelf: D
```
❌ Related books are scattered across different shelves!

**Embedding Method (Semantic Space):**
```
"Python Programming" → [0.2, 0.8, 0.1, ...]
"Machine Learning"   → [0.3, 0.7, 0.2, ...]  ← Close to Python
"Deep Learning"      → [0.4, 0.6, 0.3, ...]  ← Even closer to ML
"Cooking Recipes"    → [-0.8, 0.1, 0.9, ...] ← Far from tech books
```
✓ Similar books naturally cluster together!

## How Embeddings Work

### From Text to Numbers

```python
# Input: Text
text = "Quantum computing uses qubits"

# Step 1: Tokenization (break into words/subwords)
tokens = ["Quantum", "computing", "uses", "qubits"]

# Step 2: Neural network processing
# (This happens inside the embedding model)
hidden_states = neural_network(tokens)

# Step 3: Output vector (embedding)
embedding = [0.23, -0.15, 0.87, 0.42, -0.56, ...]  # 384 numbers
```

### Why 384 Dimensions?

Different embedding models use different dimensions:
- **BERT-base**: 768 dimensions
- **text-embedding-3-small**: 1536 dimensions (can be reduced to 384)
- **BGE-M3**: 1024 dimensions
- **Sentence-BERT**: 384 dimensions

**Trade-offs:**
- **More dimensions**: Captures more nuance, requires more storage/computation
- **Fewer dimensions**: Faster, more efficient, still captures main concepts

In Relevance Search, we use **384 dimensions** as a good balance.

## Embedding Models in Relevance Search

### Option 1: OpenRouter (text-embedding-3-small)

```python
class OpenRouterEmbeddings:
    def __init__(self, api_key, model='openai/text-embedding-3-small'):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://openrouter.ai/api/v1"

    def embed_documents(self, texts: list) -> list:
        """Embed a list of text chunks"""
        embeddings = []

        for text in texts:
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": text
                }
            )

            result = response.json()
            embedding = result['data'][0]['embedding']
            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> list:
        """Embed a single query"""
        return self.embed_documents([text])[0]
```

**Characteristics:**
- ✓ High quality (OpenAI's model)
- ✓ English-optimized
- ✓ Fast API
- ✗ Costs money (small amount per 1M tokens)
- ✗ Less effective for Chinese text

### Option 2: Gitee AI (BGE-M3)

```python
class GiteeEmbeddings:
    def __init__(self, api_key, model='bge-m3'):
        self.api_key = api_key
        self.model = model
        self.api_base = "https://ai.gitee.com/api/inference/serverless"

    def embed_documents(self, texts: list) -> list:
        """Embed documents using Gitee BGE-M3"""
        embeddings = []

        for text in texts:
            response = requests.post(
                f"{self.api_base}/{self.model}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={"input": text}
            )

            result = response.json()
            embedding = result['data'][0]['embedding']
            embeddings.append(embedding)

        return embeddings
```

**Characteristics:**
- ✓ Bilingual (Chinese + English)
- ✓ Excellent for technical terms
- ✓ Free tier available
- ✓ SOTA (State of the Art) on multilingual benchmarks
- ✗ Slightly slower than OpenRouter

### Model Selection Logic

```python
# In retrieval.py
use_gitee = (
    self.config.get("gitee_api_key") and
    self.config.get("gitee_api_key") != ""
)

if use_gitee:
    logger.info("Using Gitee AI BGE-M3 embeddings")
    embeddings = GiteeEmbeddings(
        api_key=self.config["gitee_api_key"],
        model=self.config.get("embedding_model", "bge-m3")
    )
else:
    logger.info("Using OpenRouter embeddings")
    embeddings = OpenRouterEmbeddings(
        api_key=self.config["openrouter_api_key"],
        model='openai/text-embedding-3-small'
    )
```

**Decision Tree:**
```
Check config.yaml
    │
    ├─ Gitee API key provided? ──YES──> Use BGE-M3
    │
    └─ NO ──> Use OpenRouter text-embedding-3-small
```

## Understanding Vector Similarity

### Cosine Similarity

The most common similarity metric for embeddings.

**Formula:**
```
similarity = (A · B) / (||A|| × ||B||)

Where:
  A · B = dot product
  ||A|| = magnitude (length) of vector A
  ||B|| = magnitude (length) of vector B
```

**Code Implementation:**

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    # Dot product
    dot_product = np.dot(vec_a, vec_b)

    # Magnitudes
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)

    # Cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)

    return similarity

# Example
query_vec = np.array([0.5, 0.8, 0.3])
doc1_vec = np.array([0.6, 0.7, 0.4])   # Similar
doc2_vec = np.array([-0.5, -0.8, 0.1]) # Opposite

sim1 = cosine_similarity(query_vec, doc1_vec)  # 0.989 (very similar)
sim2 = cosine_similarity(query_vec, doc2_vec)  # -0.623 (dissimilar)
```

**Range:**
- **+1.0**: Vectors point in exactly the same direction (identical meaning)
- **0.0**: Vectors are perpendicular (unrelated)
- **-1.0**: Vectors point in opposite directions (opposite meaning)

### Visual Example (2D Simplified)

```
        ^
        │
    Q   │   D1       Q = Query: "machine learning"
     \  │  /         D1 = "neural networks"
      \ │ /          D2 = "cooking recipes"
       \│/
────────●──────>
       /│\
      / │ \
     /  │  D2
        │
```

**Angles:**
- Q to D1: Small angle → High similarity (0.95)
- Q to D2: Large angle → Low similarity (0.12)

### Real Example from Relevance Search

```python
Query: "What is quantum entanglement?"
Query Embedding: [0.23, -0.15, 0.87, 0.42, ..., -0.31] (384 dims)

Chunk 1: "Quantum entanglement is a phenomenon where particles..."
Embedding: [0.25, -0.13, 0.85, 0.40, ..., -0.29]
Similarity: 0.96 ✓ TOP MATCH

Chunk 2: "Einstein called it spooky action at a distance..."
Embedding: [0.21, -0.18, 0.82, 0.38, ..., -0.34]
Similarity: 0.93 ✓ RELEVANT

Chunk 3: "Python is a programming language for data science..."
Embedding: [-0.42, 0.67, -0.21, 0.84, ..., 0.52]
Similarity: 0.15 ✗ NOT RELEVANT
```

## Embedding Quality Factors

### 1. Training Data

Embeddings "learn" meaning from training data.

**Example:**
```python
# Model trained on scientific papers
"quantum" → Close to "physics", "particle", "wave"

# Model trained on social media
"quantum" → Close to "leap", "change", "dramatic"
```

**BGE-M3** is trained on:
- Academic papers
- Wikipedia
- Technical documentation
- Multilingual web content

### 2. Context Window

How much text the model sees at once.

```python
# Short context (BERT-base: 512 tokens)
"The bank..."
↓
Could mean: Financial bank? River bank?

# Longer context (BGE-M3: 8192 tokens)
"The bank announced new interest rates..."
↓
Clearly: Financial bank!
```

### 3. Normalization

Embeddings are usually normalized to unit length.

```python
def normalize_embedding(embedding):
    """Normalize to unit length"""
    magnitude = np.linalg.norm(embedding)
    return embedding / magnitude

# Before normalization
vec = [3, 4]  # Length = 5

# After normalization
normalized = [0.6, 0.8]  # Length = 1
```

**Why normalize?**
- Makes cosine similarity equivalent to dot product
- Speeds up computation
- Prevents magnitude from affecting similarity

## Visualizing High-Dimensional Embeddings

### Dimensionality Reduction: t-SNE

We can't visualize 384 dimensions, but we can reduce to 2D:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get embeddings for multiple documents
texts = [
    "quantum computing",
    "quantum mechanics",
    "machine learning",
    "deep learning",
    "pizza recipe",
    "pasta recipe"
]

embeddings = [embed_text(text) for text in texts]  # Each: 384 dims

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, text in enumerate(texts):
    plt.annotate(text, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title("Embedding Space Visualization")
plt.show()
```

**Expected Result:**
```
        quantum mechanics
            ●
              ↘
                quantum computing
                    ●


    machine learning
        ●   ↘
              deep learning
                  ●


                            pizza recipe
                                ●   ↘
                                      pasta recipe
                                          ●
```

Physics terms cluster together, ML terms cluster together, recipes cluster together!

## Embedding Benchmarks

### MTEB (Massive Text Embedding Benchmark)

Standardized benchmark for embedding quality:

| Model | Retrieval | Clustering | Classification |
|-------|-----------|------------|----------------|
| **BGE-M3** | 54.3 | 48.2 | 71.4 |
| **text-embedding-3-small** | 52.1 | 46.8 | 69.2 |
| **BERT-base** | 41.2 | 39.5 | 62.7 |

**Higher is better**

### C-MTEB (Chinese Benchmark)

For Chinese language tasks:

| Model | Retrieval | STS (Semantic Similarity) |
|-------|-----------|---------------------------|
| **BGE-M3** | 66.1 | 67.9 |
| **text-embedding-3-small** | 58.3 | 61.2 |
| **m3e-base** | 63.5 | 64.8 |

BGE-M3 excels at Chinese! ✓

## Advanced Topics

### 1. Batch Processing

Embedding many documents at once is more efficient:

```python
# Inefficient: One at a time
embeddings = []
for text in texts:
    emb = embed_single(text)  # 100ms per call
    embeddings.append(emb)
# Total: 100 texts × 100ms = 10 seconds

# Efficient: Batch processing
embeddings = embed_batch(texts)  # 1 second for all 100
# Total: 1 second (10× faster!)
```

**Implementation in Relevance Search:**

```python
def embed_documents(self, texts: list) -> list:
    """Batch embed multiple documents"""
    # Process in batches of 100
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Single API call for entire batch
        response = requests.post(
            f"{self.api_base}/embeddings",
            json={"input": batch, "model": self.model}
        )

        batch_embeddings = [
            item['embedding']
            for item in response.json()['data']
        ]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
```

### 2. Caching Embeddings

Embeddings are deterministic (same input → same output), so cache them!

```python
import hashlib
import pickle

class EmbeddingCache:
    def __init__(self):
        self.cache = {}

    def get_hash(self, text):
        """Get hash of text for cache key"""
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text, embed_fn):
        """Get embedding from cache or compute"""
        key = self.get_hash(text)

        if key in self.cache:
            return self.cache[key]

        # Not in cache, compute
        embedding = embed_fn(text)
        self.cache[key] = embedding

        return embedding

# Usage
cache = EmbeddingCache()
emb1 = cache.get_embedding("quantum physics", embed_function)  # Computed
emb2 = cache.get_embedding("quantum physics", embed_function)  # Cached! ⚡
```

### 3. Semantic Similarity vs Lexical Similarity

**Lexical** (keyword matching):
```python
query = "automobile accident"
doc1 = "car crash"  # ✗ No matching words!
doc2 = "automobile manufacturing"  # ✓ Matches "automobile"

# Lexical matching ranks doc2 higher (wrong!)
```

**Semantic** (embeddings):
```python
query_emb = embed("automobile accident")
doc1_emb = embed("car crash")        # High similarity (0.89)
doc2_emb = embed("automobile manufacturing")  # Low similarity (0.32)

# Semantic matching ranks doc1 higher (correct!) ✓
```

### 4. Cross-lingual Embeddings

BGE-M3 is **multilingual** - similar concepts in different languages have similar embeddings:

```python
# English
en_emb = embed("machine learning")

# Chinese
zh_emb = embed("机器学习")

# Similarity
similarity(en_emb, zh_emb)  # 0.92 (very high!)
```

**Use Case:**
```
Query (English): "quantum computing"
    ↓
Can find relevant Chinese documents:
    - "量子计算简介" (Introduction to quantum computing)
    - "量子比特原理" (Principles of qubits)
```

## Practical Tips

### 1. Choosing Chunk Size

Embeddings work best with coherent chunks:

```python
# Too small (10 words)
"Quantum entanglement is a phenomenon where"
# ❌ Incomplete thought, lacks context

# Too large (5000 words)
"[Entire encyclopedia article on quantum physics]"
# ❌ Too generic, not focused

# Just right (100-150 words)
"Quantum entanglement is a phenomenon where two particles
become correlated such that the quantum state of one cannot
be described independently of the other..."
# ✓ Complete idea, good context
```

**Relevance Search uses 500 characters (~100 words)**

### 2. Handling Special Characters

```python
# Before embedding, normalize text
def normalize_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters (optional)
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    # Lowercase (optional, model-dependent)
    # text = text.lower()  # BGE-M3 is case-sensitive!

    return text.strip()

# Usage
raw = "Quantum   computing\n\n\t uses  qubits!!!"
clean = normalize_text(raw)
# "Quantum computing uses qubits!!!"

embedding = embed(clean)
```

### 3. Query Enhancement

Improve retrieval by expanding queries:

```python
# Original query
query = "ML algorithms"

# Enhanced query (more context)
enhanced = "machine learning algorithms including neural networks"

# Get embedding
embedding = embed(enhanced)  # Better retrieval!
```

## Common Issues and Solutions

### Issue 1: Poor Retrieval Quality

**Symptoms:**
- Irrelevant documents ranked high
- Relevant documents missing

**Solutions:**

1. **Try a better embedding model:**
```python
# Upgrade to BGE-M3
embeddings = GiteeEmbeddings(api_key, model='bge-m3')
```

2. **Increase chunk overlap:**
```python
# More context preservation
text_splitter = RecursiveTextSplitter(
    chunk_size=500,
    chunk_overlap=100  # Increased from 50
)
```

3. **Retrieve more candidates (higher K):**
```python
# Get top 20 instead of top 10
collection.query(query_embedding, n_results=20)
```

### Issue 2: Slow Embedding Generation

**Symptoms:**
- Long wait times
- Timeouts

**Solutions:**

1. **Use batch processing:**
```python
# Don't do this (slow)
for text in texts:
    embed_single(text)

# Do this (fast)
embed_batch(texts)
```

2. **Cache embeddings:**
```python
# Store embeddings in database for reuse
# See "Caching Embeddings" section above
```

3. **Use faster model:**
```python
# OpenRouter is generally faster than BGE-M3
embeddings = OpenRouterEmbeddings(api_key)
```

### Issue 3: Multilingual Queries

**Symptoms:**
- English query doesn't find Chinese documents
- Mixed language results are poor

**Solutions:**

1. **Use multilingual model (BGE-M3):**
```python
embeddings = GiteeEmbeddings(api_key, model='bge-m3')
```

2. **Translate query to target language:**
```python
from deep_translator import GoogleTranslator

query_en = "quantum computing"
query_zh = GoogleTranslator(source='en', target='zh-CN').translate(query_en)

# Search with both
results_en = search(embed(query_en))
results_zh = search(embed(query_zh))
combined = merge(results_en, results_zh)
```

## Evaluation and Testing

### Test Embedding Quality

```python
def test_similarity():
    """Test that similar concepts have high similarity"""

    # Similar pairs (should be > 0.7)
    assert similarity(
        embed("car"),
        embed("automobile")
    ) > 0.7

    assert similarity(
        embed("happy"),
        embed("joyful")
    ) > 0.7

    # Dissimilar pairs (should be < 0.3)
    assert similarity(
        embed("car"),
        embed("pizza")
    ) < 0.3

    print("✓ All tests passed!")

test_similarity()
```

### Benchmark Your System

```python
def benchmark_retrieval(test_cases):
    """Measure retrieval accuracy"""

    total = len(test_cases)
    correct = 0

    for query, expected_doc in test_cases:
        results = retrieve(query, k=10)

        if expected_doc in results:
            correct += 1

    accuracy = correct / total
    print(f"Retrieval Accuracy: {accuracy:.2%}")

    return accuracy

# Test cases: (query, expected relevant document)
test_cases = [
    ("what is python", "Python is a programming language..."),
    ("machine learning", "ML is a subset of AI..."),
    # ... more test cases
]

benchmark_retrieval(test_cases)
```

## Summary

### Key Takeaways

1. **Embeddings** convert text to numbers that capture meaning
2. **384 dimensions** is a good balance of quality and efficiency
3. **Cosine similarity** measures how similar two embeddings are
4. **BGE-M3** excels at multilingual and technical content
5. **OpenRouter** is fast and high-quality for English
6. **Batch processing** and **caching** improve performance

### Next Steps

- **04_LLM_INTEGRATION.md**: Learn how to use embeddings with LLMs
- **05_STREAMLIT_UI.md**: Build interfaces for embedding-based search
- **06_ADVANCED_TOPICS.md**: Production optimization techniques

You should now understand:
- ✓ What embeddings are and why they work
- ✓ How to choose between embedding models
- ✓ Vector similarity mathematics
- ✓ Practical optimization techniques
- ✓ Common issues and solutions
