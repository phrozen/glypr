# Task Instruction Prompts & Prefix Catalog

EmbeddingGemma 2 relies on **task-steered representations**. During training, short instruction prefixes were prepended to text inputs to condition the model's attention states for specific downstream tasks (e.g., distinguishing a short query from an authoritative document passage).

Using the appropriate prefix yields maximal retrieval precision. While omitting prefixes works, benchmark evaluations show measurable degradation across MTEB retrieval and classification tasks.

---

## 1. Asymmetric vs. Symmetric Tasks

Embedding tasks fall into two fundamental categories:

### A. Asymmetric Tasks (e.g., Information Retrieval, QA, Code Search)
* The query and the candidate documents have different lengths, syntactic structures, and semantic roles.
* **Queries:** Formatted with a specific task instruction prefix (e.g., `task: search result | query: {query}`).
* **Documents:** Formatted with title and body notation (`title: {title} | text: {content}`).

### B. Symmetric Tasks (e.g., Sentence Similarity, Classification, Clustering)
* All items being compared or grouped are of the same nature.
* **All Inputs:** Formatted with the same identical task prefix.

---

## 2. Complete Task Instruction Catalog

| Use Case | Task Type | `prompt_name` (SentenceTransformers) | Query Task Instruction | Document Task Instruction *(use `none` if untitled)* |
| :--- | :--- | :--- | :--- | :--- |
| **Web / Document Search** | Asymmetric | `SearchQuery` | `task: search result \| query: {query}` | `title: {title} \| text: {content}` |
| **Question Answering** | Asymmetric | `QuestionAnswering` | `task: question answering \| query: {question}` | `title: {title} \| text: {passage}` |
| **Fact Checking** | Asymmetric | `FactChecking` | `task: fact checking \| query: {claim}` | `title: {title} \| text: {evidence}` |
| **Code Retrieval** | Asymmetric | `CodeRetrieval` | `task: code retrieval \| query: {query}` | `title: {filename or symbol} \| text: {code}` |
| **Text Classification** | Symmetric | `Classification` | `task: classification \| query: {content}` | *N/A (Apply query instruction to all)* |
| **Clustering** | Symmetric | `Clustering` | `task: clustering \| query: {content}` | *N/A (Apply query instruction to all)* |
| **Semantic Similarity** | Symmetric | `SentenceSimilarity` | `task: sentence similarity \| query: {content}` | *N/A (Apply query instruction to all)* |

---

## 3. Python Implementation Examples

### Example 1: Web & Document Search (`SearchQuery`)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

# 1. Query: Pass prompt_name='SearchQuery'
query = "What causes the northern lights?"
query_emb = model.encode(query, prompt_name="SearchQuery")

# 2. Document with Title: Format manually
doc_with_title = "title: Aurora Borealis Explained | text: Charged solar wind particles collide with atmospheric gases..."
doc_emb1 = model.encode(doc_with_title)

# 3. Document without Title: Use prompt_name='Document' (applies 'title: none')
raw_text = "The aurora borealis occurs near Earth's polar regions."
doc_emb2 = model.encode(raw_text, prompt_name="Document")

print("Similarity (Titled):", model.similarity(query_emb, doc_emb1).item())
print("Similarity (Untitled):", model.similarity(query_emb, doc_emb2).item())
```

---

### Example 2: Code Retrieval (`CodeRetrieval`)

EmbeddingGemma 2 achieved **78.68 on MTEB Code v1** (+9.92 points over v1). To achieve this score, pair `prompt_name="CodeRetrieval"` on the natural language query with the filepath or function header as the `title` in the document:

```python
# Query
code_query = "Read a file asynchronously with timeout"
query_vec = model.encode(code_query, prompt_name="CodeRetrieval")

# Document: Pass filepath/function name in the title field
code_snippet = """title: src/io/reader.rs | text:
pub async fn read_with_timeout(path: &Path, timeout_ms: u64) -> Result<Vec<u8>, IoError> {
    tokio::time::timeout(Duration::from_millis(timeout_ms), tokio::fs::read(path)).await?
}"""
doc_vec = model.encode(code_snippet)

print("Code Match Score:", model.similarity(query_vec, doc_vec).item())

# Native Rust (glypr-core):
let query_vec = embedder.embed_query(code_query, "CodeRetrieval", None)?;
let doc_vec = embedder.embed_document(code_body, Some("src/io/reader.rs"), None)?;
```

---

### Example 3: Semantic Similarity & Clustering (Symmetric)

```python
sentences = [
    "The stock market experienced heavy volatility today.",
    "Equities fluctuated wildly during trading hours.",
    "Quantum computers utilize superposition and entanglement.",
]

# In symmetric tasks, apply SentenceSimilarity to all items
embeddings = model.encode(sentences, prompt_name="SentenceSimilarity")

similarity_matrix = model.similarity(embeddings, embeddings)
print("Similarity between 0 and 1:", similarity_matrix[0, 1].item())  # High
print("Similarity between 0 and 2:", similarity_matrix[0, 2].item())  # Low
```

---

## 4. Critical Rules & Common Pitfalls

### ⚠️ Gotcha 1: The `prompt_name="Document"` Trap
* In `sentence-transformers`, calling `model.encode(text, prompt_name="Document")` automatically prefixes the text with `title: none | text: {text}`.
* **If your document already has an actual title or filename**, do **not** use `prompt_name="Document"`. Instead, format the string manually:
  ```python
  # CORRECT
  model.encode(f"title: {doc_title} | text: {doc_body}")

  # WRONG (creates: 'title: none | text: title: MyTitle | text: MyBody')
  model.encode(f"title: {doc_title} | text: {doc_body}", prompt_name="Document")
  ```

### ⚠️ Gotcha 2: Prefixes Are Strictly Text-Only
* Never prepend task instructions when embedding images, audio, or video files:
  ```python
  # CORRECT
  img_emb = model.encode({"image": "chart.png"})

  # WRONG (will corrupt the representation)
  img_emb = model.encode({"image": "chart.png"}, prompt_name="SearchQuery")
  ```

### ⚠️ Gotcha 3: Consistent Pairings
* Always pair `SearchQuery` with `Document` (or titled document formatting). Comparing a raw query without a prefix against a document degrades cosine rank resolution by 3–8% on retrieval benchmarks.
