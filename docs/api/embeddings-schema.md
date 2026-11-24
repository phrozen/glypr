# `glypr` Embedding API Reference

The `glypr serve` engine exposes an OpenAI-compatible HTTP server (`POST /v1/embeddings`) with native extensions for Matryoshka dimension truncation, vector quantizations, asymmetric task conditioning, and multimodal (image/audio) inputs.

---

## 1. Request Schema (`POST /v1/embeddings`)

### Endpoint
```http
POST /v1/embeddings
Content-Type: application/json
```

### JSON Schema (Request Body)
```json
{
  "input": "string | object | array",
  "model": "string (optional, default: 'embeddinggemma-2')",
  "dimensions": "integer (optional, default: 768, values: 1..=768; multiple of 64 for 'binary')",
  "encoding_format": "string (optional, default: 'float', values: 'float' | 'int' | 'binary' | 'base64')",
  "task": "string (optional, default: 'Raw')",
  "title": "string (optional, used when task='Document')",
  "user": "string (optional)"
}
```

### Input Field Formats (`input`)

The `input` field accepts (up to `max_batch_items`, default `256`, within the default `32 MiB` body limit):
1. **Single string:** `"Search query text"`
2. **Batch array of strings:** `["First text", "Second text"]`
3. **Single multimodal object:** `{"text": "...", "image": "...", "budget": 280}`
4. **Batch array of multimodal objects / mixed items:** `["Text item", {"image": "..."}]`

#### Multimodal Object Structure
```json
{
  "text": "Optional text or query prompt",
  "image": "Base64 data URI | raw base64 string | local file path (requires --allow-local-files)",
  "audio": "Base64 data URI (WAV) | raw base64 WAV | local file path (requires --allow-local-files)",
  "budget": 280
}
```
> [!NOTE]
> Local filesystem paths are disabled over HTTP by default (`allow_local_files = false`). Pass a `data:...;base64,...` URI or raw base64 string, or explicitly start the server with `glypr serve --allow-local-files`. A single multimodal object may specify `image` or `audio`, not both.

---

## 2. Response Schema

### Standard JSON Response
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0312, -0.0154, 0.0891, "..."]
    }
  ],
  "model": "embeddinggemma-2",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

### Response Formats by `encoding_format`

| `encoding_format` | `embedding` Payload Type | Size (768d) | Compression | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`float`** *(default)* | `Array<number>` | 3,072 B | $1\times$ | Standard IEEE-754 32-bit floats. |
| **`int`** / **`int8`** | `Array<number>` | 768 B | $4\times$ | Signed 8-bit integers in $[-127, 127]$ (zero metadata). |
| **`binary`** | `string` (Base64) | 96 B | $32\times$ | 1-bit packed bitfield (96 bytes = 768 bits) for hardware POPCNT. |
| **`base64`** | `string` (Base64) | 3,072 B | $1\times$ | Raw little-endian float32 bytes encoded in Base64 (OpenAI spec). |

---

## 3. Supported Task Types (`task`)

Conditioning embeddings on a specific task applies official prompt prefixes to optimize the vector space for asymmetric retrieval:

| Task Name (`task`) | Applied Prefix Format | Recommended Use Case |
| :--- | :--- | :--- |
| **`Raw`** *(default)* | *(No prefix applied)* | Symmetric similarity, sentence parity, raw text. |
| **`SearchQuery`** | `task: search result \| query: {text}` | Queries submitted by search users. |
| **`Document`** | `title: {title} \| text: {text}` | Documents or passages stored in your index. |
| **`CodeRetrieval`** | `task: code retrieval \| query: {text}` | Natural language queries searching for code snippets. |
| **`QuestionAnswering`**| `task: question answering \| query: {text}` | Questions seeking answer passages. |
| **`FactChecking`** | `task: fact checking \| query: {text}` | Claims seeking verification evidence. |
| **`Classification`** | `task: classification \| query: {text}` | Zero-shot intent or category classification. |
| **`Clustering`** | `task: clustering \| query: {text}` | Unsupervised topic clustering. |
| **`SentenceSimilarity`**| `task: sentence similarity \| query: {text}` | Paraphrase detection and semantic similarity (STS). |
| **Custom** | `task: {custom} \| query: {text}` | Any custom domain prefix string. |

> [!NOTE]
> When `task="Document"` is passed, provide the optional `"title"` string. If omitted, `glypr` automatically formats it as `title: none | text: {text}`.

---

## 4. Supported Image Formats & Soft Token Budgets

### Image Input Types
The `image` field accepts:
1. **Base64 Data URI:** `data:image/png;base64,...`, `data:image/jpeg;base64,...`, `data:image/webp;base64,...`
2. **Raw Base64 string:** Raw base64-encoded image file bytes.
3. **Local File Path** *(opt-in only)*: Absolute or relative path on the server host when started with `--allow-local-files`.

### Container Formats Supported
* **PNG** (`.png`)
* **JPEG / JPG** (`.jpeg`, `.jpg`)
* **WebP** (`.webp`)

### Supported Audio Inputs
* **WAV** (`.wav`) via Base64 data URI, raw Base64, or opt-in local path (`8/16/24/32-bit` PCM or `32-bit` IEEE float, mono or stereo; automatically downmixed and FIR low-pass resampled to 16 kHz mono).

### Vision Soft Token Budget (`budget`)
Controls the spatial patch downsampling resolution. Supported range: `16` to `2048` tokens (defaults to `280`).

| Budget | Downsampled Spatial Grid | Recommended Use Case |
| :--- | :--- | :--- |
| **`70`** | Compact grid | Low-latency thumbnails, high-throughput pipelines. |
| **`140`** | $10 \times 14$ grid | Native trading cards ($480 \times 680$ px), sequential 1 FPS video frames. |
| **`280`** *(default)*| $14 \times 20$ grid | General photos, web illustrations, high detail. |
| **`560`** | High density | Small text detection, complex UI screenshots, documents. |
| **`1120`** | Ultra density | Extreme micro-feature analysis. |

---

## 5. Ready-to-Copy Example Payloads

### Example 1: Basic Text Search Query
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "How does gravitational lensing work?",
    "task": "SearchQuery",
    "dimensions": 256
  }'
```

### Example 2: Indexed Document with Title
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Gravitational lensing is the bending of light around massive celestial bodies...",
    "task": "Document",
    "title": "General Relativity Primer",
    "dimensions": 256,
    "encoding_format": "int"
  }'
```

### Example 3: Multimodal Image Embedding (Base64 JPEG)
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "text": "Card artwork depiction:",
      "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "budget": 140
    },
    "dimensions": 768,
    "encoding_format": "float"
  }'
```

### Example 4: Batch Query with Binary Quantization
```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "query vector 1",
      "query vector 2"
    ],
    "encoding_format": "binary"
  }'
```

---

## 6. Model & Health Introspection Endpoints

### A. Model Metadata (`GET /v1/models`)
Exposes active model properties, context boundaries, hardware device, and Apple Neural Engine (ANE) configuration:

```bash
curl http://127.0.0.1:8080/v1/models
```

#### Response Example
```json
{
  "object": "list",
  "data": [
    {
      "id": "embeddinggemma-2",
      "object": "model",
      "created": 1729123200,
      "owned_by": "glypr",
      "device": "metal",
      "vision_backend": "hybrid (ane + metal)",
      "text_backend": "hybrid (ane + metal)",
      "ane_enabled": true,
      "ane_text_enabled": true,
      "ane_text_seq_len": 512,
      "has_multimodal": true,
      "has_audio_projector": true,
      "max_context": 8320,
      "embedding_dim": 768
    }
  ]
}
```

### B. Health Status (`GET /health`)
Lightweight liveness and capability probe for Kubernetes and load balancers:

```bash
curl http://127.0.0.1:8080/health
```

#### Response Example
```json
{
  "status": "ok",
  "model": "embeddinggemma-2",
  "device": "metal",
  "multimodal": true,
  "has_audio_projector": true,
  "vision_backend": "hybrid (ane + metal)",
  "text_backend": "hybrid (ane + metal)",
  "ane_enabled": true,
  "ane_supported_patches": [1134, 2394],
  "ane_text_enabled": true,
  "ane_text_seq_len": 512
}
```

### C. Readiness Probe (`GET /ready`)
Returns `200 OK` when the inference worker channel is active, or `503 Service Unavailable` during shutdown:

```json
{
  "status": "ready",
  "model": "embeddinggemma-2"
}
```

### D. Error Response Envelope (`400` / `500` / `504`)
Invalid parameters, malformed JSON (`OpenAiJson`), disallowed local file paths, binary non-multiples of 64, empty/oversized batches, and missing projectors return `400 Bad Request` in standard OpenAI error JSON format:

```json
{
  "error": {
    "message": "Invalid dimensions: 1024. Must be between 1 and 768.",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

Server execution failures return `500` (`internal_server_error`), and requests exceeding `--request-timeout-secs` (default `120s`) return `504` (`timeout_error`).
