# Multimodal Inputs & Context Budgeting

EmbeddingGemma 2 maps text, code, images, video, and audio into a single **8,192-token context window**. All modalities share this budget within a single call.

---

## 1. Context Consumption Rates & Maximum Limits

When feeding inputs into the model, each modality consumes tokens from the 8,192-token budget at fixed rates:

| Modality | Default Token Cost | Maximum Single-Modality Capacity | Technical Specifications & Formatting |
| :--- | :--- | :--- | :--- |
| **Text & Code** | **1 token** per subword | **8,192 tokens** | SentencePiece 262k BPE tokenizer |
| **Image** | **280 tokens** per image *(default)* | **~29 images** | File paths, URLs, or in-memory PIL images |
| **Video** | **140 tokens** per frame *(default)* | **~58 frames** *(~58s at 1 FPS)* | MP4 files, sampled at 1 frame per second by default |
| **Audio** | **25 tokens** per second | **~327 seconds** *(~5.45 mins)* | Raw audio files, **16 kHz mono** format |

> **Note:** The maximum capacities above assume an isolated single modality with no accompanying text. In real-world multimodal applications, inputs draw from the shared budget:
> $$\text{Total Tokens} = N_{\text{text}} + (N_{\text{images}} \times \text{Tokens}_{\text{image}}) + (N_{\text{frames}} \times \text{Tokens}_{\text{frame}}) + (\text{Seconds}_{\text{audio}} \times 25) \le 8,192$$

---

## 2. Vision Token Budget (70 to 1,120 Tokens)

A major capability in EmbeddingGemma 2 is the **configurable vision token budget**. Users can override the default sequence length (280 tokens) to represent images and video frames with **soft token counts ranging from 70 to 1,120 tokens**:

```
70 tokens ◄──────────────────── 280 tokens ────────────────────► 1,120 tokens
(Max throughput: ~114 images)   (Default balance)                (Max fidelity: ~7 images)
```

### Vision Budget Comparison Matrix

| Budget Setting | Soft Tokens / Image | Capacity in 8k Window | Ideal Application Scenarios |
| :--- | :--- | :--- | :--- |
| **High Compression** | **70 tokens** | **~114 images / frames** | Rapid video skimming across long clips, large photo album clustering, high-throughput thumbnail classification |
| **Balanced (Default)** | **280 tokens** | **~29 images / frames** | General eCommerce product search, web pages with 1–5 images, balanced multimodal RAG |
| **High Fidelity** | **560 tokens** | **~14 images / frames** | Technical charts, slide decks, infographics, medium-density diagrams |
| **Ultra Fidelity** | **1,120 tokens** | **~7 images / frames** | **Visual Document Retrieval (ViDoRe)**, scanned invoices, patent schematics, screenshots with small text |

### Why scale the vision budget?
* **Fine-Grained Visual Extraction:** Higher sequence lengths provide more expressive spatial representations for the transformer backbone to process, capturing small text labels, fine line graphs, and complex multi-object scenes.
* **Throughput Optimization:** Lower sequence lengths drastically reduce transformer self-attention computations and memory allocation, allowing up to **~114 frames** to fit into a single forward pass.

---

## 3. Audio & Video Sampling Protocols

### Audio Specifications
* **Sample Rate:** **16 kHz**
* **Channel Count:** **Mono** (1 channel)
* **Token Rate:** Exactly **25 tokens per second** (1 token represents 40ms of audio)
* **Maximum Duration:** ~327 seconds (~5.45 minutes)

```python
# Pass audio files directly as paths or URLs
audio_emb = model.encode({"audio": "meeting_recording_16k.wav"})
```

### Video Specifications
* **Container:** MP4 / standard video container
* **Sampling Rate:** Default is **1 frame per second (1 FPS)** (configurable in pipeline)
* **Token Rate:** Default **140 tokens per frame**
* **Interleaving:** Frames are projected sequentially through the vision encoder and inserted at `<|video|>` markers.

---

## 4. Interleaving Modalities in a Single Input

You can combine text, images, video, and audio into a single composite embedding by using vocabulary placeholder tokens:
* `<|image|>`: Inserts image soft tokens
* `<|video|>`: Inserts video frame soft tokens
* `<|audio|>`: Inserts audio spectrogram tokens

### Python Example: Composite eCommerce Listing

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

# Create a rich interleaved document
composite_item = {
    "text": "Ultra-lightweight trail running shoes. <|image|> Breathable mesh upper detail: <|image|> Grip stress test on wet slate: <|video|>",
    "image": ["shoe_side.jpg", "mesh_macro.jpg"],
    "video": "traction_test.mp4",
}

# Generates a single 768d embedding capturing all text, photos, and video
listing_embedding = model.encode(composite_item)

# Direct match against natural language query
query_emb = model.encode("waterproof trail shoes with wet grip", prompt_name="SearchQuery")

similarity = model.similarity(query_emb, listing_embedding)
print(f"Product Match Score: {similarity.item():.4f}")
```

### Python Example: Video Moments Finder

```python
# Ingest an instructional video snippet with audio narration
video_clip = {
    "text": "Step 3: Solder the capacitor to ground. <|video|> <|audio|>",
    "video": "soldering_clip.mp4",
    "audio": "narration_audio.wav",
}

clip_embedding = model.encode(video_clip)

# Query by what is shown, what is said, or what is described
query_emb = model.encode("soldering capacitor to ground pin", prompt_name="SearchQuery")
print("Moment Match:", model.similarity(query_emb, clip_embedding).item())
```
