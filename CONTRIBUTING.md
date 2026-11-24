# Contributing to Glypr

Thank you for your interest in contributing to Glypr! We welcome contributions, bug reports, and optimizations to make EmbeddingGemma 2 inference faster, leaner, and more accessible.

---

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for everyone, regardless of background or experience. Please treat all contributors and users with respect and kindness.

---

## Getting Started

### Prerequisites
- **Rust Toolchain:** Stable Rust (2024 edition). Install via [rustup](https://rustup.rs/):
  ```bash
  rustup update stable
  ```
- **Apple Silicon (optional, for Metal & ANE):** macOS 14+ on an M-series Mac with Xcode Command Line Tools installed (`xcode-select --install`).
- **Linux / Cross-Platform:** Standard C build essentials (`build-essential` on Debian/Ubuntu).
- **Python Reference Environment (optional, for Tier 2 parity & Core ML export):** [`uv`](https://docs.astral.sh/uv/) (`uv sync` from `pyproject.toml`).

### Building the Project
```bash
# Build workspace in CPU SIMD mode (universal):
cargo build --release

# Build with Metal GPU, Accelerate, and Apple Neural Engine acceleration (macOS Apple Silicon):
cargo build --release --features metal,accelerate,ane
```

---

## Development & Quality Standards

Before submitting a pull request, ensure your changes adhere to our quality gates:

### 1. Code Formatting
Format all Rust files using standard `rustfmt`:
```bash
cargo fmt --all -- --check
```

### 2. Static Analysis & Clippy
All code must compile cleanly with Clippy warnings treated as errors:
```bash
# Universal (Linux & macOS):
cargo clippy --locked --workspace --all-targets -- -D warnings

# macOS Apple Silicon (Metal + FlashAttention + ANE):
cargo clippy --locked --workspace --all-targets --features metal,metal-flash-attn,ane -- -D warnings
```

### 3. Rustdoc Verification
All public APIs must be documented and build without warnings:
```bash
# Universal (Linux & macOS):
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps

# macOS Apple Silicon:
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --features metal,ane --no-deps
```

### 4. Testing Pyramid & Parity Verification

Glypr maintains a three-tier verification pyramid covering weightless CI checks, fast local suites, and full live Rust-vs-Python differential parity:

```bash
# Tier 0 (Weightless CI): synthetic tiny-model, EngineHandle::mock server contract, and CLI validation tests
GLYPR_ALLOW_MISSING_WEIGHTS=1 HF_HUB_OFFLINE=1 cargo test --locked --workspace --tests --lib

# Tier 1 (macOS Apple Silicon): golden integrity check + workspace check + fast integration tests
./tests/scripts/run_qa_suite.sh --tier1

# Tier 2 (Full Release QA): Tier 1 + --release heavy parity tests + pytest + live HTTP differential_parity.py
./tests/scripts/run_qa_suite.sh --tier2

# Standalone strict FP32 golden fixture integrity check (schema_version: 2) or live server parity
uv run python tests/scripts/verify_goldens.py --strict-fp32
uv run python tests/scripts/differential_parity.py
```

---

## Numerical Fidelity & Parity Invariants

Glypr maintains strict mathematical parity against the official PyTorch CPU **FP32** reference implementation (`SentenceTransformer("google/embeddinggemma-2")`, tracked with `schema_version: 2` golden fixtures):
- **Text Embeddings (292-vector suite):** Must meet or exceed minimum cosine similarity $\ge 0.9998$ (BF16) / $\ge 0.9996$ (Q8_0) and average similarity $\ge 0.9999$ (BF16).
- **Multimodal Projector (`mmproj`):** Must maintain cosine similarity $> 0.999$ across standard visual cases (and $\ge 0.998$ across all 24 golden cases including compact budgets and quantized audio).
- **Normalization:** Output vectors must be unit-normalized ($L_2 = 1.0 \pm 10^{-5}$) prior to Matryoshka truncation.

Any modification to attention kernels, pooling, normalization, preprocessors, or rotary embeddings must pass `corpus_parity`, `multimodal_parity`, and `run_qa_suite.sh --tier2`.

---

## Pull Request Guidelines

1. **Branching:** Create a feature branch off `main`:
   ```bash
   git checkout -b feat/my-improvement
   ```
2. **Commit Messages:** Follow standard conventional commit format (`feat: ...`, `fix: ...`, `perf: ...`, `docs: ...`).
3. **No Leaked Secrets or Local Paths:** Ensure no hardcoded personal filesystem paths (`/Users/...` or `/home/...`), credentials, or proprietary assets are included in commits.
4. **Submit PR:** Open a pull request against `main` on GitHub and verify all CI checks pass.
