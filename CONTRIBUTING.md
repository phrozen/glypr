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

### Building the Project
```bash
# Build workspace in CPU SIMD mode (universal):
cargo build --release

# Build with Metal GPU and Apple Neural Engine acceleration (macOS Apple Silicon):
cargo build --release --features metal,ane
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
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

### 3. Rustdoc Verification
All public APIs must be documented and build without warnings:
```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

### 4. Running Tests
Run the test suite across all crates:
```bash
# CPU mode:
cargo test --workspace

# Apple Silicon Metal GPU + ANE mode:
cargo test --workspace --features metal,ane
```

---

## Numerical Fidelity & Parity Invariants

Glypr maintains strict mathematical parity with the official Python reference implementation (`SentenceTransformers` / PyTorch MPS):
- **Text Embeddings (292-vector suite):** Must meet or exceed minimum cosine similarity $\ge 0.9996$ and average similarity $\ge 0.9999$.
- **Multimodal Projector (`mmproj`):** Must maintain cosine similarity $> 0.999$ across the 24 golden test cases.
- **Normalization:** Output vectors must be unit-normalized ($L_2 = 1.0 \pm 10^{-5}$) prior to Matryoshka truncation.

Any modification to attention kernels, pooling, normalization, or rotary embeddings must be verified against `multimodal_parity.rs` and `corpus_parity.rs`.

---

## Pull Request Guidelines

1. **Branching:** Create a feature branch off `main`:
   ```bash
   git checkout -b feat/my-improvement
   ```
2. **Commit Messages:** Follow standard conventional commit format (`feat: ...`, `fix: ...`, `perf: ...`, `docs: ...`).
3. **No Leaked Secrets or Local Paths:** Ensure no hardcoded personal filesystem paths (`/Users/...` or `/home/...`), credentials, or proprietary assets are included in commits.
4. **Submit PR:** Open a pull request against `main` on GitHub and verify all CI checks pass.
