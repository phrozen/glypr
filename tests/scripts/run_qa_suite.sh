#!/bin/bash
set -euo pipefail

echo "================================================================================"
echo "                   GLYPR E2E QA & PARITY VERIFICATION SUITE                     "
echo "================================================================================"

# 1. Environment & HF Cache Verification
echo -e "\n[1/4] Verifying HF Cache Resolution & Python Tools..."
uv run python tests/scripts/verify_cache.py

# 2. Python Unit & MRL Tests
echo -e "\n[2/4] Running Python Baseline Tests..."
uv run pytest tests/test_baseline.py

# 3. Build & Lint Verification
echo -e "\n[3/4] Verifying Cargo Checks & Zero-Warning Hygiene..."
cargo check --all --tests
cargo check --all --examples --features metal

# 4. Full Rust Test Suite (CPU SIMD, Golden Parity, Corpus Parity, Multimodal Parity)
echo -e "\n[4/4] Running Full Test Suite on Apple Silicon (Release + Metal + FlashAttn)..."
cargo test --all --release --features metal,metal-flash-attn -- --nocapture

echo -e "\n================================================================================"
echo "          ✓ ALL QA SUITE CHECKS & GOLDEN PARITY TESTS PASSED (> 0.999)          "
echo "================================================================================"
