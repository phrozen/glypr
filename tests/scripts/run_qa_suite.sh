#!/usr/bin/env bash
set -euo pipefail

TIER="tier1"
for arg in "$@"; do
  case "$arg" in
    --tier1)
      TIER="tier1"
      ;;
    --tier2)
      TIER="tier2"
      ;;
    -h|--help)
      echo "Usage: $0 [--tier1 | --tier2]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--tier1 | --tier2]" >&2
      exit 2
      ;;
  esac
done

run_step() {
  local label="$1"
  shift
  local start_ts=$SECONDS
  echo -e "\n==> ${label}"
  "$@"
  local elapsed=$((SECONDS - start_ts))
  echo "    [done in ${elapsed}s]"
}

SUITE_START=$SECONDS

echo "================================================================================"
echo "                   GLYPR QA & PARITY VERIFICATION SUITE (${TIER})               "
echo "================================================================================"

# Tier 1: Golden integrity, full Apple Silicon compile check, fast tests
run_step "[Tier 1] Verify golden vector integrity" \
  uv run python tests/scripts/verify_goldens.py

run_step "[Tier 1] Cargo check workspace (metal,metal-flash-attn,ane)" \
  cargo check --locked --workspace --all-targets --features metal,metal-flash-attn,ane

run_step "[Tier 1] Cargo test workspace (metal,metal-flash-attn,ane)" \
  cargo test --locked --workspace --features metal,metal-flash-attn,ane

if [[ "$TIER" == "tier2" ]]; then
  run_step "[Tier 2] Cargo ignored heavy parity tests (release + metal,metal-flash-attn,ane)" \
    cargo test --locked --workspace --release --features metal,metal-flash-attn,ane -- --ignored

  run_step "[Tier 2] Python baseline pytest suite" \
    uv run pytest tests/test_baseline.py

  run_step "[Tier 2] Live warm differential parity (Rust server vs Python baseline)" \
    uv run python tests/scripts/differential_parity.py
fi

TOTAL_ELAPSED=$((SECONDS - SUITE_START))
echo -e "\n================================================================================"
echo "          ✓ ALL ${TIER} QA CHECKS PASSED (elapsed: ${TOTAL_ELAPSED}s)"
echo "================================================================================"
