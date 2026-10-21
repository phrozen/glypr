# Test Asset Credits & Provenance

This directory contains test media assets utilized exclusively for automated regression testing, numerical precision assertions, and golden parity validation against the reference Python harness (`EmbeddingGemma 2`).

---

## Media Fixtures

### 1. Video Fixtures
- **`bbb_5s.mp4`**:
  - **Source:** *Big Buck Bunny* (5-second 1080p excerpt).
  - **Copyright:** © 2008, Blender Foundation / Netherlands Media Art Institute (https://peach.blender.org).
  - **License:** Creative Commons Attribution 3.0 Unported (CC BY 3.0).

### 2. Standard Benchmark Images
- **`baboon.png`**, **`pepper.png`**:
  - **Source:** University of Southern California Signal and Image Processing Institute (USC-SIPI) Image Database (Volume 3: Miscellaneous).
  - **Attribution:** USC-SIPI standard academic test imagery.
- **`countryside.png`**, **`dog.png`**:
  - **Source:** Permissive high-resolution photographic test fixtures for 16:9 landscape and 2:3 portrait aspect ratio budgeting tests.
- **`portrait_test.png`**, **`test_gradient.png`**, **`multimodal_test_image.png`**:
  - **Source:** Synthetic test patterns generated for deterministic multi-resolution token patch verification.

### 3. Audio Fixtures
- **`music_16k.wav`**, **`female_voice_16k.wav`**:
  - **Source:** Open audio sample clips resampled to 16 kHz mono for Conformer audio tower testing.
- **`test_tone_440hz.wav`**, **`multimodal_test_audio.wav`**:
  - **Source:** Synthetically generated 440 Hz sinusoidal and multi-frequency test waveforms.

### 4. Legacy Parity Fixtures
- **`lenna.png`**, **`lena_16k.wav`**:
  - **Notice:** Retained strictly as static regression fixtures for automated numerical parity tests (`multimodal_parity.rs`) against published golden embedding vectors. Not used in any production runtime code or public documentation.
