use anyhow::{Result, bail};
use rayon::prelude::*;
use std::sync::OnceLock;

pub const AUDIO_SAMPLE_RATE: usize = 16000;
pub const AUDIO_FRAME_LEN: usize = 320;
pub const AUDIO_HOP_LEN: usize = 160;
pub const AUDIO_FFT_LEN: usize = 512;
pub const AUDIO_NUM_BINS: usize = 257;
pub const AUDIO_NUM_MELS: usize = 128;
pub const AUDIO_MEL_FLOOR: f32 = 0.001;

/// Processed audio features ready for Audio Tower encoding
#[derive(Debug, Clone)]
pub struct ProcessedAudio {
    /// Log-mel spectrogram frames: [num_frames, 128]
    pub features: Vec<f32>,
    /// Number of time frames
    pub num_frames: usize,
    /// Number of mel frequency bins (128)
    pub num_mels: usize,
}

/// Convert frequency in Hz to HTK Mel scale
pub fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert HTK Mel scale to frequency in Hz
pub fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

static HTK_MEL_FILTERBANK: OnceLock<Vec<f32>> = OnceLock::new();

/// Cached HTK triangular Mel filterbank matrix: [257, 128]
pub fn get_htk_mel_filterbank() -> &'static [f32] {
    HTK_MEL_FILTERBANK.get_or_init(|| {
        create_htk_mel_filterbank(
            AUDIO_NUM_BINS,
            AUDIO_NUM_MELS,
            AUDIO_SAMPLE_RATE,
            0.0,
            8000.0,
        )
    })
}

static SPARSE_HTK_MEL_FILTERBANK: OnceLock<Vec<(usize, Vec<f32>)>> = OnceLock::new();

/// Cached sparse HTK Mel filterbank storing `(start_bin, non_zero_weights)` per Mel channel
fn get_sparse_htk_mel_filterbank() -> &'static [(usize, Vec<f32>)] {
    SPARSE_HTK_MEL_FILTERBANK.get_or_init(|| {
        let dense = get_htk_mel_filterbank();
        let mut sparse = Vec::with_capacity(AUDIO_NUM_MELS);
        for m in 0..AUDIO_NUM_MELS {
            let mut first = None;
            let mut last = 0;
            for b in 0..AUDIO_NUM_BINS {
                let w = dense[b * AUDIO_NUM_MELS + m];
                if w != 0.0 {
                    if first.is_none() {
                        first = Some(b);
                    }
                    last = b + 1;
                }
            }
            if let Some(start) = first {
                let mut weights = Vec::with_capacity(last - start);
                for b in start..last {
                    weights.push(dense[b * AUDIO_NUM_MELS + m]);
                }
                sparse.push((start, weights));
            } else {
                sparse.push((0, Vec::new()));
            }
        }
        sparse
    })
}

static HANN_WINDOW: OnceLock<[f32; AUDIO_FRAME_LEN]> = OnceLock::new();

/// Cached periodic Hann window of length 320
pub fn get_hann_window() -> &'static [f32; AUDIO_FRAME_LEN] {
    HANN_WINDOW.get_or_init(|| {
        let mut window = [0.0f32; AUDIO_FRAME_LEN];
        for (n, w) in window.iter_mut().enumerate() {
            *w = 0.5
                - 0.5 * (2.0 * std::f32::consts::PI * (n as f32) / (AUDIO_FRAME_LEN as f32)).cos();
        }
        window
    })
}

static BIT_REVERSAL_PAIRS: OnceLock<[(usize, usize); 240]> = OnceLock::new();

fn get_bit_reversal_pairs() -> &'static [(usize, usize); 240] {
    BIT_REVERSAL_PAIRS.get_or_init(|| {
        let mut pairs = [(0, 0); 240];
        let mut idx = 0;
        for i in 0..512 {
            let j = ((i as u32).reverse_bits() >> 23) as usize;
            if i < j {
                pairs[idx] = (i, j);
                idx += 1;
            }
        }
        pairs
    })
}

static FFT_512_TWIDDLES: OnceLock<[(f32, f32); 511]> = OnceLock::new();

fn get_fft_512_twiddles() -> &'static [(f32, f32); 511] {
    FFT_512_TWIDDLES.get_or_init(|| {
        let mut table = [(0.0f32, 0.0f32); 511];
        let mut len = 2;
        let mut offset = 0;
        while len <= 512 {
            let half = len / 2;
            let step_angle = -std::f64::consts::PI / (half as f64);
            for k in 0..half {
                let angle = step_angle * (k as f64);
                table[offset + k] = (angle.cos() as f32, angle.sin() as f32);
            }
            offset += half;
            len <<= 1;
        }
        table
    })
}

/// Generate HTK triangular Mel filterbank matrix: [257, 128]
pub fn create_htk_mel_filterbank(
    num_bins: usize,
    num_filters: usize,
    sample_rate: usize,
    min_freq: f64,
    max_freq: f64,
) -> Vec<f32> {
    let mel_min = hz_to_mel_htk(min_freq);
    let mel_max = hz_to_mel_htk(max_freq);

    let mut filter_freqs = Vec::with_capacity(num_filters + 2);
    for i in 0..=(num_filters + 1) {
        let frac = (i as f64) / ((num_filters + 1) as f64);
        let m = mel_min + frac * (mel_max - mel_min);
        filter_freqs.push(mel_to_hz_htk(m));
    }

    let mut fft_freqs = Vec::with_capacity(num_bins);
    for b in 0..num_bins {
        let f = (b as f64) * ((sample_rate / 2) as f64) / ((num_bins - 1) as f64);
        fft_freqs.push(f);
    }

    let mut filterbank = vec![0.0f32; num_bins * num_filters];

    for i in 0..num_filters {
        let left = filter_freqs[i];
        let center = filter_freqs[i + 1];
        let right = filter_freqs[i + 2];

        for b in 0..num_bins {
            let f = fft_freqs[b];
            let val = if f >= left && f <= center {
                if center > left {
                    (f - left) / (center - left)
                } else {
                    0.0
                }
            } else if f > center && f <= right {
                if right > center {
                    (right - f) / (right - center)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            filterbank[b * num_filters + i] = val as f32;
        }
    }

    filterbank
}

/// In-place Radix-2 Cooley-Tukey FFT for 512 points using precomputed bit-reversals and twiddles
pub fn fft_512_in_place(real: &mut [f32; 512], imag: &mut [f32; 512]) {
    // 1. Bit-reversal permutation
    for &(i, j) in get_bit_reversal_pairs() {
        real.swap(i, j);
        imag.swap(i, j);
    }

    // 2. Cooley-Tukey butterflies with precomputed twiddles
    let twiddles = get_fft_512_twiddles();
    let mut len = 2;
    let mut offset = 0;
    while len <= 512 {
        let half = len / 2;
        let stage_twiddles = &twiddles[offset..offset + half];

        let mut i = 0;
        while i < 512 {
            for k in 0..half {
                let (w_re, w_im) = stage_twiddles[k];
                let u_re = real[i + k];
                let u_im = imag[i + k];

                let v_re = real[i + k + half] * w_re - imag[i + k + half] * w_im;
                let v_im = real[i + k + half] * w_im + imag[i + k + half] * w_re;

                real[i + k] = u_re + v_re;
                imag[i + k] = u_im + v_im;

                real[i + k + half] = u_re - v_re;
                imag[i + k + half] = u_im - v_im;
            }
            i += len;
        }
        offset += half;
        len <<= 1;
    }
}

/// Preprocess raw 16 kHz audio samples into log-mel spectrogram features using parallel Rayon streaming
pub fn preprocess_audio(samples: &[f32]) -> Result<ProcessedAudio> {
    if samples.is_empty() {
        bail!("Input audio cannot be empty");
    }

    let pad_left = AUDIO_FRAME_LEN / 2; // 160
    let mut padded = Vec::with_capacity(pad_left + samples.len());
    padded.resize(pad_left, 0.0f32);
    padded.extend_from_slice(samples);

    let frame_size_for_unfold = AUDIO_FRAME_LEN + 1; // 321
    if padded.len() < frame_size_for_unfold {
        padded.resize(frame_size_for_unfold, 0.0f32);
    }

    let num_frames = (padded.len() - frame_size_for_unfold) / AUDIO_HOP_LEN + 1;

    let window = get_hann_window();
    let sparse_mel_filters = get_sparse_htk_mel_filterbank();

    let mut features = vec![0.0f32; num_frames * AUDIO_NUM_MELS];

    features
        .par_chunks_exact_mut(AUDIO_NUM_MELS)
        .enumerate()
        .for_each(|(frame_idx, frame_features)| {
            let start = frame_idx * AUDIO_HOP_LEN;
            let mut real_buf = [0.0f32; AUDIO_FFT_LEN];
            let mut imag_buf = [0.0f32; AUDIO_FFT_LEN];

            // Window chunk and zero-pad to 512
            for i in 0..AUDIO_FRAME_LEN {
                real_buf[i] = padded[start + i] * window[i];
            }

            // In-place FFT
            fft_512_in_place(&mut real_buf, &mut imag_buf);

            // Magnitude spectrum for first 257 bins
            let mut mag_spec = [0.0f32; AUDIO_NUM_BINS];
            for b in 0..AUDIO_NUM_BINS {
                mag_spec[b] = (real_buf[b] * real_buf[b] + imag_buf[b] * imag_buf[b]).sqrt();
            }

            // Sparse Mel filterbank projection + log(mel + 0.001)
            for (m, (start_bin, weights)) in sparse_mel_filters.iter().enumerate() {
                let mut mel_val = 0.0f32;
                let mag_slice = &mag_spec[*start_bin..*start_bin + weights.len()];
                for (&mag, &w) in mag_slice.iter().zip(weights.iter()) {
                    mel_val += mag * w;
                }
                frame_features[m] = (mel_val + AUDIO_MEL_FLOOR).ln();
            }
        });

    Ok(ProcessedAudio {
        features,
        num_frames,
        num_mels: AUDIO_NUM_MELS,
    })
}

/// Compute exact number of soft tokens produced by Audio Tower from raw 16 kHz sample count in O(1).
///
/// EmbeddingGemma 2 Audio Tower pipeline:
/// 1. Audio preprocessor pads 160 zeros at left, unpacks 321-sample frames with hop 160:
///    num_frames = (pad_left + len - 321) / 160 + 1
/// 2. AudioSubsampleConv runs two Conv2D layers with kernel=3, stride=2, padding=1:
///    L1 = (num_frames + 1) / 2
///    L2 = (L1 + 1) / 2
///    Output soft tokens = L2
pub fn compute_audio_soft_token_count(samples_len: usize) -> usize {
    if samples_len == 0 {
        return 0;
    }
    let pad_left = AUDIO_FRAME_LEN / 2; // 160
    let total_len = pad_left + samples_len;
    let len_padded = if total_len < 321 { 321 } else { total_len };
    let num_frames = (len_padded - 321) / AUDIO_HOP_LEN + 1;
    let l1 = num_frames.div_ceil(2);
    l1.div_ceil(2)
}

/// Compute total tokens required for an audio input including framing tokens (<|audio> and <audio|>).
pub fn compute_audio_total_tokens(samples_len: usize) -> usize {
    if samples_len == 0 {
        return 0;
    }
    compute_audio_soft_token_count(samples_len) + 2
}

/// Decode a WAV stream into mono 16 kHz `f32` samples normalized to `[-1.0, 1.0]`.
///
/// Supports:
/// - `SampleFormat::Int` (8-bit, 16-bit, 24-bit, 32-bit PCM)
/// - `SampleFormat::Float` (32-bit IEEE float WAV)
/// - Multi-channel (stereo, etc.) automatic averaging to mono
/// - Automatic linear resampling to 16,000 Hz if source sample rate differs
/// - Exact `32767.0` divisor for 16-bit mono 16kHz PCM to preserve bit-exact golden parity
pub fn decode_wav_reader<R: std::io::Read>(mut reader: hound::WavReader<R>) -> Result<Vec<f32>> {
    let spec = reader.spec();
    if spec.sample_rate == 0 {
        bail!("Invalid WAV sample rate: 0");
    }
    let channels = spec.channels.max(1) as usize;

    let raw_interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                bail!("Unsupported float WAV bit depth: {}", spec.bits_per_sample);
            }
            reader
                .samples::<f32>()
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| anyhow::anyhow!("Failed to read float WAV samples: {e}"))?
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if ![8, 16, 24, 32].contains(&bits) {
                bail!("Unsupported WAV bit depth: {}", bits);
            }
            if bits <= 16 {
                let scale = if bits == 8 { 127.0f32 } else { 32767.0f32 };
                reader
                    .samples::<i16>()
                    .map(|s| s.map(|v| (v as f32) / scale))
                    .collect::<std::result::Result<Vec<f32>, _>>()
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to read {bits}-bit PCM WAV samples: {e}")
                    })?
            } else {
                let scale = ((1i64 << (bits - 1)) - 1) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| (v as f32) / scale))
                    .collect::<std::result::Result<Vec<f32>, _>>()
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to read {bits}-bit PCM WAV samples: {e}")
                    })?
            }
        }
    };

    let mono: Vec<f32> = if channels == 1 {
        raw_interleaved
    } else {
        raw_interleaved
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / (channels as f32))
            .collect()
    };

    if spec.sample_rate as usize == AUDIO_SAMPLE_RATE || mono.is_empty() {
        Ok(mono)
    } else {
        let src_rate = spec.sample_rate as f64;
        let dst_rate = AUDIO_SAMPLE_RATE as f64;
        let ratio = src_rate / dst_rate;

        // Apply zero-phase windowed-sinc FIR anti-aliasing low-pass filter before downsampling.
        let filtered_storage: Vec<f32>;
        let src_samples: &[f32] = if src_rate > dst_rate {
            const HALF_TAPS: isize = 15; // 31-tap symmetric FIR
            const NUM_TAPS: usize = (2 * HALF_TAPS + 1) as usize;
            let fc = 0.45 * (dst_rate / src_rate);
            let m = (2 * HALF_TAPS) as f64;
            let mut kernel = [0.0f64; NUM_TAPS];
            let mut sum = 0.0f64;
            for (idx, k) in (-HALF_TAPS..=HALF_TAPS).enumerate() {
                let n = (k + HALF_TAPS) as f64;
                let sinc = if k == 0 {
                    2.0 * fc
                } else {
                    let x = std::f64::consts::PI * (k as f64);
                    (2.0 * fc * x).sin() / x
                };
                // Blackman window for strong stopband attenuation
                let win = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * n / m).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * n / m).cos();
                let w = sinc * win;
                kernel[idx] = w;
                sum += w;
            }
            let inv_sum = 1.0 / sum;
            let mut kernel_f32 = [0.0f32; NUM_TAPS];
            for (dst, &src) in kernel_f32.iter_mut().zip(kernel.iter()) {
                *dst = (src * inv_sum) as f32;
            }

            let max_idx = (mono.len() - 1) as isize;
            let mut out = Vec::with_capacity(mono.len());
            for i in 0..mono.len() {
                let mut acc = 0.0f32;
                let center = i as isize;
                for (k_idx, &coef) in kernel_f32.iter().enumerate() {
                    let s_idx = (center + (k_idx as isize) - HALF_TAPS).clamp(0, max_idx) as usize;
                    acc += mono[s_idx] * coef;
                }
                out.push(acc);
            }
            filtered_storage = out;
            &filtered_storage
        } else {
            &mono
        };

        let out_len = (((src_samples.len() as f64) / ratio).round() as usize).max(1);
        let mut resampled = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_pos = (i as f64) * ratio;
            let idx = src_pos.floor() as usize;
            let frac = (src_pos - (idx as f64)) as f32;
            let s0 = src_samples[idx.min(src_samples.len() - 1)];
            let s1 = src_samples[(idx + 1).min(src_samples.len() - 1)];
            resampled.push(s0 + frac * (s1 - s0));
        }
        Ok(resampled)
    }
}

/// Decode a WAV file from disk into mono 16 kHz `f32` samples.
pub fn decode_wav_file<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path.as_ref())
        .map_err(|e| anyhow::anyhow!("Failed to open WAV file {:?}: {e}", path.as_ref()))?;
    decode_wav_reader(reader)
}

/// Decode WAV audio bytes in memory into mono 16 kHz `f32` samples.
pub fn decode_wav_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .map_err(|e| anyhow::anyhow!("Failed to parse WAV audio bytes: {e}"))?;
    decode_wav_reader(reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_audio_soft_token_count_exactness() {
        assert_eq!(compute_audio_soft_token_count(0), 0);
        assert_eq!(compute_audio_total_tokens(0), 0);

        // 1.0 second (16,000 samples) -> 25 soft tokens (+2 framing = 27)
        assert_eq!(compute_audio_soft_token_count(16000), 25);
        assert_eq!(compute_audio_total_tokens(16000), 27);

        // 5.0 seconds (80,000 samples) -> 125 soft tokens (+2 framing = 127)
        assert_eq!(compute_audio_soft_token_count(80000), 125);
        assert_eq!(compute_audio_total_tokens(80000), 127);

        // ~12.27 seconds (196,348 samples, lena_16k.wav) -> 307 soft tokens
        assert_eq!(compute_audio_soft_token_count(196348), 307);
        assert_eq!(compute_audio_total_tokens(196348), 309);

        // ~12.68 seconds (202,915 samples, female_voice_16k.wav) -> 317 soft tokens
        assert_eq!(compute_audio_soft_token_count(202915), 317);
        assert_eq!(compute_audio_total_tokens(202915), 319);
    }

    #[test]
    fn test_decode_wav_formats_float_stereo_and_24bit() {
        // 1. Test 32-bit float stereo WAV
        let spec_f32 = hound::WavSpec {
            channels: 2,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut buf = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(std::io::Cursor::new(&mut buf), spec_f32).unwrap();
            // L = 0.25, R = 0.75 -> Mono = 0.5
            writer.write_sample(0.25f32).unwrap();
            writer.write_sample(0.75f32).unwrap();
            writer.write_sample(-0.5f32).unwrap();
            writer.write_sample(0.5f32).unwrap();
            writer.finalize().unwrap();
        }
        let decoded = decode_wav_bytes(&buf).unwrap();
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0] - 0.5).abs() < 1e-6);
        assert!((decoded[1] - 0.0).abs() < 1e-6);

        // 2. Test 24-bit integer mono WAV
        let spec_i24 = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 24,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf24 = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(std::io::Cursor::new(&mut buf24), spec_i24).unwrap();
            let max_24 = (1i32 << 23) - 1;
            writer.write_sample(max_24).unwrap();
            writer.finalize().unwrap();
        }
        let decoded24 = decode_wav_bytes(&buf24).unwrap();
        assert_eq!(decoded24.len(), 1);
        assert!((decoded24[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fft_twiddles_unit_magnitude() {
        let twiddles = get_fft_512_twiddles();
        assert_eq!(twiddles.len(), 511);
        let mut max_err = 0.0f32;
        for &(re, im) in twiddles.iter() {
            let mag = (re * re + im * im).sqrt();
            let err = (mag - 1.0f32).abs();
            if err > max_err {
                max_err = err;
            }
        }
        println!("Phase 2 FFT-512 twiddle max magnitude error: {:e}", max_err);
        // Phase 2: Direct 64-bit trigonometry guarantees max magnitude error < 1e-6 (near single-float epsilon)
        assert!(
            max_err < 1e-6,
            "Twiddle drift exceeded bound: {:e}",
            max_err
        );
    }

    fn make_raw_pcm_wav(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        data: &[u8],
    ) -> Vec<u8> {
        let bytes_per_sample = bits_per_sample.div_ceil(8).max(1);
        let block_align = channels * bytes_per_sample;
        let byte_rate = sample_rate * (block_align as u32);
        let data_len = data.len() as u32;
        let riff_size = 36u32 + data_len;

        let mut buf = Vec::with_capacity(44 + data.len());
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&riff_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        buf.extend_from_slice(data);
        buf
    }

    #[test]
    fn test_decode_wav_reader_rejects_unsupported_bit_depths() {
        for bad_bits in [0u16, 7, 65] {
            let bytes = make_raw_pcm_wav(16000, 1, bad_bits, &[0u8; 8]);
            let err = decode_wav_bytes(&bytes)
                .expect_err(&format!("bit depth {bad_bits} should be rejected"));
            assert!(
                err.to_string().contains("Unsupported WAV bit depth")
                    || err.to_string().contains("Failed to parse WAV"),
                "unexpected error for bit depth {bad_bits}: {err}"
            );
        }
    }

    #[test]
    fn test_decode_wav_reader_single_sample_high_rate_nonempty() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = Vec::new();
        {
            let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut buf), spec).unwrap();
            writer.write_sample(16384i16).unwrap();
            writer.finalize().unwrap();
        }
        let decoded = decode_wav_bytes(&buf).unwrap();
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - (16384.0 / 32767.0)).abs() < 1e-4);
    }

    #[test]
    fn test_decode_wav_reader_fir_lpf_attenuates_aliasing_48k_to_16k() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut buf = Vec::new();
        {
            let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut buf), spec).unwrap();
            // 12 kHz sine wave (above 16 kHz Nyquist = 8 kHz) at 48 kHz sample rate
            let n_samples = 4800; // 0.1s
            for i in 0..n_samples {
                let t = (i as f32) / 48000.0;
                let s = (2.0 * std::f32::consts::PI * 12000.0 * t).sin();
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        let decoded = decode_wav_bytes(&buf).unwrap();
        assert_eq!(decoded.len(), 1600);
        // Skip edge transient (first/last 16 samples) and verify strong stopband attenuation
        let steady = &decoded[16..decoded.len() - 16];
        let rms = (steady.iter().map(|&x| x * x).sum::<f32>() / (steady.len() as f32)).sqrt();
        // Unattenuated sine RMS is ~0.707; 31-tap Blackman FIR at fc=0.15 attenuates 12 kHz > 40 dB
        assert!(
            rms < 0.02,
            "12 kHz above-Nyquist tone was not sufficiently attenuated: rms={rms}"
        );
    }

    #[test]
    fn test_preprocess_audio_empty_and_short_edge_cases() {
        // Empty audio must fail
        let err = preprocess_audio(&[]).unwrap_err();
        assert!(err.to_string().contains("empty"), "unexpected error: {err}");

        // 100 samples (< 321 after left-pad of 160 -> 260 < 321) pads to 321 and yields 1 frame
        let short = vec![0.1f32; 100];
        let processed = preprocess_audio(&short).unwrap();
        assert_eq!(processed.num_frames, 1);
        assert_eq!(processed.num_mels, AUDIO_NUM_MELS);
        assert_eq!(processed.features.len(), AUDIO_NUM_MELS);
        assert!(
            processed.features.iter().all(|v| v.is_finite()),
            "short audio features contained non-finite values"
        );
    }

    #[test]
    fn test_decode_wav_reader_8bit_32bit_and_corrupt_empty() {
        // 8-bit integer mono PCM
        let spec_i8 = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 8,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf8 = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(std::io::Cursor::new(&mut buf8), spec_i8).unwrap();
            writer.write_sample(127i8).unwrap();
            writer.write_sample(-127i8).unwrap();
            writer.write_sample(0i8).unwrap();
            writer.finalize().unwrap();
        }
        let dec8 = decode_wav_bytes(&buf8).unwrap();
        assert_eq!(dec8.len(), 3);
        assert!((dec8[0] - 1.0).abs() < 1e-5);
        assert!((dec8[1] + 1.0).abs() < 1e-5);
        assert!((dec8[2] - 0.0).abs() < 1e-5);

        // 32-bit integer mono PCM
        let spec_i32 = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf32 = Vec::new();
        {
            let mut writer =
                hound::WavWriter::new(std::io::Cursor::new(&mut buf32), spec_i32).unwrap();
            writer.write_sample(i32::MAX).unwrap();
            writer.write_sample(0i32).unwrap();
            writer.finalize().unwrap();
        }
        let dec32 = decode_wav_bytes(&buf32).unwrap();
        assert_eq!(dec32.len(), 2);
        assert!((dec32[0] - 1.0).abs() < 1e-5);
        assert!((dec32[1] - 0.0).abs() < 1e-5);

        // Empty and corrupt WAV bytes must fail cleanly
        assert!(decode_wav_bytes(&[]).is_err());
        assert!(decode_wav_bytes(b"not a valid wav header").is_err());
    }

    #[test]
    fn test_audio_preprocessor_matches_golden_json() {
        #[derive(serde::Deserialize)]
        struct PreprocessorGolden {
            audio: AudioGolden,
        }

        #[derive(serde::Deserialize)]
        struct AudioGolden {
            sample_rate: usize,
            freq_hz: f64,
            num_samples: usize,
            num_frames: usize,
            num_mels: usize,
            features: Vec<f32>,
        }

        let golden_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden_vectors/preprocessor_golden.json");
        if !golden_path.exists() {
            let allow_missing = matches!(
                std::env::var("GLYPR_ALLOW_MISSING_WEIGHTS")
                    .unwrap_or_default()
                    .trim()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true"
            );
            if allow_missing {
                eprintln!(
                    "Skipping test_audio_preprocessor_matches_golden_json ({} missing and GLYPR_ALLOW_MISSING_WEIGHTS=1)",
                    golden_path.display()
                );
                return;
            }
            panic!(
                "Missing preprocessor golden fixture at {}",
                golden_path.display()
            );
        }

        let raw = std::fs::read_to_string(&golden_path).unwrap();
        let golden: PreprocessorGolden = serde_json::from_str(&raw).unwrap();
        let audio = golden.audio;
        assert_eq!(audio.num_mels, 128);

        let samples: Vec<f32> = (0..audio.num_samples)
            .map(|i| {
                (2.0 * std::f64::consts::PI * audio.freq_hz * (i as f64)
                    / (audio.sample_rate as f64))
                    .sin() as f32
            })
            .collect();

        let processed = preprocess_audio(&samples).unwrap();
        assert_eq!(processed.num_frames, audio.num_frames);
        assert_eq!(processed.num_mels, 128);
        assert_eq!(processed.features.len(), audio.features.len());

        let mut max_abs_diff = 0.0f32;
        for (&got, &expected) in processed.features.iter().zip(audio.features.iter()) {
            assert!(got.is_finite(), "non-finite audio feature encountered");
            max_abs_diff = max_abs_diff.max((got - expected).abs());
        }
        assert!(
            max_abs_diff < 1e-4,
            "audio preprocessor max_abs_diff={max_abs_diff:.6e} >= 1e-4"
        );
    }
}
