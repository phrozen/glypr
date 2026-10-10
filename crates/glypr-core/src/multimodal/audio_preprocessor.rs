use anyhow::{bail, Result};
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

static HANN_WINDOW: OnceLock<[f32; AUDIO_FRAME_LEN]> = OnceLock::new();

/// Cached periodic Hann window of length 320
pub fn get_hann_window() -> &'static [f32; AUDIO_FRAME_LEN] {
    HANN_WINDOW.get_or_init(|| {
        let mut window = [0.0f32; AUDIO_FRAME_LEN];
        for n in 0..AUDIO_FRAME_LEN {
            window[n] = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * (n as f32) / (AUDIO_FRAME_LEN as f32))).cos();
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
            let angle = -std::f32::consts::PI / (half as f32);
            let w_step_re = angle.cos();
            let w_step_im = angle.sin();

            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;
            for k in 0..half {
                table[offset + k] = (w_re, w_im);
                let next_w_re = w_re * w_step_re - w_im * w_step_im;
                let next_w_im = w_re * w_step_im + w_im * w_step_re;
                w_re = next_w_re;
                w_im = next_w_im;
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
    let mel_filters = get_htk_mel_filterbank();

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

            // Mel filterbank projection + log(mel + 0.001)
            for m in 0..AUDIO_NUM_MELS {
                let mut mel_val = 0.0f32;
                for b in 0..AUDIO_NUM_BINS {
                    mel_val += mag_spec[b] * mel_filters[b * AUDIO_NUM_MELS + m];
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
/// Output soft tokens = L2
pub fn compute_audio_soft_token_count(samples_len: usize) -> usize {
    if samples_len == 0 {
        return 0;
    }
    let pad_left = AUDIO_FRAME_LEN / 2; // 160
    let total_len = pad_left + samples_len;
    let len_padded = if total_len < 321 { 321 } else { total_len };
    let num_frames = (len_padded - 321) / AUDIO_HOP_LEN + 1;
    let l1 = (num_frames + 1) / 2;
    (l1 + 1) / 2
}

/// Compute total tokens required for an audio input including framing tokens (<|audio> and <audio|>).
pub fn compute_audio_total_tokens(samples_len: usize) -> usize {
    if samples_len == 0 {
        return 0;
    }
    compute_audio_soft_token_count(samples_len) + 2
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
}

