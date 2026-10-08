use anyhow::{bail, Result};

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

/// In-place Radix-2 Cooley-Tukey FFT for 512 points
pub fn fft_512_in_place(real: &mut [f32; 512], imag: &mut [f32; 512]) {
    // 1. Bit-reversal permutation (9 bits for 512 points)
    for i in 0..512 {
        let j = ((i as u32).reverse_bits() >> 23) as usize;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    // 2. Cooley-Tukey butterflies
    let mut len = 2;
    while len <= 512 {
        let half = len / 2;
        let angle = -std::f32::consts::PI / (half as f32);
        let w_step_re = angle.cos();
        let w_step_im = angle.sin();

        let mut i = 0;
        while i < 512 {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;

            for k in 0..half {
                let u_re = real[i + k];
                let u_im = imag[i + k];

                let v_re = real[i + k + half] * w_re - imag[i + k + half] * w_im;
                let v_im = real[i + k + half] * w_im + imag[i + k + half] * w_re;

                real[i + k] = u_re + v_re;
                imag[i + k] = u_im + v_im;

                real[i + k + half] = u_re - v_re;
                imag[i + k + half] = u_im - v_im;

                let next_w_re = w_re * w_step_re - w_im * w_step_im;
                let next_w_im = w_re * w_step_im + w_im * w_step_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Preprocess raw 16 kHz audio samples into log-mel spectrogram features
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

    // Periodic Hann window of length 320
    let mut window = [0.0f32; AUDIO_FRAME_LEN];
    for n in 0..AUDIO_FRAME_LEN {
        window[n] = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * (n as f32) / (AUDIO_FRAME_LEN as f32))).cos();
    }

    let mel_filters = create_htk_mel_filterbank(
        AUDIO_NUM_BINS,
        AUDIO_NUM_MELS,
        AUDIO_SAMPLE_RATE,
        0.0,
        8000.0,
    );

    let mut features = Vec::with_capacity(num_frames * AUDIO_NUM_MELS);
    let mut real_buf = [0.0f32; AUDIO_FFT_LEN];
    let mut imag_buf = [0.0f32; AUDIO_FFT_LEN];
    let mut mag_spec = [0.0f32; AUDIO_NUM_BINS];

    for frame_idx in 0..num_frames {
        let start = frame_idx * AUDIO_HOP_LEN;

        // Window chunk and zero-pad to 512
        for i in 0..AUDIO_FRAME_LEN {
            real_buf[i] = padded[start + i] * window[i];
            imag_buf[i] = 0.0;
        }
        for i in AUDIO_FRAME_LEN..AUDIO_FFT_LEN {
            real_buf[i] = 0.0;
            imag_buf[i] = 0.0;
        }

        // FFT
        fft_512_in_place(&mut real_buf, &mut imag_buf);

        // Magnitude spectrum for first 257 bins
        for b in 0..AUDIO_NUM_BINS {
            mag_spec[b] = (real_buf[b] * real_buf[b] + imag_buf[b] * imag_buf[b]).sqrt();
        }

        // Mel filterbank projection + log(mel + 0.001)
        for m in 0..AUDIO_NUM_MELS {
            let mut mel_val = 0.0f32;
            for b in 0..AUDIO_NUM_BINS {
                mel_val += mag_spec[b] * mel_filters[b * AUDIO_NUM_MELS + m];
            }
            let log_mel = (mel_val + AUDIO_MEL_FLOOR).ln();
            features.push(log_mel);
        }
    }

    Ok(ProcessedAudio {
        features,
        num_frames,
        num_mels: AUDIO_NUM_MELS,
    })
}
