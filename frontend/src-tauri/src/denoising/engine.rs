//! Denoising engine: thin wrappers around sherpa-onnx's DPDFNet speech-enhancement API
//! (ADR-0027), plus the "Observation Adding" blend used to differentiate how strongly
//! ASR vs diarization consume the denoised signal.
//!
//! Two sherpa-onnx primitives, not one, because DPDFNet is causal/recurrent (Dual-Path
//! RNN) and expects continuity across consecutive calls on a stream:
//!
//! - `OfflineDenoiser` wraps `sherpa_onnx::OfflineSpeechDenoiser`: one call on a
//!   complete buffer, used by the batch paths (import/retranscription), where the
//!   whole file is already decoded before any denoising happens.
//! - `StreamingDenoiser` wraps `sherpa_onnx::OnlineSpeechDenoiser`: consecutive calls
//!   on successive windows of a live stream, maintaining internal state between calls
//!   (unlike the offline API, which has no cross-call memory) -- used by the live
//!   recording pipeline (`audio::pipeline`).
//!
//! Both consumers get a single denoising pass per buffer/window; the ASR-bound and
//! diarization-bound copies are then produced by `blend()`, a cheap linear combination
//! of the original and denoised signal, instead of running the model twice with two
//! different `attenuation_limit_db` values. This is deliberate: `attenuation_limit_db`
//! is fixed at construction time in sherpa-onnx's config, so achieving two different
//! "how aggressive" outputs would otherwise cost 2x inference. See ADR-0027 for why
//! ASR gets a partial blend ("Observation Adding", mitigates the front-end/back-end
//! mismatch documented in the speech-enhancement-for-ASR literature) while diarization
//! gets the denoised signal in full.

use sherpa_onnx::{
    OfflineSpeechDenoiser, OfflineSpeechDenoiserConfig, OfflineSpeechDenoiserDpdfNetModelConfig,
    OfflineSpeechDenoiserModelConfig, OnlineSpeechDenoiser, OnlineSpeechDenoiserConfig,
};
use thiserror::Error;

/// `attenuation_limit_db` passed to sherpa-onnx at construction -- same value used in
/// the upstream Rust example (`offline_speech_enhancement_dpdfnet.rs`). Differentiating
/// ASR vs diarization happens downstream via `blend()`, not by varying this.
const ATTENUATION_LIMIT_DB: f32 = 12.0;

/// Blend weight ("Observation Adding") for the ASR-bound signal: `0.0` = fully
/// original, `1.0` = fully denoised. **Still a placeholder, not calibrated on our own
/// engines/audio** -- calibrate empirically against the existing 5 real reference
/// recordings (`examples/diarization_calibration.rs`) before shipping; see ADR-0027
/// and `docs/sviluppi/denoising/Roadmap e todo.md`. Set to 0.75 (leaning toward
/// denoised) rather than a naive 0.5 midpoint, based on the closest analogous finding
/// in the literature: a bridge-module study for Whisper-based ASR found the optimal
/// learned enhanced-signal weight never dropped below 0.6, reasoning that modern
/// robust ASR models rarely benefit from mixing in much of the original signal -- no
/// paper gives a value validated for our specific engine combination (DPDFNet +
/// Whisper-rs/Parakeet), so this is an informed starting point, not a calibrated one.
pub const ASR_BLEND_WET: f32 = 0.75;

/// Blend weight for the diarization-bound signal. Full denoised by default -- the
/// speech-enhancement-for-diarization literature surveyed in ADR-0027 found no
/// evidence this hurts clustering, unlike the ASR case.
pub const DIARIZATION_BLEND_WET: f32 = 1.0;

#[derive(Debug, Error)]
pub enum DenoisingEngineError {
    #[error("failed to create sherpa-onnx OfflineSpeechDenoiser (check model path)")]
    OfflineInit,
    #[error("failed to create sherpa-onnx OnlineSpeechDenoiser (check model path)")]
    OnlineInit,
}

fn model_config(model_path: &str, num_threads: i32) -> OfflineSpeechDenoiserModelConfig {
    OfflineSpeechDenoiserModelConfig {
        dpdfnet: OfflineSpeechDenoiserDpdfNetModelConfig {
            model: Some(model_path.to_string()),
            attenuation_limit_db: ATTENUATION_LIMIT_DB,
        },
        num_threads,
        ..Default::default()
    }
}

/// Whole-buffer denoiser for the batch paths (import/retranscription). One `.run()`
/// call per file -- no cross-call state to manage.
pub struct OfflineDenoiser {
    denoiser: OfflineSpeechDenoiser,
}

impl OfflineDenoiser {
    pub fn new(model_path: &str, num_threads: i32) -> Result<Self, DenoisingEngineError> {
        let config = OfflineSpeechDenoiserConfig {
            model: model_config(model_path, num_threads),
        };
        let denoiser = OfflineSpeechDenoiser::create(&config).ok_or(DenoisingEngineError::OfflineInit)?;
        Ok(Self { denoiser })
    }

    /// Denoises the whole buffer. Returns the denoised samples and the sample rate
    /// sherpa-onnx reports back -- **not assumed equal to the input rate**, since
    /// `DenoisedAudio` carries its own `sample_rate` field; callers must resample
    /// against the returned rate, not the one they passed in.
    pub fn process(&self, samples: &[f32], sample_rate: i32) -> (Vec<f32>, i32) {
        let denoised = self.denoiser.run(samples, sample_rate);
        (denoised.samples, denoised.sample_rate)
    }
}

/// Consecutive-window denoiser for the live recording pipeline. Unlike
/// `OfflineDenoiser`, this maintains internal state across `.run()` calls -- windows
/// must be fed in order, from the same stream, and `flush()` must be called once at
/// the end to drain any buffered tail.
pub struct StreamingDenoiser {
    denoiser: OnlineSpeechDenoiser,
}

impl StreamingDenoiser {
    pub fn new(model_path: &str, num_threads: i32) -> Result<Self, DenoisingEngineError> {
        let config = OnlineSpeechDenoiserConfig {
            model: model_config(model_path, num_threads),
        };
        let denoiser = OnlineSpeechDenoiser::create(&config).ok_or(DenoisingEngineError::OnlineInit)?;
        Ok(Self { denoiser })
    }

    pub fn process_chunk(&self, samples: &[f32], sample_rate: i32) -> Vec<f32> {
        self.denoiser.run(samples, sample_rate).samples
    }

    /// Drains any buffered tail at the end of a stream. Call exactly once, after the
    /// last `process_chunk`.
    pub fn flush(&self) -> Vec<f32> {
        self.denoiser.flush().samples
    }
}

/// Linear crossfade between the original and denoised signal ("Observation Adding",
/// ADR-0027): `wet = 0.0` returns `original`, `wet = 1.0` returns `denoised`. `wet` is
/// clamped to `[0, 1]`. If the two buffers differ in length (should not normally
/// happen, but sherpa-onnx's reported output length is not hard-guaranteed to match
/// the input exactly), blends only over the overlapping prefix and appends whichever
/// buffer is longer, unmodified, past that point -- avoids panicking or silently
/// truncating audio.
pub fn blend(original: &[f32], denoised: &[f32], wet: f32) -> Vec<f32> {
    let wet = wet.clamp(0.0, 1.0);
    let dry = 1.0 - wet;
    let common_len = original.len().min(denoised.len());

    let mut out = Vec::with_capacity(original.len().max(denoised.len()));
    for i in 0..common_len {
        out.push(original[i] * dry + denoised[i] * wet);
    }
    if original.len() > common_len {
        out.extend_from_slice(&original[common_len..]);
    } else if denoised.len() > common_len {
        out.extend_from_slice(&denoised[common_len..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_wet_zero_returns_original() {
        let original = vec![1.0, 2.0, 3.0];
        let denoised = vec![10.0, 20.0, 30.0];
        assert_eq!(blend(&original, &denoised, 0.0), original);
    }

    #[test]
    fn blend_wet_one_returns_denoised() {
        let original = vec![1.0, 2.0, 3.0];
        let denoised = vec![10.0, 20.0, 30.0];
        assert_eq!(blend(&original, &denoised, 1.0), denoised);
    }

    #[test]
    fn blend_wet_half_averages() {
        let original = vec![0.0, 10.0];
        let denoised = vec![10.0, 0.0];
        assert_eq!(blend(&original, &denoised, 0.5), vec![5.0, 5.0]);
    }

    #[test]
    fn blend_clamps_out_of_range_wet() {
        let original = vec![1.0, 2.0];
        let denoised = vec![10.0, 20.0];
        assert_eq!(blend(&original, &denoised, -1.0), original);
        assert_eq!(blend(&original, &denoised, 2.0), denoised);
    }

    #[test]
    fn blend_handles_denoised_longer_than_original() {
        let original = vec![1.0, 2.0];
        let denoised = vec![10.0, 20.0, 30.0, 40.0];
        // wet=1.0 degenerates to "denoised" exactly, including the extra tail.
        assert_eq!(blend(&original, &denoised, 1.0), denoised);
    }

    #[test]
    fn blend_handles_original_longer_than_denoised() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let denoised = vec![10.0, 20.0];
        // wet=0.0 degenerates to "original" exactly, including the extra tail.
        assert_eq!(blend(&original, &denoised, 0.0), original);
    }

    #[test]
    fn blend_mismatched_length_mixed_weight_does_not_panic() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let denoised = vec![10.0, 20.0];
        let result = blend(&original, &denoised, 0.5);
        assert_eq!(result, vec![5.5, 11.0, 3.0, 4.0]);
    }
}
