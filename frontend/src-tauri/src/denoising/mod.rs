//! Speech denoising module (CPU-only, sherpa-onnx/DPDFNet based).
//!
//! Reduces background noise on the mixed audio *before* it reaches either ASR or
//! diarization, independently of both -- see `docs/adr/0027-denoising-motore-dpdfnet-via-sherpa-onnx.md`
//! in the docs workspace. Not specific to diarization: it has value with diarization
//! off too, so it is gated by its own `denoising_enabled` toggle, never by
//! `diarization_enabled`.
//!
//! # Module structure
//!
//! - `model`: model catalog (single DPDFNet file), download/status management
//! - `engine`: thin wrappers around sherpa-onnx's offline/online speech-denoiser API,
//!   plus the `blend()` ("Observation Adding") helper that differentiates how strongly
//!   the ASR-bound vs diarization-bound copies of the signal are denoised
//! - `commands`: Tauri command layer for model status/download (mirrors
//!   `diarization::commands`)

pub mod commands;
pub mod debug_wav;
pub mod engine;
pub mod model;

pub use commands::{denoising_download_models, denoising_get_status};
pub use debug_wav::DebugWavWriter;
pub use engine::{
    blend, DenoisingEngineError, OfflineDenoiser, StreamingDenoiser, ASR_BLEND_WET,
    DIARIZATION_BLEND_WET,
};
pub use model::{resolve_paths_if_enabled, DenoisingModelStatus, DENOISING_MODEL_CATALOG};

/// Resolves `denoising_enabled` and constructs a `StreamingDenoiser` ready for the live
/// recording pipeline, mirroring `diarization::DiarizationSession::prepare_if_enabled`.
/// Returns `Ok(None)` when the toggle is off (zero overhead) or
/// `Ok(Some((denoiser, save_debug_files)))` when on and the model is ready --
/// `save_debug_files` is `transcript_settings.denoising_save_debug_files`, for the
/// caller to decide whether to also set up `DebugWavWriter`s (see
/// `audio::recording_commands`). Construction of the denoiser itself is synchronous --
/// the model file is small (~11MB), same assumption already made for
/// `DiarizationEngine::new()` inside `DiarizationSession::start`.
pub async fn prepare_if_enabled<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<Option<(StreamingDenoiser, bool)>, String> {
    let (model_path, save_debug_files) = match model::resolve_paths_if_enabled(app).await? {
        Some(paths) => paths,
        None => return Ok(None),
    };
    let denoiser = StreamingDenoiser::new(&model_path, 1)
        .map_err(|e| format!("Failed to initialize denoising engine: {}", e))?;
    Ok(Some((denoiser, save_debug_files)))
}
