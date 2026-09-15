//! Denoising model catalog, on-demand download, and status tracking.
//!
//! Single fixed model (DPDFNet, 48kHz high-resolution variant, ADR-0027 in the docs
//! workspace) -- unlike diarization's two files, there is only one `.onnx` file here,
//! not distributed as an archive. Same lifecycle pattern as
//! `diarization::model`/`whisper_engine`/`parakeet_engine`: on-demand download only,
//! never proactive (mirrors ADR-0010's pattern for a second, independent toggle).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The one fixed model file this feature needs.
pub struct DenoisingModelFile {
    /// Stable identifier, not shown to the user.
    pub name: &'static str,
    pub download_url: &'static str,
    pub final_filename: &'static str,
    pub size_mb: u32,
    pub license: &'static str,
    pub description: &'static str,
}

/// Official k2-fsa/sherpa-onnx GitHub release (`speech-enhancement-models` tag), same
/// family used by the diarization models -- filename confirmed against the real
/// release assets (`dpdfnet2_48khz_hr.onnx`), not guessed. See ADR-0027 for why this
/// specific variant (48kHz-native, matches the mixed audio pipeline's own sample rate)
/// over the 16kHz baseline/2/4/8 variants or GTCRN.
pub const DENOISING_MODEL_CATALOG: &[DenoisingModelFile] = &[DenoisingModelFile {
    name: "dpdfnet",
    download_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2_48khz_hr.onnx",
    final_filename: "dpdfnet2_48khz_hr.onnx",
    size_mb: 11,
    license: "Apache-2.0",
    description: "DPDFNet 48kHz high-resolution speech enhancement (Ceva-IP) -- see docs/adr/0027",
}];

/// Mirrors `diarization::model::DiarizationModelStatus` (same variants, duplicated
/// per-feature by existing project convention).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DenoisingModelStatus {
    Available,
    Missing,
    Downloading { progress: u8 },
    Corrupted { file: String },
    Error(String),
}

/// Detailed download progress, mirroring `diarization::model::DiarizationDownloadProgress`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenoisingDownloadProgress {
    pub file: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: u8,
}

/// Subfolder of the app's models directory denoising models live in, same convention
/// as `diarization/`/`parakeet/`.
pub fn models_dir(base: &Path) -> PathBuf {
    base.join("denoising")
}

pub fn model_path(base: &Path) -> PathBuf {
    models_dir(base).join(DENOISING_MODEL_CATALOG[0].final_filename)
}

/// Minimum plausible file size, in bytes, used as a cheap corruption check (same idea
/// as `diarization::model::MIN_PLAUSIBLE_BYTES`).
const MIN_PLAUSIBLE_BYTES: u64 = 1024 * 100; // 100KB

pub async fn check_status(base: &Path) -> DenoisingModelStatus {
    match tokio::fs::metadata(model_path(base)).await {
        Ok(meta) if meta.len() >= MIN_PLAUSIBLE_BYTES => DenoisingModelStatus::Available,
        Ok(_) => DenoisingModelStatus::Corrupted {
            file: model_path(base).display().to_string(),
        },
        Err(_) => DenoisingModelStatus::Missing,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DenoisingModelError {
    #[error("network error downloading {file}: {source}")]
    Network {
        file: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("failed to write {file} to disk: {source}")]
    Io {
        file: String,
        #[source]
        source: std::io::Error,
    },
}

/// Downloads the model file, calling `on_progress` after each meaningful chunk.
/// Idempotent-ish: re-downloads unconditionally when called (the caller -- the Tauri
/// command layer -- is responsible for checking `check_status` first and only calling
/// this on an explicit user "Download" action).
pub async fn download_models<F>(base: &Path, mut on_progress: F) -> Result<(), DenoisingModelError>
where
    F: FnMut(DenoisingDownloadProgress),
{
    let dir = models_dir(base);
    tokio::fs::create_dir_all(&dir).await.map_err(|e| DenoisingModelError::Io {
        file: dir.display().to_string(),
        source: e,
    })?;

    let file = &DENOISING_MODEL_CATALOG[0];
    let dest = dir.join(file.final_filename);
    let bytes = download_bytes(file.download_url, file.name, &mut on_progress).await?;
    tokio::fs::write(&dest, bytes)
        .await
        .map_err(|e| DenoisingModelError::Io { file: dest.display().to_string(), source: e })?;

    Ok(())
}

async fn download_bytes<F>(
    url: &str,
    label: &str,
    on_progress: &mut F,
) -> Result<Vec<u8>, DenoisingModelError>
where
    F: FnMut(DenoisingDownloadProgress),
{
    use futures_util::StreamExt;

    let response = reqwest::get(url).await.map_err(|e| DenoisingModelError::Network {
        file: label.to_string(),
        source: e,
    })?;
    let total_bytes = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut buffer = Vec::with_capacity(total_bytes as usize);

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| DenoisingModelError::Network {
            file: label.to_string(),
            source: e,
        })?;
        downloaded += chunk.len() as u64;
        buffer.extend_from_slice(&chunk);
        let percent = if total_bytes > 0 {
            ((downloaded as f64 / total_bytes as f64) * 100.0) as u8
        } else {
            0
        };
        on_progress(DenoisingDownloadProgress {
            file: label.to_string(),
            downloaded_bytes: downloaded,
            total_bytes,
            percent,
        });
    }

    Ok(buffer)
}

/// Reads `transcript_settings.denoising_enabled` and, if on, validates the model is
/// downloaded (same "block, don't silently skip" gate as
/// `diarization::model::resolve_paths_if_enabled`, roadmap 6h equivalent). Shared by the
/// live path (`denoising::prepare_if_enabled`, called from the two recording-start
/// entry points) and the batch paths (`audio::import`/`audio::retranscription`), so the
/// toggle-read + gate logic isn't duplicated.
///
/// Independent of `diarization_enabled` -- denoising has value even with diarization
/// off (it also improves plain ASR), so it is never gated on the diarization toggle.
///
/// Returns `Ok(None)` when the toggle is off (callers should skip denoising entirely,
/// zero overhead). Returns `Ok(Some((model_path, save_debug_files)))` when enabled and
/// ready -- `save_debug_files` is `transcript_settings.denoising_save_debug_files`
/// (default `false`, opt-in on top of `denoising_enabled`: see
/// `SettingsRepository::get_denoising_save_debug_files` for why), read here alongside
/// `denoising_enabled` since the `SqlitePool` is already in scope, rather than a
/// second pool acquisition in each caller (same reasoning as diarization's
/// `max_speakers` in `diarization::model::resolve_paths_if_enabled`).
pub async fn resolve_paths_if_enabled<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<Option<(String, bool)>, String> {
    use tauri::Manager;

    // NOTE: `app_for_state` must be a named binding, not inline `app.clone().state()` --
    // same lifetime pitfall documented in `diarization::model::resolve_paths_if_enabled`.
    let app_for_state = app.clone();
    let state: tauri::State<'_, crate::state::AppState> = app_for_state.state();
    let pool = state.db_manager.pool().clone();

    let enabled = crate::database::repositories::setting::SettingsRepository::get_denoising_enabled(&pool)
        .await
        .map_err(|e| format!("Failed to read denoising_enabled setting: {}", e))?;

    if !enabled {
        return Ok(None);
    }

    let save_debug_files = crate::database::repositories::setting::SettingsRepository::get_denoising_save_debug_files(&pool)
        .await
        .map_err(|e| format!("Failed to read denoising_save_debug_files setting: {}", e))?;

    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {}", e))?;

    match check_status(&base).await {
        DenoisingModelStatus::Available => {}
        other => {
            return Err(format!(
                "Denoising is enabled but the model is not ready ({:?}). Please download it from Settings first.",
                other
            ));
        }
    }

    Ok(Some((
        model_path(&base).to_string_lossy().into_owned(),
        save_debug_files,
    )))
}
