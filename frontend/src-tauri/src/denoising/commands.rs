//! Tauri command layer for the denoising model catalog. One model, one status/download
//! pair -- no per-variant choice, mirroring `diarization::commands`.

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Manager, Runtime};

use super::model::{check_status, download_models, DenoisingModelStatus};

// Prevents overlapping downloads if the user double-clicks -- same intent as
// `diarization::commands::DOWNLOAD_IN_PROGRESS`.
static DOWNLOAD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

#[tauri::command]
pub async fn denoising_get_status<R: Runtime>(app: AppHandle<R>) -> Result<DenoisingModelStatus, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {}", e))?;
    Ok(check_status(&base).await)
}

/// Downloads the model file, emitting `denoising-model-download-progress` events
/// throughout and `denoising-model-download-complete`/`denoising-model-download-error`
/// at the end -- same event-naming convention as `diarization-model-download-*`.
/// Triggered only by an explicit user action (Download button), never automatically.
#[tauri::command]
pub async fn denoising_download_models<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    if DOWNLOAD_IN_PROGRESS.swap(true, Ordering::SeqCst) {
        return Err("A denoising model download is already in progress".to_string());
    }

    // Hold the same lock used around Whisper/Parakeet/diarization engine lifecycle
    // (audio/common.rs) for the whole download: `download_models` writes straight to
    // the final `.onnx` filename (no temp file + rename), so a denoiser construction
    // racing a download here could read a partially-written model file.
    let _engine_lifecycle_guard = crate::audio::common::acquire_engine_lifecycle_lock().await;

    let result = download_inner(&app).await;

    DOWNLOAD_IN_PROGRESS.store(false, Ordering::SeqCst);

    match &result {
        Ok(()) => {
            let _ = app.emit("denoising-model-download-complete", ());
        }
        Err(e) => {
            let _ = app.emit("denoising-model-download-error", e.clone());
        }
    }

    result
}

async fn download_inner<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {}", e))?;

    let app_for_progress = app.clone();
    download_models(&base, move |progress| {
        let _ = app_for_progress.emit(
            "denoising-model-download-progress",
            serde_json::json!({
                "file": progress.file,
                "downloaded_bytes": progress.downloaded_bytes,
                "total_bytes": progress.total_bytes,
                "percent": progress.percent,
            }),
        );
    })
    .await
    .map_err(|e| e.to_string())
}
