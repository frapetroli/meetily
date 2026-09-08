//! Tauri command layer for the diarization model catalog (roadmap 7c). One combined
//! "Modelli diarization" card in the UI, not a per-variant list like Whisper/Parakeet --
//! there is nothing to choose, only two fixed files to check/download (`model.rs`).

use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Manager, Runtime};

use super::model::{check_status, download_models, DiarizationModelStatus};

// Prevents overlapping downloads if the user double-clicks -- same intent as the
// per-model `active_downloads` sets in whisper_engine/parakeet_engine, simplified since
// there is only ever one thing to download here (both files, together).
static DOWNLOAD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

#[tauri::command]
pub async fn diarization_get_status<R: Runtime>(app: AppHandle<R>) -> Result<DiarizationModelStatus, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {}", e))?;
    Ok(check_status(&base).await)
}

/// Downloads both model files (segmentation + embedding), emitting
/// `diarization-model-download-progress` events throughout and
/// `diarization-model-download-complete`/`diarization-model-download-error` at the end --
/// same event-naming convention as `parakeet-model-download-*`/`builtin-ai-download-*`.
/// Triggered only by an explicit user action (Download button), never automatically --
/// ADR-0010.
#[tauri::command]
pub async fn diarization_download_models<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    if DOWNLOAD_IN_PROGRESS.swap(true, Ordering::SeqCst) {
        return Err("A diarization model download is already in progress".to_string());
    }

    // Hold the same lock used around Whisper/Parakeet engine lifecycle (audio/common.rs)
    // for the whole download: `download_models` writes straight to the final `.onnx`
    // filenames (no temp file + rename), so a `DiarizationEngine::new()` call racing a
    // download here could read a partially-written model file.
    let _engine_lifecycle_guard = crate::audio::common::acquire_engine_lifecycle_lock().await;

    let result = download_inner(&app).await;

    DOWNLOAD_IN_PROGRESS.store(false, Ordering::SeqCst);

    match &result {
        Ok(()) => {
            let _ = app.emit("diarization-model-download-complete", ());
        }
        Err(e) => {
            let _ = app.emit("diarization-model-download-error", e.clone());
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
            "diarization-model-download-progress",
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
