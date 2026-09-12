//! Diarization model catalog, on-demand download, and status tracking.
//!
//! Unlike Whisper/Parakeet, there is no user-facing choice of model variant here: exactly
//! two fixed files are required (pyannote-segmentation-3.0 + 3D-Speaker embedding, ADR-0007),
//! ~33MB total. The UI (roadmap 7c) is expected to show a single combined "Modelli
//! diarization" card, not a per-variant list like `WhisperModelManager.tsx`.
//!
//! Same lifecycle pattern as `whisper_engine`/`parakeet_engine`: `DiarizationModelStatus`
//! mirrors `whisper_engine::ModelStatus`/`parakeet_engine::ModelStatus`, models live in a
//! `diarization/` subfolder of the app's models directory (same convention as Parakeet's
//! own `parakeet/` subfolder), and downloads are on-demand, never proactive (ADR-0010).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// One of the two fixed model files this feature needs.
pub struct DiarizationModelFile {
    /// `"segmentation"` or `"embedding"` -- used as a stable identifier, not shown to the user.
    pub name: &'static str,
    pub download_url: &'static str,
    /// `true` for the segmentation model, distributed as a `.tar.bz2` archive.
    pub is_archive: bool,
    /// Path of the actual `.onnx` file inside the archive, if `is_archive`.
    pub archive_member: Option<&'static str>,
    /// Filename this ends up as inside `models_dir()`, after any extraction.
    pub final_filename: &'static str,
    pub size_mb: u32,
    pub license: &'static str,
    pub description: &'static str,
}

/// Same URLs already used and validated in the project's own spikes (see
/// `docs/sviluppi/diarization/Architettura pipeline.md`, "Blocker 2" benchmark recipe) --
/// official k2-fsa/sherpa-onnx GitHub releases, not a third-party mirror.
pub const DIARIZATION_MODEL_CATALOG: &[DiarizationModelFile] = &[
    DiarizationModelFile {
        name: "segmentation",
        download_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        is_archive: true,
        archive_member: Some("sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
        final_filename: "segmentation-model.onnx",
        size_mb: 7,
        license: "MIT",
        description: "pyannote-segmentation-3.0 (ONNX export, CNRS) -- see docs/adr/0007",
    },
    DiarizationModelFile {
        name: "embedding",
        download_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        is_archive: false,
        archive_member: None,
        final_filename: "embedding-model.onnx",
        size_mb: 26,
        license: "Apache-2.0",
        description: "3D-Speaker eres2net, English (ModelScope) -- see docs/adr/0007",
    },
];

/// Mirrors `whisper_engine::ModelStatus`/`parakeet_engine::ModelStatus` (same varitants,
/// duplicated per-engine by existing project convention -- see
/// `docs/as-is/Codice morto e moduli non collegati.md` re: this not being an oversight).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiarizationModelStatus {
    Available,
    Missing,
    Downloading { progress: u8 },
    Corrupted { file: String },
    Error(String),
}

/// Detailed download progress for one file, mirroring `whisper_engine`/`parakeet_engine`'s
/// `DownloadProgress` shape (kept separate per-engine, same convention).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationDownloadProgress {
    pub file: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: u8,
}

/// Subfolder of the app's models directory diarization models live in, same convention
/// as Parakeet's own `parakeet/` subfolder (`parakeet_engine.rs::new_with_models_dir`).
pub fn models_dir(base: &Path) -> PathBuf {
    base.join("diarization")
}

pub fn segmentation_model_path(base: &Path) -> PathBuf {
    models_dir(base).join(DIARIZATION_MODEL_CATALOG[0].final_filename)
}

pub fn embedding_model_path(base: &Path) -> PathBuf {
    models_dir(base).join(DIARIZATION_MODEL_CATALOG[1].final_filename)
}

/// Minimum plausible file size, in bytes, used as a cheap corruption check (same idea as
/// `WhisperEngine::discover_models`' "at least 90% of expected size" check, simplified
/// since there's no size-variant catalog here to compare against).
const MIN_PLAUSIBLE_BYTES: u64 = 1024 * 100; // 100KB

/// Checks both files on disk and returns a single aggregate status, since the UI shows
/// one combined card, not per-file status (roadmap 7c).
pub async fn check_status(base: &Path) -> DiarizationModelStatus {
    for path in [segmentation_model_path(base), embedding_model_path(base)] {
        match tokio::fs::metadata(&path).await {
            Ok(meta) if meta.len() >= MIN_PLAUSIBLE_BYTES => continue,
            Ok(_) => {
                return DiarizationModelStatus::Corrupted {
                    file: path.display().to_string(),
                }
            }
            Err(_) => return DiarizationModelStatus::Missing,
        }
    }
    DiarizationModelStatus::Available
}

#[derive(Debug, thiserror::Error)]
pub enum DiarizationModelError {
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
    #[error("failed to extract {member} from archive: {source}")]
    Extract {
        member: String,
        #[source]
        source: std::io::Error,
    },
    #[error("expected member '{0}' not found in downloaded archive")]
    MemberNotFound(String),
}

/// Downloads and (if needed) extracts both model files, calling `on_progress` after each
/// meaningful chunk. Idempotent-ish: re-downloads unconditionally when called (the caller
/// -- the Tauri command layer, roadmap 7c -- is responsible for checking `check_status`
/// first and only calling this on an explicit user "Download" action, per ADR-0010).
pub async fn download_models<F>(base: &Path, mut on_progress: F) -> Result<(), DiarizationModelError>
where
    F: FnMut(DiarizationDownloadProgress),
{
    let dir = models_dir(base);
    tokio::fs::create_dir_all(&dir).await.map_err(|e| DiarizationModelError::Io {
        file: dir.display().to_string(),
        source: e,
    })?;

    for file in DIARIZATION_MODEL_CATALOG {
        let dest = dir.join(file.final_filename);
        let bytes = download_bytes(file.download_url, file.name, &mut on_progress).await?;

        if file.is_archive {
            let member = file
                .archive_member
                .expect("catalog entries with is_archive=true must set archive_member");
            let extracted = extract_archive_member(&bytes, member)?;
            tokio::fs::write(&dest, extracted)
                .await
                .map_err(|e| DiarizationModelError::Io { file: dest.display().to_string(), source: e })?;
        } else {
            tokio::fs::write(&dest, bytes)
                .await
                .map_err(|e| DiarizationModelError::Io { file: dest.display().to_string(), source: e })?;
        }
    }

    Ok(())
}

async fn download_bytes<F>(
    url: &str,
    label: &str,
    on_progress: &mut F,
) -> Result<Vec<u8>, DiarizationModelError>
where
    F: FnMut(DiarizationDownloadProgress),
{
    use futures_util::StreamExt;

    let response = reqwest::get(url).await.map_err(|e| DiarizationModelError::Network {
        file: label.to_string(),
        source: e,
    })?;
    let total_bytes = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut buffer = Vec::with_capacity(total_bytes as usize);

    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| DiarizationModelError::Network {
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
        on_progress(DiarizationDownloadProgress {
            file: label.to_string(),
            downloaded_bytes: downloaded,
            total_bytes,
            percent,
        });
    }

    Ok(buffer)
}

/// Reads `transcript_settings.diarization_enabled` and, if on, validates the models are
/// downloaded (roadmap 6h: recording/batch start is blocked, not silently skipped, if the
/// toggle is on but models are missing). Shared by the live path
/// (`diarization::DiarizationSession::prepare_if_enabled`) and the batch paths
/// (`audio::import`/`audio::retranscription`), so the toggle-read + gate logic isn't
/// duplicated three times.
///
/// Returns `Ok(None)` when the toggle is off (callers should skip diarization entirely,
/// zero overhead). Returns
/// `Ok(Some((segmentation_model_path, embedding_model_path, max_speakers)))` when enabled
/// and ready -- `max_speakers` is `transcript_settings.diarization_max_speakers` (see
/// `docs/adr/0024-...md`), read here alongside `diarization_enabled` since the `SqlitePool`
/// is already in scope, rather than a second pool acquisition in each of the three callers.
pub async fn resolve_paths_if_enabled<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
) -> Result<Option<(String, String, usize)>, String> {
    use tauri::Manager;

    // NOTE: `app_for_state` must be a named binding, not inline `app.clone().state()` --
    // otherwise the temporary AppHandle from `.clone()` would be dropped at the end of
    // this statement, before `state`'s borrow of it is used on the next line.
    let app_for_state = app.clone();
    let state: tauri::State<'_, crate::state::AppState> = app_for_state.state();
    let pool = state.db_manager.pool().clone();

    let enabled = crate::database::repositories::setting::SettingsRepository::get_diarization_enabled(&pool)
        .await
        .map_err(|e| format!("Failed to read diarization_enabled setting: {}", e))?;

    if !enabled {
        return Ok(None);
    }

    let max_speakers = crate::database::repositories::setting::SettingsRepository::get_diarization_max_speakers(&pool)
        .await
        .map_err(|e| format!("Failed to read diarization_max_speakers setting: {}", e))?;

    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data directory: {}", e))?;

    match check_status(&base).await {
        DiarizationModelStatus::Available => {}
        other => {
            return Err(format!(
                "Diarization is enabled but models are not ready ({:?}). Please download them from Settings first.",
                other
            ));
        }
    }

    Ok(Some((
        segmentation_model_path(&base).to_string_lossy().into_owned(),
        embedding_model_path(&base).to_string_lossy().into_owned(),
        max_speakers,
    )))
}

/// Extracts one member file from an in-memory `.tar.bz2` archive.
fn extract_archive_member(archive_bytes: &[u8], member_path: &str) -> Result<Vec<u8>, DiarizationModelError> {
    use bzip2::read::BzDecoder;
    use std::io::{Cursor, Read};
    use tar::Archive;

    let decoder = BzDecoder::new(Cursor::new(archive_bytes));
    let mut archive = Archive::new(decoder);
    let entries = archive.entries().map_err(|e| DiarizationModelError::Extract {
        member: member_path.to_string(),
        source: e,
    })?;

    for entry in entries {
        let mut entry = entry.map_err(|e| DiarizationModelError::Extract {
            member: member_path.to_string(),
            source: e,
        })?;
        let path = entry
            .path()
            .map_err(|e| DiarizationModelError::Extract { member: member_path.to_string(), source: e })?
            .to_string_lossy()
            .into_owned();
        if path == member_path {
            let mut contents = Vec::new();
            entry
                .read_to_end(&mut contents)
                .map_err(|e| DiarizationModelError::Extract { member: member_path.to_string(), source: e })?;
            return Ok(contents);
        }
    }

    Err(DiarizationModelError::MemberNotFound(member_path.to_string()))
}
