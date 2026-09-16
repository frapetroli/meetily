//! Orchestrates a `DiarizationEngine` across the lifetime of one recording/import job --
//! analogous to `audio::pipeline::AudioPipelineManager`, but for the diarization side
//! branch (ADR-0009): receives the same pre-VAD mixed-audio tap already sent to
//! `RecordingSaver`, accumulates it into ~25s windows (sherpa-onnx's segmentation API
//! works on a whole buffer per call, not a true streaming API -- see `engine.rs`), and
//! feeds those windows to `DiarizationEngine::process_chunk` as they fill up. Only
//! created when `transcript_settings.diarization_enabled` is on.

use crate::audio::recording_state::AudioChunk;
use crate::diarization::engine::{DiarizationEngine, DiarizationEngineError};
use crate::diarization::merge::SpeakerSegment;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Chunks are accumulated until they reach this many seconds before being handed to
/// `DiarizationEngine::process_chunk` -- same ~20-30s cadence already decided in
/// `docs/sviluppi/diarization/Architettura pipeline.md` (independent of the ASR chunking,
/// which is VAD-driven, not time-driven). The batch paths (`audio::import`/
/// `audio::retranscription`) deliberately do NOT mirror this chunking -- tried it
/// (commit 008e42a), measured it 2.3-2.5x *slower* on a real long file, reverted --
/// see `docs/sviluppi/diarization/Roadmap e todo.md`, voce 9i.
const WINDOW_SECONDS: f64 = 25.0;

pub struct DiarizationSession {
    /// Clone of this and hand it to `AudioPipelineManager::start()` as the third
    /// consumer of the mixed-audio tap (alongside VAD and `RecordingSaver`).
    sender: mpsc::UnboundedSender<AudioChunk>,
    task_handle: JoinHandle<Vec<SpeakerSegment>>,
}

impl DiarizationSession {
    pub fn start(
        segmentation_model_path: String,
        embedding_model_path: String,
        num_threads: i32,
        max_speakers: usize,
    ) -> Result<Self, DiarizationEngineError> {
        let mut engine = DiarizationEngine::new(&segmentation_model_path, &embedding_model_path, num_threads)?;
        let expected_sample_rate = engine.sample_rate();
        let (sender, mut receiver) = mpsc::unbounded_channel::<AudioChunk>();

        let task_handle = tokio::spawn(async move {
            let mut window: Vec<f32> = Vec::new();
            let mut window_start_time: Option<f64> = None;

            while let Some(chunk) = receiver.recv().await {
                // Same flush-signal convention as AudioPipeline's own flush chunks
                // (chunk_id near u64::MAX, empty data) -- not real audio, skip.
                if chunk.chunk_id >= u64::MAX - 10 {
                    continue;
                }
                if chunk.data.is_empty() {
                    continue;
                }

                if window_start_time.is_none() {
                    window_start_time = Some(chunk.timestamp);
                }

                if chunk.sample_rate as i32 != expected_sample_rate {
                    let resampled = crate::audio::audio_processing::resample_audio(
                        &chunk.data,
                        chunk.sample_rate,
                        expected_sample_rate as u32,
                    );
                    window.extend_from_slice(&resampled);
                } else {
                    window.extend_from_slice(&chunk.data);
                }

                let window_duration = window.len() as f64 / expected_sample_rate as f64;
                if window_duration >= WINDOW_SECONDS {
                    if let Some(start) = window_start_time {
                        // `process_chunk` runs real ONNX inference (segmentation +
                        // embedding extraction) synchronously -- CPU-bound, no `.await`
                        // points. Calling it directly here would block this tokio worker
                        // thread for however long that inference takes, once per window,
                        // for the whole call: on CPU-only hardware this starves the
                        // runtime's scheduler over a multi-minute recording (observed as
                        // live transcription/diarization silently falling behind and the
                        // saved result only covering the first few minutes). The batch
                        // path (import.rs/retranscription.rs) already avoids this via
                        // spawn_blocking; mirror that here. Ownership of `engine` and
                        // `window` is threaded through spawn_blocking and handed back so
                        // the loop can keep reusing them across windows.
                        let (result, returned_engine, returned_window) =
                            tokio::task::spawn_blocking(move || {
                                let result = engine.process_chunk(&window, start);
                                (result, engine, window)
                            })
                            .await
                            .expect("diarization process_chunk blocking task panicked");
                        engine = returned_engine;
                        window = returned_window;
                        if let Err(e) = result {
                            log::warn!("Diarization: process_chunk failed on a window: {}", e);
                        }
                    }
                    window.clear();
                    window_start_time = None;
                }
            }

            // Flush the last partial window (recording ended mid-window).
            if !window.is_empty() {
                if let Some(start) = window_start_time {
                    let (result, returned_engine, _window) =
                        tokio::task::spawn_blocking(move || {
                            let result = engine.process_chunk(&window, start);
                            (result, engine, window)
                        })
                        .await
                        .expect("diarization final process_chunk blocking task panicked");
                    engine = returned_engine;
                    if let Err(e) = result {
                        log::warn!("Diarization: final process_chunk failed: {}", e);
                    }
                }
            }

            // `finalize()` runs the final clustering pass -- not ONNX inference, but
            // still synchronous CPU work over all accumulated embeddings; spawn_blocking
            // for the same reason as process_chunk above. `max_speakers` needs no
            // round-trip (unlike `engine`/`window` above): nothing downstream needs it
            // back, so a plain `move` capture is enough.
            tokio::task::spawn_blocking(move || engine.finalize(max_speakers))
                .await
                .expect("diarization finalize blocking task panicked")
        });

        Ok(Self { sender, task_handle })
    }

    /// Clone to pass to `AudioPipelineManager::start()`'s new `diarization_sender` parameter.
    pub fn sender(&self) -> mpsc::UnboundedSender<AudioChunk> {
        self.sender.clone()
    }

    /// Signals end-of-stream (closes the channel, which ends the receive loop) and waits
    /// for the final clustering + returns the speaker-labeled segments. Called from the
    /// `diarizing` stage of `stop_recording` (or the batch equivalent for import/
    /// retranscription).
    pub async fn finish(self) -> Vec<SpeakerSegment> {
        drop(self.sender);
        self.task_handle.await.unwrap_or_else(|e| {
            log::error!("Diarization session task panicked: {}", e);
            Vec::new()
        })
    }

    /// Reads `transcript_settings.diarization_enabled` (ADR-0010) and, if on, validates
    /// the models are downloaded (roadmap 6h: same gate pattern as
    /// `validate_transcription_model_ready`/`check_can_record` for Parakeet -- recording
    /// start is blocked, not silently skipped, if the toggle is on but models are
    /// missing) and starts a session. Returns `Ok(None)` when the toggle is off --
    /// callers pass that straight through to `RecordingManager::start_recording` as
    /// `None`, for zero overhead.
    pub async fn prepare_if_enabled<R: tauri::Runtime>(
        app: &tauri::AppHandle<R>,
    ) -> Result<Option<Self>, String> {
        let paths = crate::diarization::model::resolve_paths_if_enabled(app).await?;
        let Some((segmentation_model_path, embedding_model_path, max_speakers)) = paths else {
            return Ok(None);
        };

        let session = Self::start(segmentation_model_path, embedding_model_path, 1, max_speakers)
            .map_err(|e| format!("Failed to initialize diarization engine: {}", e))?;

        Ok(Some(session))
    }
}
