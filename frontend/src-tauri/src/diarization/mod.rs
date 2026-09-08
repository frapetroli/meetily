//! Speaker diarization module (CPU-only, sherpa-onnx based).
//!
//! Adds speaker labels to Meetily's existing transcription, without introducing a new
//! ASR engine: the existing Whisper-rs/Parakeet transcription (see `audio::transcription`)
//! keeps running unchanged, and this module runs an additional, independent pipeline on the
//! same mixed audio to determine *who* is speaking, merged with the transcript at the end
//! of a call/import (see `merge.rs`).
//!
//! Opt-in, gated by `transcript_settings.diarization_enabled` (default off) -- see
//! `docs/adr/0010-diarization-opt-in-toggle-globale.md` in the docs workspace.
//!
//! # Module structure
//!
//! - `model`: model catalog (segmentation + embedding), download/status management
//! - `engine`: per-chunk segmentation + speaker embedding extraction (`DiarizationEngine`)
//! - `clustering`: final speaker clustering across all accumulated embeddings
//! - `merge`: overlap-based assignment of words to speaker segments (ADR-0003)
//!
//! # Why clustering is reimplemented instead of using sherpa-onnx's `FastClustering`
//!
//! sherpa-onnx's official Rust crate (and its C/C++ API) only exposes clustering as part
//! of the full `OfflineSpeakerDiarization::process()` pipeline (segmentation + embedding +
//! clustering together on a complete waveform) -- there is no standalone, reusable
//! clustering entry point in any public API (only Python's internal pybind11 binding has
//! one). See `docs/adr/0017-clustering-diarization-reimplementato-non-fastclustering-sherpa-onnx.md`.

pub mod clustering;
pub mod commands;
pub mod engine;
pub mod merge;
pub mod model;
pub mod session;

pub use clustering::cluster_embeddings;
pub use commands::{diarization_download_models, diarization_get_status};
pub use engine::{DiarizationEngine, DiarizationEngineError};
pub use merge::{
    assign_word_speakers, clean_speaker_turns, format_turns_as_text, group_into_speaker_turns,
    SpeakerSegment, SpeakerTurn, WordTiming, WordWithSpeaker,
};
pub use model::{DiarizationModelStatus, DIARIZATION_MODEL_CATALOG};
pub use session::DiarizationSession;
