// audio/transcription/provider.rs
//
// Defines the unified TranscriptionProvider trait and common types for all
// transcription engines (Whisper, Parakeet, future providers).

use async_trait::async_trait;
use crate::diarization::WordTiming;

// ============================================================================
// TRANSCRIPTION PROVIDER TRAIT & ERROR TYPES
// ============================================================================

/// Granular error types for transcription operations
#[derive(Debug, Clone)]
pub enum TranscriptionError {
    ModelNotLoaded,
    AudioTooShort { samples: usize, minimum: usize },
    EngineFailed(String),
    UnsupportedLanguage(String),
}

impl std::fmt::Display for TranscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotLoaded => write!(f, "No transcription model is loaded"),
            Self::AudioTooShort { samples, minimum } => write!(
                f,
                "Audio too short: {} samples (minimum {})",
                samples, minimum
            ),
            Self::EngineFailed(msg) => write!(f, "Transcription engine failed: {}", msg),
            Self::UnsupportedLanguage(lang) => {
                write!(f, "Language '{}' is not supported by this provider", lang)
            }
        }
    }
}

impl std::error::Error for TranscriptionError {}

/// Unified transcription result across all providers
#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub text: String,
    pub confidence: Option<f32>, // None if provider doesn't support confidence scores
    pub is_partial: bool,
    /// Per-word timestamps, populated only when `transcribe()` was called with
    /// `include_word_timestamps: true` (diarization toggle on -- ADR-0013). `None`
    /// otherwise, including for providers that don't support word timestamps at all.
    pub word_timestamps: Option<Vec<WordTiming>>,
}

/// Trait for transcription providers (Whisper, Parakeet, future providers)
#[async_trait]
pub trait TranscriptionProvider: Send + Sync {
    /// Transcribe audio samples to text
    ///
    /// # Arguments
    /// * `audio` - Audio samples (16kHz mono, f32 format)
    /// * `language` - Optional language hint (e.g., "en", "es", "fr")
    /// * `include_word_timestamps` - gates per-word timestamp extraction (ADR-0013);
    ///   `false` on the default (diarization-off) hot path, no behavior/cost change from
    ///   before this parameter existed. Providers that can't support this simply ignore
    ///   it and always return `word_timestamps: None`.
    ///
    /// # Returns
    /// * `TranscriptResult` with text, optional confidence, and partial flag
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
        include_word_timestamps: bool,
    ) -> std::result::Result<TranscriptResult, TranscriptionError>;

    /// Check if a model is currently loaded
    async fn is_model_loaded(&self) -> bool;

    /// Get the name of the currently loaded model
    async fn get_current_model(&self) -> Option<String>;

    /// Get the provider name (for logging/debugging)
    fn provider_name(&self) -> &'static str;
}
