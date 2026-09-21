// audio/transcription/whisper_provider.rs
//
// Whisper transcription provider implementation.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use std::sync::Arc;

/// Whisper transcription provider (wraps WhisperEngine)
pub struct WhisperProvider {
    engine: Arc<crate::whisper_engine::WhisperEngine>,
}

impl WhisperProvider {
    pub fn new(engine: Arc<crate::whisper_engine::WhisperEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl TranscriptionProvider for WhisperProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
        include_word_timestamps: bool,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        // `previous_text: None` -- this trait-based path has no live caller yet (see
        // `TranscriptionEngine::Provider`, never constructed today), so there's no cross-chunk
        // state to thread through here. Same status as `include_word_timestamps` below.
        match self
            .engine
            .transcribe_audio_with_confidence(audio, language, include_word_timestamps, None)
            .await
        {
            Ok((text, confidence, is_partial, word_timestamps)) => Ok(TranscriptResult {
                text: text.trim().to_string(),
                confidence: Some(confidence),
                is_partial,
                word_timestamps,
            }),
            Err(e) => Err(TranscriptionError::EngineFailed(e.to_string())),
        }
    }

    async fn is_model_loaded(&self) -> bool {
        self.engine.is_model_loaded().await
    }

    async fn get_current_model(&self) -> Option<String> {
        self.engine.get_current_model().await
    }

    fn provider_name(&self) -> &'static str {
        "Whisper"
    }
}
