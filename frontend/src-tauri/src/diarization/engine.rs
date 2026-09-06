//! Per-chunk segmentation + speaker embedding extraction, using the official `sherpa-onnx`
//! Rust crate (ADR-0016).
//!
//! Two sherpa-onnx primitives are combined here, neither of which alone does the whole
//! job (validated empirically with a standalone prototype against the project's own
//! spike audio/models -- see ADR-0017 and the accompanying conversation notes):
//!
//! 1. `sherpa_onnx::OfflineSpeakerDiarization::process()` is used **only** to obtain
//!    segment boundaries (`start`/`end`) for the chunk -- its own internal `speaker`
//!    label is discarded, because it is local to this one call and not comparable
//!    across chunks (sherpa-onnx exposes no standalone segmentation-only API).
//! 2. Each returned segment is **re-embedded independently** via
//!    `SpeakerEmbeddingExtractor`/`OnlineStream` (the only standalone embedding API --
//!    `OfflineSpeakerDiarizationResult` does not expose the raw embedding vectors it
//!    computes internally), producing a portable embedding that can be compared across
//!    chunks/the whole call.
//!
//! Final speaker identity is resolved once, across every chunk of the call, by
//! `clustering::cluster_embeddings` (see `finalize()`).

use crate::diarization::clustering::{cluster_embeddings, DEFAULT_CLUSTERING_THRESHOLD};
use crate::diarization::merge::SpeakerSegment;
use sherpa_onnx::{
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
};
use thiserror::Error;

/// Segments shorter than this are dropped before embedding -- empirically, embeddings
/// computed from sub-second segments are too noisy to cluster reliably (validated
/// against real audio: two ~0.7s segments in one test file had near-zero similarity to
/// every other segment, including their own true speaker).
pub const MIN_SEGMENT_DURATION_SECS: f32 = 1.0;

#[derive(Debug, Error)]
pub enum DiarizationEngineError {
    #[error("failed to create sherpa-onnx OfflineSpeakerDiarization (check segmentation model path)")]
    SegmentationInit,
    #[error("failed to create sherpa-onnx SpeakerEmbeddingExtractor (check embedding model path)")]
    EmbeddingInit,
    #[error("failed to create audio stream for embedding extraction")]
    StreamCreate,
    #[error("diarization process() call failed on this chunk")]
    ProcessFailed,
}

/// Accumulates (embedding, absolute_start, absolute_end) across every chunk of a call,
/// then clusters everything once at the end (`finalize()`). One instance per
/// recording/import job -- not reused across calls.
pub struct DiarizationEngine {
    diarizer: OfflineSpeakerDiarization,
    extractor: SpeakerEmbeddingExtractor,
    sample_rate: i32,
    accumulated: Vec<(Vec<f32>, f64, f64)>,
}

impl DiarizationEngine {
    /// `segmentation_model_path`/`embedding_model_path` point at the two on-disk ONNX
    /// files downloaded by `model.rs` (pyannote-segmentation-3.0 + 3D-Speaker embedding).
    pub fn new(
        segmentation_model_path: &str,
        embedding_model_path: &str,
        num_threads: i32,
    ) -> Result<Self, DiarizationEngineError> {
        let diar_config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(segmentation_model_path.to_string()),
                    ..Default::default()
                },
                num_threads,
                ..Default::default()
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(embedding_model_path.to_string()),
                num_threads,
                ..Default::default()
            },
            // Only used to shape the boundaries we get back -- the resulting `speaker`
            // label on each segment is discarded (see module docs). num_clusters=-1
            // (auto) is fine here: validated empirically that boundary quality doesn't
            // depend on knowing the true speaker count in advance.
            ..Default::default()
        };
        let diarizer = OfflineSpeakerDiarization::create(&diar_config)
            .ok_or(DiarizationEngineError::SegmentationInit)?;
        let sample_rate = diarizer.sample_rate();

        let embedding_config = SpeakerEmbeddingExtractorConfig {
            model: Some(embedding_model_path.to_string()),
            num_threads,
            ..Default::default()
        };
        let extractor = SpeakerEmbeddingExtractor::create(&embedding_config)
            .ok_or(DiarizationEngineError::EmbeddingInit)?;

        Ok(Self {
            diarizer,
            extractor,
            sample_rate,
            accumulated: Vec::new(),
        })
    }

    /// Sample rate expected by the segmentation model (resample audio to this before
    /// calling `process_chunk`, same contract as Whisper-rs/Parakeet's 16kHz mono input).
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }

    /// Process one ~20-30s chunk of mixed audio (mic+system, same pre-VAD tap as
    /// `RecordingSaver` -- ADR-0009/ADR-0015). `chunk_start_time` is the chunk's offset
    /// in seconds from the start of the recording, used to convert the chunk-relative
    /// segment boundaries sherpa-onnx returns into absolute recording timestamps.
    ///
    /// Chunking (not processing the whole growing buffer) keeps this incremental and
    /// avoids quadratic cost as the call gets longer -- see ADR-0004.
    pub fn process_chunk(
        &mut self,
        samples: &[f32],
        chunk_start_time: f64,
    ) -> Result<(), DiarizationEngineError> {
        let result = self
            .diarizer
            .process(samples)
            .ok_or(DiarizationEngineError::ProcessFailed)?;

        for seg in result.sort_by_start_time() {
            if seg.end - seg.start < MIN_SEGMENT_DURATION_SECS {
                continue;
            }
            let start_idx = (seg.start * self.sample_rate as f32) as usize;
            let end_idx = ((seg.end * self.sample_rate as f32) as usize).min(samples.len());
            if end_idx <= start_idx {
                continue;
            }

            let stream = self
                .extractor
                .create_stream()
                .ok_or(DiarizationEngineError::StreamCreate)?;
            stream.accept_waveform(self.sample_rate, &samples[start_idx..end_idx]);
            stream.input_finished();
            if !self.extractor.is_ready(&stream) {
                continue; // segment too short for the embedding model itself; skip
            }
            let Some(embedding) = self.extractor.compute(&stream) else {
                continue;
            };

            self.accumulated.push((
                embedding,
                chunk_start_time + seg.start as f64,
                chunk_start_time + seg.end as f64,
            ));
        }

        Ok(())
    }

    /// Cluster everything accumulated across the whole call/import job (one-shot, at
    /// end-of-stream -- ADR-0004/ADR-0009's `diarizing` stage), and return the resulting
    /// speaker-labeled segments sorted by start time. Consumes `self`: a
    /// `DiarizationEngine` is single-use per recording.
    pub fn finalize(self) -> Vec<SpeakerSegment> {
        if self.accumulated.is_empty() {
            return Vec::new();
        }

        let embeddings: Vec<Vec<f32>> = self.accumulated.iter().map(|(e, _, _)| e.clone()).collect();
        let labels = cluster_embeddings(&embeddings, DEFAULT_CLUSTERING_THRESHOLD);

        let mut segments: Vec<SpeakerSegment> = self
            .accumulated
            .into_iter()
            .zip(labels)
            .map(|((_, start, end), label)| SpeakerSegment {
                start,
                end,
                speaker: format!("speaker_{label}"),
            })
            .collect();
        segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
        segments
    }
}
