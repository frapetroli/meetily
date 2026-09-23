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
//! `clustering::cluster_embeddings_spectral_with_p` (see `finalize()`, ADR-0024).

use crate::diarization::clustering::{
    cluster_embeddings_spectral_with_p, normalize_labels, smooth_isolated_segments,
};
use crate::diarization::merge::SpeakerSegment;
use log::info;
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

    /// Dimensionality of the embedding vectors this instance's extractor produces --
    /// varies by embedding model (e.g. eres2net vs campplus vs wespeaker/titanet), never
    /// assumed fixed elsewhere in this module or in `clustering.rs` (cosine similarity is
    /// dimension-agnostic). Mainly a diagnostic for `examples/diarization_calibration.rs`
    /// when comparing embedding models (docs/adr/0030) -- confirms at a glance that a
    /// different `--embedding-model-override` really loaded a different model.
    pub fn embedding_dim(&self) -> i32 {
        self.extractor.dim()
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
        // Split timing: the FFI call below is opaque C++/ONNX Runtime (sherpa-onnx's own
        // segmentation + its internal, otherwise-discarded clustering -- see module docs),
        // outside our control; the embedding loop after it is ours, so separating the two
        // says whether a slow process_chunk() (see docs/sviluppi/diarization/Roadmap e
        // todo.md, voce 9i -- ~36.5s average per 25s window measured, unexplained) is spent
        // inside sherpa-onnx itself or in our own per-segment re-embedding.
        let diarizer_started = std::time::Instant::now();
        let result = self
            .diarizer
            .process(samples)
            .ok_or(DiarizationEngineError::ProcessFailed)?;
        let diarizer_elapsed = diarizer_started.elapsed();

        let segments = result.sort_by_start_time();
        let raw_segment_count = segments.len();
        let mut embedded_count = 0usize;

        let embedding_started = std::time::Instant::now();
        for seg in segments {
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
            embedded_count += 1;
        }
        let embedding_elapsed = embedding_started.elapsed();

        info!(
            "DiarizationEngine::process_chunk(start={:.1}s, {} samples): sherpa-onnx process()={:?} ({} raw segments), embedding loop={:?} ({} embedded)",
            chunk_start_time,
            samples.len(),
            diarizer_elapsed,
            raw_segment_count,
            embedding_elapsed,
            embedded_count
        );

        Ok(())
    }

    /// Cluster everything accumulated across the whole call/import job (one-shot, at
    /// end-of-stream -- ADR-0004/ADR-0009's `diarizing` stage), and return the resulting
    /// speaker-labeled segments sorted by start time. Consumes `self`: a
    /// `DiarizationEngine` is single-use per recording.
    ///
    /// Thin wrapper over `finalize_with_spectral` -- the production entry point called by
    /// all three call sites (session.rs/import.rs/retranscription.rs), `max_speakers`
    /// coming from `transcript_settings.diarization_max_speakers`
    /// (`SettingsRepository::get_diarization_max_speakers`, default
    /// `clustering::DEFAULT_MAX_SPEAKERS`). See `docs/adr/0024-...md`: this replaced a
    /// fixed-threshold agglomerative approach (ADR-0017/ADR-0022, removed once this
    /// superseded it in production) after real-recording calibration showed the
    /// spectral/NME-SC method dramatically closer to ground truth.
    pub fn finalize(self, max_speakers: usize) -> Vec<SpeakerSegment> {
        self.finalize_with_spectral(max_speakers)
    }

    /// Clusters the accumulated embeddings with `clustering::cluster_embeddings_spectral`
    /// (automatic speaker-count estimation via the NME-SC eigengap heuristic), capped at
    /// `max_speakers`. Called by `finalize()` (the production entry point, see its doc
    /// comment and ADR-0024) and by `examples/diarization_calibration.rs` (which calls
    /// `finalize_with_spectral_and_p` directly to also experiment with a fixed `p`).
    ///
    /// No post-hoc cluster-merging pass needed: the eigengap estimate is already bounded
    /// by `max_speakers`.
    pub fn finalize_with_spectral(&self, max_speakers: usize) -> Vec<SpeakerSegment> {
        self.finalize_with_spectral_and_p(max_speakers, None)
    }

    /// Same as `finalize_with_spectral`, but lets the caller override the p-nearest-
    /// neighbor pruning count (see `clustering::cluster_embeddings_spectral_with_p`) --
    /// exists for `examples/diarization_calibration.rs` to experiment with `p` on real
    /// recordings where the default formula under-estimated the speaker count. `None`
    /// reproduces `finalize_with_spectral`'s exact behavior.
    pub fn finalize_with_spectral_and_p(
        &self,
        max_speakers: usize,
        p_override: Option<usize>,
    ) -> Vec<SpeakerSegment> {
        if self.accumulated.is_empty() {
            return Vec::new();
        }

        // Sort by start time up front: clustering itself doesn't care about order, but
        // the post-clustering smoothing pass below (`smooth_isolated_segments`) needs
        // real temporal adjacency, and building the final segment list from this same
        // order means no separate sort is needed afterwards (unlike before this pass
        // was added, where sorting only happened at the very end).
        let mut ordered: Vec<&(Vec<f32>, f64, f64)> = self.accumulated.iter().collect();
        ordered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let embeddings: Vec<Vec<f32>> = ordered.iter().map(|(e, _, _)| e.clone()).collect();
        let starts: Vec<f64> = ordered.iter().map(|(_, start, _)| *start).collect();
        let ends: Vec<f64> = ordered.iter().map(|(_, _, end)| *end).collect();

        let labels = cluster_embeddings_spectral_with_p(&embeddings, max_speakers, p_override);
        // Resegmentation pass (docs/adr/0031): reattaches isolated single-segment
        // "sandwiches" (A, B, A) to the surrounding speaker when the sandwiched segment
        // is short and not confidently a different voice -- see `smooth_isolated_segments`'s
        // doc comment for the full rationale and the literature it's grounded in.
        let labels = smooth_isolated_segments(&embeddings, &starts, &ends, &labels);
        let labels = normalize_labels(&labels);

        starts
            .iter()
            .zip(ends.iter())
            .zip(labels)
            .map(|((start, end), label)| SpeakerSegment {
                start: *start,
                end: *end,
                speaker: format!("speaker_{label}"),
            })
            .collect()
    }
}
