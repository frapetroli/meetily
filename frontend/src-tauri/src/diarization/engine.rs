//! Per-chunk segmentation + speaker embedding extraction, using the official `sherpa-onnx`
//! Rust crate (ADR-0016).
//!
//! Two sherpa-onnx primitives are combined here, neither of which alone does the whole
//! job (validated empirically with a standalone prototype against the project's own
//! spike audio/models -- see ADR-0017 and the accompanying conversation notes):
//!
//! 1. `sherpa_onnx::OfflineSpeakerDiarization::process()` is used to obtain segment
//!    boundaries (`start`/`end`) for the chunk. Its own internal `speaker` label is
//!    *not* discarded outright any more (ADR-0032) -- it's used to merge adjacent
//!    segments sherpa-onnx already called the same speaker before re-embedding
//!    (`merge_adjacent_same_speaker`), then dropped: it is local to this one call and
//!    never compared across chunks (sherpa-onnx exposes no standalone segmentation-only
//!    API).
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

/// Maximum time gap (seconds) between two adjacent segments -- already assigned to the
/// same chunk-local speaker by sherpa-onnx's own segmentation+clustering -- for them to
/// still be merged into one span before re-embedding (see `merge_adjacent_same_speaker`).
/// Kept small deliberately: merging across a *large* gap would pull unrelated/silent
/// audio into the merged span and degrade the resulting embedding instead of stabilizing
/// it -- this closes micro-pauses/prosodic breaks within one continuous utterance, not
/// real silence between turns.
pub const MAX_MERGE_GAP_SECS: f32 = 0.5;

/// Merges adjacent `(start, end, local_speaker)` spans that share the same
/// `local_speaker` id and are close enough in time (`<= MAX_MERGE_GAP_SECS` apart) into
/// a single wider span, dropping the (now no-longer-needed) local speaker id. Input must
/// already be sorted by `start` (same order
/// `OfflineSpeakerDiarizationResult::sort_by_start_time()` returns).
///
/// Exists because sherpa-onnx's segmentation step often splits one continuous utterance
/// from a single speaker into several short adjacent segments (micro-pauses, prosodic
/// variation -- not itself a bug). Re-embedding each of those short pieces
/// *independently*, as this module did before this function existed, produces several
/// noisier embeddings instead of one stable one -- any of which can drift close enough
/// to a *different* speaker's cluster to get misassigned, producing sustained
/// alternation between two labels for what is really one continuous speaker turn
/// (reported directly against real app output, see docs/adr/0032-... in the docs
/// workspace). Merging first, using the *local* speaker id sherpa-onnx's own
/// segmentation+clustering already computed for this one chunk (valid only within this
/// call, never compared across chunks -- see module docs), avoids that without
/// weakening pyannote-segmentation's own validated ability to detect a genuine speaker
/// change with zero pause (ADR-0007's spike): two adjacent segments are only merged
/// when sherpa-onnx itself already called them the same speaker, never merely because
/// they're close together in time.
///
/// Pure function, no FFI -- takes plain tuples rather than
/// `OfflineSpeakerDiarizationSegment` directly so it's testable without the sherpa-onnx
/// crate.
pub fn merge_adjacent_same_speaker(segments: &[(f32, f32, i32)]) -> Vec<(f32, f32)> {
    let mut merged: Vec<(f32, f32, i32)> = Vec::new();
    for &(start, end, speaker) in segments {
        match merged.last_mut() {
            Some(last) if last.2 == speaker && start - last.1 <= MAX_MERGE_GAP_SECS => {
                last.1 = end;
            }
            _ => merged.push((start, end, speaker)),
        }
    }
    merged.into_iter().map(|(s, e, _)| (s, e)).collect()
}

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
            // Used to shape the boundaries we get back and, since ADR-0032, to merge
            // adjacent same-(chunk-)speaker segments before re-embedding -- see
            // `merge_adjacent_same_speaker`'s doc comment and the module docs.
            // num_clusters=-1 (auto) is fine here: validated empirically that boundary
            // quality doesn't depend on knowing the true speaker count in advance.
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

        // Merge adjacent segments sherpa-onnx's own chunk-local segmentation+clustering
        // already assigned to the same speaker, before re-embedding (ADR-0032) -- see
        // `merge_adjacent_same_speaker`'s doc comment for the full rationale.
        let merge_input: Vec<(f32, f32, i32)> =
            segments.iter().map(|s| (s.start, s.end, s.speaker)).collect();
        let merged_segments = merge_adjacent_same_speaker(&merge_input);
        let merged_segment_count = merged_segments.len();
        let mut embedded_count = 0usize;

        let embedding_started = std::time::Instant::now();
        for (seg_start, seg_end) in merged_segments {
            if seg_end - seg_start < MIN_SEGMENT_DURATION_SECS {
                continue;
            }
            let start_idx = (seg_start * self.sample_rate as f32) as usize;
            let end_idx = ((seg_end * self.sample_rate as f32) as usize).min(samples.len());
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
                chunk_start_time + seg_start as f64,
                chunk_start_time + seg_end as f64,
            ));
            embedded_count += 1;
        }
        let embedding_elapsed = embedding_started.elapsed();

        info!(
            "DiarizationEngine::process_chunk(start={:.1}s, {} samples): sherpa-onnx process()={:?} ({} raw segments, {} after same-speaker merge), embedding loop={:?} ({} embedded)",
            chunk_start_time,
            samples.len(),
            diarizer_elapsed,
            raw_segment_count,
            merged_segment_count,
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

#[cfg(test)]
mod tests {
    use super::*;

    // -- merge_adjacent_same_speaker (ADR-0032) --------------------------------------

    #[test]
    fn merge_adjacent_same_speaker_merges_close_same_speaker_segments() {
        // Same local speaker (0), gap of 0.2s (<= MAX_MERGE_GAP_SECS) -- merged into
        // one span from the first start to the second end.
        let segments = vec![(0.0f32, 4.0, 0), (4.2, 6.0, 0)];
        let merged = merge_adjacent_same_speaker(&segments);
        assert_eq!(merged, vec![(0.0, 6.0)]);
    }

    #[test]
    fn merge_adjacent_same_speaker_does_not_merge_different_speakers() {
        // Different local speakers, small gap -- sherpa-onnx itself already called
        // these different people, so they must stay separate regardless of the gap.
        let segments = vec![(0.0f32, 4.0, 0), (4.2, 6.0, 1)];
        let merged = merge_adjacent_same_speaker(&segments);
        assert_eq!(merged, vec![(0.0, 4.0), (4.2, 6.0)]);
    }

    #[test]
    fn merge_adjacent_same_speaker_does_not_merge_across_a_large_gap() {
        // Same local speaker, but the gap (5.0s) exceeds MAX_MERGE_GAP_SECS (0.5s) --
        // merging would pull unrelated/silent audio into the span, so these stay
        // separate even though sherpa-onnx called them the same speaker.
        let segments = vec![(0.0f32, 4.0, 0), (9.0, 11.0, 0)];
        let merged = merge_adjacent_same_speaker(&segments);
        assert_eq!(merged, vec![(0.0, 4.0), (9.0, 11.0)]);
    }

    #[test]
    fn merge_adjacent_same_speaker_merges_a_chain_of_three() {
        let segments = vec![(0.0f32, 2.0, 0), (2.1, 4.0, 0), (4.3, 6.0, 0)];
        let merged = merge_adjacent_same_speaker(&segments);
        assert_eq!(merged, vec![(0.0, 6.0)]);
    }

    #[test]
    fn merge_adjacent_same_speaker_keeps_a_sandwiched_different_speaker_separate() {
        // A, B, A all close together in time -- the middle B must not get bridged over,
        // and the two A segments (not adjacent to each other) must not merge either.
        let segments = vec![(0.0f32, 2.0, 0), (2.1, 3.0, 1), (3.1, 5.0, 0)];
        let merged = merge_adjacent_same_speaker(&segments);
        assert_eq!(merged, vec![(0.0, 2.0), (2.1, 3.0), (3.1, 5.0)]);
    }

    #[test]
    fn merge_adjacent_same_speaker_handles_single_segment_and_empty_input() {
        assert_eq!(merge_adjacent_same_speaker(&[]), Vec::<(f32, f32)>::new());
        assert_eq!(
            merge_adjacent_same_speaker(&[(1.0, 2.0, 0)]),
            vec![(1.0, 2.0)]
        );
    }
}
