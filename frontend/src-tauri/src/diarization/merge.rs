//! Overlap-based assignment of transcript words to speaker segments (ADR-0003,
//! `assign_word_speakers` style, as used by WhisperX). Pure functions, no I/O -- the
//! word/timestamp side comes from the already-active ASR engine (Whisper-rs or Parakeet,
//! extended per ADR-0013 to propagate per-word timestamps), the speaker-segment side
//! comes from `diarization::clustering::cluster_embeddings` grouped back into
//! `(start, end, speaker)` spans by `engine.rs`.
//!
//! Must run on the *pre-cleanup* word list (before `clean_repetitive_text()` rewrites the
//! text), so the word<->timestamp alignment used here isn't broken by that later pass --
//! see ADR-0013.

use serde::{Deserialize, Serialize};

/// One transcribed word with its absolute timestamp in the recording (seconds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTiming {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// A word after speaker assignment. `speaker` is `None` only if no speaker segment
/// exists at all for the recording (e.g. diarization produced nothing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordWithSpeaker {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub speaker: Option<String>,
}

/// One continuous speaker turn, produced by grouping consecutive same-speaker words.
/// This is the granularity actually persisted to `transcripts` (one row per turn, not
/// per chunk/word) -- see ADR-0009.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub speaker: Option<String>,
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// A speaker-labeled time span produced by the diarization clustering step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerSegment {
    pub start: f64,
    pub end: f64,
    pub speaker: String,
}

fn overlap(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

fn gap(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    if a_end <= b_start {
        b_start - a_end
    } else if b_end <= a_start {
        a_start - b_end
    } else {
        0.0 // overlapping, handled separately
    }
}

/// Assign each word the speaker of the segment it overlaps most with. If a word overlaps
/// no segment (gap in diarization, e.g. very short interjection dropped by the min-duration
/// filter in `engine.rs`), fall back to the nearest segment by time gap. Returns `None` only
/// when `segments` is empty.
pub fn assign_word_speakers(words: &[WordTiming], segments: &[SpeakerSegment]) -> Vec<WordWithSpeaker> {
    words
        .iter()
        .map(|w| {
            let speaker = if segments.is_empty() {
                None
            } else {
                let mut best_overlap = 0.0f64;
                let mut best_idx: Option<usize> = None;
                for (i, s) in segments.iter().enumerate() {
                    let ov = overlap(w.start, w.end, s.start, s.end);
                    if ov > best_overlap {
                        best_overlap = ov;
                        best_idx = Some(i);
                    }
                }
                let idx = best_idx.unwrap_or_else(|| {
                    // No overlap at all: fall back to the segment with the smallest gap.
                    segments
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            gap(w.start, w.end, a.start, a.end)
                                .partial_cmp(&gap(w.start, w.end, b.start, b.end))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                });
                Some(segments[idx].speaker.clone())
            };
            WordWithSpeaker {
                word: w.word.clone(),
                start: w.start,
                end: w.end,
                speaker,
            }
        })
        .collect()
}

/// Collapse consecutive words with the same speaker into turns. This is what actually
/// gets persisted per-row in `transcripts` (ADR-0009: "granularità per turno di speaker,
/// non per chunk VAD").
pub fn group_into_speaker_turns(words: &[WordWithSpeaker]) -> Vec<SpeakerTurn> {
    let mut turns: Vec<SpeakerTurn> = Vec::new();
    for w in words {
        match turns.last_mut() {
            Some(turn) if turn.speaker == w.speaker => {
                turn.text.push(' ');
                turn.text.push_str(&w.word);
                turn.end = w.end;
            }
            _ => {
                turns.push(SpeakerTurn {
                    speaker: w.speaker.clone(),
                    text: w.word.clone(),
                    start: w.start,
                    end: w.end,
                });
            }
        }
    }
    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(w: &str, start: f64, end: f64) -> WordTiming {
        WordTiming { word: w.to_string(), start, end }
    }

    fn span(start: f64, end: f64, speaker: &str) -> SpeakerSegment {
        SpeakerSegment { start, end, speaker: speaker.to_string() }
    }

    #[test]
    fn clean_overlap_assigns_the_containing_speaker() {
        let words = vec![word("hello", 1.0, 1.5), word("world", 5.0, 5.5)];
        let segments = vec![span(0.0, 3.0, "speaker_0"), span(3.0, 8.0, "speaker_1")];
        let result = assign_word_speakers(&words, &segments);
        assert_eq!(result[0].speaker.as_deref(), Some("speaker_0"));
        assert_eq!(result[1].speaker.as_deref(), Some("speaker_1"));
    }

    #[test]
    fn ambiguous_overlap_picks_the_larger_share() {
        // Word spans the boundary between two segments, but overlaps speaker_1 more.
        let words = vec![word("transition", 2.8, 3.4)];
        let segments = vec![span(0.0, 3.0, "speaker_0"), span(3.0, 6.0, "speaker_1")];
        let result = assign_word_speakers(&words, &segments);
        // Overlap with speaker_0: 0.2s (2.8-3.0); with speaker_1: 0.4s (3.0-3.4).
        assert_eq!(result[0].speaker.as_deref(), Some("speaker_1"));
    }

    #[test]
    fn word_with_no_overlapping_segment_falls_back_to_nearest() {
        // Gap in diarization between 3.0 and 4.0; word falls entirely in the gap but
        // closer to speaker_1's segment.
        let words = vec![word("uh", 3.8, 3.9)];
        let segments = vec![span(0.0, 3.0, "speaker_0"), span(4.0, 6.0, "speaker_1")];
        let result = assign_word_speakers(&words, &segments);
        assert_eq!(result[0].speaker.as_deref(), Some("speaker_1"));
    }

    #[test]
    fn no_segments_at_all_leaves_speaker_none() {
        let words = vec![word("hello", 1.0, 1.5)];
        let result = assign_word_speakers(&words, &[]);
        assert_eq!(result[0].speaker, None);
    }

    #[test]
    fn grouping_collapses_consecutive_same_speaker_words_into_one_turn() {
        let words = vec![
            WordWithSpeaker { word: "hello".into(), start: 0.0, end: 0.5, speaker: Some("speaker_0".into()) },
            WordWithSpeaker { word: "there".into(), start: 0.5, end: 1.0, speaker: Some("speaker_0".into()) },
            WordWithSpeaker { word: "hi".into(), start: 1.2, end: 1.5, speaker: Some("speaker_1".into()) },
            WordWithSpeaker { word: "again".into(), start: 3.0, end: 3.4, speaker: Some("speaker_0".into()) },
        ];
        let turns = group_into_speaker_turns(&words);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].text, "hello there");
        assert_eq!(turns[0].speaker.as_deref(), Some("speaker_0"));
        assert_eq!(turns[0].start, 0.0);
        assert_eq!(turns[0].end, 1.0);
        assert_eq!(turns[1].text, "hi");
        assert_eq!(turns[2].text, "again");
        assert_eq!(turns[2].speaker.as_deref(), Some("speaker_0"));
    }
}
