//! Overlap-based assignment of transcript words to speaker segments (ADR-0003,
//! `assign_word_speakers` style, as used by WhisperX). Pure functions, no I/O -- the
//! word/timestamp side comes from the already-active ASR engine (Whisper-rs or Parakeet,
//! extended per ADR-0013 to propagate per-word timestamps), the speaker-segment side
//! comes from `diarization::clustering`'s spectral clustering (ADR-0024) grouped back
//! into `(start, end, speaker)` spans by `engine.rs`.
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

/// Removes Whisper-rs hallucination/repetition artifacts from each turn's text (roadmap
/// item 2e), and re-merges consecutive turns left adjacent by a dropped turn back into
/// one. Call this right after `group_into_speaker_turns()`, before the result is saved
/// or attached to any payload.
///
/// `whisper_engine.rs::clean_repetitive_text()` already does this same cleanup for the
/// live/non-diarized text, but deliberately runs on data *before* diarization's
/// word-level merge (see the comment at its call site) -- the per-word timestamps used
/// here are intentionally the raw, uncleaned ones, so hallucinated/repeated words can
/// still reach this point. Turn granularity is exactly where it's safe to clean: a
/// turn's `start`/`end` stay valid no matter what happens to its `text`, unlike at word
/// level, where cleaning would require deciding which duplicate word's timestamp
/// survives being collapsed into one.
///
/// Reimplements the same three-pass logic as `whisper_engine.rs::clean_repetitive_text`
/// (word-repetition removal, phrase-repetition removal, meaningless-output/ratio
/// filtering) as free functions here, rather than reusing that one, to avoid touching
/// `whisper_engine.rs` at all -- it runs unconditionally for every Whisper-rs
/// transcription, diarization on or off, so any change there carries a much wider blast
/// radius than this diarization-only post-processing step.
pub fn clean_speaker_turns(turns: Vec<SpeakerTurn>) -> Vec<SpeakerTurn> {
    let cleaned: Vec<SpeakerTurn> = turns
        .into_iter()
        .filter_map(|turn| {
            let cleaned_text = clean_repetitive_text(&turn.text);
            if cleaned_text.is_empty() {
                None
            } else {
                Some(SpeakerTurn { text: cleaned_text, ..turn })
            }
        })
        .collect();

    // Re-merge turns that are now adjacent (same speaker) because a turn between them
    // got dropped above, or that were already adjacent -- cosmetic, not required for
    // correctness (each turn's speaker/boundaries stay individually valid either way).
    let mut merged: Vec<SpeakerTurn> = Vec::new();
    for turn in cleaned {
        match merged.last_mut() {
            Some(prev) if prev.speaker == turn.speaker => {
                prev.text.push(' ');
                prev.text.push_str(&turn.text);
                prev.end = turn.end;
            }
            _ => merged.push(turn),
        }
    }
    merged
}

fn is_meaningless_output(text: &str) -> bool {
    let text_lower = text.to_lowercase();
    let meaningless_patterns = [
        "thank you for watching",
        "thanks for watching",
        "like and subscribe",
        "music playing",
        "applause",
        "laughter",
        "um um um",
        "uh uh uh",
        "ah ah ah",
    ];
    if meaningless_patterns.iter().any(|p| text_lower.contains(p)) {
        return true;
    }
    let unique_chars: std::collections::HashSet<char> = text.chars().collect();
    unique_chars.len() <= 3 && text.len() > 10
}

fn remove_word_repetitions<'a>(words: &'a [&'a str]) -> Vec<&'a str> {
    let mut cleaned_words = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let current_word = words[i];
        let mut repeat_count = 1;
        while i + repeat_count < words.len() && words[i + repeat_count] == current_word {
            repeat_count += 1;
        }
        cleaned_words.push(current_word);
        i += repeat_count;
    }
    cleaned_words
}

fn remove_phrase_repetitions<'a>(words: &'a [&'a str]) -> Vec<&'a str> {
    if words.len() < 4 {
        return words.to_vec();
    }
    let mut final_words = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let mut phrase_found = false;
        for phrase_len in 2..=std::cmp::min(5, (words.len() - i) / 2) {
            if i + phrase_len * 2 <= words.len() {
                let phrase1 = &words[i..i + phrase_len];
                let phrase2 = &words[i + phrase_len..i + phrase_len * 2];
                if phrase1 == phrase2 {
                    final_words.extend_from_slice(phrase1);
                    i += phrase_len * 2;
                    phrase_found = true;
                    break;
                }
            }
        }
        if !phrase_found {
            final_words.push(words[i]);
            i += 1;
        }
    }
    final_words
}

fn calculate_repetition_ratio(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return 0.0;
    }
    let mut word_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for word in &words {
        *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
    }
    let total_words = words.len() as f32;
    let repeated_words: usize = word_counts
        .values()
        .map(|&count| if count > 1 { count - 1 } else { 0 })
        .sum();
    repeated_words as f32 / total_words
}

fn clean_repetitive_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if is_meaningless_output(text) {
        return String::new();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return text.to_string();
    }
    let cleaned_words = remove_word_repetitions(&words);
    let cleaned_words = remove_phrase_repetitions(&cleaned_words);
    let final_text = cleaned_words.join(" ");
    if calculate_repetition_ratio(&final_text) > 0.7 {
        return String::new();
    }
    final_text
}

/// Formats speaker turns as a plain-text transcript, one line per turn, e.g.:
/// `[00:00 - 00:12] speaker_0: hello there`. Pure formatting, no I/O -- callers decide
/// where (if anywhere) to write the result; see `recording_commands.rs::stop_recording`'s
/// `diarizing` stage for the one place that saves this to the recording folder today.
pub fn format_turns_as_text(turns: &[SpeakerTurn]) -> String {
    fn format_timestamp(seconds: f64) -> String {
        let total_secs = seconds.max(0.0) as u64;
        format!("{:02}:{:02}", total_secs / 60, total_secs % 60)
    }

    turns
        .iter()
        .map(|t| {
            let speaker_label = t.speaker.as_deref().unwrap_or("unknown");
            format!(
                "[{} - {}] {}: {}",
                format_timestamp(t.start),
                format_timestamp(t.end),
                speaker_label,
                t.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
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

    #[test]
    fn format_turns_as_text_renders_timestamps_and_speaker_labels() {
        let turns = vec![
            SpeakerTurn { speaker: Some("speaker_0".into()), text: "hello there".into(), start: 0.0, end: 65.0 },
            SpeakerTurn { speaker: None, text: "uh".into(), start: 65.0, end: 66.0 },
        ];
        let text = format_turns_as_text(&turns);
        assert_eq!(
            text,
            "[00:00 - 01:05] speaker_0: hello there\n[01:05 - 01:06] unknown: uh"
        );
    }

    fn turn(speaker: Option<&str>, text: &str, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn { speaker: speaker.map(String::from), text: text.to_string(), start, end }
    }

    #[test]
    fn clean_speaker_turns_removes_repeated_words_within_a_turn() {
        let turns = vec![turn(Some("speaker_0"), "hello hello hello there", 0.0, 2.0)];
        let cleaned = clean_speaker_turns(turns);
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].text, "hello there");
        assert_eq!(cleaned[0].speaker.as_deref(), Some("speaker_0"));
    }

    #[test]
    fn clean_speaker_turns_drops_meaningless_turns_and_remerges_same_speaker() {
        let turns = vec![
            turn(Some("speaker_0"), "hello there how are you", 0.0, 2.0),
            turn(Some("speaker_0"), "thank you for watching", 2.0, 4.0),
            turn(Some("speaker_0"), "I am doing well thanks", 4.0, 6.0),
        ];
        let cleaned = clean_speaker_turns(turns);
        // The meaningless middle turn is dropped entirely (not left as an empty-text
        // turn), and the two speaker_0 turns around it re-merge into one continuous
        // turn spanning the original start/end.
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].speaker.as_deref(), Some("speaker_0"));
        assert_eq!(cleaned[0].text, "hello there how are you I am doing well thanks");
        assert_eq!(cleaned[0].start, 0.0);
        assert_eq!(cleaned[0].end, 6.0);
    }

    #[test]
    fn clean_speaker_turns_keeps_different_adjacent_speakers_separate() {
        let turns = vec![
            turn(Some("speaker_0"), "hello there", 0.0, 1.0),
            turn(Some("speaker_1"), "hi back", 1.0, 2.0),
        ];
        let cleaned = clean_speaker_turns(turns);
        assert_eq!(cleaned.len(), 2);
        assert_eq!(cleaned[0].speaker.as_deref(), Some("speaker_0"));
        assert_eq!(cleaned[0].text, "hello there");
        assert_eq!(cleaned[1].speaker.as_deref(), Some("speaker_1"));
        assert_eq!(cleaned[1].text, "hi back");
    }
}
