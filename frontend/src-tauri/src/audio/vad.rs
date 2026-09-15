use anyhow::{anyhow, Result};
use silero_rs::{VadConfig, VadSession, VadTransition};
use log::{debug, info, warn};
use std::collections::VecDeque;
use std::time::Duration;

/// Silero VAD only operates at 16kHz; input is resampled to this rate, and every
/// sample count and timestamp inside this module is expressed in it.
const VAD_SAMPLE_RATE: u32 = 16000;

/// Preferred and hard-cap length for a single speech run when `bound_long_runs` is set
/// (see `ContinuousVadProcessor::new`). Mirrors the target/hard-max split proposed in
/// GitHub issue #756 for the live path's incremental-delivery problem; here it guards
/// against the same underlying failure mode on the batch path instead: a run that never
/// sees `redemption_time_ms` of qualifying silence (common on real conversational audio
/// at batch's 2000ms redemption) would otherwise accumulate without limit -- see the
/// buffer-size warning below -- and later get re-split blind by
/// `common::split_segment_at_silence()` in retranscription.rs/import.rs, which has no
/// visibility into where this run's own natural pauses actually were.
const TARGET_SEGMENT_SAMPLES: usize = 20 * VAD_SAMPLE_RATE as usize; // 20s
const HARD_MAX_SEGMENT_SAMPLES: usize = 25 * VAD_SAMPLE_RATE as usize; // 25s

/// Represents a complete speech segment detected by VAD
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub samples: Vec<f32>,
    pub start_timestamp_ms: f64,
    pub end_timestamp_ms: f64,
    pub confidence: f32,
}

/// Processes audio in 30ms chunks but returns complete speech segments
pub struct ContinuousVadProcessor {
    session: VadSession,
    chunk_size: usize,
    sample_rate: u32,
    buffer: Vec<f32>,
    speech_segments: VecDeque<SpeechSegment>,
    current_speech: Vec<f32>,
    in_speech: bool,
    processed_samples: usize,
    speech_start_sample: usize,
    // Samples of the *current* run already emitted via force_split_current_run, relative
    // to speech_start_sample. Reset to 0 on every real SpeechStart. Read-only against
    // Silero's own buffer (see force_split_current_run's doc comment for why we never
    // delete from it mid-run).
    speech_delivered_in_run: usize,
    // State tracking for smart logging
    last_logged_state: bool,
    // Whether to proactively force-close a run once it exceeds hard_max_segment_samples
    // (see HARD_MAX_SEGMENT_SAMPLES's doc comment). Off by default so this stays scoped
    // to the batch callers that opted in, leaving the live path's characteristics
    // unchanged.
    bound_long_runs: bool,
    // Instance-level so tests can override them (see set_segment_caps_for_test) to
    // exercise force_split_current_run without needing tens of real seconds of
    // VAD-recognized speech; production code always gets the module consts.
    target_segment_samples: usize,
    hard_max_segment_samples: usize,
}

impl ContinuousVadProcessor {
    /// `bound_long_runs`: when true, a speech run that exceeds `HARD_MAX_SEGMENT_SAMPLES`
    /// without a qualifying pause is proactively closed at a natural (low-energy) point
    /// instead of being left to grow unbounded. Batch callers (`get_speech_chunks_with_progress`)
    /// opt in; the live path does not, to avoid changing its already-tuned delivery cadence
    /// without the dedicated validation #756 calls for.
    pub fn new(input_sample_rate: u32, redemption_time_ms: u32, bound_long_runs: bool) -> Result<Self> {
        crate::ensure_onnx_runtime_available()?;

        // Use STRICT settings to prevent silence from reaching Whisper
        let mut config = VadConfig::default();
        config.sample_rate = VAD_SAMPLE_RATE as usize;

        // CONTINUOUS SPEECH FIX: Tuned for capturing complete 5+ second utterances
        // Previous: 0.55/0.40 with 400ms redemption was fragmenting speech into 40ms segments
        // New: More lenient thresholds + longer redemption for continuous speech
        config.positive_speech_threshold = 0.50;  // Silero default - good for continuous speech
        config.negative_speech_threshold = 0.35;  // Silero default - allows natural pauses

        // Use the caller's redemption time without additional capping. The batch
        // paths (`import.rs`, `retranscription.rs`) pass 2000ms to bridge natural
        // pauses; the live path (`pipeline.rs`) passes 500ms to reduce pause-induced
        // latency. A qualifying silence is still required; bounded uninterrupted-
        // speech delivery is tracked in #756.
        config.redemption_time = Duration::from_millis(redemption_time_ms as u64);
        config.pre_speech_pad = Duration::from_millis(300);   // Pre-speech padding for context
        config.post_speech_pad = Duration::from_millis(400);  // Increased: more context at end

        // CRITICAL FIX: Increased min_speech_time to prevent tiny 40ms fragments
        // Previous: 100ms allowed too-short segments that Whisper rejects
        // New: 250ms ensures segments are substantial enough for Whisper (>100ms requirement)
        config.min_speech_time = Duration::from_millis(250);  // Prevent tiny fragments

        debug!("Creating VAD session with: sample_rate={}Hz, redemption={}ms, min_speech={}ms, input_rate={}Hz",
               VAD_SAMPLE_RATE, redemption_time_ms, 250, input_sample_rate);

        let session = VadSession::new(config)
            .map_err(|e| anyhow!("Failed to create VAD session: {:?}", e))?;

        // VAD uses 30ms chunks at 16kHz (480 samples)
        let vad_chunk_size = (VAD_SAMPLE_RATE as f32 * 0.03) as usize; // 480 samples

        info!("VAD processor created: input={}Hz, vad={}Hz, chunk_size={} samples",
              input_sample_rate, VAD_SAMPLE_RATE, vad_chunk_size);

        Ok(Self {
            session,
            chunk_size: vad_chunk_size,
            sample_rate: input_sample_rate, // Store input rate for resampling ratio in resample_to_16k()
            buffer: Vec::with_capacity(vad_chunk_size * 2),
            speech_segments: VecDeque::new(),
            current_speech: Vec::new(),
            in_speech: false,
            processed_samples: 0,
            speech_start_sample: 0,
            speech_delivered_in_run: 0,
            // Initialize state tracking
            last_logged_state: false,
            bound_long_runs,
            target_segment_samples: TARGET_SEGMENT_SAMPLES,
            hard_max_segment_samples: HARD_MAX_SEGMENT_SAMPLES,
        })
    }

    #[cfg(test)]
    fn set_segment_caps_for_test(&mut self, target_samples: usize, hard_max_samples: usize) {
        self.target_segment_samples = target_samples;
        self.hard_max_segment_samples = hard_max_samples;
    }

    /// Process incoming audio samples and return any complete speech segments
    /// Handles resampling from input sample rate to 16kHz for VAD processing
    pub fn process_audio(&mut self, samples: &[f32]) -> Result<Vec<SpeechSegment>> {
        // Resample to 16kHz if needed
        let resampled_audio = if self.sample_rate == 16000 {
            samples.to_vec()
        } else {
            self.resample_to_16k(samples)?
        };

        self.buffer.extend_from_slice(&resampled_audio);
        let mut completed_segments = Vec::new();

        // Process complete 30ms chunks (480 samples at 16kHz)
        while self.buffer.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.buffer.drain(..self.chunk_size).collect();
            self.process_chunk(&chunk)?;

            // Extract any completed speech segments
            while let Some(segment) = self.speech_segments.pop_front() {
                completed_segments.push(segment);
            }
        }

        Ok(completed_segments)
    }

    /// Improved resampling from input sample rate to 16kHz with anti-aliasing
    /// Uses linear interpolation and basic low-pass filtering for better quality
    fn resample_to_16k(&self, samples: &[f32]) -> Result<Vec<f32>> {
        if self.sample_rate == 16000 {
            return Ok(samples.to_vec());
        }

        // Calculate downsampling ratio
        let ratio = self.sample_rate as f64 / 16000.0;
        let output_len = (samples.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(output_len);

        // Apply simple low-pass filter before downsampling to reduce aliasing
        let cutoff_freq = 0.4; // Normalized frequency (0.4 * Nyquist)
        let mut filtered_samples = Vec::with_capacity(samples.len());

        // Simple moving average filter (basic low-pass)
        let filter_size = (self.sample_rate as f64 / (cutoff_freq * self.sample_rate as f64)) as usize;
        let filter_size = std::cmp::max(1, std::cmp::min(filter_size, 5)); // Limit filter size

        for i in 0..samples.len() {
            let start = if i >= filter_size { i - filter_size } else { 0 };
            let end = std::cmp::min(i + filter_size + 1, samples.len());
            let sum: f32 = samples[start..end].iter().sum();
            filtered_samples.push(sum / (end - start) as f32);
        }

        // Linear interpolation downsampling
        for i in 0..output_len {
            let source_pos = i as f64 * ratio;
            let source_index = source_pos as usize;
            let fraction = source_pos - source_index as f64;

            if source_index + 1 < filtered_samples.len() {
                // Linear interpolation
                let sample1 = filtered_samples[source_index];
                let sample2 = filtered_samples[source_index + 1];
                let interpolated = sample1 + (sample2 - sample1) * fraction as f32;
                resampled.push(interpolated);
            } else if source_index < filtered_samples.len() {
                resampled.push(filtered_samples[source_index]);
            }
        }

        debug!("Resampled from {} samples ({}Hz) to {} samples (16kHz) with anti-aliasing",
               samples.len(), self.sample_rate, resampled.len());

        Ok(resampled)
    }

    /// Flush any remaining audio and return final speech segments
    pub fn flush(&mut self) -> Result<Vec<SpeechSegment>> {
        debug!("VAD flush: in_speech={}, current_speech_len={}, buffer_len={}, speech_segments_queued={}",
              self.in_speech, self.current_speech.len(), self.buffer.len(), self.speech_segments.len());

        let mut completed_segments = Vec::new();
        // Preserve the real post-resampling endpoint before padding the final VAD frame.
        let real_end_sample = self.processed_samples + self.buffer.len();

        // Process any remaining buffered audio
        if !self.buffer.is_empty() {
            let remaining = self.buffer.clone();
            self.buffer.clear();

            // Pad to chunk size if needed
            let mut padded_chunk = remaining;
            if padded_chunk.len() < self.chunk_size {
                padded_chunk.resize(self.chunk_size, 0.0);
            }

            self.process_chunk(&padded_chunk)?;
        }

        // Force end any ongoing speech
        if self.in_speech && !self.current_speech.is_empty() {
            let real_sample_count = real_end_sample
                .checked_sub(self.speech_start_sample)
                .filter(|count| *count > 0)
                .ok_or_else(|| {
                    anyhow!(
                        "VAD flush invariant violated: active speech interval [{}, {}) is empty or reversed",
                        self.speech_start_sample,
                        real_end_sample
                    )
                })?;
            let active_speech = self.session.get_current_speech();
            if active_speech.len() < real_sample_count {
                return Err(anyhow!(
                    "VAD flush invariant violated: Silero active speech buffer has {} samples, but [{}, {}) requires {}",
                    active_speech.len(),
                    self.speech_start_sample,
                    real_end_sample,
                    real_sample_count
                ));
            }
            // Skip whatever force_split_current_run already delivered for this run (see
            // its doc comment) -- Silero's own buffer holds the *whole* run since it was
            // never deleted from, so without this the tail here would duplicate every
            // forced-cut segment already emitted.
            let delivered = self.speech_delivered_in_run.min(real_sample_count);
            if real_sample_count > delivered {
                let samples = active_speech[delivered..real_sample_count].to_vec();
                let start_ms = ((self.speech_start_sample + delivered) as f64
                    / VAD_SAMPLE_RATE as f64)
                    * 1000.0;
                let end_ms = (real_end_sample as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;

                debug!("VAD flush: Force-ending speech - start={}ms, end={}ms, duration={}ms, samples={}",
                      start_ms, end_ms, end_ms - start_ms, samples.len());

                let segment = SpeechSegment {
                    samples,
                    start_timestamp_ms: start_ms,
                    end_timestamp_ms: end_ms,
                    confidence: 0.8, // Estimated confidence for forced end
                };

                self.speech_segments.push_back(segment);
            }
            self.current_speech.clear();
            self.in_speech = false;
        }

        // Extract all remaining segments
        while let Some(segment) = self.speech_segments.pop_front() {
            completed_segments.push(segment);
        }

        Ok(completed_segments)
    }

    fn process_chunk(&mut self, chunk: &[f32]) -> Result<()> {
        // Track accumulated speech buffer size to detect memory issues
        let current_speech_size = self.current_speech.len();
        if current_speech_size > 1_000_000 {
            // More than ~62 seconds of accumulated speech at 16kHz
            warn!("VAD: Accumulated speech buffer is large: {} samples ({:.1}s) - possible memory issue",
                  current_speech_size, current_speech_size as f64 / 16000.0);
        }

        let transitions = self.session.process(chunk)
            .map_err(|e| anyhow!("VAD processing failed: {}", e))?;

        // Log transitions for debugging
        if !transitions.is_empty() {
            debug!("VAD transitions at sample {}: {} transitions", self.processed_samples, transitions.len());
        }

        // Handle VAD transitions
        for transition in transitions {
            match transition {
                VadTransition::SpeechStart { timestamp_ms } => {
                    // Only log if state changed
                    if !self.last_logged_state {
                        debug!("VAD: Speech started at {}ms", timestamp_ms);
                        self.last_logged_state = true;
                    }
                    self.in_speech = true;
                    // `timestamp_ms` is ALREADY session-absolute: silero computes it as
                    // `processed_duration() - pre_speech_pad`, where `processed_duration()`
                    // is every sample the network has seen this session. Adding our own
                    // session-absolute `processed_samples` to it double-counted the
                    // position, producing a start timestamp of roughly 2x the true one.
                    //
                    // The only reader is the end-of-recording flush below, so the bug
                    // surfaced once per recording, on the final segment — which landed at
                    // ~2x the file duration and sorted to the end of the transcript.
                    self.speech_start_sample = timestamp_ms * VAD_SAMPLE_RATE as usize / 1000;
                    self.speech_delivered_in_run = 0;
                    self.current_speech.clear();
                }
                VadTransition::SpeechEnd { start_timestamp_ms, end_timestamp_ms, samples } => {
                    // Only log if we were previously in speech state
                    if self.last_logged_state {
                        debug!("VAD: Speech ended at {}ms (duration: {}ms)", end_timestamp_ms, end_timestamp_ms - start_timestamp_ms);
                        self.last_logged_state = false;
                    }
                    self.in_speech = false;

                    // Use samples from VAD transition if available, otherwise use accumulated samples
                    let full_run_samples = if !samples.is_empty() {
                        samples
                    } else {
                        self.current_speech.clone()
                    };

                    // Skip whatever force_split_current_run already delivered for this run
                    // (see its doc comment) -- both sources above cover the *whole* run, so
                    // without this we'd duplicate every forced-cut segment already emitted.
                    let delivered = self.speech_delivered_in_run.min(full_run_samples.len());
                    let speech_samples = full_run_samples[delivered..].to_vec();
                    let adjusted_start_ms = start_timestamp_ms as f64
                        + (delivered as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;

                    if !speech_samples.is_empty() {
                        let segment = SpeechSegment {
                            samples: speech_samples,
                            start_timestamp_ms: adjusted_start_ms,
                            end_timestamp_ms: end_timestamp_ms as f64,
                            confidence: 0.9, // VAD confidence
                        };

                        info!("VAD: Completed speech segment: {:.1}ms duration, {} samples",
                              end_timestamp_ms as f64 - adjusted_start_ms, segment.samples.len());

                        self.speech_segments.push_back(segment);
                    }

                    self.current_speech.clear();
                }
            }
        }

        // Accumulate speech if we're currently in a speech state
        if self.in_speech {
            self.current_speech.extend_from_slice(chunk);
        }

        if self.bound_long_runs && self.in_speech {
            let current_position = self.processed_samples + chunk.len();
            let undelivered_len = current_position
                .saturating_sub(self.speech_start_sample)
                .saturating_sub(self.speech_delivered_in_run);
            if undelivered_len > self.hard_max_segment_samples {
                self.force_split_current_run()?;
            }
        }

        self.processed_samples += chunk.len();
        Ok(())
    }

    /// Proactively closes the current speech run at a natural (low-energy) point instead
    /// of letting it grow unbounded.
    ///
    /// Deliberately does **not** use `VadSession::take_until()` to actually remove the
    /// emitted prefix from Silero's internal buffer, even though that's what its doc
    /// comment suggests it's for: `take_until` updates `speech_start_ms` (what
    /// `get_current_speech()` reads) but *not* the separate `start_ms` copy embedded in
    /// the state machine's own `VadState::Speech` variant, which is what an eventual real
    /// `SpeechEnd` internally re-derives its own `get_speech()` call from. Once enough has
    /// been deleted via `take_until` to fall behind that stale, untouched original start,
    /// the very next natural pause panics deep inside `VadSession::process()` ("Duration
    /// ...ms is outside of session audio range") -- confirmed empirically against
    /// silero-rs rev 26a6460. So instead this only *reads* `get_current_speech()` (whole
    /// run, always valid) and tracks how much of it we've already delivered ourselves via
    /// `speech_delivered_in_run`; the real `SpeechEnd`/`flush()` paths trim that same
    /// prefix before emitting their own (final) segment. Silero's own buffer still grows
    /// for the full run underneath this (same memory profile as before bound_long_runs),
    /// but delivery -- what retranscription.rs/import.rs actually see -- stays bounded.
    fn force_split_current_run(&mut self) -> Result<()> {
        let active_speech = self.session.get_current_speech();
        if self.speech_delivered_in_run >= active_speech.len() {
            return Ok(());
        }
        let undelivered = &active_speech[self.speech_delivered_in_run..];
        let target_rel = self.target_segment_samples.min(undelivered.len());
        let cut_rel = find_silence_cut_point(undelivered, target_rel);
        if cut_rel == 0 {
            return Ok(());
        }

        let segment_samples = undelivered[..cut_rel].to_vec();
        let start_abs = self.speech_start_sample + self.speech_delivered_in_run;
        let cut_abs = start_abs + cut_rel;

        let start_ms = (start_abs as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;
        let end_ms = (cut_abs as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;

        info!(
            "VAD: Forcing segment boundary at {:.1}s (run exceeded {:.0}s without a qualifying pause), {} samples",
            end_ms / 1000.0,
            self.hard_max_segment_samples as f64 / VAD_SAMPLE_RATE as f64,
            segment_samples.len()
        );

        self.speech_segments.push_back(SpeechSegment {
            samples: segment_samples,
            start_timestamp_ms: start_ms,
            end_timestamp_ms: end_ms,
            confidence: 0.8, // Forced boundary, not a real VAD-detected pause -- same convention as flush()'s force-end
        });

        self.speech_delivered_in_run += cut_rel;

        Ok(())
    }
}

/// Search for a low-energy point near `target` within `samples`, so a proactively-split
/// run gets closed at a natural pause rather than an arbitrary sample boundary. Same
/// technique as `common::split_segment_at_silence()` (100ms energy window, +/-3s search
/// radius, same silence threshold), but operating on a single in-progress buffer with a
/// single target instead of a list of already-closed segments each needing full
/// re-splitting -- different enough shapes that sharing one helper wasn't a clean fit.
fn find_silence_cut_point(samples: &[f32], target: usize) -> usize {
    const ENERGY_WINDOW: usize = VAD_SAMPLE_RATE as usize / 10; // 100ms
    const SEARCH_RADIUS: usize = VAD_SAMPLE_RATE as usize * 3; // +/-3s
    const SILENCE_RMS_THRESHOLD: f32 = 0.02;
    const STEP: usize = VAD_SAMPLE_RATE as usize / 100; // 10ms

    if samples.len() <= ENERGY_WINDOW {
        return samples.len();
    }

    let search_start = target.saturating_sub(SEARCH_RADIUS);
    let search_end = (target + SEARCH_RADIUS).min(samples.len() - ENERGY_WINDOW);

    let mut best_split = target.min(samples.len());
    let mut best_rms = f32::MAX;

    if search_start < search_end {
        let mut idx = search_start;
        while idx + ENERGY_WINDOW <= search_end {
            let window = &samples[idx..idx + ENERGY_WINDOW];
            let rms = (window.iter().map(|s| s * s).sum::<f32>() / ENERGY_WINDOW as f32).sqrt();
            if rms < best_rms {
                best_rms = rms;
                best_split = idx + ENERGY_WINDOW / 2;
            }
            idx += STEP;
        }
    }

    if best_rms <= SILENCE_RMS_THRESHOLD {
        best_split
    } else {
        // No clear pause nearby -- cut at the exact target rather than an open-ended
        // search, same fallback spirit as split_segment_at_silence's overlap.
        target.min(samples.len())
    }
}

/// Legacy function for backward compatibility - now uses the optimized approach
pub fn extract_speech_16k(samples_mono_16k: &[f32]) -> Result<Vec<f32>> {
    let mut processor = ContinuousVadProcessor::new(16000, 400, false)?;

    // Process all audio
    let mut all_segments = processor.process_audio(samples_mono_16k)?;
    let final_segments = processor.flush()?;
    all_segments.extend(final_segments);

    // Concatenate all speech segments
    let mut result = Vec::new();
    let num_segments = all_segments.len();
    for segment in &all_segments {
        result.extend_from_slice(&segment.samples);
    }

    // Apply balanced energy filtering for very short segments
    if result.len() < 1600 { // Less than 100ms at 16kHz
        let input_energy: f32 = samples_mono_16k.iter().map(|&x| x * x).sum::<f32>() / samples_mono_16k.len() as f32;
        let rms = input_energy.sqrt();
        let peak = samples_mono_16k.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        // BALANCED FIX: Lowered thresholds to preserve quiet speech while still filtering silence
        // Previous aggressive values (0.08/0.15) were discarding valid quiet speech
        // New values (0.03/0.08) are more balanced - catch quiet speech, reject pure silence
        if rms < 0.2 || peak < 0.20 {
            info!("-----VAD detected silence/noise (RMS: {:.6}, Peak: {:.6}), skipping to prevent hallucinations-----", rms, peak);
            return Ok(Vec::new());
        } else {
            info!("VAD detected speech with sufficient energy (RMS: {:.6}, Peak: {:.6})", rms, peak);
            return Ok(samples_mono_16k.to_vec());
        }
    }

    debug!("VAD: Processed {} samples, extracted {} speech samples from {} segments",
           samples_mono_16k.len(), result.len(), num_segments);

    Ok(result)
}

/// Simple convenience function to get speech chunks from audio
/// Uses the optimized ContinuousVadProcessor with configurable redemption time
pub fn get_speech_chunks(samples_mono_16k: &[f32], redemption_time_ms: u32) -> Result<Vec<SpeechSegment>> {
    get_speech_chunks_with_progress(samples_mono_16k, redemption_time_ms, |_, _| true)
}

/// Get speech chunks with progress callback and cancellation support
/// The callback receives (progress_percent, segments_found) and returns false to cancel
pub fn get_speech_chunks_with_progress<F>(
    samples_mono_16k: &[f32],
    redemption_time_ms: u32,
    mut progress_callback: F,
) -> Result<Vec<SpeechSegment>>
where
    F: FnMut(u32, usize) -> bool,
{
    // Batch/offline caller (import.rs, retranscription.rs): bound long runs so a
    // recording with no qualifying pause doesn't accumulate unboundedly -- see
    // HARD_MAX_SEGMENT_SAMPLES's doc comment.
    let mut processor = ContinuousVadProcessor::new(16000, redemption_time_ms, true)?;

    let total_samples = samples_mono_16k.len();

    // For large files (>1 minute at 16kHz = 960,000 samples), process in chunks with progress logging
    const LARGE_FILE_THRESHOLD: usize = 960_000;
    const CHUNK_SIZE: usize = 160_000; // 10 seconds at 16kHz

    let mut all_segments = Vec::new();

    if total_samples > LARGE_FILE_THRESHOLD {
        info!("VAD: Processing large file ({} samples = {:.1}s), will log progress...",
              total_samples, total_samples as f64 / 16000.0);

        let mut processed = 0;
        let mut last_progress = 0u32;
        let mut chunk_count = 0;
        let total_chunks = (total_samples + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for chunk in samples_mono_16k.chunks(CHUNK_SIZE) {
            chunk_count += 1;

            let start_time = std::time::Instant::now();
            let segments = processor.process_audio(chunk)?;
            let elapsed = start_time.elapsed();

            // Debug log for chunk processing details
            debug!("VAD: Chunk {}/{} processed in {:?}, found {} segments",
                  chunk_count, total_chunks, elapsed, segments.len());

            // Warn if chunk processing took too long (>1 second)
            if elapsed.as_secs() > 1 {
                warn!("VAD: Chunk {} took {:?} - possible performance issue", chunk_count, elapsed);
            }

            all_segments.extend(segments);

            processed += chunk.len();
            let progress = ((processed * 100) / total_samples) as u32;

            // Call progress callback every 5%
            if progress >= last_progress + 5 {
                debug!("VAD: Progress {}% ({} segments found so far)", progress, all_segments.len());

                // Check for cancellation
                if !progress_callback(progress, all_segments.len()) {
                    info!("VAD: Cancelled by callback at {}%", progress);
                    return Err(anyhow!("VAD processing cancelled"));
                }

                last_progress = progress;
            }
        }

        let final_segments = processor.flush()?;
        all_segments.extend(final_segments);

        info!("VAD: Complete! Found {} speech segments", all_segments.len());
    } else {
        // Small file - process all at once
        all_segments = processor.process_audio(samples_mono_16k)?;
        let final_segments = processor.flush()?;
        all_segments.extend(final_segments);
    }

    Ok(all_segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic speech-like audio with alternating speech/silence
    fn generate_test_audio_with_speech(duration_seconds: f32, sample_rate: u32) -> Vec<f32> {
        let total_samples = (duration_seconds * sample_rate as f32) as usize;
        let mut samples = vec![0.0f32; total_samples];

        // Create speech-like patterns: bursts of sine waves with varying amplitude
        // Speech every 10 seconds for 5 seconds
        let speech_interval = 10.0; // seconds between speech starts
        let speech_duration = 5.0;  // seconds of speech

        for i in 0..total_samples {
            let time = i as f32 / sample_rate as f32;
            let cycle_time = time % speech_interval;

            // Speech occurs in the first `speech_duration` seconds of each cycle
            if cycle_time < speech_duration {
                // Generate speech-like signal: multiple frequencies with amplitude modulation
                let freq1 = 200.0 + (time * 50.0).sin() * 100.0; // Varying fundamental
                let freq2 = freq1 * 2.0; // Harmonic
                let freq3 = freq1 * 3.0; // Another harmonic

                let amplitude = 0.3 + 0.1 * (time * 5.0).sin(); // Amplitude modulation
                samples[i] = amplitude * (
                    0.5 * (2.0 * std::f32::consts::PI * freq1 * time).sin() +
                    0.3 * (2.0 * std::f32::consts::PI * freq2 * time).sin() +
                    0.2 * (2.0 * std::f32::consts::PI * freq3 * time).sin()
                );
            }
            // else: silence (already 0.0)
        }

        samples
    }

    #[test]
    fn test_vad_chunked_vs_single_processing() {
        // Generate 60 seconds of audio with speech patterns at 16kHz
        let audio = generate_test_audio_with_speech(60.0, 16000);
        println!("Generated {} samples ({:.1}s)", audio.len(), audio.len() as f32 / 16000.0);

        // Process all at once (like small files)
        let segments_single = get_speech_chunks(&audio, 2000).expect("Single processing failed");
        println!("Single processing found {} segments", segments_single.len());

        // Process in chunks (like large files)
        let segments_chunked = get_speech_chunks_with_progress(&audio, 2000, |progress, segments| {
            println!("Chunked progress: {}%, {} segments", progress, segments);
            true // Don't cancel
        }).expect("Chunked processing failed");
        println!("Chunked processing found {} segments", segments_chunked.len());

        // Both should find the same number of segments (approximately)
        // Allow some variance due to chunk boundary effects
        let diff = (segments_single.len() as i32 - segments_chunked.len() as i32).abs();
        assert!(diff <= 1,
            "Chunked and single processing found different segment counts: {} vs {} (diff: {})",
            segments_single.len(), segments_chunked.len(), diff);
    }

    #[test]
    fn test_vad_large_file_progress() {
        // Generate 120 seconds (2 minutes) of audio - triggers large file threshold
        let audio = generate_test_audio_with_speech(120.0, 16000);
        let total_samples = audio.len();
        println!("Generated {} samples ({:.1}s)", total_samples, total_samples as f32 / 16000.0);

        // This should trigger the large file path (>960,000 samples)
        assert!(total_samples > 960_000, "Audio should be large enough to trigger chunked processing");

        let mut progress_updates = Vec::new();
        let segments = get_speech_chunks_with_progress(&audio, 2000, |progress, segments| {
            progress_updates.push((progress, segments));
            true // Don't cancel
        }).expect("Processing failed");

        println!("Found {} segments with {} progress updates", segments.len(), progress_updates.len());

        // The synthetic signal is not real speech, so Silero may merge it into
        // one long segment. This test is specifically for the large-file path:
        // it must still emit speech and report monotonic progress through 100%.
        assert!(!segments.is_empty(), "Expected at least one speech segment");
        assert!(
            segments.iter().all(|segment| !segment.samples.is_empty()
                && segment.end_timestamp_ms > segment.start_timestamp_ms),
            "Expected all speech segments to contain audio with positive duration"
        );

        // Should have received progress updates
        assert!(!progress_updates.is_empty(), "Expected progress updates for large file");
        assert_eq!(
            progress_updates.last().map(|(progress, _)| *progress),
            Some(100),
            "Expected progress to reach 100%"
        );
        assert!(
            progress_updates
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0),
            "Expected progress updates to increase monotonically: {:?}",
            progress_updates
        );
    }

    #[test]
    fn test_vad_cancellation() {
        let audio = generate_test_audio_with_speech(120.0, 16000);

        // Cancel at 50%
        let result = get_speech_chunks_with_progress(&audio, 2000, |progress, _| {
            progress < 50 // Cancel when reaching 50%
        });

        // Should return error due to cancellation
        assert!(result.is_err(), "Expected cancellation error");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("cancelled"), "Error should mention cancellation: {}", err_msg);
    }

    #[test]
    fn test_vad_continuous_processor_state_across_chunks() {
        // Test that VAD state is correctly maintained across chunk boundaries
        let mut processor =
            ContinuousVadProcessor::new(16000, 2000, true).expect("Failed to create processor");

        // Generate audio with a speech segment that spans a chunk boundary
        let chunk_size = 160_000; // 10 seconds
        let audio = generate_test_audio_with_speech(30.0, 16000); // 30 seconds

        // Process in 10-second chunks
        let mut all_segments = Vec::new();
        for (i, chunk) in audio.chunks(chunk_size).enumerate() {
            let segments = processor.process_audio(chunk).expect("Processing failed");
            println!("Chunk {}: processed {} samples, found {} segments", i, chunk.len(), segments.len());
            all_segments.extend(segments);
        }

        // Flush remaining
        let final_segments = processor.flush().expect("Flush failed");
        all_segments.extend(final_segments);

        println!("Total segments found: {}", all_segments.len());

        // Should find speech segments
        assert!(all_segments.len() >= 1, "Expected at least 1 speech segment");
    }

    #[test]
    fn test_vad_400ms_vs_2000ms_segmentation() {
        // Demonstrates why 2000ms redemption is needed for batch processing:
        // 400ms creates excessive fragmentation, 2000ms bridges natural pauses.
        //
        // Audio pattern: 60s with 5s speech / 5s silence cycles
        // Natural pauses within speech (sentence gaps) are 500ms-1.5s
        let audio = generate_test_audio_with_speech(60.0, 16000);

        let segments_400 = get_speech_chunks(&audio, 400).expect("400ms processing failed");
        let segments_2000 = get_speech_chunks(&audio, 2000).expect("2000ms processing failed");

        println!(
            "400ms redemption: {} segments, 2000ms redemption: {} segments",
            segments_400.len(),
            segments_2000.len()
        );

        // 2000ms should produce fewer or equal segments (bridges more pauses)
        assert!(
            segments_2000.len() <= segments_400.len(),
            "2000ms redemption ({} segments) should not produce more segments than 400ms ({} segments)",
            segments_2000.len(),
            segments_400.len()
        );

        // Verify segments have reasonable durations with 2000ms
        for (i, seg) in segments_2000.iter().enumerate() {
            let duration_ms = seg.end_timestamp_ms - seg.start_timestamp_ms;
            println!("2000ms segment {}: {:.0}ms duration", i, duration_ms);
            // Each segment should be at least 250ms (min_speech_time)
            assert!(duration_ms >= 200.0, "Segment {} too short: {:.0}ms", i, duration_ms);
        }
    }
    /// Leading silence, then speech that runs to the end of the buffer.
    ///
    /// This is the shape that matters for the flush path: an utterance that begins
    /// late in a long session and is still in progress when recording stops.
    fn generate_late_speech_audio(
        silence_seconds: f32,
        speech_seconds: f32,
        sample_rate: u32,
    ) -> Vec<f32> {
        let silence_samples = (silence_seconds * sample_rate as f32) as usize;
        let speech = generate_test_audio_with_speech(speech_seconds, sample_rate);

        let mut samples = vec![0.0f32; silence_samples];
        samples.extend_from_slice(&speech);
        samples
    }

    /// `speech_start_sample` records where the current utterance began, so it can
    /// never point past the number of samples the VAD has actually seen.
    ///
    /// It used to, because it was computed as `processed_samples + timestamp_ms` where
    /// silero's `timestamp_ms` is ALREADY session-absolute
    /// (`processed_duration() - pre_speech_pad`), which doubled the position. The only
    /// reader is the force-end branch in `flush()`, so in production the corruption
    /// escaped as one phantom segment per recording, timestamped past the end of the
    /// audio. The error grows with how late the utterance starts, which is why it took
    /// a long recording to surface.
    #[test]
    fn test_speech_start_sample_never_exceeds_processed_samples() {
        // 20s of silence, then 3s of speech still running when the buffer ends.
        let audio = generate_late_speech_audio(20.0, 3.0, 16000);

        let mut processor =
            ContinuousVadProcessor::new(16000, 2000, true).expect("Failed to create processor");
        processor
            .process_audio(&audio)
            .expect("process_audio failed");

        assert!(
            processor.in_speech,
            "expected to still be mid-speech at the end of the buffer; the invariant \
             below would not be exercised otherwise"
        );

        assert!(
            processor.speech_start_sample <= processor.processed_samples,
            "speech_start_sample ({}) is past processed_samples ({}) - \
             session-absolute timestamp double-count regression. \
             In seconds: start={:.2}s vs processed={:.2}s",
            processor.speech_start_sample,
            processor.processed_samples,
            processor.speech_start_sample as f64 / VAD_SAMPLE_RATE as f64,
            processor.processed_samples as f64 / VAD_SAMPLE_RATE as f64,
        );
    }

    /// A forced segment's timestamps and samples must describe the same real audio interval.
    #[test]
    fn test_flush_segment_timestamps_stay_within_audio_duration() {
        let audio = generate_late_speech_audio(20.0, 3.0, 16000);
        let audio_duration_ms = (audio.len() as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;

        assert_eq!(audio.len(), 368_000);
        assert_eq!(
            audio.len() % 480,
            320,
            "fixture must require 160 samples of terminal VAD padding"
        );

        let mut processor =
            ContinuousVadProcessor::new(16000, 2000, true).expect("Failed to create processor");

        let segments = processor
            .process_audio(&audio)
            .expect("process_audio failed");
        assert!(
            segments.is_empty(),
            "process_audio completed a segment, so flush() would not exercise force-end"
        );
        assert!(
            processor.in_speech,
            "expected to still be mid-speech before flush()"
        );

        let flushed = processor.flush().expect("flush failed");
        assert_eq!(
            flushed.len(),
            1,
            "force-end must emit exactly one segment"
        );

        let segment = &flushed[0];
        let start_sample = ((segment.start_timestamp_ms / 1000.0)
            * VAD_SAMPLE_RATE as f64)
            .round() as usize;

        assert!(
            segment.start_timestamp_ms <= audio_duration_ms,
            "segment starts at {:.0}ms, beyond the {:.0}ms of audio supplied",
            segment.start_timestamp_ms,
            audio_duration_ms
        );
        assert_eq!(
            segment.end_timestamp_ms, 23_000.0,
            "forced segment must end at the real audio endpoint"
        );
        assert!(
            segment.end_timestamp_ms <= audio_duration_ms,
            "segment ends at {:.0}ms, beyond the {:.0}ms of audio supplied",
            segment.end_timestamp_ms,
            audio_duration_ms
        );
        assert!(
            segment.end_timestamp_ms >= segment.start_timestamp_ms,
            "segment ends before it starts: {:.0}ms -> {:.0}ms",
            segment.start_timestamp_ms,
            segment.end_timestamp_ms
        );
        assert_eq!(
            segment.samples.as_slice(),
            &audio[start_sample..],
            "forced payload must contain the exact real audio interval named by its timestamps"
        );
        assert_eq!(segment.samples.len(), audio.len() - start_sample);

        let timestamp_sample_count = (((segment.end_timestamp_ms
            - segment.start_timestamp_ms)
            / 1000.0)
            * VAD_SAMPLE_RATE as f64)
            .round() as usize;
        assert_eq!(
            timestamp_sample_count,
            segment.samples.len(),
            "timestamp duration and payload length must describe the same sample interval"
        );

        let payload_end_ms = segment.start_timestamp_ms
            + (segment.samples.len() as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;
        assert!(
            (payload_end_ms - segment.end_timestamp_ms).abs() < 0.001,
            "payload ends at {payload_end_ms:.3}ms, timestamp ends at {:.3}ms",
            segment.end_timestamp_ms
        );

        assert!(
            processor.flush().expect("second flush failed").is_empty(),
            "flush() must not emit the same forced segment twice"
        );
    }

    /// Continuous speech-like audio with no silence gaps at all, for exercising the
    /// bound_long_runs hard-cap path directly, without depending on a natural pause
    /// ever occurring.
    fn generate_continuous_speech_audio(duration_seconds: f32, sample_rate: u32) -> Vec<f32> {
        let total_samples = (duration_seconds * sample_rate as f32) as usize;
        let mut samples = vec![0.0f32; total_samples];
        for (i, sample) in samples.iter_mut().enumerate() {
            let time = i as f32 / sample_rate as f32;
            let freq1 = 200.0 + (time * 50.0).sin() * 100.0;
            let freq2 = freq1 * 2.0;
            let freq3 = freq1 * 3.0;
            let amplitude = 0.3 + 0.1 * (time * 5.0).sin();
            *sample = amplitude
                * (0.5 * (2.0 * std::f32::consts::PI * freq1 * time).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * freq2 * time).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * freq3 * time).sin());
        }
        samples
    }

    /// With bound_long_runs on, a speech run must be proactively closed in multiple
    /// contiguous segments instead of growing unbounded -- validates the actual mechanism
    /// (`get_current_speech()` read + `speech_delivered_in_run` bookkeeping, see
    /// `force_split_current_run`'s doc comment for why not `take_until()`) against a real
    /// Silero session. Uses small test-only caps (`set_segment_caps_for_test`) rather than
    /// the real 25s hard max: a real trained net won't sustain "speech" on a crude
    /// synthetic tone for tens of seconds (confirmed empirically -- this generator's
    /// output is only recognized for roughly its first ~1.2s), so forcing the *mechanism*
    /// to trigger well within that reliably-recognized window is what's testable here
    /// without real speech audio (unavailable in this environment).
    ///
    /// This is the same underlying bug found on real long recordings (see
    /// docs/as-is/Pipeline audio.md, "Bug scoperto: segmentazione VAD che non chiude su
    /// audio lunghi"): without bound_long_runs, a run with no qualifying pause stays open
    /// until flush(), then gets blind-chopped downstream with no VAD signal at all.
    #[test]
    fn test_bound_long_runs_forces_contiguous_segment_boundaries() {
        let audio = generate_continuous_speech_audio(5.0, 16000);

        let mut processor =
            ContinuousVadProcessor::new(16000, 2000, true).expect("Failed to create processor");
        processor.set_segment_caps_for_test(3200, 4800); // 200ms target, 300ms hard max
        let mut segments = processor.process_audio(&audio).expect("process_audio failed");
        segments.extend(processor.flush().expect("flush failed"));

        assert!(
            segments.len() >= 2,
            "expected multiple segments from a run exceeding the 300ms test hard cap, got {}",
            segments.len()
        );

        for seg in &segments {
            let duration_s = (seg.end_timestamp_ms - seg.start_timestamp_ms) / 1000.0;
            assert!(
                duration_s <= 0.35, // 300ms cap + slack for search-window/post-pad rounding
                "segment exceeds the test hard cap: {:.3}s",
                duration_s
            );
            let expected_samples = ((seg.end_timestamp_ms - seg.start_timestamp_ms) / 1000.0
                * VAD_SAMPLE_RATE as f64)
                .round() as i64;
            assert!(
                (seg.samples.len() as i64 - expected_samples).abs() <= 1,
                "segment payload length {} doesn't match its own timestamps ({} expected)",
                seg.samples.len(),
                expected_samples
            );
        }

        // No gaps and no overlaps between consecutive segments -- the property that
        // matters most: no audio dropped or duplicated across a forced boundary.
        for pair in segments.windows(2) {
            assert!(
                (pair[1].start_timestamp_ms - pair[0].end_timestamp_ms).abs() < 1.0,
                "gap or overlap between forced segments: {:.1}ms -> {:.1}ms",
                pair[0].end_timestamp_ms,
                pair[1].start_timestamp_ms
            );
        }
    }

    /// Same as above, but the buffer ends *before* any natural pause -- so the run is
    /// still open when flush() runs, exercising flush()'s own delivered-prefix trim
    /// (`force_split_current_run`'s SpeechEnd-path trim is a different branch, covered by
    /// the test above where a real SpeechEnd does fire before end-of-audio).
    #[test]
    fn test_bound_long_runs_forces_boundaries_then_flush_closes_remainder() {
        // Empirically (this generator, this model): SpeechStart doesn't fire until
        // ~1.02s of input has been fed (this signal's own ramp-up, not a fixed constant),
        // and a real SpeechEnd fires around ~1.63s on the 5s version above. 1.4s sits
        // inside that window, so the run is still open when flush() runs.
        let audio = generate_continuous_speech_audio(1.4, 16000);

        let mut processor =
            ContinuousVadProcessor::new(16000, 2000, true).expect("Failed to create processor");
        processor.set_segment_caps_for_test(3200, 4800); // 200ms target, 300ms hard max
        let mut segments = processor.process_audio(&audio).expect("process_audio failed");
        let flushed = processor.flush().expect("flush failed");

        assert!(
            !flushed.is_empty(),
            "expected flush() to close the still-open remainder after forced cuts"
        );
        segments.extend(flushed);

        assert!(
            segments.len() >= 2,
            "expected at least one forced cut plus the flush-closed remainder, got {}",
            segments.len()
        );

        for pair in segments.windows(2) {
            assert!(
                (pair[1].start_timestamp_ms - pair[0].end_timestamp_ms).abs() < 1.0,
                "gap or overlap between forced segments: {:.1}ms -> {:.1}ms",
                pair[0].end_timestamp_ms,
                pair[1].start_timestamp_ms
            );
        }

        // No segment (forced or flush-closed) may extend past the real audio duration.
        let audio_duration_ms = (audio.len() as f64 / VAD_SAMPLE_RATE as f64) * 1000.0;
        let last = segments.last().expect("at least one segment");
        assert!(
            last.end_timestamp_ms <= audio_duration_ms + 1.0,
            "flush-closed segment ends at {:.1}ms, beyond the {:.1}ms of audio supplied",
            last.end_timestamp_ms,
            audio_duration_ms
        );
    }
}
