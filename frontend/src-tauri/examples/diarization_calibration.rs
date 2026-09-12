//! Diagnostic tool (not shipped app behavior): runs the real diarization pipeline
//! (`app_lib::diarization::DiarizationEngine`) against real recordings and compares the
//! spectral/NME-SC clustering result (the production method, ADR-0024) to a known
//! reference transcript.
//!
//! Expensive ONNX inference (`process_chunk`) runs exactly once per recording; clustering
//! itself (`finalize_with_spectral_and_p`) is cheap and re-run per `--nearest-neighbors`
//! value against the same accumulated embeddings.
//!
//! Usage:
//!   cargo run --release --example diarization_calibration -- \
//!     --data-dir <path> --models-dir <path> \
//!     [--max-speakers 20] [--nearest-neighbors 10,15,25]
//!
//! `--data-dir` can also come from the `MEETILY_CALIBRATION_DIR` env var.
//!
//! `--data-dir` layout: one subfolder per recording, each containing exactly one audio
//! file (mp4/wav/m4a/mp3/mov/webm) and a `reference.txt` in the format
//! `[HH:MM:SS] SPEAKER_NN: text` (one turn per line, blank lines allowed).
//!
//! This directory must live OUTSIDE this repo (and outside the parent workspace) --
//! these are real, unredacted meeting recordings. The tool refuses to run if `--data-dir`
//! resolves inside this repo's working tree.

use anyhow::{anyhow, bail, Context, Result};
use app_lib::audio::decoder::decode_audio_file;
use app_lib::diarization::{DiarizationEngine, SpeakerSegment};
use clap::Parser;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Calibrate the spectral/NME-SC diarization clustering against real recordings")]
struct Cli {
    /// Folder with one subfolder per recording (audio file + reference.txt). Falls back
    /// to MEETILY_CALIBRATION_DIR if omitted.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Folder containing segmentation-model.onnx and embedding-model.onnx (already
    /// downloaded via the app's diarization toggle at least once).
    #[arg(long)]
    models_dir: PathBuf,

    /// Upper bound passed to the spectral-clustering method
    /// (`DiarizationEngine::finalize_with_spectral`). Pass a generous value, not a tight
    /// guess at the real speaker count -- see `cluster_embeddings_spectral`'s doc comment
    /// on why capping too tightly can collapse the estimate well below the cap.
    #[arg(long, default_value_t = 20)]
    max_speakers: usize,

    /// Comma-separated list of p-nearest-neighbor pruning counts to try with the spectral
    /// method (see `cluster_embeddings_spectral_with_p`), instead of always using its
    /// default per-recording NME auto-search (`nme_select_p`, in `clustering.rs`). Omit to
    /// run once with the auto-search only. Useful for experimenting on recordings where
    /// the auto-search under- or over-estimates the speaker count -- see
    /// docs/sviluppi/diarization/Architettura pipeline.md, "Primo confronto reale...",
    /// "Tentativo di fix: p fisso...". A fixed p was tried and found not to generalize
    /// across recordings of different sizes -- see that same doc. Example:
    /// `--nearest-neighbors 10,15,25`.
    #[arg(long)]
    nearest_neighbors: Option<String>,
}

struct GroundTruthTurn {
    start: f64,
    end: f64,
    speaker: String,
}

struct Recording {
    name: String,
    audio_path: PathBuf,
    reference_path: PathBuf,
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .with_context(|| format!("invalid integer value: {part:?}"))
        })
        .collect()
}

/// Refuses a `data_dir` that resolves inside this repo's working tree (or its parent
/// workspace) -- these are real, confidential meeting recordings and must never risk
/// being picked up by a later `git add`.
fn assert_data_dir_outside_repo(data_dir: &Path) -> Result<()> {
    let data_dir_canonical = data_dir
        .canonicalize()
        .with_context(|| format!("--data-dir does not exist: {}", data_dir.display()))?;

    // CARGO_MANIFEST_DIR (baked in at compile time) is frontend/src-tauri -- walk up two
    // levels to the repo root (frontend/src-tauri -> frontend -> repo root), then one more
    // to also cover the parent workspace directory the repo is checked out into.
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .ancestors()
        .nth(2)
        .ok_or_else(|| anyhow!("could not resolve repo root from CARGO_MANIFEST_DIR"))?;
    let workspace_root = repo_root.parent().unwrap_or(repo_root);

    for boundary in [repo_root, workspace_root] {
        if let Ok(boundary_canonical) = boundary.canonicalize() {
            if data_dir_canonical.starts_with(&boundary_canonical) {
                bail!(
                    "--data-dir ({}) resolves inside {} -- real meeting recordings must be \
                     kept completely outside both the repo and its parent workspace. Pick a \
                     folder elsewhere, e.g. ~/meetily-calibration/.",
                    data_dir_canonical.display(),
                    boundary_canonical.display()
                );
            }
        }
    }
    Ok(())
}

fn discover_recordings(data_dir: &Path) -> Result<Vec<Recording>> {
    const AUDIO_EXTS: &[&str] = &["mp4", "wav", "m4a", "mp3", "mov", "webm"];
    let mut recordings = Vec::new();

    for entry in std::fs::read_dir(data_dir)
        .with_context(|| format!("cannot read --data-dir: {}", data_dir.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let subdir = entry.path();
        let name = subdir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        let reference_path = subdir.join("reference.txt");
        if !reference_path.exists() {
            eprintln!("skipping {name}: no reference.txt");
            continue;
        }

        let audio_candidates: Vec<PathBuf> = std::fs::read_dir(&subdir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| AUDIO_EXTS.contains(&e.to_lowercase().as_str()))
                    .unwrap_or(false)
            })
            .collect();

        match audio_candidates.len() {
            0 => eprintln!("skipping {name}: no audio file found ({AUDIO_EXTS:?})"),
            1 => recordings.push(Recording {
                name,
                audio_path: audio_candidates[0].clone(),
                reference_path,
            }),
            _ => eprintln!(
                "skipping {name}: found {} audio files, expected exactly 1",
                audio_candidates.len()
            ),
        }
    }

    recordings.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(recordings)
}

/// `[HH:MM:SS] SPEAKER_NN: text` per line. The format has no explicit turn-end time --
/// inferred as the next turn's start (last turn: start + 5.0, an arbitrary fallback since
/// there is nothing better available).
fn parse_reference_transcript(path: &Path) -> Result<Vec<GroundTruthTurn>> {
    let re = Regex::new(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+(SPEAKER_\d+):\s*(.*)$")?;
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read reference transcript: {}", path.display()))?;

    struct Raw {
        start: f64,
        speaker: String,
    }
    let mut raw: Vec<Raw> = Vec::new();
    for line in content.lines() {
        let Some(caps) = re.captures(line.trim()) else {
            continue;
        };
        let h: f64 = caps[1].parse()?;
        let m: f64 = caps[2].parse()?;
        let s: f64 = caps[3].parse()?;
        raw.push(Raw {
            start: h * 3600.0 + m * 60.0 + s,
            speaker: caps[4].to_string(),
        });
    }

    if raw.is_empty() {
        bail!("no turns matched in {} -- unexpected format?", path.display());
    }

    let mut turns = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let end = raw.get(i + 1).map(|r| r.start).unwrap_or(raw[i].start + 5.0);
        turns.push(GroundTruthTurn {
            start: raw[i].start,
            end,
            speaker: raw[i].speaker.clone(),
        });
    }
    Ok(turns)
}

/// Predicted speaker segment active at `t`, or the temporally nearest one if `t` falls in
/// a gap (segments shorter than the min-duration filter get dropped upstream, so small
/// gaps are expected and shouldn't just be treated as "no prediction").
fn segment_at<'a>(segments: &'a [SpeakerSegment], t: f64) -> Option<&'a SpeakerSegment> {
    segments
        .iter()
        .min_by(|a, b| {
            let dist = |s: &SpeakerSegment| {
                if t >= s.start && t < s.end {
                    0.0
                } else {
                    (t - s.start).abs().min((t - s.end).abs())
                }
            };
            dist(a).partial_cmp(&dist(b)).unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Majority-vote label mapping (cluster purity, no Hungarian algorithm needed for this
/// few speakers): each predicted label is assigned to whichever true speaker it overlaps
/// most often at ground-truth turn midpoints, then turn accuracy is scored under that
/// mapping.
fn score_against_ground_truth(segments: &[SpeakerSegment], ground_truth: &[GroundTruthTurn]) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }

    let mut samples: Vec<(Option<String>, &str)> = Vec::with_capacity(ground_truth.len());
    for turn in ground_truth {
        let midpoint = (turn.start + turn.end) / 2.0;
        let predicted_label = segment_at(segments, midpoint).map(|s| s.speaker.clone());
        samples.push((predicted_label, turn.speaker.as_str()));
    }

    let mut contingency: HashMap<String, HashMap<&str, usize>> = HashMap::new();
    for (predicted, truth) in &samples {
        if let Some(label) = predicted {
            *contingency.entry(label.clone()).or_default().entry(truth).or_insert(0) += 1;
        }
    }

    let label_to_truth: HashMap<String, &str> = contingency
        .into_iter()
        .map(|(label, counts)| {
            let best = counts.into_iter().max_by_key(|(_, count)| *count).map(|(t, _)| t).unwrap_or("");
            (label, best)
        })
        .collect();

    let correct = samples
        .iter()
        .filter(|(predicted, truth)| {
            predicted
                .as_ref()
                .and_then(|label| label_to_truth.get(label))
                .map(|mapped| mapped == truth)
                .unwrap_or(false)
        })
        .count();

    100.0 * correct as f64 / samples.len() as f64
}

fn distinct_speaker_count(segments: &[SpeakerSegment]) -> usize {
    segments
        .iter()
        .map(|s| s.speaker.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len()
}

fn distinct_true_speaker_count(ground_truth: &[GroundTruthTurn]) -> usize {
    ground_truth
        .iter()
        .map(|t| t.speaker.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let data_dir = cli
        .data_dir
        .or_else(|| std::env::var("MEETILY_CALIBRATION_DIR").ok().map(PathBuf::from))
        .ok_or_else(|| anyhow!("--data-dir not given and MEETILY_CALIBRATION_DIR not set"))?;
    assert_data_dir_outside_repo(&data_dir)?;

    // One entry per p value to try with the spectral method; `None` means "use the
    // default per-recording NME auto-search". Absent --nearest-neighbors -> a single
    // default-only run.
    let p_values: Vec<Option<usize>> = match &cli.nearest_neighbors {
        Some(s) => parse_usize_list(s)?.into_iter().map(Some).collect(),
        None => vec![None],
    };

    let segmentation_model_path = cli.models_dir.join("segmentation-model.onnx");
    let embedding_model_path = cli.models_dir.join("embedding-model.onnx");
    for p in [&segmentation_model_path, &embedding_model_path] {
        if !p.exists() {
            bail!(
                "model file not found: {} -- download the diarization models once via the \
                 app (Settings > toggle diarization on > Download), then point --models-dir \
                 at that folder",
                p.display()
            );
        }
    }

    let recordings = discover_recordings(&data_dir)?;
    if recordings.is_empty() {
        bail!("no valid recordings found under {}", data_dir.display());
    }
    println!("Found {} recording(s) to calibrate against.\n", recordings.len());

    let mut spectral_results: Vec<(String, String, usize, usize, f64)> = Vec::new(); // (name, p_label, detected, true, accuracy)

    for recording in &recordings {
        println!("=== {} ===", recording.name);
        let ground_truth = parse_reference_transcript(&recording.reference_path)?;
        let true_speaker_count = distinct_true_speaker_count(&ground_truth);
        println!(
            "  reference: {} turns, {} distinct speakers",
            ground_truth.len(),
            true_speaker_count
        );

        println!("  decoding + running ONNX segmentation/embedding (once)...");
        let decoded = decode_audio_file(&recording.audio_path)
            .with_context(|| format!("failed to decode {}", recording.audio_path.display()))?;
        let samples = decoded.to_whisper_format();

        // num_threads=1, same as the production batch path (audio/import.rs).
        let mut engine = DiarizationEngine::new(
            segmentation_model_path.to_str().unwrap(),
            embedding_model_path.to_str().unwrap(),
            1,
        )
        .map_err(|e| anyhow!("failed to init DiarizationEngine: {e}"))?;
        engine
            .process_chunk(&samples, 0.0)
            .map_err(|e| anyhow!("process_chunk failed: {e}"))?;

        // One run per p value in p_values (just [None] -- the default NME auto-search --
        // if --nearest-neighbors was omitted).
        for &p_override in &p_values {
            let spectral_segments = engine.finalize_with_spectral_and_p(cli.max_speakers, p_override);
            let spectral_detected = distinct_speaker_count(&spectral_segments);
            let spectral_accuracy = score_against_ground_truth(&spectral_segments, &ground_truth);
            let p_label = p_override.map(|p| p.to_string()).unwrap_or_else(|| "auto".to_string());
            println!(
                "  spectral (max_speakers={}, p={}) | detected {:>3} | true {:>3} | turn acc {:>5.1}%",
                cli.max_speakers, p_label, spectral_detected, true_speaker_count, spectral_accuracy
            );
            spectral_results.push((
                recording.name.clone(),
                p_label,
                spectral_detected,
                true_speaker_count,
                spectral_accuracy,
            ));
        }

        println!();
    }

    println!(
        "\n=== Spettrale (max_speakers={}), dettaglio per registrazione ===",
        cli.max_speakers
    );
    println!(
        "  {:>30} | {:>6} | {:>9} | {:>9} | {:>12}",
        "registrazione", "p", "detected", "true", "turn acc %"
    );
    for (name, p_label, detected, truth, accuracy) in &spectral_results {
        println!(
            "  {:>30} | {:>6} | {:>9} | {:>9} | {:>12.1}",
            name, p_label, detected, truth, accuracy
        );
    }

    // Aggregate per p value tried, in first-seen order -- lets a --nearest-neighbors
    // sweep be compared side by side without re-running the tool once per value.
    let mut seen_p_labels: Vec<String> = Vec::new();
    for (_, p_label, ..) in &spectral_results {
        if !seen_p_labels.contains(p_label) {
            seen_p_labels.push(p_label.clone());
        }
    }
    println!(
        "\n=== Spettrale, aggregato per valore di p (media su {} registrazioni) ===",
        recordings.len()
    );
    println!("  {:>6} | {:>16} | {:>12}", "p", "avg |detected-true|", "avg turn acc %");
    for p_label in &seen_p_labels {
        let matching: Vec<&(String, String, usize, usize, f64)> = spectral_results
            .iter()
            .filter(|(_, p, ..)| p == p_label)
            .collect();
        let n = matching.len() as f64;
        let avg_count_error: f64 = matching
            .iter()
            .map(|(_, _, d, t, _)| (*d as f64 - *t as f64).abs())
            .sum::<f64>()
            / n;
        let avg_accuracy: f64 = matching.iter().map(|(_, _, _, _, a)| a).sum::<f64>() / n;
        println!("  {:>6} | {:>16.2} | {:>12.1}", p_label, avg_count_error, avg_accuracy);
    }

    Ok(())
}
