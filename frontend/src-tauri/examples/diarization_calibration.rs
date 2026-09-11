//! Diagnostic tool (not shipped app behavior): runs the real diarization pipeline
//! (`app_lib::diarization::DiarizationEngine`) against real recordings and compares the
//! result to a known reference transcript, across a grid of clustering/reattach
//! thresholds -- same methodology as the original "Blocker 2" calibration table in
//! `docs/sviluppi/diarization/Roadmap e todo.md` (outside this repo), but on real
//! multi-speaker Italian recordings instead of 4 clean/synthetic samples.
//!
//! Expensive ONNX inference (`process_chunk`) runs exactly once per recording; the grid
//! sweep then only re-runs `DiarizationEngine::finalize_with_thresholds` (cheap
//! clustering) per grid point against the same accumulated embeddings.
//!
//! Usage:
//!   cargo run --release --example diarization_calibration -- \
//!     --data-dir <path> --models-dir <path> \
//!     [--clustering-thresholds 0.5,0.55,0.6,0.65,0.7] \
//!     [--reattach-thresholds 0.35,0.40,0.45,0.50]
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
use app_lib::diarization::clustering::MIN_CLUSTER_SIZE;
use app_lib::diarization::{DiarizationEngine, SpeakerSegment};
use clap::Parser;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Sweep diarization clustering/reattach thresholds against real recordings")]
struct Cli {
    /// Folder with one subfolder per recording (audio file + reference.txt). Falls back
    /// to MEETILY_CALIBRATION_DIR if omitted.
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Folder containing segmentation-model.onnx and embedding-model.onnx (already
    /// downloaded via the app's diarization toggle at least once).
    #[arg(long)]
    models_dir: PathBuf,

    /// Comma-separated list of DEFAULT_CLUSTERING_THRESHOLD values to try.
    #[arg(long, default_value = "0.5,0.55,0.6,0.65,0.7")]
    clustering_thresholds: String,

    /// Comma-separated list of REATTACH_THRESHOLD values to try.
    #[arg(long, default_value = "0.35,0.40,0.45,0.50")]
    reattach_thresholds: String,

    /// Upper bound passed to the experimental spectral-clustering method
    /// (`DiarizationEngine::finalize_with_spectral`). Pass a generous value, not a tight
    /// guess at the real speaker count -- see `cluster_embeddings_spectral`'s doc comment
    /// on why capping too tightly can collapse the estimate well below the cap.
    #[arg(long, default_value_t = 20)]
    max_speakers: usize,
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

struct GridResult {
    clustering_threshold: f32,
    reattach_threshold: f32,
    detected_speaker_count: usize,
    true_speaker_count: usize,
    turn_accuracy_pct: f64,
}

fn parse_threshold_list(s: &str) -> Result<Vec<f32>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<f32>()
                .with_context(|| format!("invalid threshold value: {part:?}"))
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

    let clustering_thresholds = parse_threshold_list(&cli.clustering_thresholds)?;
    let reattach_thresholds = parse_threshold_list(&cli.reattach_thresholds)?;

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

    let mut all_results: Vec<(String, GridResult)> = Vec::new();
    let mut spectral_results: Vec<(String, usize, usize, f64)> = Vec::new(); // (name, detected, true, accuracy)

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

        println!(
            "  {:>6} | {:>8} | {:>9} | {:>9} | {:>12}",
            "clust", "reattach", "detected", "true", "turn acc %"
        );
        for &ct in &clustering_thresholds {
            for &rt in &reattach_thresholds {
                let segments = engine.finalize_with_thresholds(ct, MIN_CLUSTER_SIZE, rt);
                let detected = distinct_speaker_count(&segments);
                let accuracy = score_against_ground_truth(&segments, &ground_truth);
                println!(
                    "  {:>6.2} | {:>8.2} | {:>9} | {:>9} | {:>12.1}",
                    ct, rt, detected, true_speaker_count, accuracy
                );
                all_results.push((
                    recording.name.clone(),
                    GridResult {
                        clustering_threshold: ct,
                        reattach_threshold: rt,
                        detected_speaker_count: detected,
                        true_speaker_count,
                        turn_accuracy_pct: accuracy,
                    },
                ));
            }
        }

        // Experimental spectral method -- same already-accumulated embeddings, no second
        // ONNX inference pass. Printed separately since it isn't part of the
        // threshold/reattach grid (it has its own single parameter, max_speakers).
        let spectral_segments = engine.finalize_with_spectral(cli.max_speakers);
        let spectral_detected = distinct_speaker_count(&spectral_segments);
        let spectral_accuracy = score_against_ground_truth(&spectral_segments, &ground_truth);
        println!(
            "  spectral (max_speakers={}) | detected {:>3} | true {:>3} | turn acc {:>5.1}%",
            cli.max_speakers, spectral_detected, true_speaker_count, spectral_accuracy
        );
        spectral_results.push((recording.name.clone(), spectral_detected, true_speaker_count, spectral_accuracy));

        println!();
    }

    println!("=== Aggregato per combinazione di soglie (media su {} registrazioni) ===", recordings.len());
    println!(
        "  {:>6} | {:>8} | {:>16} | {:>12}",
        "clust", "reattach", "avg |detected-true|", "avg turn acc %"
    );
    for &ct in &clustering_thresholds {
        for &rt in &reattach_thresholds {
            let matching: Vec<&GridResult> = all_results
                .iter()
                .map(|(_, r)| r)
                .filter(|r| r.clustering_threshold == ct && r.reattach_threshold == rt)
                .collect();
            let n = matching.len() as f64;
            let avg_count_error: f64 = matching
                .iter()
                .map(|r| (r.detected_speaker_count as f64 - r.true_speaker_count as f64).abs())
                .sum::<f64>()
                / n;
            let avg_accuracy: f64 = matching.iter().map(|r| r.turn_accuracy_pct).sum::<f64>() / n;
            println!(
                "  {:>6.2} | {:>8.2} | {:>16.2} | {:>12.1}",
                ct, rt, avg_count_error, avg_accuracy
            );
        }
    }

    println!(
        "\n=== Spettrale (max_speakers={}), media su {} registrazioni ===",
        cli.max_speakers,
        recordings.len()
    );
    println!("  {:>30} | {:>9} | {:>9} | {:>12}", "registrazione", "detected", "true", "turn acc %");
    for (name, detected, truth, accuracy) in &spectral_results {
        println!("  {:>30} | {:>9} | {:>9} | {:>12.1}", name, detected, truth, accuracy);
    }
    let n_spectral = spectral_results.len() as f64;
    let spectral_avg_count_error: f64 = spectral_results
        .iter()
        .map(|(_, d, t, _)| (*d as f64 - *t as f64).abs())
        .sum::<f64>()
        / n_spectral;
    let spectral_avg_accuracy: f64 =
        spectral_results.iter().map(|(_, _, _, a)| a).sum::<f64>() / n_spectral;
    println!(
        "  {:>30} | avg |detected-true| {:>6.2} | avg turn acc {:>5.1}%",
        "AGGREGATO", spectral_avg_count_error, spectral_avg_accuracy
    );

    Ok(())
}
