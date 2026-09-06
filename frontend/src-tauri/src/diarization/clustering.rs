//! Final speaker clustering, reimplemented in Rust (agglomerative, average-linkage,
//! cosine similarity) -- see `docs/adr/0017-...md` for why this isn't a wrapper around
//! sherpa-onnx's `FastClustering` (not exposed standalone in any public API).
//!
//! Threshold empirically recalibrated to ~0.6 for this pipeline shape (independently
//! re-computed per-segment embeddings, not sherpa-onnx's internal ones -- 0.85 was only
//! valid for the internal clustering used by the old spike). Validated against the same
//! test audio used in the original spikes: 5/6 files gave the exact expected speaker
//! count, the 6th improved from 9 spurious clusters down to 3 with one residual
//! ambiguous segment.
//!
//! Default recommended threshold, used when `diarization/model.rs`'s config doesn't
//! override it.
pub const DEFAULT_CLUSTERING_THRESHOLD: f32 = 0.6;

/// Cosine similarity between two equal-length vectors. Returns 0.0 if either vector has
/// zero norm (degenerate embedding), rather than dividing by zero.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Agglomerative clustering (average-linkage) over speaker embeddings.
///
/// Repeatedly merges the two clusters with the highest average pairwise cosine
/// similarity, as long as that similarity is >= `threshold`. Returns one cluster label
/// per input embedding (labels are 0-indexed and contiguous, in no particular order).
///
/// `embeddings` should already be filtered to reasonably long segments (very short
/// segments produce noisy embeddings that don't cluster reliably -- see
/// `engine.rs::MIN_SEGMENT_DURATION_SECS`).
pub fn cluster_embeddings(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    loop {
        if clusters.len() <= 1 {
            break;
        }

        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for &a in &clusters[i] {
                    for &b in &clusters[j] {
                        sum += cosine_similarity(&embeddings[a], &embeddings[b]);
                        count += 1;
                    }
                }
                let avg = sum / count as f32;
                if best.map_or(true, |(_, _, best_sim)| avg > best_sim) {
                    best = Some((i, j, avg));
                }
            }
        }

        match best {
            Some((i, j, sim)) if sim >= threshold => {
                let mut merged = clusters[i].clone();
                merged.extend(clusters[j].iter().copied());
                // Remove the higher index first so the lower index stays valid.
                clusters.remove(j);
                clusters.remove(i);
                clusters.push(merged);
            }
            _ => break,
        }
    }

    let mut labels = vec![0usize; n];
    for (cluster_id, members) in clusters.iter().enumerate() {
        for &member in members {
            labels[member] = cluster_id;
        }
    }
    labels
}

/// Number of distinct clusters produced by `cluster_embeddings`'s output.
pub fn distinct_speaker_count(labels: &[usize]) -> usize {
    labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(mostly: usize, dim: usize) -> Vec<f32> {
        let mut v = vec![0.01f32; dim];
        v[mostly] = 1.0;
        v
    }

    #[test]
    fn empty_input_returns_empty_labels() {
        assert_eq!(cluster_embeddings(&[], 0.6), Vec::<usize>::new());
    }

    #[test]
    fn single_embedding_is_its_own_cluster() {
        let labels = cluster_embeddings(&[vec![1.0, 0.0, 0.0]], 0.6);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn two_clearly_separated_speakers_cluster_correctly() {
        // Two tight groups of near-identical vectors (same speaker), far apart from
        // each other (different speaker) -- mirrors the real embedding similarity
        // matrices observed in the validation prototype (e.g. 3-two-speakers-en.wav).
        let dim = 8;
        let embeddings = vec![
            unit(0, dim),
            unit(0, dim),
            unit(0, dim),
            unit(1, dim),
            unit(1, dim),
        ];
        let labels = cluster_embeddings(&embeddings, 0.6);
        assert_eq!(distinct_speaker_count(&labels), 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn ambiguous_segment_does_not_force_a_merge_below_threshold() {
        // A segment equidistant from both real clusters (similarity below threshold to
        // both) should not be merged into either -- it becomes its own singleton
        // cluster rather than silently attaching to the nearest one. This mirrors the
        // one residual hard segment observed in 2-two-speakers-en.wav.
        let dim = 8;
        let mut ambiguous = vec![0.01f32; dim];
        ambiguous[0] = 0.5;
        ambiguous[1] = 0.5;

        let embeddings = vec![unit(0, dim), unit(0, dim), unit(1, dim), unit(1, dim), ambiguous];
        let labels = cluster_embeddings(&embeddings, 0.85);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
        // The ambiguous segment must not share a label with either real cluster.
        assert_ne!(labels[4], labels[0]);
        assert_ne!(labels[4], labels[2]);
    }

    #[test]
    fn cosine_similarity_of_identical_vectors_is_one() {
        let v = vec![0.3, 0.1, -0.2, 0.7];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_similarity_handles_zero_vector_without_panicking() {
        let zero = vec![0.0, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&zero, &v), 0.0);
    }
}
