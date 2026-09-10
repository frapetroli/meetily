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

/// Clusters smaller than this (after `cluster_embeddings`) are treated as unreliable --
/// likely a noisy/short segment that failed to reach `DEFAULT_CLUSTERING_THRESHOLD`
/// against its true speaker, rather than a genuine extra participant -- and become
/// candidates for `reattach_small_clusters`.
pub const MIN_CLUSTER_SIZE: usize = 2;

/// Secondary threshold used only by `reattach_small_clusters`, never by the main
/// `cluster_embeddings` pass -- applied only to clusters smaller than `MIN_CLUSTER_SIZE`,
/// giving noisy tiny clusters (short backchannel interjections, cross-talk) a second
/// chance to reattach to an existing speaker instead of minting a new one. See
/// `docs/sviluppi/diarization/Architettura pipeline.md`, section "Validazione reale...".
///
/// **Calibrated against real data, not a guess**: see `docs/adr/0022-reattach-threshold-
/// ricalibrato-0.60-dati-reali.md`. A grid sweep (`examples/diarization_calibration.rs`)
/// against 5 real multi-speaker recordings with reference transcripts showed per-turn
/// accuracy climbing as this threshold rises from 0.45 up to `DEFAULT_CLUSTERING_THRESHOLD`
/// (0.6), then plateauing exactly there -- pushing higher only inflated the spurious
/// speaker count further with no further accuracy gain (sometimes literally zero, e.g.
/// stuck at 90.6% from 0.60 through 0.70 on one real recording). Intentionally set equal
/// to `DEFAULT_CLUSTERING_THRESHOLD` rather than lower, contrary to this constant's
/// original (unvalidated) design intent of being "more permissive" -- the data didn't
/// support that. **Honest caveat**: even at this calibrated value, detected speaker counts
/// on real recordings stay far above ground truth (e.g. 55 vs 4, 389 vs 8) -- this
/// threshold alone does not solve the underlying over-segmentation, it only measurably
/// improves it. The remaining gap likely comes from embedding quality on short/noisy real
/// segments, not from this threshold -- separate future work, not yet investigated.
pub const REATTACH_THRESHOLD: f32 = 0.60;

/// Second pass over `cluster_embeddings`'s output: folds clusters smaller than
/// `min_cluster_size` into the nearest larger cluster, if their average cosine similarity
/// to it clears `reattach_threshold`. Otherwise a small cluster is left exactly as-is --
/// a genuinely ambiguous or distinct short segment still ends up as its own cluster,
/// same as before this pass existed.
///
/// Runs a single pass over the small clusters (does not re-check newly grown clusters
/// against each other, and never merges two small clusters together) -- sufficient for
/// the observed failure mode (many independent singletons scattered around a couple of
/// real speakers), and avoids the risk of chaining unrelated tiny clusters into one.
///
/// If every cluster is "small" (no cluster reaches `min_cluster_size`), there is nothing
/// reliable to reattach to, so the input is returned unchanged.
pub fn reattach_small_clusters(
    embeddings: &[Vec<f32>],
    labels: &[usize],
    min_cluster_size: usize,
    reattach_threshold: f32,
) -> Vec<usize> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let mut members_by_label: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        members_by_label.entry(label).or_default().push(idx);
    }

    let (small, large): (Vec<_>, Vec<_>) = members_by_label
        .into_iter()
        .partition(|(_, members)| members.len() < min_cluster_size);

    if large.is_empty() {
        return labels.to_vec();
    }

    let mut new_labels = labels.to_vec();
    for (_, small_members) in &small {
        let mut best: Option<(usize, f32)> = None;
        for (large_label, large_members) in &large {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for &a in small_members {
                for &b in large_members {
                    sum += cosine_similarity(&embeddings[a], &embeddings[b]);
                    count += 1;
                }
            }
            let avg = sum / count as f32;
            if best.map_or(true, |(_, best_sim)| avg > best_sim) {
                best = Some((*large_label, avg));
            }
        }
        if let Some((target_label, sim)) = best {
            if sim >= reattach_threshold {
                for &member in small_members {
                    new_labels[member] = target_label;
                }
            }
        }
    }

    new_labels
}

/// Remaps arbitrary label ids (e.g. left non-contiguous by `reattach_small_clusters`
/// folding some labels away entirely) to a contiguous 0-indexed range, preserving
/// first-seen order. Cosmetic only: never changes which inputs share a label.
pub fn normalize_labels(labels: &[usize]) -> Vec<usize> {
    let mut next_id = 0usize;
    let mut remap: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    labels
        .iter()
        .map(|&label| {
            *remap.entry(label).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            })
        })
        .collect()
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

    #[test]
    fn singleton_reattaches_to_the_nearest_real_cluster() {
        // s is unit-length by construction (0.5^2 + 0.15^2 + 0.853^2 ~= 1), so its cosine
        // similarity to c0=[1,0,0] is exactly 0.5 and to c1=[0,1,0] exactly 0.15 -- below
        // DEFAULT_CLUSTERING_THRESHOLD (0.6) to both, so the main pass leaves it a
        // singleton, but above the reattach threshold used in this test (0.45) to c0 only.
        // Deliberately a literal here, not the live `REATTACH_THRESHOLD` constant -- these
        // tests exercise `reattach_small_clusters`'s logic at a fixed, hand-verified
        // boundary, independent of whatever the constant is calibrated to today (see
        // docs/adr/0022-reattach-threshold-ricalibrato-0.60-dati-reali.md), same style as
        // `ambiguous_segment_does_not_force_a_merge_below_threshold` above already uses a
        // literal 0.85 instead of `DEFAULT_CLUSTERING_THRESHOLD`.
        let c0 = vec![1.0, 0.0, 0.0];
        let c1 = vec![0.0, 1.0, 0.0];
        let s = vec![0.5, 0.15, 0.853];
        let embeddings = vec![c0.clone(), c0.clone(), c0, c1.clone(), c1, s];

        let labels = cluster_embeddings(&embeddings, DEFAULT_CLUSTERING_THRESHOLD);
        assert_eq!(distinct_speaker_count(&labels), 3, "singleton must not auto-merge under the main threshold");
        let singleton_label = labels[5];
        assert_ne!(singleton_label, labels[0]);
        assert_ne!(singleton_label, labels[3]);

        let reattached = reattach_small_clusters(&embeddings, &labels, MIN_CLUSTER_SIZE, 0.45);
        assert_eq!(distinct_speaker_count(&reattached), 2, "singleton should fold into cluster 0");
        assert_eq!(reattached[5], reattached[0]);
        assert_ne!(reattached[5], reattached[3]);
    }

    #[test]
    fn singleton_too_far_from_everything_stays_its_own_cluster() {
        // Same construction as above, but similarity to both real clusters (0.3 and 0.2)
        // is below the 0.45 reattach threshold used here -- forcing a merge here would be
        // exactly the "silently attach to the nearest cluster regardless of confidence"
        // behavior the ambiguous-segment test above already guards against for the main
        // pass. See the literal-vs-constant note in the test above.
        let c0 = vec![1.0, 0.0, 0.0];
        let c1 = vec![0.0, 1.0, 0.0];
        let s = vec![0.3, 0.2, 0.9327];
        let embeddings = vec![c0.clone(), c0.clone(), c0, c1.clone(), c1, s];

        let labels = cluster_embeddings(&embeddings, DEFAULT_CLUSTERING_THRESHOLD);
        let reattached = reattach_small_clusters(&embeddings, &labels, MIN_CLUSTER_SIZE, 0.45);
        assert_eq!(distinct_speaker_count(&reattached), 3, "below-threshold singleton must remain isolated");
        assert_ne!(reattached[5], reattached[0]);
        assert_ne!(reattached[5], reattached[3]);
    }

    #[test]
    fn no_large_cluster_to_reattach_to_leaves_labels_unchanged() {
        // Three mutually distant embeddings: cluster_embeddings gives each its own label,
        // and every resulting cluster is "small" (size 1 < MIN_CLUSTER_SIZE) -- there is
        // nothing reliable to fold into, so the pass must be a no-op regardless of threshold.
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let labels = cluster_embeddings(&embeddings, DEFAULT_CLUSTERING_THRESHOLD);
        assert_eq!(distinct_speaker_count(&labels), 3);

        let reattached = reattach_small_clusters(&embeddings, &labels, MIN_CLUSTER_SIZE, 0.45);
        assert_eq!(reattached, labels);
    }

    #[test]
    fn no_small_clusters_present_leaves_labels_unchanged() {
        let dim = 8;
        let embeddings = vec![unit(0, dim), unit(0, dim), unit(1, dim), unit(1, dim)];
        let labels = cluster_embeddings(&embeddings, DEFAULT_CLUSTERING_THRESHOLD);
        let reattached = reattach_small_clusters(&embeddings, &labels, MIN_CLUSTER_SIZE, 0.45);
        assert_eq!(reattached, labels);
    }

    #[test]
    fn normalize_labels_remaps_to_contiguous_ids_preserving_first_seen_order() {
        assert_eq!(normalize_labels(&[5, 5, 2, 2, 9]), vec![0, 0, 1, 1, 2]);
        assert_eq!(normalize_labels(&[]), Vec::<usize>::new());
        assert_eq!(normalize_labels(&[0, 0, 0]), vec![0, 0, 0]);
    }
}
