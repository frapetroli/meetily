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

use nalgebra::{DMatrix, SymmetricEigen};

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

// ============================================================================
// Spectral clustering (experimental alternative to `cluster_embeddings`)
// ============================================================================
//
// NME-SC family (eigengap heuristic on a pruned, normalized-Laplacian cosine affinity
// graph -- Park et al. 2019 "Auto-Tuning Spectral Clustering for Speaker Diarization
// Using Normalized Maximum Eigengap"; von Luxburg 2007 "A Tutorial on Spectral
// Clustering"; Ng-Jordan-Weiss 2002 for the normalized-Laplacian + row-normalized-
// eigenvector formulation). Estimates the number of speakers automatically instead of
// requiring a fixed similarity threshold -- see docs/sviluppi/diarization/Architettura
// pipeline.md ("Prima calibrazione reale...") for why `cluster_embeddings`'s threshold
// approach, even recalibrated (ADR-0022), still leaves real recordings wildly
// over-segmented (e.g. 55 detected vs 4 true speakers). Experimental: not wired into
// `DiarizationEngine::finalize()` (production) yet -- only into
// `finalize_with_spectral()`, used by `examples/diarization_calibration.rs` to A/B test
// against the threshold approach on real recordings before any production decision.
//
// This implements the *core* technique, not the full NME auto-search over the
// neighbor-pruning parameter `p` (Park et al.'s "Normalized Maximum Eigengap" search
// tries many candidate `p` values and picks the one maximizing a normalized eigengap
// score) -- a single defensible formula for `p` is used instead (see
// `nearest_neighbor_count`). Flagged as a documented possible future refinement, not
// blocking for this first experiment.

/// p-nearest-neighbor count for the graph-pruning step, before computing the Laplacian.
/// `ln(n)` matches the classic random-graph connectivity threshold (a k-NN graph needs
/// roughly `k ~ log(n)` neighbors per node to be connected with high probability -- von
/// Luxburg 2007, graph construction section); the `+1` is a small safety margin since
/// that result is asymptotic and this project's `n` is small (single digits to a few
/// hundred segments after `MIN_SEGMENT_DURATION_SECS` filtering), where the asymptotics
/// are weak. Deliberately not a fixed *fraction* of `n` (the other common heuristic,
/// e.g. 2-5%): at n=32 that would give p=1, risking disconnecting a genuine same-speaker
/// subgroup that happens to have few segments -- fraction-based heuristics are tuned for
/// corpora with thousands of points, not this project's regime.
fn nearest_neighbor_count(n: usize) -> usize {
    let raw = (n as f64).ln().round() as usize + 1;
    raw.clamp(2, n.saturating_sub(1).max(2))
}

/// Prunes the full affinity matrix to each node's `p` nearest neighbors, then
/// symmetrizes via `max(A, Aᵀ)` (the "OR" graph: keep edge (i,j) if *either* i ranks j
/// in its own top-p or j ranks i in its own top-p) -- not the average, and not the
/// stricter "mutual"/AND graph (both sides must agree). In this project's small-`n`
/// regime, over-sparsification (accidentally disconnecting a genuine sub-cluster) is the
/// bigger risk than over-connection, and averaging would halve a genuine strong edge
/// whenever only one side reciprocated it (e.g. a "hub" segment with many strong
/// neighbors getting outvoted by a low-degree segment that didn't rank it in its own
/// top-p) -- the more permissive OR convention avoids that.
///
/// Does not preserve the diagonal (self-similarity) -- graph Laplacians are defined over
/// edges between distinct nodes, a self-loop would double-count in the degree.
fn prune_to_p_nearest_neighbors(affinity: &[Vec<f64>], p: usize) -> Vec<Vec<f64>> {
    let n = affinity.len();
    let mut directed = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut neighbors: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        neighbors.sort_by(|&a, &b| {
            affinity[i][b]
                .partial_cmp(&affinity[i][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in neighbors.iter().take(p) {
            directed[i][j] = affinity[i][j];
        }
    }
    let mut sym = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            sym[i][j] = directed[i][j].max(directed[j][i]);
        }
    }
    sym
}

const SPECTRAL_DEGREE_EPSILON: f64 = 1e-9;

/// Lloyd's k-means on already-embedded points (deterministic "farthest-first" init, no
/// random restarts -- deliberate: with N/K both small here (spectral embeddings of a
/// few hundred segments into under ~20 dimensions at most), a single deterministic run
/// is both fast enough and, crucially, exactly reproducible for hand-verified unit
/// tests. Farthest-first is also a good fit for this specific input: row-normalized
/// spectral embeddings tend to already be well-separated per true cluster, so greedily
/// seeding one center per far-apart point reliably lands one seed per cluster.
fn kmeans(points: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = points.len();
    if k <= 1 || n <= 1 {
        return vec![0; n];
    }
    let dim = points[0].len();

    fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    let mut centers: Vec<Vec<f64>> = vec![points[0].clone()];
    let mut min_dist: Vec<f64> = points.iter().map(|p| sq_dist(p, &centers[0])).collect();
    while centers.len() < k {
        let (next, _) = min_dist
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        centers.push(points[next].clone());
        let last = centers.len() - 1;
        for (i, p) in points.iter().enumerate() {
            min_dist[i] = min_dist[i].min(sq_dist(p, &centers[last]));
        }
    }

    const KMEANS_MAX_ITERS: usize = 100;
    let mut assignment = vec![0usize; n];
    for _ in 0..KMEANS_MAX_ITERS {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let (best_c, _) = centers
                .iter()
                .enumerate()
                .map(|(c, ctr)| (c, sq_dist(p, ctr)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            if assignment[i] != best_c {
                assignment[i] = best_c;
                changed = true;
            }
        }

        let mut sums = vec![vec![0.0f64; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            counts[assignment[i]] += 1;
            for d in 0..dim {
                sums[assignment[i]][d] += p[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Empty cluster: reseed at the point farthest from its own current center.
                let (farthest, _) = points
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (i, sq_dist(p, &centers[assignment[i]])))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                centers[c] = points[farthest].clone();
            } else {
                centers[c] = sums[c].iter().map(|s| s / counts[c] as f64).collect();
            }
        }
        if !changed {
            break;
        }
    }
    assignment
}

/// Spectral clustering with automatic speaker-count estimation. See the module-level
/// comment above this section for the algorithm family and rationale. `max_speakers`
/// upper-bounds the estimated count (clamped to >= 1) -- pass a generous value (e.g. 20),
/// not a tight guess: if the true cluster count is at or beyond the cap, the eigengap
/// search can collapse the estimate well below the cap rather than simply clipping to it
/// (see the doc comment on the K-estimation loop below, and the
/// `spectral_true_k_exceeding_max_speakers_never_exceeds_the_cap` test).
///
/// Returns one 0-indexed cluster label per input embedding -- same contract as
/// `cluster_embeddings`, drops into the same downstream pipeline (`normalize_labels`,
/// `SpeakerSegment` construction) unchanged.
pub fn cluster_embeddings_spectral(embeddings: &[Vec<f32>], max_speakers: usize) -> Vec<usize> {
    cluster_embeddings_spectral_with_p(embeddings, max_speakers, None)
}

/// Same as `cluster_embeddings_spectral`, but lets the caller override the p-nearest-
/// neighbor pruning count instead of always using `nearest_neighbor_count`'s `ln(n)+1`
/// formula. Exists for `examples/diarization_calibration.rs` to experiment with `p` on
/// real recordings (the two hardest calibration files, with more true speakers, under-
/// estimated the speaker count with the default formula -- see docs/sviluppi/diarization/
/// Architettura pipeline.md, "Primo confronto reale..." -- a larger `p` is one plausible
/// fix, to be validated empirically like everything else in this file, not assumed).
/// `None` reproduces the exact default-formula behavior of `cluster_embeddings_spectral`.
pub fn cluster_embeddings_spectral_with_p(
    embeddings: &[Vec<f32>],
    max_speakers: usize,
    p_override: Option<usize>,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    let max_speakers = max_speakers.max(1);

    // N=2 is a special case, not just an optimization: for exactly 2 nodes, the
    // symmetric normalized Laplacian's eigenvalues are always exactly {0, 2} whenever
    // the two points have any positive similarity, *regardless of the similarity's
    // magnitude* (the degree term cancels out algebraically: D^(-1/2) A D^(-1/2) reduces
    // to [[0,1],[1,0]] for any single positive edge weight w>0, whose eigenvalues are
    // {1,-1}, giving Laplacian eigenvalues {1-1, 1-(-1)} = {0,2} independent of w). The
    // eigengap step is therefore mathematically uninformative at n=2 -- it cannot tell
    // "barely similar" from "very similar" -- so a direct threshold comparison is used
    // instead (the one deliberate point of contact with the calibrated threshold
    // approach).
    if n == 2 {
        let similar =
            cosine_similarity(&embeddings[0], &embeddings[1]) >= DEFAULT_CLUSTERING_THRESHOLD;
        return if similar || max_speakers < 2 {
            vec![0, 0]
        } else {
            vec![0, 1]
        };
    }

    // Affinity matrix: cosine similarity, negatives clipped to zero (graph Laplacian
    // theory assumes nonnegative edge weights).
    let mut affinity = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]).max(0.0) as f64;
            affinity[i][j] = sim;
            affinity[j][i] = sim;
        }
    }

    let p = p_override
        .unwrap_or_else(|| nearest_neighbor_count(n))
        .clamp(2, n.saturating_sub(1).max(2));
    let mut affinity_final = prune_to_p_nearest_neighbors(&affinity, p);

    let mut degree = vec![0.0f64; n];
    for i in 0..n {
        degree[i] = affinity_final[i].iter().sum();
        if degree[i] < SPECTRAL_DEGREE_EPSILON {
            // Degenerate: this segment has ~zero similarity to every other segment
            // (should not happen with real speech embeddings, but guarded rather than
            // assumed impossible). Treat as its own trivial component via a self-loop.
            affinity_final[i][i] = 1.0;
            degree[i] = 1.0;
        }
    }

    // Symmetric normalized Laplacian: L = I - D^(-1/2) A D^(-1/2). Chosen over the
    // unnormalized L = D - A (whose eigenvalues scale with max degree, not comparable
    // in magnitude across recordings of very different sizes -- bad for a "biggest gap"
    // heuristic meant to run unattended on many different recordings) and over the
    // random-walk normalization (Ng-Jordan-Weiss 2002 established the symmetric
    // normalization as the robust choice for exactly this eigengap-based approach).
    let d_inv_sqrt: Vec<f64> = degree.iter().map(|d| 1.0 / d.sqrt()).collect();
    let laplacian_raw = DMatrix::from_fn(n, n, |r, c| {
        let identity = if r == c { 1.0 } else { 0.0 };
        identity - d_inv_sqrt[r] * affinity_final[r][c] * d_inv_sqrt[c]
    });
    // Defensive re-symmetrization against floating-point asymmetry, built explicitly via
    // from_fn (not nalgebra's operator overloads) to keep every step here unambiguous
    // without a compiler on hand to double-check trait resolution.
    let laplacian_t = laplacian_raw.transpose();
    let laplacian = DMatrix::from_fn(n, n, |r, c| 0.5 * (laplacian_raw[(r, c)] + laplacian_t[(r, c)]));

    let eigen = SymmetricEigen::new(laplacian);

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_eigenvalues: Vec<f64> = order.iter().map(|&i| eigen.eigenvalues[i]).collect();

    // Eigengap: the biggest jump between consecutive sorted eigenvalues, searched only
    // within 1..=k_max so K is never 0 and never exceeds the cap. Ties prefer the
    // smaller k -- deliberate, since the entire premise of this experiment is that the
    // threshold approach *over*-clusters.
    //
    // Non-obvious, hand-verified behavior: if the graph's true number of components
    // exceeds max_speakers, this does NOT generally land on k = max_speakers. If every
    // candidate gap within the capped search range sits on the same near-zero
    // "eigenvalue-zero plateau" (i.e. the capped range ends before the real structural
    // gap), every gap in range ties at ~0 and the tie-break rule collapses the estimate
    // down to k=1 -- still respects "never exceed the cap", just not the naively
    // expected "clip to the cap". See the
    // spectral_true_k_exceeding_max_speakers_never_exceeds_the_cap test. This is why
    // callers should pass a generous max_speakers, not a tight guess.
    let k_max = max_speakers.min(n - 1);
    let mut best_k = 1usize;
    let mut best_gap = f64::MIN;
    for k in 1..=k_max {
        let gap = sorted_eigenvalues[k] - sorted_eigenvalues[k - 1];
        if gap > best_gap {
            best_gap = gap;
            best_k = k;
        }
    }
    let k = best_k;

    if k == 1 {
        return vec![0; n];
    }

    // Row-normalized spectral embedding (Ng-Jordan-Weiss 2002): in the ideal
    // block-diagonal case, points in the same connected component get eigenvector
    // entries that are constant but scaled by that point's own degree -- row-normalizing
    // to unit length removes this per-point scale so members of the same true cluster
    // map to the *same* point on the unit hypersphere regardless of individual degree
    // variation (e.g. a "chatty" segment with many neighbors vs. a sparser one).
    let mut rows: Vec<Vec<f64>> = vec![vec![0.0; k]; n];
    for row in 0..n {
        for (col, &eig_idx) in order.iter().take(k).enumerate() {
            rows[row][col] = eigen.eigenvectors[(row, eig_idx)];
        }
        let norm: f64 = rows[row].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > SPECTRAL_DEGREE_EPSILON {
            for v in rows[row].iter_mut() {
                *v /= norm;
            }
        }
    }

    kmeans(&rows, k)
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

    // ------------------------------------------------------------------------------
    // cluster_embeddings_spectral
    //
    // These deliberately use *exactly* orthogonal one-hot vectors (`vec![1.0, 0.0, ...]`),
    // not the `unit(mostly, dim)` helper above (which has 0.01 leakage in every other
    // dimension). With exact orthogonality, cross-cluster cosine similarity is a literal
    // 0.0, so the pruned/symmetrized graph is exactly block-diagonal *regardless* of the
    // p-nearest-neighbor selection or tie-breaking order -- which is what makes the
    // expected eigenvalues hand-computable at all without running the code. `unit()`'s
    // leakage would reintroduce small nonzero cross-similarities whose effect on the
    // eigengap can't be pinned down by hand.
    // ------------------------------------------------------------------------------

    #[test]
    fn spectral_empty_input_returns_empty_labels() {
        assert_eq!(cluster_embeddings_spectral(&[], 5), Vec::<usize>::new());
    }

    #[test]
    fn spectral_single_embedding_is_its_own_cluster() {
        assert_eq!(cluster_embeddings_spectral(&[vec![1.0, 0.0, 0.0]], 5), vec![0]);
    }

    #[test]
    fn spectral_two_similar_embeddings_merge() {
        // Identical vectors: cosine similarity 1.0 >= DEFAULT_CLUSTERING_THRESHOLD (0.6)
        // -- n=2 is a special case (see cluster_embeddings_spectral's doc comment): the
        // eigengap is mathematically uninformative at n=2, so this falls back to a
        // direct threshold comparison.
        let e = vec![1.0, 0.0, 0.0];
        assert_eq!(cluster_embeddings_spectral(&[e.clone(), e], 5), vec![0, 0]);
    }

    #[test]
    fn spectral_two_dissimilar_embeddings_split() {
        // Orthogonal vectors: cosine similarity 0.0 < DEFAULT_CLUSTERING_THRESHOLD.
        let labels = cluster_embeddings_spectral(&[vec![1.0, 0.0], vec![0.0, 1.0]], 5);
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn spectral_two_well_separated_clusters_estimates_k_two() {
        // Two orthogonal one-hot directions (dim=2), 3 identical copies each. Hand-derived
        // eigenvalues: each group of 3 identical vectors forms a complete weight-1 K3
        // subgraph (cross-group similarity is exactly 0.0, so it can never contribute to
        // the pruned graph regardless of which neighbors p-NN selects). A weight-1 K3 has
        // adjacency eigenvalues {2, -1, -1} (standard complete-graph spectrum) with
        // uniform degree d=2, so D^(-1/2)AD^(-1/2) = A/2 has eigenvalues {1, -0.5, -0.5},
        // giving Laplacian eigenvalues I - that = {0, 1.5, 1.5} per block. Two disjoint
        // blocks -> overall sorted eigenvalues [0, 0, 1.5, 1.5, 1.5, 1.5]. Gaps:
        // gap(1)=0, gap(2)=1.5, gap(3..5)=0 -- unique max at k=2.
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let embeddings = vec![a.clone(), a.clone(), a, b.clone(), b.clone(), b];
        let labels = cluster_embeddings_spectral(&embeddings, 5);
        assert_eq!(distinct_speaker_count(&labels), 2);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn spectral_three_well_separated_clusters_estimates_k_three() {
        // Three orthogonal one-hot directions (dim=3), 2 identical copies each. Each pair
        // forms a disjoint weight-1 K2 (single edge, degree=1): D^(-1/2)AD^(-1/2) reduces
        // to [[0,1],[1,0]] (eigenvalues {1,-1}), giving Laplacian eigenvalues {0, 2} per
        // block. Three disjoint blocks -> sorted eigenvalues [0, 0, 0, 2, 2, 2]. Gaps:
        // gap(1)=0, gap(2)=0, gap(3)=2, gap(4)=0, gap(5)=0 -- unique max at k=3.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];
        let embeddings = vec![a.clone(), a, b.clone(), b, c.clone(), c];
        let labels = cluster_embeddings_spectral(&embeddings, 5);
        assert_eq!(distinct_speaker_count(&labels), 3);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[2]);
        assert_ne!(labels[0], labels[4]);
        assert_ne!(labels[2], labels[4]);
    }

    #[test]
    fn spectral_true_k_exceeding_max_speakers_never_exceeds_the_cap() {
        // Same embeddings as spectral_three_well_separated_clusters_estimates_k_three
        // (true K=3, eigenvalues [0,0,0,2,2,2] exactly), but max_speakers=2 this time.
        //
        // Hand-verified, non-obvious outcome: capping does NOT land on K=2 here. With
        // k_max=2, only gap(1)=eigenvalues[1]-eigenvalues[0]=0-0=0 and
        // gap(2)=eigenvalues[2]-eigenvalues[1]=0-0=0 are in range -- both exactly 0 (the
        // real signal, gap(3)=2, sits beyond the cap and is never examined). The tie-break
        // rule (prefer the smaller k) collapses the estimate all the way to K=1, not K=2.
        // This still satisfies "never exceed max_speakers" (1 <= 2), just not via the
        // naively-expected "clip to the cap" path -- documented here rather than
        // discovered later against real over-segmented recordings, which is exactly why
        // `examples/diarization_calibration.rs` should pass a generous max_speakers, not
        // a tight guess.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];
        let embeddings = vec![a.clone(), a, b.clone(), b, c.clone(), c];
        let labels = cluster_embeddings_spectral(&embeddings, 2);
        assert!(
            distinct_speaker_count(&labels) <= 2,
            "must never exceed max_speakers"
        );
        assert_eq!(
            distinct_speaker_count(&labels),
            1,
            "collapses to 1 here -- see comment above"
        );
    }

    #[test]
    fn spectral_with_p_none_matches_the_default_formula() {
        // cluster_embeddings_spectral must be exactly cluster_embeddings_spectral_with_p
        // with p_override=None -- a regression guard on the delegation itself, not on the
        // algorithm (already covered by the tests above).
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let embeddings = vec![a.clone(), a.clone(), a, b.clone(), b.clone(), b];
        let via_default = cluster_embeddings_spectral(&embeddings, 5);
        let via_explicit_none = cluster_embeddings_spectral_with_p(&embeddings, 5, None);
        assert_eq!(via_default, via_explicit_none);
    }

    #[test]
    fn spectral_with_p_override_clamps_without_panicking() {
        // Extreme p_override values (0, and far beyond n) must clamp instead of panicking
        // -- exact resulting labels aren't asserted here (unlike the hand-verified tests
        // above, changing p can change the graph structure in ways not worth re-deriving
        // by hand for this regression guard), only that the function stays well-behaved.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];
        let embeddings = vec![a.clone(), a, b.clone(), b, c.clone(), c];

        let low = cluster_embeddings_spectral_with_p(&embeddings, 5, Some(0));
        assert_eq!(low.len(), embeddings.len());

        let high = cluster_embeddings_spectral_with_p(&embeddings, 5, Some(1000));
        assert_eq!(high.len(), embeddings.len());
    }
}
