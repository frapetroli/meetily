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
// NME-SC (Normalized Maximum Eigengap Spectral Clustering) -- Park, Han, Kumar,
// Narayanan, "Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized
// Maximum Eigengap", IEEE Signal Processing Letters 2019 (arXiv:2003.02405), later
// adopted into NVIDIA NeMo's diarization pipeline. This is a faithful translation of the
// authors' own reference implementation (github.com/tango4j/Auto-Tuning-Spectral-
// Clustering, spectral_opt.py -- `NMEanalysis`/`estimate_num_of_spkrs`/
// `get_kneighbors_conn`/`get_X_conn_from_dist`/`getLaplacian`/`getLamdaGaplist`, fetched
// and verified against that source in September 2026), not an independent design -- see
// docs/sviluppi/diarization/Architettura pipeline.md ("Tentativo di fix: p fisso...") for
// why a single p value (however chosen) proved unable to generalize across recordings of
// very different size and noise characteristics, which is exactly the problem this
// per-recording auto-search exists to solve. Estimates the number of speakers
// automatically instead of requiring a fixed similarity threshold -- see "Prima
// calibrazione reale..." for why `cluster_embeddings`'s threshold approach, even
// recalibrated (ADR-0022), still leaves real recordings wildly over-segmented.
// Experimental: not wired into `DiarizationEngine::finalize()` (production) yet -- only
// into `finalize_with_spectral()`, used by `examples/diarization_calibration.rs` to A/B
// test against the threshold approach on real recordings before any production decision.
//
// Key differences from this file's first (non-reference) attempt at this technique,
// corrected after reading the real reference source instead of re-deriving the method
// from theory alone:
// - The k-NN graph is a BINARY connectivity graph (0/1, then 0/0.5/1 after averaging
//   symmetrization), not a graph weighted by the actual cosine similarity values.
// - The Laplacian is the *unnormalized* `L = D - A`, not the symmetric normalized
//   `I - D^-1/2 A D^-1/2` used before -- comparability across recordings of different
//   sizes comes instead from normalizing the *eigengap* by the largest eigenvalue when
//   scoring candidate `p` values (see `nme_select_p`), not from normalizing the
//   Laplacian itself.
// - `p` is not one fixed formula: many candidate values are tried per recording (up to
//   `MAX_RP_THRESHOLD` of the segment count, matching the reference's own default) and
//   the one minimizing a specific ratio -- (fraction of segments connected) / (how
//   confident the resulting eigengap is) -- is kept. This is the actual "auto-tuning"
//   the NME name refers to; the first attempt's `ln(n)+1` formula was a simplification
//   that a real calibration run (see the doc referenced above) showed does not
//   generalize: a single p that works for a small recording actively hurts a large one
//   and vice versa.
// - No row-normalization of the eigenvectors before k-means (the reference hardcodes
//   `norm_laplacian=False`) -- the first attempt added a Ng-Jordan-Weiss
//   row-normalization step the reference does not use; removed for fidelity.
//
// Deliberate, documented deviation kept from the first attempt: k-means (see `kmeans`
// below) uses a deterministic "farthest-first" initialization and a single run, not
// scikit-learn's `KMeans(n_init=10)` (10 random restarts) that the reference uses --
// determinism was worth more than matching this specific detail, given there is no
// compiler available in the sandbox this was written in to catch a randomness-related
// bug before it reaches a real machine.
//
// Honest testing caveat (see the test module): the exact hand-derived eigenvalues used
// to verify this file's first attempt do not transfer cleanly to this version. Two real
// quirks of the reference algorithm interact badly with the perfectly-orthogonal,
// exactly-zero-cross-similarity test vectors used before: (1) the diagonal (self-
// similarity, always the maximum possible value) is only zeroed out *after* neighbor
// selection, so one neighbor "slot" per node is always spent on itself -- faithfully
// reproduced here, see `build_knn_connectivity`'s doc comment; (2) the connectivity
// fallback below actively fights exactly-zero cross-cluster similarity (it keeps raising
// `p` until the *whole* graph is one connected component, which an exactly
// block-diagonal similarity matrix resists by construction). Real embeddings never have
// exactly zero cross-speaker similarity, so this is a test-construction artifact, not a
// real-world concern -- but it does mean the tests below verify sub-components and
// qualitative/structural outcomes on realistic (non-adversarial) inputs, rather than
// exact end-to-end eigenvalues the way the previous version's tests could.

/// Builds the k-nearest-neighbor connectivity graph used by the NME search and the final
/// clustering: for each node, keep only its `p` most similar neighbors (by cosine
/// similarity, `similarity[i]` treated as node `i`'s row, self-similarity included in the
/// ranking), as a binary (0/1) connection, then symmetrize by *averaging* (not max) -- an
/// edge only one side reciprocated ends at weight 0.5, one both sides agree on ends at
/// weight 1.0. Matches the reference's `get_kneighbors_conn`/`get_X_conn_from_dist`
/// exactly, including the quirk that self-similarity (always the maximum possible value)
/// occupies one of the `p` neighbor slots -- harmless, since
/// `laplacian_from_connectivity` zeroes the diagonal before computing degrees, same as
/// the reference's `getLaplacian`. A direct, real consequence worth knowing: `p=1` alone
/// is always degenerate (every node's only selected neighbor is itself, zeroed away by
/// the Laplacian step, leaving zero real edges) -- `nme_select_p`'s connectivity fallback
/// exists partly to route around exactly this.
fn build_knn_connectivity(similarity: &[Vec<f64>], p: usize) -> Vec<Vec<f64>> {
    let n = similarity.len();
    let mut directed = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            similarity[i][b]
                .partial_cmp(&similarity[i][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in idx.iter().take(p.min(n)) {
            directed[j][i] = 1.0;
        }
    }
    let mut sym = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            sym[i][j] = 0.5 * (directed[i][j] + directed[j][i]);
        }
    }
    sym
}

/// True if the connectivity graph (any nonzero entry treated as an edge) is a single
/// connected component, via breadth-first search from node 0. Matches the reference's
/// `isFullyConnected`/`_graph_connected_component`. `n == 0` is vacuously true (never
/// actually reached here -- `cluster_embeddings_spectral_with_p` returns early for
/// `n <= 2` before any graph is built).
fn is_fully_connected(graph: &[Vec<f64>]) -> bool {
    let n = graph.len();
    if n == 0 {
        return true;
    }
    let mut visited = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    visited[0] = true;
    queue.push_back(0usize);
    let mut count = 1usize;
    while let Some(node) = queue.pop_front() {
        for (neighbor, &weight) in graph[node].iter().enumerate() {
            if weight > 0.0 && !visited[neighbor] {
                visited[neighbor] = true;
                count += 1;
                queue.push_back(neighbor);
            }
        }
    }
    count == n
}

/// Unnormalized graph Laplacian `L = D - A`, `D` the diagonal degree matrix (row sums of
/// `|A|`, diagonal of `A` excluded from the sum and zeroed in `L`). Matches the
/// reference's `getLaplacian` exactly -- deliberately *not* the symmetric-normalized
/// Laplacian this file's first attempt used (see the module-level comment above for why
/// comparability across recordings comes from elsewhere in this version).
fn laplacian_from_connectivity(graph: &[Vec<f64>]) -> DMatrix<f64> {
    let n = graph.len();
    let degree: Vec<f64> = (0..n)
        .map(|i| (0..n).filter(|&j| j != i).map(|j| graph[i][j].abs()).sum())
        .collect();
    DMatrix::from_fn(n, n, |r, c| if r == c { degree[r] } else { -graph[r][c] })
}

/// Ascending-sorted eigenvalues of a symmetric matrix, its (unsorted) eigenvector matrix,
/// and the permutation needed to read off eigenvector *columns* in the same sorted order
/// (nalgebra's `SymmetricEigen` returns both eigenvalues and eigenvectors unsorted).
fn sorted_eigen(matrix: DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>, Vec<usize>) {
    let n = matrix.nrows();
    let eigen = SymmetricEigen::new(matrix);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_values: Vec<f64> = order.iter().map(|&i| eigen.eigenvalues[i]).collect();
    (sorted_values, eigen.eigenvectors, order)
}

/// Consecutive differences between ascending-sorted eigenvalues: `gaps[i] =
/// eigenvalues[i+1] - eigenvalues[i]`, one shorter than `eigenvalues`. Matches the
/// reference's `getLamdaGaplist`.
fn eigengaps(sorted_eigenvalues: &[f64]) -> Vec<f64> {
    sorted_eigenvalues.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Estimated speaker count from a gap sequence: the position of the biggest gap within
/// the first `max_speakers` gaps, plus one (`gaps[0]` separates the 1st and 2nd
/// eigenvalues, i.e. supports a 2-cluster split -- the reference's
/// `estimate_num_of_spkrs` numbers this "+1" the same way). Ties prefer the earliest
/// (smallest-k) gap, matching numpy's `argmax`. Never returns 0; returns 1 if `gaps` is
/// empty (not expected to happen given this is only called with n>=3, so gaps has at
/// least 2 entries, but guarded rather than assumed).
fn k_from_gaps(gaps: &[f64], max_speakers: usize) -> usize {
    if gaps.is_empty() {
        return 1;
    }
    let search_len = max_speakers.min(gaps.len());
    let mut best_idx = 0usize;
    let mut best_gap = f64::MIN;
    for (i, &gap) in gaps.iter().take(search_len).enumerate() {
        if gap > best_gap {
            best_gap = gap;
            best_idx = i;
        }
    }
    best_idx + 1
}

const MAX_RP_THRESHOLD: f64 = 0.25; // reference default: search p up to 25% of n
const NME_SEARCH_P_VOLUME: usize = 500; // reference default: at most this many candidates
const NME_EPSILON: f64 = 1e-10; // reference's `eps`, guards the two divisions below

/// The NME auto-search: tries multiple `p` values (up to `MAX_RP_THRESHOLD` of the
/// segment count) and keeps the one minimizing `(p/n) / (best_normalized_gap + eps)` --
/// balancing "as sparse a graph as possible" against "still shows a clean, confident
/// cluster split" (a small p with a messy/absent gap gives a *large* ratio, same as a
/// large p that's needlessly over-connected -- the minimum sits at the sparsest p that
/// still finds real structure). `best_normalized_gap` is the biggest gap within the
/// first `max_speakers` gaps, divided by the largest eigenvalue overall (`+eps`) --
/// matches the reference's `NMEanalysis` exactly, including the fallback (mirroring
/// `gc_thres_min_gc`) that raises `p` from 1 upward until the graph is fully connected,
/// if the ratio-minimizing choice wasn't.
fn nme_select_p(similarity: &[Vec<f64>], max_speakers: usize) -> usize {
    let n = similarity.len();
    // Deviation from the reference, justified by a real project-specific concern: for
    // small n (short recordings, few accumulated segments -- realistic here in a way the
    // reference's original speaker-diarization corpora, with hundreds to thousands of
    // frames, never faced), `floor(n * MAX_RP_THRESHOLD)` alone can collapse to exactly
    // 1, meaning the "search" only ever considers p=1 -- which is *always* degenerate
    // (see build_knn_connectivity's doc comment: self always wins the ranking, so p=1
    // alone selects zero real edges for anyone) and would make this function trivially
    // always report an empty/fully-disconnected graph regardless of the true structure.
    // Flooring the search ceiling at 3 gives even small recordings a non-trivial p range
    // to search (still capped at n-1, so this can't ask for more neighbors than exist).
    let max_n = (((n as f64) * MAX_RP_THRESHOLD).floor() as usize)
        .max(3)
        .min(n.saturating_sub(1).max(1));
    let candidate_count = max_n.min(NME_SEARCH_P_VOLUME).max(1);
    let p_candidates: Vec<usize> = if candidate_count == 1 {
        vec![1]
    } else {
        (0..candidate_count)
            .map(|i| {
                let t = i as f64 / (candidate_count - 1) as f64;
                (1.0 + t * (max_n as f64 - 1.0)).round() as usize
            })
            .collect()
    };

    let mut best_p = p_candidates[0];
    let mut best_ratio = f64::MAX;
    for &p in &p_candidates {
        let graph = build_knn_connectivity(similarity, p);
        let laplacian = laplacian_from_connectivity(&graph);
        let (sorted_values, _, _) = sorted_eigen(laplacian);
        let gaps = eigengaps(&sorted_values);
        let search_len = max_speakers.min(gaps.len());
        if search_len == 0 {
            continue;
        }
        let best_gap_value = gaps
            .iter()
            .take(search_len)
            .cloned()
            .fold(f64::MIN, f64::max);
        let max_eigenvalue = sorted_values.iter().cloned().fold(f64::MIN, f64::max);
        let normalized_gap = best_gap_value / (max_eigenvalue + NME_EPSILON);
        let ratio = (p as f64 / n as f64) / (normalized_gap + NME_EPSILON);
        if ratio < best_ratio {
            best_ratio = ratio;
            best_p = p;
        }
    }

    let chosen_graph = build_knn_connectivity(similarity, best_p);
    if !is_fully_connected(&chosen_graph) {
        for p in 1..=max_n {
            let graph = build_knn_connectivity(similarity, p);
            if is_fully_connected(&graph) {
                return p;
            }
        }
    }
    best_p
}

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
/// neighbor pruning count instead of running the NME auto-search (`nme_select_p`) for it.
/// Exists for `examples/diarization_calibration.rs` to experiment with a fixed `p` on
/// real recordings -- a real calibration run (see docs/sviluppi/diarization/Architettura
/// pipeline.md, "Tentativo di fix: p fisso...") showed that a single fixed `p` cannot
/// generalize across recordings of very different size, which is exactly why the
/// default (`None`) now runs the full per-recording search instead of a fixed formula.
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

    // N=2 is a special case, not just an optimization: with only 2 nodes there is only
    // ever one possible edge, so no eigengap search can say anything a direct similarity
    // comparison couldn't (and, per the module-level comment above, `p=1` alone -- the
    // only value that could apply here -- is degenerate by construction). Unaffected by
    // the Laplacian-formulation change above/below: this bypasses the graph/Laplacian
    // machinery entirely either way.
    if n == 2 {
        let similar =
            cosine_similarity(&embeddings[0], &embeddings[1]) >= DEFAULT_CLUSTERING_THRESHOLD;
        return if similar || max_speakers < 2 {
            vec![0, 0]
        } else {
            vec![0, 1]
        };
    }

    // Full cosine similarity matrix, diagonal included (self-similarity = 1.0, always
    // the maximum possible value -- see build_knn_connectivity's doc comment on why that
    // matters). No negative-clipping here, unlike the first attempt: the reference never
    // clips, since the matrix is only ever used for *ranking* neighbors (build_knn_
    // connectivity), and the eventual edge weight is binary regardless of the exact
    // similarity value once a neighbor is selected.
    let mut similarity = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        similarity[i][i] = 1.0;
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]) as f64;
            similarity[i][j] = sim;
            similarity[j][i] = sim;
        }
    }

    let p = p_override
        .unwrap_or_else(|| nme_select_p(&similarity, max_speakers))
        .max(1);

    let graph = build_knn_connectivity(&similarity, p);
    let laplacian = laplacian_from_connectivity(&graph);
    let (sorted_values, eigenvectors, order) = sorted_eigen(laplacian);
    let gaps = eigengaps(&sorted_values);
    let k = k_from_gaps(&gaps, max_speakers).max(1).min(max_speakers);

    if k == 1 {
        return vec![0; n];
    }

    // No row-normalization here (unlike the first attempt's Ng-Jordan-Weiss step) --
    // the reference hardcodes `norm_laplacian=False` and uses the raw eigenvector
    // entries directly; removed for fidelity, see the module-level comment above.
    let mut rows: Vec<Vec<f64>> = vec![vec![0.0; k]; n];
    for row in 0..n {
        for (col, &eig_idx) in order.iter().take(k).enumerate() {
            rows[row][col] = eigenvectors[(row, eig_idx)];
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
    // Sub-components of the NME-SC spectral clustering pipeline (build_knn_connectivity,
    // is_fully_connected, eigengaps, k_from_gaps) -- exactly hand-verifiable in
    // isolation, unlike the full pipeline (see the note on the end-to-end tests further
    // below on why exact eigenvalues stopped being hand-derivable once this file switched
    // from its own first attempt to a faithful translation of the real NME-SC reference).
    // ------------------------------------------------------------------------------

    #[test]
    fn eigengaps_computes_consecutive_differences() {
        assert_eq!(eigengaps(&[0.0, 0.0, 2.0, 2.5]), vec![0.0, 2.0, 0.5]);
        assert_eq!(eigengaps(&[1.0]), Vec::<f64>::new());
        assert_eq!(eigengaps(&[]), Vec::<f64>::new());
    }

    #[test]
    fn k_from_gaps_picks_the_biggest_gap_plus_one() {
        // Biggest gap (3.0) sits at index 2 -> k = 2 + 1 = 3.
        assert_eq!(k_from_gaps(&[0.1, 0.05, 3.0, 0.2], 4), 3);
    }

    #[test]
    fn k_from_gaps_ties_prefer_the_earliest_index() {
        assert_eq!(k_from_gaps(&[1.0, 1.0, 0.2], 3), 1);
    }

    #[test]
    fn k_from_gaps_capping_does_not_simply_clip_to_the_cap() {
        // The real signal (3.0 at index 2) sits *beyond* max_speakers=2's search window
        // (only indices 0..2 are examined) -- capping picks the best *within that
        // window* (both 0.1 and 0.05 are candidates, 0.1 at index 0 wins), landing on
        // k=1, not k=2. Same non-obvious "doesn't clip to the cap" behavior the previous
        // version of this file's tests demonstrated end-to-end -- reproduced here as an
        // exact, trivial arithmetic check instead, since it no longer requires running
        // the full eigendecomposition pipeline to exercise.
        assert_eq!(k_from_gaps(&[0.1, 0.05, 3.0, 0.2], 2), 1);
    }

    #[test]
    fn k_from_gaps_empty_returns_one() {
        assert_eq!(k_from_gaps(&[], 5), 1);
    }

    #[test]
    fn is_fully_connected_detects_a_single_chain() {
        // 0-1-2-3 chain: one connected component.
        let graph = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        assert!(is_fully_connected(&graph));
    }

    #[test]
    fn is_fully_connected_detects_two_disjoint_pairs() {
        // Edges (0-1) and (2-3) only -- two components, not one.
        let graph = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        assert!(!is_fully_connected(&graph));
    }

    #[test]
    fn build_knn_connectivity_keeps_top_p_and_averages_symmetrization() {
        // 3 points, similarity[i] strictly ranked (no ties) so top-p selection is
        // unambiguous: node 0's neighbors by similarity are [2 (0.9), 1 (0.5)] (self
        // excluded from this description, though it participates in the real ranking --
        // see the function's doc comment); node 1's are [0 (0.5), 2 (0.1)]; node 2's are
        // [0 (0.9), 1 (0.1)]. With p=2 (self + 1 real neighbor each): node0 keeps 2 (its
        // top real pick), node1 keeps 0, node2 keeps 0. Edge (0,2) is picked by *both*
        // sides -> symmetrized weight 1.0. Edge (0,1) is picked by node1 only (node0
        // preferred node2 over node1) -> symmetrized weight 0.5. Edge (1,2) picked by
        // neither -> 0.0.
        let similarity = vec![
            vec![1.0, 0.5, 0.9],
            vec![0.5, 1.0, 0.1],
            vec![0.9, 0.1, 1.0],
        ];
        let graph = build_knn_connectivity(&similarity, 2);
        assert_eq!(graph[0][2], 1.0);
        assert_eq!(graph[2][0], 1.0);
        assert_eq!(graph[0][1], 0.5);
        assert_eq!(graph[1][0], 0.5);
        assert_eq!(graph[1][2], 0.0);
        assert_eq!(graph[2][1], 0.0);
    }

    #[test]
    fn build_knn_connectivity_p_one_is_always_empty_after_self_is_excluded() {
        // Documents the real quirk relied on elsewhere (nme_select_p's small-n floor,
        // and the connectivity fallback): with p=1, every node's only selected neighbor
        // is itself (self-similarity is always the maximum possible value), so the
        // *connectivity* graph -- which laplacian_from_connectivity later reads with the
        // diagonal excluded -- carries no real edges at all.
        let similarity = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.1],
            vec![0.2, 0.1, 1.0],
        ];
        let graph = build_knn_connectivity(&similarity, 1);
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert_eq!(graph[i][j], 0.0, "no real edge expected at p=1");
                }
            }
        }
    }

    // ------------------------------------------------------------------------------
    // cluster_embeddings_spectral -- end-to-end tests.
    //
    // Honest note on verification level (see the module-level comment above
    // cluster_embeddings_spectral_with_p for the full explanation): this file's first
    // attempt at spectral clustering used exactly-orthogonal one-hot test vectors and
    // hand-derived the exact expected eigenvalues, which was possible because that
    // version's math (weighted graph, symmetric normalized Laplacian) made the
    // block-diagonal structure clean and tie-free. This version's real reference
    // algorithm (binary graph, self always occupying one neighbor slot, a connectivity
    // fallback that actively resists exactly-zero cross-cluster similarity) makes exact
    // ties and exactly-zero cross-similarity into *adversarial* inputs, not safe test
    // vectors, for reasons traced through in detail while designing this rewrite. The
    // tests below instead use clearly-separated but non-adversarial vectors (distinct,
    // non-tied similarities; small but nonzero cross-group similarity, mimicking real
    // embeddings, which never have exactly-zero cross-speaker similarity either) and
    // check only the qualitative, structurally-guaranteed outcome (same true group ->
    // same label, different true group -> different label) rather than exact
    // eigenvalues -- resting on spectral clustering's well-established general
    // guarantee on clearly block-structured data, not on a hand re-derivation of this
    // specific case. This is a real, deliberate reduction in verification rigor compared
    // to the sub-component tests above, made necessary by the algorithm change; the
    // actual calibration run against 5 real recordings (see docs/sviluppi/diarization/
    // Architettura pipeline.md) is what ultimately validates this, not these tests alone.
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
        // -- n=2 is a special case (see cluster_embeddings_spectral_with_p's doc
        // comment): the eigengap is mathematically uninformative at n=2, so this falls
        // back to a direct threshold comparison, unaffected by the NME rewrite.
        let e = vec![1.0, 0.0, 0.0];
        assert_eq!(cluster_embeddings_spectral(&[e.clone(), e], 5), vec![0, 0]);
    }

    #[test]
    fn spectral_two_dissimilar_embeddings_split() {
        // Orthogonal vectors: cosine similarity 0.0 < DEFAULT_CLUSTERING_THRESHOLD.
        let labels = cluster_embeddings_spectral(&[vec![1.0, 0.0], vec![0.0, 1.0]], 5);
        assert_ne!(labels[0], labels[1]);
    }

    /// Tiny deterministic PRNG (fixed seed, no external `rand` dependency) used only to
    /// build isotropic-looking noise for `spectral_two_clearly_separated_groups`. Returns
    /// values roughly in [-0.5, 0.5).
    fn lcg_next(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((*state >> 16) & 0x7fff) as f32 / 32768.0 - 0.5
    }

    #[test]
    fn spectral_two_clearly_separated_groups() {
        // 40 points, two groups of 20 (dim = 2 dominant + 12 noise dims). An earlier
        // version of this test perturbed each member along a *single shared* axis by a
        // small per-index amount (e.g. member i's noise = 0.01*i on one dimension) to
        // avoid exact ties. That is a real bug in the test, discovered empirically: a
        // single shared perturbation axis puts every member of the group on a 1-D
        // manifold, so the induced k-NN graph is a *path*, not a blob -- and a path
        // graph's Laplacian spectrum has its own internal harmonics (consecutive gaps
        // comparable in size to the gap that separates the two true groups). The
        // eigengap heuristic legitimately read those harmonics as extra sub-clusters and
        // reported 4 groups instead of 2 -- correct behavior of the algorithm on data
        // that was not actually blob-shaped, not a bug in `cluster_embeddings_spectral`.
        //
        // Real speaker embeddings vary across many largely-independent dimensions around
        // a per-speaker centroid, which is what this construction approximates: each
        // member gets independent pseudo-random noise (fixed-seed LCG) spread across 12
        // dimensions, so no single axis dominates the within-group similarity structure.
        // Group size is also increased from 8 to 20 so the automatic k-NN pruning
        // (capped at `max_n = floor(0.25*n)`, `nme_select_p`) has enough same-group
        // candidates to build a properly dense (non-chain, non-hub) subgraph.
        let mut seed = 42u32;
        let n_per_group = 20;
        let noise_dims = 12;
        let mut embeddings = Vec::with_capacity(n_per_group * 2);
        for _ in 0..n_per_group {
            let mut v = vec![1.0f32, 0.05];
            for _ in 0..noise_dims {
                v.push(0.05 * lcg_next(&mut seed));
            }
            embeddings.push(v);
        }
        for _ in 0..n_per_group {
            let mut v = vec![0.05f32, 1.0];
            for _ in 0..noise_dims {
                v.push(0.05 * lcg_next(&mut seed));
            }
            embeddings.push(v);
        }
        let labels = cluster_embeddings_spectral(&embeddings, 10);
        assert_eq!(distinct_speaker_count(&labels), 2, "expected exactly 2 groups");
        for i in 1..n_per_group {
            assert_eq!(labels[0], labels[i], "group A member {i} split off from the rest of group A");
        }
        for i in (n_per_group + 1)..(2 * n_per_group) {
            assert_eq!(labels[n_per_group], labels[i], "group B member {i} split off from the rest of group B");
        }
        assert_ne!(labels[0], labels[n_per_group], "group A and group B were merged into one cluster");
    }

    #[test]
    fn spectral_with_p_none_matches_the_default_search() {
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
    fn spectral_with_p_override_does_not_panic() {
        // Extreme p_override values (0, and far beyond n) must not panic -- exact
        // resulting labels aren't asserted here, only that the function stays
        // well-behaved regardless of what p is forced to.
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
