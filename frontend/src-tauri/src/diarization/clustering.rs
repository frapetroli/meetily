//! Final speaker clustering -- see `docs/adr/0017-...md` for why this isn't a wrapper
//! around sherpa-onnx's `FastClustering` (not exposed standalone in any public API).
//!
//! Production uses spectral clustering (NME-SC, automatic speaker-count estimation via
//! the eigengap heuristic -- see the dedicated section below), adopted in
//! `docs/adr/0024-...md` after real-recording testing showed it far closer to ground
//! truth than the fixed-threshold agglomerative approach this file used previously
//! (`docs/adr/0017-...md`/`docs/adr/0022-...md`, removed once the spectral method
//! replaced it in production -- see git history on this file if that old approach is ever
//! needed again for reference).

use nalgebra::{DMatrix, SymmetricEigen};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Direct cosine-similarity threshold used only by `cluster_embeddings_spectral_with_p`'s
/// `n == 2` special case (see its doc comment below) -- with only 2 embeddings there's no
/// eigengap to compute, so this is a plain "same speaker or not" comparison. Historically
/// this was also the main agglomerative clustering's threshold (`docs/adr/0017-...md`),
/// before that approach was removed once spectral clustering replaced it in production
/// (`docs/adr/0024-...md`) -- kept at the same calibrated value since it was independently
/// re-validated for that N=2 case, not because the old approach still exists.
pub const DEFAULT_CLUSTERING_THRESHOLD: f32 = 0.6;

/// Default cap passed to `cluster_embeddings_spectral`/`finalize_with_spectral` when the
/// user hasn't configured `transcript_settings.diarization_max_speakers` -- see
/// `docs/adr/0024-...md`. Generous on purpose (the eigengap search can collapse its
/// estimate well below this cap, but never above it -- see `k_from_gaps`'s doc comment),
/// not a tight guess at a typical meeting's real speaker count.
pub const DEFAULT_MAX_SPEAKERS: usize = 20;

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

/// Number of distinct clusters/labels in a clustering result.
pub fn distinct_speaker_count(labels: &[usize]) -> usize {
    labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
}

/// Remaps arbitrary label ids (e.g. left non-contiguous after some labels are folded
/// away) to a contiguous 0-indexed range, preserving first-seen order. Cosmetic only:
/// never changes which inputs share a label.
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
// Spectral clustering (production method, ADR-0024)
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
// calibrazione reale..." for why the fixed-threshold approach this file used previously
// (removed, ADR-0024), even recalibrated (ADR-0022), still left real recordings wildly
// over-segmented.
// `DiarizationEngine::finalize()` (production, all three call sites: session.rs/
// import.rs/retranscription.rs) calls this via `finalize_with_spectral()` -- see
// ADR-0024. `examples/diarization_calibration.rs` also calls it directly (with a `p`
// override) to keep experimenting with `p` on real recordings.
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
// K-means (see `kmeans` below) now matches the reference's `n_init=10` -- 10 restarts
// with k-means++ initialization (the reference's own default), picking the run with the
// lowest inertia, same selection rule scikit-learn uses. The one remaining deliberate
// deviation: the reference's restarts are unseeded (genuinely random), ours use fixed
// seeds 0..9, so the result stays exactly reproducible run-to-run -- there is no
// compiler available in the sandbox this was written in to catch a randomness-related
// bug before it reaches a real machine. The original single deterministic "farthest-
// first" run from this file's first version of `kmeans` is kept as an extra 11th
// candidate alongside the 10 seeded restarts (not a reference behavior -- our own
// addition), so this change can only match or improve on the previous behavior, never
// regress it.
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
                // Reference: `np.linspace(...).astype(int)` truncates toward zero, it does
                // not round to the nearest integer -- `as usize` on a positive f64 already
                // truncates the same way, so no explicit `.round()`/`.floor()` call is needed.
                let t = i as f64 / (candidate_count - 1) as f64;
                (1.0 + t * (max_n as f64 - 1.0)) as usize
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

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Deterministic "farthest-first" seeding: kept from this file's first version of
/// `kmeans`, now used as one extra candidate alongside the seeded k-means++ restarts
/// below (see `kmeans`'s doc comment). Greedily picks, after an arbitrary first center,
/// whichever remaining point is farthest from every center chosen so far -- a good fit
/// for row-normalized spectral embeddings, which tend to already be well-separated per
/// true cluster.
fn farthest_first_init(points: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
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
    centers
}

/// k-means++ seeding (the reference's/scikit-learn's default `KMeans` initialization):
/// first center uniformly random, each subsequent center picked with probability
/// proportional to its squared distance from the nearest already-chosen center --
/// spreads the initial centers out, rather than risking several landing in the same
/// true cluster the way a purely uniform random pick could.
fn kmeanspp_init(points: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut centers: Vec<Vec<f64>> = vec![points[rng.gen_range(0..n)].clone()];
    let mut min_dist: Vec<f64> = points.iter().map(|p| sq_dist(p, &centers[0])).collect();
    while centers.len() < k {
        let total: f64 = min_dist.iter().sum();
        let next = if total <= 0.0 {
            // All remaining points exactly coincide with an existing center (degenerate
            // input, e.g. duplicate embeddings) -- weighted sampling can't break the tie,
            // fall back to picking uniformly at random.
            rng.gen_range(0..n)
        } else {
            WeightedIndex::new(&min_dist)
                .expect("total > 0 checked above, so at least one weight is positive")
                .sample(rng)
        };
        centers.push(points[next].clone());
        let last = centers.len() - 1;
        for (i, p) in points.iter().enumerate() {
            min_dist[i] = min_dist[i].min(sq_dist(p, &centers[last]));
        }
    }
    centers
}

/// Lloyd's algorithm from a given set of initial centers. Returns the final assignment
/// plus its inertia (sum of squared point-to-assigned-center distances) -- the same
/// quantity scikit-learn's `KMeans(n_init=...)` uses to pick the best of several runs.
fn lloyd_iterate(points: &[Vec<f64>], k: usize, mut centers: Vec<Vec<f64>>) -> (Vec<usize>, f64) {
    let n = points.len();
    let dim = points[0].len();

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

    let inertia: f64 = points
        .iter()
        .enumerate()
        .map(|(i, p)| sq_dist(p, &centers[assignment[i]]))
        .sum();
    (assignment, inertia)
}

/// Reference: scikit-learn `KMeans(n_init=10)` -- 10 restarts, keep the one with the
/// lowest inertia. See the module-level comment above for the one deliberate deviation
/// (fixed seeds instead of the reference's genuine randomness) and the extra 11th
/// candidate kept from this file's first version.
const KMEANS_N_INIT: usize = 10;

fn kmeans(points: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = points.len();
    if k <= 1 || n <= 1 {
        return vec![0; n];
    }

    let (mut best_assignment, mut best_inertia) =
        lloyd_iterate(points, k, farthest_first_init(points, k));

    for seed in 0..KMEANS_N_INIT as u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (assignment, inertia) = lloyd_iterate(points, k, kmeanspp_init(points, k, &mut rng));
        if inertia < best_inertia {
            best_inertia = inertia;
            best_assignment = assignment;
        }
    }

    best_assignment
}

/// Spectral clustering with automatic speaker-count estimation. See the module-level
/// comment above this section for the algorithm family and rationale. `max_speakers`
/// upper-bounds the estimated count (clamped to >= 1) -- pass a generous value (e.g. 20),
/// not a tight guess: if the true cluster count is at or beyond the cap, the eigengap
/// search can collapse the estimate well below the cap rather than simply clipping to it
/// (see the doc comment on the K-estimation loop below, and the
/// `k_from_gaps_capping_does_not_simply_clip_to_the_cap` test).
///
/// Returns one 0-indexed cluster label per input embedding, feeding straight into the
/// same downstream pipeline (`normalize_labels`, `SpeakerSegment` construction).
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
