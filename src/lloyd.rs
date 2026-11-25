//! Standard Lloyd's k-means baseline for apples-to-apples comparison with e-means.

use std::time::Instant;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::distance::{AssignBackend, AssignOut, Metric, assign_rows};
use crate::emeans::{EmeansError, EpochStat};
use crate::spherical::normalize_into;

/// Configuration for [`LloydKmeans`].
#[derive(Debug, Clone)]
pub struct LloydConfig {
    pub d: usize,
    pub k: usize,
    pub niter: usize,
    pub metric: Metric,
    pub backend: AssignBackend,
    pub seed: u64,
    pub chunk_size: usize,
}

impl LloydConfig {
    #[must_use]
    pub fn new(d: usize, k: usize) -> Self {
        Self {
            d,
            k,
            niter: 25,
            metric: Metric::Euclidean,
            backend: AssignBackend::Accelerate,
            seed: 1234,
            chunk_size: 65_536,
        }
    }
}

/// Output of [`LloydKmeans::train`].
#[derive(Debug, Clone)]
pub struct LloydResult {
    pub centroids: Vec<f32>,
    pub assignments: Vec<usize>,
    pub iteration_stats: Vec<EpochStat>,
}

/// Standard Lloyd's k-means using the same assignment kernels as [`crate::Emeans`].
pub struct LloydKmeans {
    cfg: LloydConfig,
}

impl LloydKmeans {
    #[must_use]
    pub fn new(cfg: LloydConfig) -> Self {
        Self { cfg }
    }

    pub fn train(
        &self,
        x: &[f32],
        init_centroids: Option<&[f32]>,
    ) -> Result<LloydResult, EmeansError> {
        let d = self.cfg.d;
        let k = self.cfg.k;
        if d == 0 {
            return Err(EmeansError::InvalidDimension);
        }
        if k == 0 {
            return Err(EmeansError::InvalidClusterCount);
        }
        if x.is_empty() || !x.len().is_multiple_of(d) {
            return Err(EmeansError::InvalidDataLength { len: x.len(), d });
        }
        let n = x.len() / d;

        let mut centroids: Vec<f32> = match init_centroids {
            Some(c) => {
                if c.len() != k * d {
                    return Err(EmeansError::InvalidInitCentroids {
                        expected: k * d,
                        got: c.len(),
                    });
                }
                c.to_vec()
            }
            None => {
                let mut rng = StdRng::seed_from_u64(self.cfg.seed);
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(&mut rng);
                let mut c = Vec::with_capacity(k * d);
                for &idx in perm.iter().take(k) {
                    c.extend_from_slice(&x[idx * d..(idx + 1) * d]);
                }
                c
            }
        };

        if self.cfg.metric == Metric::Spherical {
            let mut norm_buf = vec![0.0f32; k * d];
            normalize_into(&centroids, &mut norm_buf, d);
            centroids = norm_buf;
        }

        let start = Instant::now();
        let mut distances = vec![0.0f32; n];
        let mut assignments = vec![0usize; n];
        let mut iteration_stats = Vec::with_capacity(self.cfg.niter);

        for _iter in 0..self.cfg.niter {
            for (chunk_idx, x_chunk) in x.chunks(self.cfg.chunk_size * d).enumerate() {
                let m = x_chunk.len() / d;
                let off = chunk_idx * self.cfg.chunk_size;
                let mut out = AssignOut {
                    d1: &mut distances[off..off + m],
                    l1: &mut assignments[off..off + m],
                    d2: None,
                    l2: None,
                };
                assign_rows(
                    x_chunk,
                    &centroids,
                    d,
                    k,
                    self.cfg.metric,
                    self.cfg.backend,
                    &mut out,
                );
            }

            let mut obj = 0.0f64;
            if self.cfg.metric == Metric::Spherical {
                obj += 2.0 * (n as f64);
            }
            for &dist in &distances {
                obj += dist as f64;
            }

            let mut sums = vec![0.0f32; k * d];
            let mut counts = vec![0usize; k];
            for (i, &lbl) in assignments.iter().enumerate() {
                counts[lbl] += 1;
                let row = &x[i * d..(i + 1) * d];
                let s_row = &mut sums[lbl * d..(lbl + 1) * d];
                for (s, &v) in s_row.iter_mut().zip(row.iter()) {
                    *s += v;
                }
            }

            let count_sq: f64 = counts.iter().map(|&c| (c as f64) * (c as f64)).sum();
            let imbalance = (k as f64) * count_sq / ((n as f64) * (n as f64)).max(1.0);

            for j in 0..k {
                if counts[j] > 0 {
                    let inv = 1.0 / (counts[j] as f32);
                    for i in 0..d {
                        centroids[j * d + i] = sums[j * d + i] * inv;
                    }
                }
            }

            // Revive empty clusters by splitting the largest cluster (Faiss style).
            let mut n_splits = 0usize;
            for j in 0..k {
                if counts[j] == 0 {
                    let mut best_donor = 0;
                    let mut best_cnt = 0;
                    for (idx, &c) in counts.iter().enumerate() {
                        if c > best_cnt {
                            best_cnt = c;
                            best_donor = idx;
                        }
                    }
                    if best_cnt > 1 {
                        let half = best_cnt / 2;
                        counts[best_donor] -= half;
                        counts[j] = half;
                        for i in 0..d {
                            let val = centroids[best_donor * d + i];
                            centroids[j * d + i] = val;
                        }
                        let eps = 1e-4f32;
                        centroids[best_donor * d] += eps;
                        centroids[j * d] -= eps;
                        n_splits += 1;
                    }
                }
            }

            if self.cfg.metric == Metric::Spherical {
                let mut norm_buf = vec![0.0f32; k * d];
                normalize_into(&centroids, &mut norm_buf, d);
                centroids = norm_buf;
            }

            iteration_stats.push(EpochStat {
                obj,
                time_secs: start.elapsed().as_secs_f64(),
                imbalance_factor: imbalance,
                n_splits,
            });
        }

        Ok(LloydResult {
            centroids,
            assignments,
            iteration_stats,
        })
    }
}
