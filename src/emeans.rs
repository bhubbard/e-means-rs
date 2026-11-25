//! Core E-means clustering implementation (FP32 CPU, Algorithm 1).

use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use thiserror::Error;

use crate::distance::{AssignBackend, AssignOut, Metric, assign_rows};
use crate::recipe::{
    BatchLaw, CHECK_UNUSED_EVERY, INIT_USAGE, REMOVAL_THRESHOLD, Recipe, RevivalMode, epoch_seed,
    resolve_recipe, scheduled_alpha,
};
use crate::spherical::normalize_into;

/// Errors returned by E-means configuration and execution.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum EmeansError {
    #[error("dimension d must be > 0")]
    InvalidDimension,
    #[error("cluster count k must be > 0")]
    InvalidClusterCount,
    #[error("nepochs must be finite and > 0, got {0}")]
    InvalidEpochs(f64),
    #[error("data length ({len}) must be a non-zero multiple of d ({d})")]
    InvalidDataLength { len: usize, d: usize },
    #[error("expected {expected} initial centroid floats (k*d), got {got}")]
    InvalidInitCentroids { expected: usize, got: usize },
    #[error("batch_size must be > 0")]
    InvalidBatchSize,
    #[error("unused_threshold must be >= 0, got {0}")]
    InvalidThreshold(f64),
    #[error("invalid state: {0}")]
    InvalidState(&'static str),
}

/// Whether `train` captures final-epoch cluster assignments for every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureMode {
    #[default]
    Capture,
    None,
}

/// Closing-pass handover strategy for rows held by revived centroids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HandoverMode {
    #[default]
    SecondNearest,
    None,
}

/// Configuration for [`Emeans`].
#[derive(Debug, Clone, PartialEq)]
pub struct EmeansConfig {
    pub d: usize,
    pub k: usize,
    pub nepochs: f64,
    pub batch_size: Option<usize>,
    pub alpha0: Option<f64>,
    pub batch_law: BatchLaw,
    pub revival: RevivalMode,
    pub metric: Metric,
    pub backend: AssignBackend,
    pub seed: u64,
    pub max_points_per_centroid: Option<usize>,
    pub assignment_chunk_size: usize,
    pub capture: CaptureMode,
    pub handover: HandoverMode,
}

impl EmeansConfig {
    /// Create a default configuration for `d`-dimensional vectors and `k` clusters.
    #[must_use]
    pub fn new(d: usize, k: usize) -> Self {
        Self {
            d,
            k,
            nepochs: 5.0,
            batch_size: None,
            alpha0: None,
            batch_law: BatchLaw::Auto,
            revival: RevivalMode::CostBased,
            metric: Metric::Euclidean,
            backend: AssignBackend::Accelerate,
            seed: 1234,
            max_points_per_centroid: None,
            assignment_chunk_size: 65_536,
            capture: CaptureMode::Capture,
            handover: HandoverMode::SecondNearest,
        }
    }

    /// Return a builder for [`EmeansConfig`].
    #[must_use]
    pub fn builder(d: usize, k: usize) -> EmeansBuilder {
        EmeansBuilder {
            cfg: Self::new(d, k),
        }
    }

    pub fn validate(&self) -> Result<(), EmeansError> {
        if self.d == 0 {
            return Err(EmeansError::InvalidDimension);
        }
        if self.k == 0 {
            return Err(EmeansError::InvalidClusterCount);
        }
        if !self.nepochs.is_finite() || self.nepochs <= 0.0 {
            return Err(EmeansError::InvalidEpochs(self.nepochs));
        }
        if let Some(b) = self.batch_size
            && b == 0
        {
            return Err(EmeansError::InvalidBatchSize);
        }
        if let RevivalMode::MassBased(t) = self.revival
            && (!t.is_finite() || t < 0.0)
        {
            return Err(EmeansError::InvalidThreshold(t));
        }
        Ok(())
    }
}

/// Fluent builder for [`EmeansConfig`].
#[derive(Debug, Clone)]
pub struct EmeansBuilder {
    cfg: EmeansConfig,
}

impl EmeansBuilder {
    #[must_use]
    pub fn nepochs(mut self, nepochs: f64) -> Self {
        self.cfg.nepochs = nepochs;
        self
    }

    #[must_use]
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.cfg.batch_size = Some(batch_size);
        self
    }

    #[must_use]
    pub fn alpha0(mut self, alpha0: f64) -> Self {
        self.cfg.alpha0 = Some(alpha0);
        self
    }

    #[must_use]
    pub fn batch_law(mut self, batch_law: BatchLaw) -> Self {
        self.cfg.batch_law = batch_law;
        self
    }

    #[must_use]
    pub fn revival(mut self, revival: RevivalMode) -> Self {
        self.cfg.revival = revival;
        self
    }

    #[must_use]
    pub fn metric(mut self, metric: Metric) -> Self {
        self.cfg.metric = metric;
        self
    }

    #[must_use]
    pub fn backend(mut self, backend: AssignBackend) -> Self {
        self.cfg.backend = backend;
        self
    }

    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.cfg.seed = seed;
        self
    }

    #[must_use]
    pub fn max_points_per_centroid(mut self, max_ppc: usize) -> Self {
        self.cfg.max_points_per_centroid = Some(max_ppc);
        self
    }

    #[must_use]
    pub fn assignment_chunk_size(mut self, chunk_size: usize) -> Self {
        self.cfg.assignment_chunk_size = chunk_size.max(1);
        self
    }

    #[must_use]
    pub fn capture(mut self, capture: CaptureMode) -> Self {
        self.cfg.capture = capture;
        self
    }

    #[must_use]
    pub fn handover(mut self, handover: HandoverMode) -> Self {
        self.cfg.handover = handover;
        self
    }

    pub fn build(self) -> Result<Emeans, EmeansError> {
        Emeans::new(self.cfg)
    }

    pub fn build_config(self) -> Result<EmeansConfig, EmeansError> {
        self.cfg.validate()?;
        Ok(self.cfg)
    }
}

/// Per-epoch telemetry recorded during training.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochStat {
    pub obj: f64,
    pub time_secs: f64,
    pub imbalance_factor: f64,
    pub n_splits: usize,
}

/// Output of [`Emeans::train`].
#[derive(Debug, Clone)]
pub struct TrainResult {
    pub centroids: Vec<f32>,
    pub assignments: Option<Vec<isize>>,
    pub obj: Vec<f64>,
    pub iteration_stats: Vec<EpochStat>,
    pub recipe: Recipe,
    pub n_splits: usize,
}

/// Output of [`Emeans::assign`].
#[derive(Debug, Clone)]
pub struct AssignResult {
    /// Squared L2 distances for Euclidean, or inner-product similarities for Spherical.
    pub distances: Vec<f32>,
    pub labels: Vec<usize>,
}

/// Batch statistics produced by [`Emeans::accumulate`] and consumed by [`Emeans::update`].
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub sums: Vec<f32>,
    pub counts: Vec<f32>,
    pub objective: f64,
    pub n: usize,
    pub(crate) removal: Option<Vec<f32>>,
}

struct SplitEvent {
    target: Vec<usize>,
    source: Vec<usize>,
    born: usize,
}

struct HandoverState {
    done: usize,
    events: Vec<SplitEvent>,
    moved_at: Vec<isize>,
}

/// Stateful E-means clustering engine.
pub struct Emeans {
    cfg: EmeansConfig,
    recipe: Option<Recipe>,
    centroids: Vec<f32>,
    unit_centroids: Vec<f32>,
    unit_dirty: bool,
    usage: Vec<f32>,
    removal_stats: Option<Vec<f32>>,
    axes: Vec<f32>,
    axis_scale: f64,
    pass_counts: Vec<f64>,
    pass_sums: Option<Vec<f32>>,
    objective: f64,
    step: usize,
    epoch: usize,
    epoch_rows_done: usize,
    splits_before_epoch: usize,
    n_splits: usize,
    iteration_stats: Vec<EpochStat>,
    start_time: Option<Instant>,
    split_rng: StdRng,
    handover: Option<HandoverState>,
    finalized: bool,
}

impl Emeans {
    /// Create a new [`Emeans`] instance from a validated configuration.
    pub fn new(cfg: EmeansConfig) -> Result<Self, EmeansError> {
        cfg.validate()?;
        let split_rng = StdRng::seed_from_u64(cfg.seed);
        Ok(Self {
            cfg,
            recipe: None,
            centroids: Vec::new(),
            unit_centroids: Vec::new(),
            unit_dirty: true,
            usage: Vec::new(),
            removal_stats: None,
            axes: Vec::new(),
            axis_scale: 1.0,
            pass_counts: Vec::new(),
            pass_sums: None,
            objective: 0.0,
            step: 0,
            epoch: 0,
            epoch_rows_done: 0,
            splits_before_epoch: 0,
            n_splits: 0,
            iteration_stats: Vec::new(),
            start_time: None,
            split_rng,
            handover: None,
            finalized: false,
        })
    }

    /// Access the resolved training recipe, if [`begin`](Self::begin) or [`train`](Self::train) has been called.
    #[must_use]
    pub fn recipe(&self) -> Option<&Recipe> {
        self.recipe.as_ref()
    }

    /// Borrow the effective centroids (`k * d`), unit-normalized if `Metric::Spherical`.
    pub fn current_centroids(&mut self) -> &[f32] {
        self.ensure_readout();
        if self.cfg.metric == Metric::Spherical {
            &self.unit_centroids
        } else {
            &self.centroids
        }
    }

    fn ensure_readout(&mut self) {
        if self.cfg.metric == Metric::Spherical && self.unit_dirty {
            if self.unit_centroids.len() != self.centroids.len() {
                self.unit_centroids.resize(self.centroids.len(), 0.0);
            }
            normalize_into(&self.centroids, &mut self.unit_centroids, self.cfg.d);
            self.unit_dirty = false;
        }
    }

    fn keeps_removal_stats(&self) -> bool {
        match self.cfg.revival {
            RevivalMode::CostBased => true,
            RevivalMode::MassBased(t) => t > 0.0,
            RevivalMode::Disabled => false,
        }
    }

    fn uses_cost_revival(&self) -> bool {
        self.cfg.revival == RevivalMode::CostBased && self.cfg.k > 1
    }

    /// Initialize model state for caller-driven streaming training over `global_n` rows.
    pub fn begin(&mut self, global_n: usize, init_centroids: &[f32]) -> Result<(), EmeansError> {
        if self.recipe.is_some() {
            return Err(EmeansError::InvalidState(
                "create a new Emeans instance for another run",
            ));
        }
        let expected = self.cfg.k * self.cfg.d;
        if init_centroids.len() != expected {
            return Err(EmeansError::InvalidInitCentroids {
                expected,
                got: init_centroids.len(),
            });
        }
        if global_n == 0 {
            return Err(EmeansError::InvalidDataLength {
                len: 0,
                d: self.cfg.d,
            });
        }

        self.centroids = init_centroids.to_vec();
        if self.cfg.metric == Metric::Spherical {
            let mut norm_buf = vec![0.0f32; expected];
            normalize_into(&self.centroids, &mut norm_buf, self.cfg.d);
            self.centroids = norm_buf;
            self.unit_centroids = self.centroids.clone();
            self.unit_dirty = false;
        }

        self.usage = vec![INIT_USAGE; self.cfg.k];
        self.split_rng = StdRng::seed_from_u64(self.cfg.seed);

        if self.keeps_removal_stats() {
            self.removal_stats = Some(vec![0.0f32; self.cfg.k * (self.cfg.d + 1)]);
            self.axis_scale = 1.0;
            self.axes = init_random_axes(self.cfg.k, self.cfg.d, &mut self.split_rng);
        }

        let recipe = resolve_recipe(
            global_n,
            self.cfg.k,
            self.cfg.d,
            self.cfg.nepochs,
            self.cfg.batch_law,
            self.cfg.batch_size,
            self.cfg.alpha0,
        );
        self.recipe = Some(recipe);
        self.step = 0;
        self.epoch = 0;
        self.n_splits = 0;
        self.iteration_stats.clear();
        self.start_epoch();
        self.start_time = Some(Instant::now());
        Ok(())
    }

    fn start_epoch(&mut self) {
        self.epoch_rows_done = 0;
        self.splits_before_epoch = self.n_splits;
        self.pass_counts = vec![0.0f64; self.cfg.k];
        self.objective = 0.0;
        let is_last = self
            .recipe
            .as_ref()
            .is_some_and(|r| self.epoch + 1 == r.epoch_rows.len());
        self.pass_sums = if is_last {
            Some(vec![0.0f32; self.cfg.k * self.cfg.d])
        } else {
            None
        };
    }

    /// Accumulate batch statistics for a contiguous slice of rows `batch` (`m * d`).
    pub fn accumulate(&mut self, batch: &[f32]) -> Result<BatchStats, EmeansError> {
        let d = self.cfg.d;
        if batch.is_empty() || !batch.len().is_multiple_of(d) {
            return Err(EmeansError::InvalidDataLength {
                len: batch.len(),
                d,
            });
        }
        let n = batch.len() / d;
        self.accumulate_indexed(batch, &(0..n).collect::<Vec<_>>(), None, None)
    }

    fn accumulate_indexed(
        &mut self,
        x: &[f32],
        indices: &[usize],
        mut labels_out: Option<&mut [usize]>,
        mut second_labels_out: Option<&mut [isize]>,
    ) -> Result<BatchStats, EmeansError> {
        let d = self.cfg.d;
        let k = self.cfg.k;
        let n = indices.len();
        let chunk_size = self.cfg.assignment_chunk_size;
        let need_removal = self.removal_stats.is_some();
        let need_second_dist = need_removal && self.uses_cost_revival();

        self.ensure_readout();
        let readout = if self.cfg.metric == Metric::Spherical {
            &self.unit_centroids
        } else {
            &self.centroids
        };

        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0.0f32; k];
        let mut batch_removal = if need_removal {
            Some(vec![0.0f32; k * (d + 1)])
        } else {
            None
        };
        let mut objective = 0.0f64;

        let mut gathered = Vec::new();
        let mut d1_buf = Vec::new();
        let mut l1_buf = Vec::new();
        let mut d2_buf = Vec::new();
        let mut l2_buf = Vec::new();

        for (chunk_idx, idx_chunk) in indices.chunks(chunk_size).enumerate() {
            let m = idx_chunk.len();
            gathered.clear();
            gathered.reserve(m * d);
            for &row_id in idx_chunk {
                gathered.extend_from_slice(&x[row_id * d..(row_id + 1) * d]);
            }

            d1_buf.resize(m, 0.0);
            l1_buf.resize(m, 0);
            if need_second_dist {
                d2_buf.resize(m, 0.0);
            }
            if second_labels_out.is_some() && need_second_dist {
                l2_buf.resize(m, 0);
            }

            {
                let mut out = AssignOut {
                    d1: &mut d1_buf[..m],
                    l1: &mut l1_buf[..m],
                    d2: if need_second_dist {
                        Some(&mut d2_buf[..m])
                    } else {
                        None
                    },
                    l2: if second_labels_out.is_some() && need_second_dist {
                        Some(&mut l2_buf[..m])
                    } else {
                        None
                    },
                };
                assign_rows(
                    &gathered,
                    readout,
                    d,
                    k,
                    self.cfg.metric,
                    self.cfg.backend,
                    &mut out,
                );
            }

            let offset = chunk_idx * chunk_size;
            if let Some(ref mut l_out) = labels_out {
                l_out[offset..offset + m].copy_from_slice(&l1_buf[..m]);
            }
            if let Some(ref mut s_out) = second_labels_out {
                if need_second_dist {
                    for (dst, &src) in s_out[offset..offset + m].iter_mut().zip(l2_buf[..m].iter())
                    {
                        *dst = src as isize;
                    }
                } else {
                    s_out[offset..offset + m].fill(-1);
                }
            }

            if self.cfg.metric == Metric::Spherical {
                objective += 2.0 * (m as f64);
            }
            for &dist in &d1_buf[..m] {
                objective += dist as f64;
            }

            // Accumulate cluster sums and counts.
            for r in 0..m {
                let lbl = l1_buf[r];
                counts[lbl] += 1.0;
                let row = &gathered[r * d..(r + 1) * d];
                let c_sum = &mut sums[lbl * d..(lbl + 1) * d];
                for (s, &v) in c_sum.iter_mut().zip(row.iter()) {
                    *s += v;
                }
            }

            // Accumulate Oja projection & removal margin (d2 - d1).
            if let Some(ref mut rem) = batch_removal {
                for r in 0..m {
                    let lbl = l1_buf[r];
                    let row = &gathered[r * d..(r + 1) * d];
                    let cen = &readout[lbl * d..(lbl + 1) * d];
                    let axis = &self.axes[lbl * d..(lbl + 1) * d];
                    let mut p = 0.0f32;
                    for j in 0..d {
                        p += (row[j] - cen[j]) * axis[j];
                    }
                    let rem_row = &mut rem[lbl * (d + 1)..(lbl + 1) * (d + 1)];
                    for j in 0..d {
                        rem_row[j] += p * (row[j] - cen[j]);
                    }
                    let margin = if need_second_dist {
                        let diff = d2_buf[r] - d1_buf[r];
                        if diff.is_finite() { diff } else { 0.0 }
                    } else {
                        0.0
                    };
                    rem_row[d] += margin;
                }
            }
        }

        Ok(BatchStats {
            sums,
            counts,
            objective,
            n,
            removal: batch_removal,
        })
    }

    /// Apply one batch's statistics via EMA and run periodic cluster revival.
    pub fn update(&mut self, stats: &BatchStats) -> Result<(), EmeansError> {
        let recipe = self
            .recipe
            .as_ref()
            .ok_or(EmeansError::InvalidState("call begin() before update()"))?;
        let alpha = scheduled_alpha(recipe.alpha0, self.step, recipe.total_steps);
        let decay = 1.0 - alpha;
        let replace_usage = (alpha - 1.0).abs() < f64::EPSILON;
        let d = self.cfg.d;
        let k = self.cfg.k;

        self.objective += stats.objective;

        if let (Some(rem_stats), Some(batch_rem)) = (&mut self.removal_stats, &stats.removal) {
            let alpha_f = alpha as f32;
            let decay_f = decay as f32;
            // Update removal cost R_j (column d).
            for j in 0..k {
                let idx = j * (d + 1) + d;
                rem_stats[idx] = decay_f * rem_stats[idx] + alpha_f * batch_rem[idx];
            }
            // Decay U via scalar factor `axis_scale`.
            let mut scale = self.axis_scale * decay;
            if scale <= 2.0f64.powi(-32) {
                let sf = scale as f32;
                for j in 0..k {
                    let row = &mut rem_stats[j * (d + 1)..j * (d + 1) + d];
                    for v in row.iter_mut() {
                        *v *= sf;
                    }
                }
                scale = 1.0;
            }
            self.axis_scale = scale;
            let inv_scale = (alpha / scale) as f32;
            for j in 0..k {
                if stats.counts[j] > 0.0 {
                    let u_row = &mut rem_stats[j * (d + 1)..j * (d + 1) + d];
                    let b_row = &batch_rem[j * (d + 1)..j * (d + 1) + d];
                    let mut norm_sq = 0.0f64;
                    for i in 0..d {
                        u_row[i] += inv_scale * b_row[i];
                        let val = u_row[i] as f64;
                        norm_sq += val * val;
                    }
                    let norm = norm_sq.sqrt();
                    if norm > 0.0 {
                        let inv_norm = 1.0 / norm;
                        let ax_row = &mut self.axes[j * d..(j + 1) * d];
                        for i in 0..d {
                            ax_row[i] = ((u_row[i] as f64) * inv_norm) as f32;
                        }
                    }
                }
            }
        }

        for j in 0..k {
            self.pass_counts[j] += stats.counts[j] as f64;
        }
        if let Some(ref mut p_sums) = self.pass_sums {
            for (dst, &src) in p_sums.iter_mut().zip(stats.sums.iter()) {
                *dst += src;
            }
        }

        let decay_f = decay as f32;
        let alpha_f = alpha as f32;
        if !replace_usage {
            for u in self.usage.iter_mut() {
                *u *= decay_f;
            }
        }

        for j in 0..k {
            let cnt = stats.counts[j];
            if cnt > 0.0 {
                let retained = if replace_usage { 0.0 } else { self.usage[j] };
                let mass = retained + alpha_f * cnt;
                let weight = alpha_f * cnt / mass;
                let c_row = &mut self.centroids[j * d..(j + 1) * d];
                let s_row = &stats.sums[j * d..(j + 1) * d];
                if (weight - 1.0).abs() < f32::EPSILON {
                    for i in 0..d {
                        c_row[i] = s_row[i] / cnt;
                    }
                } else {
                    for i in 0..d {
                        let mean = s_row[i] / cnt;
                        c_row[i] += weight * (mean - c_row[i]);
                    }
                }
                self.usage[j] = mass;
            }
        }

        self.unit_dirty = true;

        if (self.step + 1).is_multiple_of(CHECK_UNUSED_EVERY) {
            self.revive_unused();
        }

        self.step += 1;
        self.epoch_rows_done += stats.n;
        Ok(())
    }

    fn revive_unused(&mut self) {
        let d = self.cfg.d;
        let k = self.cfg.k;

        let (sources, targets): (Vec<usize>, Vec<usize>) = if self.uses_cost_revival() {
            let Some(ref rem) = self.removal_stats else {
                return;
            };
            let mut r_vals = Vec::with_capacity(k);
            let mut axis_norms = Vec::with_capacity(k);
            for j in 0..k {
                r_vals.push(rem[j * (d + 1) + d] as f64);
                let mut nsq = 0.0f64;
                for i in 0..d {
                    let v = rem[j * (d + 1) + i] as f64;
                    nsq += v * v;
                }
                axis_norms.push(self.axis_scale * nsq.sqrt());
            }
            let med = median_f64(&r_vals);
            let mut kept: Vec<usize> = (0..k)
                .filter(|&j| r_vals[j] <= 0.0 || r_vals[j] < REMOVAL_THRESHOLD * med)
                .collect();
            if kept.is_empty() {
                return;
            }
            kept.sort_by(|&a, &b| r_vals[a].total_cmp(&r_vals[b]));

            let mut is_kept = vec![false; k];
            for &j in &kept {
                is_kept[j] = true;
            }
            let mut donors: Vec<usize> = (0..k)
                .filter(|&j| !is_kept[j] && self.usage[j] > 0.0)
                .collect();
            if donors.is_empty() {
                return;
            }
            donors.sort_by(|&a, &b| axis_norms[b].total_cmp(&axis_norms[a]));

            let n = kept.len().min(donors.len());
            (donors[..n].to_vec(), kept[..n].to_vec())
        } else if let RevivalMode::MassBased(tau) = self.cfg.revival {
            if tau <= 0.0 {
                return;
            }
            let total_usage: f32 = self.usage.iter().sum();
            let threshold = total_usage * (tau as f32) / (k as f32);
            let unused: Vec<usize> = (0..k).filter(|&j| self.usage[j] < threshold).collect();
            let donors: Vec<usize> = (0..k)
                .filter(|&j| self.usage[j] >= threshold && self.usage[j] > 0.0)
                .collect();
            let nsplit = unused.len().min(donors.len());
            if nsplit == 0 {
                return;
            }
            let mut scored: Vec<(f64, usize)> = donors
                .iter()
                .map(|&j| {
                    let u: f64 = self.split_rng.random::<f64>().max(f64::MIN_POSITIVE);
                    let key = (self.usage[j] as f64).ln() - (-u.ln()).ln();
                    (key, j)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            let src: Vec<usize> = scored.into_iter().take(nsplit).map(|(_, j)| j).collect();
            let tgt: Vec<usize> = unused.into_iter().take(nsplit).collect();
            (src, tgt)
        } else {
            return;
        };

        if sources.is_empty() {
            return;
        }
        self.apply_splits(&sources, &targets);
    }

    fn apply_splits(&mut self, sources: &[usize], targets: &[usize]) {
        let d = self.cfg.d;

        for (&src, &tgt) in sources.iter().zip(targets.iter()) {
            let spread = if let Some(ref rem) = self.removal_stats {
                let mut nsq = 0.0f64;
                for i in 0..d {
                    let v = rem[src * (d + 1) + i] as f64;
                    nsq += v * v;
                }
                let norm = self.axis_scale * nsq.sqrt();
                let mass = self.usage[src] as f64;
                if norm > 0.0 && mass > 0.0 && norm.is_finite() && mass.is_finite() {
                    (2.0 * norm / (std::f64::consts::PI * mass)).sqrt()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let mut donor_c = vec![0.0f32; d];
            let mut child_c = vec![0.0f32; d];
            for i in 0..d {
                let c = self.centroids[src * d + i] as f64;
                let off = spread * (self.axes.get(src * d + i).copied().unwrap_or(0.0) as f64);
                donor_c[i] = (c - off) as f32;
                child_c[i] = (c + off) as f32;
            }
            if donor_c == child_c {
                let v = child_c[0];
                let next = v.next_up();
                child_c[0] = if next.is_finite() {
                    next
                } else {
                    v.next_down()
                };
            }

            let half_usage = self.usage[src] * 0.5;
            self.usage[src] = half_usage;
            self.usage[tgt] = half_usage;
            self.centroids[src * d..(src + 1) * d].copy_from_slice(&donor_c);
            self.centroids[tgt * d..(tgt + 1) * d].copy_from_slice(&child_c);

            if let Some(ref mut rem) = self.removal_stats {
                for i in 0..=d {
                    rem[src * (d + 1) + i] *= 0.5;
                    rem[tgt * (d + 1) + i] = rem[src * (d + 1) + i];
                }
                let src_ax = self.axes[src * d..(src + 1) * d].to_vec();
                self.axes[tgt * d..(tgt + 1) * d].copy_from_slice(&src_ax);
            }

            if self.pass_sums.is_some() && self.uses_cost_revival() && self.handover.is_none() {
                if let Some(ref mut p_sums) = self.pass_sums {
                    p_sums[src * d..(src + 1) * d].fill(0.0);
                    p_sums[tgt * d..(tgt + 1) * d].fill(0.0);
                }
                self.pass_counts[src] = 0.0;
                self.pass_counts[tgt] = 0.0;
            }
        }

        if let Some(ref mut h) = self.handover {
            for &tgt in targets {
                h.moved_at[tgt] = h.done as isize;
            }
            h.events.push(SplitEvent {
                target: targets.to_vec(),
                source: sources.to_vec(),
                born: h.done,
            });
        }

        self.n_splits += sources.len();
        self.unit_dirty = true;
    }

    /// Finish the current epoch and record its objective and imbalance factor.
    pub fn end_epoch(&mut self) -> Result<EpochStat, EmeansError> {
        let recipe = self
            .recipe
            .as_ref()
            .ok_or(EmeansError::InvalidState("call begin() before end_epoch()"))?;
        let rows = recipe.epoch_rows[self.epoch] as f64;
        let count_sq: f64 = self.pass_counts.iter().map(|&c| c * c).sum();
        let time_secs = self.start_time.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let stat = EpochStat {
            obj: self.objective,
            time_secs,
            imbalance_factor: (self.cfg.k as f64) * count_sq / (rows * rows).max(1.0),
            n_splits: self.n_splits - self.splits_before_epoch,
        };
        self.iteration_stats.push(stat.clone());
        self.epoch += 1;
        if self.epoch < recipe.epoch_rows.len() {
            self.start_epoch();
        }
        Ok(stat)
    }

    /// Compute final closing means from last-epoch assignments.
    pub fn finalize(&mut self) -> Result<&[f32], EmeansError> {
        if self.finalized {
            return Ok(&self.centroids);
        }
        let d = self.cfg.d;
        let k = self.cfg.k;
        let p_sums = self.pass_sums.as_ref().ok_or(EmeansError::InvalidState(
            "finalize called before last epoch",
        ))?;

        match self.cfg.metric {
            Metric::Euclidean => {
                for j in 0..k {
                    let cnt = self.pass_counts[j];
                    if cnt > 0.0 {
                        let inv = 1.0 / (cnt as f32);
                        for i in 0..d {
                            self.centroids[j * d + i] = p_sums[j * d + i] * inv;
                        }
                    }
                }
            }
            Metric::Spherical => {
                let mut means = vec![0.0f32; k * d];
                for j in 0..k {
                    let cnt = self.pass_counts[j];
                    if cnt > 0.0 {
                        let inv = 1.0 / (cnt as f32);
                        for i in 0..d {
                            means[j * d + i] = p_sums[j * d + i] * inv;
                        }
                    } else {
                        means[j * d..(j + 1) * d]
                            .copy_from_slice(&self.centroids[j * d..(j + 1) * d]);
                    }
                }
                normalize_into(&means, &mut self.centroids, d);
            }
        }
        self.finalized = true;
        self.unit_dirty = true;
        Ok(&self.centroids)
    }

    /// Train on `x` (row-major `n * d`), optionally using `init_centroids` (`k * d`).
    pub fn train(
        &mut self,
        x: &[f32],
        init_centroids: Option<&[f32]>,
    ) -> Result<TrainResult, EmeansError> {
        let d = self.cfg.d;
        if x.is_empty() || !x.len().is_multiple_of(d) {
            return Err(EmeansError::InvalidDataLength { len: x.len(), d });
        }
        let total_n = x.len() / d;

        if let Some(max_ppc) = self.cfg.max_points_per_centroid {
            let limit = max_ppc.saturating_mul(self.cfg.k);
            if total_n > limit {
                let mut rng = StdRng::seed_from_u64(self.cfg.seed);
                let mut indices: Vec<usize> = (0..total_n).collect();
                indices.shuffle(&mut rng);
                indices.truncate(limit);
                indices.sort_unstable();

                let mut sub_x = Vec::with_capacity(limit * d);
                for &idx in &indices {
                    sub_x.extend_from_slice(&x[idx * d..(idx + 1) * d]);
                }
                let mut res = self.train_rows(&sub_x, init_centroids)?;
                if let Some(sampled) = res.assignments.take() {
                    let mut full = vec![-1isize; total_n];
                    for (&orig_idx, &lbl) in indices.iter().zip(sampled.iter()) {
                        full[orig_idx] = lbl;
                    }
                    res.assignments = Some(full);
                }
                return Ok(res);
            }
        }

        self.train_rows(x, init_centroids)
    }

    fn train_rows(
        &mut self,
        x: &[f32],
        init_centroids: Option<&[f32]>,
    ) -> Result<TrainResult, EmeansError> {
        let d = self.cfg.d;
        let k = self.cfg.k;
        let n = x.len() / d;

        let init_buf: Vec<f32> = match init_centroids {
            Some(c) => c.to_vec(),
            None => {
                if n == k {
                    x.to_vec()
                } else {
                    let mut rng = StdRng::seed_from_u64(self.cfg.seed);
                    let mut perm: Vec<usize> = (0..n).collect();
                    perm.shuffle(&mut rng);
                    let mut c = Vec::with_capacity(k * d);
                    for &idx in perm.iter().take(k) {
                        c.extend_from_slice(&x[idx * d..(idx + 1) * d]);
                    }
                    c
                }
            }
        };

        self.begin(n, &init_buf)?;
        let recipe = self.recipe.clone().expect("recipe initialized by begin");

        let capture = self.cfg.capture == CaptureMode::Capture;
        let use_handover =
            capture && self.uses_cost_revival() && self.cfg.handover == HandoverMode::SecondNearest;

        let mut assignments = if capture {
            Some(vec![-1isize; n])
        } else {
            None
        };
        let mut second_labels = if use_handover {
            Some(vec![-1isize; n])
        } else {
            None
        };

        let mut last_order = Vec::new();
        let num_epochs = recipe.epoch_rows.len();

        for (epoch_idx, &rows) in recipe.epoch_rows.iter().enumerate() {
            let is_last = epoch_idx + 1 == num_epochs;
            if use_handover && is_last {
                self.handover = Some(HandoverState {
                    done: 0,
                    events: Vec::new(),
                    moved_at: vec![-1isize; k],
                });
            }

            let mut rng = StdRng::seed_from_u64(epoch_seed(self.cfg.seed, epoch_idx));
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut rng);

            for batch_start in (0..rows).step_by(recipe.batch_size) {
                let batch_end = (batch_start + recipe.batch_size).min(rows);
                let batch_idx = &order[batch_start..batch_end];

                if is_last && let Some(ref mut assign_vec) = assignments {
                    let mut lbl_buf = vec![0usize; batch_idx.len()];
                    let mut sec_buf = if second_labels.is_some() {
                        Some(vec![-1isize; batch_idx.len()])
                    } else {
                        None
                    };
                    let stats = self.accumulate_indexed(
                        x,
                        batch_idx,
                        Some(&mut lbl_buf),
                        sec_buf.as_deref_mut(),
                    )?;
                    for (i, &orig_idx) in batch_idx.iter().enumerate() {
                        assign_vec[orig_idx] = lbl_buf[i] as isize;
                    }
                    if let (Some(sec_vec), Some(s_buf)) = (&mut second_labels, sec_buf) {
                        for (i, &orig_idx) in batch_idx.iter().enumerate() {
                            sec_vec[orig_idx] = s_buf[i];
                        }
                    }
                    if let Some(ref mut h) = self.handover {
                        h.done += batch_idx.len();
                    }
                    self.update(&stats)?;
                } else {
                    let stats = self.accumulate_indexed(x, batch_idx, None, None)?;
                    self.update(&stats)?;
                }
            }

            self.end_epoch()?;
            if is_last {
                last_order = order;
            }
        }

        if let (Some(h), Some(assign_vec), Some(sec_vec)) =
            (self.handover.take(), &mut assignments, &second_labels)
        {
            self.handover_finish(
                h,
                &last_order,
                recipe.epoch_rows[num_epochs - 1],
                assign_vec,
                sec_vec,
                x,
            );
        }

        self.finalize()?;
        let obj: Vec<f64> = self.iteration_stats.iter().map(|s| s.obj).collect();
        Ok(TrainResult {
            centroids: self.centroids.clone(),
            assignments,
            obj,
            iteration_stats: self.iteration_stats.clone(),
            recipe,
            n_splits: self.n_splits,
        })
    }

    fn handover_finish(
        &mut self,
        h: HandoverState,
        order: &[usize],
        rows: usize,
        labels: &mut [isize],
        second: &[isize],
        x: &[f32],
    ) {
        if !h.moved_at.iter().any(|&m| m >= 0) {
            return;
        }
        let d = self.cfg.d;
        let k = self.cfg.k;
        let moved_at = &h.moved_at;

        let mut handed = Vec::new();
        let mut search = Vec::new();

        for (rank, &idx) in order.iter().enumerate().take(rows) {
            let lbl = labels[idx];
            if lbl < 0 {
                continue;
            }
            if (rank as isize) < moved_at[lbl as usize] {
                let n2 = second[idx];
                if n2 >= 0 && moved_at[n2 as usize] <= (rank as isize) {
                    labels[idx] = n2;
                    handed.push(idx);
                } else {
                    search.push(idx);
                }
            }
        }

        if !search.is_empty() {
            self.ensure_readout();
            let readout = if self.cfg.metric == Metric::Spherical {
                self.unit_centroids.clone()
            } else {
                self.centroids.clone()
            };
            let chunk_size = self.cfg.assignment_chunk_size;
            for chunk in search.chunks(chunk_size) {
                let m = chunk.len();
                let mut gathered = Vec::with_capacity(m * d);
                for &idx in chunk {
                    gathered.extend_from_slice(&x[idx * d..(idx + 1) * d]);
                }
                let mut d1 = vec![0.0f32; m];
                let mut l1 = vec![0usize; m];
                let mut out = AssignOut {
                    d1: &mut d1,
                    l1: &mut l1,
                    d2: None,
                    l2: None,
                };
                assign_rows(
                    &gathered,
                    &readout,
                    d,
                    k,
                    self.cfg.metric,
                    self.cfg.backend,
                    &mut out,
                );
                for (&idx, &new_lbl) in chunk.iter().zip(l1.iter()) {
                    labels[idx] = new_lbl as isize;
                }
            }
        }

        // Twin reassignment and closing-sum rebuild for affected centroids.
        let mut stale = vec![false; labels.len()];
        for &idx in handed.iter().chain(search.iter()) {
            stale[idx] = true;
        }

        // Flatten events + sentinel (target=0, source=k, born=0) sorted by (source, born).
        let mut flat: Vec<(usize, usize, usize)> = vec![(k, 0, 0)];
        for ev in &h.events {
            for (&t, &s) in ev.target.iter().zip(ev.source.iter()) {
                flat.push((s, t, ev.born));
            }
        }
        flat.sort_by_key(|&(s, _, b)| (s, b));

        self.ensure_readout();
        let readout = if self.cfg.metric == Metric::Spherical {
            &self.unit_centroids
        } else {
            &self.centroids
        };

        for (rank, &idx) in order[..rows].iter().enumerate() {
            if stale[idx] {
                continue;
            }
            let lbl = labels[idx];
            if lbl < 0 {
                continue;
            }
            let lbl_u = lbl as usize;
            // Find first event for `lbl_u` with `born > rank`.
            let pos = flat.partition_point(|&(s, _, b)| (s, b) <= (lbl_u, rank));
            if pos < flat.len() {
                let (src, child, born) = flat[pos];
                if src == lbl_u && born > rank && moved_at[child] == (born as isize) {
                    let row = &x[idx * d..(idx + 1) * d];
                    let own_c = &readout[lbl_u * d..(lbl_u + 1) * d];
                    let child_c = &readout[child * d..(child + 1) * d];
                    let near_own = l2_sqr_f64(row, own_c);
                    let near_child = l2_sqr_f64(row, child_c);
                    if near_child < near_own || (near_child == near_own && child < lbl_u) {
                        labels[idx] = child as isize;
                    }
                }
            }
        }

        let mut rebuild = vec![false; k];
        for &(s, t, _) in &flat {
            if s < k {
                rebuild[s] = true;
                rebuild[t] = true;
            }
        }
        for &idx in handed.iter().chain(search.iter()) {
            let lbl = labels[idx];
            if lbl >= 0 {
                rebuild[lbl as usize] = true;
            }
        }

        if let Some(ref mut p_sums) = self.pass_sums {
            for j in 0..k {
                if rebuild[j] {
                    p_sums[j * d..(j + 1) * d].fill(0.0);
                    self.pass_counts[j] = 0.0;
                }
            }
            for &idx in &order[..rows] {
                let lbl = labels[idx];
                if lbl >= 0 {
                    let j = lbl as usize;
                    if rebuild[j] {
                        self.pass_counts[j] += 1.0;
                        let row = &x[idx * d..(idx + 1) * d];
                        let s_row = &mut p_sums[j * d..(j + 1) * d];
                        for i in 0..d {
                            s_row[i] += row[i];
                        }
                    }
                }
            }
        }
    }

    /// Assign query vectors `x` (`n * d`) to the nearest trained centroid.
    pub fn assign(&self, x: &[f32]) -> Result<AssignResult, EmeansError> {
        let d = self.cfg.d;
        let k = self.cfg.k;
        if x.is_empty() || !x.len().is_multiple_of(d) {
            return Err(EmeansError::InvalidDataLength { len: x.len(), d });
        }
        if !self.finalized && self.centroids.is_empty() {
            return Err(EmeansError::InvalidState("model has not been trained"));
        }
        let n = x.len() / d;
        let mut distances = vec![0.0f32; n];
        let mut labels = vec![0usize; n];
        let chunk_size = self.cfg.assignment_chunk_size;

        for (chunk_idx, x_chunk) in x.chunks(chunk_size * d).enumerate() {
            let m = x_chunk.len() / d;
            let off = chunk_idx * chunk_size;
            let mut out = AssignOut {
                d1: &mut distances[off..off + m],
                l1: &mut labels[off..off + m],
                d2: None,
                l2: None,
            };
            assign_rows(
                x_chunk,
                &self.centroids,
                d,
                k,
                self.cfg.metric,
                self.cfg.backend,
                &mut out,
            );
        }

        if self.cfg.metric == Metric::Spherical {
            for dist in distances.iter_mut() {
                *dist *= -0.5;
            }
        }

        Ok(AssignResult { distances, labels })
    }
}

fn init_random_axes(k: usize, d: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut axes = vec![0.0f32; k * d];
    for row in axes.chunks_exact_mut(d) {
        let mut sq = 0.0f64;
        for v in row.iter_mut() {
            // Box-Muller standard normal
            let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *v = z as f32;
            sq += z * z;
        }
        let norm = sq.sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for v in row.iter_mut() {
                *v = ((*v as f64) * inv) as f32;
            }
        }
    }
    axes
}

fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

#[inline]
fn l2_sqr_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = (x as f64) - (y as f64);
        sum += diff * diff;
    }
    sum
}
