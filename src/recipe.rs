//! Scaling laws, batch/step recipes, cosine annealing, and epoch scheduling
//! for e-means (Algorithm 1 of Bessa et al., 2026).

/// Minimum step factor relative to `alpha0` in the cosine schedule.
pub const ALPHA_MIN_FACTOR: f64 = 0.1;

/// Number of logical batch updates between cluster revival checks.
pub const CHECK_UNUSED_EVERY: usize = 5;

/// Initial EMA mass `N_j` for each centroid (`2^-16`).
pub const INIT_USAGE: f32 = 1.0 / 65536.0;

/// Default mass-based revival threshold `tau`: revive `N_j < tau * mean(N)`.
pub const DEFAULT_UNUSED_THRESHOLD: f64 = 0.1;

/// Removal cost threshold `tau_R`: revive if `R_j <= 0` or `R_j < 0.3 * median(R)`.
pub const REMOVAL_THRESHOLD: f64 = 0.3;

/// Above this dimension, `BatchLaw::Auto` selects `Throughput`; otherwise `Quality`.
pub const AUTO_QUALITY_MAX_D: usize = 16;

/// At or below this dimension, mass-based default revival turns off (`0.0`).
pub const NO_REVIVAL_MAX_D: usize = 2;

const THROUGHPUT_LAW: (f64, f64, f64, f64) = (0.035_134, 0.134_749, 0.876_473, 0.813_344);
const QUALITY_LAW: (f64, f64, f64, f64) = (0.004_391_75, 0.134_749, 0.876_473, 0.813_344);
const ALPHA_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

/// Batch-size scaling law selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchLaw {
    /// Selects `Quality` for `d <= 16` and `Throughput` for `d > 16`.
    #[default]
    Auto,
    /// Larger batch size optimized for high-dimensional throughput.
    Throughput,
    /// ~8x smaller batch size for better quality in low dimensions.
    Quality,
}

/// Resolved concrete batch law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedBatchLaw {
    Throughput,
    Quality,
    Manual,
}

/// Cluster revival strategy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RevivalMode {
    /// Cost-based revival using removal cost `R_j = EMA(d2 - d1)` and Oja's rule.
    #[default]
    CostBased,
    /// Mass-based revival: revive clusters with `N_j < threshold * mean(N)`.
    MassBased(f64),
    /// Disable cluster revival.
    Disabled,
}

/// Resolved training plan for a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    pub batch_law: ResolvedBatchLaw,
    pub batch_size: usize,
    pub alpha0: f64,
    pub nepochs: f64,
    pub steps_per_pass: usize,
    pub total_steps: usize,
    pub epoch_rows: Vec<usize>,
    pub epoch_steps: Vec<usize>,
}

/// Round a positive value to the nearest power of two.
#[must_use]
pub fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let exp = value.log2().floor() as u32;
    let low = 1usize << exp;
    let high = low.saturating_mul(2);
    if (value / low as f64) < (high as f64 / value) {
        low
    } else {
        high
    }
}

/// Initial EMA step `alpha0` coupled to `batch_size` by the fitted step law.
#[must_use]
pub fn step_law(n: usize, k: usize, nepochs: f64, batch_size: usize) -> f64 {
    let (c, ak, ab, an, at) = ALPHA_LAW;
    let alpha0 = c
        * (k as f64).powf(ak)
        * (batch_size as f64).powf(ab)
        * (n as f64).powf(an)
        * nepochs.powf(at);
    alpha0.clamp(1e-4, 1.0)
}

/// Resolve `BatchLaw` from dimension `d`.
#[must_use]
pub fn resolve_batch_law(law: BatchLaw, d: usize) -> ResolvedBatchLaw {
    match law {
        BatchLaw::Auto => {
            if d <= AUTO_QUALITY_MAX_D {
                ResolvedBatchLaw::Quality
            } else {
                ResolvedBatchLaw::Throughput
            }
        }
        BatchLaw::Throughput => ResolvedBatchLaw::Throughput,
        BatchLaw::Quality => ResolvedBatchLaw::Quality,
    }
}

/// Compute row counts per epoch: whole passes, rounded fractional prefix, then closing pass.
#[must_use]
pub fn epoch_rows(n: usize, nepochs: f64) -> Vec<usize> {
    let whole = nepochs.floor() as usize;
    let frac = nepochs - (whole as f64);
    // Match Python's round-half-to-even for fractional pass.
    let remainder = (frac * n as f64).round_ties_even() as usize;
    if whole == 0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole - 1];
    if remainder > 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// Derive a reproducible permutation seed for `epoch` without colliding with `seed`.
#[must_use]
pub fn epoch_seed(seed: u64, epoch: usize) -> u64 {
    let base = seed & 0xFFFF_FFFF;
    let mut s = (base.wrapping_mul(1_000_003).wrapping_add(epoch as u64)) & 0xFFFF_FFFF;
    if s == base {
        s = s.wrapping_add(0x9E37_79B9) & 0xFFFF_FFFF;
    }
    s
}

/// Cosine-anneal `alpha0` toward `ALPHA_MIN_FACTOR * alpha0`.
#[must_use]
pub fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = (step.min(total_steps) as f64) / (total_steps.max(1) as f64);
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Resolve the full training [`Recipe`] from dataset size, cluster count, and options.
#[must_use]
pub fn resolve_recipe(
    n: usize,
    k: usize,
    d: usize,
    nepochs: f64,
    batch_law: BatchLaw,
    batch_size: Option<usize>,
    alpha0: Option<f64>,
) -> Recipe {
    let resolved_law = if batch_size.is_some() {
        ResolvedBatchLaw::Manual
    } else {
        resolve_batch_law(batch_law, d)
    };

    let resolved_batch = match batch_size {
        Some(b) => b.max(1).min(n.max(1)),
        None => {
            let (c, bk, bn, bt) = match resolved_law {
                ResolvedBatchLaw::Quality => QUALITY_LAW,
                ResolvedBatchLaw::Throughput | ResolvedBatchLaw::Manual => THROUGHPUT_LAW,
            };
            let raw = c * (k as f64).powf(bk) * (n as f64).powf(bn) * nepochs.powf(bt);
            let mut b = round_power_of_two(raw);
            let cap = match resolved_law {
                ResolvedBatchLaw::Throughput => round_power_of_two(n as f64 / 2.0),
                ResolvedBatchLaw::Quality | ResolvedBatchLaw::Manual => (n / 2).max(1),
            };
            b = b.min(cap).min(n.max(1)).max(1);
            b
        }
    };

    let resolved_alpha0 = alpha0.unwrap_or_else(|| step_law(n, k, nepochs, resolved_batch));
    let rows = epoch_rows(n, nepochs);
    let epoch_steps: Vec<usize> = rows.iter().map(|&r| r.div_ceil(resolved_batch)).collect();
    let total_steps = epoch_steps.iter().sum();
    let steps_per_pass = n.div_ceil(resolved_batch);

    Recipe {
        batch_law: resolved_law,
        batch_size: resolved_batch,
        alpha0: resolved_alpha0,
        nepochs,
        steps_per_pass,
        total_steps,
        epoch_rows: rows,
        epoch_steps,
    }
}
