//! Nearest-centroid assignment kernels (k=1 and k=2) for Euclidean L2^2 and Spherical IP.

use rayon::prelude::*;

/// Distance metric for clustering and assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    /// Squared Euclidean distance `||x - c||^2`.
    #[default]
    Euclidean,
    /// Cosine similarity on unit-normalized vectors (maximizes `x . c`).
    Spherical,
}

/// Execution backend for nearest-centroid assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssignBackend {
    /// Apple Accelerate `cblas_sgemm` + Rayon (falls back to `PureRust` on non-macOS).
    #[default]
    Accelerate,
    /// Pure Rust parallel tiled GEMM / SIMD-friendly L2/IP + Rayon.
    PureRust,
    /// Single-threaded scalar baseline.
    Scalar,
}

/// Output buffers for nearest-centroid assignment.
pub struct AssignOut<'a> {
    pub d1: &'a mut [f32],
    pub l1: &'a mut [usize],
    pub d2: Option<&'a mut [f32]>,
    pub l2: Option<&'a mut [usize]>,
}

#[cfg(all(target_os = "macos", feature = "accelerate"))]
mod cblas {
    pub const CBLAS_ROW_MAJOR: i32 = 101;
    pub const CBLAS_NO_TRANS: i32 = 111;
    pub const CBLAS_TRANS: i32 = 112;

    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}

#[inline]
pub(crate) fn l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let chunks = a.len() / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    sum += (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn sq_norm_f32(a: &[f32]) -> f32 {
    dot_f32(a, a)
}

/// Assign `x` (`n x d`) to nearest (and optionally 2nd-nearest) centroid in `centroids` (`k x d`).
///
/// For `Metric::Spherical`, `d1` (and `d2`) store `-2.0 * inner_product`, matching
/// `NumpyEmeans._assign_rows` so that `2.0 + d1 = 2 - 2 * cos`.
pub fn assign_rows(
    x: &[f32],
    centroids: &[f32],
    d: usize,
    k: usize,
    metric: Metric,
    backend: AssignBackend,
    out: &mut AssignOut<'_>,
) {
    let n = x.len() / d;
    if n == 0 || k == 0 {
        return;
    }

    match backend {
        AssignBackend::Scalar => assign_scalar(x, centroids, d, k, metric, out),
        AssignBackend::PureRust => assign_pure_rust(x, centroids, d, k, metric, out),
        AssignBackend::Accelerate => {
            #[cfg(all(target_os = "macos", feature = "accelerate"))]
            {
                if d >= 4 && k >= 16 && n >= 16 {
                    assign_accelerate(x, centroids, d, k, metric, out);
                } else {
                    assign_pure_rust(x, centroids, d, k, metric, out);
                }
            }
            #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
            {
                assign_pure_rust(x, centroids, d, k, metric, out);
            }
        }
    }
}

fn assign_row_direct(
    row: &[f32],
    centroids: &[f32],
    d: usize,
    k: usize,
    metric: Metric,
    need_second: bool,
) -> (f32, usize, f32, usize) {
    let mut best_d1 = f32::INFINITY;
    let mut best_l1 = 0usize;
    let mut best_d2 = f32::INFINITY;
    let mut best_l2 = 0usize;

    for (j, c) in centroids.chunks_exact(d).enumerate().take(k) {
        let dist = match metric {
            Metric::Euclidean => l2_sqr(row, c),
            Metric::Spherical => -2.0 * dot_f32(row, c),
        };
        if dist < best_d1 {
            best_d2 = best_d1;
            best_l2 = best_l1;
            best_d1 = dist;
            best_l1 = j;
        } else if need_second && dist < best_d2 {
            best_d2 = dist;
            best_l2 = j;
        }
    }

    (best_d1, best_l1, best_d2, best_l2)
}

fn assign_scalar(
    x: &[f32],
    centroids: &[f32],
    d: usize,
    k: usize,
    metric: Metric,
    out: &mut AssignOut<'_>,
) {
    let need_second = out.d2.is_some() || out.l2.is_some();
    for (i, row) in x.chunks_exact(d).enumerate() {
        let (d1, l1, d2, l2) = assign_row_direct(row, centroids, d, k, metric, need_second);
        out.d1[i] = d1;
        out.l1[i] = l1;
        if let Some(ref mut d2_buf) = out.d2 {
            d2_buf[i] = d2;
        }
        if let Some(ref mut l2_buf) = out.l2 {
            l2_buf[i] = l2;
        }
    }
}

fn assign_pure_rust(
    x: &[f32],
    centroids: &[f32],
    d: usize,
    k: usize,
    metric: Metric,
    out: &mut AssignOut<'_>,
) {
    let n = x.len() / d;
    let need_second = out.d2.is_some() || out.l2.is_some();

    // For small k*d or small n, parallel direct scan avoids extra norm buffers.
    if n * k < 65_536 || d < 8 {
        let mut results = vec![(0.0f32, 0usize, 0.0f32, 0usize); n];
        results.par_iter_mut().enumerate().for_each(|(i, slot)| {
            let row = &x[i * d..(i + 1) * d];
            *slot = assign_row_direct(row, centroids, d, k, metric, need_second);
        });
        write_results(&results, out);
        return;
    }

    // Tiled pure-Rust GEMM-like dot-product accumulation with precomputed norms.
    let c_norms: Vec<f32> = match metric {
        Metric::Euclidean => centroids.chunks_exact(d).map(sq_norm_f32).collect(),
        Metric::Spherical => Vec::new(),
    };

    const ROW_BLOCK: usize = 64;
    let mut results = vec![(0.0f32, 0usize, 0.0f32, 0usize); n];

    results
        .par_chunks_mut(ROW_BLOCK)
        .enumerate()
        .for_each(|(blk_idx, res_chunk)| {
            let row_start = blk_idx * ROW_BLOCK;
            let m = res_chunk.len();
            let mut dots = vec![0.0f32; m * k];

            for (r, dot_row) in dots.chunks_exact_mut(k).enumerate() {
                let row = &x[(row_start + r) * d..(row_start + r + 1) * d];
                for (j, c) in centroids.chunks_exact(d).enumerate().take(k) {
                    dot_row[j] = dot_f32(row, c);
                }
            }

            for (r, slot) in res_chunk.iter_mut().enumerate() {
                let row = &x[(row_start + r) * d..(row_start + r + 1) * d];
                let x_norm = match metric {
                    Metric::Euclidean => sq_norm_f32(row),
                    Metric::Spherical => 0.0,
                };
                let dot_row = &dots[r * k..(r + 1) * k];
                *slot = argmin_from_dots(dot_row, x_norm, &c_norms, metric, need_second);
            }
        });

    write_results(&results, out);
}

#[cfg(all(target_os = "macos", feature = "accelerate"))]
fn assign_accelerate(
    x: &[f32],
    centroids: &[f32],
    d: usize,
    k: usize,
    metric: Metric,
    out: &mut AssignOut<'_>,
) {
    let n = x.len() / d;
    let need_second = out.d2.is_some() || out.l2.is_some();

    let c_norms: Vec<f32> = match metric {
        Metric::Euclidean => centroids.chunks_exact(d).map(sq_norm_f32).collect(),
        Metric::Spherical => Vec::new(),
    };

    // Process in row blocks so the `block_rows x k` score matrix fits comfortably in L2/L3 cache.
    let block_rows = (262_144 / k.max(1)).clamp(32, 1024);
    let mut results = vec![(0.0f32, 0usize, 0.0f32, 0usize); n];

    results
        .par_chunks_mut(block_rows)
        .enumerate()
        .for_each(|(blk_idx, res_chunk)| {
            let row_start = blk_idx * block_rows;
            let m = res_chunk.len();
            let x_blk = &x[row_start * d..(row_start + m) * d];
            let mut dots = vec![0.0f32; m * k];

            // SAFETY: x_blk has m*d elements, centroids has k*d elements, dots has m*k elements.
            unsafe {
                cblas::cblas_sgemm(
                    cblas::CBLAS_ROW_MAJOR,
                    cblas::CBLAS_NO_TRANS,
                    cblas::CBLAS_TRANS,
                    m as i32,
                    k as i32,
                    d as i32,
                    1.0,
                    x_blk.as_ptr(),
                    d as i32,
                    centroids.as_ptr(),
                    d as i32,
                    0.0,
                    dots.as_mut_ptr(),
                    k as i32,
                );
            }

            for (r, slot) in res_chunk.iter_mut().enumerate() {
                let row = &x_blk[r * d..(r + 1) * d];
                let x_norm = match metric {
                    Metric::Euclidean => sq_norm_f32(row),
                    Metric::Spherical => 0.0,
                };
                let dot_row = &dots[r * k..(r + 1) * k];
                *slot = argmin_from_dots(dot_row, x_norm, &c_norms, metric, need_second);
            }
        });

    write_results(&results, out);
}

#[inline]
fn argmin_from_dots(
    dot_row: &[f32],
    x_norm: f32,
    c_norms: &[f32],
    metric: Metric,
    need_second: bool,
) -> (f32, usize, f32, usize) {
    let mut best_d1 = f32::INFINITY;
    let mut best_l1 = 0usize;
    let mut best_d2 = f32::INFINITY;
    let mut best_l2 = 0usize;

    match metric {
        Metric::Euclidean => {
            for (j, (&dot, &c_norm)) in dot_row.iter().zip(c_norms.iter()).enumerate() {
                let dist = (x_norm + c_norm - 2.0 * dot).max(0.0);
                if dist < best_d1 {
                    best_d2 = best_d1;
                    best_l2 = best_l1;
                    best_d1 = dist;
                    best_l1 = j;
                } else if need_second && dist < best_d2 {
                    best_d2 = dist;
                    best_l2 = j;
                }
            }
        }
        Metric::Spherical => {
            for (j, &dot) in dot_row.iter().enumerate() {
                let dist = -2.0 * dot;
                if dist < best_d1 {
                    best_d2 = best_d1;
                    best_l2 = best_l1;
                    best_d1 = dist;
                    best_l1 = j;
                } else if need_second && dist < best_d2 {
                    best_d2 = dist;
                    best_l2 = j;
                }
            }
        }
    }

    (best_d1, best_l1, best_d2, best_l2)
}

fn write_results(results: &[(f32, usize, f32, usize)], out: &mut AssignOut<'_>) {
    for (i, &(d1, l1, d2, l2)) in results.iter().enumerate() {
        out.d1[i] = d1;
        out.l1[i] = l1;
        if let Some(ref mut d2_buf) = out.d2 {
            d2_buf[i] = d2;
        }
        if let Some(ref mut l2_buf) = out.l2 {
            l2_buf[i] = l2;
        }
    }
}
