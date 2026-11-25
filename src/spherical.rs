//! Bounded, zero-preserving row normalization for spherical e-means.

/// Project rows of `source` (shape `n x d`) to unit L2 norm into `out`.
///
/// Zero rows remain zero. `source` and `out` may alias if they have the same
/// slice (use [`normalize_in_place`] for in-place normalization).
///
/// # Panics
///
/// Panics if `d == 0` or `source.len() != out.len()` or `source.len() % d != 0`.
pub fn normalize_into(source: &[f32], out: &mut [f32], d: usize) {
    assert!(d > 0, "dimension d must be > 0");
    assert_eq!(
        source.len(),
        out.len(),
        "source len ({}) must match out len ({})",
        source.len(),
        out.len()
    );
    assert_eq!(source.len() % d, 0, "buffer len must be divisible by d");

    for (src_row, dst_row) in source.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        normalize_row(src_row, dst_row);
    }
}

/// Normalize rows of `buf` (shape `n x d`) in place, preserving zero rows.
///
/// # Panics
///
/// Panics if `d == 0` or `buf.len() % d != 0`.
pub fn normalize_in_place(buf: &mut [f32], d: usize) {
    assert!(d > 0, "dimension d must be > 0");
    assert_eq!(buf.len() % d, 0, "buffer len must be divisible by d");

    for row in buf.chunks_exact_mut(d) {
        let max_abs = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if max_abs == 0.0 {
            continue;
        }
        let inv_max = 1.0 / max_abs;
        let mut sq_sum = 0.0f64;
        for v in row.iter_mut() {
            let scaled = *v * inv_max;
            *v = scaled;
            sq_sum += (scaled as f64) * (scaled as f64);
        }
        let norm = sq_sum.sqrt() as f32;
        if norm > 0.0 {
            let inv_norm = 1.0 / norm;
            for v in row.iter_mut() {
                *v *= inv_norm;
            }
        }
    }
}

fn normalize_row(src: &[f32], dst: &mut [f32]) {
    let max_abs = src.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if max_abs == 0.0 {
        dst.fill(0.0);
        return;
    }
    let inv_max = 1.0 / max_abs;
    let mut sq_sum = 0.0f64;
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        let scaled = *s * inv_max;
        *d = scaled;
        sq_sum += (scaled as f64) * (scaled as f64);
    }
    let norm = sq_sum.sqrt() as f32;
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        for d in dst.iter_mut() {
            *d *= inv_norm;
        }
    }
}
