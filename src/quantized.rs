//! Quantized representations (INT8 and FP8 E4M3) for memory-efficient e-means.

use crate::emeans::{Emeans, EmeansConfig, EmeansError, TrainResult};

/// Symmetric INT8 quantizer with per-dataset or per-vector scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Int8Quantizer {
    pub scale: f32,
}

impl Int8Quantizer {
    /// Fit a symmetric INT8 scale from `x` (`max(|x|) / 127.0`).
    #[must_use]
    pub fn fit(x: &[f32]) -> Self {
        let max_abs = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        Self { scale }
    }

    /// Encode `f32` slice to `i8`.
    #[must_use]
    pub fn encode(&self, x: &[f32]) -> Vec<i8> {
        let inv = 1.0 / self.scale;
        x.iter()
            .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
            .collect()
    }

    /// Decode `i8` slice back to `f32`.
    #[must_use]
    pub fn decode(&self, q: &[i8]) -> Vec<f32> {
        q.iter().map(|&v| (v as f32) * self.scale).collect()
    }
}

/// Software IEEE-like FP8 E4M3 (1 sign, 4 exponent, 3 mantissa, bias=7) encoder/decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp8E4M3;

impl Fp8E4M3 {
    const MAX_FINITE: f32 = 448.0;

    /// Encode a single `f32` into FP8 E4M3 byte.
    #[inline]
    #[must_use]
    pub fn encode_f32(val: f32) -> u8 {
        if val.is_nan() {
            return 0x7F;
        }
        let sign = ((val.to_bits() >> 31) & 1) as u8;
        let abs = val.abs();
        if abs == 0.0 {
            return sign << 7;
        }
        let clamped = abs.min(Self::MAX_FINITE);
        let bits = clamped.to_bits();
        let exp32 = ((bits >> 23) & 0xFF) as i32 - 127;
        let mant32 = bits & 0x7F_FFFF;

        let exp8 = exp32 + 7;
        if exp8 <= 0 {
            // Subnormal FP8
            let shift = 1 - exp8;
            if shift > 3 {
                return sign << 7;
            }
            let mant = ((0x80_0000 | mant32) >> (20 + shift)) as u8;
            return (sign << 7) | (mant & 0x07);
        }
        if exp8 >= 15 {
            // Max finite: exp=15, mant=6 (0b111_1110 = 448)
            return (sign << 7) | 0x7E;
        }
        // Round to nearest 3-bit mantissa
        let mant = ((mant32 + (1 << 19)) >> 20) as u8;
        if mant >= 8 {
            let next_exp = (exp8 + 1) as u8;
            if next_exp >= 15 {
                return (sign << 7) | 0x7E;
            }
            (sign << 7) | (next_exp << 3)
        } else {
            (sign << 7) | ((exp8 as u8) << 3) | (mant & 0x07)
        }
    }

    /// Decode a single FP8 E4M3 byte into `f32`.
    #[inline]
    #[must_use]
    pub fn decode_u8(byte: u8) -> f32 {
        let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (byte >> 3) & 0x0F;
        let mant = byte & 0x07;
        if exp == 0 {
            if mant == 0 {
                return sign * 0.0;
            }
            return sign * (mant as f32) * 2.0f32.powi(-9);
        }
        if exp == 15 && mant == 7 {
            return f32::NAN;
        }
        let val = (1.0 + (mant as f32) / 8.0) * 2.0f32.powi(exp as i32 - 7);
        sign * val
    }

    /// Encode a slice of `f32` into FP8 E4M3 bytes.
    #[must_use]
    pub fn encode_slice(x: &[f32]) -> Vec<u8> {
        x.iter().map(|&v| Self::encode_f32(v)).collect()
    }

    /// Decode a slice of FP8 E4M3 bytes into `f32`.
    #[must_use]
    pub fn decode_slice(q: &[u8]) -> Vec<f32> {
        q.iter().map(|&b| Self::decode_u8(b)).collect()
    }
}

/// Scale-fitted FP8 E4M3 quantizer (`scale = max(|x|) / 448.0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fp8Quantizer {
    pub scale: f32,
}

impl Fp8Quantizer {
    /// Fit an FP8 E4M3 scale from `x`.
    #[must_use]
    pub fn fit(x: &[f32]) -> Self {
        let max_abs = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs > 0.0 {
            max_abs / Fp8E4M3::MAX_FINITE
        } else {
            1.0
        };
        Self { scale }
    }

    /// Encode `f32` slice to scaled FP8 E4M3 bytes.
    #[must_use]
    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        let inv = 1.0 / self.scale;
        x.iter().map(|&v| Fp8E4M3::encode_f32(v * inv)).collect()
    }

    /// Decode scaled FP8 E4M3 bytes back to `f32`.
    #[must_use]
    pub fn decode(&self, q: &[u8]) -> Vec<f32> {
        q.iter()
            .map(|&b| Fp8E4M3::decode_u8(b) * self.scale)
            .collect()
    }
}

/// Quantized E-means trainer supporting INT8 and FP8 E4M3 encoded inputs.
pub struct QuantizedEmeans {
    cfg: EmeansConfig,
}

impl QuantizedEmeans {
    #[must_use]
    pub fn new(cfg: EmeansConfig) -> Self {
        Self { cfg }
    }

    /// Train e-means on INT8-quantized data (`x` is encoded to INT8, then trained in decoded FP32).
    pub fn train_int8(
        &self,
        x: &[f32],
        init_centroids: Option<&[f32]>,
    ) -> Result<(TrainResult, Int8Quantizer), EmeansError> {
        let q = Int8Quantizer::fit(x);
        let encoded = q.encode(x);
        let decoded = q.decode(&encoded);
        let mut em = Emeans::new(self.cfg.clone())?;
        let res = em.train(&decoded, init_centroids)?;
        Ok((res, q))
    }

    /// Train e-means on scale-fitted FP8-E4M3-quantized data.
    pub fn train_fp8(
        &self,
        x: &[f32],
        init_centroids: Option<&[f32]>,
    ) -> Result<(TrainResult, Fp8Quantizer), EmeansError> {
        let q = Fp8Quantizer::fit(x);
        let encoded = q.encode(x);
        let decoded = q.decode(&encoded);
        let mut em = Emeans::new(self.cfg.clone())?;
        let res = em.train(&decoded, init_centroids)?;
        Ok((res, q))
    }
}
