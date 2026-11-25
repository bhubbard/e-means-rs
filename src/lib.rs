//! `e-means`: Rust port of Meta FAIR's e-means clustering algorithm.
//!
//! Implements Algorithm 1 from *"e-means: outperforming k-means in one-tenth the epochs"*
//! (Bessa et al., 2026) with cost-based and mass-based cluster revival, Oja's rule
//! long-axis splitting, closing handover, spherical mode, and INT8/FP8 quantization.

pub mod distance;
pub mod emeans;
pub mod ivf_flat;
pub mod lloyd;
pub mod quantized;
pub mod recipe;
pub mod sharded;
pub mod spherical;

pub use distance::{AssignBackend, Metric};
pub use emeans::{
    AssignResult, BatchStats, CaptureMode, Emeans, EmeansBuilder, EmeansConfig, EmeansError,
    EpochStat, HandoverMode, TrainResult,
};
pub use ivf_flat::IndexIVFFlat;
pub use lloyd::{LloydConfig, LloydKmeans, LloydResult};
pub use quantized::{Fp8E4M3, Fp8Quantizer, Int8Quantizer, QuantizedEmeans};
pub use recipe::{BatchLaw, Recipe, ResolvedBatchLaw, RevivalMode};
pub use sharded::{ShardedEmeans, ShardedEmeansIvf};
