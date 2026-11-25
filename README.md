# e-means-rs

Pure Rust port of Meta FAIR's [`faiss/contrib/e_means`](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means) (*e-means: outperforming k-means in one-tenth the epochs*, Bessa et al., 2026).

## Features

- **Core E-means (`Emeans`)**: Algorithm 1 with `Throughput`, `Quality`, and `Auto` (`d <= 16` vs `d > 16`) batch/step scaling laws, cosine step annealing, fractional epoch budgets, and exact closing means.
- **Cluster Revival**:
  - `RevivalMode::CostBased`: Removal cost `R_j = EMA(d2 - d1)` + Oja's rule long-axis splitting (`c_j -/+ sqrt(2 ||U_j|| / (pi N_j)) u_j`), plus `second_nearest` closing handover and twin reassignment.
  - `RevivalMode::MassBased(tau)`: Weighted Gumbel-top-k donor sampling.
- **Spherical E-means (`Metric::Spherical`)**: Zero-preserving bounded row normalization and inner-product assignment.
- **Assignment Backends (`AssignBackend`)**:
  - `Accelerate` (default on macOS): Apple `cblas_sgemm` + Rayon.
  - `PureRust`: Parallel tiled dot-product + Rayon.
  - `Scalar`: Single-threaded baseline.
- **Quantized (`QuantizedEmeans`)**: Symmetric `Int8Quantizer` and scale-fitted `Fp8Quantizer` (E4M3).
- **Large-Scale (`ShardedEmeans`, `IndexIVFFlat`, `ShardedEmeansIvf`)**: Multi-shard centroid ownership with parallel 2-NN reduction, cross-shard revival, and coarse IVF-Flat routing.

## Quick Start

```rust
use e_means::{EmeansConfig, Metric};

let d = 64;
let k = 256;
let data = vec![0.0f32; 10_000 * d];

let mut em = EmeansConfig::builder(d, k)
    .nepochs(5.0)
    .metric(Metric::Euclidean)
    .build()?;

let result = em.train(&data, None)?;
let assigned = em.assign(&data)?;
# Ok::<(), e_means::EmeansError>(())
```

## Test & Benchmark

```bash
cargo test
cargo run --release --bin bench_compare
```
