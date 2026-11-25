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

## Benchmarks (Apple M3 Pro, Release Build)

### e-means vs Lloyd's k-means

| Workload | N | d | K | e-means | e-means Loss | Lloyd's | Lloyd Loss | Speedup | Loss Delta | Splits |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| Euclidean High-D (3ep) | 50000 | 64 | 256 | 3.0 ep (0.066s) | 16.7439 | 25 ep (0.099s) | 17.6293 | 1.52x | -5.02% | 912 |
| Euclidean High-D (5ep) | 50000 | 64 | 256 | 5.0 ep (0.087s) | 17.2852 | 25 ep (0.121s) | 17.6293 | 1.39x | -1.95% | 932 |
| Euclidean Low-D (Quality) | 100000 | 16 | 256 | 3.0 ep (0.147s) | 2.2756 | 25 ep (0.124s) | 2.8435 | 0.85x | -19.97% | 1250 |
| Euclidean Large (128D) | 100000 | 128 | 512 | 3.0 ep (0.214s) | 41.1182 | 25 ep (0.397s) | 35.2341 | 1.85x | +16.70% | 1253 |
| Spherical (64D) | 50000 | 64 | 256 | 3.0 ep (0.059s) | 0.1165 | 25 ep (0.076s) | 0.2106 | 1.28x | -44.70% | 197 |

### Backend Comparison (`n=50000, d=64, k=256, 3 epochs, CostBased`)

| Backend | Wall Time (s) | Speedup vs Scalar | Final Loss | Splits |
|---|---:|---:|---:|---:|
| Scalar (1-thread) | 1.210 | 1.00x | 13.5658 | 790 |
| PureRust (Rayon) | 0.195 | 6.20x | 13.5658 | 790 |
| Accelerate (BLAS+Rayon) | 0.058 | 20.94x | 13.5658 | 790 |

### Precision / Quantized E-means (`n=50000, d=64, k=256, 5 epochs`)

| Representation | Wall Time (s) | Final Loss (on FP32 x) | Delta vs FP32 |
|---|---:|---:|---:|
| FP32 | 0.066 | 16.2754 | +0.00% |
| INT8 | 0.068 | 16.6151 | +2.09% |
| FP8 E4M3 | 0.084 | 13.0040 | -20.10% |

### Sharded & IVF-Flat (`n=50000, d=64, k=1024`)

- `ShardedEmeans` (4 shards, 3 ep): `0.618s`, Loss `9.4830`, Splits `2323`
- `IndexIVFFlat` 2-NN (`nlist=64, nprobe=4`, 10,000 queries): `0.0247s` (`97.5%` recall@1, `ndis=4927550`) vs Exhaustive `Accelerate` `0.0034s`
