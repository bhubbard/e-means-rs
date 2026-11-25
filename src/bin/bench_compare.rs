//! Benchmark comparison: e-means vs Lloyd's k-means, backends, and quantized variants.

use std::time::Instant;

use e_means::distance::{AssignOut, assign_rows};
use e_means::spherical::normalize_in_place;
use e_means::{
    AssignBackend, BatchLaw, CaptureMode, EmeansConfig, IndexIVFFlat, LloydConfig, LloydKmeans,
    Metric, QuantizedEmeans, RevivalMode, ShardedEmeans,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

fn generate_mixture(n: usize, d: usize, n_modes: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut modes = vec![0.0f32; n_modes * d];
    for v in modes.iter_mut() {
        let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.random::<f32>();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }

    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        // Skewed cluster sizes (Zipf-like) to stress-test cluster revival.
        let r: f32 = rng.random::<f32>();
        let mode = ((r * r) * (n_modes as f32)) as usize % n_modes;
        for j in 0..d {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random::<f32>();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            data[i * d + j] = modes[mode * d + j] + spread * noise;
        }
    }
    data
}

fn sample_init(x: &[f32], n: usize, d: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    let mut init = Vec::with_capacity(k * d);
    for &idx in perm.iter().take(k) {
        init.extend_from_slice(&x[idx * d..(idx + 1) * d]);
    }
    init
}

fn eval_loss(x: &[f32], centroids: &[f32], n: usize, d: usize, k: usize, metric: Metric) -> f64 {
    let mut d1 = vec![0.0f32; n];
    let mut l1 = vec![0usize; n];
    let mut out = AssignOut {
        d1: &mut d1,
        l1: &mut l1,
        d2: None,
        l2: None,
    };
    assign_rows(
        x,
        centroids,
        d,
        k,
        metric,
        AssignBackend::Accelerate,
        &mut out,
    );
    let mut total = 0.0f64;
    if metric == Metric::Spherical {
        total += 2.0 * (n as f64);
    }
    for &v in &d1 {
        total += v as f64;
    }
    total / (n as f64)
}

fn run_emeans_vs_lloyd(
    label: &str,
    n: usize,
    d: usize,
    k: usize,
    metric: Metric,
    emeans_epochs: f64,
    lloyd_iters: usize,
) {
    let mut x = generate_mixture(n, d, k, 0.35, 2026);
    if metric == Metric::Spherical {
        normalize_in_place(&mut x, d);
    }
    let init = sample_init(&x, n, d, k, 1234);

    // 1. e-means
    let t0 = Instant::now();
    let mut em = EmeansConfig::builder(d, k)
        .nepochs(emeans_epochs)
        .metric(metric)
        .batch_law(BatchLaw::Auto)
        .revival(RevivalMode::CostBased)
        .capture(CaptureMode::None)
        .build()
        .expect("valid emeans config");
    let em_res = em.train(&x, Some(&init)).expect("emeans train");
    let em_time = t0.elapsed().as_secs_f64();
    let em_loss = eval_loss(&x, &em_res.centroids, n, d, k, metric);

    // 2. Lloyd's k-means
    let mut lcfg = LloydConfig::new(d, k);
    lcfg.niter = lloyd_iters;
    lcfg.metric = metric;
    let t1 = Instant::now();
    let l_res = LloydKmeans::new(lcfg)
        .train(&x, Some(&init))
        .expect("lloyd train");
    let l_time = t1.elapsed().as_secs_f64();
    let l_loss = eval_loss(&x, &l_res.centroids, n, d, k, metric);

    let speedup = l_time / em_time;
    let loss_delta_pct = ((em_loss - l_loss) / l_loss) * 100.0;

    println!(
        "| {label:<26} | {n:>7} | {d:>3} | {k:>4} | {emeans_epochs:>4.1} ep ({em_time:>6.3}s) | {em_loss:>8.4} | {lloyd_iters:>2} ep ({l_time:>6.3}s) | {l_loss:>8.4} | {speedup:>5.2}x | {loss_delta_pct:>+6.2}% | {:>4} |",
        em_res.n_splits
    );
}

fn run_backend_bench(n: usize, d: usize, k: usize) {
    let x = generate_mixture(n, d, k, 0.35, 999);
    let init = sample_init(&x, n, d, k, 1234);

    println!("\n### Backend Comparison (n={n}, d={d}, k={k}, 3 epochs, CostBased)\n");
    println!("| Backend | Wall Time (s) | Speedup vs Scalar | Final Loss | Splits |");
    println!("|---|---:|---:|---:|---:|");

    let mut scalar_time = 0.0f64;
    for (name, backend) in [
        ("Scalar (1-thread)", AssignBackend::Scalar),
        ("PureRust (Rayon)", AssignBackend::PureRust),
        ("Accelerate (BLAS+Rayon)", AssignBackend::Accelerate),
    ] {
        let t0 = Instant::now();
        let mut em = EmeansConfig::builder(d, k)
            .nepochs(3.0)
            .backend(backend)
            .capture(CaptureMode::None)
            .build()
            .expect("valid config");
        let res = em.train(&x, Some(&init)).expect("train");
        let dt = t0.elapsed().as_secs_f64();
        if backend == AssignBackend::Scalar {
            scalar_time = dt;
        }
        let loss = eval_loss(&x, &res.centroids, n, d, k, Metric::Euclidean);
        println!(
            "| {name:<23} | {dt:>13.3} | {:>16.2}x | {loss:>10.4} | {:>6} |",
            scalar_time / dt,
            res.n_splits
        );
    }
}

fn run_quantized_bench(n: usize, d: usize, k: usize) {
    let x = generate_mixture(n, d, k, 0.35, 777);
    let init = sample_init(&x, n, d, k, 1234);

    println!("\n### Precision / Quantized E-means (n={n}, d={d}, k={k}, 5 epochs)\n");
    println!("| Representation | Wall Time (s) | Final Loss (on FP32 x) | Delta vs FP32 |");
    println!("|---|---:|---:|---:|");

    let t0 = Instant::now();
    let cfg = EmeansConfig::builder(d, k)
        .nepochs(5.0)
        .capture(CaptureMode::None)
        .build_config()
        .expect("valid");
    let mut em = e_means::Emeans::new(cfg.clone()).expect("valid");
    let fp32_res = em.train(&x, Some(&init)).expect("fp32");
    let fp32_time = t0.elapsed().as_secs_f64();
    let fp32_loss = eval_loss(&x, &fp32_res.centroids, n, d, k, Metric::Euclidean);
    println!("| FP32           | {fp32_time:>13.3} | {fp32_loss:>22.4} |        +0.00% |");

    let qe = QuantizedEmeans::new(cfg);

    let t1 = Instant::now();
    let (i8_res, _) = qe.train_int8(&x, Some(&init)).expect("int8");
    let i8_time = t1.elapsed().as_secs_f64();
    let i8_loss = eval_loss(&x, &i8_res.centroids, n, d, k, Metric::Euclidean);
    println!(
        "| INT8           | {i8_time:>13.3} | {i8_loss:>22.4} | {:>+12.2}% |",
        ((i8_loss - fp32_loss) / fp32_loss) * 100.0
    );

    let t2 = Instant::now();
    let (fp8_res, _) = qe.train_fp8(&x, Some(&init)).expect("fp8");
    let fp8_time = t2.elapsed().as_secs_f64();
    let fp8_loss = eval_loss(&x, &fp8_res.centroids, n, d, k, Metric::Euclidean);
    println!(
        "| FP8 E4M3       | {fp8_time:>13.3} | {fp8_loss:>22.4} | {:>+12.2}% |",
        ((fp8_loss - fp32_loss) / fp32_loss) * 100.0
    );
}

fn main() {
    println!("### e-means vs Lloyd's k-means Benchmark (Apple M3 Pro, Release Build)\n");
    println!(
        "| Workload | N | d | K | e-means | e-means Loss | Lloyd's | Lloyd Loss | Speedup | Loss Delta | Splits |"
    );
    println!("|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|");

    run_emeans_vs_lloyd(
        "Euclidean High-D (3ep)",
        50_000,
        64,
        256,
        Metric::Euclidean,
        3.0,
        25,
    );
    run_emeans_vs_lloyd(
        "Euclidean High-D (5ep)",
        50_000,
        64,
        256,
        Metric::Euclidean,
        5.0,
        25,
    );
    run_emeans_vs_lloyd(
        "Euclidean Low-D (Quality)",
        100_000,
        16,
        256,
        Metric::Euclidean,
        3.0,
        25,
    );
    run_emeans_vs_lloyd(
        "Euclidean Large (128D)",
        100_000,
        128,
        512,
        Metric::Euclidean,
        3.0,
        25,
    );
    run_emeans_vs_lloyd(
        "Spherical (64D)",
        50_000,
        64,
        256,
        Metric::Spherical,
        3.0,
        25,
    );

    run_backend_bench(50_000, 64, 256);
    run_quantized_bench(50_000, 64, 256);
    run_ivf_and_sharded_bench(50_000, 64, 1024);
}

fn run_ivf_and_sharded_bench(n: usize, d: usize, k: usize) {
    let x = generate_mixture(n, d, k, 0.35, 2027);
    let init = sample_init(&x, n, d, k, 1234);

    // Sharded (4 shards) vs Single-model Emeans
    println!("\n### ShardedEmeans (4 shards) & IVF-Flat 2-NN (n={n}, d={d}, k={k})\n");
    let cfg = EmeansConfig::builder(d, k)
        .nepochs(3.0)
        .capture(CaptureMode::None)
        .build_config()
        .expect("valid");
    let t_sh = Instant::now();
    let sh_res = ShardedEmeans::new(cfg, 4)
        .expect("sharded")
        .train(&x, Some(&init))
        .expect("train");
    let dt_sh = t_sh.elapsed().as_secs_f64();
    let loss_sh = eval_loss(&x, &sh_res.centroids, n, d, k, Metric::Euclidean);
    println!(
        "- ShardedEmeans (4 shards, 3 ep): `{dt_sh:.3}s`, Loss `{loss_sh:.4}`, Splits `{}`",
        sh_res.n_splits
    );

    // IVF-Flat (nlist=64, nprobe=4) vs Exhaustive 2-NN on 10k queries against 1024 centroids
    let nq = 10_000;
    let queries = &x[..nq * d];
    let nlist = 64;
    let coarse = sample_init(&sh_res.centroids, k, d, nlist, 7);
    let t_ivf = Instant::now();
    let mut ivf = IndexIVFFlat::new(&coarse, d, 4).expect("ivf");
    ivf.add(&sh_res.centroids).expect("add");
    let (_, ivf_ids) = ivf.search_2nn(queries, None).expect("ivf 2nn");
    let dt_ivf = t_ivf.elapsed().as_secs_f64();

    let t_exh = Instant::now();
    let mut d1 = vec![0.0f32; nq];
    let mut l1 = vec![0usize; nq];
    let mut d2 = vec![0.0f32; nq];
    let mut l2 = vec![0usize; nq];
    assign_rows(
        queries,
        &sh_res.centroids,
        d,
        k,
        Metric::Euclidean,
        AssignBackend::Accelerate,
        &mut AssignOut {
            d1: &mut d1,
            l1: &mut l1,
            d2: Some(&mut d2),
            l2: Some(&mut l2),
        },
    );
    let dt_exh = t_exh.elapsed().as_secs_f64();
    let recall_1: f64 = l1
        .iter()
        .enumerate()
        .filter(|&(i, &lbl)| ivf_ids[i * 2] == lbl as isize)
        .count() as f64
        / (nq as f64);
    println!(
        "- IVF-Flat 2-NN (nlist={nlist}, nprobe=4, {nq} queries): `{dt_ivf:.4}s` (recall@1={:.1}%, ndis={}) vs Exhaustive Accelerate `{dt_exh:.4}s`",
        recall_1 * 100.0,
        ivf.ndis
    );
}
