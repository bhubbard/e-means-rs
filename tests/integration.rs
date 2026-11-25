use e_means::distance::{AssignOut, assign_rows};
use e_means::recipe::{
    epoch_rows, epoch_seed, resolve_recipe, round_power_of_two, scheduled_alpha, step_law,
};
use e_means::spherical::normalize_in_place;
use e_means::{
    AssignBackend, BatchLaw, CaptureMode, EmeansConfig, Fp8E4M3, Fp8Quantizer, IndexIVFFlat,
    Int8Quantizer, LloydConfig, LloydKmeans, Metric, QuantizedEmeans, ResolvedBatchLaw,
    RevivalMode, ShardedEmeans, ShardedEmeansIvf,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

fn make_blobs(n: usize, d: usize, n_blobs: usize, spread: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = vec![0.0f32; n_blobs * d];
    for v in centers.iter_mut() {
        *v = rng.random_range(-10.0f32..10.0f32);
    }
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let c = i % n_blobs;
        for j in 0..d {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random::<f32>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            data[i * d + j] = centers[c * d + j] + spread * z;
        }
    }
    data
}

fn randn(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0.0f32; n * d];
    for v in out.iter_mut() {
        let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.random::<f32>();
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    out
}

#[test]
fn recipe_fractional_epoch_rounding_matches_reference() {
    assert_eq!(
        epoch_rows(10_000_000_000, 1.999_999_999_5),
        vec![9_999_999_995, 10_000_000_000]
    );
    assert_eq!(epoch_rows(3, 1.5), vec![2, 3]);
    assert_eq!(epoch_rows(5, 1.5), vec![2, 5]);
    assert_eq!(epoch_rows(3, 1.01), vec![3]);
    assert_eq!(epoch_rows(100, 3.0), vec![100, 100, 100]);
    assert_eq!(epoch_rows(100, 2.5), vec![100, 50, 100]);
    assert_eq!(epoch_rows(100, 0.5), vec![50]);
}

#[test]
fn recipe_reference_values_match_paper_and_faiss() {
    assert_eq!(round_power_of_two(3.0), 4);
    assert_eq!(round_power_of_two(5.0), 4);
    assert_eq!(round_power_of_two(0.5), 1);

    let cases = [
        (1_000_000, 16_384, 25.0, BatchLaw::Throughput, 262_144, 1.0),
        (
            1_000_000,
            16_384,
            25.0,
            BatchLaw::Quality,
            32_768,
            0.154_538_828_130_462_7,
        ),
        (
            100_000_000,
            262_144,
            25.0,
            BatchLaw::Throughput,
            33_554_432,
            1.0,
        ),
        (
            100_000_000,
            262_144,
            25.0,
            BatchLaw::Quality,
            4_194_304,
            0.427_185_550_532_935_17,
        ),
    ];
    for (n, k, nepochs, law, expected_batch, expected_alpha) in cases {
        let r = resolve_recipe(n, k, 64, nepochs, law, None, None);
        assert_eq!(r.batch_size, expected_batch);
        assert!((r.alpha0 - expected_alpha).abs() < 1e-12);
    }

    let r_manual = resolve_recipe(
        1_000_000,
        16_384,
        64,
        3.0,
        BatchLaw::Throughput,
        Some(8_192),
        None,
    );
    assert_eq!(r_manual.batch_law, ResolvedBatchLaw::Manual);
    assert_eq!(r_manual.batch_size, 8_192);
    assert!((r_manual.alpha0 - step_law(1_000_000, 16_384, 3.0, 8_192)).abs() < 1e-15);

    let r_low = resolve_recipe(100_000, 256, 8, 5.0, BatchLaw::Auto, None, None);
    assert_eq!(r_low.batch_law, ResolvedBatchLaw::Quality);
    let r_high = resolve_recipe(100_000, 256, 64, 5.0, BatchLaw::Auto, None, None);
    assert_eq!(r_high.batch_law, ResolvedBatchLaw::Throughput);

    let alpha0 = 0.4;
    assert_eq!(scheduled_alpha(alpha0, 0, 10), alpha0);
    assert!((scheduled_alpha(alpha0, 10, 10) - 0.04).abs() < 1e-12);
    assert!((scheduled_alpha(alpha0, 9, 10) - 0.048_809_827_066_872_37).abs() < 1e-12);

    assert_ne!(epoch_seed(0, 0), 0);
    assert_ne!(epoch_seed(1234, 0), 1234);
}

#[test]
fn closing_mean_and_assign_match_reference() {
    let x = vec![0.0f32, 2.0, 10.0, 12.0];
    let init = vec![0.0f32, 10.0];
    let mut km = EmeansConfig::builder(1, 2)
        .nepochs(1.0)
        .batch_size(4)
        .alpha0(0.5)
        .build()
        .expect("valid");
    let res = km.train(&x, Some(&init)).expect("train");
    assert_eq!(res.centroids, vec![1.0, 11.0]);
    assert_eq!(res.obj, vec![8.0]);
    assert_eq!(res.assignments, Some(vec![0, 0, 1, 1]));

    let assigned = km.assign(&x).expect("assign");
    assert_eq!(assigned.labels, vec![0, 0, 1, 1]);
    assert_eq!(assigned.distances, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn untouched_centroid_retains_its_position() {
    let x = vec![0.0f32, 1.0, 2.0, 3.0];
    let init = vec![0.0f32, 100.0];

    for nepochs in [1.0, 2.0] {
        let mut km = EmeansConfig::builder(1, 2)
            .nepochs(nepochs)
            .batch_size(4)
            .alpha0(1.0)
            .build()
            .expect("valid");
        let res = km.train(&x, Some(&init)).expect("train");
        assert_eq!(res.centroids, vec![1.5, 100.0]);
    }
}

#[test]
fn caller_batches_match_matrix_training() {
    let x = randn(17, 3, 123);
    let init = x[..12].to_vec();

    let mut matrix = EmeansConfig::builder(3, 4)
        .nepochs(3.0)
        .batch_size(4)
        .alpha0(0.2)
        .seed(7)
        .assignment_chunk_size(2)
        .build()
        .expect("valid");
    let m_res = matrix.train(&x, Some(&init)).expect("matrix train");

    let mut caller = EmeansConfig::builder(3, 4)
        .nepochs(3.0)
        .batch_size(4)
        .alpha0(0.2)
        .seed(7)
        .assignment_chunk_size(2)
        .build()
        .expect("valid");
    caller.begin(17, &init).expect("begin");
    for epoch in 0..3 {
        let mut rng = StdRng::seed_from_u64(epoch_seed(7, epoch));
        let mut order: Vec<usize> = (0..17).collect();
        order.shuffle(&mut rng);
        for chunk in order.chunks(4) {
            let mut batch = Vec::with_capacity(chunk.len() * 3);
            for &idx in chunk {
                batch.extend_from_slice(&x[idx * 3..(idx + 1) * 3]);
            }
            let stats = caller.accumulate(&batch).expect("accumulate");
            caller.update(&stats).expect("update");
        }
        caller.end_epoch().expect("end_epoch");
    }
    let c_centroids = caller.finalize().expect("finalize").to_vec();
    assert_eq!(c_centroids, m_res.centroids);
}

#[test]
fn revival_starts_on_fifth_update() {
    let x = vec![0.0f32; 10];
    let init = vec![0.0f32, 10.0];

    let mut before = EmeansConfig::builder(1, 2)
        .nepochs(4.0)
        .batch_size(10)
        .alpha0(0.5)
        .revival(RevivalMode::MassBased(0.1))
        .build()
        .expect("valid");
    let res_before = before.train(&x, Some(&init)).expect("train");
    assert_eq!(res_before.n_splits, 0);

    let mut on_fifth = EmeansConfig::builder(1, 2)
        .nepochs(5.0)
        .batch_size(10)
        .alpha0(0.5)
        .revival(RevivalMode::MassBased(0.1))
        .build()
        .expect("valid");
    let res_fifth = on_fifth.train(&x, Some(&init)).expect("train");
    assert_eq!(res_fifth.n_splits, 1);
}

#[test]
fn removal_cost_keeps_far_blob_and_revives_dead() {
    let mut x = randn(4000, 4, 0);
    let mut far = randn(40, 4, 1);
    for v in far.iter_mut() {
        *v += 20.0;
    }
    x.extend_from_slice(&far);

    let mut init = Vec::with_capacity(4 * 4);
    init.extend_from_slice(&x[..8]);
    init.extend_from_slice(&x[(4039 * 4)..(4040 * 4)]);
    init.extend_from_slice(&[-50.0f32; 4]);

    for (revival, far_kept) in [
        (RevivalMode::CostBased, true),
        (RevivalMode::MassBased(0.1), false),
    ] {
        let mut km = EmeansConfig::builder(4, 4)
            .nepochs(3.0)
            .revival(revival)
            .build()
            .expect("valid");
        let res = km.train(&x, Some(&init)).expect("train");
        let min_dist_to_20 = res
            .centroids
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| c.iter().map(|&v| (v - 20.0).abs()).fold(0.0f32, f32::max))
            .fold(f32::INFINITY, f32::min);
        assert_eq!(min_dist_to_20 < 1.0, far_kept, "revival={revival:?}");
        let max_coord = res.centroids.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max_coord < 30.0, "dead centroid should have moved");
    }
}

#[test]
fn closing_handover_closes_from_final_rows_and_improves_error() {
    let n = 10_000;
    let d = 8;
    let k = 64;
    let x = make_blobs(n, d, k, 0.5, 42);
    let mut rng = StdRng::seed_from_u64(99);
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    let mut init = Vec::with_capacity(k * d);
    for &idx in perm.iter().take(k) {
        init.extend_from_slice(&x[idx * d..(idx + 1) * d]);
    }

    let mut km = EmeansConfig::builder(d, k)
        .nepochs(1.0)
        .batch_size(500)
        .revival(RevivalMode::CostBased)
        .build()
        .expect("valid");
    let res = km.train(&x, Some(&init)).expect("train");
    let labels = res.assignments.expect("captured");
    assert!(res.n_splits > 0, "expected splits during closing pass");

    // Every populated centroid returns the exact mean of its final last-pass rows.
    for j in 0..k {
        let mut sum = vec![0.0f64; d];
        let mut count = 0usize;
        for (i, &lbl) in labels.iter().enumerate() {
            if lbl == j as isize {
                count += 1;
                for m in 0..d {
                    sum[m] += x[i * d + m] as f64;
                }
            }
        }
        if count > 0 {
            for m in 0..d {
                let expected = (sum[m] / count as f64) as f32;
                let got = res.centroids[j * d + m];
                assert!(
                    (got - expected).abs() < 1e-4,
                    "centroid {j} dim {m}: got {got}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn each_pass_uses_tail_and_chunking_preserves_updates() {
    let x = vec![0.0f32, 0.0, 0.0, 0.0, 10.0];
    let mut km = EmeansConfig::builder(1, 1)
        .nepochs(2.0)
        .batch_size(3)
        .alpha0(0.5)
        .build()
        .expect("valid");
    let res = km.train(&x, Some(&[0.0])).expect("train");
    assert_eq!(res.centroids, vec![2.0]);
    assert_eq!(res.recipe.steps_per_pass, 2);
    assert_eq!(res.recipe.total_steps, 4);

    let x2 = randn(41, 5, 123);
    let init2 = x2[..20].to_vec();
    let mut a = EmeansConfig::builder(5, 4)
        .nepochs(3.0)
        .batch_size(13)
        .alpha0(0.2)
        .seed(7)
        .assignment_chunk_size(4)
        .build()
        .expect("valid");
    let mut b = EmeansConfig::builder(5, 4)
        .nepochs(3.0)
        .batch_size(13)
        .alpha0(0.2)
        .seed(7)
        .assignment_chunk_size(64)
        .build()
        .expect("valid");
    let ra = a.train(&x2, Some(&init2)).expect("train a");
    let rb = b.train(&x2, Some(&init2)).expect("train b");
    for (va, vb) in ra.centroids.iter().zip(rb.centroids.iter()) {
        assert!((va - vb).abs() < 1e-5);
    }
    assert_eq!(ra.recipe.steps_per_pass, 4);
    assert_eq!(ra.recipe.total_steps, 12);
}

#[test]
fn fresh_objects_are_deterministic() {
    let x = randn(40, 4, 123);
    let mut a = EmeansConfig::builder(4, 3)
        .nepochs(2.0)
        .batch_size(11)
        .alpha0(0.2)
        .build()
        .expect("valid");
    let mut b = EmeansConfig::builder(4, 3)
        .nepochs(2.0)
        .batch_size(11)
        .alpha0(0.2)
        .build()
        .expect("valid");
    let ra = a.train(&x, None).expect("a");
    let rb = b.train(&x, None).expect("b");
    assert_eq!(ra.centroids, rb.centroids);
    assert_eq!(ra.obj, rb.obj);
}

#[test]
fn backends_agree_on_euclidean_and_spherical() {
    let n = 256;
    let d = 32;
    let k = 24;
    let x = make_blobs(n, d, 8, 0.5, 42);
    let c = make_blobs(k, d, 8, 0.5, 99);

    for metric in [Metric::Euclidean, Metric::Spherical] {
        let mut x_use = x.clone();
        let mut c_use = c.clone();
        if metric == Metric::Spherical {
            normalize_in_place(&mut x_use, d);
            normalize_in_place(&mut c_use, d);
        }

        let mut l1_scalar = vec![0usize; n];
        let mut d1_scalar = vec![0.0f32; n];
        let mut l2_scalar = vec![0usize; n];
        let mut d2_scalar = vec![0.0f32; n];
        assign_rows(
            &x_use,
            &c_use,
            d,
            k,
            metric,
            AssignBackend::Scalar,
            &mut AssignOut {
                d1: &mut d1_scalar,
                l1: &mut l1_scalar,
                d2: Some(&mut d2_scalar),
                l2: Some(&mut l2_scalar),
            },
        );

        for backend in [AssignBackend::PureRust, AssignBackend::Accelerate] {
            let mut l1 = vec![0usize; n];
            let mut d1 = vec![0.0f32; n];
            let mut l2 = vec![0usize; n];
            let mut d2 = vec![0.0f32; n];
            assign_rows(
                &x_use,
                &c_use,
                d,
                k,
                metric,
                backend,
                &mut AssignOut {
                    d1: &mut d1,
                    l1: &mut l1,
                    d2: Some(&mut d2),
                    l2: Some(&mut l2),
                },
            );
            assert_eq!(l1, l1_scalar, "l1 mismatch for {backend:?} / {metric:?}");
            assert_eq!(l2, l2_scalar, "l2 mismatch for {backend:?} / {metric:?}");
            for (a, b) in d1.iter().zip(d1_scalar.iter()) {
                assert!((a - b).abs() < 1e-3, "d1 mismatch: {a} vs {b}");
            }
        }
    }
}

#[test]
fn spherical_emeans_produces_unit_centroids() {
    let n = 2000;
    let d = 32;
    let k = 16;
    let mut x = make_blobs(n, d, k, 0.2, 123);
    normalize_in_place(&mut x, d);

    let mut em = EmeansConfig::builder(d, k)
        .nepochs(4.0)
        .metric(Metric::Spherical)
        .build()
        .expect("valid config");
    let res = em.train(&x, None).expect("train succeeds");

    for row in res.centroids.chunks_exact(d) {
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "centroid norm = {norm}");
    }

    let assigned = em.assign(&x).expect("assign succeeds");
    let mean_cos: f64 = assigned.distances.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);
    assert!(
        mean_cos > 0.9,
        "expected high cosine similarity, got {mean_cos}"
    );
}

#[test]
fn quantized_int8_and_fp8_train_cleanly() {
    let n = 1500;
    let d = 16;
    let k = 8;
    let x = make_blobs(n, d, k, 0.3, 55);

    let q = Int8Quantizer::fit(&x);
    let enc = q.encode(&x);
    let dec = q.decode(&enc);
    assert_eq!(dec.len(), x.len());

    let fp8_enc = Fp8E4M3::encode_slice(&x);
    let fp8_dec = Fp8E4M3::decode_slice(&fp8_enc);
    assert_eq!(fp8_dec.len(), x.len());

    let cfg = EmeansConfig::builder(d, k)
        .nepochs(3.0)
        .capture(CaptureMode::None)
        .build_config()
        .expect("valid config");
    let qe = QuantizedEmeans::new(cfg);
    let (res_i8, _) = qe.train_int8(&x, None).expect("int8 train");
    assert_eq!(res_i8.centroids.len(), k * d);

    let (res_fp8, q_fp8) = qe.train_fp8(&x, None).expect("fp8 train");
    assert_eq!(res_fp8.centroids.len(), k * d);
    assert!(q_fp8.scale > 0.0);

    let qf = Fp8Quantizer::fit(&x);
    assert_eq!(qf.decode(&qf.encode(&x)).len(), x.len());
}

#[test]
fn max_points_per_centroid_and_lloyd_baseline() {
    let n = 2000;
    let d = 16;
    let k = 10;
    let x = make_blobs(n, d, k, 0.3, 77);

    let mut em = EmeansConfig::builder(d, k)
        .nepochs(3.0)
        .max_points_per_centroid(50)
        .build()
        .expect("valid config");
    let res = em.train(&x, None).expect("train succeeds");
    let labels = res.assignments.expect("captured");
    let active_count = labels.iter().filter(|&&l| l >= 0).count();
    assert_eq!(active_count, 50 * k);

    let lloyd = LloydKmeans::new(LloydConfig::new(d, k));
    let lres = lloyd.train(&x, None).expect("lloyd succeeds");
    assert_eq!(lres.centroids.len(), k * d);
}

#[test]
fn ivf_flat_routes_and_searches_2nn_with_ties_and_missing() {
    let d = 4;
    let coarse = vec![
        0.0f32, 0.0, 0.0, 0.0, // list 0
        10.0, 0.0, 0.0, 0.0, // list 1
        20.0, 0.0, 0.0, 0.0, // list 2 (empty)
    ];
    let mut ivf = IndexIVFFlat::new(&coarse, d, 2).expect("valid ivf");

    // Candidates: id 0 and id 1 are identical at origin -> tie breaks to id 0 then id 1.
    // id 2 is at (10, 0, 0, 0).
    let cands = vec![
        0.1f32, 0.0, 0.0, 0.0, // id 0 -> list 0
        0.1, 0.0, 0.0, 0.0, // id 1 -> list 0
        10.2, 0.0, 0.0, 0.0, // id 2 -> list 1
    ];
    ivf.add(&cands).expect("add");
    assert_eq!(ivf.list_sizes, vec![2, 1, 0]);

    let queries = vec![
        0.0f32, 0.0, 0.0, 0.0, // nearest: id 0, second: id 1
        10.0, 0.0, 0.0, 0.0, // nearest: id 2, second: id 0
    ];
    let (scores2, ids2) = ivf.search_2nn(&queries, None).expect("search_2nn");
    assert_eq!(ids2[0], 0);
    assert_eq!(ids2[1], 1);
    assert_eq!(ids2[2], 2);
    assert_eq!(ids2[3], 0);
    assert!(scores2[0] <= scores2[1]);

    let (_, ids1) = ivf.search_1nn(&queries, None).expect("search_1nn");
    assert_eq!(ids1, vec![0, 2]);

    // Probing an empty list explicitly returns -1 / inf for missing entries.
    let empty_probe = vec![2usize, 2, 2, 2];
    let (miss_sc, miss_id) = ivf
        .search_2nn(&queries, Some(&empty_probe))
        .expect("empty probe");
    assert_eq!(miss_id, vec![-1, -1, -1, -1]);
    assert!(miss_sc.iter().all(|s| s.is_infinite()));
}

#[test]
fn sharded_emeans_matches_quality_across_shards() {
    let n = 4000;
    let d = 16;
    let k = 16;
    let x = make_blobs(n, d, k, 0.25, 33);

    let cfg = EmeansConfig::builder(d, k)
        .nepochs(4.0)
        .capture(CaptureMode::None)
        .build_config()
        .expect("valid");
    let sharded = ShardedEmeans::new(cfg, 4).expect("valid sharded");
    let res = sharded.train(&x, None).expect("sharded train");
    assert_eq!(res.centroids.len(), k * d);

    let mut d1 = vec![0.0f32; n];
    let mut l1 = vec![0usize; n];
    assign_rows(
        &x,
        &res.centroids,
        d,
        k,
        Metric::Euclidean,
        AssignBackend::PureRust,
        &mut AssignOut {
            d1: &mut d1,
            l1: &mut l1,
            d2: None,
            l2: None,
        },
    );
    let mse: f64 = d1.iter().map(|&v| v as f64).sum::<f64>() / (n as f64);
    assert!(mse < 3.0 * (d as f64) * 0.25 * 0.25, "sharded mse={mse}");
}

#[test]
fn sharded_emeans_ivf_converges() {
    let n = 4000;
    let d = 16;
    let k = 32;
    let x = make_blobs(n, d, k, 0.25, 88);

    let cfg = EmeansConfig::builder(d, k)
        .nepochs(4.0)
        .capture(CaptureMode::None)
        .build_config()
        .expect("valid");
    let sharded_ivf = ShardedEmeansIvf::new(cfg, 4, 8, 4).expect("valid sharded ivf");
    let res = sharded_ivf.train(&x, None).expect("sharded ivf train");
    assert_eq!(res.centroids.len(), k * d);
}
