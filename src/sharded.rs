//! Sharded-centroid E-means (`EmeansShardedCentroids` CPU/Rayon port).
//!
//! Partitions `k` centroids across `n_shards` independent shard owners, evaluates
//! shard-local 2-NN assignments in parallel, merges global top-2 results, and
//! performs global cost-based revival across shard boundaries.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::distance::{AssignBackend, AssignOut, Metric, assign_rows};
use crate::emeans::{EmeansConfig, EmeansError, EpochStat, TrainResult};
use crate::recipe::{
    CHECK_UNUSED_EVERY, INIT_USAGE, REMOVAL_THRESHOLD, epoch_seed, resolve_recipe, scheduled_alpha,
};

struct Shard {
    centroid_begin: usize,
    local_k: usize,
    centroids: Vec<f32>,
    usage: Vec<f32>,
    removal_stats: Vec<f32>,
    axes: Vec<f32>,
    axis_scale: f64,
    pass_counts: Vec<f64>,
    pass_sums: Option<Vec<f32>>,
}

/// Multi-shard E-means trainer partitioning centroids across `n_shards` owners.
pub struct ShardedEmeans {
    cfg: EmeansConfig,
    n_shards: usize,
}

impl ShardedEmeans {
    /// Create a sharded trainer with `n_shards` centroid partitions.
    pub fn new(cfg: EmeansConfig, n_shards: usize) -> Result<Self, EmeansError> {
        cfg.validate()?;
        if n_shards == 0 || n_shards > cfg.k {
            return Err(EmeansError::InvalidState(
                "n_shards must be between 1 and k",
            ));
        }
        Ok(Self { cfg, n_shards })
    }

    /// Train on `x` (`n * d`) using `n_shards` parallel centroid shards.
    pub fn train(
        &self,
        x: &[f32],
        init_centroids: Option<&[f32]>,
    ) -> Result<TrainResult, EmeansError> {
        let d = self.cfg.d;
        let k = self.cfg.k;
        if x.is_empty() || !x.len().is_multiple_of(d) {
            return Err(EmeansError::InvalidDataLength { len: x.len(), d });
        }
        let n = x.len() / d;

        let init_buf: Vec<f32> = match init_centroids {
            Some(c) => {
                if c.len() != k * d {
                    return Err(EmeansError::InvalidInitCentroids {
                        expected: k * d,
                        got: c.len(),
                    });
                }
                c.to_vec()
            }
            None => {
                let mut rng = StdRng::seed_from_u64(self.cfg.seed);
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(&mut rng);
                let mut c = Vec::with_capacity(k * d);
                for &idx in perm.iter().take(k) {
                    c.extend_from_slice(&x[idx * d..(idx + 1) * d]);
                }
                c
            }
        };

        let mut rng = StdRng::seed_from_u64(self.cfg.seed);
        let base_k = k / self.n_shards;
        let rem_k = k % self.n_shards;
        let mut shards = Vec::with_capacity(self.n_shards);
        let mut offset = 0usize;
        for s in 0..self.n_shards {
            let local_k = base_k + usize::from(s < rem_k);
            let mut axes = vec![0.0f32; local_k * d];
            for row in axes.chunks_exact_mut(d) {
                let mut sq = 0.0f64;
                for v in row.iter_mut() {
                    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                    let u2: f64 = rng.random::<f64>();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    *v = z as f32;
                    sq += z * z;
                }
                let inv = 1.0 / sq.sqrt().max(1e-12);
                for v in row.iter_mut() {
                    *v = ((*v as f64) * inv) as f32;
                }
            }
            shards.push(Shard {
                centroid_begin: offset,
                local_k,
                centroids: init_buf[offset * d..(offset + local_k) * d].to_vec(),
                usage: vec![INIT_USAGE; local_k],
                removal_stats: vec![0.0f32; local_k * (d + 1)],
                axes,
                axis_scale: 1.0,
                pass_counts: vec![0.0f64; local_k],
                pass_sums: None,
            });
            offset += local_k;
        }

        let recipe = resolve_recipe(
            n,
            k,
            d,
            self.cfg.nepochs,
            self.cfg.batch_law,
            self.cfg.batch_size,
            self.cfg.alpha0,
        );

        let mut step = 0usize;
        let mut n_splits = 0usize;
        let mut iteration_stats = Vec::with_capacity(recipe.epoch_rows.len());
        let start_time = std::time::Instant::now();
        let num_epochs = recipe.epoch_rows.len();

        for (epoch_idx, &rows) in recipe.epoch_rows.iter().enumerate() {
            let is_last = epoch_idx + 1 == num_epochs;
            let splits_before = n_splits;
            let mut epoch_obj = 0.0f64;
            for sh in &mut shards {
                sh.pass_counts.fill(0.0);
                sh.pass_sums = if is_last {
                    Some(vec![0.0f32; sh.local_k * d])
                } else {
                    None
                };
            }

            let mut e_rng = StdRng::seed_from_u64(epoch_seed(self.cfg.seed, epoch_idx));
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut e_rng);

            for batch_start in (0..rows).step_by(recipe.batch_size) {
                let batch_end = (batch_start + recipe.batch_size).min(rows);
                let batch_idx = &order[batch_start..batch_end];
                let m = batch_idx.len();

                let mut batch_x = Vec::with_capacity(m * d);
                for &id in batch_idx {
                    batch_x.extend_from_slice(&x[id * d..(id + 1) * d]);
                }

                // Parallel 2-NN search on every shard.
                type ShardHits = (Vec<f32>, Vec<usize>, Vec<f32>, Vec<usize>);
                let shard_hits: Vec<ShardHits> = shards
                    .par_iter()
                    .map(|sh| {
                        let mut d1 = vec![0.0f32; m];
                        let mut l1 = vec![0usize; m];
                        let mut d2 = vec![f32::INFINITY; m];
                        let mut l2 = vec![0usize; m];
                        let mut out = AssignOut {
                            d1: &mut d1,
                            l1: &mut l1,
                            d2: Some(&mut d2),
                            l2: Some(&mut l2),
                        };
                        assign_rows(
                            &batch_x,
                            &sh.centroids,
                            d,
                            sh.local_k,
                            Metric::Euclidean,
                            AssignBackend::PureRust,
                            &mut out,
                        );
                        (d1, l1, d2, l2)
                    })
                    .collect();

                // Global top-2 reduction across shards.
                let mut global_d1 = vec![f32::INFINITY; m];
                let mut global_l1 = vec![0usize; m];
                let mut global_d2 = vec![f32::INFINITY; m];

                for (sh_idx, (d1, l1, d2, _)) in shard_hits.iter().enumerate() {
                    let base = shards[sh_idx].centroid_begin;
                    for r in 0..m {
                        let cand1_d = d1[r];
                        let cand1_l = base + l1[r];
                        if cand1_d < global_d1[r]
                            || (cand1_d == global_d1[r] && cand1_l < global_l1[r])
                        {
                            global_d2[r] = global_d1[r];
                            global_d1[r] = cand1_d;
                            global_l1[r] = cand1_l;
                        } else if cand1_d < global_d2[r] {
                            global_d2[r] = cand1_d;
                        }
                        let cand2_d = d2[r];
                        if cand2_d < global_d2[r] {
                            global_d2[r] = cand2_d;
                        }
                    }
                }

                for &dist in &global_d1 {
                    epoch_obj += dist as f64;
                }

                let alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps);
                let decay = 1.0 - alpha;
                let alpha_f = alpha as f32;
                let decay_f = decay as f32;

                // Update each shard independently.
                shards.par_iter_mut().for_each(|sh| {
                    let end = sh.centroid_begin + sh.local_k;
                    let mut sums = vec![0.0f32; sh.local_k * d];
                    let mut counts = vec![0.0f32; sh.local_k];
                    let mut b_rem = vec![0.0f32; sh.local_k * (d + 1)];

                    for r in 0..m {
                        let g_lbl = global_l1[r];
                        if g_lbl >= sh.centroid_begin && g_lbl < end {
                            let loc = g_lbl - sh.centroid_begin;
                            counts[loc] += 1.0;
                            let row = &batch_x[r * d..(r + 1) * d];
                            let s_row = &mut sums[loc * d..(loc + 1) * d];
                            let c_row = &sh.centroids[loc * d..(loc + 1) * d];
                            let ax_row = &sh.axes[loc * d..(loc + 1) * d];
                            let mut p = 0.0f32;
                            for i in 0..d {
                                s_row[i] += row[i];
                                p += (row[i] - c_row[i]) * ax_row[i];
                            }
                            let rem_row = &mut b_rem[loc * (d + 1)..(loc + 1) * (d + 1)];
                            for i in 0..d {
                                rem_row[i] += p * (row[i] - c_row[i]);
                            }
                            let margin = (global_d2[r] - global_d1[r]).max(0.0);
                            if margin.is_finite() {
                                rem_row[d] += margin;
                            }
                        }
                    }

                    for j in 0..sh.local_k {
                        sh.pass_counts[j] += counts[j] as f64;
                        let r_idx = j * (d + 1) + d;
                        sh.removal_stats[r_idx] =
                            decay_f * sh.removal_stats[r_idx] + alpha_f * b_rem[r_idx];
                    }
                    if let Some(ref mut ps) = sh.pass_sums {
                        for (dst, &src) in ps.iter_mut().zip(sums.iter()) {
                            *dst += src;
                        }
                    }

                    let mut scale = sh.axis_scale * decay;
                    if scale <= 2.0f64.powi(-32) {
                        let sf = scale as f32;
                        for j in 0..sh.local_k {
                            for v in &mut sh.removal_stats[j * (d + 1)..j * (d + 1) + d] {
                                *v *= sf;
                            }
                        }
                        scale = 1.0;
                    }
                    sh.axis_scale = scale;
                    let inv_scale = (alpha / scale) as f32;

                    for j in 0..sh.local_k {
                        sh.usage[j] *= decay_f;
                        if counts[j] > 0.0 {
                            let u_row = &mut sh.removal_stats[j * (d + 1)..j * (d + 1) + d];
                            let b_row = &b_rem[j * (d + 1)..j * (d + 1) + d];
                            let mut nsq = 0.0f64;
                            for i in 0..d {
                                u_row[i] += inv_scale * b_row[i];
                                let val = u_row[i] as f64;
                                nsq += val * val;
                            }
                            let norm = nsq.sqrt();
                            if norm > 0.0 {
                                let inv_n = 1.0 / norm;
                                for i in 0..d {
                                    sh.axes[j * d + i] = ((u_row[i] as f64) * inv_n) as f32;
                                }
                            }

                            let mass = sh.usage[j] + alpha_f * counts[j];
                            let w = alpha_f * counts[j] / mass;
                            let c_row = &mut sh.centroids[j * d..(j + 1) * d];
                            let s_row = &sums[j * d..(j + 1) * d];
                            for i in 0..d {
                                let mean = s_row[i] / counts[j];
                                c_row[i] += w * (mean - c_row[i]);
                            }
                            sh.usage[j] = mass;
                        }
                    }
                });

                // Global cost-based revival every 5 steps.
                if (step + 1).is_multiple_of(CHECK_UNUSED_EVERY) && k > 1 {
                    let mut all_r = Vec::with_capacity(k);
                    let mut all_usage = Vec::with_capacity(k);
                    let mut all_anorm = Vec::with_capacity(k);
                    for sh in &shards {
                        for j in 0..sh.local_k {
                            all_r.push(sh.removal_stats[j * (d + 1) + d] as f64);
                            all_usage.push(sh.usage[j] as f64);
                            let mut nsq = 0.0f64;
                            for i in 0..d {
                                let v = sh.removal_stats[j * (d + 1) + i] as f64;
                                nsq += v * v;
                            }
                            all_anorm.push(sh.axis_scale * nsq.sqrt());
                        }
                    }
                    let mut sorted_r = all_r.clone();
                    sorted_r.sort_by(f64::total_cmp);
                    let med = if sorted_r.len().is_multiple_of(2) {
                        0.5 * (sorted_r[sorted_r.len() / 2 - 1] + sorted_r[sorted_r.len() / 2])
                    } else {
                        sorted_r[sorted_r.len() / 2]
                    };

                    let mut kept: Vec<usize> = (0..k)
                        .filter(|&j| all_r[j] <= 0.0 || all_r[j] < REMOVAL_THRESHOLD * med)
                        .collect();
                    if !kept.is_empty() {
                        kept.sort_by(|&a, &b| all_r[a].total_cmp(&all_r[b]));
                        let mut is_kept = vec![false; k];
                        for &j in &kept {
                            is_kept[j] = true;
                        }
                        let mut donors: Vec<usize> = (0..k)
                            .filter(|&j| !is_kept[j] && all_usage[j] > 0.0)
                            .collect();
                        donors.sort_by(|&a, &b| all_anorm[b].total_cmp(&all_anorm[a]));
                        let ns = kept.len().min(donors.len());
                        for p in 0..ns {
                            let src_g = donors[p];
                            let tgt_g = kept[p];
                            let (src_sh, src_loc) = locate_shard(&shards, src_g);
                            let (tgt_sh, tgt_loc) = locate_shard(&shards, tgt_g);

                            let norm = all_anorm[src_g];
                            let mass = all_usage[src_g];
                            let spread = if norm > 0.0 && mass > 0.0 {
                                (2.0 * norm / (std::f64::consts::PI * mass)).sqrt()
                            } else {
                                0.0
                            };

                            let mut donor_c = vec![0.0f32; d];
                            let mut child_c = vec![0.0f32; d];
                            for i in 0..d {
                                let c = shards[src_sh].centroids[src_loc * d + i] as f64;
                                let off = spread * (shards[src_sh].axes[src_loc * d + i] as f64);
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

                            let half_u = shards[src_sh].usage[src_loc] * 0.5;
                            shards[src_sh].usage[src_loc] = half_u;
                            shards[tgt_sh].usage[tgt_loc] = half_u;
                            shards[src_sh].centroids[src_loc * d..(src_loc + 1) * d]
                                .copy_from_slice(&donor_c);
                            shards[tgt_sh].centroids[tgt_loc * d..(tgt_loc + 1) * d]
                                .copy_from_slice(&child_c);

                            let src_rem: Vec<f32> = shards[src_sh].removal_stats
                                [src_loc * (d + 1)..(src_loc + 1) * (d + 1)]
                                .iter()
                                .map(|&v| v * 0.5)
                                .collect();
                            shards[src_sh].removal_stats
                                [src_loc * (d + 1)..(src_loc + 1) * (d + 1)]
                                .copy_from_slice(&src_rem);
                            shards[tgt_sh].removal_stats
                                [tgt_loc * (d + 1)..(tgt_loc + 1) * (d + 1)]
                                .copy_from_slice(&src_rem);
                            let src_ax =
                                shards[src_sh].axes[src_loc * d..(src_loc + 1) * d].to_vec();
                            shards[tgt_sh].axes[tgt_loc * d..(tgt_loc + 1) * d]
                                .copy_from_slice(&src_ax);

                            n_splits += 1;
                        }
                    }
                }

                step += 1;
            }

            let count_sq: f64 = shards
                .iter()
                .flat_map(|sh| sh.pass_counts.iter())
                .map(|&c| c * c)
                .sum();
            iteration_stats.push(EpochStat {
                obj: epoch_obj,
                time_secs: start_time.elapsed().as_secs_f64(),
                imbalance_factor: (k as f64) * count_sq / ((rows as f64) * (rows as f64)).max(1.0),
                n_splits: n_splits - splits_before,
            });
        }

        // Gather final closing centroids from all shards.
        let mut final_centroids = Vec::with_capacity(k * d);
        for sh in &shards {
            let ps = sh
                .pass_sums
                .as_ref()
                .expect("last epoch allocated pass_sums");
            for j in 0..sh.local_k {
                let cnt = sh.pass_counts[j];
                if cnt > 0.0 {
                    let inv = 1.0 / (cnt as f32);
                    for i in 0..d {
                        final_centroids.push(ps[j * d + i] * inv);
                    }
                } else {
                    final_centroids.extend_from_slice(&sh.centroids[j * d..(j + 1) * d]);
                }
            }
        }

        let obj = iteration_stats.iter().map(|s| s.obj).collect();
        Ok(TrainResult {
            centroids: final_centroids,
            assignments: None,
            obj,
            iteration_stats,
            recipe,
            n_splits,
        })
    }
}

fn locate_shard(shards: &[Shard], global_id: usize) -> (usize, usize) {
    for (s_idx, sh) in shards.iter().enumerate() {
        if global_id >= sh.centroid_begin && global_id < sh.centroid_begin + sh.local_k {
            return (s_idx, global_id - sh.centroid_begin);
        }
    }
    unreachable!("global_id {global_id} out of bounds")
}

/// Sharded-centroid E-means using per-shard [`crate::IndexIVFFlat`] for large-`K` assignment
/// (Rust port of `sharded_centroids_ivf.py`).
pub struct ShardedEmeansIvf {
    cfg: EmeansConfig,
    n_shards: usize,
    nlist: usize,
    nprobe: usize,
}

impl ShardedEmeansIvf {
    /// Create a sharded IVF trainer with `n_shards`, `nlist` coarse lists, and `nprobe` probes.
    pub fn new(
        cfg: EmeansConfig,
        n_shards: usize,
        nlist: usize,
        nprobe: usize,
    ) -> Result<Self, EmeansError> {
        cfg.validate()?;
        if n_shards == 0 || n_shards > cfg.k {
            return Err(EmeansError::InvalidState(
                "n_shards must be between 1 and k",
            ));
        }
        if nlist == 0 || nprobe == 0 || nprobe > nlist {
            return Err(EmeansError::InvalidState(
                "nprobe must be between 1 and nlist",
            ));
        }
        Ok(Self {
            cfg,
            n_shards,
            nlist,
            nprobe,
        })
    }

    /// Train on `x` (`n * d`) using coarse IVF routing across sharded centroid owners.
    pub fn train(
        &self,
        x: &[f32],
        init_centroids: Option<&[f32]>,
    ) -> Result<TrainResult, EmeansError> {
        use crate::distance::dot_f32;
        use crate::ivf_flat::IndexIVFFlat;

        let d = self.cfg.d;
        let k = self.cfg.k;
        if x.is_empty() || !x.len().is_multiple_of(d) {
            return Err(EmeansError::InvalidDataLength { len: x.len(), d });
        }
        let n = x.len() / d;

        let init_buf: Vec<f32> = match init_centroids {
            Some(c) => {
                if c.len() != k * d {
                    return Err(EmeansError::InvalidInitCentroids {
                        expected: k * d,
                        got: c.len(),
                    });
                }
                c.to_vec()
            }
            None => {
                let mut rng = StdRng::seed_from_u64(self.cfg.seed);
                let mut perm: Vec<usize> = (0..n).collect();
                perm.shuffle(&mut rng);
                let mut c = Vec::with_capacity(k * d);
                for &idx in perm.iter().take(k) {
                    c.extend_from_slice(&x[idx * d..(idx + 1) * d]);
                }
                c
            }
        };

        // Sample coarse centroids from initial centroids (or data if k < nlist).
        let mut coarse_rng = StdRng::seed_from_u64(self.cfg.seed.wrapping_add(1));
        let mut coarse = Vec::with_capacity(self.nlist * d);
        let src_rows = if k >= self.nlist { &init_buf } else { x };
        let src_n = src_rows.len() / d;
        let mut c_perm: Vec<usize> = (0..src_n).collect();
        c_perm.shuffle(&mut coarse_rng);
        for &idx in c_perm.iter().take(self.nlist) {
            coarse.extend_from_slice(&src_rows[idx * d..(idx + 1) * d]);
        }

        let base_k = k / self.n_shards;
        let rem_k = k % self.n_shards;
        let mut shards = Vec::with_capacity(self.n_shards);
        let mut offset = 0usize;
        for s in 0..self.n_shards {
            let local_k = base_k + usize::from(s < rem_k);
            shards.push((
                offset,
                local_k,
                init_buf[offset * d..(offset + local_k) * d].to_vec(),
                vec![INIT_USAGE; local_k],
                vec![0.0f64; local_k],
                None::<Vec<f32>>,
            ));
            offset += local_k;
        }

        let recipe = resolve_recipe(
            n,
            k,
            d,
            self.cfg.nepochs,
            self.cfg.batch_law,
            self.cfg.batch_size,
            self.cfg.alpha0,
        );
        let mut step = 0usize;
        let mut iteration_stats = Vec::with_capacity(recipe.epoch_rows.len());
        let start_time = std::time::Instant::now();
        let num_epochs = recipe.epoch_rows.len();

        for (epoch_idx, &rows) in recipe.epoch_rows.iter().enumerate() {
            let is_last = epoch_idx + 1 == num_epochs;
            let mut epoch_obj = 0.0f64;
            for sh in &mut shards {
                sh.4.fill(0.0);
                sh.5 = if is_last {
                    Some(vec![0.0f32; sh.1 * d])
                } else {
                    None
                };
            }

            let mut e_rng = StdRng::seed_from_u64(epoch_seed(self.cfg.seed, epoch_idx));
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut e_rng);

            for batch_start in (0..rows).step_by(recipe.batch_size) {
                let batch_end = (batch_start + recipe.batch_size).min(rows);
                let batch_idx = &order[batch_start..batch_end];
                let m = batch_idx.len();

                let mut batch_x = Vec::with_capacity(m * d);
                let mut x_norms = Vec::with_capacity(m);
                for &id in batch_idx {
                    let row = &x[id * d..(id + 1) * d];
                    batch_x.extend_from_slice(row);
                    x_norms.push(dot_f32(row, row));
                }

                // Build per-shard IVF-Flat indexes and compute global list occupancy.
                let mut shard_ivfs: Vec<IndexIVFFlat> = shards
                    .iter()
                    .map(|sh| {
                        let mut ivf =
                            IndexIVFFlat::new(&coarse, d, self.nprobe).expect("valid ivf");
                        ivf.add(&sh.2).expect("add shard centroids");
                        ivf
                    })
                    .collect();

                let mut global_nonempty = vec![false; self.nlist];
                for ivf in &shard_ivfs {
                    for (l, &sz) in ivf.list_sizes.iter().enumerate() {
                        if sz > 0 {
                            global_nonempty[l] = true;
                        }
                    }
                }
                for ivf in &mut shard_ivfs {
                    let _ = ivf.set_routing_nonempty(&global_nonempty);
                }

                let routed = shard_ivfs[0].route(&batch_x)?;
                let shard_hits: Vec<(Vec<f32>, Vec<isize>)> = shard_ivfs
                    .par_iter_mut()
                    .map(|ivf| ivf.search_1nn(&batch_x, Some(&routed)).expect("ivf 1nn"))
                    .collect();

                let mut global_l1 = vec![0usize; m];
                let mut global_d1 = vec![f32::INFINITY; m];
                for (sh_idx, (sc, id)) in shard_hits.iter().enumerate() {
                    let base = shards[sh_idx].0;
                    for r in 0..m {
                        if id[r] >= 0 {
                            let dist = (x_norms[r] + sc[r]).max(0.0);
                            let g_id = base + (id[r] as usize);
                            if dist < global_d1[r] || (dist == global_d1[r] && g_id < global_l1[r])
                            {
                                global_d1[r] = dist;
                                global_l1[r] = g_id;
                            }
                        }
                    }
                }

                for &dist in &global_d1 {
                    if dist.is_finite() {
                        epoch_obj += dist as f64;
                    }
                }

                let alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps);
                let decay_f = (1.0 - alpha) as f32;
                let alpha_f = alpha as f32;

                shards.par_iter_mut().for_each(|sh| {
                    let (begin, local_k, centroids, usage, pass_counts, pass_sums) = sh;
                    let end = *begin + *local_k;
                    let mut sums = vec![0.0f32; *local_k * d];
                    let mut counts = vec![0.0f32; *local_k];
                    for r in 0..m {
                        let g = global_l1[r];
                        if g >= *begin && g < end {
                            let loc = g - *begin;
                            counts[loc] += 1.0;
                            let row = &batch_x[r * d..(r + 1) * d];
                            let s_row = &mut sums[loc * d..(loc + 1) * d];
                            for i in 0..d {
                                s_row[i] += row[i];
                            }
                        }
                    }
                    for j in 0..*local_k {
                        pass_counts[j] += counts[j] as f64;
                        usage[j] *= decay_f;
                        if counts[j] > 0.0 {
                            let mass = usage[j] + alpha_f * counts[j];
                            let w = alpha_f * counts[j] / mass;
                            let c_row = &mut centroids[j * d..(j + 1) * d];
                            let s_row = &sums[j * d..(j + 1) * d];
                            for i in 0..d {
                                let mean = s_row[i] / counts[j];
                                c_row[i] += w * (mean - c_row[i]);
                            }
                            usage[j] = mass;
                        }
                    }
                    if let Some(ps) = pass_sums {
                        for (dst, &src) in ps.iter_mut().zip(sums.iter()) {
                            *dst += src;
                        }
                    }
                });

                step += 1;
            }

            let count_sq: f64 = shards
                .iter()
                .flat_map(|sh| sh.4.iter())
                .map(|&c| c * c)
                .sum();
            iteration_stats.push(EpochStat {
                obj: epoch_obj,
                time_secs: start_time.elapsed().as_secs_f64(),
                imbalance_factor: (k as f64) * count_sq / ((rows as f64) * (rows as f64)).max(1.0),
                n_splits: 0,
            });
        }

        let mut final_centroids = Vec::with_capacity(k * d);
        for sh in &shards {
            let ps = sh.5.as_ref().expect("last epoch allocated pass_sums");
            for j in 0..sh.1 {
                let cnt = sh.4[j];
                if cnt > 0.0 {
                    let inv = 1.0 / (cnt as f32);
                    for i in 0..d {
                        final_centroids.push(ps[j * d + i] * inv);
                    }
                } else {
                    final_centroids.extend_from_slice(&sh.2[j * d..(j + 1) * d]);
                }
            }
        }

        let obj = iteration_stats.iter().map(|s| s.obj).collect();
        Ok(TrainResult {
            centroids: final_centroids,
            assignments: None,
            obj,
            iteration_stats,
            recipe,
            n_splits: 0,
        })
    }
}
