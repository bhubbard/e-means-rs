//! IVF-Flat index for fast approximate 1-NN and 2-NN candidate search
//! (Rust port of `ivf_flat.py` and `ivf_fine.py`).

use rayon::prelude::*;

use crate::distance::{AssignBackend, AssignOut, Metric, assign_rows, dot_f32};
use crate::emeans::EmeansError;

/// Coarse-quantized IVF-Flat index supporting 1-NN and 2-NN search with
/// non-empty list routing and exact lowest-ID tie-breaking.
#[derive(Debug, Clone)]
pub struct IndexIVFFlat {
    pub d: usize,
    pub nlist: usize,
    pub nprobe: usize,
    pub centroids: Vec<f32>,
    centroids_norms: Vec<f32>,
    routing_nonempty: Vec<bool>,
    pub list_sizes: Vec<usize>,
    xb_lims: Vec<usize>,
    pub ids: Vec<usize>,
    vectors: Vec<f32>,
    vectors_norms: Vec<f32>,
    pub ndis: usize,
    pub coarse_ndis: usize,
}

impl IndexIVFFlat {
    /// Create a new IVF-Flat index from coarse `centroids` (`nlist * d`).
    pub fn new(centroids: &[f32], d: usize, nprobe: usize) -> Result<Self, EmeansError> {
        if d == 0 {
            return Err(EmeansError::InvalidDimension);
        }
        if centroids.is_empty() || !centroids.len().is_multiple_of(d) {
            return Err(EmeansError::InvalidDataLength {
                len: centroids.len(),
                d,
            });
        }
        let nlist = centroids.len() / d;
        if nprobe == 0 || nprobe > nlist {
            return Err(EmeansError::InvalidState(
                "nprobe must be between 1 and nlist",
            ));
        }

        let centroids_norms: Vec<f32> = centroids.chunks_exact(d).map(|c| dot_f32(c, c)).collect();

        Ok(Self {
            d,
            nlist,
            nprobe,
            centroids: centroids.to_vec(),
            centroids_norms,
            routing_nonempty: vec![false; nlist],
            list_sizes: vec![0; nlist],
            xb_lims: Vec::new(),
            ids: Vec::new(),
            vectors: Vec::new(),
            vectors_norms: Vec::new(),
            ndis: 0,
            coarse_ndis: 0,
        })
    }

    /// Assign candidate vectors `x` (`n * d`) to coarse lists and build inverted lists.
    pub fn add(&mut self, x: &[f32]) -> Result<(), EmeansError> {
        if x.is_empty() || !x.len().is_multiple_of(self.d) {
            return Err(EmeansError::InvalidDataLength {
                len: x.len(),
                d: self.d,
            });
        }
        let n = x.len() / self.d;
        self.coarse_ndis += n * self.nlist;

        let mut d1 = vec![0.0f32; n];
        let mut list_nos = vec![0usize; n];
        let mut out = AssignOut {
            d1: &mut d1,
            l1: &mut list_nos,
            d2: None,
            l2: None,
        };
        assign_rows(
            x,
            &self.centroids,
            self.d,
            self.nlist,
            Metric::Euclidean,
            AssignBackend::Accelerate,
            &mut out,
        );

        let mut counts = vec![0usize; self.nlist];
        for &l in &list_nos {
            counts[l] += 1;
        }

        let mut lims = vec![0usize; self.nlist + 1];
        for l in 0..self.nlist {
            lims[l + 1] = lims[l] + counts[l];
        }

        // Stable bucket sort of candidate IDs by coarse list.
        let mut order = vec![0usize; n];
        let mut write_pos = lims[..self.nlist].to_vec();
        for (i, &l) in list_nos.iter().enumerate() {
            order[write_pos[l]] = i;
            write_pos[l] += 1;
        }

        let mut vectors = vec![0.0f32; n * self.d];
        let mut vectors_norms = vec![0.0f32; n];
        for (slot, &orig_id) in order.iter().enumerate() {
            let src = &x[orig_id * self.d..(orig_id + 1) * self.d];
            vectors[slot * self.d..(slot + 1) * self.d].copy_from_slice(src);
            vectors_norms[slot] = dot_f32(src, src);
        }

        self.list_sizes = counts.clone();
        self.routing_nonempty = counts.iter().map(|&c| c > 0).collect();
        self.xb_lims = lims;
        self.ids = order;
        self.vectors = vectors;
        self.vectors_norms = vectors_norms;
        Ok(())
    }

    /// Override which lists are routable (e.g., global occupancy across shards).
    pub fn set_routing_nonempty(&mut self, nonempty: &[bool]) -> Result<(), EmeansError> {
        if nonempty.len() != self.nlist {
            return Err(EmeansError::InvalidState(
                "nonempty length must equal nlist",
            ));
        }
        self.routing_nonempty = nonempty.to_vec();
        Ok(())
    }

    /// Route each query row in `x` (`nq * d`) to its `nprobe` nearest non-empty lists.
    pub fn route(&mut self, x: &[f32]) -> Result<Vec<usize>, EmeansError> {
        if self.xb_lims.is_empty() && !self.routing_nonempty.iter().any(|&b| b) {
            return Err(EmeansError::InvalidState(
                "add candidates or set routing occupancy before route",
            ));
        }
        if x.is_empty() || !x.len().is_multiple_of(self.d) {
            return Err(EmeansError::InvalidDataLength {
                len: x.len(),
                d: self.d,
            });
        }
        let nq = x.len() / self.d;
        self.coarse_ndis += nq * self.nlist;

        let d = self.d;
        let nlist = self.nlist;
        let nprobe = self.nprobe;
        let centroids = &self.centroids;
        let c_norms = &self.centroids_norms;
        let nonempty = &self.routing_nonempty;

        let mut routed = vec![0usize; nq * nprobe];
        routed
            .par_chunks_mut(nprobe)
            .enumerate()
            .for_each(|(q_idx, dst)| {
                let q = &x[q_idx * d..(q_idx + 1) * d];
                let mut scores: Vec<(f32, usize)> = (0..nlist)
                    .map(|l| {
                        let sc = if nonempty[l] {
                            c_norms[l] - 2.0 * dot_f32(q, &centroids[l * d..(l + 1) * d])
                        } else {
                            f32::INFINITY
                        };
                        (sc, l)
                    })
                    .collect();
                scores.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                for (p, &(_, l)) in scores.iter().take(nprobe).enumerate() {
                    dst[p] = l;
                }
            });

        Ok(routed)
    }

    /// Search for the 2 nearest distinct candidates in the probed lists for each query.
    ///
    /// Returns `(scores, ids)` each of length `nq * 2`. Scores omit `||q||^2`
    /// (`score = ||c||^2 - 2 <q, c>`). Missing entries have `f32::INFINITY` and `-1`.
    pub fn search_2nn(
        &mut self,
        x: &[f32],
        coarse_idx: Option<&[usize]>,
    ) -> Result<(Vec<f32>, Vec<isize>), EmeansError> {
        if self.xb_lims.is_empty() {
            return Err(EmeansError::InvalidState("add candidates before search"));
        }
        if x.is_empty() || !x.len().is_multiple_of(self.d) {
            return Err(EmeansError::InvalidDataLength {
                len: x.len(),
                d: self.d,
            });
        }
        let nq = x.len() / self.d;
        let owned_idx;
        let idx = match coarse_idx {
            Some(c) => {
                if c.len() != nq * self.nprobe {
                    return Err(EmeansError::InvalidState(
                        "coarse_idx must have length nq * nprobe",
                    ));
                }
                c
            }
            None => {
                owned_idx = self.route(x)?;
                &owned_idx
            }
        };

        for &l in idx {
            self.ndis += self.list_sizes[l];
        }

        let d = self.d;
        let nprobe = self.nprobe;
        let lims = &self.xb_lims;
        let ids = &self.ids;
        let vecs = &self.vectors;
        let v_norms = &self.vectors_norms;

        let mut out_scores = vec![f32::INFINITY; nq * 2];
        let mut out_ids = vec![-1isize; nq * 2];

        out_scores
            .par_chunks_mut(2)
            .zip(out_ids.par_chunks_mut(2))
            .enumerate()
            .for_each(|(q_idx, (sc_slot, id_slot))| {
                let q = &x[q_idx * d..(q_idx + 1) * d];
                let probes = &idx[q_idx * nprobe..(q_idx + 1) * nprobe];

                let mut best1_sc = f32::INFINITY;
                let mut best1_id = isize::MAX;
                let mut best2_sc = f32::INFINITY;
                let mut best2_id = isize::MAX;

                for &lst in probes {
                    let b0 = lims[lst];
                    let b1 = lims[lst + 1];
                    for pos in b0..b1 {
                        let cand_id = ids[pos] as isize;
                        let cand_vec = &vecs[pos * d..(pos + 1) * d];
                        let sc = v_norms[pos] - 2.0 * dot_f32(q, cand_vec);
                        if !sc.is_finite() {
                            continue;
                        }
                        if cand_id == best1_id {
                            if sc < best1_sc {
                                best1_sc = sc;
                            }
                            continue;
                        }
                        if (sc < best1_sc) || (sc == best1_sc && cand_id < best1_id) {
                            best2_sc = best1_sc;
                            best2_id = best1_id;
                            best1_sc = sc;
                            best1_id = cand_id;
                        } else if (sc < best2_sc) || (sc == best2_sc && cand_id < best2_id) {
                            best2_sc = sc;
                            best2_id = cand_id;
                        }
                    }
                }

                if best1_id != isize::MAX {
                    sc_slot[0] = best1_sc;
                    id_slot[0] = best1_id;
                }
                if best2_id != isize::MAX {
                    sc_slot[1] = best2_sc;
                    id_slot[1] = best2_id;
                }
            });

        Ok((out_scores, out_ids))
    }

    /// Search for the 1 nearest candidate in the probed lists for each query.
    pub fn search_1nn(
        &mut self,
        x: &[f32],
        coarse_idx: Option<&[usize]>,
    ) -> Result<(Vec<f32>, Vec<isize>), EmeansError> {
        let (scores2, ids2) = self.search_2nn(x, coarse_idx)?;
        let nq = x.len() / self.d;
        let mut s1 = Vec::with_capacity(nq);
        let mut i1 = Vec::with_capacity(nq);
        for q in 0..nq {
            s1.push(scores2[q * 2]);
            i1.push(ids2[q * 2]);
        }
        Ok((s1, i1))
    }
}
