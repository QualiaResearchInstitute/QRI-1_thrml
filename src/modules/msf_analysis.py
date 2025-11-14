"""
MSF (Master Stability Function) analysis for cluster synchronization.

Implements fast, torch-native utilities to:
- Sanitize coupling matrices for graph/Laplacian operations
- Build cluster indicator matrices and quotient matrices (Eq. 9-style)
- Compute intra-cluster Laplacian spectra (transverse modes)
- Estimate discrete-time transverse Lyapunov exponents (surrogate)
- Predict regimes: cluster_sync | chimera | asynchronous
- Estimate critical coupling thresholds per cluster

Notes:
- This module uses a phase-only linearization surrogate for discrete-time
  Kuramoto-Sakaguchi-type updates around synchronization:
    η_{t+1} ≈ [I − dt * σ * cos(α) * L_k] η_t
  where L_k is the intra-cluster graph Laplacian. For a mode with eigenvalue λ:
    g(λ) = |1 − dt * σ * cos(α) * λ|
    Λ(λ) ≈ log g(λ)
  Stability if Λ(λ) < 0 for all transverse modes (i.e., all nonzero Laplacian eigenvalues).

- For clusters of size 0 or 1, there are no transverse degrees of freedom;
  we set λ_max = 0 and treat them as trivially stable in this approximation.

- Masked entries:
  Upstream code may use -inf to mask edges. We map -inf, +inf, and NaN to 0,
  relu to enforce nonnegativity, and symmetrize to obtain a valid weight matrix
  for Laplacian construction.

- Critical coupling thresholds (per cluster):
    σ_crit,k ≈ 2 / (dt * cos(α) * λ_max,k)   when cos(α) > 0 and λ_max,k > 0
  If cos(α) ≤ 0 or λ_max,k = 0, we return +inf for σ_crit,k in this surrogate.

Author: QRI Resonance Transformer MSF integration
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import math
import torch


def _to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.tensor(x, device=device, dtype=dtype)


class MSFAnalyzer:
    """
    Master Stability Function analyzer for cluster synchronization.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    # -----------------------------
    # Sanitization and helpers
    # -----------------------------
    def sanitize_weights(
        self,
        K: torch.Tensor,
        nonnegative: bool = True,
        symmetrize: bool = True,
        zero_out_diag: bool = False,
    ) -> torch.Tensor:
        """
        Sanitize coupling matrix for Laplacian computations.

        - Replace NaN/±Inf with 0
        - Optionally relu to enforce nonnegative weights
        - Optionally symmetrize: 0.5 * (W + W^T)
        - Optionally zero out diagonal

        Args:
            K: [B, N, N]
        Returns:
            W: [B, N, N] sanitized
        """
        assert K.dim() == 3, f"sanitize_weights: expected [B,N,N], got {list(K.shape)}"
        W = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        if nonnegative:
            W = torch.relu(W)
        if symmetrize:
            W = 0.5 * (W + W.transpose(-1, -2))
        if zero_out_diag:
            B, N, _ = W.shape
            eye = torch.eye(N, device=W.device, dtype=W.dtype).unsqueeze(0).expand(B, -1, -1)
            W = W * (1.0 - eye)
        return W

    def cluster_masks(self, assignments: torch.Tensor, n_clusters: Optional[int] = None) -> List[torch.Tensor]:
        """
        Build boolean masks per cluster for a single batch item.

        Args:
            assignments: [N] int64 cluster ids
            n_clusters: optional number of clusters; defaults to max(assignments)+1
        Returns:
            masks: list of [N] boolean masks (length = n_clusters)
        """
        assert assignments.dim() == 1, "cluster_masks expects 1D assignments per batch"
        if n_clusters is None:
            n_clusters = int(assignments.max().item()) + 1 if assignments.numel() > 0 else 0
        masks = []
        for c in range(n_clusters):
            masks.append(assignments == c)
        return masks

    def build_indicator(self, assignments: torch.Tensor, n_clusters: Optional[int] = None) -> torch.Tensor:
        """
        Build indicator matrices P from assignments for an entire batch.

        Args:
            assignments: [B, N] int64
            n_clusters: optional global number of clusters; if None, uses max across batch + 1
        Returns:
            P: [B, N, K] binary indicator
        """
        assert assignments.dim() == 2, f"build_indicator: expected [B,N], got {list(assignments.shape)}"
        B, N = assignments.shape
        if n_clusters is None:
            n_clusters = int(assignments.max().item()) + 1 if assignments.numel() > 0 else 0
        device = assignments.device
        dtype = torch.float32
        P = torch.zeros((B, N, n_clusters), device=device, dtype=dtype)
        # scatter 1s
        # safeguard empty case
        if n_clusters == 0 or N == 0:
            return P
        idx_b = torch.arange(B, device=device).unsqueeze(-1).expand(B, N)  # [B,N]
        idx_n = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # [B,N]
        # Clamp out-of-range cluster ids
        a = torch.clamp(assignments, 0, n_clusters - 1)
        P[idx_b.reshape(-1), idx_n.reshape(-1), a.reshape(-1)] = 1.0
        return P

    # -----------------------------
    # Quotient matrix (Eq. 9 style)
    # -----------------------------
    def compute_quotient_matrix(
        self,
        K: torch.Tensor,
        assignments: torch.Tensor,
        n_clusters: Optional[int] = None,
        sanitize: bool = True,
    ) -> torch.Tensor:
        """
        Compute quotient matrix Q at cluster-level:
            Q = (P^T P)^(-1) P^T W P
        where W is sanitized K.

        Args:
            K: [B, N, N]
            assignments: [B, N] long
            n_clusters: optional K (clusters)
            sanitize: whether to sanitize K -> W

        Returns:
            Q: [B, K, K]
        """
        assert K.dim() == 3 and assignments.dim() == 2
        B, N, _ = K.shape
        P = self.build_indicator(assignments, n_clusters=n_clusters)  # [B, N, K]
        _, _, Kc = P.shape
        if sanitize:
            W = self.sanitize_weights(K, nonnegative=True, symmetrize=True, zero_out_diag=False)
        else:
            W = K

        # Compute P^T P and its inverse (with epsilon)
        Pt = P.transpose(-1, -2)                     # [B, K, N]
        PtP = Pt @ P                                  # [B, K, K]
        eye_k = torch.eye(Kc, device=W.device, dtype=W.dtype).unsqueeze(0).expand(B, -1, -1)
        PtP_safe = PtP + self.eps * eye_k
        try:
            PtP_inv = torch.linalg.inv(PtP_safe)     # [B, K, K]
        except Exception:
            # fallback to cpu if needed
            PtP_inv = torch.linalg.inv(PtP_safe.to("cpu")).to(PtP_safe.device, dtype=PtP_safe.dtype)

        # Compute Q = (P^T P)^(-1) P^T W P
        Q = PtP_inv @ (Pt @ (W @ P))                 # [B, K, K]
        # Symmetrize for numerical stability
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        return Q

    # -----------------------------
    # Cluster Laplacian spectra
    # -----------------------------
    def cluster_laplacian_eigs(
        self,
        K: torch.Tensor,
        assignments: torch.Tensor,
        return_all: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Compute intra-cluster Laplacian eigenvalues per batch and cluster.

        Args:
            K: [B, N, N] (will be sanitized)
            assignments: [B, N] long

        Returns:
            spectra: list length B, each a dict:
                {
                    'per_cluster': {
                        c: { 'lambdas': tensor[M], 'lambda_max': float }
                    },
                    'n_clusters': int
                }
        """
        assert K.dim() == 3 and assignments.dim() == 2
        B, N, _ = K.shape
        W = self.sanitize_weights(K, nonnegative=True, symmetrize=True, zero_out_diag=False)
        spectra: List[Dict[str, Any]] = []

        for b in range(B):
            a_b = assignments[b]  # [N]
            n_clusters_b = int(a_b.max().item()) + 1 if a_b.numel() > 0 else 0
            per_cluster: Dict[int, Dict[str, Any]] = {}
            for c in range(n_clusters_b):
                mask = (a_b == c)
                idx = torch.nonzero(mask, as_tuple=False).flatten()
                if idx.numel() <= 1:
                    # No transverse DOF
                    lambdas = torch.zeros(0, device=K.device, dtype=K.dtype)
                    lambda_max = _to_tensor(0.0, device=K.device, dtype=K.dtype)
                    per_cluster[c] = {'lambdas': lambdas, 'lambda_max': lambda_max}
                    continue
                Wk = W[b][idx][:, idx]  # [m, m]
                # Laplacian L = D - W
                d = torch.sum(Wk, dim=-1)
                L = torch.diag(d) - Wk
                # eigh on device with fallback
                try:
                    lams, _V = torch.linalg.eigh(L)
                except Exception:
                    Lcpu = L.detach().to("cpu")
                    lcpu, _Vcpu = torch.linalg.eigh(Lcpu)
                    lams = lcpu.to(device=K.device, dtype=K.dtype)
                # Sort ascending
                lams, _ = torch.sort(lams, dim=0)
                # Exclude zero mode if exists (numerically near zero)
                if lams.numel() > 0:
                    # Consider all nonzero modes as transverse; keep them in return_all
                    # λ_max among transverse modes
                    if lams.numel() > 1:
                        lambda_max = torch.max(lams[1:])
                    else:
                        lambda_max = _to_tensor(0.0, device=K.device, dtype=K.dtype)
                else:
                    lambda_max = _to_tensor(0.0, device=K.device, dtype=K.dtype)
                per_cluster[c] = {
                    'lambdas': lams if return_all else torch.empty(0, device=K.device, dtype=K.dtype),
                    'lambda_max': lambda_max,
                }
            spectra.append({'per_cluster': per_cluster, 'n_clusters': n_clusters_b})
        return spectra

    # -----------------------------
    # Transverse Lyapunov surrogate
    # -----------------------------
    def compute_transverse_lyapunov_exponents(
        self,
        coupling_matrix: torch.Tensor,
        cluster_assignments: torch.Tensor,
        dt: float,
        sigma: float,
        alpha: float = 0.0,
        use_amplitude_scaling: bool = False,
        amplitudes: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute per-cluster transverse Lyapunov exponents using a discrete-time surrogate.

        Args:
            coupling_matrix: [B, N, N]
            cluster_assignments: [B, N] long
            dt: float time step
            sigma: scalar coupling multiplier (effective)
            alpha: Sakaguchi phase lag
            use_amplitude_scaling: if True and amplitudes provided, scale lambda_max by mean amp of cluster
            amplitudes: optional [B, N]

        Returns:
            {
              'Lambda': List[Dict[int, float]],      # per-batch dict of per-cluster Λ_k
              'lambda_max': List[Dict[int, float]],  # per-batch dict of per-cluster λ_max,k
              'stable_flags': List[Dict[int, bool]],
              'gamma': float(dt * sigma * cos(alpha)),
              'cos_alpha': float,
            }
        """
        assert coupling_matrix.dim() == 3 and cluster_assignments.dim() == 2
        B, N, _ = coupling_matrix.shape
        # Compute spectra
        spectra = self.cluster_laplacian_eigs(coupling_matrix, cluster_assignments, return_all=False)
        cos_alpha = math.cos(float(alpha))
        gamma = float(dt) * float(sigma) * float(cos_alpha)

        Lambda_list: List[Dict[int, float]] = []
        lmax_list: List[Dict[int, float]] = []
        stable_list: List[Dict[int, bool]] = []

        for b in range(B):
            per_cluster = spectra[b]['per_cluster']
            Lambdas: Dict[int, float] = {}
            Lmaxs: Dict[int, float] = {}
            Stab: Dict[int, bool] = {}
            # Optional amplitude scaling
            amp_b = amplitudes[b] if (use_amplitude_scaling and isinstance(amplitudes, torch.Tensor)) else None
            for c, d in per_cluster.items():
                lmax = float(d['lambda_max'].detach().item()) if isinstance(d['lambda_max'], torch.Tensor) else float(d['lambda_max'])
                if amp_b is not None:
                    # scale by mean amplitude of cluster (heuristic)
                    mask = (cluster_assignments[b] == c)
                    if torch.any(mask):
                        mean_amp = float(torch.mean(amp_b[mask]).detach().item())
                        lmax = lmax * max(self.eps, mean_amp)
                # Growth factor and Lyapunov surrogate
                g = abs(1.0 - gamma * lmax)
                # Avoid log(0)
                Lambda = math.log(max(self.eps, g))
                Lambdas[c] = float(Lambda)
                Lmaxs[c] = float(lmax)
                # Stable if Λ < 0, require gamma > 0 and lmax > 0 for meaningful criterion
                Stab[c] = (Lambda < 0.0) and (gamma > 0.0) and (lmax > 0.0)
            Lambda_list.append(Lambdas)
            lmax_list.append(Lmaxs)
            stable_list.append(Stab)

        return {
            'Lambda': Lambda_list,
            'lambda_max': lmax_list,
            'stable_flags': stable_list,
            'gamma': gamma,
            'cos_alpha': cos_alpha,
        }

    def predict_dynamical_regime(self, Lambda_per_batch: Dict[int, float] | List[Dict[int, float]]) -> List[str]:
        """
        Predict regime per batch from per-cluster Λ_k:
          - all Λ_k < 0: cluster_sync
          - all Λ_k > 0: asynchronous
          - mixed: chimera

        Args:
            Lambda_per_batch: dict per-batch, or list of dicts
        Returns:
            regimes: list of strings per batch
        """
        # Normalize to list[dict]
        if isinstance(Lambda_per_batch, dict):
            Lambda_list = [Lambda_per_batch]
        else:
            Lambda_list = list(Lambda_per_batch)

        regimes: List[str] = []
        for Lambdas in Lambda_list:
            if len(Lambdas) == 0:
                regimes.append("asynchronous")
                continue
            vals = list(Lambdas.values())
            all_neg = all(v < 0.0 for v in vals)
            all_pos = all(v > 0.0 for v in vals)
            if all_neg:
                regimes.append("cluster_sync")
            elif all_pos:
                regimes.append("asynchronous")
            else:
                regimes.append("chimera")
        return regimes

    def predict_critical_coupling(
        self,
        coupling_matrix: torch.Tensor,
        cluster_assignments: torch.Tensor,
        dt: float,
        alpha: float = 0.0,
    ) -> List[Dict[int, float]]:
        """
        Estimate σ_crit per cluster via:
            σ_crit,k ≈ 2 / (dt * cos(α) * λ_max,k)

        Returns:
            List length B of dicts { cluster_id: sigma_crit_k }
        """
        assert coupling_matrix.dim() == 3 and cluster_assignments.dim() == 2
        spectra = self.cluster_laplacian_eigs(coupling_matrix, cluster_assignments, return_all=False)
        cos_alpha = math.cos(float(alpha))
        out: List[Dict[int, float]] = []
        for b in range(len(spectra)):
            per_cluster = spectra[b]['per_cluster']
            res: Dict[int, float] = {}
            for c, d in per_cluster.items():
                lmax = float(d['lambda_max'].detach().item()) if isinstance(d['lambda_max'], torch.Tensor) else float(d['lambda_max'])
                if lmax > 0.0 and cos_alpha > 0.0 and dt > 0.0:
                    res[c] = float(2.0 / (dt * cos_alpha * lmax))
                else:
                    res[c] = float('inf')
            out.append(res)
        return out
