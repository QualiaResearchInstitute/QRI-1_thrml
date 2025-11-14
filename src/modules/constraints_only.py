"""
ConstraintsOnlyCoreML
- ANE-friendly constraints path for Core ML export.
- Runs embeddings + optional beam-splitter + spectral head (+ optional FFN/LN) and emits:
    - processed features x'  [B, T, C]
    - Q, K, V projections    [B, T, C] each
This keeps ops within Conv/Linear/Elementwise to maximize ANE compatibility.

Intended for use as the Core ML subgraph. The resonance (Kuramoto/Stuart-Landau) loop runs on Metal GPU.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from resonance_transformer.modules.beam_splitter import BeamSplitterUnitaryStack
from resonance_transformer.modules.spectral_threshold import SpectralThresholdHead


class ConstraintsOnlyCoreML(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 512,
        use_beam_splitter: bool = True,
        bs_layers: int = 2,
        use_spectral_head: bool = True,
        spectral_bands: int = 4,
        add_ffn_ln: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            vocab_size: tokenizer vocab
            d_model: model width (must be even if use_beam_splitter=True)
            max_seq_len: max positions for position embeddings
            use_beam_splitter: include beam-splitter unitary stack (ANE-friendly)
            bs_layers: number of beam-splitter layers
            use_spectral_head: include spectral threshold head (ANE-friendly)
            spectral_bands: number of bands for spectral thresholding
            add_ffn_ln: optionally include a light FFN + LN block
            dropout: dropout prob (kept 0 for export)
        """
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Optional ANE-friendly constraint modules
        self.beam_splitter = BeamSplitterUnitaryStack(d_model, n_layers=int(bs_layers)) if use_beam_splitter else None
        self.spectral_head = SpectralThresholdHead(d_model, bands=int(spectral_bands), fixed_basis="dct8", learn_thresholds=True) if use_spectral_head else None

        # Optional light FFN + LN for additional shaping
        self.add_ffn_ln = bool(add_ffn_ln)
        if self.add_ffn_ln:
            d_ff = 4 * d_model
            self.ln1 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            self.ln2 = nn.LayerNorm(d_model)

        # QKV projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [B, T] int64
        Returns:
            x_prime: [B, T, C]
            Q: [B, T, C]
            K: [B, T, C]
            V: [B, T, C]
        """
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0)

        tok = self.token_embedding(input_ids)             # [B, T, C]
        pos_emb = self.position_embedding(pos)            # [1, T, C]
        x = self.drop(tok + pos_emb)

        if self.beam_splitter is not None:
            x = self.beam_splitter(x)

        if self.spectral_head is not None:
            x = self.spectral_head(x)

        if self.add_ffn_ln:
            h = self.ln1(x)
            h = self.ff(h)
            x = self.ln2(x + h)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return x, Q, K, V
