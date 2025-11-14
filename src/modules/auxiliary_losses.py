"""
Auxiliary Loss Functions for Resonance Transformer

Uses CDNS metrics, order parameter variance, and spectral bands as auxiliary
losses to guide training toward desired dynamical regimes.
"""

from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn


def cdns_auxiliary_loss(
    metrics: Dict,
    target_consonance: float = 0.6,
    target_dissonance: float = 0.2,
    weight_consonance: float = 0.01,
    weight_dissonance: float = 0.01,
) -> torch.Tensor:
    """
    Auxiliary loss based on CDNS metrics.
    
    Encourages the model to maintain target consonance and dissonance levels.
    
    Args:
        metrics: Dictionary containing CDNS metrics
        target_consonance: Target consonance value (order parameter)
        target_dissonance: Target dissonance value (XY energy)
        weight_consonance: Weight for consonance loss
        weight_dissonance: Weight for dissonance loss
    
    Returns:
        Scalar loss tensor
    """
    loss = torch.tensor(0.0, device='cpu')
    
    # Extract CDNS metrics
    cdns = metrics.get('cdns', {})
    if not isinstance(cdns, dict):
        return loss
    
    # Consonance loss (squared error from target)
    if 'consonance' in cdns:
        consonance = cdns['consonance']
        if isinstance(consonance, torch.Tensor):
            # Take mean if multi-dimensional
            if consonance.numel() > 1:
                consonance = consonance.mean()
            loss = loss + weight_consonance * (consonance - target_consonance) ** 2
    
    # Dissonance loss (squared error from target)
    if 'dissonance' in cdns:
        dissonance = cdns['dissonance']
        if isinstance(dissonance, torch.Tensor):
            if dissonance.numel() > 1:
                dissonance = dissonance.mean()
            loss = loss + weight_dissonance * (dissonance - target_dissonance) ** 2
    
    return loss


def metastability_band_loss_torch(
    order_param_variance: torch.Tensor,
    var_lo: float = 0.05,
    var_hi: float = 0.3,
    weight: float = 0.01,
) -> torch.Tensor:
    """
    Encourage order parameter variance to lie within a target band.
    
    Penalizes both too-low (locking) and too-high (chaos) variance.
    This encourages metastability (R ≈ 0.6 with moderate variance).
    
    Args:
        order_param_variance: Variance of order parameter [batch] or scalar
        var_lo: Lower bound for variance band
        var_hi: Upper bound for variance band
        weight: Loss weight
    
    Returns:
        Scalar loss tensor
    """
    if isinstance(order_param_variance, torch.Tensor):
        if order_param_variance.numel() > 1:
            var_mean = order_param_variance.mean()
        else:
            var_mean = order_param_variance.item()
    else:
        var_mean = float(order_param_variance)
    
    # Penalty for being below lower bound
    err_low = torch.clamp(torch.tensor(var_lo) - var_mean, min=0.0)
    
    # Penalty for being above upper bound
    err_high = torch.clamp(var_mean - torch.tensor(var_hi), min=0.0)
    
    band_err = err_low + err_high
    
    return weight * band_err


def criticality_loss_torch(
    order_parameter: torch.Tensor,
    target_R: float = 0.6,
    weight: float = 0.01,
) -> torch.Tensor:
    """
    Loss to maintain target order parameter (criticality).
    
    Args:
        order_parameter: Order parameter R [batch] or scalar
        target_R: Target order parameter value
        weight: Loss weight
    
    Returns:
        Scalar loss tensor
    """
    if isinstance(order_parameter, torch.Tensor):
        if order_parameter.numel() > 1:
            R_mean = order_parameter.mean()
        else:
            R_mean = order_parameter
    else:
        R_mean = torch.tensor(float(order_parameter))
    
    return weight * (R_mean - target_R) ** 2


def spectral_band_loss_torch(
    spectral_bands: torch.Tensor,
    target_distribution: Optional[torch.Tensor] = None,
    weight: float = 0.005,
) -> torch.Tensor:
    """
    Loss to encourage desired spectral band energy distribution.
    
    Args:
        spectral_bands: Spectral band energies [num_bands] or [batch, num_bands]
        target_distribution: Target distribution over bands (None = uniform)
        weight: Loss weight
    
    Returns:
        Scalar loss tensor
    """
    if spectral_bands.numel() == 0:
        return torch.tensor(0.0)
    
    # Normalize to distribution
    if spectral_bands.dim() > 1:
        # Average over batch
        bands = spectral_bands.mean(dim=0)
    else:
        bands = spectral_bands
    
    # Normalize to probabilities
    bands_softmax = torch.softmax(bands, dim=0)
    
    if target_distribution is None:
        # Uniform target
        num_bands = bands_softmax.shape[0]
        target_distribution = torch.ones(num_bands) / num_bands
    
    # KL divergence or MSE
    loss = torch.nn.functional.mse_loss(bands_softmax, target_distribution)
    
    return weight * loss


def aggregate_auxiliary_losses(
    metrics: Dict,
    weights: Optional[Dict[str, float]] = None,
    target_R: float = 0.6,
    target_consonance: float = 0.6,
    target_dissonance: float = 0.2,
    var_lo: float = 0.05,
    var_hi: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate all auxiliary losses from metrics.
    
    Args:
        metrics: Dictionary of resonance metrics
        weights: Dictionary of loss weights (defaults used if None)
        target_R: Target order parameter
        target_consonance: Target consonance
        target_dissonance: Target dissonance
        var_lo: Lower bound for variance band
        var_hi: Upper bound for variance band
    
    Returns:
        Dictionary of loss components and total loss
    """
    if weights is None:
        weights = {
            'cdns': 0.01,
            'criticality': 0.01,
            'metastability': 0.01,
            'spectral': 0.005,
        }
    
    losses = {}
    
    # CDNS loss
    if 'cdns' in metrics:
        losses['cdns'] = cdns_auxiliary_loss(
            metrics,
            target_consonance=target_consonance,
            target_dissonance=target_dissonance,
            weight_consonance=weights.get('cdns', 0.01),
            weight_dissonance=weights.get('cdns', 0.01),
        )
    
    # Criticality loss
    if 'final_order_parameter' in metrics:
        order_param = metrics['final_order_parameter']
        if isinstance(order_param, torch.Tensor):
            losses['criticality'] = criticality_loss_torch(
                order_param,
                target_R=target_R,
                weight=weights.get('criticality', 0.01),
            )
    
    # Metastability band loss
    if 'order_param_variance' in metrics:
        var_R = metrics['order_param_variance']
        if isinstance(var_R, torch.Tensor):
            losses['metastability'] = metastability_band_loss_torch(
                var_R,
                var_lo=var_lo,
                var_hi=var_hi,
                weight=weights.get('metastability', 0.01),
            )
    
    # Spectral band loss
    if 'spectral_bands' in metrics:
        spectral_bands = metrics['spectral_bands']
        if isinstance(spectral_bands, torch.Tensor):
            losses['spectral'] = spectral_band_loss_torch(
                spectral_bands,
                weight=weights.get('spectral', 0.005),
            )
    
    # Total loss
    total = sum(losses.values()) if losses else torch.tensor(0.0)
    losses['total'] = total
    
    return losses


