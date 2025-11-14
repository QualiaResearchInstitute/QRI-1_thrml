"""
Unified Device Orchestrator - The Singularity

A self-optimizing system that transparently routes operations across ANE/MPS/CPU,
learns optimal placements through profiling, and adapts in real-time.

This is the "singularity" - device placement becomes completely transparent,
with the system learning and optimizing itself.
"""

from __future__ import annotations

import math
import random
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from statistics import median
import torch
import torch.nn as nn
from functools import wraps

from modules.device_manager import DeviceManager, get_device_manager


@dataclass
class OperationProfile:
    """Profile of an operation's performance on different devices."""
    operation_type: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    execution_time_ms: float
    memory_used_mb: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


ENERGY_COST_MAP = {
    "cpu": 1.0,
    "cuda": 1.25,
    "mps": 0.7,
    "ane": 0.35,
    "coreml": 0.35,
}

DEFAULT_METRIC_WEIGHTS = {
    "time": 0.4,
    "success": 0.3,
    "headroom": 0.2,
    "energy": 0.1,
}

MAX_RECENT_SAMPLES = 32


@dataclass
class BanditArmStats:
    """Contextual bandit arm statistics for a specific device."""
    device: str
    successes: int = 0
    failures: int = 0
    total_time_ms: float = 0.0
    total_time_sq: float = 0.0
    total_energy_cost: float = 0.0
    avg_memory_headroom: float = 1.0
    samples: int = 0
    last_updated: float = field(default_factory=time.time)
    recent_times: List[float] = field(default_factory=list)
    recent_rewards: List[float] = field(default_factory=list)

    def record(
        self,
        time_ms: float,
        success: bool,
        headroom: float,
        energy_cost: float,
        reward: float,
        max_recent: int = MAX_RECENT_SAMPLES,
    ) -> None:
        prev_samples = self.samples
        self.samples += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_time_ms += time_ms
        self.total_time_sq += time_ms ** 2
        self.total_energy_cost += energy_cost
        if prev_samples == 0:
            self.avg_memory_headroom = headroom
        else:
            self.avg_memory_headroom = (
                (self.avg_memory_headroom * prev_samples) + headroom
            ) / (prev_samples + 1)
        self.recent_times.append(time_ms)
        self.recent_rewards.append(reward)
        if len(self.recent_times) > max_recent:
            self.recent_times = self.recent_times[-max_recent:]
        if len(self.recent_rewards) > max_recent:
            self.recent_rewards = self.recent_rewards[-max_recent:]
        self.last_updated = time.time()

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.0

    @property
    def mean_time_ms(self) -> float:
        return self.total_time_ms / self.samples if self.samples else float("inf")

    @property
    def mean_energy_cost(self) -> float:
        return self.total_energy_cost / self.samples if self.samples else ENERGY_COST_MAP.get(self.device, 1.0)


@dataclass
class DeviceStrategy:
    """Learned strategy for device placement."""
    operation_signature: str  # e.g., "linear_512_1024_float32"
    preferred_device: str
    fallback_devices: List[str] = field(default_factory=list)
    avg_time_ms: float = 0.0
    success_rate: float = 1.0
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)
    arms: Dict[str, BanditArmStats] = field(default_factory=dict)
    pareto_front: List[str] = field(default_factory=list)
    hysteresis_anchor: Optional[str] = None


class UnifiedDeviceOrchestrator:
    """
    The Singularity: Self-optimizing device orchestrator.
    
    Features:
    - Transparent operation routing
    - Performance profiling and learning
    - Adaptive optimization
    - Hybrid execution support
    - Automatic fallback
    """
    
    def __init__(
        self,
        device_manager: Optional[DeviceManager] = None,
        enable_profiling: bool = True,
        enable_learning: bool = True,
        profile_cache_path: Optional[str] = None,
        min_samples_for_learning: int = 3,
        exploration_coef: float = 0.3,
        reward_time_scale: float = 100.0,
        drift_threshold: float = 0.35,
        drift_window: int = 5,
        hysteresis_margin: float = 0.05,
        metric_weights: Optional[Dict[str, float]] = None,
        recent_window: int = MAX_RECENT_SAMPLES,
    ):
        """
        Initialize the unified device orchestrator.
        
        Args:
            device_manager: Device manager instance (creates new if None)
            enable_profiling: Enable performance profiling
            enable_learning: Enable learning from profiles
            profile_cache_path: Path to save/load learned strategies
            min_samples_for_learning: Minimum samples before trusting learned strategy
        """
        self.device_manager = device_manager or get_device_manager()
        self.enable_profiling = enable_profiling
        self.enable_learning = enable_learning
        self.profile_cache_path = profile_cache_path
        self.min_samples_for_learning = min_samples_for_learning
        self.exploration_coef = exploration_coef
        self.reward_time_scale = reward_time_scale
        self.drift_threshold = drift_threshold
        self.drift_window = max(3, drift_window)
        self.hysteresis_margin = hysteresis_margin
        self.metric_weights = metric_weights or DEFAULT_METRIC_WEIGHTS.copy()
        self.recent_window = recent_window
        
        # Learned strategies: operation_signature -> DeviceStrategy
        self.strategies: Dict[str, DeviceStrategy] = {}
        
        # Performance profiles: operation_signature -> List[OperationProfile]
        self.profiles: Dict[str, List[OperationProfile]] = defaultdict(list)
        
        # Thread lock for thread-safe updates
        self._lock = threading.Lock()
        
        # Load cached strategies if available
        if profile_cache_path and Path(profile_cache_path).exists():
            self._load_strategies()
    
    def _device_candidates(self) -> List[str]:
        """Return list of candidate device strings for bandit selection."""
        candidates = {"cpu"}
        if self.device_manager.primary_device is not None:
            candidates.add(str(self.device_manager.primary_device))
        if self.device_manager.has_cuda:
            candidates.add("cuda")
        if self.device_manager.has_mps:
            candidates.add("mps")
        if getattr(self.device_manager, "has_ane", False):
            candidates.add("ane")
        return list(candidates)
    
    def _energy_cost_for_device(self, device: str) -> float:
        return ENERGY_COST_MAP.get(device.split(":")[0], 1.0)
    
    def _build_context_features(
        self,
        operation_type: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        tensor_size_bytes: Optional[int],
        device_state: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        batch_size = shape[0] if shape else None
        seq_len = shape[1] if len(shape) > 1 else None
        return {
            "operation_type": operation_type,
            "shape": shape,
            "dtype": str(dtype),
            "tensor_size_bytes": tensor_size_bytes,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "device_state": device_state,
        }
    
    @staticmethod
    def _estimate_tensor_bytes(shape: Tuple[int, ...], dtype: torch.dtype) -> Optional[int]:
        if not shape:
            return None
        try:
            element_size = torch.tensor([], dtype=dtype).element_size()
        except Exception:
            element_size = torch.tensor([], dtype=torch.float32).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= max(1, int(dim))
        return total_elements * element_size
    
    def _operation_signature(
        self,
        operation_type: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> str:
        """Create a signature for an operation."""
        dtype_str = str(dtype).replace("torch.", "")
        shape_str = "_".join(map(str, shape))
        return f"{operation_type}_{shape_str}_{dtype_str}"
    
    def _ensure_strategy(self, signature: str) -> DeviceStrategy:
        if signature not in self.strategies:
            preferred = str(self.device_manager.primary_device or torch.device("cpu"))
            self.strategies[signature] = DeviceStrategy(
                operation_signature=signature,
                preferred_device=preferred,
            )
        return self.strategies[signature]
    
    def _ensure_arm(self, strategy: DeviceStrategy, device: str) -> BanditArmStats:
        key = device.split(":")[0]
        if key not in strategy.arms:
            strategy.arms[key] = BanditArmStats(device=key)
        return strategy.arms[key]
    
    def _get_optimal_device(
        self,
        operation_type: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        tensor_size_bytes: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """
        Get optimal device for an operation, using learned strategies if available.
        
        Returns:
            (device_name, is_learned) tuple
        """
        signature = self._operation_signature(operation_type, shape, dtype)
        
        # Check if we have a learned strategy / bandit policy
        if self.enable_learning and signature in self.strategies and context is not None:
            learned_device = self._bandit_select_device(signature, context)
            if learned_device:
                return learned_device, True
        
        # Fall back to device manager heuristics
        device = self.device_manager.get_device_for_operation(
            operation_type,
            tensor_size_bytes,
        )
        return str(device), False
    
    def _profile_operation(
        self,
        operation_type: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        execution_time_ms: float,
        memory_used_mb: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance profile."""
        if not self.enable_profiling:
            return
        
        metadata = metadata or {}
        context = metadata.get("context", {})
        device_state = context.get("device_state", {})
        device_key = device.type
        headroom = device_state.get(device_key, {}).get("memory_headroom", 1.0)
        energy_cost = self._energy_cost_for_device(device_key)
        reward = self._evaluate_reward(execution_time_ms, success, headroom, energy_cost)
        
        profile = OperationProfile(
            operation_type=operation_type,
            shape=shape,
            dtype=str(dtype).replace("torch.", ""),
            device=str(device),
            execution_time_ms=execution_time_ms,
            memory_used_mb=memory_used_mb,
            success=success,
            metadata=metadata,
        )
        
        signature = self._operation_signature(operation_type, shape, dtype)
        
        with self._lock:
            self.profiles[signature].append(profile)
            
            # Keep only recent profiles (last 100)
            if len(self.profiles[signature]) > 100:
                self.profiles[signature] = self.profiles[signature][-100:]
            
            # Update strategy if learning is enabled
            if self.enable_learning:
                strategy = self._ensure_strategy(signature)
                arm = self._ensure_arm(strategy, device_key)
                arm.record(
                    execution_time_ms,
                    success,
                    headroom,
                    energy_cost,
                    reward,
                    self.recent_window,
                )
                self._detect_drift(strategy, arm)
                self._update_strategy(signature)
    
    def _evaluate_reward(
        self,
        time_ms: float,
        success: bool,
        headroom: float,
        energy_cost: float,
    ) -> float:
        time_score = 1.0 / (1.0 + (time_ms / self.reward_time_scale))
        success_score = 1.0 if success else 0.0
        headroom_score = max(0.0, min(1.0, headroom))
        energy_score = 1.0 / (1.0 + energy_cost)
        weights = self.metric_weights
        return (
            weights["time"] * time_score
            + weights["success"] * success_score
            + weights["headroom"] * headroom_score
            + weights["energy"] * energy_score
        )
    
    def _detect_drift(self, strategy: DeviceStrategy, arm: BanditArmStats) -> bool:
        if len(arm.recent_times) < self.drift_window or arm.samples < self.drift_window * 2:
            return False
        recent_avg = sum(arm.recent_times[-self.drift_window:]) / self.drift_window
        overall_avg = arm.mean_time_ms
        if not math.isfinite(overall_avg) or overall_avg <= 0:
            return False
        delta = abs(recent_avg - overall_avg) / max(overall_avg, 1e-3)
        if delta > self.drift_threshold:
            arm.successes = int(arm.successes * 0.5)
            arm.failures = int(arm.failures * 0.5)
            arm.total_time_ms *= 0.5
            arm.total_time_sq *= 0.5
            arm.total_energy_cost *= 0.5
            strategy.sample_count = max(0, strategy.sample_count - self.drift_window)
            strategy.last_updated = time.time()
            return True
        return False
    
    def _update_strategy(self, signature: str) -> None:
        """Update learned strategy based on bandit arm statistics."""
        strategy = self._ensure_strategy(signature)
        if not strategy.arms:
            return
        
        total_samples = sum(arm.samples for arm in strategy.arms.values())
        if total_samples < 1:
            return
        
        pareto = self._compute_pareto_front(strategy)
        if not pareto:
            pareto = list(strategy.arms.keys())
        strategy.pareto_front = pareto
        
        candidate_scores: Dict[str, float] = {}
        for device in pareto:
            arm = strategy.arms[device]
            candidate_scores[device] = self._ucb_score(arm, total_samples)
        
        if not candidate_scores:
            return
        
        best_device = self._apply_hysteresis(strategy, candidate_scores)
        strategy.preferred_device = best_device
        strategy.avg_time_ms = strategy.arms[best_device].mean_time_ms
        strategy.success_rate = strategy.arms[best_device].success_rate
        strategy.sample_count = total_samples
        strategy.last_updated = time.time()
        
        fallback_devices = sorted(
            (d for d in candidate_scores.keys() if d != best_device),
            key=lambda d: candidate_scores[d],
            reverse=True,
        )
        strategy.fallback_devices = fallback_devices[:2]
        
        if self.profile_cache_path and (total_samples % 10 == 0):
            self._save_strategies()
    
    def _compute_pareto_front(self, strategy: DeviceStrategy) -> List[str]:
        devices = list(strategy.arms.keys())
        pareto: List[str] = []
        for device in devices:
            dominated = False
            metrics = self._arm_metrics(strategy.arms[device])
            for other in devices:
                if other == device:
                    continue
                if self._dominates(self._arm_metrics(strategy.arms[other]), metrics):
                    dominated = True
                    break
            if not dominated:
                pareto.append(device)
        return pareto
    
    @staticmethod
    def _dominates(other: Dict[str, float], target: Dict[str, float]) -> bool:
        better_or_equal = all(other[k] >= target[k] for k in other.keys())
        strictly_better = any(other[k] > target[k] for k in other.keys())
        return better_or_equal and strictly_better
    
    def _arm_metrics(self, arm: BanditArmStats) -> Dict[str, float]:
        time_value = arm.mean_time_ms
        if arm.recent_times:
            try:
                time_value = median(arm.recent_times)
            except Exception:
                time_value = arm.mean_time_ms
        time_score = 1.0 / (1.0 + (time_value / self.reward_time_scale))
        success_score = arm.success_rate
        headroom_score = max(0.0, min(1.0, arm.avg_memory_headroom))
        energy_score = 1.0 / (1.0 + arm.mean_energy_cost)
        return {
            "time": time_score,
            "success": success_score,
            "headroom": headroom_score,
            "energy": energy_score,
        }
    
    def _ucb_score(self, arm: BanditArmStats, total_samples: int) -> float:
        metrics = self._arm_metrics(arm)
        success_alpha = arm.successes + 1
        success_beta = arm.failures + 1
        success_sample = random.betavariate(success_alpha, success_beta)
        blended_success = 0.5 * metrics["success"] + 0.5 * success_sample
        weights = self.metric_weights
        base_reward = (
            weights["time"] * metrics["time"]
            + weights["success"] * blended_success
            + weights["headroom"] * metrics["headroom"]
            + weights["energy"] * metrics["energy"]
        )
        exploration = self.exploration_coef * math.sqrt(
            math.log(max(total_samples, 1) + 1) / (arm.samples + 1)
        )
        return base_reward + exploration
    
    def _apply_hysteresis(
        self,
        strategy: DeviceStrategy,
        scores: Dict[str, float],
    ) -> str:
        best_device = max(scores, key=scores.get)
        current = strategy.preferred_device
        if current in scores:
            current_score = scores[current]
            best_score = scores[best_device]
            if best_device != current and best_score < current_score * (1 + self.hysteresis_margin):
                return current
        strategy.hysteresis_anchor = best_device
        return best_device
    
    def _bandit_select_device(
        self,
        signature: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        strategy = self.strategies.get(signature)
        if not strategy:
            return None
        for candidate in self._device_candidates():
            self._ensure_arm(strategy, candidate.split(":")[0])
        if strategy.sample_count < self.min_samples_for_learning:
            return None
        total_samples = sum(arm.samples for arm in strategy.arms.values())
        if total_samples < self.min_samples_for_learning:
            return None
        
        if not strategy.pareto_front:
            strategy.pareto_front = self._compute_pareto_front(strategy)
        candidate_devices = strategy.pareto_front or list(strategy.arms.keys())
        candidate_scores: Dict[str, float] = {}
        device_state = context.get("device_state", {})
        for device in candidate_devices:
            arm = strategy.arms[device]
            score = self._ucb_score(arm, total_samples)
            state = device_state.get(device, {})
            if state:
                headroom = state.get("memory_headroom")
                if headroom is not None:
                    score *= (0.8 + 0.2 * max(0.0, min(1.0, headroom)))
            candidate_scores[device] = score
        
        if not candidate_scores:
            return None
        return self._apply_hysteresis(strategy, candidate_scores)
    
    def execute_with_profiling(
        self,
        operation: Callable,
        operation_type: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute an operation with automatic device routing and profiling.
        
        This is the core "singularity" method - operations just work,
        with optimal device placement happening transparently.
        """
        tensor_size_bytes = self._estimate_tensor_bytes(shape, dtype)
        signature = self._operation_signature(operation_type, shape, dtype)
        device_state = self.device_manager.get_device_state_snapshot()
        context = self._build_context_features(
            operation_type,
            shape,
            dtype,
            tensor_size_bytes,
            device_state,
        )
        # Get optimal device
        device_name, is_learned = self._get_optimal_device(
            operation_type,
            shape,
            dtype,
            tensor_size_bytes,
            context,
        )
        device = torch.device(device_name)
        
        # Move tensors to device if needed
        args_device = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                args_device.append(arg.to(device))
            else:
                args_device.append(arg)
        
        kwargs_device = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs_device[key] = value.to(device)
            else:
                kwargs_device[key] = value
        
        # Execute with profiling
        start_time = time.time()
        memory_before = 0.0
        memory_after = 0.0
        
        try:
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated(device) / (1024**2)
            
            result = operation(*args_device, **kwargs_device)
            
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated(device) / (1024**2)
            
            execution_time_ms = (time.time() - start_time) * 1000
            memory_used_mb = max(0, memory_after - memory_before)
            
            # Profile successful execution
            self._profile_operation(
                operation_type,
                shape,
                dtype,
                device,
                execution_time_ms,
                memory_used_mb,
                success=True,
                metadata={"learned": is_learned, "context": context},
            )
            
            return result
            
        except RuntimeError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Profile failed execution
            self._profile_operation(
                operation_type,
                shape,
                dtype,
                device,
                execution_time_ms,
                0.0,
                success=False,
                metadata={"error": str(e), "learned": is_learned, "context": context},
            )
            
            # Try fallback device if available
            if signature in self.strategies:
                strategy = self.strategies[signature]
                if strategy.fallback_devices:
                    fallback_device = torch.device(strategy.fallback_devices[0])
                    print(f"⚠️  Operation failed on {device}, trying fallback {fallback_device}")
                    
                    # Retry on fallback
                    args_fallback = [
                        arg.to(fallback_device) if isinstance(arg, torch.Tensor) else arg
                        for arg in args
                    ]
                    kwargs_fallback = {
                        k: (v.to(fallback_device) if isinstance(v, torch.Tensor) else v)
                        for k, v in kwargs.items()
                    }
                    
                    return operation(*args_fallback, **kwargs_fallback)
            
            # If no fallback, try CPU
            if device.type != "cpu":
                print(f"⚠️  Operation failed on {device}, falling back to CPU")
                args_cpu = [
                    arg.to("cpu") if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ]
                kwargs_cpu = {
                    k: (v.to("cpu") if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                }
                return operation(*args_cpu, **kwargs_cpu)
            
            raise
    
    def wrap_module(self, module: nn.Module) -> nn.Module:
        """
        Wrap a PyTorch module to use unified device orchestration.
        
        This makes device placement completely transparent!
        """
        original_forward = module.forward
        
        @wraps(original_forward)
        def orchestrated_forward(*args, **kwargs):
            # Extract tensor info for routing
            if args and isinstance(args[0], torch.Tensor):
                input_tensor = args[0]
                operation_type = module.__class__.__name__.lower()
                shape = tuple(input_tensor.shape)
                dtype = input_tensor.dtype
                
                # Execute with profiling
                return self.execute_with_profiling(
                    original_forward,
                    operation_type,
                    shape,
                    dtype,
                    *args,
                    **kwargs,
                )
            
            return original_forward(*args, **kwargs)
        
        module.forward = orchestrated_forward
        return module
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned strategies and performance."""
        total_profiles = sum(len(profiles) for profiles in self.profiles.values())
        learned_strategies = len([
            s for s in self.strategies.values()
            if s.sample_count >= self.min_samples_for_learning
        ])
        
        return {
            "total_profiles": total_profiles,
            "unique_operations": len(self.profiles),
            "learned_strategies": learned_strategies,
            "strategies": {
                sig: {
                    "preferred_device": s.preferred_device,
                    "avg_time_ms": s.avg_time_ms,
                    "success_rate": s.success_rate,
                    "sample_count": s.sample_count,
                    "pareto_front": s.pareto_front,
                }
                for sig, s in self.strategies.items()
            },
        }
    
    def _save_strategies(self) -> None:
        """Save learned strategies to disk."""
        if not self.profile_cache_path:
            return
        
        try:
            strategies_data = {
                sig: asdict(strategy)
                for sig, strategy in self.strategies.items()
            }
            
            Path(self.profile_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.profile_cache_path, 'w') as f:
                json.dump(strategies_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save strategies: {e}")
    
    def _load_strategies(self) -> None:
        """Load learned strategies from disk."""
        if not self.profile_cache_path or not Path(self.profile_cache_path).exists():
            return
        
        try:
            with open(self.profile_cache_path, 'r') as f:
                strategies_data = json.load(f)
            
            for sig, data in strategies_data.items():
                arms_data = data.pop("arms", {}) or {}
                data.setdefault("preferred_device", data.get("preferred_device", "cpu"))
                strategy = DeviceStrategy(**data)
                strategy.arms = {
                    dev: BanditArmStats(**arm_dict)
                    for dev, arm_dict in arms_data.items()
                }
                self.strategies[sig] = strategy
        except Exception as e:
            print(f"Warning: Failed to load strategies: {e}")


# Global orchestrator instance
_global_orchestrator: Optional[UnifiedDeviceOrchestrator] = None


def get_orchestrator() -> UnifiedDeviceOrchestrator:
    """Get or create the global unified device orchestrator."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = UnifiedDeviceOrchestrator()
    return _global_orchestrator


def orchestrate(operation_type: str):
    """
    Decorator to automatically orchestrate a function/operation.
    
    Usage:
        @orchestrate("linear")
        def my_linear_layer(x, weight):
            return torch.nn.functional.linear(x, weight)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            orchestrator = get_orchestrator()
            
            # Extract tensor info
            input_tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_tensor = arg
                    break
            
            if input_tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        input_tensor = value
                        break
            
            if input_tensor is not None:
                shape = tuple(input_tensor.shape)
                dtype = input_tensor.dtype
                
                return orchestrator.execute_with_profiling(
                    func,
                    operation_type,
                    shape,
                    dtype,
                    *args,
                    **kwargs,
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
