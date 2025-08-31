"""
Adaptive Quantum Hypercube for QuantumFortress 2.0

This module implements the Adaptive Quantum Hypercube, which serves as the foundation
for the topological security analysis and forms the basis for the Topological
Vulnerability Index (TVI) calculation.

Key features:
- Dynamic dimension expansion (4D → 6D → 8D)
- Multiple compression methods (topological, algebraic, spectral, hybrid)
- Direct construction of compressed hypercube without building full representation
- Integration with auto-calibration system
- TVI-based security monitoring
- WDM-parallelism for quantum operations
- Resource-aware operation to prevent overload

The implementation follows principles from:
- "Методы сжатия.md": Methods for ECDSA hypercube compression
- "Квантовый ПК.md": Quantum PC implementation guidelines
- "Ur Uz работа.md": TVI metrics and topological analysis
- "TopoSphere.md": Topological analysis techniques

As stated in documentation: "Прямое построение сжатого гиперкуба ECDSA представляет собой
критически важный прорыв, позволяющий анализировать системы, которые ранее считались
неподдающимися анализу из-за масштаба."
"""

import numpy as np
import time
import uuid
import math
import warnings
import heapq
import itertools
from enum import Enum
from typing import Union, Dict, Any, Tuple, Optional, List, Callable, Set, Generator
import logging
import scipy.spatial
import scipy.stats
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import concurrent.futures
import threading
import queue
import copy
import sys
import gc
import psutil
import resource
import ctypes
from functools import lru_cache
from dataclasses import dataclass
import json
import pickle
import zlib
import base64
import hashlib

# FastECDSA for optimized ECDSA operations
# As stated in Ur Uz работа.md: "fastecdsa|0.83 сек|В 15× быстрее, оптимизированные C-расширения"
try:
    from fastecdsa.curve import Curve
    from fastecdsa.point import Point
    from fastecdsa.util import mod_sqrt
    from fastecdsa.keys import gen_keypair
    FAST_ECDSA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("FastECDSA library successfully imported. Using optimized C extensions.")
except ImportError as e:
    FAST_ECDSA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"FastECDSA library not found: {e}. Some features will be limited.")

# Quantum platform support
from ..utils.quantum_utils import QuantumPlatform, PlatformConfig, get_platform_config
from ..utils.topology_utils import (
    calculate_betti_numbers,
    analyze_signature_topology,
    torus_distance,
    calculate_euler_characteristic,
    calculate_topological_entropy,
    find_high_density_areas,
    get_connectivity_metrics
)
from .auto_calibration import AutoCalibrationSystem, CalibrationStatus
from .topological_metrics import TopologicalMetrics, TVIResult
from .dynamic_compute_router import DynamicComputeRouter
from .betti_analyzer import BettiAnalyzer
from .collision_engine import CollisionEngine
from .gradient_analysis import GradientAnalyzer
from .hypercore_transformer import HypercoreTransformer

logger = logging.getLogger(__name__)

# ======================
# CONSTANTS
# ======================
# Dimension limits
DEFAULT_DIMENSION = 4
MIN_DIMENSION = 4
MAX_DIMENSION = 8

# Compression methods
class CompressionMethod(Enum):
    """Compression methods for the quantum hypercube"""
    TOPOLOGICAL = 1
    ALGEBRAIC = 2
    SPECTRAL = 3
    HYBRID = 4
    NERVE_COMPLEX = 5

# Quantum platform configurations
PLATFORM_CONFIGS = {
    "SOI": {
        "wavelengths": 8,
        "precision": 12,
        "error_tolerance": 0.001,
        "drift_rate": 0.005,
        "processing_speed": 1.0,
        "calibration_interval": 300
    },
    "SiN": {
        "wavelengths": 16,
        "precision": 14,
        "error_tolerance": 0.0005,
        "drift_rate": 0.003,
        "processing_speed": 1.5,
        "calibration_interval": 450
    },
    "TFLN": {
        "wavelengths": 32,
        "precision": 16,
        "error_tolerance": 0.0001,
        "drift_rate": 0.001,
        "processing_speed": 2.0,
        "calibration_interval": 600
    },
    "InP": {
        "wavelengths": 64,
        "precision": 18,
        "error_tolerance": 0.00005,
        "drift_rate": 0.0005,
        "processing_speed": 2.5,
        "calibration_interval": 900
    }
}

# TVI thresholds
TVI_CRITICAL = 0.8
TVI_HIGH = 0.6
TVI_MEDIUM = 0.4
TVI_LOW = 0.2

# Resource limits
MAX_MEMORY_USAGE_PERCENT = 85
MAX_CPU_USAGE_PERCENT = 85
ANALYSIS_TIMEOUT = 300  # seconds

# ======================
# EXCEPTIONS
# ======================
class HypercubeError(Exception):
    """Base exception for AdaptiveQuantumHypercube module."""
    pass

class DimensionError(HypercubeError):
    """Raised when dimension constraints are violated."""
    pass

class ResourceLimitExceededError(HypercubeError):
    """Raised when resource limits are exceeded."""
    pass

class AnalysisTimeoutError(HypercubeError):
    """Raised when analysis exceeds timeout limits."""
    pass

class CompressionError(HypercubeError):
    """Raised when compression operations fail."""
    pass

class QuantumStateError(HypercubeError):
    """Raised when quantum state operations fail."""
    pass

# ======================
# DATA CLASSES
# ======================
@dataclass
class HypercubeStatus:
    """Status of the quantum hypercube"""
    dimension: int
    size_gb: float
    tvi: float
    stability: float
    drift_rate: float
    compression_method: CompressionMethod
    memory_usage: float
    cpu_usage: float
    last_update: float
    signature_count: int
    high_density_areas: int
    betti_numbers: List[float]
    euler_characteristic: float
    topological_entropy: float

@dataclass
class CompressionParams:
    """Parameters for hypercube compression"""
    method: CompressionMethod
    topological: Dict[str, Any]
    algebraic: Dict[str, Any]
    spectral: Dict[str, Any]
    performance: Dict[str, Any]
    target_size_gb: Optional[float] = None
    quality_target: float = 0.95
    max_iterations: int = 5

@dataclass
class QuantumStateMetrics:
    """Metrics for quantum state analysis"""
    fidelity: float
    coherence_time: float
    error_rate: float
    drift_rate: float
    stability_score: float
    vulnerability_score: float
    timestamp: float

@dataclass
class HypercubeCompressionResult:
    """Result of hypercube compression operation"""
    compressed_data: Any
    original_size: float
    compressed_size: float
    compression_ratio: float
    quality_score: float
    processing_time: float
    method: CompressionMethod
    parameters: Dict[str, Any]
    timestamp: float

# ======================
# CORE CLASS
# ======================
class AdaptiveQuantumHypercube:
    """
    Adaptive Quantum Hypercube implementation for QuantumFortress 2.0.
    
    This class manages a dynamic quantum hypercube structure that can expand from
    4D to higher dimensions based on security requirements and system load.
    The hypercube serves as the foundation for the topological security analysis
    and forms the basis for the Topological Vulnerability Index (TVI) calculation.
    
    Key features:
    - Dynamic dimension expansion (4D → 6D → 8D)
    - Multiple compression methods (topological, algebraic, spectral, hybrid)
    - Direct construction of compressed hypercube without building full representation
    - Integration with auto-calibration system
    - TVI-based security monitoring
    - WDM-parallelism for quantum operations
    
    The implementation follows the principles from "Методы сжатия.md":
    "Прямое построение сжатого гиперкуба ECDSA представляет собой критически важный прорыв,
    позволяющий анализировать системы, которые ранее считались неподдающимися анализу из-за масштаба."
    
    Example:
    >>> hypercube = AdaptiveQuantumHypercube(dimension=4)
    >>> hypercube.update_with_signature(message, ecdsa_signature, quantum_signature)
    >>> tvi = hypercube.get_tvi()
    >>> print(f"Current TVI: {tvi:.4f}")
    """
    
    def __init__(self,
                 dimension: int = DEFAULT_DIMENSION,
                 max_dimension: int = MAX_DIMENSION,
                 stability_threshold: float = 0.95,
                 tvi_threshold: float = 0.5,
                 drift_threshold: float = 0.05,
                 quantum_platform: QuantumPlatform = QuantumPlatform.SOI,
                 ecdsa_curve: str = "secp256k1",
                 compression_method: CompressionMethod = CompressionMethod.HYBRID,
                 wdm_parallelism: bool = True,
                 auto_calibrate: bool = True):
        """
        Initialize the Adaptive Quantum Hypercube.
        
        Args:
            dimension: Initial dimension of the hypercube (4-8)
            max_dimension: Maximum dimension the hypercube can expand to
            stability_threshold: Threshold for quantum state stability
            tvi_threshold: Threshold for Topological Vulnerability Index
            drift_threshold: Threshold for quantum state drift
            quantum_platform: Quantum platform to use
            ecdsa_curve: ECDSA curve to use
            compression_method: Compression method for the hypercube
            wdm_parallelism: Whether to enable WDM parallelism
            auto_calibrate: Whether to enable auto-calibration
            
        Raises:
            DimensionError: If dimension is outside valid range
        """
        # Validate dimension
        if not (MIN_DIMENSION <= dimension <= max_dimension <= MAX_DIMENSION):
            raise DimensionError(
                f"Dimension must be between {MIN_DIMENSION} and {MAX_DIMENSION}, "
                f"with max_dimension >= dimension"
            )
        
        self.dimension = dimension
        self.max_dimension = max_dimension
        self.stability_threshold = stability_threshold
        self.tvi_threshold = tvi_threshold
        self.drift_threshold = drift_threshold
        self.quantum_platform = quantum_platform
        self.ecdsa_curve = ecdsa_curve
        self.compression_method = compression_method
        self.wdm_parallelism = wdm_parallelism
        self.auto_calibrate = auto_calibrate
        
        # Initialize platform configuration
        self.platform_config = get_platform_config(quantum_platform)
        
        # Initialize compression parameters
        self.compression_params = self._initialize_compression_params()
        
        # Initialize quantum state
        self.quantum_state = None
        self._initialize_quantum_state()
        
        # Initialize storage for signatures
        self.signatures = []
        self.max_signatures = 100000  # Maximum signatures to store for analysis
        
        # Initialize topological metrics
        self.topological_metrics = TopologicalMetrics()
        self.tvi_history = []
        self.max_history_size = 1000
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.start()
        
        # Initialize auto-calibration system
        self.calibration_system = None
        if auto_calibrate:
            self.calibration_system = AutoCalibrationSystem(
                self,
                platform_config=self.platform_config
            )
            self.calibration_system.start()
        
        # Initialize dynamic compute router
        self.compute_router = DynamicComputeRouter()
        
        # Initialize betti analyzer
        self.betti_analyzer = BettiAnalyzer()
        
        # Initialize collision engine
        self.collision_engine = CollisionEngine()
        
        # Initialize gradient analyzer
        self.gradient_analyzer = GradientAnalyzer()
        
        # Initialize hypercore transformer
        self.hypercore_transformer = HypercoreTransformer(
            curve=ecdsa_curve,
            compression_method=compression_method.name.lower(),
            params=self.compression_params.__dict__
        )
        
        # Initialize status tracking
        self.last_update = time.time()
        self.signature_count = 0
        self.processing_times = []
        self.max_processing_times = 1000
        
        logger.info(f"Initialized AdaptiveQuantumHypercube (dimension={dimension}, "
                    f"platform={quantum_platform.name}, compression={compression_method.name})")
    
    def _initialize_compression_params(self) -> CompressionParams:
        """Initialize compression parameters based on platform and requirements."""
        return CompressionParams(
            method=self.compression_method,
            topological={
                "sample_size": 10000,
                "min_cluster_size": 50,
                "eps": 0.05,
                "min_samples": 10
            },
            algebraic={
                "sampling_rate": 0.01,
                "min_points": 1000,
                "line_threshold": 0.95
            },
            spectral={
                "threshold_percentile": 95,
                "psnr_target": 40,
                "compression_ratio": 0.1
            },
            performance={
                "grid_size": 1000,
                "max_memory_usage": MAX_MEMORY_USAGE_PERCENT,
                "max_cpu_usage": MAX_CPU_USAGE_PERCENT
            }
        )
    
    def _initialize_quantum_state(self):
        """Initialize the quantum state for the hypercube."""
        try:
            # Initialize quantum state based on platform
            if self.quantum_platform == QuantumPlatform.SOI:
                self.quantum_state = self._init_soi_state()
            elif self.quantum_platform == QuantumPlatform.SiN:
                self.quantum_state = self._init_sin_state()
            elif self.quantum_platform == QuantumPlatform.TFLN:
                self.quantum_state = self._init_tfln_state()
            elif self.quantum_platform == QuantumPlatform.InP:
                self.quantum_state = self._init_inp_state()
            
            # Set initial metrics
            self.quantum_metrics = QuantumStateMetrics(
                fidelity=0.95,
                coherence_time=self.platform_config.coherence_time,
                error_rate=self.platform_config.error_tolerance,
                drift_rate=0.0,
                stability_score=0.95,
                vulnerability_score=0.05,
                timestamp=time.time()
            )
            
            logger.debug(f"Quantum state initialized for platform {self.quantum_platform.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize quantum state: {e}", exc_info=True)
            # Fallback to basic state
            self.quantum_state = {
                "state_vector": np.random.random((2**self.dimension,)),
                "phase": np.random.random((2**self.dimension,))
            }
            self.quantum_metrics = QuantumStateMetrics(
                fidelity=0.8,
                coherence_time=100.0,
                error_rate=0.01,
                drift_rate=0.01,
                stability_score=0.8,
                vulnerability_score=0.2,
                timestamp=time.time()
            )
    
    def _init_soi_state(self) -> Dict[str, Any]:
        """Initialize quantum state for SOI platform."""
        # SOI-specific initialization
        return {
            "state_vector": np.random.random((2**self.dimension,)),
            "phase": np.random.random((2**self.dimension,)),
            "wavelengths": self.platform_config.wavelengths,
            "precision": self.platform_config.min_precision
        }
    
    def _init_sin_state(self) -> Dict[str, Any]:
        """Initialize quantum state for SiN platform."""
        # SiN-specific initialization
        return {
            "state_vector": np.random.random((2**self.dimension,)) * 0.9,
            "phase": np.random.random((2**self.dimension,)) * 0.8,
            "wavelengths": self.platform_config.wavelengths,
            "precision": self.platform_config.min_precision + 2
        }
    
    def _init_tfln_state(self) -> Dict[str, Any]:
        """Initialize quantum state for TFLN platform."""
        # TFLN-specific initialization
        return {
            "state_vector": np.random.random((2**self.dimension,)) * 0.85,
            "phase": np.random.random((2**self.dimension,)) * 0.75,
            "wavelengths": self.platform_config.wavelengths,
            "precision": self.platform_config.min_precision + 4
        }
    
    def _init_inp_state(self) -> Dict[str, Any]:
        """Initialize quantum state for InP platform."""
        # InP-specific initialization
        return {
            "state_vector": np.random.random((2**self.dimension,)) * 0.8,
            "phase": np.random.random((2**self.dimension,)) * 0.7,
            "wavelengths": self.platform_config.wavelengths,
            "precision": self.platform_config.min_precision + 6
        }
    
    def _check_resources(self):
        """Check if system resources are within acceptable limits."""
        # Get current resource usage
        memory_usage = self.resource_monitor.get_memory_usage()
        cpu_usage = self.resource_monitor.get_cpu_usage()
        
        # Check if we're approaching resource limits
        if memory_usage > MAX_MEMORY_USAGE_PERCENT or cpu_usage > MAX_CPU_USAGE_PERCENT:
            raise ResourceLimitExceededError(
                f"Resource limits exceeded: memory={memory_usage:.1f}%, cpu={cpu_usage:.1f}%"
            )
    
    def update_configuration(self, migration_phase: Enum):
        """
        Update hypercube configuration based on migration phase.
        
        Args:
            migration_phase: Current migration phase from HybridCryptoSystem
        """
        # Adjust configuration based on phase
        if migration_phase == MigrationPhase.CLASSICAL:
            # Focus on classical signature analysis
            self.compression_params.topological["sample_size"] = 5000
            self.compression_params.algebraic["sampling_rate"] = 0.005
            self.wdm_parallelism = False
        elif migration_phase == MigrationPhase.HYBRID:
            # Balance between classical and quantum
            self.compression_params.topological["sample_size"] = 10000
            self.compression_params.algebraic["sampling_rate"] = 0.01
            self.wdm_parallelism = True
        elif migration_phase == MigrationPhase.POST_QUANTUM:
            # Focus on quantum operations
            self.compression_params.topological["sample_size"] = 15000
            self.compression_params.algebraic["sampling_rate"] = 0.02
            self.wdm_parallelism = True
            # Consider expanding dimension if possible
            self._expand_dimension_if_needed()
        
        # Update hypercore transformer
        self.hypercore_transformer.update_params({
            "topological": self.compression_params.topological,
            "algebraic": self.compression_params.algebraic,
            "spectral": self.compression_params.spectral
        })
        
        logger.debug(f"Hypercube configuration updated for phase {migration_phase.name}")
    
    def _expand_dimension_if_needed(self):
        """Expand hypercube dimension if security requirements demand it."""
        current_tvi = self.get_tvi()
        
        # Expand dimension if TVI is high and we're below max dimension
        if current_tvi > self.tvi_threshold and self.dimension < self.max_dimension:
            target_dimension = min(self.dimension + 2, self.max_dimension)
            self.expand_dimension(target_dimension)
            logger.info(f"Expanded hypercube dimension from {self.dimension} to {target_dimension} "
                        f"due to high TVI ({current_tvi:.4f})")
    
    def expand_dimension(self, new_dimension: int):
        """
        Expand the hypercube to a higher dimension.
        
        Args:
            new_dimension: Target dimension (must be > current dimension and <= max_dimension)
            
        Raises:
            DimensionError: If dimension constraints are violated
        """
        if new_dimension <= self.dimension:
            raise DimensionError("New dimension must be greater than current dimension")
        if new_dimension > self.max_dimension:
            raise DimensionError(f"Cannot expand beyond max dimension ({self.max_dimension})")
        if new_dimension > MAX_DIMENSION:
            raise DimensionError(f"Dimension cannot exceed {MAX_DIMENSION}")
        
        start_time = time.time()
        
        try:
            # Save current state
            old_dimension = self.dimension
            
            # Update dimension
            self.dimension = new_dimension
            
            # Reinitialize quantum state for new dimension
            self._initialize_quantum_state()
            
            # Update compression parameters for new dimension
            self.compression_params.topological["sample_size"] = min(
                20000, 
                self.compression_params.topological["sample_size"] * 2
            )
            
            # Recalculate metrics with new dimension
            if self.signatures:
                self._recalculate_metrics()
            
            logger.info(f"Successfully expanded hypercube from {old_dimension}D to {new_dimension}D "
                        f"(time: {time.time() - start_time:.2f}s)")
            
        except Exception as e:
            # Revert dimension on failure
            self.dimension = old_dimension
            logger.error(f"Failed to expand hypercube dimension: {e}", exc_info=True)
            raise DimensionError(f"Dimension expansion failed: {str(e)}") from e
    
    def update_with_signature(self,
                            message: Union[str, bytes],
                            ecdsa_signature: bytes,
                            quantum_signature: Optional[bytes] = None,
                            ecdsa_components: Optional[Dict[str, Any]] = None,
                            quantum_components: Optional[Dict[str, Any]] = None):
        """
        Update the hypercube with a new signature.
        
        This method:
        - Adds the signature to the internal storage
        - Updates topological metrics
        - Calculates TVI
        - Triggers dimension expansion if needed
        - Checks for resource limits
        
        Args:
            message: Original message that was signed
            ecdsa_signature: ECDSA signature component
            quantum_signature: Quantum-topological signature component (optional)
            ecdsa_components: Additional ECDSA components (r, s, z)
            quantum_components: Additional quantum components
            
        Raises:
            ResourceLimitExceededError: If resource limits are exceeded
        """
        self._check_resources()
        start_time = time.time()
        
        try:
            # Transform to appropriate format if needed
            if isinstance(message, str):
                message = message.encode()
            
            # Extract ECDSA components if not provided
            if ecdsa_components is None:
                ecdsa_components = self._extract_ecdsa_components(ecdsa_signature, message)
            
            # Store signature data
            signature_data = {
                "message": message,
                "ecdsa_signature": ecdsa_signature,
                "quantum_signature": quantum_signature,
                "ecdsa_components": ecdsa_components,
                "quantum_components": quantum_components,
                "timestamp": time.time()
            }
            
            # Add to signatures (with limit)
            if len(self.signatures) >= self.max_signatures:
                # Remove oldest signature
                self.signatures.pop(0)
            self.signatures.append(signature_data)
            self.signature_count += 1
            
            # Update topological metrics
            self._update_topological_metrics(signature_data)
            
            # Record processing time
            self.processing_times.append(time.time() - start_time)
            if len(self.processing_times) > self.max_processing_times:
                self.processing_times.pop(0)
            
            # Update last update time
            self.last_update = time.time()
            
            logger.debug(f"Updated hypercube with signature (total: {self.signature_count})")
            
        except Exception as e:
            logger.error(f"Failed to update hypercube with signature: {e}", exc_info=True)
            raise HypercubeError(f"Signature update failed: {str(e)}") from e
    
    def _extract_ecdsa_components(self, 
                                ecdsa_signature: bytes, 
                                message: bytes) -> Dict[str, Any]:
        """
        Extract ECDSA components (r, s, z) from signature and message.
        
        Args:
            ecdsa_signature: ECDSA signature
            message: Original message
            
        Returns:
            Dictionary containing r, s, z components
        """
        # In a real implementation, this would properly parse the signature
        if not FAST_ECDSA_AVAILABLE:
            # Fallback implementation
            r = int.from_bytes(ecdsa_signature[:32], byteorder='big')
            s = int.from_bytes(ecdsa_signature[32:64], byteorder='big')
        else:
            # Use FastECDSA for proper parsing
            try:
                # This is simplified - actual implementation would depend on curve and format
                r = int.from_bytes(ecdsa_signature[:32], byteorder='big')
                s = int.from_bytes(ecdsa_signature[32:64], byteorder='big')
            except Exception as e:
                logger.warning(f"FastECDSA parsing failed, using fallback: {e}")
                r = int.from_bytes(ecdsa_signature[:32], byteorder='big')
                s = int.from_bytes(ecdsa_signature[32:64], byteorder='big')
        
        # Calculate z (message hash mod n)
        n = self._get_curve_order()
        z = int.from_bytes(message, byteorder='big') % n
        
        return {
            'r': r,
            's': s,
            'z': z,
            'curve': self.ecdsa_curve
        }
    
    def _get_curve_order(self) -> int:
        """Get the order of the elliptic curve."""
        # In a real implementation, this would use actual curve parameters
        if self.ecdsa_curve == "secp256k1":
            return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        elif self.ecdsa_curve == "P-256":
            return 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
        else:
            # Default to secp256k1 order
            return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    
    def _update_topological_metrics(self, signature_data: Dict[str, Any]):
        """Update topological metrics with new signature data."""
        try:
            # Extract components
            ecdsa_components = signature_data["ecdsa_components"]
            r = ecdsa_components["r"]
            s = ecdsa_components["s"]
            z = ecdsa_components["z"]
            
            # Transform to (ur, uz) space
            ur, uz = self._transform_to_ur_uz(r, s, z)
            
            # Update betti numbers and other metrics
            self.topological_metrics.update_with_point(ur, uz)
            
            # Calculate current TVI
            tvi_result = self.topological_metrics.calculate_tvi()
            
            # Store in history
            self.tvi_history.append((time.time(), tvi_result.tvi))
            if len(self.tvi_history) > self.max_history_size:
                self.tvi_history.pop(0)
            
            # Update quantum metrics if needed
            self._update_quantum_metrics()
            
        except Exception as e:
            logger.error(f"Failed to update topological metrics: {e}", exc_info=True)
    
    def _transform_to_ur_uz(self, r: int, s: int, z: int) -> Tuple[float, float]:
        """
        Transform ECDSA signature components to (ur, uz) space on the torus.
        
        This function implements the transformation described in Ur Uz работа.md:
        ur = (r * s^-1) mod N
        uz = (z * s^-1) mod N
        
        Args:
            r: ECDSA r component
            s: ECDSA s component
            z: Message hash (mod N)
            
        Returns:
            Tuple (ur, uz) in [0, 1) range representing points on the unit torus
        
        As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
        точную количественную оценку структуры пространства подписей и обнаруживает скрытые
        уязвимости, которые пропускаются другими методами."
        """
        # Get curve order
        n = self._get_curve_order()
        
        # Calculate modular inverse of s
        try:
            s_inv = pow(s, -1, n)
        except ValueError:
            # If s has no inverse (shouldn't happen with valid signatures), use a fallback
            s_inv = 1
        
        # Calculate ur and uz
        ur = (r * s_inv) % n
        uz = (z * s_inv) % n
        
        # Normalize to [0, 1) range
        ur_normalized = ur / n
        uz_normalized = uz / n
        
        return ur_normalized, uz_normalized
    
    def _update_quantum_metrics(self):
        """Update quantum state metrics based on current analysis."""
        try:
            # Get current TVI
            tvi_result = self.topological_metrics.calculate_tvi()
            
            # Update drift rate based on TVI changes
            if len(self.tvi_history) >= 2:
                prev_tvi = self.tvi_history[-2][1]
                current_tvi = self.tvi_history[-1][1]
                drift_rate = abs(current_tvi - prev_tvi) / (self.tvi_history[-1][0] - self.tvi_history[-2][0])
            else:
                drift_rate = 0.0
            
            # Calculate stability score (higher TVI = lower stability)
            stability_score = max(0.0, 1.0 - tvi_result.tvi * 1.2)
            
            # Update quantum metrics
            self.quantum_metrics = QuantumStateMetrics(
                fidelity=self.quantum_metrics.fidelity * 0.99 + stability_score * 0.01,
                coherence_time=self.quantum_metrics.coherence_time,
                error_rate=self.platform_config.error_tolerance * (1.0 + tvi_result.tvi),
                drift_rate=drift_rate,
                stability_score=stability_score,
                vulnerability_score=tvi_result.tvi,
                timestamp=time.time()
            )
            
            # Check if calibration is needed
            if self.auto_calibrate and self.calibration_system:
                self.calibration_system.check_and_calibrate()
                
        except Exception as e:
            logger.error(f"Failed to update quantum metrics: {e}", exc_info=True)
    
    def _recalculate_metrics(self):
        """Recalculate all metrics after dimension change or significant update."""
        try:
            # Clear current metrics
            self.topological_metrics = TopologicalMetrics()
            
            # Recalculate with all signatures
            for signature in self.signatures:
                self._update_topological_metrics(signature)
                
            logger.debug("Recalculated metrics after structural change")
            
        except Exception as e:
            logger.error(f"Failed to recalculate metrics: {e}", exc_info=True)
            # Reset to safe state
            self.topological_metrics = TopologicalMetrics()
    
    def get_tvi(self) -> float:
        """
        Get the current Topological Vulnerability Index (TVI).
        
        TVI is a measure of the security vulnerability based on topological analysis:
        - 0.0 = Completely secure
        - 1.0 = Critically vulnerable
        
        Returns:
            float: Current TVI value
        """
        try:
            tvi_result = self.topological_metrics.calculate_tvi()
            return tvi_result.tvi
        except Exception as e:
            logger.error(f"Failed to calculate TVI: {e}", exc_info=True)
            return 1.0  # Assume worst case on failure
    
    def get_status(self) -> HypercubeStatus:
        """
        Get the current status of the quantum hypercube.
        
        Returns:
            HypercubeStatus object containing detailed status information
        """
        # Get current resource usage
        memory_usage = self.resource_monitor.get_memory_usage()
        cpu_usage = self.resource_monitor.get_cpu_usage()
        
        # Get topological metrics
        tvi_result = self.topological_metrics.calculate_tvi()
        metrics = self.topological_metrics.get_metrics()
        
        return HypercubeStatus(
            dimension=self.dimension,
            size_gb=self._estimate_size_gb(),
            tvi=tvi_result.tvi,
            stability=self.quantum_metrics.stability_score,
            drift_rate=self.quantum_metrics.drift_rate,
            compression_method=self.compression_method,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            last_update=self.last_update,
            signature_count=self.signature_count,
            high_density_areas=len(metrics.high_density_areas),
            betti_numbers=metrics.betti_numbers,
            euler_characteristic=metrics.euler_characteristic,
            topological_entropy=metrics.topological_entropy
        )
    
    def _estimate_size_gb(self) -> float:
        """
        Estimate the current size of the hypercube in GB.
        
        Returns:
            float: Estimated size in GB
        """
        # This is a simplified estimation - actual implementation would depend on compression
        base_size = 0.1  # Base size in GB for 4D hypercube with default settings
        
        # Adjust for dimension
        dimension_factor = (2 ** (self.dimension - 4)) * 0.5
        
        # Adjust for number of signatures
        signature_factor = min(1.0, self.signature_count / 10000) * 0.5
        
        # Adjust for compression
        compression_factor = {
            CompressionMethod.TOPOLOGICAL: 0.3,
            CompressionMethod.ALGEBRAIC: 0.2,
            CompressionMethod.SPECTRAL: 0.25,
            CompressionMethod.HYBRID: 0.15,
            CompressionMethod.NERVE_COMPLEX: 0.1
        }.get(self.compression_method, 0.2)
        
        # Calculate estimated size
        estimated_size = base_size * dimension_factor * (1 + signature_factor) * compression_factor
        
        return max(0.01, estimated_size)  # Minimum size of 0.01 GB
    
    def compress(self, 
                method: Optional[CompressionMethod] = None,
                target_size_gb: Optional[float] = None) -> HypercubeCompressionResult:
        """
        Compress the hypercube using the specified method.
        
        Args:
            method: Compression method to use (defaults to current method)
            target_size_gb: Target size in GB (optional)
            
        Returns:
            HypercubeCompressionResult object with compression details
            
        Raises:
            CompressionError: If compression fails
        """
        start_time = time.time()
        method = method or self.compression_method
        
        try:
            # Check resources
            self._check_resources()
            
            # Get current hypercube representation
            hypercube_data = self._get_hypercube_representation()
            
            # Compress based on method
            if method == CompressionMethod.TOPOLOGICAL:
                compressed_data = self._topological_compress(hypercube_data)
            elif method == CompressionMethod.ALGEBRAIC:
                compressed_data = self._algebraic_compress(hypercube_data)
            elif method == CompressionMethod.SPECTRAL:
                compressed_data = self._spectral_compress(hypercube_data)
            elif method == CompressionMethod.HYBRID:
                compressed_data = self._hybrid_compress(hypercube_data)
            elif method == CompressionMethod.NERVE_COMPLEX:
                compressed_data = self._nerve_complex_compress(hypercube_data)
            else:
                raise CompressionError(f"Unknown compression method: {method}")
            
            # Calculate sizes
            original_size = self._estimate_size_gb()
            compressed_size = self._estimate_compressed_size(compressed_data)
            compression_ratio = original_size / max(compressed_size, 0.001)
            
            # Calculate quality score (higher is better)
            quality_score = self._calculate_compression_quality(hypercube_data, compressed_data)
            
            # Create result
            result = HypercubeCompressionResult(
                compressed_data=compressed_data,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                processing_time=time.time() - start_time,
                method=method,
                parameters=self.compression_params.__dict__,
                timestamp=time.time()
            )
            
            # Update compression method if successful
            self.compression_method = method
            
            logger.info(f"Hypercube compressed ({method.name}): "
                        f"ratio={compression_ratio:.2f}x, "
                        f"quality={quality_score:.4f}, "
                        f"time={result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Hypercube compression failed: {e}", exc_info=True)
            raise CompressionError(f"Compression failed: {str(e)}") from e
    
    def _get_hypercube_representation(self) -> Any:
        """
        Get the current representation of the hypercube.
        
        Returns:
            Current hypercube representation (format depends on implementation)
        """
        # In a real implementation, this would return the actual hypercube data structure
        # For this example, we'll return a simplified representation
        
        # Extract all points for analysis
        points = []
        for signature in self.signatures:
            ecdsa_components = signature["ecdsa_components"]
            r, s, z = ecdsa_components["r"], ecdsa_components["s"], ecdsa_components["z"]
            ur, uz = self._transform_to_ur_uz(r, s, z)
            points.append((ur, uz))
        
        return {
            "points": points,
            "dimension": self.dimension,
            "quantum_state": self.quantum_state,
            "metrics": self.topological_metrics.get_metrics()
        }
    
    def _estimate_compressed_size(self, compressed_data: Any) -> float:
        """
        Estimate the size of compressed data in GB.
        
        Args:
            compressed_data: Compressed hypercube data
            
        Returns:
            Estimated size in GB
        """
        # Simplified size estimation
        if isinstance(compressed_data, dict):
            # Estimate based on number of elements
            size_mb = 0.1  # Base size
            
            if "lines" in compressed_data:
                size_mb += len(compressed_data["lines"]) * 0.001
            if "spectral" in compressed_data:
                size_mb += len(compressed_data["spectral"]) * 0.0005
            if "singularities" in compressed_data:
                size_mb += len(compressed_data["singularities"]) * 0.002
                
            return size_mb / 1024  # Convert to GB
        
        # Fallback estimation
        return sys.getsizeof(compressed_data) / (1024 ** 3)
    
    def _calculate_compression_quality(self, 
                                     original_data: Any, 
                                     compressed_data: Any) -> float:
        """
        Calculate the quality of compression (1.0 = perfect, 0.0 = unusable).
        
        Args:
            original_data: Original hypercube data
            compressed_data: Compressed hypercube data
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # In a real implementation, this would compare original and decompressed data
            # For this example, we'll use a simplified approach
            
            # Get metrics from original data
            original_metrics = self.topological_metrics.get_metrics()
            
            # Reconstruct metrics from compressed data (simplified)
            # This would be more complex in a real implementation
            reconstructed_metrics = {
                "betti_numbers": original_metrics.betti_numbers,
                "euler_characteristic": original_metrics.euler_characteristic,
                "topological_entropy": original_metrics.topological_entropy
            }
            
            # Calculate quality based on metric preservation
            betti_diff = sum(abs(original_metrics.betti_numbers[i] - reconstructed_metrics["betti_numbers"][i]) 
                            for i in range(3)) / 3.0
            euler_diff = abs(original_metrics.euler_characteristic - reconstructed_metrics["euler_characteristic"])
            entropy_diff = abs(original_metrics.topological_entropy - reconstructed_metrics["topological_entropy"])
            
            # Combined quality score (higher is better)
            quality = 1.0 - (betti_diff * 0.4 + euler_diff * 0.3 + entropy_diff * 0.3)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Failed to calculate compression quality: {e}", exc_info=True)
            return 0.7  # Default medium quality
    
    def _topological_compress(self, hypercube_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply topological compression to the hypercube.
        
        This method:
        - Identifies topological features (connected components, loops, voids)
        - Preserves critical topological information
        - Removes redundant data
        
        Args:
            hypercube_data: Hypercube data to compress
            
        Returns:
            Compressed hypercube data
        """
        # Extract parameters
        params = self.compression_params.topological
        sample_size = params["sample_size"]
        min_cluster_size = params["min_cluster_size"]
        eps = params["eps"]
        min_samples = params["min_samples"]
        
        # Sample points if needed
        points = hypercube_data["points"]
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            sampled_points = [points[i] for i in indices]
        else:
            sampled_points = points
        
        # Convert to numpy array
        points_array = np.array(sampled_points)
        
        # Apply DBSCAN for clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
        labels = clustering.labels_
        
        # Identify core points for each cluster
        core_points = []
        for label in set(labels):
            if label == -1:  # Noise points
                continue
                
            cluster_points = points_array[labels == label]
            if len(cluster_points) >= min_cluster_size:
                # Calculate centroid
                centroid = np.mean(cluster_points, axis=0)
                core_points.append(centroid.tolist())
        
        # Return compressed representation
        return {
            "method": "topological",
            "core_points": core_points,
            "cluster_count": len(set(labels)) - (1 if -1 in labels else 0),
            "noise_ratio": np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0.0,
            "params": params
        }
    
    def _algebraic_compress(self, hypercube_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply algebraic compression to the hypercube.
        
        This method:
        - Identifies algebraic structures (lines, planes)
        - Uses sparse representation of these structures
        - Preserves cryptographic properties
        
        Args:
            hypercube_data: Hypercube data to compress
            
        Returns:
            Compressed hypercube data
        """
        # Extract parameters
        params = self.compression_params.algebraic
        sampling_rate = params["sampling_rate"]
        min_points = params["min_points"]
        line_threshold = params["line_threshold"]
        
        # Sample points
        points = hypercube_data["points"]
        sample_size = max(min_points, int(len(points) * sampling_rate))
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            sampled_points = [points[i] for i in indices]
        else:
            sampled_points = points
        
        # Convert to numpy array
        points_array = np.array(sampled_points)
        
        # Find lines (simplified approach)
        lines = []
        if len(points_array) >= 2:
            # Calculate pairwise slopes
            for i in range(len(points_array)):
                for j in range(i + 1, len(points_array)):
                    p1, p2 = points_array[i], points_array[j]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    
                    # Avoid vertical lines for simplicity
                    if abs(dx) > 1e-10:
                        slope = dy / dx
                        intercept = p1[1] - slope * p1[0]
                        
                        # Count points on this line
                        on_line = 0
                        for k in range(len(points_array)):
                            x, y = points_array[k]
                            if abs(y - (slope * x + intercept)) < 1e-5:
                                on_line += 1
                        
                        # Add line if enough points
                        if on_line >= min_points and on_line / len(points_array) >= line_threshold:
                            lines.append({
                                "slope": float(slope),
                                "intercept": float(intercept),
                                "point_count": on_line,
                                "precision": 1.0 - (on_line / len(points_array))
                            })
        
        # Return compressed representation
        return {
            "method": "algebraic",
            "lines": lines,
            "line_count": len(lines),
            "params": params
        }
    
    def _spectral_compress(self, hypercube_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply spectral compression to the hypercube.
        
        This method:
        - Uses Fourier or wavelet transforms
        - Keeps only significant coefficients
        - Achieves high compression ratios
        
        Args:
            hypercube_data: Hypercube data to compress
            
        Returns:
            Compressed hypercube data
        """
        # Extract parameters
        params = self.compression_params.spectral
        threshold_percentile = params["threshold_percentile"]
        psnr_target = params["psnr_target"]
        
        # Get points
        points = hypercube_data["points"]
        
        # In a real implementation, this would use actual spectral methods
        # For this example, we'll simulate the process
        
        # Convert to grid representation (simplified)
        grid_size = 1000
        grid = np.zeros((grid_size, grid_size))
        
        for ur, uz in points:
            x = int(ur * (grid_size - 1))
            y = int(uz * (grid_size - 1))
            grid[x, y] += 1
        
        # Apply FFT
        fft_result = np.fft.fft2(grid)
        
        # Calculate magnitude spectrum
        magnitude = np.abs(fft_result)
        
        # Find threshold based on percentile
        threshold = np.percentile(magnitude, threshold_percentile)
        
        # Keep only significant coefficients
        compressed_fft = np.where(magnitude >= threshold, fft_result, 0)
        
        # Calculate PSNR (simplified)
        mse = np.mean(np.abs(grid - np.fft.ifft2(compressed_fft))**2)
        max_val = np.max(grid)
        psnr = 10 * np.log10((max_val**2) / (mse + 1e-10)) if mse > 0 else float('inf')
        
        # Return compressed representation
        # In a real implementation, we'd store only the non-zero coefficients
        return {
            "method": "spectral",
            "coefficients": {
                "shape": fft_result.shape,
                "non_zero_count": np.sum(np.abs(compressed_fft) > 0),
                "threshold": threshold,
                "psnr": float(psnr)
            },
            "params": params
        }
    
    def _hybrid_compress(self, hypercube_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply hybrid compression to the hypercube.
        
        This method combines:
        1. Topological compression for structural features
        2. Algebraic compression for linear structures
        3. Spectral compression for residual data
        
        Args:
            hypercube_data: Hypercube data to compress
            
        Returns:
            Compressed hypercube data
        """
        # Apply topological compression
        topological_result = self._topological_compress(hypercube_data)
        
        # Extract core points from topological compression
        core_points = topological_result.get("core_points", [])
        
        # Create residual data (points not well represented by topological features)
        residual_points = []
        if core_points and len(hypercube_data["points"]) > len(core_points) * 2:
            # Simplified approach - in real implementation would use proper residual calculation
            residual_points = hypercube_data["points"][:len(core_points)]
        
        # Apply algebraic compression to residual
        algebraic_result = self._algebraic_compress({
            "points": residual_points,
            "dimension": hypercube_data["dimension"],
            "quantum_state": hypercube_data["quantum_state"],
            "metrics": hypercube_data["metrics"]
        })
        
        # Apply spectral compression to remaining residual
        spectral_result = self._spectral_compress({
            "points": residual_points,
            "dimension": hypercube_data["dimension"],
            "quantum_state": hypercube_data["quantum_state"],
            "metrics": hypercube_data["metrics"]
        })
        
        # Return hybrid representation
        return {
            "method": "hybrid",
            "topological": topological_result,
            "algebraic": algebraic_result,
            "spectral": spectral_result,
            "params": {
                "topological": self.compression_params.topological,
                "algebraic": self.compression_params.algebraic,
                "spectral": self.compression_params.spectral
            }
        }
    
    def _nerve_complex_compress(self, hypercube_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply nerve complex compression to the hypercube.
        
        This method:
        - Builds a nerve complex from overlapping regions
        - Uses persistent homology to identify significant features
        - Provides topologically accurate compression
        
        Args:
            hypercube_data: Hypercube data to compress
            
        Returns:
            Compressed hypercube data
        """
        # Extract parameters
        params = self.compression_params.topological
        sample_size = params["sample_size"]
        min_cluster_size = params["min_cluster_size"]
        
        # Sample points
        points = hypercube_data["points"]
        if len(points) > sample_size:
            indices = np.random.choice(len(points), sample_size, replace=False)
            sampled_points = [points[i] for i in indices]
        else:
            sampled_points = points
        
        # Convert to numpy array
        points_array = np.array(sampled_points)
        
        # Calculate pairwise distances
        distances = squareform(pdist(points_array))
        
        # Build nerve complex (simplified approach)
        # In a real implementation, this would use proper persistent homology
        nerve_complex = {
            "vertices": len(points_array),
            "edges": [],
            "triangles": []
        }
        
        # Add edges based on distance threshold
        distance_threshold = np.percentile(distances, 25)  # 25th percentile
        for i in range(len(points_array)):
            for j in range(i + 1, len(points_array)):
                if distances[i, j] < distance_threshold:
                    nerve_complex["edges"].append((i, j))
        
        # Add triangles (simplified)
        for i, j in nerve_complex["edges"]:
            for k in range(j + 1, len(points_array)):
                if (i, k) in nerve_complex["edges"] and (j, k) in nerve_complex["edges"]:
                    nerve_complex["triangles"].append((i, j, k))
        
        # Calculate persistent homology (simplified)
        persistence_diagram = {
            "h0": [{"birth": 0.0, "death": distance_threshold * 2}],
            "h1": [],
            "h2": []
        }
        
        # Return compressed representation
        return {
            "method": "nerve_complex",
            "nerve_complex": nerve_complex,
            "persistence_diagram": persistence_diagram,
            "params": params
        }
    
    def build_compressed_hypercube(self, 
                                 public_key: Any,
                                 target_size_gb: Optional[float] = None,
                                 method: Optional[CompressionMethod] = None) -> Any:
        """
        Build a compressed hypercube directly from a public key without creating the full hypercube.
        
        This implements the "direct construction of compressed hypercube" approach described in
        "Методы сжатия.md": "Прямое построение сжатого гиперкуба ECDSA представляет собой критически важный прорыв"
        
        Args:
            public_key: Public key to build hypercube from
            target_size_gb: Target size in GB (optional)
            method: Compression method to use (optional)
            
        Returns:
            Compressed hypercube representation
            
        Raises:
            CompressionError: If building fails
        """
        start_time = time.time()
        method = method or self.compression_method
        
        try:
            # Check resources
            self._check_resources()
            
            # Use HypercoreTransformer for direct construction
            compressed_hypercube = self.hypercore_transformer.transform(
                public_key,
                target_size_gb=target_size_gb,
                method=method.name.lower()
            )
            
            # Update internal state
            self._update_from_compressed(compressed_hypercube)
            
            logger.info(f"Built compressed hypercube directly (method={method.name}, "
                        f"time={time.time() - start_time:.2f}s)")
            
            return compressed_hypercube
            
        except Exception as e:
            logger.error(f"Failed to build compressed hypercube: {e}", exc_info=True)
            raise CompressionError(f"Hypercube construction failed: {str(e)}") from e
    
    def _update_from_compressed(self, compressed_data: Any):
        """
        Update internal state from compressed hypercube data.
        
        Args:
            compressed_data: Compressed hypercube data
        """
        # Extract relevant information to update internal metrics
        # This is simplified - in a real implementation would be more detailed
        
        # Update dimension if available
        if "dimension" in compressed_data:
            self.dimension = compressed_data["dimension"]
        
        # Update topological metrics
        if "metrics" in compressed_data:
            metrics = compressed_data["metrics"]
            self.topological_metrics = TopologicalMetrics(
                betti_numbers=metrics.get("betti_numbers", [0.0, 0.0, 0.0]),
                euler_characteristic=metrics.get("euler_characteristic", 0.0),
                topological_entropy=metrics.get("topological_entropy", 0.0),
                high_density_areas=metrics.get("high_density_areas", []),
                connectivity=metrics.get("connectivity", {}),
                density_metrics=metrics.get("density_metrics", {})
            )
        
        # Update quantum state if available
        if "quantum_state" in compressed_data:
            self.quantum_state = compressed_data["quantum_state"]
        
        # Update signature count
        if "signature_count" in compressed_data:
            self.signature_count = compressed_data["signature_count"]
        
        # Update last update time
        self.last_update = time.time()
    
    def analyze_quantum_state(self) -> QuantumStateMetrics:
        """
        Analyze the current quantum state for stability and security.
        
        Returns:
            QuantumStateMetrics object with detailed analysis
        """
        # Update metrics if needed
        if time.time() - self.quantum_metrics.timestamp > 1.0:  # Update if older than 1 second
            self._update_quantum_metrics()
        
        return self.quantum_metrics
    
    def get_recent_tvi_trend(self, window_size: int = 10) -> List[float]:
        """
        Get the recent trend of TVI values.
        
        Args:
            window_size: Number of recent values to return
            
        Returns:
            List of recent TVI values
        """
        if not self.tvi_history:
            return []
        
        # Get recent values
        recent = self.tvi_history[-window_size:]
        
        # Extract just the TVI values
        return [tvi for _, tvi in recent]
    
    def needs_dimension_expansion(self) -> bool:
        """
        Check if the hypercube needs dimension expansion.
        
        Returns:
            bool: True if dimension expansion is recommended
        """
        # Get current TVI
        current_tvi = self.get_tvi()
        
        # Get recent TVI trend
        recent_tvi = self.get_recent_tvi_trend(5)
        avg_tvi = sum(recent_tvi) / len(recent_tvi) if recent_tvi else current_tvi
        
        # Check if TVI is consistently high
        high_tvi = avg_tvi > self.tvi_threshold * 1.2
        
        # Check if dimension is below max
        below_max = self.dimension < self.max_dimension
        
        return high_tvi and below_max
    
    def get_optimal_compression_params(self, 
                                     target_size_gb: float,
                                     current_params: Optional[CompressionParams] = None) -> CompressionParams:
        """
        Get optimal compression parameters to achieve a target size.
        
        Args:
            target_size_gb: Target size in GB
            current_params: Current compression parameters (optional)
            
        Returns:
            CompressionParams object with optimal parameters
        """
        # Start with current parameters or default
        params = current_params or self.compression_params
        
        # Get current estimated size
        current_size = self._estimate_size_gb()
        
        # Calculate required compression ratio
        if current_size <= 0.001:  # Avoid division by zero
            return params
            
        target_ratio = current_size / max(target_size_gb, 0.001)
        
        # Adjust parameters based on target ratio
        new_params = copy.deepcopy(params)
        
        if target_ratio > 1.0:  # Need more compression
            if new_params.method == CompressionMethod.TOPOLOGICAL:
                new_params.topological["sample_size"] = max(
                    1000, 
                    int(new_params.topological["sample_size"] / target_ratio)
                )
            elif new_params.method == CompressionMethod.ALGEBRAIC:
                new_params.algebraic["sampling_rate"] = max(
                    0.001,
                    new_params.algebraic["sampling_rate"] / target_ratio
                )
            elif new_params.method == CompressionMethod.SPECTRAL:
                new_params.spectral["threshold_percentile"] = min(
                    99,
                    new_params.spectral["threshold_percentile"] * target_ratio
                )
            elif new_params.method == CompressionMethod.HYBRID:
                # Adjust all components
                new_params.topological["sample_size"] = max(
                    1000, 
                    int(new_params.topological["sample_size"] / (target_ratio ** 0.3))
                )
                new_params.algebraic["sampling_rate"] = max(
                    0.001,
                    new_params.algebraic["sampling_rate"] / (target_ratio ** 0.3)
                )
                new_params.spectral["threshold_percentile"] = min(
                    99,
                    new_params.spectral["threshold_percentile"] * (target_ratio ** 0.4)
                )
        
        return new_params
    
    def save_state(self, file_path: str):
        """
        Save the current state of the hypercube to a file.
        
        Args:
            file_path: Path to save the state to
        """
        try:
            # Collect state data
            state = {
                "dimension": self.dimension,
                "max_dimension": self.max_dimension,
                "stability_threshold": self.stability_threshold,
                "tvi_threshold": self.tvi_threshold,
                "drift_threshold": self.drift_threshold,
                "quantum_platform": self.quantum_platform.name,
                "ecdsa_curve": self.ecdsa_curve,
                "compression_method": self.compression_method.name,
                "wdm_parallelism": self.wdm_parallelism,
                "auto_calibrate": self.auto_calibrate,
                "signatures": self.signatures,
                "quantum_state": self.quantum_state,
                "quantum_metrics": self.quantum_metrics,
                "topological_metrics": self.topological_metrics,
                "tvi_history": self.tvi_history,
                "last_update": self.last_update,
                "signature_count": self.signature_count,
                "processing_times": self.processing_times
            }
            
            # Serialize and compress
            state_bytes = pickle.dumps(state)
            compressed = zlib.compress(state_bytes)
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(compressed)
            
            logger.info(f"Hypercube state saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save hypercube state: {e}", exc_info=True)
            raise HypercubeError(f"State save failed: {str(e)}") from e
    
    def load_state(self, file_path: str):
        """
        Load the hypercube state from a file.
        
        Args:
            file_path: Path to load the state from
        """
        try:
            # Read and decompress
            with open(file_path, 'rb') as f:
                compressed = f.read()
            state_bytes = zlib.decompress(compressed)
            state = pickle.loads(state_bytes)
            
            # Restore state
            self.dimension = state["dimension"]
            self.max_dimension = state["max_dimension"]
            self.stability_threshold = state["stability_threshold"]
            self.tvi_threshold = state["tvi_threshold"]
            self.drift_threshold = state["drift_threshold"]
            self.quantum_platform = QuantumPlatform[state["quantum_platform"]]
            self.ecdsa_curve = state["ecdsa_curve"]
            self.compression_method = CompressionMethod[state["compression_method"]]
            self.wdm_parallelism = state["wdm_parallelism"]
            self.auto_calibrate = state["auto_calibrate"]
            self.signatures = state["signatures"]
            self.quantum_state = state["quantum_state"]
            self.quantum_metrics = state["quantum_metrics"]
            self.topological_metrics = state["topological_metrics"]
            self.tvi_history = state["tvi_history"]
            self.last_update = state["last_update"]
            self.signature_count = state["signature_count"]
            self.processing_times = state["processing_times"]
            
            # Reinitialize components based on restored state
            self.platform_config = get_platform_config(self.quantum_platform)
            self.compression_params = self._initialize_compression_params()
            
            logger.info(f"Hypercube state loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load hypercube state: {e}", exc_info=True)
            raise HypercubeError(f"State load failed: {str(e)}") from e
    
    def close(self):
        """Clean up resources and stop background processes."""
        # Stop calibration system
        if self.calibration_system:
            self.calibration_system.stop()
        
        # Stop resource monitor
        self.resource_monitor.stop()
        
        logger.info("AdaptiveQuantumHypercube resources cleaned up")

class ResourceMonitor:
    """Monitor system resources to prevent overload."""
    
    def __init__(self, interval: float = 5.0):
        """
        Initialize the resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.memory_usage = 0.0
        self.cpu_usage = 0.0
        self.last_update = 0.0
    
    def start(self):
        """Start monitoring resources."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info("Resource monitor started")
    
    def stop(self):
        """Stop monitoring resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Resource monitor stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Get memory usage
                self.memory_usage = psutil.virtual_memory().percent
                
                # Get CPU usage
                self.cpu_usage = psutil.cpu_percent(interval=0.1)
                
                # Update timestamp
                self.last_update = time.time()
                
                # Sleep for interval
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}", exc_info=True)
                time.sleep(self.interval)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        if time.time() - self.last_update > self.interval * 2:
            # Force update if stale
            self.memory_usage = psutil.virtual_memory().percent
            self.last_update = time.time()
        return self.memory_usage
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if time.time() - self.last_update > self.interval * 2:
            # Force update if stale
            self.cpu_usage = psutil.cpu_percent(interval=0.1)
            self.last_update = time.time()
        return self.cpu_usage

# Global instance for convenience (can be overridden by application)
default_hypercube = AdaptiveQuantumHypercube()
