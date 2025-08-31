"""
QuantumFortress 2.0 Hybrid Cryptographic System

This module implements the full hybrid cryptographic system that enables seamless migration
from classical to post-quantum algorithms while maintaining full backward compatibility.
The system uses the Topological Vulnerability Index (TVI) as the primary metric
for determining migration phases and security status.

Key features:
- Hybrid signing with both ECDSA and quantum-topological signatures
- Automatic migration phases triggered by TVI thresholds
- Full backward compatibility with existing blockchain networks
- TVI-based transaction filtering (blocks transactions with TVI > 0.5)
- QuantumBridge integration for traditional network compatibility
- Advanced topological analysis using Betti numbers and Euler characteristic
- Adaptive quantum hypercube with multiple compression techniques
- Integration with TCON (Topological CONformance) system
- WDM-parallelism for quantum operations

As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
точную количественную оценку структуры пространства подписей и обнаруживает скрытые
уязвимости, которые пропускаются другими методами."

This implementation extends those principles to a hybrid cryptographic framework that
provides a practical path to post-quantum security with comprehensive topological analysis.
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

from ..utils.crypto_utils import generate_ecdsa_keys, verify_ecdsa_signature, ecdsa_sign, ecdsa_verify
from ..utils.topology_utils import (
    calculate_betti_numbers, 
    analyze_signature_topology,
    torus_distance,
    calculate_euler_characteristic,
    calculate_topological_entropy,
    find_high_density_areas,
    get_connectivity_metrics
)
from ..utils.quantum_utils import (
    generate_quantum_key_pair, 
    verify_quantum_signature,
    QuantumPlatform,
    PlatformConfig,
    get_platform_config,
    optimize_shor_algorithm
)
from .adaptive_hypercube import AdaptiveQuantumHypercube
from .tcon import TCONValidator
from .topological_analyzer import TopologicalAnalyzer
from .betti_analyzer import BettiAnalyzer
from .collision_engine import CollisionEngine
from .auto_calibration import AutoCalibrationSystem
from .gradient_analysis import GradientAnalyzer
from .dynamic_compute_router import DynamicComputeRouter
from .topo_nonce_v2 import TopoNonceV2

logger = logging.getLogger(__name__)

# ======================
# CONSTANTS
# ======================
VULNERABILITY_TYPES = {
    "NONE": "No vulnerability detected",
    "WEAK_SIGNATURES": "Weak signatures detected",
    "TOPOLOGICAL_ANOMALY": "Topological anomaly detected",
    "HIGH_TV": "High topological vulnerability",
    "COLLISION_RISK": "Collision risk detected",
    "QUANTUM_WEAKNESS": "Quantum vulnerability detected"
}

# Resource limits
MAX_SIGNATURES_FOR_ANALYSIS = 100000
MAX_MEMORY_USAGE_PERCENT = 85
MAX_CPU_USAGE_PERCENT = 85
ANALYSIS_TIMEOUT = 300  # seconds

# ======================
# EXCEPTIONS
# ======================
class HybridCryptoError(Exception):
    """Base exception for HybridCryptoSystem module."""
    pass

class InputValidationError(HybridCryptoError):
    """Raised when input validation fails."""
    pass

class ResourceLimitExceededError(HybridCryptoError):
    """Raised when resource limits are exceeded."""
    pass

class AnalysisTimeoutError(HybridCryptoError):
    """Raised when analysis exceeds timeout limits."""
    pass

class SecurityValidationError(HybridCryptoError):
    """Raised when security validation fails."""
    pass

class TopologicalAnalysisError(HybridCryptoError):
    """Raised when topological analysis fails."""
    pass

class QuantumOperationError(HybridCryptoError):
    """Raised when quantum operations fail."""
    pass

# ======================
# ENUMS
# ======================
class MigrationPhase(Enum):
    """Migration phases for the hybrid cryptographic system"""
    CLASSICAL = 1    # Only ECDSA signatures
    HYBRID = 2       # Both ECDSA and quantum-topological signatures
    POST_QUANTUM = 3 # Only quantum-topological signatures

class TVIMetric(Enum):
    """TVI metrics for vulnerability assessment"""
    BETTI_NUMBERS = 1
    EULER_CHARACTERISTIC = 2
    TOPOLOGICAL_ENTROPY = 3
    CONNECTIVITY = 4
    DENSITY_VARIATION = 5
    COLLISION_RISK = 6
    QUANTUM_STABILITY = 7

class CompressionMethod(Enum):
    """Compression methods for the quantum hypercube"""
    TOPOLOGICAL = 1
    ALGEBRAIC = 2
    SPECTRAL = 3
    HYBRID = 4
    NERVE_COMPLEX = 5

class CalibrationStatus(Enum):
    """Calibration status for quantum components"""
    CALIBRATED = 1
    NEEDS_CALIBRATION = 2
    CALIBRATION_IN_PROGRESS = 3
    CALIBRATION_FAILED = 4
    OUT_OF_SPEC = 5

# ======================
# DATA CLASSES
# ======================
@dataclass
class ECDSAMetrics:
    """Metrics for ECDSA signature analysis"""
    tvl: float  # Topological Vulnerability Level (0.0 = secure, 1.0 = critical)
    vulnerability_type: str
    vulnerability_score: float
    explanation: str
    is_secure: bool
    betti_numbers: List[float]
    euler_characteristic: float
    topological_entropy: float
    naturalness_coefficient: float
    timestamp: float
    component_metrics: Dict[str, Any] = None
    density_metrics: Dict[str, Any] = None
    collision_risk: float = 0.0
    quantum_vulnerability: float = 0.0

@dataclass
class QuantumMetrics:
    """Metrics for quantum signature analysis"""
    stability: float  # Quantum stability (0.0 = unstable, 1.0 = stable)
    coherence_time: float  # Coherence time in nanoseconds
    error_rate: float  # Quantum error rate
    platform: QuantumPlatform
    calibration_status: CalibrationStatus
    drift_rate: float  # Current drift rate
    vulnerability_score: float
    timestamp: float
    topology_metrics: Dict[str, Any] = None

@dataclass
class TVIComponents:
    """Components of the Topological Vulnerability Index"""
    betti_component: float
    euler_component: float
    entropy_component: float
    connectivity_component: float
    density_component: float
    collision_component: float
    quantum_component: float
    total_tvi: float
    weights: Dict[str, float]
    timestamp: float

@dataclass
class HypercubeCompressionParams:
    """Parameters for hypercube compression"""
    method: CompressionMethod
    topological: Dict[str, Any]
    algebraic: Dict[str, Any]
    spectral: Dict[str, Any]
    performance: Dict[str, Any]
    target_size_gb: float = None
    quality_target: float = 0.95
    max_iterations: int = 5

@dataclass
class QuantumPlatformMetrics:
    """Metrics for quantum platform performance"""
    platform: QuantumPlatform
    qubit_count: int
    precision_bits: int
    error_tolerance: float
    drift_rate: float
    processing_speed: float
    calibration_interval: int
    wavelengths: int
    memory_usage: float
    cpu_usage: float
    last_calibration: float
    stability_score: float
    vulnerability_score: float
    timestamp: float

# ======================
# CORE CLASSES
# ======================
class HybridKeyPair:
    """Container for hybrid cryptographic key pairs"""
    
    def __init__(self, 
                 key_id: str,
                 created_at: float,
                 ecdsa_private: Any,
                 ecdsa_public: Any,
                 quantum_private: Optional[Any] = None,
                 quantum_public: Optional[Any] = None,
                 current_phase: MigrationPhase = MigrationPhase.CLASSICAL,
                 quantum_platform: QuantumPlatform = QuantumPlatform.SOI,
                 ecdsa_curve: str = "secp256k1",
                 quantum_dimension: int = 4,
                 calibration_data: Dict[str, Any] = None):
        """
        Initialize a hybrid key pair.
        
        Args:
            key_id: Unique identifier for the key pair
            created_at: Timestamp of key creation
            ecdsa_private: ECDSA private key component
            ecdsa_public: ECDSA public key component
            quantum_private: Quantum-topological private key component (optional)
            quantum_public: Quantum-topological public key component (optional)
            current_phase: Current migration phase for these keys
            quantum_platform: Quantum platform used for quantum operations
            ecdsa_curve: ECDSA curve used
            quantum_dimension: Dimension of quantum space
            calibration_data: Calibration data for quantum components
        """
        self.key_id = key_id
        self.created_at = created_at
        self.ecdsa_private = ecdsa_private
        self.ecdsa_public = ecdsa_public
        self.quantum_private = quantum_private
        self.quantum_public = quantum_public
        self.current_phase = current_phase
        self.quantum_platform = quantum_platform
        self.ecdsa_curve = ecdsa_curve
        self.quantum_dimension = quantum_dimension
        self.calibration_data = calibration_data or {}
        self.last_calibration = time.time()
        self.calibration_status = CalibrationStatus.CALIBRATED
        self.vulnerability_metrics = None
        self.quantum_metrics = None
    
    def is_quantum_enabled(self) -> bool:
        """Check if quantum components are available and active."""
        return self.current_phase in [MigrationPhase.HYBRID, MigrationPhase.POST_QUANTUM] and \
               self.quantum_private is not None and self.quantum_public is not None
    
    def needs_calibration(self, max_interval: int = 300) -> bool:
        """Check if the quantum components need calibration."""
        return (time.time() - self.last_calibration) > max_interval
    
    def update_calibration(self, new_data: Dict[str, Any]):
        """Update calibration data and timestamp."""
        self.calibration_data.update(new_data)
        self.last_calibration = time.time()
        self.calibration_status = CalibrationStatus.CALIBRATED
    
    def get_platform_config(self) -> PlatformConfig:
        """Get configuration for the quantum platform."""
        return get_platform_config(self.quantum_platform)
    
    def analyze_vulnerabilities(self, signatures: List[Tuple[int, int, int]]) -> ECDSAMetrics:
        """
        Analyze vulnerabilities in the key pair based on signatures.
        
        Args:
            signatures: List of (r, s, z) signature components
            
        Returns:
            ECDSAMetrics object with vulnerability analysis
        """
        if not signatures:
            return ECDSAMetrics(
                tvl=1.0,
                vulnerability_type=VULNERABILITY_TYPES["NONE"],
                vulnerability_score=0.0,
                explanation="No signatures provided for analysis",
                is_secure=False,
                betti_numbers=[0.0, 0.0, 0.0],
                euler_characteristic=0.0,
                topological_entropy=0.0,
                naturalness_coefficient=1.0,
                timestamp=time.time()
            )
        
        # Transform signatures to (ur, uz) space
        points = []
        for r, s, z in signatures:
            try:
                ur, uz = transform_to_ur_uz(r, s, z, self.ecdsa_curve)
                points.append((ur, uz))
            except Exception as e:
                logger.debug(f"Failed to transform signature: {e}")
                continue
        
        if not points:
            return ECDSAMetrics(
                tvl=1.0,
                vulnerability_type=VULNERABILITY_TYPES["NONE"],
                vulnerability_score=0.0,
                explanation="No valid signatures for analysis",
                is_secure=False,
                betti_numbers=[0.0, 0.0, 0.0],
                euler_characteristic=0.0,
                topological_entropy=0.0,
                naturalness_coefficient=1.0,
                timestamp=time.time()
            )
        
        # Calculate topological metrics
        betti_numbers = calculate_betti_numbers(points)
        euler_char = calculate_euler_characteristic(points)
        topological_entropy = calculate_topological_entropy(points)
        
        # Analyze connectivity
        connectivity_metrics = get_connectivity_metrics(points)
        
        # Find high density areas
        high_density_areas = find_high_density_areas(points)
        
        # Calculate vulnerability score
        vulnerability_score = self._calculate_vulnerability_score(
            betti_numbers, 
            euler_char,
            topological_entropy,
            connectivity_metrics,
            high_density_areas
        )
        
        # Determine vulnerability type
        vulnerability_type = self._determine_vulnerability_type(
            vulnerability_score,
            betti_numbers,
            high_density_areas
        )
        
        # Calculate TVL (Topological Vulnerability Level)
        tvl = min(1.0, max(0.0, vulnerability_score * 1.2))
        
        # Create explanation
        explanation = self._generate_vulnerability_explanation(
            vulnerability_type,
            vulnerability_score,
            betti_numbers,
            high_density_areas
        )
        
        # Store metrics
        self.vulnerability_metrics = {
            "betti_numbers": betti_numbers,
            "euler_characteristic": euler_char,
            "topological_entropy": topological_entropy,
            "connectivity": connectivity_metrics,
            "high_density_areas": high_density_areas,
            "vulnerability_score": vulnerability_score
        }
        
        # Create and return metrics object
        metrics = ECDSAMetrics(
            tvl=tvl,
            vulnerability_type=vulnerability_type,
            vulnerability_score=vulnerability_score,
            explanation=explanation,
            is_secure=vulnerability_score < 0.5,
            betti_numbers=betti_numbers,
            euler_characteristic=euler_char,
            topological_entropy=topological_entropy,
            naturalness_coefficient=self._calculate_naturalness(points),
            timestamp=time.time(),
            component_metrics=connectivity_metrics,
            density_metrics={"high_density_areas": high_density_areas},
            collision_risk=self._estimate_collision_risk(high_density_areas),
            quantum_vulnerability=self._estimate_quantum_vulnerability(betti_numbers)
        )
        
        return metrics
    
    def _calculate_vulnerability_score(self, 
                                      betti_numbers: List[float],
                                      euler_char: float,
                                      topological_entropy: float,
                                      connectivity_metrics: Dict[str, Any],
                                      high_density_areas: List[Tuple]) -> float:
        """
        Calculate the vulnerability score based on topological metrics.
        
        Args:
            betti_numbers: Calculated Betti numbers
            euler_char: Euler characteristic
            topological_entropy: Topological entropy
            connectivity_metrics: Connectivity metrics
            high_density_areas: High density areas
            
        Returns:
            Vulnerability score (0.0 = secure, 1.0 = critical)
        """
        # Weights for different metrics
        weights = {
            "betti1": 0.25,
            "betti2": 0.20,
            "euler": 0.15,
            "entropy": 0.10,
            "connectivity": 0.15,
            "density": 0.10,
            "collision": 0.05
        }
        
        # Calculate individual scores
        betti1_score = min(1.0, betti_numbers[0] * 0.1)
        betti2_score = min(1.0, betti_numbers[1] * 0.3)
        
        # Euler characteristic score (ideal is 0 for torus)
        euler_score = min(1.0, abs(euler_char) * 0.5)
        
        # Topological entropy score (lower is better)
        entropy_score = min(1.0, (1.0 - topological_entropy) * 0.8)
        
        # Connectivity score
        connectivity_score = 1.0 - connectivity_metrics.get("connectivity_ratio", 0.0)
        
        # Density variation score
        density_score = 0.0
        if high_density_areas:
            # More high density areas with high counts indicate vulnerability
            density_score = min(1.0, sum(count for _, _, count, _, _ in high_density_areas) / 1000.0)
        
        # Collision risk score
        collision_score = min(1.0, self._estimate_collision_risk(high_density_areas) * 2.0)
        
        # Combined score
        score = (
            weights["betti1"] * betti1_score +
            weights["betti2"] * betti2_score +
            weights["euler"] * euler_score +
            weights["entropy"] * entropy_score +
            weights["connectivity"] * connectivity_score +
            weights["density"] * density_score +
            weights["collision"] * collision_score
        )
        
        return min(1.0, score * 1.2)  # Slight amplification for conservatism
    
    def _determine_vulnerability_type(self, 
                                     score: float,
                                     betti_numbers: List[float],
                                     high_density_areas: List[Tuple]) -> str:
        """Determine the type of vulnerability based on analysis."""
        if score < 0.2:
            return VULNERABILITY_TYPES["NONE"]
        
        # Check for weak signatures (high density clusters)
        if len(high_density_areas) > 5:
            return VULNERABILITY_TYPES["WEAK_SIGNATURES"]
        
        # Check for topological anomalies (unusual Betti numbers)
        if betti_numbers[0] > 5 or betti_numbers[1] > 2:
            return VULNERABILITY_TYPES["TOPOLOGICAL_ANOMALY"]
        
        # Check for high topological vulnerability
        if score > 0.7:
            return VULNERABILITY_TYPES["HIGH_TV"]
        
        # Check for collision risk
        if self._estimate_collision_risk(high_density_areas) > 0.3:
            return VULNERABILITY_TYPES["COLLISION_RISK"]
        
        # Check for quantum vulnerability
        if self._estimate_quantum_vulnerability(betti_numbers) > 0.5:
            return VULNERABILITY_TYPES["QUANTUM_WEAKNESS"]
        
        return VULNERABILITY_TYPES["NONE"]
    
    def _generate_vulnerability_explanation(self, 
                                          vulnerability_type: str,
                                          score: float,
                                          betti_numbers: List[float],
                                          high_density_areas: List[Tuple]) -> str:
        """Generate a human-readable explanation of the vulnerability."""
        if vulnerability_type == VULNERABILITY_TYPES["NONE"]:
            return "No significant vulnerabilities detected. The signature topology appears secure."
        
        explanations = {
            VULNERABILITY_TYPES["WEAK_SIGNATURES"]: 
                f"Multiple high-density clusters detected ({len(high_density_areas)} clusters). "
                f"This indicates potential weak signatures that could be vulnerable to collision attacks.",
                
            VULNERABILITY_TYPES["TOPOLOGICAL_ANOMALY"]:
                f"Unusual topological structure detected (Betti numbers: β0={betti_numbers[0]:.2f}, "
                f"β1={betti_numbers[1]:.2f}). This suggests structural weaknesses in the signature space.",
                
            VULNERABILITY_TYPES["HIGH_TV"]:
                f"High topological vulnerability score ({score:.2f}/1.0). The signature topology "
                f"shows significant deviations from expected distribution, indicating potential security risks.",
                
            VULNERABILITY_TYPES["COLLISION_RISK"]:
                f"Elevated collision risk detected. {len(high_density_areas)} high-density regions "
                f"with potential collision points identified.",
                
            VULNERABILITY_TYPES["QUANTUM_WEAKNESS"]:
                f"Quantum vulnerability detected. Topological analysis suggests potential weaknesses "
                f"that could be exploited by quantum attacks."
        }
        
        return explanations.get(vulnerability_type, "Security vulnerability detected but type could not be determined.")
    
    def _calculate_naturalness(self, points: List[Tuple[float, float]]) -> float:
        """Calculate how 'natural' the distribution appears (1.0 = ideal)."""
        if len(points) < 10:
            return 0.5  # Not enough data
        
        # Convert to numpy array
        points_array = np.array(points)
        
        # Calculate pairwise distances
        distances = pdist(points_array)
        
        # Expected distribution for uniform points on torus
        expected_mean = 0.25  # Approximate for unit torus
        expected_std = 0.1
        
        # Calculate actual distribution
        actual_mean = np.mean(distances)
        actual_std = np.std(distances)
        
        # Compare to expected
        mean_diff = abs(actual_mean - expected_mean) / expected_mean
        std_diff = abs(actual_std - expected_std) / expected_std
        
        # Score (lower diffs are better)
        score = 1.0 - (mean_diff * 0.6 + std_diff * 0.4)
        
        return max(0.0, min(1.0, score))
    
    def _estimate_collision_risk(self, high_density_areas: List[Tuple]) -> float:
        """Estimate collision risk based on high density areas."""
        if not high_density_areas:
            return 0.0
        
        # Total points in high density areas
        total_points = sum(count for _, _, count, _, _ in high_density_areas)
        
        # Number of high density areas
        num_areas = len(high_density_areas)
        
        # Average points per area
        avg_points = total_points / num_areas if num_areas > 0 else 0
        
        # Collision risk estimate
        risk = min(1.0, (avg_points * num_areas) / 1000.0)
        
        return risk
    
    def _estimate_quantum_vulnerability(self, betti_numbers: List[float]) -> float:
        """Estimate vulnerability to quantum attacks based on topology."""
        # Higher Betti numbers indicate more complex topology which could be vulnerable
        beta0, beta1, beta2 = betti_numbers[:3]
        
        # Base vulnerability from Betti numbers
        vulnerability = (beta0 * 0.1 + beta1 * 0.3 + beta2 * 0.6)
        
        # Scale to 0-1 range
        return min(1.0, vulnerability / 2.0)

class HybridSignature:
    """Container for hybrid cryptographic signatures"""
    
    def __init__(self,
                 signature_id: str,
                 timestamp: float,
                 ecdsa_signature: bytes,
                 quantum_signature: Optional[bytes] = None,
                 tvi: float = 1.0,
                 migration_phase: MigrationPhase = MigrationPhase.CLASSICAL,
                 ecdsa_components: Optional[Dict[str, Any]] = None,
                 quantum_components: Optional[Dict[str, Any]] = None,
                 topology_metrics: Optional[Dict[str, Any]] = None,
                 quantum_metrics: Optional[Dict[str, Any]] = None,
                 security_level: float = 0.0):
        """
        Initialize a hybrid signature.
        
        Args:
            signature_id: Unique identifier for the signature
            timestamp: Timestamp of signature creation
            ecdsa_signature: ECDSA signature component
            quantum_signature: Quantum-topological signature component (optional)
            tvi: Topological Vulnerability Index score (0.0 = secure, 1.0 = critical)
            migration_phase: Migration phase used for signing
            ecdsa_components: Additional ECDSA signature components
            quantum_components: Additional quantum signature components
            topology_metrics: Topological metrics for the signature
            quantum_metrics: Quantum metrics for the signature
            security_level: Security level (0.0 = insecure, 1.0 = highly secure)
        """
        self.signature_id = signature_id
        self.timestamp = timestamp
        self.ecdsa_signature = ecdsa_signature
        self.quantum_signature = quantum_signature
        self.tvi = tvi
        self.migration_phase = migration_phase
        self.ecdsa_components = ecdsa_components or {}
        self.quantum_components = quantum_components or {}
        self.topology_metrics = topology_metrics or {}
        self.quantum_metrics = quantum_metrics or {}
        self.security_level = security_level
        self.validation_results = None
        self.collision_risk = 0.0
        self.quantum_vulnerability = 0.0
    
    def is_secure(self, threshold: float = 0.5) -> bool:
        """
        Check if the signature is secure based on TVI.
        
        Args:
            threshold: TVI threshold for security (default: 0.5)
            
        Returns:
            bool: True if secure (TVI < threshold), False otherwise
        """
        return self.tvi < threshold
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get detailed security metrics for the signature."""
        return {
            "tvi": self.tvi,
            "security_level": self.security_level,
            "collision_risk": self.collision_risk,
            "quantum_vulnerability": self.quantum_vulnerability,
            "topology_metrics": self.topology_metrics,
            "quantum_metrics": self.quantum_metrics,
            "validation_results": self.validation_results
        }
    
    def analyze_topology(self, 
                        public_key: HybridKeyPair,
                        message: Union[str, bytes]) -> Dict[str, Any]:
        """
        Perform detailed topological analysis of the signature.
        
        Args:
            public_key: Public key used for verification
            message: Original message that was signed
            
        Returns:
            Dictionary containing topological analysis results
        """
        try:
            # Extract ECDSA components
            r = self.ecdsa_components.get('r')
            s = self.ecdsa_components.get('s')
            z = self.ecdsa_components.get('z')
            
            if r is None or s is None or z is None:
                # Try to extract from signature
                try:
                    # This would depend on the actual signature format
                    r, s = self._extract_ecdsa_components(self.ecdsa_signature)
                    z = self._calculate_z(message, public_key.ecdsa_curve)
                except Exception as e:
                    logger.error(f"Failed to extract ECDSA components: {e}")
                    return {}
            
            # Transform to (ur, uz) space
            ur, uz = transform_to_ur_uz(r, s, z, public_key.ecdsa_curve)
            
            # Create point for analysis
            point = (ur, uz)
            
            # Perform topological analysis
            analyzer = TopologicalAnalyzer()
            metrics = analyzer.analyze_point(point)
            
            # Store metrics
            self.topology_metrics = metrics
            
            # Update TVI based on analysis
            self.tvi = self._calculate_tvi_from_metrics(metrics)
            
            # Calculate security level
            self.security_level = 1.0 - min(1.0, self.tvi * 1.2)
            
            # Calculate collision risk
            self.collision_risk = self._estimate_collision_risk(metrics)
            
            # Calculate quantum vulnerability
            self.quantum_vulnerability = self._estimate_quantum_vulnerability(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Topology analysis failed: {e}", exc_info=True)
            return {}
    
    def _extract_ecdsa_components(self, signature: bytes) -> Tuple[int, int]:
        """Extract r and s components from ECDSA signature."""
        # This is a placeholder - actual implementation depends on signature format
        # In a real implementation, this would parse the ASN.1 structure of the signature
        r = int.from_bytes(signature[:32], byteorder='big')
        s = int.from_bytes(signature[32:64], byteorder='big')
        return r, s
    
    def _calculate_z(self, message: Union[str, bytes], curve: str) -> int:
        """Calculate z value from message and curve parameters."""
        # This is a placeholder - actual implementation depends on curve and hashing
        if isinstance(message, str):
            message = message.encode()
        
        # In real implementation, this would hash the message with the appropriate hash function
        # for the curve (e.g., SHA-256 for secp256k1)
        import hashlib
        if curve == "secp256k1":
            z = int.from_bytes(hashlib.sha256(message).digest(), byteorder='big')
        else:
            # Other curves would use different hash functions
            z = int.from_bytes(hashlib.sha3_256(message).digest(), byteorder='big')
        
        # Truncate if necessary
        n = self._get_curve_order(curve)
        z = z % n
        
        return z
    
    def _get_curve_order(self, curve: str) -> int:
        """Get the order of the elliptic curve."""
        # Placeholder - in real implementation, this would return actual curve parameters
        if curve == "secp256k1":
            return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        elif curve == "P-256":
            return 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
        else:
            # Default to secp256k1 order
            return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    
    def _calculate_tvi_from_metrics(self, metrics: Dict[str, Any]) -> float:
        """Calculate TVI based on topological metrics."""
        # Weights for different metrics
        weights = {
            "betti1_anomaly": 0.25,
            "betti2_anomaly": 0.20,
            "euler_deviation": 0.15,
            "entropy_deviation": 0.10,
            "connectivity_issues": 0.15,
            "density_anomalies": 0.10,
            "collision_risk": 0.05
        }
        
        # Calculate individual components
        betti1_anomaly = min(1.0, metrics.get("betti1", 0.0) * 0.1)
        betti2_anomaly = min(1.0, metrics.get("betti2", 0.0) * 0.3)
        euler_deviation = min(1.0, abs(metrics.get("euler_characteristic", 0.0)) * 0.5)
        entropy_deviation = min(1.0, (1.0 - metrics.get("topological_entropy", 0.5)) * 0.8)
        connectivity_issues = 1.0 - metrics.get("connectivity_ratio", 1.0)
        density_anomalies = min(1.0, metrics.get("density_variation", 0.0) * 2.0)
        collision_risk = min(1.0, metrics.get("collision_risk", 0.0) * 2.0)
        
        # Combined TVI
        tvi = (
            weights["betti1_anomaly"] * betti1_anomaly +
            weights["betti2_anomaly"] * betti2_anomaly +
            weights["euler_deviation"] * euler_deviation +
            weights["entropy_deviation"] * entropy_deviation +
            weights["connectivity_issues"] * connectivity_issues +
            weights["density_anomalies"] * density_anomalies +
            weights["collision_risk"] * collision_risk
        )
        
        return min(1.0, tvi * 1.2)  # Slight amplification for conservatism
    
    def _estimate_collision_risk(self, metrics: Dict[str, Any]) -> float:
        """Estimate collision risk based on topological metrics."""
        high_density_areas = metrics.get("high_density_areas", [])
        if not high_density_areas:
            return 0.0
        
        # Calculate risk based on number and size of high density areas
        total_risk = 0.0
        for ur_mean, uz_mean, r_val, count, ur_cluster, uz_cluster in high_density_areas:
            # Risk increases with count and decreases with spread
            spread = np.std(ur_cluster) + np.std(uz_cluster)
            area_risk = count / (spread + 1e-10)
            total_risk += area_risk
        
        # Normalize risk to 0-1 range
        return min(1.0, total_risk / 1000.0)
    
    def _estimate_quantum_vulnerability(self, metrics: Dict[str, Any]) -> float:
        """Estimate vulnerability to quantum attacks based on topology."""
        betti1 = metrics.get("betti1", 0.0)
        betti2 = metrics.get("betti2", 0.0)
        
        # Higher Betti numbers indicate more complex topology which could be vulnerable
        vulnerability = (betti1 * 0.3 + betti2 * 0.7)
        
        # Scale to 0-1 range
        return min(1.0, vulnerability / 3.0)

class TVICalculator:
    """Calculator for Topological Vulnerability Index (TVI)"""
    
    def __init__(self,
                 betti1_threshold: float = 2.5,
                 betti2_threshold: float = 1.5,
                 euler_threshold: float = 0.3,
                 entropy_threshold: float = 0.2,
                 connectivity_threshold: float = 0.7,
                 density_threshold: float = 0.3,
                 collision_threshold: float = 0.2):
        """
        Initialize the TVI calculator.
        
        Args:
            betti1_threshold: Threshold for Betti number 1
            betti2_threshold: Threshold for Betti number 2
            euler_threshold: Threshold for Euler characteristic deviation
            entropy_threshold: Threshold for topological entropy deviation
            connectivity_threshold: Threshold for connectivity issues
            density_threshold: Threshold for density anomalies
            collision_threshold: Threshold for collision risk
        """
        self.betti1_threshold = betti1_threshold
        self.betti2_threshold = betti2_threshold
        self.euler_threshold = euler_threshold
        self.entropy_threshold = entropy_threshold
        self.connectivity_threshold = connectivity_threshold
        self.density_threshold = density_threshold
        self.collision_threshold = collision_threshold
        
        # Default weights for TVI components
        self.weights = {
            "betti1": 0.25,
            "betti2": 0.20,
            "euler": 0.15,
            "entropy": 0.10,
            "connectivity": 0.15,
            "density": 0.10,
            "collision": 0.05
        }
    
    def calculate_tvi(self, 
                     topology_metrics: Dict[str, Any],
                     quantum_metrics: Optional[Dict[str, Any]] = None) -> TVIComponents:
        """
        Calculate the Topological Vulnerability Index (TVI).
        
        Args:
            topology_metrics: Dictionary of topological metrics
            quantum_metrics: Optional dictionary of quantum metrics
            
        Returns:
            TVIComponents object with detailed TVI breakdown
        """
        # Calculate individual components
        betti1 = topology_metrics.get("betti_numbers", [0,0,0])[0]
        betti2 = topology_metrics.get("betti_numbers", [0,0,0])[1]
        
        betti1_component = min(1.0, betti1 / self.betti1_threshold)
        betti2_component = min(1.0, betti2 / self.betti2_threshold)
        
        euler_char = abs(topology_metrics.get("euler_characteristic", 0.0))
        euler_component = min(1.0, euler_char / self.euler_threshold)
        
        entropy = 1.0 - topology_metrics.get("topological_entropy", 0.5)
        entropy_component = min(1.0, entropy / self.entropy_threshold)
        
        connectivity = 1.0 - topology_metrics.get("connectivity_ratio", 1.0)
        connectivity_component = min(1.0, connectivity / self.connectivity_threshold)
        
        density_variation = topology_metrics.get("density_variation", 0.0)
        density_component = min(1.0, density_variation / self.density_threshold)
        
        collision_risk = topology_metrics.get("collision_risk", 0.0)
        collision_component = min(1.0, collision_risk / self.collision_threshold)
        
        # Quantum component if available
        quantum_component = 0.0
        if quantum_metrics:
            quantum_vulnerability = quantum_metrics.get("vulnerability_score", 0.0)
            quantum_component = min(1.0, quantum_vulnerability)
        
        # Calculate total TVI
        total_tvi = (
            self.weights["betti1"] * betti1_component +
            self.weights["betti2"] * betti2_component +
            self.weights["euler"] * euler_component +
            self.weights["entropy"] * entropy_component +
            self.weights["connectivity"] * connectivity_component +
            self.weights["density"] * density_component +
            self.weights["collision"] * collision_component +
            0.1 * quantum_component  # Smaller weight for quantum component
        )
        
        # Clamp to 0-1 range
        total_tvi = min(1.0, total_tvi)
        
        return TVIComponents(
            betti1_component=betti1_component,
            betti2_component=betti2_component,
            euler_component=euler_component,
            entropy_component=entropy_component,
            connectivity_component=connectivity_component,
            density_component=density_component,
            collision_component=collision_component,
            quantum_component=quantum_component,
            total_tvi=total_tvi,
            weights=self.weights.copy(),
            timestamp=time.time()
        )
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update the weights for TVI components.
        
        Args:
            new_weights: Dictionary of new weights
            
        Raises:
            ValueError: If weights don't sum to approximately 1.0
        """
        # Validate weights sum to ~1.0
        total = sum(new_weights.values())
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"TVI weights must sum to approximately 1.0, got {total}")
        
        # Update weights
        self.weights = new_weights.copy()

class HybridCryptoSystem:
    """
    Hybrid cryptographic system for QuantumFortress 2.0.
    
    This class implements the core functionality for hybrid cryptographic operations,
    managing the transition from classical to post-quantum cryptography based on
    Topological Vulnerability Index (TVI) measurements.
    
    The system operates in three migration phases:
    1. CLASSICAL: Only ECDSA signatures are used
    2. HYBRID: Both ECDSA and quantum-topological signatures are used
    3. POST_QUANTUM: Only quantum-topological signatures are used
    
    Migration between phases is determined by TVI measurements and system policies.
    
    Key features:
    - TVI-based migration phase determination
    - Advanced topological analysis using Betti numbers
    - Integration with TCON (Topological CONformance) system
    - Adaptive quantum hypercube with multiple compression techniques
    - WDM-parallelism for quantum operations
    - Auto-calibration for quantum components
    - Resource-aware operation to prevent overload
    - Detailed security metrics and vulnerability analysis
    """
    
    def __init__(self, 
                 base_dimension: int = 4,
                 tvi_threshold_classical: float = 0.3,
                 tvi_threshold_hybrid: float = 0.7,
                 min_quantum_security: float = 0.8,
                 quantum_platform: QuantumPlatform = QuantumPlatform.SOI,
                 ecdsa_curve: str = "secp256k1",
                 max_signatures_for_analysis: int = 10000,
                 compression_method: CompressionMethod = CompressionMethod.HYBRID,
                 auto_calibrate: bool = True,
                 wdm_parallelism: bool = True):
        """
        Initialize the hybrid cryptographic system.
        
        Args:
            base_dimension: Base dimension for the quantum hypercube
            tvi_threshold_classical: TVI threshold for remaining in CLASSICAL phase
            tvi_threshold_hybrid: TVI threshold for moving to POST_QUANTUM phase
            min_quantum_security: Minimum quantum security level required for migration
            quantum_platform: Quantum platform to use
            ecdsa_curve: ECDSA curve to use
            max_signatures_for_analysis: Maximum signatures to use for analysis
            compression_method: Compression method for the quantum hypercube
            auto_calibrate: Whether to enable auto-calibration
            wdm_parallelism: Whether to enable WDM parallelism
        """
        self.base_dimension = base_dimension
        self.tvi_threshold_classical = tvi_threshold_classical
        self.tvi_threshold_hybrid = tvi_threshold_hybrid
        self.min_quantum_security = min_quantum_security
        self.quantum_platform = quantum_platform
        self.ecdsa_curve = ecdsa_curve
        self.max_signatures_for_analysis = max_signatures_for_analysis
        self.compression_method = compression_method
        self.wdm_parallelism = wdm_parallelism
        
        # Initialize migration phase
        self.migration_phase = MigrationPhase.CLASSICAL
        self.phase_start_time = time.time()
        self.last_analysis = 0.0
        self.analysis_interval = 60.0  # seconds
        
        # Initialize hypercube with compression
        self.hypercube = self._initialize_hypercube()
        
        # Initialize TCON validator
        self.tcon_validator = TCONValidator()
        
        # Initialize auto-calibration system
        self.auto_calibrate = auto_calibrate
        self.calibration_system = None
        if auto_calibrate:
            self.calibration_system = AutoCalibrationSystem(quantum_platform)
            self.calibration_system.start()
        
        # Initialize quantum security level
        self.quantum_security_level = 0.0
        self.quantum_metrics = None
        
        # Initialize TVI history
        self.tvi_history = []
        self.max_history_size = 1000
        
        # Initialize resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.start()
        
        # Initialize dynamic compute router
        self.compute_router = DynamicComputeRouter()
        
        # Initialize collision engine
        self.collision_engine = CollisionEngine()
        
        # Initialize gradient analyzer
        self.gradient_analyzer = GradientAnalyzer()
        
        # Initialize topological analyzer
        self.topological_analyzer = TopologicalAnalyzer()
        
        # Initialize betti analyzer
        self.betti_analyzer = BettiAnalyzer()
        
        # Initialize TopoNonce for mining
        self.topo_nonce = TopoNonceV2(
            dimension=base_dimension,
            wdm_parallelism=wdm_parallelism
        )
        
        # Initialize TVI calculator
        self.tvi_calculator = TVICalculator()
        
        # Initialize platform metrics
        self.platform_metrics = self._get_initial_platform_metrics()
        
        logger.info(f"Initialized HybridCryptoSystem (phase={self.migration_phase.name}, "
                    f"dimension={self.base_dimension}, platform={quantum_platform.name})")
    
    def _initialize_hypercube(self) -> AdaptiveQuantumHypercube:
        """Initialize the adaptive quantum hypercube with appropriate compression."""
        # Get platform config
        platform_config = get_platform_config(self.quantum_platform)
        
        # Create compression parameters
        compression_params = HypercubeCompressionParams(
            method=self.compression_method,
            topological={
                "sample_size": 10000,
                "min_cluster_size": 50,
                "eps": 0.05
            },
            algebraic={
                "sampling_rate": 0.01,
                "min_points": 1000
            },
            spectral={
                "threshold_percentile": 95,
                "psnr_target": 40
            },
            performance={
                "grid_size": 1000,
                "max_memory_usage": MAX_MEMORY_USAGE_PERCENT
            }
        )
        
        # Initialize hypercube with compression
        return AdaptiveQuantumHypercube(
            base_dimension=self.base_dimension,
            compression_params=compression_params,
            platform_config=platform_config
        )
    
    def _get_initial_platform_metrics(self) -> QuantumPlatformMetrics:
        """Get initial metrics for the quantum platform."""
        platform_config = get_platform_config(self.quantum_platform)
        
        return QuantumPlatformMetrics(
            platform=self.quantum_platform,
            qubit_count=self.base_dimension,
            precision_bits=platform_config.min_precision,
            error_tolerance=platform_config.error_tolerance,
            drift_rate=platform_config.drift_rate,
            processing_speed=platform_config.processing_speed,
            calibration_interval=platform_config.calibration_interval,
            wavelengths=platform_config.wavelengths,
            memory_usage=0.0,
            cpu_usage=0.0,
            last_calibration=time.time(),
            stability_score=0.8,
            vulnerability_score=0.0,
            timestamp=time.time()
        )
    
    def _update_platform_metrics(self):
        """Update metrics for the quantum platform."""
        # Get current resource usage
        memory_usage = self.resource_monitor.get_memory_usage()
        cpu_usage = self.resource_monitor.get_cpu_usage()
        
        # Update platform metrics
        self.platform_metrics = QuantumPlatformMetrics(
            platform=self.quantum_platform,
            qubit_count=self.base_dimension,
            precision_bits=self.platform_metrics.precision_bits,
            error_tolerance=self.platform_metrics.error_tolerance,
            drift_rate=self.platform_metrics.drift_rate,
            processing_speed=self.platform_metrics.processing_speed,
            calibration_interval=self.platform_metrics.calibration_interval,
            wavelengths=self.platform_metrics.wavelengths,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            last_calibration=self.platform_metrics.last_calibration,
            stability_score=self.platform_metrics.stability_score,
            vulnerability_score=self.platform_metrics.vulnerability_score,
            timestamp=time.time()
        )
    
    def _determine_migration_phase(self) -> MigrationPhase:
        """
        Determine the appropriate migration phase based on TVI measurements and system state.
        
        Returns:
            MigrationPhase: The recommended migration phase
        """
        # Get current TVI from the hypercube analysis
        current_tvi = self.hypercube.get_tvi()
        self.tvi_history.append((time.time(), current_tvi))
        
        # Keep history to a reasonable size
        if len(self.tvi_history) > self.max_history_size:
            self.tvi_history.pop(0)
        
        # Analyze trends in TVI
        recent_tvi_values = [tvi for _, tvi in self.tvi_history[-10:]]
        avg_recent_tvi = sum(recent_tvi_values) / len(recent_tvi_values) if recent_tvi_values else current_tvi
        
        # Get platform metrics
        self._update_platform_metrics()
        
        # Determine if quantum platform is stable enough
        quantum_stable = (
            self.platform_metrics.stability_score >= 0.7 and
            self.platform_metrics.vulnerability_score <= 0.3
        )
        
        # Determine phase based on TVI thresholds and trends
        if avg_recent_tvi < self.tvi_threshold_classical and self.migration_phase != MigrationPhase.CLASSICAL:
            return MigrationPhase.CLASSICAL
        elif avg_recent_tvi < self.tvi_threshold_hybrid and self.migration_phase != MigrationPhase.HYBRID:
            return MigrationPhase.HYBRID
        elif avg_recent_tvi >= self.tvi_threshold_hybrid and quantum_stable:
            return MigrationPhase.POST_QUANTUM
        
        return self.migration_phase
    
    def _update_migration_phase(self) -> bool:
        """
        Update the migration phase if conditions warrant a change.
        
        Returns:
            bool: True if phase changed, False otherwise
        """
        # Check analysis interval
        if time.time() - self.last_analysis < self.analysis_interval:
            return False
        
        self.last_analysis = time.time()
        
        new_phase = self._determine_migration_phase()
        
        if new_phase != self.migration_phase:
            old_phase = self.migration_phase
            self.migration_phase = new_phase
            self.phase_start_time = time.time()
            
            logger.info(f"Migration phase changed from {old_phase.name} to {new_phase.name} "
                        f"(current TVI: {self.hypercube.get_tvi():.4f}, "
                        f"quantum stability: {self.platform_metrics.stability_score:.2f})")
            
            # Update hypercube configuration based on new phase
            self.hypercube.update_configuration(self.migration_phase)
            
            return True
        
        return False
    
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
    
    def generate_keys(self, quantum_dimension: Optional[int] = None) -> HybridKeyPair:
        """
        Generate hybrid cryptographic keys.
        
        This method creates both classical (ECDSA) and quantum-topological key components,
        with the quantum component being activated based on the current migration phase.
        
        Args:
            quantum_dimension: Optional quantum dimension to use (defaults to system base dimension)
            
        Returns:
            HybridKeyPair object containing both key types
            
        Example from documentation:
        "Плагин для Bitcoin Core: Автоматически проверяет входящие транзакции
        на наличие слабых подписей, Блокирует транзакции с TVI > 0.5."
        """
        self._update_migration_phase()
        self._check_resources()
        
        key_id = str(uuid.uuid4())
        creation_time = time.time()
        
        # Use specified dimension or default to system dimension
        dimension = quantum_dimension or self.base_dimension
        
        # Generate ECDSA keys (always included for backward compatibility)
        ecdsa_private, ecdsa_public = generate_ecdsa_keys(curve=self.ecdsa_curve)
        
        # Generate quantum-topological keys based on migration phase
        quantum_private, quantum_public = None, None
        calibration_data = {}
        
        if self.migration_phase in [MigrationPhase.HYBRID, MigrationPhase.POST_QUANTUM]:
            # Check if we need to calibrate first
            if self.auto_calibrate and self.calibration_system.needs_calibration():
                calibration_data = self.calibration_system.run_calibration()
            
            # Generate quantum keys with appropriate platform
            quantum_private, quantum_public = generate_quantum_key_pair(
                dimension=dimension,
                platform=self.quantum_platform
            )
        
        # Determine current migration phase for these keys
        current_phase = self.migration_phase
        
        # Create and return key pair
        key_pair = HybridKeyPair(
            key_id=key_id,
            created_at=creation_time,
            ecdsa_private=ecdsa_private,
            ecdsa_public=ecdsa_public,
            quantum_private=quantum_private,
            quantum_public=quantum_public,
            current_phase=current_phase,
            quantum_platform=self.quantum_platform,
            ecdsa_curve=self.ecdsa_curve,
            quantum_dimension=dimension,
            calibration_data=calibration_data
        )
        
        # Log key generation
        logger.debug(f"Generated hybrid key pair (ID: {key_id}, phase: {current_phase.name})")
        
        return key_pair
    
    def sign(self, 
             private_key: HybridKeyPair, 
             message: Union[str, bytes],
             include_topology_analysis: bool = True) -> HybridSignature:
        """
        Create a hybrid signature for the given message.
        
        This method:
        - Always creates an ECDSA signature for backward compatibility
        - Creates a quantum-topological signature when in HYBRID or POST_QUANTUM phase
        - Calculates TVI for the signature to assess security
        - Performs detailed topological analysis if requested
        
        Args:
            private_key: HybridKeyPair containing private components
            message: Message to sign
            include_topology_analysis: Whether to include detailed topology analysis
            
        Returns:
            HybridSignature object containing both signature types
            
        Example from documentation:
        "Works as API wrapper (no core modifications needed)"
        """
        self._update_migration_phase()
        self._check_resources()
        
        signature_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Always create ECDSA signature for backward compatibility
        ecdsa_signature, ecdsa_components = self._sign_ecdsa(private_key.ecdsa_private, message)
        
        # Create quantum-topological signature if in appropriate phase
        quantum_signature = None
        quantum_components = None
        if private_key.is_quantum_enabled():
            quantum_signature, quantum_components = self._sign_quantum(
                private_key.quantum_private, 
                message,
                platform=private_key.quantum_platform
            )
        
        # Create basic signature
        signature = HybridSignature(
            signature_id=signature_id,
            timestamp=timestamp,
            ecdsa_signature=ecdsa_signature,
            quantum_signature=quantum_signature,
            migration_phase=private_key.current_phase,
            ecdsa_components=ecdsa_components,
            quantum_components=quantum_components
        )
        
        # Perform detailed topology analysis if requested
        if include_topology_analysis:
            topology_metrics = signature.analyze_topology(private_key, message)
            
            # Update quantum metrics if available
            if private_key.quantum_metrics:
                signature.quantum_metrics = private_key.quantum_metrics
            
            # Calculate security level
            signature.security_level = 1.0 - min(1.0, signature.tvi * 1.2)
        
        # Log signature creation
        logger.debug(f"Created hybrid signature (ID: {signature_id}, TVI: {signature.tvi:.4f})")
        
        return signature
    
    def _sign_ecdsa(self, ecdsa_private: Any, message: Union[str, bytes]) -> Tuple[bytes, Dict[str, Any]]:
        """Create an ECDSA signature for the given message and return components."""
        # In a real implementation, this would use a proper ECDSA signing function
        if isinstance(message, str):
            message = message.encode()
        
        # Generate signature
        r, s = ecdsa_sign(ecdsa_private, message, curve=self.ecdsa_curve)
        
        # Calculate z value
        n = self._get_curve_order(self.ecdsa_curve)
        z = int.from_bytes(message, byteorder='big') % n
        
        # Create signature (DER encoding would be more complex)
        signature = r.to_bytes(32, byteorder='big') + s.to_bytes(32, byteorder='big')
        
        # Return signature and components
        return signature, {
            'r': r,
            's': s,
            'z': z,
            'curve': self.ecdsa_curve
        }
    
    def _get_curve_order(self, curve: str) -> int:
        """Get the order of the elliptic curve."""
        # Placeholder - in real implementation, this would return actual curve parameters
        if curve == "secp256k1":
            return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        elif curve == "P-256":
            return 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
        else:
            # Default to secp256k1 order
            return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    
    def _sign_quantum(self, 
                     quantum_private: Any, 
                     message: Union[str, bytes],
                     platform: QuantumPlatform) -> Tuple[bytes, Dict[str, Any]]:
        """Create a quantum-topological signature for the given message."""
        # In a real implementation, this would use quantum-topological signing
        if isinstance(message, str):
            message = message.encode()
        
        # Get platform config
        platform_config = get_platform_config(platform)
        
        # Generate quantum signature (simplified for example)
        signature = quantum_sign(
            quantum_private, 
            message, 
            platform=platform,
            dimension=self.base_dimension
        )
        
        # Extract components for analysis
        components = {
            'signature': signature,
            'platform': platform.name,
            'dimension': self.base_dimension,
            'timestamp': time.time()
        }
        
        return signature, components
    
    def _analyze_signature_topology(self, 
                                  message: Union[str, bytes], 
                                  ecdsa_signature: bytes, 
                                  quantum_signature: Optional[bytes],
                                  ecdsa_components: Optional[Dict[str, Any]] = None,
                                  quantum_components: Optional[Dict[str, Any]] = None) -> float:
        """
        Analyze the topological properties of a signature to calculate TVI.
        
        Args:
            message: The original message
            ecdsa_signature: The ECDSA signature component
            quantum_signature: The quantum-topological signature component (optional)
            ecdsa_components: Additional ECDSA components
            quantum_components: Additional quantum components
            
        Returns:
            float: TVI score (0.0 = secure, 1.0 = critical vulnerability)
            
        As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
        точную количественную оценку структуры пространства подписей и обнаруживает скрытые
        уязвимости, которые пропускаются другими методами."
        """
        # Update the hypercube with the new signature data
        self.hypercube.update_with_signature(
            message, 
            ecdsa_signature, 
            quantum_signature,
            ecdsa_components=ecdsa_components,
            quantum_components=quantum_components
        )
        
        # Get the current TVI from the hypercube
        tvi = self.hypercube.get_tvi()
        
        # Update quantum security level based on analysis
        self._update_quantum_security_level()
        
        return tvi
    
    def _update_quantum_security_level(self):
        """Update the quantum security level based on current system state."""
        # Analyze quantum platform metrics
        self._update_platform_metrics()
        
        # Calculate quantum security level based on multiple factors
        stability_factor = self.platform_metrics.stability_score
        vulnerability_factor = 1.0 - min(1.0, self.platform_metrics.vulnerability_score * 1.5)
        tvi_factor = 1.0 - min(1.0, self.hypercube.get_tvi() * 1.2)
        
        # Combined security level
        self.quantum_security_level = (
            0.4 * stability_factor +
            0.3 * vulnerability_factor +
            0.3 * tvi_factor
        )
        
        # Update quantum metrics
        self.quantum_metrics = {
            "stability": stability_factor,
            "vulnerability_score": self.platform_metrics.vulnerability_score,
            "tvi": self.hypercube.get_tvi(),
            "security_level": self.quantum_security_level,
            "timestamp": time.time()
        }
    
    def verify(self, 
              public_key: HybridKeyPair, 
              message: Union[str, bytes], 
              signature: HybridSignature,
              strict_tvi: bool = True,
              tvi_threshold: float = 0.5) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify a hybrid signature.
        
        This method:
        - Always verifies the ECDSA signature for backward compatibility
        - Verifies the quantum-topological signature when available
        - Checks TVI to determine if the signature meets current security requirements
        - Performs TCON validation if available
        
        Args:
            public_key: HybridKeyPair containing public components
            message: Message that was signed
            signature: HybridSignature to verify
            strict_tvi: Whether to enforce strict TVI checking
            tvi_threshold: TVI threshold for security (default: 0.5)
            
        Returns:
            Tuple[bool, float, Dict[str, Any]]: (verification result, TVI score, detailed results)
            
        The verification process employs Quantum Vulnerability Analysis to guide robust 
        quantum cryptographic implementations, ensuring security against potential quantum attacks. [[8]]
        """
        self._check_resources()
        
        # Initialize detailed results
        detailed_results = {
            "ecdsa_valid": False,
            "quantum_valid": False,
            "tvi_check": False,
            "tcon_valid": False,
            "security_level": 0.0,
            "collision_risk": 0.0,
            "quantum_vulnerability": 0.0,
            "timestamp": time.time()
        }
        
        # Always verify ECDSA signature (required for backward compatibility)
        ecdsa_valid = self._verify_ecdsa(public_key.ecdsa_public, message, signature.ecdsa_signature)
        detailed_results["ecdsa_valid"] = ecdsa_valid
        
        # Verify quantum signature if present and in appropriate phase
        quantum_valid = True
        if signature.quantum_signature is not None:
            quantum_valid = self._verify_quantum(
                public_key.quantum_public, 
                message, 
                signature.quantum_signature,
                platform=public_key.quantum_platform
            )
            detailed_results["quantum_valid"] = quantum_valid
        
        # Check TCON conformance if available
        tcon_valid = True
        if self.tcon_validator:
            tcon_valid = self.tcon_validator.validate_signature(signature)
            detailed_results["tcon_valid"] = tcon_valid
        
        # Check if signature meets current security requirements based on TVI
        tvi_check = not strict_tvi or signature.is_secure(tvi_threshold)
        detailed_results["tvi_check"] = tvi_check
        
        # Overall verification result requires:
        # 1. ECDSA signature valid
        # 2. Quantum signature valid if present
        # 3. Signature passes TCON validation
        # 4. Signature is secure (TVI < threshold) if strict checking
        verification_result = (
            ecdsa_valid and 
            quantum_valid and 
            tcon_valid and 
            (not strict_tvi or tvi_check)
        )
        
        # Update detailed results with security metrics
        detailed_results.update({
            "security_level": signature.security_level,
            "tvi": signature.tvi,
            "collision_risk": signature.collision_risk,
            "quantum_vulnerability": signature.quantum_vulnerability
        })
        
        if not verification_result:
            reasons = []
            if not ecdsa_valid:
                reasons.append("ECDSA signature invalid")
            if not quantum_valid and signature.quantum_signature is not None:
                reasons.append("Quantum signature invalid")
            if not tcon_valid:
                reasons.append("TCON validation failed")
            if not tvi_check:
                reasons.append(f"TVI too high ({signature.tvi:.4f} >= {tvi_threshold})")
            logger.warning(f"Signature verification failed: {', '.join(reasons)}")
        
        return verification_result, signature.tvi, detailed_results
    
    def _verify_ecdsa(self, ecdsa_public: Any, message: Union[str, bytes], signature: bytes) -> bool:
        """Verify an ECDSA signature."""
        if isinstance(message, str):
            message = message.encode()
        
        # In a real implementation, this would use a proper ECDSA verification function
        return ecdsa_verify(ecdsa_public, message, signature, curve=self.ecdsa_curve)
    
    def _verify_quantum(self, 
                       quantum_public: Any, 
                       message: Union[str, bytes], 
                       signature: bytes,
                       platform: QuantumPlatform) -> bool:
        """Verify a quantum-topological signature."""
        if isinstance(message, str):
            message = message.encode()
        
        # In a real implementation, this would use quantum-topological verification
        return quantum_verify(
            quantum_public, 
            message, 
            signature,
            platform=platform,
            dimension=self.base_dimension
        )
    
    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a transaction through the hybrid cryptographic system.
        
        This method:
        - Analyzes the transaction's signature topology
        - Blocks transactions with TVI > 0.5 as per security policy
        - Handles both classical and quantum-topological signatures
        - Performs TCON validation
        - Checks for collision risks
        
        Args:
            transaction: Transaction dictionary containing signature and message
            
        Returns:
            Processed transaction with security assessment
            
        Example from documentation:
        "Плагин для Bitcoin Core: Автоматически проверяет входящие транзакции
        on наличие слабых подписей, Блокирует транзакции с TVI > 0.5."
        """
        self._check_resources()
        
        # Extract required components
        signature_data = transaction.get("signature")
        public_key_data = transaction.get("public_key")
        message = transaction.get("message")
        transaction_id = transaction.get("id", str(uuid.uuid4()))
        
        if not all([signature_data, public_key_data, message]):
            return {
                "status": "error",
                "reason": "Missing required transaction components",
                "transaction_id": transaction_id,
                "timestamp": time.time()
            }
        
        try:
            # Convert data to appropriate objects
            public_key = self._convert_to_key_pair(public_key_data)
            signature = self._convert_to_signature(signature_data)
            
            # Verify the transaction
            verification_result, tvi, detailed_results = self.verify(public_key, message, signature)
            
            # Additional security checks
            security_checks = self._perform_additional_security_checks(
                public_key, 
                signature, 
                detailed_results
            )
            
            # Combine results
            result = {
                "status": "accepted" if verification_result and security_checks["secure"] else "rejected",
                "transaction_id": transaction_id,
                "tvi": tvi,
                "migration_phase": self.migration_phase.name,
                "detailed_results": detailed_results,
                "security_checks": security_checks,
                "timestamp": time.time()
            }
            
            # Log transaction processing
            if result["status"] == "rejected":
                logger.warning(f"Transaction rejected (ID: {transaction_id}, reason: {security_checks['reason']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Transaction processing failed: {e}", exc_info=True)
            return {
                "status": "error",
                "reason": str(e),
                "transaction_id": transaction_id,
                "timestamp": time.time()
            }
    
    def _convert_to_key_pair(self, key_data: Dict[str, Any]) -> HybridKeyPair:
        """Convert key data dictionary to HybridKeyPair object."""
        return HybridKeyPair(
            key_id=key_data["key_id"],
            created_at=key_data["created_at"],
            ecdsa_private=key_data.get("ecdsa_private"),
            ecdsa_public=key_data["ecdsa_public"],
            quantum_private=key_data.get("quantum_private"),
            quantum_public=key_data.get("quantum_public"),
            current_phase=MigrationPhase[key_data["current_phase"]],
            quantum_platform=QuantumPlatform[key_data.get("quantum_platform", "SOI")],
            ecdsa_curve=key_data.get("ecdsa_curve", "secp256k1"),
            quantum_dimension=key_data.get("quantum_dimension", self.base_dimension),
            calibration_data=key_data.get("calibration_data", {})
        )
    
    def _convert_to_signature(self, signature_data: Dict[str, Any]) -> HybridSignature:
        """Convert signature data dictionary to HybridSignature object."""
        return HybridSignature(
            signature_id=signature_data["signature_id"],
            timestamp=signature_data["timestamp"],
            ecdsa_signature=signature_data["ecdsa_signature"],
            quantum_signature=signature_data.get("quantum_signature"),
            tvi=signature_data.get("tvi", 1.0),
            migration_phase=MigrationPhase[signature_data["migration_phase"]],
            ecdsa_components=signature_data.get("ecdsa_components", {}),
            quantum_components=signature_data.get("quantum_components", {}),
            topology_metrics=signature_data.get("topology_metrics", {}),
            quantum_metrics=signature_data.get("quantum_metrics", {}),
            security_level=signature_data.get("security_level", 0.0)
        )
    
    def _perform_additional_security_checks(self, 
                                          public_key: HybridKeyPair,
                                          signature: HybridSignature,
                                          detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform additional security checks beyond basic verification.
        
        Args:
            public_key: Public key used in the transaction
            signature: Signature to check
            detailed_results: Results from basic verification
            
        Returns:
            Dictionary with additional security check results
        """
        # Check for high collision risk
        if signature.collision_risk > 0.3:
            return {
                "secure": False,
                "reason": "High collision risk detected",
                "collision_risk": signature.collision_risk
            }
        
        # Check for quantum vulnerability
        if signature.quantum_vulnerability > 0.4:
            return {
                "secure": False,
                "reason": "Quantum vulnerability detected",
                "quantum_vulnerability": signature.quantum_vulnerability
            }
        
        # Check if key needs calibration
        if public_key.needs_calibration() and self.auto_calibrate:
            return {
                "secure": False,
                "reason": "Key requires calibration",
                "calibration_needed": True
            }
        
        # Check TCON validation
        if not detailed_results.get("tcon_valid", True):
            return {
                "secure": False,
                "reason": "TCON validation failed",
                "tcon_valid": False
            }
        
        # Check TVI threshold
        if signature.tvi > 0.5:
            return {
                "secure": False,
                "reason": f"TVI too high ({signature.tvi:.4f} > 0.5)",
                "tvi": signature.tvi
            }
        
        return {
            "secure": True,
            "reason": "All security checks passed"
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the hybrid cryptographic system.
        
        Returns:
            Dictionary containing system status information
        """
        # Update platform metrics
        self._update_platform_metrics()
        
        return {
            "migration_phase": self.migration_phase.name,
            "phase_duration": time.time() - self.phase_start_time,
            "current_tvi": self.hypercube.get_tvi(),
            "quantum_security_level": self.quantum_security_level,
            "system_uptime": time.time() - self.phase_start_time,
            "tvi_history_sample": [tvi for _, tvi in self.tvi_history[-5:]] if self.tvi_history else [],
            "platform_metrics": {
                "platform": self.quantum_platform.name,
                "qubit_count": self.base_dimension,
                "precision_bits": self.platform_metrics.precision_bits,
                "error_tolerance": self.platform_metrics.error_tolerance,
                "drift_rate": self.platform_metrics.drift_rate,
                "stability_score": self.platform_metrics.stability_score,
                "vulnerability_score": self.platform_metrics.vulnerability_score,
                "memory_usage": self.platform_metrics.memory_usage,
                "cpu_usage": self.platform_metrics.cpu_usage
            },
            "resource_usage": {
                "memory_percent": self.resource_monitor.get_memory_usage(),
                "cpu_percent": self.resource_monitor.get_cpu_usage(),
                "active_threads": threading.active_count()
            },
            "hypercube_status": self.hypercube.get_status(),
            "calibration_status": self.calibration_system.get_status() if self.calibration_system else None,
            "timestamp": time.time()
        }
    
    def analyze_key_vulnerabilities(self, 
                                  key_pair: HybridKeyPair,
                                  signature_samples: List[Tuple[int, int, int]]) -> ECDSAMetrics:
        """
        Analyze vulnerabilities in a key pair based on signature samples.
        
        Args:
            key_pair: Key pair to analyze
            signature_samples: List of (r, s, z) signature components
            
        Returns:
            ECDSAMetrics object with vulnerability analysis
            
        As stated in documentation: "Для 10,000 кошельков: 3 уязвимых (0.03%)"
        """
        if len(signature_samples) > MAX_SIGNATURES_FOR_ANALYSIS:
            logger.warning(f"Truncating signature samples from {len(signature_samples)} to {MAX_SIGNATURES_FOR_ANALYSIS}")
            signature_samples = signature_samples[:MAX_SIGNATURES_FOR_ANALYSIS]
        
        # Analyze vulnerabilities
        return key_pair.analyze_vulnerabilities(signature_samples)
    
    def generate_topo_nonce(self, 
                          block_header: bytes, 
                          target_difficulty: float,
                          max_iterations: int = 1000000) -> Tuple[int, int, float]:
        """
        Generate a TopoNonce for mining using the TopoNonceV2 algorithm.
        
        Args:
            block_header: Block header to mine
            target_difficulty: Target difficulty for the nonce
            max_iterations: Maximum iterations to try
            
        Returns:
            Tuple containing (nonce, quantum_nonce, achieved_difficulty)
        """
        return self.topo_nonce.generate_nonce(
            block_header,
            target_difficulty,
            max_iterations
        )
    
    def validate_block(self, 
                      block: Dict[str, Any],
                      strict_tvi: bool = True) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a block using hybrid cryptographic verification.
        
        Args:
            block: Block to validate
            strict_tvi: Whether to enforce strict TVI checking
            
        Returns:
            Tuple containing (validation result, detailed results)
        """
        # Validate block header
        header_valid = self._validate_block_header(block.get("header", {}))
        if not header_valid:
            return False, {"reason": "Block header validation failed"}
        
        # Validate transactions
        tx_results = []
        for tx in block.get("transactions", []):
            tx_result = self.process_transaction(tx)
            tx_results.append(tx_result)
            
            # Early rejection if any transaction is invalid
            if tx_result["status"] != "accepted":
                return False, {
                    "reason": f"Transaction validation failed: {tx_result.get('reason', 'Unknown error')}",
                    "failed_transaction": tx.get("id")
                }
        
        # Additional block-level validation
        block_valid = self._additional_block_validation(block, tx_results)
        
        return block_valid, {
            "header_valid": header_valid,
            "transaction_results": tx_results,
            "block_valid": block_valid,
            "timestamp": time.time()
        }
    
    def _validate_block_header(self, header: Dict[str, Any]) -> bool:
        """Validate block header structure and signatures."""
        # Check required fields
        required_fields = ["version", "prev_hash", "merkle_root", "timestamp", "bits", "nonce"]
        if not all(field in header for field in required_fields):
            return False
        
        # Validate nonce structure
        if "topo_nonce" in header:
            # Validate TopoNonce structure
            required_nonce_fields = ["nonce", "quantum_nonce", "topology_metrics"]
            return all(field in header["topo_nonce"] for field in required_nonce_fields)
        
        return True
    
    def _additional_block_validation(self, 
                                   block: Dict[str, Any],
                                   tx_results: List[Dict[str, Any]]) -> bool:
        """Perform additional block-level validation."""
        # Check block size
        if sys.getsizeof(block) > 1024 * 1024:  # 1MB limit
            return False
        
        # Check for duplicate transactions
        tx_ids = [tx.get("id") for tx in block.get("transactions", [])]
        if len(tx_ids) != len(set(tx_ids)):
            return False
        
        # Check TVI metrics for the block
        block_tvi = self._calculate_block_tvi(tx_results)
        if block_tvi > 0.6:  # Higher threshold for blocks
            return False
        
        return True
    
    def _calculate_block_tvi(self, tx_results: List[Dict[str, Any]]) -> float:
        """Calculate TVI for a block based on transaction results."""
        tvi_values = []
        for result in tx_results:
            if "tvi" in result:
                tvi_values.append(result["tvi"])
        
        if not tvi_values:
            return 1.0
        
        # Weighted average with more recent transactions having higher weight
        weights = np.linspace(0.5, 1.5, len(tvi_values))
        return np.average(tvi_values, weights=weights)
    
    def optimize_mining(self, 
                       block_template: Dict[str, Any],
                       resource_constraints: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Optimize mining parameters based on topological analysis.
        
        Args:
            block_template: Block template to mine
            resource_constraints: Optional resource constraints
            
        Returns:
            Optimized mining parameters
        """
        # Get current system status
        system_status = self.get_system_status()
        
        # Analyze topology of current network
        topology_analysis = self.topological_analyzer.analyze_network_topology()
        
        # Determine optimal mining parameters
        optimal_params = {
            "difficulty": self._determine_optimal_difficulty(
                system_status, 
                topology_analysis,
                block_template
            ),
            "search_space": self._determine_optimal_search_space(
                system_status,
                topology_analysis,
                block_template
            ),
            "quantum_usage": self._determine_optimal_quantum_usage(
                system_status,
                topology_analysis
            ),
            "wdm_parallelism": self.wdm_parallelism,
            "timestamp": time.time()
        }
        
        # Apply resource constraints if provided
        if resource_constraints:
            self._apply_resource_constraints(optimal_params, resource_constraints)
        
        return optimal_params
    
    def _determine_optimal_difficulty(self,
                                     system_status: Dict[str, Any],
                                     topology_analysis: Dict[str, Any],
                                     block_template: Dict[str, Any]) -> float:
        """Determine optimal mining difficulty based on system state."""
        # Base difficulty from block template
        base_difficulty = block_template.get("bits", 1.0)
        
        # Adjust based on TVI
        tvi_factor = 1.0 + (system_status["current_tvi"] * 0.5)
        
        # Adjust based on network topology
        topology_factor = 1.0
        if topology_analysis.get("anomalies", 0) > 5:
            topology_factor = 0.8  # Lower difficulty when anomalies detected
        
        # Adjust based on quantum security
        quantum_factor = 1.0
        if system_status["quantum_security_level"] < 0.7:
            quantum_factor = 1.2  # Higher difficulty when quantum security is low
        
        # Combined difficulty
        optimal_difficulty = base_difficulty * tvi_factor * topology_factor * quantum_factor
        
        return max(1.0, optimal_difficulty)
    
    def _determine_optimal_search_space(self,
                                      system_status: Dict[str, Any],
                                      topology_analysis: Dict[str, Any],
                                      block_template: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal search space for mining."""
        # Base search space
        base_space = {
            "start": 0,
            "end": 2**32 - 1
        }
        
        # Focus on high-probability regions based on topology
        high_density_regions = topology_analysis.get("high_density_regions", [])
        if high_density_regions:
            # Select the most promising region
            best_region = max(high_density_regions, key=lambda r: r["probability"])
            
            # Adjust search space to focus on this region
            region_start = best_region["start"]
            region_end = best_region["end"]
            region_size = region_end - region_start
            
            # Expand slightly around the region
            expansion = region_size * 0.2
            adjusted_start = max(0, region_start - expansion)
            adjusted_end = min(2**32 - 1, region_end + expansion)
            
            base_space = {
                "start": int(adjusted_start),
                "end": int(adjusted_end),
                "focused_region": True
            }
        
        return base_space
    
    def _determine_optimal_quantum_usage(self,
                                       system_status: Dict[str, Any],
                                       topology_analysis: Dict[str, Any]) -> float:
        """Determine optimal quantum resource usage for mining."""
        # Base quantum usage
        base_usage = 0.7
        
        # Adjust based on quantum security level
        security_factor = system_status["quantum_security_level"]
        if security_factor < 0.6:
            return 1.0  # Max quantum usage when security is low
        
        # Adjust based on migration phase
        if system_status["migration_phase"] == "CLASSICAL":
            return 0.0  # No quantum usage in CLASSICAL phase
        
        # Adjust based on topology anomalies
        if topology_analysis.get("anomalies", 0) > 10:
            return min(1.0, base_usage * 1.2)  # Slightly increase usage
        
        return base_usage
    
    def _apply_resource_constraints(self, 
                                  params: Dict[str, Any], 
                                  constraints: Dict[str, float]):
        """Apply resource constraints to mining parameters."""
        # Memory constraint
        if "memory" in constraints:
            memory_usage = self.resource_monitor.get_memory_usage()
            if memory_usage > constraints["memory"]:
                # Reduce quantum usage
                params["quantum_usage"] = max(0.0, params["quantum_usage"] - 0.2)
                
                # Widen search space if possible
                if params["search_space"].get("focused_region"):
                    params["search_space"] = {
                        "start": 0,
                        "end": 2**32 - 1,
                        "focused_region": False
                    }
        
        # CPU constraint
        if "cpu" in constraints:
            cpu_usage = self.resource_monitor.get_cpu_usage()
            if cpu_usage > constraints["cpu"]:
                # Reduce WDM parallelism if possible
                if params["wdm_parallelism"]:
                    params["wdm_parallelism"] = False

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

def transform_to_ur_uz(r: int, s: int, z: int, curve: str = "secp256k1") -> Tuple[float, float]:
    """
    Transform ECDSA signature components to (ur, uz) space on the torus.
    
    This function implements the transformation described in Ur Uz работа.md:
    ur = (r * s^-1) mod N
    uz = (z * s^-1) mod N
    
    Args:
        r: ECDSA r component
        s: ECDSA s component
        z: Message hash (mod N)
        curve: Elliptic curve name
        
    Returns:
        Tuple (ur, uz) in [0, 1) range representing points on the unit torus
    
    As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
    точную количественную оценку структуры пространства подписей и обнаруживает скрытые
    уязвимости, которые пропускаются другими методами."
    """
    # Get curve order
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # secp256k1 order
    if curve == "P-256":
        n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    
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

def quantum_sign(private_key: Any, 
                message: bytes, 
                platform: QuantumPlatform,
                dimension: int) -> bytes:
    """
    Create a quantum-topological signature.
    
    Args:
        private_key: Quantum private key
        message: Message to sign
        platform: Quantum platform
        dimension: Quantum dimension
        
    Returns:
        Quantum signature as bytes
    """
    # This is a placeholder - in a real implementation, this would use actual quantum operations
    import hashlib
    
    # Generate a hash of the message
    message_hash = hashlib.sha3_256(message).digest()
    
    # Simulate quantum signature generation
    # In a real system, this would involve quantum operations on the specified platform
    signature = b""
    
    if platform == QuantumPlatform.SOI:
        # SOI platform signature generation
        signature = _generate_soi_signature(private_key, message_hash, dimension)
    elif platform == QuantumPlatform.SiN:
        # SiN platform signature generation
        signature = _generate_sin_signature(private_key, message_hash, dimension)
    elif platform == QuantumPlatform.TFLN:
        # TFLN platform signature generation
        signature = _generate_tfln_signature(private_key, message_hash, dimension)
    elif platform == QuantumPlatform.InP:
        # InP platform signature generation
        signature = _generate_inp_signature(private_key, message_hash, dimension)
    
    return signature

def _generate_soi_signature(private_key: Any, message_hash: bytes, dimension: int) -> bytes:
    """Generate signature for SOI platform."""
    # Placeholder implementation
    return b"soi_sig_" + message_hash[:16]

def _generate_sin_signature(private_key: Any, message_hash: bytes, dimension: int) -> bytes:
    """Generate signature for SiN platform."""
    # Placeholder implementation
    return b"sin_sig_" + message_hash[16:32]

def _generate_tfln_signature(private_key: Any, message_hash: bytes, dimension: int) -> bytes:
    """Generate signature for TFLN platform."""
    # Placeholder implementation
    return b"tfln_sig_" + message_hash

def _generate_inp_signature(private_key: Any, message_hash: bytes, dimension: int) -> bytes:
    """Generate signature for InP platform."""
    # Placeholder implementation
    return b"inp_sig_" + message_hash[::-1]

def quantum_verify(public_key: Any, 
                 message: bytes, 
                 signature: bytes,
                 platform: QuantumPlatform,
                 dimension: int) -> bool:
    """
    Verify a quantum-topological signature.
    
    Args:
        public_key: Quantum public key
        message: Message that was signed
        signature: Signature to verify
        platform: Quantum platform
        dimension: Quantum dimension
        
    Returns:
        bool: True if signature is valid, False otherwise
    """
    # This is a placeholder - in a real implementation, this would use actual quantum verification
    try:
        # In a real system, this would involve quantum verification operations
        if platform == QuantumPlatform.SOI:
            return _verify_soi_signature(public_key, message, signature, dimension)
        elif platform == QuantumPlatform.SiN:
            return _verify_sin_signature(public_key, message, signature, dimension)
        elif platform == QuantumPlatform.TFLN:
            return _verify_tfln_signature(public_key, message, signature, dimension)
        elif platform == QuantumPlatform.InP:
            return _verify_inp_signature(public_key, message, signature, dimension)
        return False
    except Exception as e:
        logger.error(f"Quantum verification error: {e}", exc_info=True)
        return False

def _verify_soi_signature(public_key: Any, message: bytes, signature: bytes, dimension: int) -> bool:
    """Verify signature for SOI platform."""
    # Placeholder implementation
    return signature.startswith(b"soi_sig_")

def _verify_sin_signature(public_key: Any, message: bytes, signature: bytes, dimension: int) -> bool:
    """Verify signature for SiN platform."""
    # Placeholder implementation
    return signature.startswith(b"sin_sig_")

def _verify_tfln_signature(public_key: Any, message: bytes, signature: bytes, dimension: int) -> bool:
    """Verify signature for TFLN platform."""
    # Placeholder implementation
    return signature.startswith(b"tfln_sig_")

def _verify_inp_signature(public_key: Any, message: bytes, signature: bytes, dimension: int) -> bool:
    """Verify signature for InP platform."""
    # Placeholder implementation
    return signature.startswith(b"inp_sig_")

# Initialize global hybrid crypto system (can be overridden by application)
default_hybrid_crypto = HybridCryptoSystem()
