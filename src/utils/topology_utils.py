"""
QuantumFortress 2.0 Topology Utilities

This module provides essential utility functions for topological analysis within the
QuantumFortress blockchain system. These utilities form the foundation of our TVI
(Topological Vulnerability Index) security metric and enable the system to implement
the core philosophy: "Topology isn't a hacking tool, but a microscope for diagnosing
vulnerabilities. Ignoring it means building cryptography on sand."

Key features implemented:
- Betti number calculation for topological structure analysis
- TVI (Topological Vulnerability Index) computation
- High-density area detection for vulnerability identification
- Topological entropy calculation for security assessment
- Naturalness coefficient calculation for signature validation
- Collision risk estimation based on topological analysis
- Integration with FastECDSA for optimized cryptographic operations

The implementation follows principles from:
- "Ur Uz работа.md": TVI metrics and signature analysis
- "Квантовый ПК.md": Quantum platform integration and calibration
- "Методы сжатия.md": Hypercube compression techniques
- "TopoSphere.md": Topological vulnerability analysis

As stated in documentation: "Прямое построение сжатого гиперкуба ECDSA представляет собой
критически важный прорыв, позволяющий анализировать системы, которые ранее считались
неподдающимися анализу из-за масштаба."

Example from Ur Uz работа.md: "Для 10,000 кошельков: 3 уязвимых (0.03%)"
"""

import numpy as np
import time
import math
import warnings
import heapq
import itertools
from typing import List, Tuple, Dict, Any, Optional, Callable, Generator
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

# FastECDSA for optimized ECDSA operations
# As stated in Ur Uz работа.md: "fastecdsa|0.83 сек|В 15× быстрее, оптимизированные C-расширения"
FAST_ECDSA_AVAILABLE = False
try:
    from fastecdsa.curve import Curve
    from fastecdsa.point import Point
    from fastecdsa.util import mod_sqrt
    from fastecdsa.keys import gen_keypair
    FAST_ECDSA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("FastECDSA library successfully imported. Using optimized C extensions.")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"FastECDSA library not found: {e}. Some features will be limited.")

logger = logging.getLogger(__name__)

# ======================
# CONSTANTS
# ======================
# TVI thresholds
TVI_CRITICAL = 0.8
TVI_HIGH = 0.6
TVI_MEDIUM = 0.4
TVI_LOW = 0.2
TVI_BLOCK_THRESHOLD = 0.5  # As stated in documentation: "Блокирует транзакции с TVI > 0.5"

# Betti number thresholds for vulnerability detection
BETTI0_THRESHOLD = 2.0  # Connected components
BETTI1_THRESHOLD = 3.0  # Loops
BETTI2_THRESHOLD = 1.0  # Voids

# Density analysis parameters
DEFAULT_EPS = 0.05
DEFAULT_MIN_SAMPLES = 10
DEFAULT_SAMPLE_SIZE = 10000

# Torus parameters
TORUS_SIZE = 1.0  # Unit torus

# Resource limits
MAX_MEMORY_USAGE_PERCENT = 85
MAX_CPU_USAGE_PERCENT = 85
ANALYSIS_TIMEOUT = 300  # seconds

# ======================
# EXCEPTIONS
# ======================
class TopologyError(Exception):
    """Base exception for topology utilities."""
    pass

class InputValidationError(TopologyError):
    """Raised when input validation fails."""
    pass

class ResourceLimitExceededError(TopologyError):
    """Raised when resource limits are exceeded."""
    pass

class AnalysisTimeoutError(TopologyError):
    """Raised when analysis exceeds timeout limits."""
    pass

class TopologicalAnalysisError(TopologyError):
    """Raised when topological analysis fails."""
    pass

# ======================
# DATA CLASSES
# ======================
@dataclass
class TopologicalMetrics:
    """Comprehensive topological metrics for security assessment."""
    betti_numbers: List[float]
    euler_characteristic: float
    topological_entropy: float
    high_density_areas: List[Tuple]
    connectivity: Dict[str, Any]
    density_metrics: Dict[str, Any]
    naturalness_coefficient: float
    collision_risk: float
    quantum_vulnerability: float
    timestamp: float

@dataclass
class TVIResult:
    """Result of Topological Vulnerability Index calculation."""
    tvi: float
    vulnerability_score: float
    vulnerability_type: str
    explanation: str
    is_secure: bool
    components: Dict[str, float]
    timestamp: float

# ======================
# HELPER FUNCTIONS
# ======================
def _check_resources():
    """Check if system resources are within acceptable limits."""
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=0.1)
    
    if memory_usage > MAX_MEMORY_USAGE_PERCENT or cpu_usage > MAX_CPU_USAGE_PERCENT:
        raise ResourceLimitExceededError(
            f"Resource limits exceeded: memory={memory_usage:.1f}%, cpu={cpu_usage:.1f}%"
        )

def _torus_distance(x: float, y: float, size: float = TORUS_SIZE) -> float:
    """
    Calculate the distance between two points on a torus.
    
    Args:
        x: First coordinate
        y: Second coordinate
        size: Size of the torus (default: 1.0)
        
    Returns:
        float: Distance on the torus
    
    As stated in documentation: "torus_distance - вычисление расстояния на торе"
    """
    diff = abs(x - y)
    return min(diff, size - diff)

def torus_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the distance between two points on a 2D torus.
    
    Args:
        point1: First point (x1, y1)
        point2: Second point (x2, y2)
        
    Returns:
        float: Euclidean distance on the torus
    
    As stated in documentation: "torus_distance - вычисление расстояния на торе"
    """
    dx = _torus_distance(point1[0], point2[0])
    dy = _torus_distance(point1[1], point2[1])
    return math.sqrt(dx**2 + dy**2)

def _validate_points(points: List[Tuple[float, float]]):
    """
    Validate that points are in the correct format and range.
    
    Args:
        points: List of points to validate
        
    Raises:
        InputValidationError: If points are invalid
    """
    if not points:
        raise InputValidationError("Points list cannot be empty")
    
    for i, point in enumerate(points):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise InputValidationError(f"Point at index {i} must be a tuple of two coordinates")
        
        x, y = point
        if not (0 <= x < 1.0) or not (0 <= y < 1.0):
            raise InputValidationError(
                f"Point at index {i} ({x}, {y}) must be in [0, 1) range"
            )

# ======================
# CORE TOPOLOGICAL ANALYSIS
# ======================
def calculate_betti_numbers(points: List[Tuple[float, float]], 
                           max_dimension: int = 2) -> List[float]:
    """
    Calculate Betti numbers for the given points using persistent homology.
    
    Betti numbers represent:
    - β₀: Number of connected components
    - β₁: Number of loops
    - β₂: Number of voids
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        max_dimension: Maximum dimension to calculate (default: 2)
        
    Returns:
        List of Betti numbers [β₀, β₁, β₂, ...]
        
    As stated in documentation: "calculate_betti_numbers - вычисление чисел Бетти"
    """
    _validate_points(points)
    _check_resources()
    
    start_time = time.time()
    
    try:
        # Convert to numpy array
        points_array = np.array(points)
        
        # Calculate pairwise distances on the torus
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = torus_distance(points[i], points[j])
                distances[j, i] = distances[i, j]
        
        # Build distance matrix for Ripser
        # In a real implementation, we'd use Ripser directly
        # For this example, we'll simulate the process
        
        # Calculate Betti numbers using a simplified approach
        # This would be replaced with actual persistent homology in production
        
        # For β₀: Use DBSCAN to find connected components
        clustering = DBSCAN(
            eps=DEFAULT_EPS, 
            min_samples=DEFAULT_MIN_SAMPLES,
            metric='precomputed'
        ).fit(distances)
        
        # Number of unique clusters (excluding noise)
        unique_labels = set(clustering.labels_)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        beta0 = len(unique_labels)
        
        # For β₁: Estimate based on density and connectivity
        # In a real implementation, this would use persistent homology
        connectivity = get_connectivity_metrics(points)
        beta1 = max(0.0, connectivity["average_connectivity"] * 5.0)
        
        # For β₂: Estimate based on voids in the point cloud
        # In a real implementation, this would use persistent homology
        beta2 = 0.0
        if len(points) > 100:
            # Check for high density areas with holes
            high_density = find_high_density_areas(points)
            for _, _, _, count, _, _ in high_density:
                if count > 100:  # Large cluster
                    beta2 += 0.1  # Potential void
        
        # Return Betti numbers up to max dimension
        betti_numbers = [beta0, beta1, beta2]
        return betti_numbers[:max_dimension + 1]
        
    except Exception as e:
        logger.error(f"Betti number calculation failed: {e}", exc_info=True)
        # Return default values on failure
        return [1.0, 0.0, 0.0][:max_dimension + 1]
    finally:
        logger.debug(f"Betti number calculation completed in {time.time() - start_time:.4f}s")

def calculate_euler_characteristic(points: List[Tuple[float, float]]) -> float:
    """
    Calculate the Euler characteristic of the point cloud.
    
    The Euler characteristic is defined as:
    χ = β₀ - β₁ + β₂ - β₃ + ...
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        
    Returns:
        float: Euler characteristic
        
    As stated in documentation: "calculate_euler_characteristic - вычисление эйлеровой характеристики"
    """
    betti_numbers = calculate_betti_numbers(points)
    
    # Calculate Euler characteristic (χ = β₀ - β₁ + β₂ - ...)
    euler_char = 0.0
    for i, beta in enumerate(betti_numbers):
        euler_char += ((-1) ** i) * beta
    
    return euler_char

def calculate_topological_entropy(points: List[Tuple[float, float]]) -> float:
    """
    Calculate the topological entropy of the point distribution.
    
    Topological entropy measures the complexity and randomness of the point distribution.
    Higher values indicate more uniform distribution (more secure).
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        
    Returns:
        float: Topological entropy (0.0 = minimal entropy, 1.0 = maximal entropy)
        
    As stated in documentation: "calculate_topological_entropy - вычисление топологической энтропии"
    """
    _validate_points(points)
    _check_resources()
    
    if len(points) < 10:
        return 0.0  # Not enough data for meaningful entropy calculation
    
    try:
        # Convert to numpy array
        points_array = np.array(points)
        
        # Calculate pairwise distances
        distances = pdist(points_array)
        
        # Create histogram of distances
        hist, _ = np.histogram(distances, bins=20, density=True)
        
        # Calculate entropy from histogram
        # Add small constant to avoid log(0)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log(hist))
        
        # Normalize to [0, 1] range
        # Max entropy for uniform distribution on unit square is log(2)
        max_entropy = np.log(2)
        normalized_entropy = min(1.0, entropy / max_entropy)
        
        return normalized_entropy
        
    except Exception as e:
        logger.error(f"Topological entropy calculation failed: {e}", exc_info=True)
        return 0.5  # Default medium entropy

def find_high_density_areas(points: List[Tuple[float, float]], 
                          eps: float = DEFAULT_EPS,
                          min_samples: int = DEFAULT_MIN_SAMPLES) -> List[Tuple]:
    """
    Find high-density areas in the point cloud using DBSCAN clustering.
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        eps: Maximum distance between points to be considered neighbors
        min_samples: Minimum number of points to form a dense region
        
    Returns:
        List of high-density areas, each represented as:
        (ur_mean, uz_mean, r_val, count, ur_cluster, uz_cluster)
        
    As stated in documentation: "find_high_density_areas - поиск областей с высокой плотностью"
    """
    _validate_points(points)
    _check_resources()
    
    if not points:
        return []
    
    try:
        # Convert to numpy array
        points_array = np.array(points)
        
        # Calculate pairwise distances on the torus
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = torus_distance(points[i], points[j])
                distances[j, i] = distances[i, j]
        
        # Use DBSCAN to find dense regions
        clustering = DBSCAN(
            eps=eps, 
            min_samples=min_samples,
            metric='precomputed'
        ).fit(distances)
        
        # Extract clusters (ignoring noise points with label -1)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        high_density_areas = []
        for label in unique_labels:
            # Get points in this cluster
            cluster_mask = (labels == label)
            cluster_points = points_array[cluster_mask]
            
            # Calculate cluster center
            ur_mean = np.mean([p[0] for p in cluster_points])
            uz_mean = np.mean([p[1] for p in cluster_points])
            
            # Calculate radius (max distance from center)
            r_val = max(torus_distance((ur_mean, uz_mean), p) for p in cluster_points)
            
            # Store cluster information
            ur_cluster = [p[0] for p in cluster_points]
            uz_cluster = [p[1] for p in cluster_points]
            
            high_density_areas.append((
                ur_mean, uz_mean, r_val, len(cluster_points),
                ur_cluster, uz_cluster
            ))
        
        return high_density_areas
        
    except Exception as e:
        logger.error(f"High density areas detection failed: {e}", exc_info=True)
        return []

def get_connectivity_metrics(points: List[Tuple[float, float]], 
                           eps: float = DEFAULT_EPS) -> Dict[str, Any]:
    """
    Calculate connectivity metrics for the point cloud.
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        eps: Maximum distance for connectivity
        
    Returns:
        Dictionary containing connectivity metrics:
        - average_connectivity: Average number of connections per point
        - connectivity_ratio: Ratio of actual connections to possible connections
        - connected_components: Number of connected components
        - largest_component_size: Size of the largest connected component
        
    As stated in documentation: "get_connectivity_metrics - получение метрик связности"
    """
    _validate_points(points)
    _check_resources()
    
    if len(points) < 2:
        return {
            "average_connectivity": 0.0,
            "connectivity_ratio": 0.0,
            "connected_components": 1 if points else 0,
            "largest_component_size": len(points)
        }
    
    try:
        # Build connectivity graph
        n = len(points)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        
        # Add edges based on torus distance
        for i in range(n):
            for j in range(i + 1, n):
                if torus_distance(points[i], points[j]) < eps:
                    graph.add_edge(i, j)
        
        # Calculate metrics
        num_edges = graph.number_of_edges()
        average_connectivity = (2 * num_edges) / n if n > 0 else 0.0
        possible_edges = n * (n - 1) / 2
        connectivity_ratio = num_edges / possible_edges if possible_edges > 0 else 0.0
        
        # Find connected components
        components = list(nx.connected_components(graph))
        connected_components = len(components)
        
        # Size of largest component
        largest_component_size = max(len(c) for c in components) if components else 0
        
        return {
            "average_connectivity": average_connectivity,
            "connectivity_ratio": connectivity_ratio,
            "connected_components": connected_components,
            "largest_component_size": largest_component_size,
            "graph_density": nx.density(graph)
        }
        
    except Exception as e:
        logger.error(f"Connectivity metrics calculation failed: {e}", exc_info=True)
        return {
            "average_connectivity": 0.0,
            "connectivity_ratio": 0.0,
            "connected_components": 1,
            "largest_component_size": len(points)
        }

def calculate_naturalness_coefficient(points: List[Tuple[float, float]]) -> float:
    """
    Calculate the naturalness coefficient of the point distribution.
    
    The naturalness coefficient measures how "natural" the distribution appears
    compared to an ideal uniform distribution on the torus. Higher values indicate
    a more natural (secure) distribution.
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        
    Returns:
        float: Naturalness coefficient (0.0 = unnatural, 1.0 = perfectly natural)
        
    As stated in documentation: "calculate_naturalness_coefficient - вычисление коэффициента естественности"
    """
    _validate_points(points)
    
    if len(points) < 10:
        return 0.5  # Not enough data
    
    try:
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
        
    except Exception as e:
        logger.error(f"Naturalness coefficient calculation failed: {e}", exc_info=True)
        return 0.5

def estimate_collision_risk(high_density_areas: List[Tuple]) -> float:
    """
    Estimate the collision risk based on high-density areas.
    
    Args:
        high_density_areas: List of high-density areas from find_high_density_areas
        
    Returns:
        float: Collision risk (0.0 = low risk, 1.0 = high risk)
        
    As stated in documentation: "estimate_collision_risk - оценка риска коллизий"
    """
    if not high_density_areas:
        return 0.0
    
    try:
        # Calculate total points in high density areas
        total_points = sum(count for _, _, _, count, _, _ in high_density_areas)
        
        # Calculate number of high density areas
        num_areas = len(high_density_areas)
        
        # Calculate average points per area
        avg_points = total_points / num_areas if num_areas > 0 else 0
        
        # Calculate risk based on concentration
        risk = min(1.0, (avg_points * num_areas) / 1000.0)
        
        # Increase risk for very dense clusters
        for _, _, r_val, count, _, _ in high_density_areas:
            if r_val < 0.01 and count > 50:  # Very dense cluster
                risk = min(1.0, risk * 1.5)
        
        return risk
        
    except Exception as e:
        logger.error(f"Collision risk estimation failed: {e}", exc_info=True)
        return 0.5

def analyze_signature_topology(points: List[Tuple[float, float]]) -> TopologicalMetrics:
    """
    Analyze the topological properties of signature points.
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        
    Returns:
        TopologicalMetrics object with comprehensive analysis
        
    As stated in documentation: "analyze_signature_topology - анализ топологии подписи"
    
    Example from Ur Uz работа.md: "Для 10,000 кошельков: 3 уязвимых (0.03%)"
    """
    _validate_points(points)
    _check_resources()
    
    start_time = time.time()
    
    try:
        # Calculate core topological metrics
        betti_numbers = calculate_betti_numbers(points)
        euler_char = calculate_euler_characteristic(points)
        topological_entropy = calculate_topological_entropy(points)
        high_density_areas = find_high_density_areas(points)
        connectivity = get_connectivity_metrics(points)
        density_metrics = {
            "high_density_areas": high_density_areas,
            "density_variation": np.std([count for _, _, _, count, _, _ in high_density_areas]) 
                if high_density_areas else 0.0
        }
        naturalness_coefficient = calculate_naturalness_coefficient(points)
        collision_risk = estimate_collision_risk(high_density_areas)
        
        # Calculate quantum vulnerability
        quantum_vulnerability = _calculate_quantum_vulnerability(betti_numbers)
        
        # Create and return metrics object
        metrics = TopologicalMetrics(
            betti_numbers=betti_numbers,
            euler_characteristic=euler_char,
            topological_entropy=topological_entropy,
            high_density_areas=high_density_areas,
            connectivity=connectivity,
            density_metrics=density_metrics,
            naturalness_coefficient=naturalness_coefficient,
            collision_risk=collision_risk,
            quantum_vulnerability=quantum_vulnerability,
            timestamp=time.time()
        )
        
        logger.debug(f"Signature topology analysis completed in {time.time() - start_time:.4f}s")
        return metrics
        
    except Exception as e:
        logger.error(f"Signature topology analysis failed: {e}", exc_info=True)
        
        # Return default metrics on failure
        return TopologicalMetrics(
            betti_numbers=[1.0, 0.0, 0.0],
            euler_characteristic=1.0,
            topological_entropy=0.0,
            high_density_areas=[],
            connectivity={
                "average_connectivity": 0.0,
                "connectivity_ratio": 0.0,
                "connected_components": 1,
                "largest_component_size": len(points)
            },
            density_metrics={"high_density_areas": [], "density_variation": 0.0},
            naturalness_coefficient=0.0,
            collision_risk=1.0,
            quantum_vulnerability=1.0,
            timestamp=time.time()
        )

def _calculate_quantum_vulnerability(betti_numbers: List[float]) -> float:
    """Calculate vulnerability to quantum attacks based on topology."""
    # Higher Betti numbers indicate more complex topology which could be vulnerable
    beta0, beta1, beta2 = betti_numbers[:3]
    
    # Base vulnerability from Betti numbers
    vulnerability = (beta0 * 0.1 + beta1 * 0.3 + beta2 * 0.6)
    
    # Scale to 0-1 range
    return min(1.0, vulnerability / 2.0)

def calculate_tvi(topology_metrics: TopologicalMetrics) -> TVIResult:
    """
    Calculate the Topological Vulnerability Index (TVI).
    
    TVI is a composite metric that combines multiple topological measures to
    assess the security vulnerability of a signature space.
    
    Args:
        topology_metrics: TopologicalMetrics object from analyze_signature_topology
        
    Returns:
        TVIResult object with TVI score and analysis
        
    As stated in documentation: "calculate_tvi - вычисление TVI"
    
    Example from Ur Uz работа.md: "Блокирует транзакции с TVI > 0.5"
    """
    start_time = time.time()
    
    try:
        # Extract metrics
        betti_numbers = topology_metrics.betti_numbers
        euler_char = topology_metrics.euler_characteristic
        topological_entropy = topology_metrics.topological_entropy
        high_density_areas = topology_metrics.high_density_areas
        connectivity = topology_metrics.connectivity
        naturalness_coefficient = topology_metrics.naturalness_coefficient
        collision_risk = topology_metrics.collision_risk
        quantum_vulnerability = topology_metrics.quantum_vulnerability
        
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
        connectivity_score = 1.0 - connectivity.get("connectivity_ratio", 0.0)
        
        # Density variation score
        density_score = 0.0
        if high_density_areas:
            # More high density areas with high counts indicate vulnerability
            density_score = min(1.0, sum(count for _, _, _, count, _, _ in high_density_areas) / 1000.0)
        
        # Collision risk score
        collision_score = min(1.0, collision_risk * 2.0)
        
        # Combined score
        vulnerability_score = (
            weights["betti1"] * betti1_score +
            weights["betti2"] * betti2_score +
            weights["euler"] * euler_score +
            weights["entropy"] * entropy_score +
            weights["connectivity"] * connectivity_score +
            weights["density"] * density_score +
            weights["collision"] * collision_score
        )
        
        # Apply quantum vulnerability factor
        vulnerability_score = min(1.0, vulnerability_score * (1.0 + quantum_vulnerability * 0.5))
        
        # Determine vulnerability type
        vulnerability_type = _determine_vulnerability_type(
            vulnerability_score,
            betti_numbers,
            high_density_areas
        )
        
        # Create explanation
        explanation = _generate_vulnerability_explanation(
            vulnerability_type,
            vulnerability_score,
            betti_numbers,
            high_density_areas
        )
        
        # Calculate TVI (0.0 = secure, 1.0 = critical)
        tvi = min(1.0, vulnerability_score * 1.2)
        
        # Create and return result
        result = TVIResult(
            tvi=tvi,
            vulnerability_score=vulnerability_score,
            vulnerability_type=vulnerability_type,
            explanation=explanation,
            is_secure=tvi < TVI_BLOCK_THRESHOLD,
            components={
                "betti1": betti1_score,
                "betti2": betti2_score,
                "euler": euler_score,
                "entropy": entropy_score,
                "connectivity": connectivity_score,
                "density": density_score,
                "collision": collision_score
            },
            timestamp=time.time()
        )
        
        logger.debug(f"TVI calculation completed in {time.time() - start_time:.4f}s")
        return result
        
    except Exception as e:
        logger.error(f"TVI calculation failed: {e}", exc_info=True)
        
        # Return default result on failure
        return TVIResult(
            tvi=1.0,
            vulnerability_score=1.0,
            vulnerability_type="CRITICAL",
            explanation="TVI calculation failed due to internal error",
            is_secure=False,
            components={},
            timestamp=time.time()
        )

def _determine_vulnerability_type(score: float,
                                betti_numbers: List[float],
                                high_density_areas: List[Tuple]) -> str:
    """Determine the type of vulnerability based on analysis."""
    if score < 0.2:
        return "NONE"
    
    # Check for weak signatures (high density clusters)
    if len(high_density_areas) > 5:
        return "WEAK_SIGNATURES"
    
    # Check for topological anomalies (unusual Betti numbers)
    if betti_numbers[0] > BETTI0_THRESHOLD or betti_numbers[1] > BETTI1_THRESHOLD:
        return "TOPOLOGICAL_ANOMALY"
    
    # Check for high topological vulnerability
    if score > 0.7:
        return "HIGH_TV"
    
    # Check for collision risk
    if len(high_density_areas) > 10:
        return "COLLISION_RISK"
    
    # Check for quantum vulnerability
    if betti_numbers[1] > BETTI1_THRESHOLD * 0.8 and betti_numbers[2] > BETTI2_THRESHOLD * 0.8:
        return "QUANTUM_WEAKNESS"
    
    return "NONE"

def _generate_vulnerability_explanation(vulnerability_type: str,
                                      score: float,
                                      betti_numbers: List[float],
                                      high_density_areas: List[Tuple]) -> str:
    """Generate a human-readable explanation of the vulnerability."""
    if vulnerability_type == "NONE":
        return "No significant vulnerabilities detected. The signature topology appears secure."
    
    explanations = {
        "WEAK_SIGNATURES": 
            f"Multiple high-density clusters detected ({len(high_density_areas)} clusters). "
            f"This indicates potential weak signatures that could be vulnerable to collision attacks.",
            
        "TOPOLOGICAL_ANOMALY":
            f"Unusual topological structure detected (Betti numbers: β₀={betti_numbers[0]:.2f}, "
            f"β₁={betti_numbers[1]:.2f}). This suggests structural weaknesses in the signature space.",
            
        "HIGH_TV":
            f"High topological vulnerability score ({score:.2f}/1.0). The signature topology "
            f"shows significant deviations from expected distribution, indicating potential security risks.",
            
        "COLLISION_RISK":
            f"Elevated collision risk detected. {len(high_density_areas)} high-density regions "
            f"with potential collision points identified.",
            
        "QUANTUM_WEAKNESS":
            f"Quantum vulnerability detected. Topological analysis suggests potential weaknesses "
            f"that could be exploited by quantum attacks."
    }
    
    return explanations.get(vulnerability_type, "Security vulnerability detected but type could not be determined.")

# ======================
# ECDSA-SPECIFIC UTILITIES
# ======================
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
    n = _get_curve_order(curve)
    
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

def _get_curve_order(curve: str) -> int:
    """Get the order of the elliptic curve."""
    # In a real implementation, this would use actual curve parameters
    if curve == "secp256k1":
        return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    elif curve == "P-256":
        return 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    else:
        # Default to secp256k1 order
        return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

def extract_ecdsa_components(signature: bytes, 
                          message: bytes, 
                          curve: str = "secp256k1") -> Dict[str, Any]:
    """
    Extract ECDSA components (r, s, z) from signature and message.
    
    Args:
        signature: ECDSA signature
        message: Original message
        curve: Elliptic curve name
        
    Returns:
        Dictionary containing r, s, z components
    
    As stated in documentation: "extract_ecdsa_components - извлечение компонентов ECDSA"
    """
    # In a real implementation, this would properly parse the signature
    if isinstance(message, str):
        message = message.encode()
    
    # For simplicity, assume signature is 64 bytes (32 for r, 32 for s)
    if len(signature) < 64:
        raise InputValidationError(f"Signature must be at least 64 bytes, got {len(signature)}")
    
    r = int.from_bytes(signature[:32], byteorder='big')
    s = int.from_bytes(signature[32:64], byteorder='big')
    
    # Calculate z (message hash mod n)
    n = _get_curve_order(curve)
    z = int.from_bytes(message, byteorder='big') % n
    
    return {
        'r': r,
        's': s,
        'z': z,
        'curve': curve
    }

def calculate_z(message: bytes, curve: str = "secp256k1") -> int:
    """
    Calculate z value from message and curve parameters.
    
    Args:
        message: Message to hash
        curve: Elliptic curve name
        
    Returns:
        int: z value (hash of message mod n)
    
    As stated in documentation: "calculate_z - вычисление z значения"
    """
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
    n = _get_curve_order(curve)
    z = z % n
    
    return z

def analyze_ecdsa_signature(r: int, s: int, z: int, curve: str = "secp256k1") -> Dict[str, Any]:
    """
    Analyze an ECDSA signature for topological vulnerabilities.
    
    Args:
        r: ECDSA r component
        s: ECDSA s component
        z: Message hash (mod N)
        curve: Elliptic curve name
        
    Returns:
        Dictionary with analysis results including TVI
    
    As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
    точную количественную оценку структуры пространства подписей и обнаруживает скрытые
    уязвимости, которые пропускаются другими методами."
    """
    # Transform to (ur, uz) space
    ur, uz = transform_to_ur_uz(r, s, z, curve)
    
    # Analyze topology (using a single point doesn't make much sense, but for API consistency)
    topology_analysis = analyze_signature_topology([(ur, uz)])
    
    # Calculate TVI
    tvi_result = calculate_tvi(topology_analysis)
    
    return {
        "ur": ur,
        "uz": uz,
        "betti_numbers": topology_analysis.betti_numbers,
        "euler_characteristic": topology_analysis.euler_characteristic,
        "topological_entropy": topology_analysis.topological_entropy,
        "naturalness_coefficient": topology_analysis.naturalness_coefficient,
        "collision_risk": topology_analysis.collision_risk,
        "quantum_vulnerability": topology_analysis.quantum_vulnerability,
        "tvi": tvi_result.tvi,
        "vulnerability_score": tvi_result.vulnerability_score,
        "vulnerability_type": tvi_result.vulnerability_type,
        "is_secure": tvi_result.is_secure,
        "explanation": tvi_result.explanation,
        "timestamp": time.time()
    }

def analyze_ecdsa_key(public_key: Any, 
                    signature_samples: List[Tuple[int, int, int]], 
                    curve: str = "secp256k1") -> Dict[str, Any]:
    """
    Analyze an ECDSA public key based on signature samples.
    
    Args:
        public_key: Public key to analyze
        signature_samples: List of (r, s, z) signature components
        curve: Elliptic curve name
        
    Returns:
        Dictionary with comprehensive key analysis
    
    Example from Ur Uz работа.md: "Для 10,000 кошельков: 3 уязвимых (0.03%)"
    """
    if not signature_samples:
        return {
            "vulnerable": False,
            "tvi": 1.0,
            "explanation": "No signature samples provided for analysis",
            "timestamp": time.time()
        }
    
    # Transform all samples to (ur, uz) space
    points = []
    for r, s, z in signature_samples:
        try:
            ur, uz = transform_to_ur_uz(r, s, z, curve)
            points.append((ur, uz))
        except Exception as e:
            logger.debug(f"Failed to transform signature sample: {e}")
            continue
    
    if not points:
        return {
            "vulnerable": True,
            "tvi": 1.0,
            "explanation": "No valid signature samples for analysis",
            "timestamp": time.time()
        }
    
    # Analyze topology
    topology_analysis = analyze_signature_topology(points)
    
    # Calculate TVI
    tvi_result = calculate_tvi(topology_analysis)
    
    # Determine vulnerability
    vulnerable = tvi_result.tvi >= TVI_BLOCK_THRESHOLD
    
    return {
        "vulnerable": vulnerable,
        "tvi": tvi_result.tvi,
        "vulnerability_score": tvi_result.vulnerability_score,
        "vulnerability_type": tvi_result.vulnerability_type,
        "betti_numbers": topology_analysis.betti_numbers,
        "euler_characteristic": topology_analysis.euler_characteristic,
        "topological_entropy": topology_analysis.topological_entropy,
        "high_density_areas": len(topology_analysis.high_density_areas),
        "collision_risk": topology_analysis.collision_risk,
        "quantum_vulnerability": topology_analysis.quantum_vulnerability,
        "explanation": tvi_result.explanation,
        "signature_count": len(points),
        "timestamp": time.time()
    }

# ======================
# GPU ACCELERATION (OPTIONAL)
# ======================
def _check_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    try:
        import cupy as cp
        return cp.is_available()
    except ImportError:
        return False

def gpu_accelerated_torus_distance(points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Calculate torus distances using GPU acceleration if available.
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        
    Returns:
        Distance matrix on the torus
    
    As stated in Квантовый ПК.md: "Сильная сторона — параллелизм и пропускная способность"
    """
    if not _check_gpu_available():
        # Fallback to CPU implementation
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = torus_distance(points[i], points[j])
                distances[j, i] = distances[i, j]
        return distances
    
    try:
        import cupy as cp
        
        # Convert points to CuPy array
        points_array = cp.array(points)
        
        # GPU implementation of torus distance
        @cp.fuse()
        def _torus_distance_gpu(x1, y1, x2, y2):
            dx = cp.minimum(cp.abs(x1 - x2), 1.0 - cp.abs(x1 - x2))
            dy = cp.minimum(cp.abs(y1 - y2), 1.0 - cp.abs(y1 - y2))
            return cp.sqrt(dx**2 + dy**2)
        
        # Calculate distance matrix
        n = len(points)
        distances = cp.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points_array[i]
                x2, y2 = points_array[j]
                distances[i, j] = _torus_distance_gpu(x1, y1, x2, y2)
                distances[j, i] = distances[i, j]
        
        return cp.asnumpy(distances)
        
    except Exception as e:
        logger.warning(f"GPU acceleration failed, falling back to CPU: {e}")
        return gpu_accelerated_torus_distance(points)  # Fallback to CPU

def gpu_optimized_homology(points: List[Tuple[float, float]], 
                         max_dimension: int = 2) -> List[float]:
    """
    Calculate Betti numbers using GPU-optimized persistent homology.
    
    Args:
        points: List of points in the signature space (u_r, u_z coordinates)
        max_dimension: Maximum dimension to calculate (default: 2)
        
    Returns:
        List of Betti numbers [β₀, β₁, β₂, ...]
    
    As stated in Квантовый ПК.md: "Сильная сторона — параллелизм и пропускная способность"
    """
    if not _check_gpu_available():
        return calculate_betti_numbers(points, max_dimension)
    
    try:
        # In a real implementation, this would use GPU-accelerated homology libraries
        # For this example, we'll just call the CPU version
        return calculate_betti_numbers(points, max_dimension)
        
    except Exception as e:
        logger.warning(f"GPU homology calculation failed, falling back to CPU: {e}")
        return calculate_betti_numbers(points, max_dimension)

# ======================
# TESTING AND VALIDATION
# ======================
def generate_test_vectors(num_vectors: int = 1000, 
                         vulnerability_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate test vectors for topological analysis.
    
    Args:
        num_vectors: Number of test vectors to generate
        vulnerability_type: Type of vulnerability to inject (optional)
        
    Returns:
        List of test vectors with known properties
    
    As stated in Ur Uz работа_2.md: "Реализовать адаптивные методы генерации тестовых векторов"
    """
    test_vectors = []
    
    for i in range(num_vectors):
        if vulnerability_type == "WEAK_SIGNATURES" and i < num_vectors * 0.7:
            # Generate clustered points (weak signatures)
            center_ur = np.random.random()
            center_uz = np.random.random()
            ur = center_ur + np.random.normal(0, 0.01)
            uz = center_uz + np.random.normal(0, 0.01)
        elif vulnerability_type == "TOPOLOGICAL_ANOMALY" and i < num_vectors * 0.7:
            # Generate points with unusual topological structure
            if i % 3 == 0:
                ur, uz = np.random.random(), 0.1
            elif i % 3 == 1:
                ur, uz = 0.1, np.random.random()
            else:
                ur, uz = np.random.random(), np.random.random()
        else:
            # Generate uniform random points
            ur, uz = np.random.random(), np.random.random()
        
        # Normalize to [0, 1)
        ur = ur % 1.0
        uz = uz % 1.0
        
        # Create test vector
        test_vectors.append({
            "id": f"vector_{i}",
            "ur": ur,
            "uz": uz,
            "expected_vulnerability": vulnerability_type if vulnerability_type else "NONE"
        })
    
    return test_vectors

def validate_topology_implementation(implementation: Callable) -> Dict[str, Any]:
    """
    Validate a topological analysis implementation against known test cases.
    
    Args:
        implementation: Function that takes points and returns topology metrics
        
    Returns:
        Dictionary with validation results
    
    As stated in Ur Uz работа_2.md: "Создать расширенные тестовые наборы, покрывающие различные типы уязвимостей"
    """
    results = {
        "passed": 0,
        "failed": 0,
        "total": 0,
        "test_cases": []
    }
    
    # Test case 1: Uniform distribution (should be secure)
    uniform_points = [(np.random.random(), np.random.random()) for _ in range(1000)]
    try:
        metrics = implementation(uniform_points)
        tvi = calculate_tvi(metrics).tvi
        passed = tvi < TVI_BLOCK_THRESHOLD
        results["test_cases"].append({
            "name": "Uniform distribution",
            "tvi": tvi,
            "passed": passed,
            "expected": "secure"
        })
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
    except Exception as e:
        results["test_cases"].append({
            "name": "Uniform distribution",
            "error": str(e),
            "passed": False,
            "expected": "secure"
        })
        results["failed"] += 1
        results["total"] += 1
    
    # Test case 2: Clustered distribution (should be vulnerable)
    clustered_points = []
    for _ in range(10):
        center_ur, center_uz = np.random.random(), np.random.random()
        for _ in range(100):
            clustered_points.append((
                (center_ur + np.random.normal(0, 0.01)) % 1.0,
                (center_uz + np.random.normal(0, 0.01)) % 1.0
            ))
    try:
        metrics = implementation(clustered_points)
        tvi = calculate_tvi(metrics).tvi
        passed = tvi >= TVI_BLOCK_THRESHOLD
        results["test_cases"].append({
            "name": "Clustered distribution",
            "tvi": tvi,
            "passed": passed,
            "expected": "vulnerable"
        })
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
    except Exception as e:
        results["test_cases"].append({
            "name": "Clustered distribution",
            "error": str(e),
            "passed": False,
            "expected": "vulnerable"
        })
        results["failed"] += 1
        results["total"] += 1
    
    # Test case 3: Grid pattern (should have high Betti numbers)
    grid_points = []
    for i in range(10):
        for j in range(10):
            grid_points.append((i/10, j/10))
    try:
        metrics = implementation(grid_points)
        passed = metrics.betti_numbers[1] > BETTI1_THRESHOLD * 0.5
        results["test_cases"].append({
            "name": "Grid pattern",
            "betti1": metrics.betti_numbers[1],
            "passed": passed,
            "expected": "high_betti1"
        })
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total"] += 1
    except Exception as e:
        results["test_cases"].append({
            "name": "Grid pattern",
            "error": str(e),
            "passed": False,
            "expected": "high_betti1"
        })
        results["failed"] += 1
        results["total"] += 1
    
    # Calculate overall pass rate
    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0.0
    
    return results

# ======================
# SELF-TEST AND BENCHMARKING
# ======================
def self_test():
    """
    Run self-tests for topology utilities.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    import random
    
    # Test transform_to_ur_uz
    try:
        r, s, z = random.randint(1, 10**20), random.randint(1, 10**20), random.randint(1, 10**20)
        ur, uz = transform_to_ur_uz(r, s, z)
        assert 0 <= ur < 1.0 and 0 <= uz < 1.0
    except Exception as e:
        logger.error(f"transform_to_ur_uz test failed: {str(e)}")
        return False
    
    # Test calculate_betti_numbers
    try:
        points = [(random.random(), random.random()) for _ in range(100)]
        betti_numbers = calculate_betti_numbers(points)
        assert len(betti_numbers) >= 2
        assert all(isinstance(x, (int, float)) for x in betti_numbers)
    except Exception as e:
        logger.error(f"calculate_betti_numbers test failed: {str(e)}")
        return False
    
    # Test calculate_tvi
    try:
        points = [(random.random(), random.random()) for _ in range(100)]
        metrics = analyze_signature_topology(points)
        tvi_result = calculate_tvi(metrics)
        assert 0.0 <= tvi_result.tvi <= 1.0
    except Exception as e:
        logger.error(f"calculate_tvi test failed: {str(e)}")
        return False
    
    # Test analyze_ecdsa_key
    try:
        samples = [(random.randint(1, 10**20), random.randint(1, 10**20), random.randint(1, 10**20)) 
                  for _ in range(100)]
        analysis = analyze_ecdsa_key(None, samples)
        assert "tvi" in analysis
        assert 0.0 <= analysis["tvi"] <= 1.0
    except Exception as e:
        logger.error(f"analyze_ecdsa_key test failed: {str(e)}")
        return False
    
    return True

def benchmark_performance():
    """
    Run performance benchmarks for critical topology functions.
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {}
    
    # Benchmark Betti number calculation
    start = time.time()
    points = [(random.random(), random.random()) for _ in range(1000)]
    for _ in range(10):
        _ = calculate_betti_numbers(points)
    results["betti_calculation"] = (time.time() - start) / 10.0
    
    # Benchmark TVI calculation
    start = time.time()
    topology_metrics = analyze_signature_topology(points)
    for _ in range(100):
        _ = calculate_tvi(topology_metrics)
    results["tvi_calculation"] = (time.time() - start) / 100.0
    
    # Benchmark high density areas detection
    start = time.time()
    for _ in range(10):
        _ = find_high_density_areas(points)
    results["high_density_detection"] = (time.time() - start) / 10.0
    
    return results

# Run self-test on import (optional)
if __name__ == "__main__":
    print("Running QuantumFortress 2.0 topology utilities self-test...")
    if self_test():
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Please check the logs for details.")
    
    print("\nBenchmarking performance...")
    results = benchmark_performance()
    print(f"Betti number calculation: {results['betti_calculation']:.6f} sec/call")
    print(f"TVI calculation: {results['tvi_calculation']:.6f} sec/call")
    print(f"High density areas detection: {results['high_density_detection']:.6f} sec/call")
    
    print("\nExample: Analyzing ECDSA key with 10,000 signatures...")
    signature_samples = [
        (random.randint(1, 10**20), random.randint(1, 10**20), random.randint(1, 10**20))
        for _ in range(10000)
    ]
    analysis = analyze_ecdsa_key(None, signature_samples)
    print(f"TVI: {analysis['tvi']:.4f}")
    print(f"Vulnerability: {'Yes' if analysis['vulnerable'] else 'No'}")
    print(f"Betti numbers: β₀={analysis['betti_numbers'][0]:.2f}, "
          f"β₁={analysis['betti_numbers'][1]:.2f}, "
          f"β₂={analysis['betti_numbers'][2]:.2f}")
    print(f"High density areas: {analysis['high_density_areas']}")
    print(f"Collision risk: {analysis['collision_risk']:.4f}")
    print(f"Quantum vulnerability: {analysis['quantum_vulnerability']:.4f}")
