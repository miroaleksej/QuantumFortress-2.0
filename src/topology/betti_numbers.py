"""
betti_numbers.py - Module for calculating and analyzing Betti numbers for cryptographic security.

This module implements the key principle from Ur Uz работа.md: "Применение чисел Бетти к анализу
ECDSA-Torus предоставляет точную количественную оценку структуры пространства подписей и обнаруживает
скрытые уязвимости, которые пропускаются другими методами."

The module provides tools to:
- Calculate Betti numbers for point clouds on the torus
- Analyze topological structure of ECDSA signature spaces
- Detect vulnerabilities through topological anomalies
- Provide quantitative metrics for cryptographic security

Author: Quantum Topology Research Group
Institution: Tambov Research Institute of Quantum Topology
Email: miro-aleksej@yandex.ru
"""

import numpy as np
import gudhi
import math
from typing import List, Tuple, Dict, Any, Optional
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Expected Betti numbers for secure ECDSA-Torus implementation
SECURE_BETTI_NUMBERS = {
    0: 1.0,  # One connected component
    1: 2.0,  # Two independent cycles (horizontal and vertical)
    2: 1.0   # One internal void
}

# Thresholds for security analysis
BETTI_DEVIATION_THRESHOLD = 0.5  # Maximum acceptable deviation from expected values
TVI_SECURE_THRESHOLD = 0.5        # TVI threshold for secure implementation
TVI_WARNING_THRESHOLD = 0.7       # TVI threshold for warning state
CURVATURE_SMOOTH_THRESHOLD = 0.3  # Threshold for curvature analysis
ENTROPY_THRESHOLD = 0.15          # Threshold for topological entropy

class TopologicalMetrics:
    """
    Container for topological metrics used in security analysis.
    
    This class implements the comprehensive topological security assessment
    as described in Ur Uz работа.md and topology_utils.txt.
    """
    def __init__(self,
                 betti_numbers: List[float],
                 euler_characteristic: float,
                 topological_entropy: float,
                 naturalness_coefficient: float,
                 tvi: float,
                 is_secure: bool,
                 vulnerability_type: str,
                 explanation: str,
                 timestamp: float):
        self.betti_numbers = betti_numbers
        self.euler_characteristic = euler_characteristic
        self.topological_entropy = topological_entropy
        self.naturalness_coefficient = naturalness_coefficient
        self.tvi = tvi
        self.is_secure = is_secure
        self.vulnerability_type = vulnerability_type
        self.explanation = explanation
        self.timestamp = timestamp

def _torus_distance(point1: Tuple[float, float], point2: Tuple[float, float], 
                  max_angle: float = 2 * np.pi) -> float:
    """
    Calculate distance on the torus surface with proper wrapping.
    
    Args:
        point1: First point (u_r, u_z) in [0,1) space
        point2: Second point (u_r, u_z) in [0,1) space
        max_angle: Maximum angle (2π for full circle)
        
    Returns:
        float: Geodesic distance on the torus
    
    Reference: Ur Uz работа_2.md - Proper torus metric implementation
    """
    u_r1, u_z1 = point1
    u_r2, u_z2 = point2
    
    # Convert to radians for proper torus distance
    theta1, phi1 = u_r1 * max_angle, u_z1 * max_angle
    theta2, phi2 = u_z2 * max_angle, u_z2 * max_angle
    
    # Calculate minimal distances considering wrapping
    d_theta = min(abs(theta1 - theta2), max_angle - abs(theta1 - theta2))
    d_phi = min(abs(phi1 - phi2), max_angle - abs(phi1 - phi2))
    
    # Euclidean distance on the flat torus
    return np.sqrt(d_theta**2 + d_phi**2)

def _create_torus_rips_complex(points: List[Tuple[float, float]], 
                             max_edge_length: float) -> gudhi.SimplexTree:
    """
    Create a Rips complex for points on a torus with proper distance metric.
    
    Args:
        points: List of points in (u_r, u_z) space
        max_edge_length: Maximum edge length for the complex
        
    Returns:
        gudhi.SimplexTree: Rips complex for the point cloud
    
    Reference: Implementation based on Ur Uz работа.md and Prototype_TopoMine.txt
    """
    # Create distance matrix with torus metric
    n = len(points)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = _torus_distance(points[i], points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Create Rips complex
    rips = gudhi.RipsComplex(
        distance_matrix=dist_matrix,
        max_edge_length=max_edge_length
    )
    
    return rips.create_simplex_tree(max_dimension=3)

def _estimate_betti_numbers(points: List[Tuple[float, float]], 
                          max_dimension: int = 2) -> List[float]:
    """
    Estimate Betti numbers for the point cloud on the torus.
    
    Args:
        points: List of points in (u_r, u_z) space
        max_dimension: Maximum dimension to calculate
        
    Returns:
        List[float]: Estimated Betti numbers [β₀, β₁, β₂, ...]
        
    As stated in Ur Uz работа.md: "Применение чисел Бетти к анализу ECDSA-Torus"
    """
    if not points or len(points) < 5:
        logger.warning("Not enough points for meaningful topological analysis")
        return [0.0] * (max_dimension + 1)
    
    # Determine appropriate epsilon for Rips complex
    # Using k-nearest neighbors to estimate density
    if len(points) > 10:
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(points)
        distances, _ = nn.kneighbors(points)
        avg_knn_dist = np.mean(distances[:, -1])
        epsilon = avg_knn_dist * 1.5
    else:
        epsilon = 0.1  # Default value for small datasets
    
    # Create Rips complex with proper torus metric
    try:
        rips = gudhi.RipsComplex(
            points=points,
            max_edge_length=epsilon,
            distance=lambda x, y: _torus_distance(x, y)
        )
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
        
        # Compute persistent homology
        persistence = simplex_tree.persistence()
        
        # Extract Betti numbers
        betti = [0.0] * (max_dimension + 1)
        for dim in range(max_dimension + 1):
            # Count infinite intervals (persistent features)
            infinite_intervals = [p for p in persistence if p[0] == dim and p[1][1] == float('inf')]
            # Count long intervals (persistent beyond threshold)
            long_intervals = [p for p in persistence if p[0] == dim and 
                             p[1][1] != float('inf') and (p[1][1] - p[1][0]) > epsilon * 0.5]
            
            betti[dim] = len(infinite_intervals) + len(long_intervals)
            
        return betti
    except Exception as e:
        logger.error(f"Error computing Betti numbers: {str(e)}")
        return [0.0] * (max_dimension + 1)

def calculate_topological_deviation(betti_numbers: List[float],
                                 expected: Dict[int, float] = SECURE_BETTI_NUMBERS) -> float:
    """
    Calculate the deviation of observed Betti numbers from expected values.
    
    This implements the key principle from Ur Uz работа.md: "Применение чисел Бетти к анализу
    ECDSA-Torus предоставляет точную количественную оценку структуры пространства подписей"
    
    Args:
        betti_numbers: Observed Betti numbers [β₀, β₁, β₂, ...]
        expected: Dictionary of expected Betti numbers by dimension
        
    Returns:
        float: Normalized deviation score (0.0 to 1.0)
        
    Example:
        >>> calculate_topological_deviation([1.0, 1.8, 1.0])
        0.1
    """
    deviation = 0.0
    for dim, expected_val in expected.items():
        if dim < len(betti_numbers):
            actual_val = betti_numbers[dim]
            # Calculate relative deviation, with smoothing to avoid division by zero
            dim_deviation = abs(actual_val - expected_val) / (expected_val + 1e-10)
            deviation += dim_deviation
    
    # Normalize by number of dimensions checked
    return min(1.0, deviation / len(expected))

def _calculate_euler_characteristic(betti_numbers: List[float]) -> float:
    """
    Calculate Euler characteristic from Betti numbers.
    
    For a torus: χ = β₀ - β₁ + β₂ = 1 - 2 + 1 = 0
    
    Args:
        betti_numbers: List of Betti numbers [β₀, β₁, β₂, ...]
        
    Returns:
        float: Euler characteristic value
    """
    euler_char = 0.0
    for i, beta in enumerate(betti_numbers):
        euler_char += ((-1) ** i) * beta
    return euler_char

def _calculate_topological_entropy(points: List[Tuple[float, float]], 
                                 grid_size: int = 32) -> float:
    """
    Calculate topological entropy of point distribution on torus.
    
    Args:
        points: List of points in (u_r, u_z) space
        grid_size: Size of the grid for discretization
        
    Returns:
        float: Topological entropy value between 0 and 1
    """
    if not points:
        return 0.0
    
    grid = np.zeros((grid_size, grid_size))
    
    # Fill the grid with point density
    for ur, uz in points:
        x = int(ur * grid_size) % grid_size
        y = int(uz * grid_size) % grid_size
        grid[x, y] += 1
    
    # Normalize to get probabilities
    total = np.sum(grid)
    if total == 0:
        return 0.0
    probabilities = grid / total
    
    # Calculate entropy: H = -Σ p_i log(p_i)
    entropy = 0.0
    for p in probabilities.flatten():
        if p > 0:
            entropy -= p * np.log(p)
    
    # Normalize by log of grid cells to get value between 0 and 1
    max_entropy = np.log(grid_size * grid_size)
    return entropy / max_entropy if max_entropy > 0 else 0.0

def _calculate_naturalness_coefficient(points: List[Tuple[float, float]]) -> float:
    """
    Calculate the naturalness coefficient of the point distribution.
    
    This measures how "natural" or expected the distribution is compared to
    the ideal uniform distribution on the torus.
    
    Args:
        points: List of points in (u_r, u_z) space
        
    Returns:
        float: Naturalness coefficient between 0 and 1 (lower is better)
    """
    if not points:
        return 1.0
    
    # Calculate expected uniform distribution
    n = len(points)
    expected_count = 1.0 / n
    
    # Calculate observed distribution using nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(5, n))
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    
    # Average distance to neighbors as a measure of local density
    avg_distances = np.mean(distances[:, 1:], axis=1)
    
    # Normalize to get a coefficient
    max_dist = np.max(avg_distances)
    if max_dist < 1e-10:
        return 0.0
    
    # Values closer to 0 indicate more uniform/natural distribution
    return np.mean(avg_distances) / max_dist

def _determine_vulnerability_type(betti_numbers: List[float],
                                topological_entropy: float,
                                naturalness_coefficient: float,
                                euler_characteristic: float) -> str:
    """
    Determine the type of vulnerability based on topological metrics.
    
    Args:
        betti_numbers: Calculated Betti numbers
        topological_entropy: Calculated topological entropy
        naturalness_coefficient: Calculated naturalness coefficient
        euler_characteristic: Calculated Euler characteristic
        
    Returns:
        str: Vulnerability type ("none", "topological_structure", "entropy_deficiency",
             "predictability", "manifold_distortion", or "unknown")
    """
    # Check for specific vulnerability patterns
    
    # Topological structure anomaly
    if len(betti_numbers) > 1 and abs(betti_numbers[1] - SECURE_BETTI_NUMBERS[1]) > BETTI_DEVIATION_THRESHOLD:
        return "topological_structure"
    
    # Entropy deficiency
    expected_entropy = math.log(len(betti_numbers) + 1) * 0.7
    if topological_entropy < expected_entropy * 0.6:
        return "entropy_deficiency"
    
    # Predictability vulnerability
    if naturalness_coefficient > 0.4:
        return "predictability"
    
    # Manifold distortion
    if abs(euler_characteristic) > CURVATURE_SMOOTH_THRESHOLD:
        return "manifold_distortion"
    
    return "none" if all(b > 0 for b in betti_numbers) else "unknown"

def _generate_vulnerability_explanation(tvi: float,
                                      vulnerability_type: str,
                                      betti_numbers: List[float],
                                      topological_entropy: float,
                                      naturalness_coefficient: float) -> str:
    """
    Generate a human-readable explanation for the vulnerability assessment.
    
    Args:
        tvi: Topological Vulnerability Index score
        vulnerability_type: Type of vulnerability detected
        betti_numbers: Betti numbers from analysis
        topological_entropy: Topological entropy value
        naturalness_coefficient: Naturalness coefficient value
        
    Returns:
        str: Detailed explanation of the vulnerability assessment
    """
    if tvi < TVI_SECURE_THRESHOLD:
        return ("No significant vulnerabilities detected. Topological structure is sound "
                "with uniform distribution across the signature space.")
    
    explanations = {
        "topological_structure": (
            f"Topological structure anomaly detected (β₁ = {betti_numbers[1]:.2f}, "
            f"expected ≈ {SECURE_BETTI_NUMBERS[1]}). This indicates potential weaknesses in the "
            "signature space structure that could be exploited."),
        
        "entropy_deficiency": (
            f"Topological entropy deficiency ({topological_entropy:.4f} is below "
            "expected threshold). This suggests insufficient randomness in the "
            "signature generation process, potentially allowing prediction of future "
            "signatures."),
        
        "predictability": (
            f"High predictability detected (naturalness coefficient = {naturalness_coefficient:.4f} > 0.4). "
            "This indicates patterns in the signature space that could be exploited "
            "to predict nonce values."),
        
        "manifold_distortion": (
            f"Manifold distortion detected (Euler characteristic = {euler_characteristic:.4f}, "
            f"expected ≈ 0.0). This suggests the topological structure has been compromised, "
            "potentially indicating a systematic vulnerability in the implementation.")
    }
    
    return explanations.get(vulnerability_type, 
                           "Security vulnerability detected with anomalous topological metrics.")

def _calculate_vulnerability_score(betti_numbers: List[float],
                                 euler_char: float,
                                 topological_entropy: float,
                                 naturalness_coefficient: float) -> float:
    """
    Calculate the vulnerability score based on topological metrics.
    
    Args:
        betti_numbers: Calculated Betti numbers [β₀, β₁, β₂]
        euler_char: Calculated Euler characteristic
        topological_entropy: Calculated topological entropy
        naturalness_coefficient: Calculated naturalness coefficient
        
    Returns:
        float: Vulnerability score (0.0 to 1.0)
    """
    # Weighted combination of different vulnerability indicators
    weights = {
        "betti_deviation": 0.4,
        "euler_deviation": 0.2,
        "entropy_deficiency": 0.2,
        "naturalness": 0.2
    }
    
    # Calculate betti deviation score
    betti_dev = 0.0
    for dim, expected in SECURE_BETTI_NUMBERS.items():
        if dim < len(betti_numbers):
            betti_dev += abs(betti_numbers[dim] - expected) / expected
    betti_dev /= len(SECURE_BETTI_NUMBERS)
    betti_score = min(1.0, betti_dev)
    
    # Euler characteristic deviation (expected is 0 for torus)
    euler_score = min(1.0, abs(euler_char) / 2.0)
    
    # Entropy deficiency (higher entropy is better)
    entropy_score = max(0.0, 1.0 - topological_entropy)
    
    # Naturalness (lower is better)
    naturalness_score = naturalness_coefficient
    
    # Weighted average
    score = (weights["betti_deviation"] * betti_score +
             weights["euler_deviation"] * euler_score +
             weights["entropy_deficiency"] * entropy_score +
             weights["naturalness"] * naturalness_score)
    
    return min(1.0, score * 1.5)  # Amplify slightly for better sensitivity

def analyze_signature_topology(points: List[Tuple[float, float]]) -> TopologicalMetrics:
    """
    Analyze the topological structure of signature space for security assessment.
    
    Args:
        points: List of points in (u_r, u_z) space from ECDSA signatures
        
    Returns:
        TopologicalMetrics: Comprehensive security assessment
        
    Example from Ur Uz работа.md: "Для 10,000 кошельков: 3 уязвимых (0.03%)"
    """
    start_time = time.time()
    
    # Calculate Betti numbers
    betti_numbers = _estimate_betti_numbers(points)
    
    # Calculate Euler characteristic (χ = β₀ - β₁ + β₂ - ...)
    euler_char = _calculate_euler_characteristic(betti_numbers)
    
    # Calculate topological entropy
    topological_entropy = _calculate_topological_entropy(points)
    
    # Calculate naturalness coefficient
    naturalness_coefficient = _calculate_naturalness_coefficient(points)
    
    # Calculate vulnerability score
    vulnerability_score = _calculate_vulnerability_score(
        betti_numbers, euler_char, topological_entropy, naturalness_coefficient
    )
    
    # Determine if secure
    is_secure = vulnerability_score < TVI_SECURE_THRESHOLD
    
    # Determine vulnerability type
    vulnerability_type = _determine_vulnerability_type(
        betti_numbers, topological_entropy, naturalness_coefficient, euler_char
    )
    
    # Generate explanation
    explanation = _generate_vulnerability_explanation(
        vulnerability_score, vulnerability_type, betti_numbers,
        topological_entropy, naturalness_coefficient
    )
    
    # Log performance
    duration = time.time() - start_time
    logger.debug(f"Topology analysis completed in {duration:.4f}s for {len(points)} points")
    
    return TopologicalMetrics(
        betti_numbers=betti_numbers,
        euler_characteristic=euler_char,
        topological_entropy=topological_entropy,
        naturalness_coefficient=naturalness_coefficient,
        tvi=min(1.0, vulnerability_score),
        is_secure=is_secure,
        vulnerability_type=vulnerability_type,
        explanation=explanation,
        timestamp=time.time()
    )

def validate_topology_integrity(points: List[Tuple[float, float]], 
                              max_deviation: float = BETTI_DEVIATION_THRESHOLD) -> bool:
    """
    Validate the topological integrity of signature space.
    
    Args:
        points: List of points in (u_r, u_z) space
        max_deviation: Maximum allowed deviation from expected topology
        
    Returns:
        bool: True if topology is intact, False otherwise
    """
    if not points:
        return False
    
    betti_numbers = _estimate_betti_numbers(points)
    deviation = calculate_topological_deviation(betti_numbers)
    
    return deviation <= max_deviation

def get_tvi_for_wallet(points: List[Tuple[float, float]]) -> float:
    """
    Calculate Topological Vulnerability Index (TVI) for a wallet.
    
    Args:
        points: List of signature points for the wallet
        
    Returns:
        float: TVI score between 0.0 (secure) and 1.0 (vulnerable)
    """
    if not points:
        return 1.0  # No data = maximum vulnerability
    
    metrics = analyze_signature_topology(points)
    return metrics.tvi

def analyze_vulnerabilities_via_betti(signatures: List[Tuple[int, int, int, float, float]]) -> Dict[str, Any]:
    """
    Analyze vulnerabilities through Betti numbers for ECDSA signatures.
    
    This function implements the comprehensive vulnerability analysis
    described in Ur Uz работа_2.md.
    
    Args:
        signatures: List of signatures [(r, s, z, ur, uz)]
        
    Returns:
        Dict[str, Any]: Analysis results with Betti numbers, vulnerabilities, etc.
    """
    # Extract (u_r, u_z) points
    points = [(ur, uz) for _, _, _, ur, uz in signatures]
    
    # Perform topological analysis
    topology_metrics = analyze_signature_topology(points)
    
    # Prepare result
    result = {
        "betti_numbers": topology_metrics.betti_numbers,
        "euler_characteristic": topology_metrics.euler_characteristic,
        "topological_entropy": topology_metrics.topological_entropy,
        "naturalness_coefficient": topology_metrics.naturalness_coefficient,
        "tvi": topology_metrics.tvi,
        "is_secure": topology_metrics.is_secure,
        "vulnerability_type": topology_metrics.vulnerability_type,
        "explanation": topology_metrics.explanation,
        "signature_count": len(signatures),
        "timestamp": topology_metrics.timestamp
    }
    
    return result

def is_vulnerable_wallet(points: List[Tuple[float, float]]) -> bool:
    """
    Determine if a wallet is vulnerable based on topological analysis.
    
    Args:
        points: List of signature points for the wallet
        
    Returns:
        bool: True if wallet is vulnerable, False otherwise
    """
    tvi = get_tvi_for_wallet(points)
    return tvi >= TVI_SECURE_THRESHOLD

def get_vulnerability_severity(tvi: float) -> str:
    """
    Get severity level for a given TVI score.
    
    Args:
        tvi: Topological Vulnerability Index score
        
    Returns:
        str: Severity level ("SECURE", "WARNING", "CRITICAL")
    """
    if tvi < TVI_SECURE_THRESHOLD:
        return "SECURE"
    elif tvi < TVI_WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "CRITICAL"

# For backward compatibility with older implementations
calculate_betti_numbers = _estimate_betti_numbers
analyze_torus_structure = analyze_signature_topology
