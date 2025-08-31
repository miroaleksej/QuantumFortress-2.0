"""
QuantumFortress 2.0 Topology Module

This module provides the public API for all topological analysis functions used throughout
the QuantumFortress 2.0 system. It serves as the central entry point for topology-related
operations, including persistent homology, Betti number calculation, TVI (Topological
Vulnerability Index) computation, and topological compression.

Key features:
- Unified interface for topological analysis of ECDSA signatures
- Direct construction of compressed hypercube without building full representation
- TVI-based security monitoring and vulnerability detection
- Integration with quantum state analysis
- GPU acceleration for performance-critical operations
- Self-calibrating topological analysis based on system conditions

The implementation follows principles from:
- "Ur Uz работа.md": TVI metrics and signature analysis
- "Квантовый ПК.md": Quantum platform integration and calibration
- "Методы сжатия.md": Hypercube compression techniques
- "TopoSphere.md": Topological vulnerability analysis

As stated in documentation: "Прямое построение сжатого гиперкуба ECDSA представляет собой
критически важный прорыв, позволяющий анализировать системы, которые ранее считались
неподдающимися анализу из-за масштаба."
"""

__version__ = "2.0.0"
__author__ = "Quantum Topology Research Group"
__email__ = "miro-aleksej@yandex.ru"
__institution__ = "Tambov Research Institute of Quantum Topology"
__license__ = "MIT"
__description__ = "QuantumFortress 2.0 Topology Module"

# Import all topology modules
try:
    from .betti_numbers import (
        calculate_betti_numbers,
        analyze_betti_numbers,
        get_betti_thresholds,
        betti_number_analysis
    )
    BETTI_NUMBERS_AVAILABLE = True
except ImportError as e:
    BETTI_NUMBERS_AVAILABLE = False
    import warnings
    warnings.warn(f"Betti numbers utilities not fully available: {e}", ImportWarning)

try:
    from .homology import (
        PersistentHomologyAnalyzer,
        analyze_persistence_diagram,
        build_rips_complex,
        calculate_persistent_homology
    )
    HOMOLOGY_AVAILABLE = True
except ImportError as e:
    HOMOLOGY_AVAILABLE = False
    import warnings
    warnings.warn(f"Homology utilities not fully available: {e}", ImportWarning)

try:
    from .sheaf_theory import (
        SheafAnalyzer,
        analyze_sheaf_structure,
        calculate_sheaf_cohomology,
        detect_sheaf_anomalies
    )
    SHEAF_THEORY_AVAILABLE = True
except ImportError as e:
    SHEAF_THEORY_AVAILABLE = False
    import warnings
    warnings.warn(f"Sheaf theory utilities not fully available: {e}", ImportWarning)

try:
    from .metrics import (
        TopologicalMetrics,
        TVIResult,
        calculate_tvi,
        calculate_topological_entropy,
        calculate_naturalness_coefficient,
        get_tvi_thresholds,
        analyze_topological_vulnerability
    )
    METRICS_AVAILABLE = True
except ImportError as e:
    METRICS_AVAILABLE = False
    import warnings
    warnings.warn(f"Topological metrics not fully available: {e}", ImportWarning)

try:
    from .optimized_cache import (
        TopologicallyOptimizedCache,
        update_cache_strategy,
        get_cached_result,
        clear_topology_cache
    )
    CACHE_AVAILABLE = True
except ImportError as e:
    CACHE_AVAILABLE = False
    import warnings
    warnings.warn(f"Topology cache utilities not fully available: {e}", ImportWarning)

try:
    from .compressor import (
        TopologicalCompressor,
        compress_topology,
        decompress_topology,
        get_compression_ratio
    )
    COMPRESSOR_AVAILABLE = True
except ImportError as e:
    COMPRESSOR_AVAILABLE = False
    import warnings
    warnings.warn(f"Topology compressor not fully available: {e}", ImportWarning)

try:
    from .tcon import (
        TCONValidator,
        check_conformance,
        get_conformance_report,
        analyze_conformance_metrics
    )
    TCON_AVAILABLE = True
except ImportError as e:
    TCON_AVAILABLE = False
    import warnings
    warnings.warn(f"TCON utilities not fully available: {e}", ImportWarning)

try:
    from .hyperbolic import (
        HyperbolicClustering,
        analyze_hyperbolic_structure,
        calculate_hyperbolic_curvature,
        detect_hyperbolic_anomalies
    )
    HYPERBOLIC_AVAILABLE = True
except ImportError as e:
    HYPERBOLIC_AVAILABLE = False
    import warnings
    warnings.warn(f"Hyperbolic analysis utilities not fully available: {e}", ImportWarning)

try:
    from .metrics_extension import (
        ExtendedTopologyMetrics,
        calculate_topological_deviation,
        analyze_signature_topology,
        calculate_euler_characteristic,
        get_high_density_regions
    )
    METRICS_EXTENSION_AVAILABLE = True
except ImportError as e:
    METRICS_EXTENSION_AVAILABLE = False
    import warnings
    warnings.warn(f"Extended topology metrics not fully available: {e}", ImportWarning)

# GPU acceleration modules (optional)
try:
    from .gpu import (
        is_gpu_available,
        gpu_accelerated_homology,
        gpu_optimized_torus_distance,
        gpu_accelerated_tvi
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Public API
__all__ = [
    # Betti numbers
    "calculate_betti_numbers",
    "analyze_betti_numbers",
    "get_betti_thresholds",
    "betti_number_analysis",
    
    # Homology
    "PersistentHomologyAnalyzer",
    "analyze_persistence_diagram",
    "build_rips_complex",
    "calculate_persistent_homology",
    
    # Sheaf theory
    "SheafAnalyzer",
    "analyze_sheaf_structure",
    "calculate_sheaf_cohomology",
    "detect_sheaf_anomalies",
    
    # Topological metrics
    "TopologicalMetrics",
    "TVIResult",
    "calculate_tvi",
    "calculate_topological_entropy",
    "calculate_naturalness_coefficient",
    "get_tvi_thresholds",
    "analyze_topological_vulnerability",
    
    # Optimized cache
    "TopologicallyOptimizedCache",
    "update_cache_strategy",
    "get_cached_result",
    "clear_topology_cache",
    
    # Compressor
    "TopologicalCompressor",
    "compress_topology",
    "decompress_topology",
    "get_compression_ratio",
    
    # TCON
    "TCONValidator",
    "check_conformance",
    "get_conformance_report",
    "analyze_conformance_metrics",
    
    # Hyperbolic analysis
    "HyperbolicClustering",
    "analyze_hyperbolic_structure",
    "calculate_hyperbolic_curvature",
    "detect_hyperbolic_anomalies",
    
    # Extended metrics
    "ExtendedTopologyMetrics",
    "calculate_topological_deviation",
    "analyze_signature_topology",
    "calculate_euler_characteristic",
    "get_high_density_regions",
    
    # GPU acceleration
    "is_gpu_available",
    "gpu_accelerated_homology",
    "gpu_optimized_torus_distance",
    "gpu_accelerated_tvi",
    
    # Additional utilities
    "torus_distance",
    "transform_to_ur_uz",
    "estimate_collision_risk",
    "analyze_ecdsa_signature",
    "analyze_ecdsa_key",
    "ResourceMonitor",
    "DynamicComputeRouter",
    "BettiAnalyzer",
    "CollisionEngine",
    "GradientAnalyzer",
    "HypercoreTransformer"
]

# Import additional utilities that might be needed
try:
    from ..utils.topology_utils import (
        torus_distance,
        transform_to_ur_uz,
        estimate_collision_risk,
        analyze_ecdsa_signature,
        analyze_ecdsa_key
    )
except ImportError:
    pass

# Conditional imports for optional components
try:
    from ..utils.resource_monitor import ResourceMonitor
    __all__.append("ResourceMonitor")
except ImportError:
    pass

try:
    from ..utils.dynamic_compute_router import DynamicComputeRouter
    __all__.append("DynamicComputeRouter")
except ImportError:
    pass

try:
    from ..utils.betti_analyzer import BettiAnalyzer
    __all__.append("BettiAnalyzer")
except ImportError:
    pass

try:
    from ..utils.collision_engine import CollisionEngine
    __all__.append("CollisionEngine")
except ImportError:
    pass

try:
    from ..utils.gradient_analysis import GradientAnalyzer
    __all__.append("GradientAnalyzer")
except ImportError:
    pass

try:
    from ..utils.hypercore_transformer import HypercoreTransformer
    __all__.append("HypercoreTransformer")
except ImportError:
    pass

# Helper functions to check availability of components
def is_betti_numbers_available() -> bool:
    """Check if Betti numbers utilities are available."""
    return BETTI_NUMBERS_AVAILABLE

def is_homology_available() -> bool:
    """Check if homology utilities are available."""
    return HOMOLOGY_AVAILABLE

def is_sheaf_theory_available() -> bool:
    """Check if sheaf theory utilities are available."""
    return SHEAF_THEORY_AVAILABLE

def is_metrics_available() -> bool:
    """Check if topological metrics utilities are available."""
    return METRICS_AVAILABLE

def is_cache_available() -> bool:
    """Check if topology cache utilities are available."""
    return CACHE_AVAILABLE

def is_compressor_available() -> bool:
    """Check if topology compressor utilities are available."""
    return COMPRESSOR_AVAILABLE

def is_tcon_available() -> bool:
    """Check if TCON utilities are available."""
    return TCON_AVAILABLE

def is_hyperbolic_available() -> bool:
    """Check if hyperbolic analysis utilities are available."""
    return HYPERBOLIC_AVAILABLE

def is_metrics_extension_available() -> bool:
    """Check if extended topology metrics are available."""
    return METRICS_EXTENSION_AVAILABLE

def get_available_components() -> Dict[str, bool]:
    """
    Get availability status of all topology components.
    
    Returns:
        Dictionary with component names and their availability status
    """
    return {
        "betti_numbers": is_betti_numbers_available(),
        "homology": is_homology_available(),
        "sheaf_theory": is_sheaf_theory_available(),
        "metrics": is_metrics_available(),
        "cache": is_cache_available(),
        "compressor": is_compressor_available(),
        "tcon": is_tcon_available(),
        "hyperbolic": is_hyperbolic_available(),
        "metrics_extension": is_metrics_extension_available(),
        "gpu": GPU_AVAILABLE
    }

def verify_environment() -> bool:
    """
    Verify that the environment meets minimum requirements for topology analysis.
    
    Returns:
        bool: True if environment is suitable, False otherwise
        
    Raises:
        EnvironmentError: If critical components are missing
    """
    available = get_available_components()
    
    # Check critical components
    if not available["betti_numbers"]:
        raise EnvironmentError("Critical: Betti numbers utilities are not available")
    if not available["homology"]:
        raise EnvironmentError("Critical: Homology utilities are not available")
    if not available["metrics"]:
        raise EnvironmentError("Critical: Topological metrics utilities are not available")
    
    # Check recommended components
    if not available["tcon"]:
        import warnings
        warnings.warn("TCON utilities are not available - security validation will be limited", RuntimeWarning)
    if not available["compressor"]:
        import warnings
        warnings.warn("Topology compressor is not available - compression features will be limited", RuntimeWarning)
    if not available["gpu"] and available["metrics"]:
        import warnings
        warnings.warn("GPU acceleration is not available - performance may be suboptimal", RuntimeWarning)
    
    return True

# Version information
def get_version_info() -> Dict[str, Any]:
    """
    Get detailed version information for the topology module.
    
    Returns:
        Dictionary containing version information
    """
    return {
        "version": __version__,
        "components": get_available_components(),
        "platform": {
            "python_version": sys.version,
            "system": platform.system(),
            "processor": platform.processor()
        },
        "timestamp": time.time()
    }

# Initialize environment verification
try:
    verify_environment()
    logger = logging.getLogger(__name__)
    logger.info("QuantumFortress 2.0 topology environment verified successfully")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Environment verification failed: {str(e)}")
    # Don't raise the exception here - let individual components handle their dependencies

# Add any additional initialization code here
def initialize():
    """
    Initialize the topology module.
    
    This function should be called before using any topology functions.
    It sets up logging, verifies the environment, and initializes any
    required resources.
    """
    # Configure logging if not already configured
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Verify environment
    try:
        verify_environment()
        logger.info("QuantumFortress 2.0 topology module initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize topology module: {str(e)}")
        raise

# Initialize on import
initialize()

# Additional helper functions based on documentation
def get_tvi_thresholds() -> Dict[str, float]:
    """
    Get standard TVI (Topological Vulnerability Index) thresholds.
    
    Returns:
        Dictionary with TVI threshold values
        
    As stated in documentation: "Блокирует транзакции с TVI > 0.5."
    """
    return {
        "critical": 0.8,
        "high": 0.6,
        "medium": 0.4,
        "low": 0.2,
        "block_threshold": 0.5
    }

def is_secure_tvi(tvi: float) -> bool:
    """
    Check if a TVI value indicates a secure signature.
    
    Args:
        tvi: Topological Vulnerability Index value
        
    Returns:
        bool: True if secure (TVI < block threshold), False otherwise
    """
    thresholds = get_tvi_thresholds()
    return tvi < thresholds["block_threshold"]

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
    
    # Analyze topology
    topology_analysis = analyze_signature_topology([(ur, uz)])
    
    # Calculate TVI
    tvi = calculate_tvi(topology_analysis)
    
    return {
        "ur": ur,
        "uz": uz,
        "betti_numbers": topology_analysis.get("betti_numbers", [0, 0, 0]),
        "euler_characteristic": topology_analysis.get("euler_characteristic", 0.0),
        "topological_entropy": topology_analysis.get("topological_entropy", 0.0),
        "tvi": tvi,
        "is_secure": is_secure_tvi(tvi),
        "vulnerability_type": "NONE" if is_secure_tvi(tvi) else "TOPOLOGICAL_ANOMALY",
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
    vulnerable = tvi_result.tvi >= get_tvi_thresholds()["block_threshold"]
    
    return {
        "vulnerable": vulnerable,
        "tvi": tvi_result.tvi,
        "vulnerability_score": tvi_result.vulnerability_score,
        "vulnerability_type": tvi_result.vulnerability_type,
        "betti_numbers": topology_analysis.betti_numbers,
        "euler_characteristic": topology_analysis.euler_characteristic,
        "topological_entropy": topology_analysis.topological_entropy,
        "high_density_areas": len(topology_analysis.high_density_areas),
        "signature_count": len(points),
        "explanation": tvi_result.explanation,
        "timestamp": time.time()
    }

def get_betti_thresholds() -> Dict[str, float]:
    """
    Get standard thresholds for Betti numbers to detect vulnerabilities.
    
    Returns:
        Dictionary with Betti number threshold values
        
    As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
    точную количественную оценку структуры пространства подписей и обнаруживает скрытые
    уязвимости, которые пропускаются другими методами."
    """
    return {
        "betti0": 2.0,  # Connected components
        "betti1": 3.0,  # Loops
        "betti2": 1.0   # Voids
    }

def analyze_betti_numbers(betti_numbers: List[float]) -> Dict[str, Any]:
    """
    Analyze Betti numbers for potential vulnerabilities.
    
    Args:
        betti_numbers: List of Betti numbers [β₀, β₁, β₂, ...]
        
    Returns:
        Dictionary with analysis results
    """
    thresholds = get_betti_thresholds()
    analysis = {
        "betti0_anomaly": betti_numbers[0] > thresholds["betti0"],
        "betti1_anomaly": betti_numbers[1] > thresholds["betti1"],
        "betti2_anomaly": betti_numbers[2] > thresholds["betti2"],
        "total_anomalies": 0,
        "vulnerability_score": 0.0
    }
    
    # Count anomalies
    if analysis["betti0_anomaly"]:
        analysis["total_anomalies"] += 1
    if analysis["betti1_anomaly"]:
        analysis["total_anomalies"] += 1
    if analysis["betti2_anomaly"]:
        analysis["total_anomalies"] += 1
    
    # Calculate vulnerability score
    analysis["vulnerability_score"] = min(1.0, analysis["total_anomalies"] * 0.3)
    
    return analysis

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
    # Get the size of the torus (unit torus)
    size = 1.0
    
    # Calculate differences with toroidal wrap-around
    dx = abs(point1[0] - point2[0])
    dy = abs(point1[1] - point2[1])
    
    # Apply toroidal wrap-around
    dx = min(dx, size - dx)
    dy = min(dy, size - dy)
    
    # Euclidean distance
    return math.sqrt(dx**2 + dy**2)

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
    n = get_curve_order(curve)
    
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

def get_curve_order(curve: str) -> int:
    """
    Get the order of the elliptic curve.
    
    Args:
        curve: Elliptic curve name
        
    Returns:
        Order of the curve
    """
    # In a real implementation, this would use actual curve parameters
    if curve == "secp256k1":
        return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    elif curve == "P-256":
        return 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    else:
        # Default to secp256k1 order
        return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

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

# FastECDSA availability check
FAST_ECDSA_AVAILABLE = False
try:
    from fastecdsa.curve import Curve
    from fastecdsa.util import mod_sqrt
    from fastecdsa.point import Point
    FAST_ECDSA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("FastECDSA library detected - using optimized C extensions for ECDSA operations")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("FastECDSA library not found - falling back to pure Python ECDSA implementation")

# Optional: Performance benchmarking
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

# Optional: Self-test function
def self_test():
    """
    Run self-tests for topology functions.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    import random
    
    # Test torus_distance
    try:
        # Test normal distance
        dist1 = torus_distance((0.2, 0.3), (0.4, 0.5))
        assert 0.0 <= dist1 <= math.sqrt(2)
        
        # Test wrap-around distance
        dist2 = torus_distance((0.1, 0.1), (0.9, 0.9))
        assert dist2 < 0.5  # Should be shorter due to wrap-around
    except Exception as e:
        logger.error(f"torus_distance test failed: {str(e)}")
        return False
    
    # Test transform_to_ur_uz
    try:
        r, s, z = random.randint(1, 10**20), random.randint(1, 10**20), random.randint(1, 10**20)
        ur, uz = transform_to_ur_uz(r, s, z)
        assert 0.0 <= ur < 1.0 and 0.0 <= uz < 1.0
    except Exception as e:
        logger.error(f"transform_to_ur_uz test failed: {str(e)}")
        return False
    
    # Test calculate_betti_numbers
    try:
        points = [(random.random(), random.random()) for _ in range(100)]
        betti_numbers = calculate_betti_numbers(points)
        assert len(betti_numbers) >= 2
        assert all(isinstance(x, float) for x in betti_numbers)
    except Exception as e:
        logger.error(f"calculate_betti_numbers test failed: {str(e)}")
        return False
    
    # Test calculate_tvi
    try:
        points = [(random.random(), random.random()) for _ in range(100)]
        topology_metrics = analyze_signature_topology(points)
        tvi_result = calculate_tvi(topology_metrics)
        assert 0.0 <= tvi_result.tvi <= 1.0
    except Exception as e:
        logger.error(f"calculate_tvi test failed: {str(e)}")
        return False
    
    return True

# Run self-test on import (optional)
if __name__ == "__main__":
    print("Running QuantumFortress 2.0 topology module self-test...")
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
    
    print("\nExample: Transforming signature to (ur, uz) space...")
    r, s, z = random.randint(1, 10**20), random.randint(1, 10**20), random.randint(1, 10**20)
    ur, uz = transform_to_ur_uz(r, s, z)
    print(f"r: {r}")
    print(f"s: {s}")
    print(f"z: {z}")
    print(f"ur: {ur:.6f}")
    print(f"uz: {uz:.6f}")
