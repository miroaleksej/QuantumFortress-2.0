"""
QuantumFortress 2.0 Utility Module

This module provides the public API for all utility functions used throughout the
QuantumFortress 2.0 system. It serves as the main entry point for utility functions
related to cryptography, topology, quantum computing, and performance optimization.

Key components:
- Cryptography utilities for ECDSA and hybrid signatures
- Topological analysis functions for TVI calculation
- Quantum state manipulation and analysis
- SIMD/GPU acceleration for performance-critical operations
- Integration with FastECDSA for optimized cryptographic operations

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
__description__ = "QuantumFortress 2.0 Utility Module"

# Import all utility modules
try:
    from .crypto_utils import (
        ecdsa_sign,
        ecdsa_verify,
        generate_ecdsa_keys,
        hash_message,
        inv,
        scalar_multiply,
        generate_quantum_key_pair,
        verify_quantum_signature,
        fastecdsa_available,
        get_curve_order,
        transform_to_ur_uz,
        extract_ecdsa_components,
        calculate_z,
        quantum_sign,
        quantum_verify
    )
    CRYPTO_UTILS_AVAILABLE = True
except ImportError as e:
    CRYPTO_UTILS_AVAILABLE = False
    import warnings
    warnings.warn(f"Crypto utilities not fully available: {e}", ImportWarning)

try:
    from .topology_utils import (
        calculate_topological_deviation,
        betti_number_analysis,
        analyze_torus_structure,
        calculate_tvi,
        project_to_torus,
        calculate_betti_numbers,
        analyze_signature_topology,
        torus_distance,
        calculate_euler_characteristic,
        calculate_topological_entropy,
        find_high_density_areas,
        get_connectivity_metrics,
        calculate_topological_metrics,
        analyze_topological_vulnerability,
        get_high_density_regions,
        calculate_naturalness_coefficient,
        estimate_collision_risk
    )
    TOPOLOGY_UTILS_AVAILABLE = True
except ImportError as e:
    TOPOLOGY_UTILS_AVAILABLE = False
    import warnings
    warnings.warn(f"Topology utilities not fully available: {e}", ImportWarning)

try:
    from .quantum_utils import (
        quantum_state_fidelity,
        apply_quantum_gate,
        measure_state,
        create_uniform_superposition,
        generate_quantum_circuit,
        QuantumPlatform,
        PlatformConfig,
        get_platform_config,
        optimize_shor_algorithm,
        optimize_grover_algorithm,
        apply_phase_correction,
        apply_amplitude_correction,
        calculate_coherence_time,
        generate_quantum_noise_profile,
        calculate_quantum_vulnerability,
        get_quantum_platform_metrics
    )
    QUANTUM_UTILS_AVAILABLE = True
except ImportError as e:
    QUANTUM_UTILS_AVAILABLE = False
    import warnings
    warnings.warn(f"Quantum utilities not fully available: {e}", ImportWarning)

try:
    from .simd_utils import (
        simd_parallelize,
        gpu_accelerate,
        vectorized_torus_distance,
        parallel_betti_calculation,
        gpu_optimized_homology,
        vectorized_projection,
        parallel_collision_detection,
        gpu_accelerated_tvi
    )
    SIMD_UTILS_AVAILABLE = True
except ImportError:
    SIMD_UTILS_AVAILABLE = False
    # SIMD/GPU acceleration is optional but recommended for performance

# Public API
__all__ = [
    # Cryptography utilities
    "ecdsa_sign",
    "ecdsa_verify",
    "generate_ecdsa_keys",
    "hash_message",
    "inv",
    "scalar_multiply",
    "generate_quantum_key_pair",
    "verify_quantum_signature",
    "fastecdsa_available",
    "get_curve_order",
    "transform_to_ur_uz",
    "extract_ecdsa_components",
    "calculate_z",
    "quantum_sign",
    "quantum_verify",
    
    # Topology utilities
    "calculate_topological_deviation",
    "betti_number_analysis",
    "analyze_torus_structure",
    "calculate_tvi",
    "project_to_torus",
    "calculate_betti_numbers",
    "analyze_signature_topology",
    "torus_distance",
    "calculate_euler_characteristic",
    "calculate_topological_entropy",
    "find_high_density_areas",
    "get_connectivity_metrics",
    "calculate_topological_metrics",
    "analyze_topological_vulnerability",
    "get_high_density_regions",
    "calculate_naturalness_coefficient",
    "estimate_collision_risk",
    
    # Quantum utilities
    "quantum_state_fidelity",
    "apply_quantum_gate",
    "measure_state",
    "create_uniform_superposition",
    "generate_quantum_circuit",
    "QuantumPlatform",
    "PlatformConfig",
    "get_platform_config",
    "optimize_shor_algorithm",
    "optimize_grover_algorithm",
    "apply_phase_correction",
    "apply_amplitude_correction",
    "calculate_coherence_time",
    "generate_quantum_noise_profile",
    "calculate_quantum_vulnerability",
    "get_quantum_platform_metrics",
    
    # SIMD/GPU optimizations
    "simd_parallelize",
    "gpu_accelerate",
    "vectorized_torus_distance",
    "parallel_betti_calculation",
    "gpu_optimized_homology",
    "vectorized_projection",
    "parallel_collision_detection",
    "gpu_accelerated_tvi",
    
    # Additional utilities
    "ResourceMonitor",
    "DynamicComputeRouter",
    "BettiAnalyzer",
    "CollisionEngine",
    "TopologicalMetrics",
    "TVIResult",
    "CalibrationStatus",
    "MigrationPhase",
    "CompressionMethod",
    "HypercubeCompressionResult",
    "TopoNonceV2",
    "TCONValidator",
    "HypercoreTransformer",
    "GradientAnalyzer"
]

# Conditional imports for optional components
try:
    from .resource_monitor import ResourceMonitor
    __all__.append("ResourceMonitor")
except ImportError:
    pass

try:
    from .dynamic_compute_router import DynamicComputeRouter
    __all__.append("DynamicComputeRouter")
except ImportError:
    pass

try:
    from .betti_analyzer import BettiAnalyzer
    __all__.append("BettiAnalyzer")
except ImportError:
    pass

try:
    from .collision_engine import CollisionEngine
    __all__.append("CollisionEngine")
except ImportError:
    pass

try:
    from .topological_metrics import TopologicalMetrics, TVIResult
    __all__.extend(["TopologicalMetrics", "TVIResult"])
except ImportError:
    pass

try:
    from .auto_calibration import CalibrationStatus
    __all__.append("CalibrationStatus")
except ImportError:
    pass

try:
    from .hybrid_crypto import MigrationPhase
    __all__.append("MigrationPhase")
except ImportError:
    pass

try:
    from .adaptive_hypercube import CompressionMethod, HypercubeCompressionResult
    __all__.extend(["CompressionMethod", "HypercubeCompressionResult"])
except ImportError:
    pass

try:
    from .topo_nonce_v2 import TopoNonceV2
    __all__.append("TopoNonceV2")
except ImportError:
    pass

try:
    from .tcon import TCONValidator
    __all__.append("TCONValidator")
except ImportError:
    pass

try:
    from .hypercore_transformer import HypercoreTransformer
    __all__.append("HypercoreTransformer")
except ImportError:
    pass

try:
    from .gradient_analysis import GradientAnalyzer
    __all__.append("GradientAnalyzer")
except ImportError:
    pass

# Helper functions to check availability of components
def is_crypto_utils_available() -> bool:
    """Check if cryptography utilities are available."""
    return CRYPTO_UTILS_AVAILABLE

def is_topology_utils_available() -> bool:
    """Check if topology utilities are available."""
    return TOPOLOGY_UTILS_AVAILABLE

def is_quantum_utils_available() -> bool:
    """Check if quantum utilities are available."""
    return QUANTUM_UTILS_AVAILABLE

def is_simd_utils_available() -> bool:
    """Check if SIMD/GPU acceleration utilities are available."""
    return SIMD_UTILS_AVAILABLE

def get_available_components() -> Dict[str, bool]:
    """
    Get availability status of all utility components.
    
    Returns:
        Dictionary with component names and their availability status
    """
    return {
        "crypto_utils": is_crypto_utils_available(),
        "topology_utils": is_topology_utils_available(),
        "quantum_utils": is_quantum_utils_available(),
        "simd_utils": is_simd_utils_available()
    }

def verify_environment() -> bool:
    """
    Verify that the environment meets minimum requirements for QuantumFortress 2.0.
    
    Returns:
        bool: True if environment is suitable, False otherwise
        
    Raises:
        EnvironmentError: If critical components are missing
    """
    available = get_available_components()
    
    # Check critical components
    if not available["crypto_utils"]:
        raise EnvironmentError("Critical: Cryptography utilities are not available")
    if not available["topology_utils"]:
        raise EnvironmentError("Critical: Topology utilities are not available")
    
    # Check recommended components
    if not available["quantum_utils"]:
        import warnings
        warnings.warn("Quantum utilities are not available - quantum features will be limited", RuntimeWarning)
    if not available["simd_utils"]:
        import warnings
        warnings.warn("SIMD/GPU acceleration is not available - performance may be suboptimal", RuntimeWarning)
    
    # Check FastECDSA availability
    if 'fastecdsa_available' in globals() and not fastecdsa_available:
        import warnings
        warnings.warn("FastECDSA is not available - ECDSA operations will be slower", RuntimeWarning)
    
    return True

# Initialize environment verification
try:
    verify_environment()
    logger = logging.getLogger(__name__)
    logger.info("QuantumFortress 2.0 utilities environment verified successfully")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Environment verification failed: {str(e)}")
    # Don't raise the exception here - let individual components handle their dependencies

# Version information
def get_version_info() -> Dict[str, Any]:
    """
    Get detailed version information for the utility module.
    
    Returns:
        Dictionary containing version information
    """
    return {
        "version": __version__,
        "components": {
            "crypto_utils": CRYPTO_UTILS_AVAILABLE,
            "topology_utils": TOPOLOGY_UTILS_AVAILABLE,
            "quantum_utils": QUANTUM_UTILS_AVAILABLE,
            "simd_utils": SIMD_UTILS_AVAILABLE
        },
        "fastecdsa": fastecdsa_available if 'fastecdsa_available' in globals() else False,
        "platform": {
            "python_version": sys.version,
            "system": platform.system(),
            "processor": platform.processor()
        },
        "timestamp": time.time()
    }

# Optional: Register atexit handler to clean up resources
try:
    import atexit
    
    def _cleanup():
        """Cleanup function to be called at program exit."""
        logger = logging.getLogger(__name__)
        logger.info("Cleaning up QuantumFortress 2.0 utilities...")
        
        # Add any cleanup logic here
        # For example, stopping background threads, closing connections, etc.
        
        logger.info("Cleanup completed")
    
    atexit.register(_cleanup)
except ImportError:
    pass

# Export version information
__version_info__ = get_version_info()

# Add any additional initialization code here
def initialize():
    """
    Initialize the utility module.
    
    This function should be called before using any utility functions.
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
        logger.info("QuantumFortress 2.0 utilities initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize utilities: {str(e)}")
        raise

# Initialize on import
initialize()

# Add any additional helper functions here
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

def get_quantum_platform_names() -> List[str]:
    """
    Get names of all supported quantum platforms.
    
    Returns:
        List of quantum platform names
    """
    return [platform.name for platform in QuantumPlatform]

def get_default_platform() -> QuantumPlatform:
    """
    Get the default quantum platform.
    
    Returns:
        QuantumPlatform: Default platform (SOI)
    """
    return QuantumPlatform.SOI

# Additional helper functions based on documentation
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
    topology_analysis = analyze_torus_structure([(ur, uz)])
    
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

def validate_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a transaction based on topological analysis.
    
    Args:
        transaction: Transaction dictionary containing signature components
        
    Returns:
        Dictionary with validation results
        
    As stated in documentation: "Плагин для Bitcoin Core: Автоматически проверяет входящие транзакции
    на наличие слабых подписей, Блокирует транзакции с TVI > 0.5."
    """
    # Extract signature components
    signature = transaction.get("signature", {})
    r = signature.get("r")
    s = signature.get("s")
    z = signature.get("z")
    curve = signature.get("curve", "secp256k1")
    
    if r is None or s is None or z is None:
        return {
            "valid": False,
            "reason": "Missing signature components",
            "tvi": 1.0,
            "timestamp": time.time()
        }
    
    # Analyze signature
    analysis = analyze_ecdsa_signature(r, s, z, curve)
    
    # Check TVI
    if not analysis["is_secure"]:
        return {
            "valid": False,
            "reason": f"TVI too high ({analysis['tvi']:.4f} > 0.5)",
            "tvi": analysis["tvi"],
            "timestamp": time.time()
        }
    
    # Additional validation steps would go here
    
    return {
        "valid": True,
        "tvi": analysis["tvi"],
        "betti_numbers": analysis["betti_numbers"],
        "timestamp": time.time()
    }

# FastECDSA availability check (must be defined after imports)
try:
    from fastecdsa.curve import Curve
    from fastecdsa.util import mod_sqrt
    from fastecdsa.point import Point
    FAST_ECDSA_AVAILABLE = True
except ImportError:
    FAST_ECDSA_AVAILABLE = False

# Export FastECDSA availability
fastecdsa_available = FAST_ECDSA_AVAILABLE

# Add any additional initialization that depends on FastECDSA
if FAST_ECDSA_AVAILABLE:
    logger = logging.getLogger(__name__)
    logger.info("FastECDSA library detected - using optimized C extensions for ECDSA operations")
else:
    logger = logging.getLogger(__name__)
    logger.warning("FastECDSA library not found - falling back to pure Python ECDSA implementation")

# Optional: Performance benchmarking
def benchmark_performance():
    """
    Run performance benchmarks for critical utility functions.
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {}
    
    # Benchmark ECDSA signing
    start = time.time()
    for _ in range(1000):
        _, _ = generate_ecdsa_keys()
    results["ecdsa_keygen"] = (time.time() - start) / 1000.0
    
    # Benchmark topology analysis
    start = time.time()
    points = [(random.random(), random.random()) for _ in range(1000)]
    _ = calculate_betti_numbers(points)
    results["betti_calculation"] = time.time() - start
    
    # Benchmark TVI calculation
    start = time.time()
    for _ in range(100):
        r, s, z = random.randint(1, 10**20), random.randint(1, 10**20), random.randint(1, 10**20)
        _ = analyze_ecdsa_signature(r, s, z)
    results["tvi_calculation"] = (time.time() - start) / 100.0
    
    return results

# Optional: Self-test function
def self_test():
    """
    Run self-tests for utility functions.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    import random
    
    # Test ECDSA key generation
    try:
        private_key, public_key = generate_ecdsa_keys()
        assert private_key is not None
        assert public_key is not None
    except Exception as e:
        logger.error(f"ECDSA key generation test failed: {str(e)}")
        return False
    
    # Test topology analysis
    try:
        points = [(random.random(), random.random()) for _ in range(100)]
        betti_numbers = calculate_betti_numbers(points)
        assert len(betti_numbers) == 3
        assert all(isinstance(x, float) for x in betti_numbers)
    except Exception as e:
        logger.error(f"Topology analysis test failed: {str(e)}")
        return False
    
    # Test TVI calculation
    try:
        r, s, z = random.randint(1, 10**20), random.randint(1, 10**20), random.randint(1, 10**20)
        analysis = analyze_ecdsa_signature(r, s, z)
        assert "tvi" in analysis
        assert 0.0 <= analysis["tvi"] <= 1.0
    except Exception as e:
        logger.error(f"TVI calculation test failed: {str(e)}")
        return False
    
    # Test quantum platform configuration
    try:
        platform = get_default_platform()
        config = get_platform_config(platform)
        assert isinstance(config, PlatformConfig)
    except Exception as e:
        logger.error(f"Quantum platform test failed: {str(e)}")
        return False
    
    return True

# Run self-test on import (optional)
if __name__ == "__main__":
    print("Running QuantumFortress 2.0 utility module self-test...")
    if self_test():
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Please check the logs for details.")
    
    print("\nBenchmarking performance...")
    results = benchmark_performance()
    print(f"ECDSA key generation: {results['ecdsa_keygen']:.6f} sec/op")
    print(f"Betti number calculation: {results['betti_calculation']:.6f} sec")
    print(f"TVI calculation: {results['tvi_calculation']:.6f} sec/op")
    
    print("\nVersion information:")
    print(json.dumps(get_version_info(), indent=2))
    
    print("\nTVI thresholds:")
    print(json.dumps(get_tvi_thresholds(), indent=2))
    
    print("\nQuantum platforms:")
    print(json.dumps(get_quantum_platform_names(), indent=2))
