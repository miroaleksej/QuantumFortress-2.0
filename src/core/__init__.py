"""QuantumFortress 2.0 Core Module

This module provides the foundational components for the QuantumFortress 2.0 system,
implementing the core quantum-topological security framework. It serves as the central
hub for the adaptive quantum hypercube, hybrid cryptography, and auto-calibration systems.

Key components exported:
- AdaptiveQuantumHypercube: The dynamic 4D-8D quantum structure that forms the security backbone
- AutoCalibrationSystem: Continuous monitoring and correction of quantum state drift
- HybridCryptoSystem: Seamless transition between classical and post-quantum cryptography
- TopologicalMetrics: Quantitative measurement of security through topological analysis

The implementation follows our core philosophy:
"Topology isn't a hacking tool, but a microscope for diagnosing vulnerabilities.
Ignoring it means building cryptography on sand."

For more information, see the architecture documentation at:
/docs/architecture/core
"""

__version__ = "2.0.0"
__author__ = "Mironov A.A., Tambov Research Institute of Quantum Topology"
__email__ = "miro-aleksej@yandex.ru"
__license__ = "MIT"
__description__ = "Core quantum-topological security framework for QuantumFortress 2.0"

# Import core components
from .adaptive_hypercube import AdaptiveQuantumHypercube
from .auto_calibration import AutoCalibrationSystem
from .hybrid_crypto import HybridCryptoSystem
from .topological_metrics import TopologicalMetrics
from .quantum_state import QuantumStateMonitor
from .dynamic_compute_router import DynamicComputeRouter
from .hypercore_transformer import HypercoreTransformer

# Export public API
__all__ = [
    'AdaptiveQuantumHypercube',
    'AutoCalibrationSystem',
    'HybridCryptoSystem',
    'TopologicalMetrics',
    'QuantumStateMonitor',
    'DynamicComputeRouter',
    'HypercoreTransformer',
    'DEFAULT_DIMENSION',
    'MAX_DIMENSION',
    'TVI_THRESHOLD',
    'SYSTEM_STABILITY_THRESHOLD'
]

# Core configuration constants
DEFAULT_DIMENSION = 4
MAX_DIMENSION = 8
TVI_THRESHOLD = 0.5  # Topological Vulnerability Index threshold for security
SYSTEM_STABILITY_THRESHOLD = 0.95  # Minimum stability percentage

def get_system_info() -> dict:
    """Get comprehensive information about the QuantumFortress core system.
    
    Returns:
        Dictionary containing system metadata, version, and capabilities
    """
    return {
        "version": __version__,
        "name": "QuantumFortress Core",
        "description": __description__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "core_components": [
            "Adaptive Quantum Hypercube",
            "Auto-Calibration System",
            "Hybrid Cryptography",
            "Topological Metrics",
            "Quantum State Monitoring"
        ],
        "current_dimension": DEFAULT_DIMENSION,
        "max_dimension": MAX_DIMENSION,
        "tvi_threshold": TVI_THRESHOLD,
        "stability_threshold": SYSTEM_STABILITY_THRESHOLD,
        "warranty": "Topology-based security implementation - not theoretical speculation"
    }

def initialize_quantum_system(dimension: int = DEFAULT_DIMENSION) -> dict:
    """Initialize the QuantumFortress core system with specified dimension.
    
    Args:
        dimension: Target dimension for the quantum hypercube (4-8)
        
    Returns:
        Dictionary containing initialized system components
        
    Raises:
        ValueError: If dimension is outside valid range (4-8)
    """
    if not (DEFAULT_DIMENSION <= dimension <= MAX_DIMENSION):
        raise ValueError(f"Dimension must be between {DEFAULT_DIMENSION} and {MAX_DIMENSION}")
    
    # Initialize core components
    hypercube = AdaptiveQuantumHypercube(dimension=dimension)
    calibration_system = AutoCalibrationSystem(hypercube)
    crypto_system = HybridCryptoSystem()
    metrics = TopologicalMetrics()
    
    # Start background services
    calibration_system.start()
    
    return {
        "hypercube": hypercube,
        "calibration_system": calibration_system,
        "crypto_system": crypto_system,
        "metrics": metrics,
        "dimension": dimension,
        "status": "initialized",
        "message": f"QuantumFortress core system initialized at {dimension}D"
    }

# Initialize package-level logger
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# For debugging and development
if __name__ == "__main__":
    print("QuantumFortress 2.0 Core Module")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print("\nSystem Information:")
    import json
    print(json.dumps(get_system_info(), indent=2))
