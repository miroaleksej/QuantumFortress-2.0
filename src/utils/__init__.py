"""
QuantumFortress 2.0 Utility Package

This package contains essential utility modules that support the core functionality
of the QuantumFortress blockchain system. These utilities provide critical helper
functions for cryptography, topology analysis, quantum operations, and performance
optimization that are used throughout the system.

The utilities implement key principles from our research:
- "Topology isn't a hacking tool, but a microscope for diagnosing vulnerabilities"
- Efficient WDM parallelism for 4.5x speedup in critical operations
- TVI (Topological Vulnerability Index) as the foundation for security assessment
- Seamless integration between classical and post-quantum cryptographic operations

These modules are designed to be:
- Highly optimized for performance-critical operations
- Thread-safe for concurrent usage in distributed systems
- Backward compatible with existing blockchain infrastructure
- Extensible for future quantum-topological innovations

As emphasized in Квантовый ПК.md: "Система авто-калибровки как обязательная часть рантайма"
("The auto-calibration system is a mandatory part of the runtime")
"""

# Import core utility modules
from .crypto_utils import (
    ecdsa_sign,
    ecdsa_verify,
    generate_ecdsa_keys,
    hash_message,
    inv,
    scalar_multiply
)
from .topology_utils import (
    calculate_topological_deviation,
    betti_number_analysis,
    analyze_torus_structure,
    calculate_tvi,
    project_to_torus
)
from .quantum_utils import (
    quantum_state_fidelity,
    apply_quantum_gate,
    measure_state,
    create_uniform_superposition,
    generate_quantum_circuit
)
from .simd_optimizations import (
    simd_parallelize,
    gpu_accelerate,
    optimize_for_cpu_features,
    vectorized_operations
)

# Package metadata
__version__ = "2.0.0"
__author__ = "Mironov A.A."
__email__ = "miro-aleksej@yandex.ru"
__institution__ = "Tambov Research Institute of Quantum Topology"

# Public API
__all__ = [
    # Cryptography utilities
    "ecdsa_sign",
    "ecdsa_verify",
    "generate_ecdsa_keys",
    "hash_message",
    "inv",
    "scalar_multiply",
    
    # Topology utilities
    "calculate_topological_deviation",
    "betti_number_analysis",
    "analyze_torus_structure",
    "calculate_tvi",
    "project_to_torus",
    
    # Quantum utilities
    "quantum_state_fidelity",
    "apply_quantum_gate",
    "measure_state",
    "create_uniform_superposition",
    "generate_quantum_circuit",
    
    # SIMD/GPU optimizations
    "simd_parallelize",
    "gpu_accelerate",
    "optimize_for_cpu_features",
    "vectorized_operations"
]

# Initialize package-level logger
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def get_utility_info() -> dict:
    """
    Get information about the utility package.
    
    Returns:
        Dictionary containing utility package information
    """
    return {
        "version": __version__,
        "modules": [
            "crypto_utils",
            "topology_utils",
            "quantum_utils",
            "simd_optimizations"
        ],
        "description": "Core utility functions for QuantumFortress 2.0",
        "author": __author__,
        "email": __email__,
        "institution": __institution__
    }

def verify_optimization_capabilities() -> dict:
    """
    Verify the availability of optimization capabilities.
    
    Returns:
        Dictionary with optimization capability status
        
    As stated in Квантовый ПК.md: 
    "Сильная сторона — параллелизм и пропускная способность; 
    слабое место — дрейф и разрядность, которые лечатся калибровкой и грамотной архитектурой."
    """
    import platform
    import numpy as np
    
    # Check SIMD capabilities
    simd_available = False
    try:
        # This is a simplified check - real implementation would be more comprehensive
        import os
        cpu_info = os.popen('lscpu').read() if platform.system() != 'Windows' else ''
        simd_available = 'avx' in cpu_info.lower() or 'sse' in cpu_info.lower()
    except:
        pass
    
    # Check GPU capabilities
    gpu_available = False
    try:
        import cupy
        gpu_available = True
    except ImportError:
        pass
    
    return {
        "platform": platform.system(),
        "processor": platform.processor(),
        "simd_available": simd_available,
        "gpu_available": gpu_available,
        "numpy_version": np.__version__,
        "numpy_simd_enabled": hasattr(np, '_mkl_version') or 'AVX' in np.__config__.get_info('atlas_info').get('define_macros', '')
    }
