"""
bridge/__init__.py - QuantumBridge integration module for QuantumFortress 2.0.

This module provides seamless integration between QuantumFortress 2.0 and existing
blockchain networks through the QuantumBridge system. Unlike traditional integration
approaches that require core modifications, QuantumBridge operates as an API wrapper
that requires no changes to the underlying blockchain infrastructure.

Key features:
- API wrapper implementation requiring no core modifications
- Automatic conversion of ECDSA keys to quantum-topological equivalents
- Bitcoin and Ethereum compatibility through dedicated adapters
- TVI-based transaction filtering for enhanced security
- Seamless migration path for vulnerable wallets

The QuantumBridge system is based on the fundamental principle from Ur Uz работа.md:
"Множество решений уравнения ECDSA топологически эквивалентно двумерному тору S¹ × S¹"
(The set of solutions to the ECDSA equation is topologically equivalent to the 2D torus S¹ × S¹).

Author: Quantum Topology Research Group
Institution: Tambov Research Institute of Quantum Topology
Email: miro-aleksej@yandex.ru
"""

# Import core bridge components
from .quantum_bridge import QuantumBridge, convert_ecdsa_to_quantum, process_transaction
from .bitcoin_adapter import BitcoinAdapter, filter_transaction as bitcoin_filter
from .ethereum_adapter import EthereumAdapter, process_smart_contract

# Export main classes and functions for convenient import
__all__ = [
    # QuantumBridge components
    'QuantumBridge',
    'convert_ecdsa_to_quantum',
    'process_transaction',
    
    # Bitcoin integration components
    'BitcoinAdapter',
    'bitcoin_filter',
    
    # Ethereum integration components
    'EthereumAdapter',
    'process_smart_contract',
    
    # Constants and utility functions
    'TVI_SECURE_THRESHOLD',
    'TVI_WARNING_THRESHOLD',
    'MIGRATION_PHASES'
]

# Configuration constants (also defined in metrics.py for consistency)
TVI_SECURE_THRESHOLD = 0.3       # TVI threshold for secure implementation
TVI_WARNING_THRESHOLD = 0.5      # TVI threshold for warning state

# Migration phases as defined in Ur Uz работа.md
MIGRATION_PHASES = {
    0: "No migration needed (TVI < 0.1)",
    1: "Recommended migration (0.1 <= TVI < 0.5)",
    2: "Critical migration required (TVI >= 0.5)"
}

def get_migration_phase(tvi: float) -> int:
    """
    Determine the migration phase based on TVI.
    
    Migration phases:
    - 0: No migration needed (TVI < 0.1)
    - 1: Recommended migration (0.1 <= TVI < 0.5)
    - 2: Critical migration required (TVI >= 0.5)
    
    Args:
        tvi: Topological Vulnerability Index
        
    Returns:
        int: Migration phase (0, 1, or 2)
    
    Example from Ur Uz работа.md: "Блокирует транзакции с TVI > 0.5"
    """
    if tvi < 0.1:
        return 0
    elif tvi < 0.5:
        return 1
    else:
        return 2

def is_transaction_secure(tvi: float) -> bool:
    """
    Determine if a transaction is secure based on TVI.
    
    Args:
        tvi: Topological Vulnerability Index
        
    Returns:
        bool: True if transaction is secure, False otherwise
    
    Example from Ur Uz работа.md: "Блокирует транзакции с TVI > 0.5"
    """
    return tvi < TVI_WARNING_THRESHOLD

def get_bridge_status() -> Dict[str, Any]:
    """
    Get current status of the QuantumBridge system.
    
    Returns:
        Dict[str, Any]: Bridge status information
    
    Example:
        >>> get_bridge_status()
        {
            "status": "active",
            "connected_networks": ["Bitcoin", "Ethereum"],
            "tv_thresholds": {"secure": 0.3, "warning": 0.5},
            "migration_phases": {0: "No migration needed", ...}
        }
    """
    from quantum_fortress.topology.metrics import get_vulnerability_severity
    
    return {
        "status": "active",
        "version": "2.0",
        "connected_networks": ["Bitcoin", "Ethereum"],
        "tv_thresholds": {
            "secure": TVI_SECURE_THRESHOLD,
            "warning": TVI_WARNING_THRESHOLD
        },
        "migration_phases": MIGRATION_PHASES,
        "severity_levels": {
            "SECURE": get_vulnerability_severity(0.0),
            "WARNING": get_vulnerability_severity(0.3),
            "CRITICAL": get_vulnerability_severity(0.6)
        },
        "timestamp": time.time()
    }

# Backward compatibility with older implementations
Bridge = QuantumBridge
ECDSAConverter = convert_ecdsa_to_quantum

# Documentation for the bridge module
__doc__ += f"""

# QuantumBridge Integration Documentation

## Core Principles

QuantumBridge is the integration layer of QuantumFortress 2.0 that enables seamless
compatibility with existing blockchain networks. It's built on four key principles:

1. **Non-Invasive Integration**: QuantumBridge operates as an API wrapper that
   requires no core modifications to existing blockchain implementations.

2. **Topological Security**: Transactions are filtered based on TVI (Topological
   Vulnerability Index), blocking those with TVI > 0.5 as stated in Ur Uz работа.md.

3. **Hybrid Cryptography**: Provides a smooth migration path through three phases:
   - Phase 0 (TVI < 0.1): No migration needed
   - Phase 1 (0.1 ≤ TVI < 0.5): Recommended migration
   - Phase 2 (TVI ≥ 0.5): Critical migration required

4. **Network-Specific Adapters**: Dedicated adapters for Bitcoin and Ethereum
   handle network-specific requirements while maintaining a unified interface.

## Integration Workflow

1. **Transaction Processing**:
   - Incoming transaction is analyzed for topological metrics
   - TVI is calculated using topological analysis
   - Transaction is filtered if TVI > 0.5
   - Secure transactions are processed with quantum-topological signatures

2. **Key Conversion**:
   ```python
   from quantum_fortress.bridge import convert_ecdsa_to_quantum
   
   # Convert traditional ECDSA key to quantum-topological equivalent
   quantum_key = convert_ecdsa_to_quantum(ecdsa_private_key)
