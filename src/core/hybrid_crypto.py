"""
QuantumFortress 2.0 Hybrid Cryptographic System

This module implements the hybrid cryptographic system that enables seamless migration
from classical to post-quantum algorithms while maintaining full backward compatibility.
The system uses the Topological Vulnerability Index (TVI) as the primary metric
for determining migration phases and security status.

Key features:
- Hybrid signing with both ECDSA and quantum-topological signatures
- Automatic migration phases triggered by TVI thresholds
- Full backward compatibility with existing blockchain networks
- TVI-based transaction filtering (blocks transactions with TVI > 0.5)
- QuantumBridge integration for traditional network compatibility

As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
точную количественную оценку структуры пространства подписей и обнаруживает скрытые
уязвимости, которые пропускаются другими методами."

This implementation extends those principles to a hybrid cryptographic framework that
provides a practical path to post-quantum security.
"""

import numpy as np
import time
import uuid
from enum import Enum
from typing import Union, Dict, Any, Tuple, Optional
import logging

from ..utils.crypto_utils import generate_ecdsa_keys, verify_ecdsa_signature
from ..utils.topology_utils import calculate_betti_numbers, analyze_signature_topology
from ..utils.quantum_utils import generate_quantum_key_pair, verify_quantum_signature
from .adaptive_hypercube import AdaptiveQuantumHypercube

logger = logging.getLogger(__name__)

class MigrationPhase(Enum):
    """Migration phases for the hybrid cryptographic system"""
    CLASSICAL = 1    # Only ECDSA signatures
    HYBRID = 2       # Both ECDSA and quantum-topological signatures
    POST_QUANTUM = 3 # Only quantum-topological signatures

class HybridKeyPair:
    """Container for hybrid cryptographic key pairs"""
    
    def __init__(self, 
                 key_id: str,
                 created_at: float,
                 ecdsa_private: Any,
                 ecdsa_public: Any,
                 quantum_private: Optional[Any] = None,
                 quantum_public: Optional[Any] = None,
                 current_phase: MigrationPhase = MigrationPhase.CLASSICAL):
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
        """
        self.key_id = key_id
        self.created_at = created_at
        self.ecdsa_private = ecdsa_private
        self.ecdsa_public = ecdsa_public
        self.quantum_private = quantum_private
        self.quantum_public = quantum_public
        self.current_phase = current_phase
    
    def is_quantum_enabled(self) -> bool:
        """Check if quantum components are available and active."""
        return self.current_phase in [MigrationPhase.HYBRID, MigrationPhase.POST_QUANTUM] and \
               self.quantum_private is not None and self.quantum_public is not None

class HybridSignature:
    """Container for hybrid cryptographic signatures"""
    
    def __init__(self,
                 signature_id: str,
                 timestamp: float,
                 ecdsa_signature: bytes,
                 quantum_signature: Optional[bytes] = None,
                 tvi: float = 1.0,
                 migration_phase: MigrationPhase = MigrationPhase.CLASSICAL):
        """
        Initialize a hybrid signature.
        
        Args:
            signature_id: Unique identifier for the signature
            timestamp: Timestamp of signature creation
            ecdsa_signature: ECDSA signature component
            quantum_signature: Quantum-topological signature component (optional)
            tvi: Topological Vulnerability Index score (0.0 = secure, 1.0 = critical)
            migration_phase: Migration phase used for signing
        """
        self.signature_id = signature_id
        self.timestamp = timestamp
        self.ecdsa_signature = ecdsa_signature
        self.quantum_signature = quantum_signature
        self.tvi = tvi
        self.migration_phase = migration_phase
    
    def is_secure(self, threshold: float = 0.5) -> bool:
        """
        Check if the signature is secure based on TVI.
        
        Args:
            threshold: TVI threshold for security (default: 0.5)
            
        Returns:
            bool: True if secure (TVI < threshold), False otherwise
        """
        return self.tvi < threshold

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
    """
    
    def __init__(self, 
                 base_dimension: int = 4,
                 tvi_threshold_classical: float = 0.3,
                 tvi_threshold_hybrid: float = 0.7,
                 min_quantum_security: float = 0.8):
        """
        Initialize the hybrid cryptographic system.
        
        Args:
            base_dimension: Base dimension for the quantum hypercube
            tvi_threshold_classical: TVI threshold for remaining in CLASSICAL phase
            tvi_threshold_hybrid: TVI threshold for moving to POST_QUANTUM phase
            min_quantum_security: Minimum quantum security level required for migration
            
        Topological Vulnerability Analysis (TVA) combines data from network scans and known 
        vulnerabilities into a model of the network security environment to determine 
        appropriate cryptographic approaches. [[9]]
        """
        self.base_dimension = base_dimension
        self.tvi_threshold_classical = tvi_threshold_classical
        self.tvi_threshold_hybrid = tvi_threshold_hybrid
        self.min_quantum_security = min_quantum_security
        self.migration_phase = MigrationPhase.CLASSICAL
        self.phase_start_time = time.time()
        self.last_analysis = 0.0
        self.hypercube = AdaptiveQuantumHypercube(base_dimension)
        self.quantum_security_level = 0.0
        self.tvi_history = []
        
        logger.info(f"Initialized HybridCryptoSystem (phase={self.migration_phase.name}, "
                    f"dimension={self.base_dimension})")
    
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
        if len(self.tvi_history) > 1000:
            self.tvi_history.pop(0)
        
        # Analyze trends in TVI
        recent_tvi_values = [tvi for _, tvi in self.tvi_history[-10:]]
        avg_recent_tvi = sum(recent_tvi_values) / len(recent_tvi_values) if recent_tvi_values else current_tvi
        
        # Determine phase based on TVI thresholds and trends
        if avg_recent_tvi < self.tvi_threshold_classical and self.migration_phase != MigrationPhase.CLASSICAL:
            return MigrationPhase.CLASSICAL
        elif avg_recent_tvi < self.tvi_threshold_hybrid and self.migration_phase != MigrationPhase.HYBRID:
            return MigrationPhase.HYBRID
        elif avg_recent_tvi >= self.tvi_threshold_hybrid and self.quantum_security_level >= self.min_quantum_security:
            return MigrationPhase.POST_QUANTUM
        
        return self.migration_phase
    
    def _update_migration_phase(self) -> bool:
        """
        Update the migration phase if conditions warrant a change.
        
        Returns:
            bool: True if phase changed, False otherwise
        """
        new_phase = self._determine_migration_phase()
        
        if new_phase != self.migration_phase:
            old_phase = self.migration_phase
            self.migration_phase = new_phase
            self.phase_start_time = time.time()
            
            logger.info(f"Migration phase changed from {old_phase.name} to {new_phase.name} "
                        f"(current TVI: {self.hypercube.get_tvi():.4f})")
            return True
        
        return False
    
    def generate_keys(self) -> HybridKeyPair:
        """
        Generate hybrid cryptographic keys.
        
        This method creates both classical (ECDSA) and quantum-topological key components,
        with the quantum component being activated based on the current migration phase.
        
        Returns:
            HybridKeyPair object containing both key types
            
        Example from documentation:
        "Плагин для Bitcoin Core: Автоматически проверяет входящие транзакции
        на наличие слабых подписей, Блокирует транзакции с TVI > 0.5."
        """
        self._update_migration_phase()
        
        key_id = str(uuid.uuid4())
        creation_time = time.time()
        
        # Generate ECDSA keys (always included for backward compatibility)
        ecdsa_private, ecdsa_public = generate_ecdsa_keys()
        
        # Generate quantum-topological keys based on migration phase
        quantum_private, quantum_public = None, None
        if self.migration_phase in [MigrationPhase.HYBRID, MigrationPhase.POST_QUANTUM]:
            quantum_private, quantum_public = generate_quantum_key_pair(self.base_dimension)
        
        # Determine current migration phase for these keys
        current_phase = self.migration_phase
        
        return HybridKeyPair(
            key_id=key_id,
            created_at=creation_time,
            ecdsa_private=ecdsa_private,
            ecdsa_public=ecdsa_public,
            quantum_private=quantum_private,
            quantum_public=quantum_public,
            current_phase=current_phase
        )
    
    def sign(self, private_key: HybridKeyPair, message: Union[str, bytes]) -> HybridSignature:
        """
        Create a hybrid signature for the given message.
        
        This method:
        - Always creates an ECDSA signature for backward compatibility
        - Creates a quantum-topological signature when in HYBRID or POST_QUANTUM phase
        - Calculates TVI for the signature to assess security
        
        Args:
            private_key: HybridKeyPair containing private components
            message: Message to sign
            
        Returns:
            HybridSignature object containing both signature types
            
        Example from documentation:
        "Works as API wrapper (no core modifications needed)"
        """
        signature_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Always create ECDSA signature for backward compatibility
        ecdsa_signature = self._sign_ecdsa(private_key.ecdsa_private, message)
        
        # Create quantum-topological signature if in appropriate phase
        quantum_signature = None
        if private_key.is_quantum_enabled():
            quantum_signature = self._sign_quantum(private_key.quantum_private, message)
        
        # Analyze topology of the signature to calculate TVI
        tvi = self._analyze_signature_topology(message, ecdsa_signature, quantum_signature)
        
        return HybridSignature(
            signature_id=signature_id,
            timestamp=timestamp,
            ecdsa_signature=ecdsa_signature,
            quantum_signature=quantum_signature,
            tvi=tvi,
            migration_phase=private_key.current_phase
        )
    
    def _sign_ecdsa(self, ecdsa_private: Any, message: Union[str, bytes]) -> bytes:
        """Create an ECDSA signature for the given message."""
        # In a real implementation, this would use a proper ECDSA signing function
        # Here we're simulating the process
        return b"mock_ecdsa_signature_" + str(uuid.uuid4()).encode()
    
    def _sign_quantum(self, quantum_private: Any, message: Union[str, bytes]) -> bytes:
        """Create a quantum-topological signature for the given message."""
        # In a real implementation, this would use quantum-topological signing
        # Here we're simulating the process
        return b"mock_quantum_signature_" + str(uuid.uuid4()).encode()
    
    def _analyze_signature_topology(self, 
                                  message: Union[str, bytes], 
                                  ecdsa_signature: bytes, 
                                  quantum_signature: Optional[bytes]) -> float:
        """
        Analyze the topological properties of a signature to calculate TVI.
        
        This method transforms the signature into topological space and analyzes
        its structure to detect vulnerabilities that might be missed by traditional methods.
        
        Args:
            message: The original message
            ecdsa_signature: The ECDSA signature component
            quantum_signature: The quantum-topological signature component (optional)
            
        Returns:
            float: TVI score (0.0 = secure, 1.0 = critical vulnerability)
            
        As stated in documentation: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
        точную количественную оценку структуры пространства подписей и обнаруживает скрытые
        уязвимости, которые пропускаются другими методами."
        """
        # Update the hypercube with the new signature data
        self.hypercube.update_with_signature(message, ecdsa_signature, quantum_signature)
        
        # Get the current TVI from the hypercube
        tvi = self.hypercube.get_tvi()
        
        # Update quantum security level based on analysis
        self._update_quantum_security_level()
        
        return tvi
    
    def _update_quantum_security_level(self):
        """Update the quantum security level based on current system state."""
        # This would analyze the quantum implementation's resistance to known attacks
        # In a real implementation, this would use actual quantum security metrics
        
        # For demonstration, we'll use a simple calculation based on TVI
        current_tvi = self.hypercube.get_tvi()
        self.quantum_security_level = max(0.0, min(1.0, 1.0 - current_tvi * 0.8))
    
    def verify(self, public_key: HybridKeyPair, message: Union[str, bytes], 
              signature: HybridSignature) -> Tuple[bool, float]:
        """
        Verify a hybrid signature.
        
        This method:
        - Always verifies the ECDSA signature for backward compatibility
        - Verifies the quantum-topological signature when available
        - Checks TVI to determine if the signature meets current security requirements
        
        Args:
            public_key: HybridKeyPair containing public components
            message: Message that was signed
            signature: HybridSignature to verify
            
        Returns:
            Tuple[bool, float]: (verification result, TVI score)
            
        The verification process employs Quantum Vulnerability Analysis to guide robust 
        quantum cryptographic implementations, ensuring security against potential quantum attacks. [[8]]
        """
        # Always verify ECDSA signature (required for backward compatibility)
        ecdsa_valid = self._verify_ecdsa(public_key.ecdsa_public, message, signature.ecdsa_signature)
        
        # Verify quantum signature if present and in appropriate phase
        quantum_valid = True
        if signature.quantum_signature is not None:
            quantum_valid = self._verify_quantum(public_key.quantum_public, 
                                               message, 
                                               signature.quantum_signature)
        
        # Check if signature meets current security requirements based on TVI
        is_secure = signature.is_secure()
        
        # Overall verification result requires:
        # 1. ECDSA signature valid
        # 2. Quantum signature valid if present
        # 3. Signature is secure (TVI < threshold)
        verification_result = ecdsa_valid and quantum_valid and is_secure
        
        if not verification_result:
            reasons = []
            if not ecdsa_valid:
                reasons.append("ECDSA signature invalid")
            if not quantum_valid and signature.quantum_signature is not None:
                reasons.append("Quantum signature invalid")
            if not is_secure:
                reasons.append(f"TVI too high ({signature.tvi:.4f} >= 0.5)")
            logger.warning(f"Signature verification failed: {', '.join(reasons)}")
        
        return verification_result, signature.tvi
    
    def _verify_ecdsa(self, ecdsa_public: Any, message: Union[str, bytes], 
                     signature: bytes) -> bool:
        """Verify an ECDSA signature."""
        # In a real implementation, this would use a proper ECDSA verification function
        # Here we're simulating the process
        return True  # Assume valid for demonstration
    
    def _verify_quantum(self, quantum_public: Any, message: Union[str, bytes], 
                       signature: bytes) -> bool:
        """Verify a quantum-topological signature."""
        # In a real implementation, this would use quantum-topological verification
        # Here we're simulating the process
        return quantum_public is not None  # Assume valid if quantum_public exists
    
    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a transaction through the hybrid cryptographic system.
        
        This method:
        - Analyzes the transaction's signature topology
        - Blocks transactions with TVI > 0.5 as per security policy
        - Handles both classical and quantum-topological signatures
        
        Args:
            transaction: Transaction dictionary containing signature and message
            
        Returns:
            Processed transaction with security assessment
            
        Example from documentation:
        "Плагин для Bitcoin Core: Автоматически проверяет входящие транзакции
        на наличие слабых подписей, Блокирует транзакции с TVI > 0.5."
        """
        # Extract signature and public key
        signature = transaction.get("signature")
        public_key = transaction.get("public_key")
        message = transaction.get("message")
        
        if not all([signature, public_key, message]):
            return {
                "status": "error",
                "reason": "Missing required transaction components"
            }
        
        # Analyze topology
        tvi_result = self._analyze_signature_topology(
            message, 
            signature.get("ecdsa_signature"), 
            signature.get("quantum_signature")
        )
        
        # Block transaction if TVI is too high
        if tvi_result > 0.5:
            return {
                "status": "rejected",
                "reason": "Transaction blocked due to high TVI",
                "tvi": tvi_result
            }
        
        # Verify the transaction
        verification_result, _ = self.verify(public_key, message, signature)
        
        if not verification_result:
            return {
                "status": "rejected",
                "reason": "Signature verification failed",
                "tvi": tvi_result
            }
        
        return {
            "status": "accepted",
            "tvi": tvi_result,
            "migration_phase": self.migration_phase.name
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the hybrid cryptographic system.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            "migration_phase": self.migration_phase.name,
            "phase_duration": time.time() - self.phase_start_time,
            "current_tvi": self.hypercube.get_tvi(),
            "quantum_security_level": self.quantum_security_level,
            "system_age": time.time() - self.phase_start_time,
            "tvi_history_sample": [tvi for _, tvi in self.tvi_history[-5:]] if self.tvi_history else []
        }
