"""
Hybrid Cryptographic System for QuantumFortress 2.0

This module implements the hybrid cryptographic system that enables seamless migration
from classical to post-quantum algorithms while maintaining full backward compatibility.
The system uses the Topological Vulnerability Index (TVI) as the primary metric for
determining migration phases and security status.

Key features:
- Hybrid signing with both ECDSA and quantum-topological signatures
- Automatic migration phases triggered by TVI thresholds
- Full backward compatibility with existing blockchain networks
- TVI-based transaction filtering (blocks transactions with TVI > 0.5)
- QuantumBridge integration for traditional network compatibility

As stated in Ur Uz работа.md: "Применение чисел Бетти к анализу ECDSA-Torus предоставляет
точную количественную оценку структуры пространства подписей и обнаруживает скрытые
уязвимости, которые пропускаются другими методами."

This implementation extends those principles to a hybrid cryptographic framework that
provides a practical path to post-quantum security.
"""

import time
import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass

from quantum_fortress.core.adaptive_hypercube import AdaptiveQuantumHypercube
from quantum_fortress.core.metrics import TopologicalMetrics, TVIResult
from quantum_fortress.topology.homology import HomologyAnalyzer
from quantum_fortress.crypto.quantum_sig import QuantumSignature
from quantum_fortress.utils.crypto_utils import (
    ecdsa_sign,
    ecdsa_verify,
    generate_ecdsa_keys,
    hash_message
)

# Configure module logger
import logging
logger = logging.getLogger(__name__)

# Constants
TVI_SECURE_THRESHOLD = 0.5  # Threshold for secure implementation
TVI_WARNING_THRESHOLD = 0.7  # Threshold for warning state
TVI_CRITICAL_THRESHOLD = 0.8  # Threshold for critical vulnerability
MIGRATION_PHASES = 3  # Total number of migration phases
DEFAULT_BASE_DIMENSION = 4
MIN_SIGNATURES_FOR_ANALYSIS = 100  # Minimum signatures for reliable TVI calculation


class MigrationPhase(Enum):
    """Migration phases from classical to post-quantum cryptography"""
    CLASSICAL_ONLY = 0  # Only classical algorithms (ECDSA)
    HYBRID = 1  # Hybrid mode (ECDSA + quantum-topological)
    POST_QUANTUM = 2  # Full post-quantum mode


@dataclass
class MigrationStatus:
    """Status of the migration process"""
    current_phase: MigrationPhase
    tvi_threshold: float
    signatures_analyzed: int
    secure_wallets: int
    vulnerable_wallets: int
    last_analysis: float
    phase_start_time: float
    estimated_completion: Optional[float] = None
    security_level: float = 100.0  # 0-100 scale


@dataclass
class HybridSignature:
    """Represents a hybrid signature combining classical and quantum-topological components"""
    ecdsa: Optional[Tuple[int, int]] = None  # (r, s) for ECDSA
    quantum: Optional[Dict[str, Any]] = None  # Quantum-topological signature
    migration_phase: int = 0
    tvi: float = 1.0
    signature_id: str = ""
    timestamp: float = 0.0


@dataclass
class HybridKeyPair:
    """Represents a hybrid key pair with both classical and quantum-topological components"""
    ecdsa: Optional[Dict[str, Any]] = None  # ECDSA key components
    quantum: Optional[Dict[str, Any]] = None  # Quantum-topological key components
    migration_phase: int = 0
    tvi_threshold: float = TVI_SECURE_THRESHOLD
    key_id: str = ""
    creation_time: float = 0.0
    last_usage: float = 0.0


class HybridCryptoSystem:
    """
    Hybrid Cryptographic System for QuantumFortress 2.0
    
    This class implements a cryptographic system that seamlessly migrates from classical
    to post-quantum algorithms based on topological security metrics. The system:
    - Maintains full backward compatibility with existing blockchain networks
    - Uses TVI as the primary metric for migration decisions
    - Automatically adjusts security parameters based on network conditions
    - Provides quantitative security metrics instead of subjective assessments
    
    The implementation follows the philosophy: "Topology isn't a hacking tool, but a microscope
    for diagnosing vulnerabilities. Ignoring it means building cryptography on sand."
    
    Example:
        >>> crypto = HybridCryptoSystem(base_dimension=4)
        >>> keys = crypto.generate_keys()
        >>> signature = crypto.sign(keys["private"], "transaction_data")
        >>> is_valid = crypto.verify(keys["public"], "transaction_data", signature)
        >>> print(f"TVI: {signature.tvi:.4f} (Phase: {signature.migration_phase})")
    """
    
    def __init__(self, base_dimension: int = DEFAULT_BASE_DIMENSION, 
                 hypercube: Optional[AdaptiveQuantumHypercube] = None):
        """
        Initialize the hybrid cryptographic system.
        
        Args:
            base_dimension: Base dimension for the quantum hypercube
            hypercube: Optional pre-configured quantum hypercube
        """
        self.base_dimension = base_dimension
        self.hypercube = hypercube or AdaptiveQuantumHypercube(dimension=base_dimension)
        self.topology_analyzer = HomologyAnalyzer(dimension=base_dimension)
        self.migration_phase = MigrationPhase.CLASSICAL_ONLY
        self.signatures_analyzed = 0
        self.secure_wallets = 0
        self.vulnerable_wallets = 0
        self.phase_start_time = time.time()
        self.last_analysis = 0.0
        
        logger.info(
            f"Initialized HybridCryptoSystem (phase={self.migration_phase.name}, "
            f"dimension={self.base_dimension})"
        )
    
    def generate_keys(self) -> HybridKeyPair:
        """
        Generate hybrid cryptographic keys.
        
        This method creates both classical (ECDSA) and quantum-topological key components,
        with the quantum component being activated based on the current migration phase.
        
        Returns:
            HybridKeyPair object containing both key types
            
        Example from Ur Uz работа.md:
            "Плагин для Bitcoin Core: Автоматически проверяет входящие транзакции 
            на наличие слабых подписей, Блокирует транзакции с TVI > 0.5."
        """
        import uuid
        key_id = str(uuid.uuid4())
        creation_time = time.time()
        
        # Generate ECDSA keys (always included for backward compatibility)
        ecdsa_private, ecdsa_public = generate_ecdsa_keys()
        
        # Generate quantum-topological keys based on migration phase
        quantum_keys = None
        if self.migration_phase in [MigrationPhase.HYBRID, MigrationPhase.POST_QUANTUM]:
            quantum_keys = self._generate_quantum_keys()
        
        # Determine current migration phase for these keys
        current_phase = self._determine_migration_phase()
        
        logger.debug(f"Generated hybrid keys (phase={current_phase.name}, id={key_id})")
        return HybridKeyPair(
            ecdsa={
                "private": ecdsa_private,
                "public": ecdsa_public
            },
            quantum=quantum_keys,
            migration_phase=current_phase.value,
            tvi_threshold=TVI_SECURE_THRESHOLD,
            key_id=key_id,
            creation_time=creation_time,
            last_usage=creation_time
        )
    
    def _generate_quantum_keys(self) -> Dict[str, Any]:
        """
        Generate quantum-topological cryptographic keys.
        
        This method creates keys within the quantum hypercube structure, where:
        - The private key is a trajectory through the hypercube
        - The public key is a projection onto the hypercube surface
        
        Returns:
            Dictionary containing quantum key components
        """
        # Generate random trajectory in the hypercube
        trajectory = self._generate_quantum_trajectory()
        
        # Get starting point for public key
        start_point = trajectory[0]
        
        # Compute public key as projection
        public_key = self._project_to_surface(start_point)
        
        return {
            "trajectory": trajectory,
            "current_position": 0,
            "dimension": self.hypercube.dimension,
            "topology_metrics": self.hypercube.get_current_metrics()
        }
    
    def _generate_quantum_trajectory(self, length: int = 100) -> List[Tuple[float, float]]:
        """
        Generate a random quantum trajectory through the hypercube.
        
        Args:
            length: Length of the trajectory
            
        Returns:
            List of points representing the trajectory
        """
        trajectory = []
        current_point = (np.random.random(), np.random.random())
        
        for _ in range(length):
            # Move to a new point with small random step
            step_r = (np.random.random() - 0.5) * 0.2
            step_z = (np.random.random() - 0.5) * 0.2
            
            # Ensure we stay on the torus (wrap around)
            new_r = (current_point[0] + step_r) % 1.0
            new_z = (current_point[1] + step_z) % 1.0
            
            trajectory.append((new_r, new_z))
            current_point = (new_r, new_z)
        
        return trajectory
    
    def _project_to_surface(self, point: Tuple[float, float]) -> Dict[str, Any]:
        """
        Project a point in the hypercube to the surface for public key generation.
        
        Args:
            point: Point in the hypercube space
            
        Returns:
            Dictionary representing the projected public key
        """
        # In a real implementation, this would use actual cryptographic operations
        # For demonstration, we'll create a simple hash-based projection
        import hashlib
        projection = hashlib.sha256(f"{point[0]},{point[1]}".encode()).hexdigest()
        
        return {
            "x": projection[:32],
            "y": projection[32:],
            "point": point
        }
    
    def sign(self, private_key: HybridKeyPair, 
             message: Union[str, bytes]) -> HybridSignature:
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
            
        Example from Ur Uz работа.md:
            "Works as API wrapper (no core modifications needed)"
        """
        import uuid
        signature_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Always sign with ECDSA for backward compatibility
        ecdsa_signature = ecdsa_sign(private_key.ecdsa["private"], message)
        
        # Sign with quantum-topological method if in appropriate phase
        quantum_signature = None
        if private_key.migration_phase >= MigrationPhase.HYBRID.value:
            quantum_signature = self._quantum_sign(
                private_key.quantum, 
                message, 
                ecdsa_signature
            )
        
        # Analyze the signature topology
        tvi_result = self._analyze_signature_topology(ecdsa_signature, message)
        
        # Update migration phase if needed
        self._update_migration_phase()
        
        logger.info(
            f"Created hybrid signature (phase={private_key.migration_phase}, "
            f"TVI={tvi_result.tvi:.4f}, id={signature_id})"
        )
        return HybridSignature(
            ecdsa=ecdsa_signature,
            quantum=quantum_signature,
            migration_phase=private_key.migration_phase,
            tvi=tvi_result.tvi,
            signature_id=signature_id,
            timestamp=timestamp
        )
    
    def _quantum_sign(self, quantum_private: Dict[str, Any], 
                      message: Union[str, bytes], 
                      ecdsa_signature: Tuple[int, int]) -> Dict[str, Any]:
        """
        Create a quantum-topological signature.
        
        Args:
            quantum_private: Quantum private key components
            message: Message to sign
            ecdsa_signature: Corresponding ECDSA signature
            
        Returns:
            Dictionary containing quantum signature components
        """
        # Get current position in trajectory
        current_pos = quantum_private["current_position"]
        trajectory = quantum_private["trajectory"]
        
        # Get current point in trajectory
        ur, uz = trajectory[current_pos]
        
        # Hash the message
        z = hash_message(message)
        
        # Use ECDSA signature components for consistency
        r, s = ecdsa_signature
        
        # Generate quantum signature components
        # In a real implementation, this would use actual quantum operations
        quantum_r = (r * (1 + ur)) % quantum_private["dimension"]
        quantum_s = (s * (1 + uz)) % quantum_private["dimension"]
        
        # Update position in trajectory (cyclic)
        quantum_private["current_position"] = (current_pos + 1) % len(trajectory)
        
        return {
            "r": quantum_r,
            "s": quantum_s,
            "ur": ur,
            "uz": uz,
            "position": current_pos,
            "dimension": quantum_private["dimension"]
        }
    
    def verify(self, public_key: HybridKeyPair, 
               message: Union[str, bytes], 
               signature: HybridSignature) -> bool:
        """
        Verify a hybrid signature.
        
        This method:
        - Always verifies the ECDSA component for backward compatibility
        - Verifies the quantum-topological component when present
        - Checks TVI to ensure signature security
        - Rejects signatures with TVI > TVI_SECURE_THRESHOLD
        
        Args:
            public_key: HybridKeyPair containing public components
            message: Message that was signed
            signature: HybridSignature to verify
            
        Returns:
            bool: True if signature is valid and secure, False otherwise
        """
        # Always verify ECDSA component (required for backward compatibility)
        ecdsa_valid = ecdsa_verify(public_key.ecdsa["public"], message, signature.ecdsa)
        if not ecdsa_valid:
            logger.warning("ECDSA signature verification failed")
            return False
        
        # Verify quantum component if present
        quantum_valid = True
        if signature.quantum is not None:
            quantum_valid = self._quantum_verify(
                public_key.quantum, 
                message, 
                signature.quantum, 
                signature.ecdsa
            )
            if not quantum_valid:
                logger.warning("Quantum signature verification failed")
        
        # Check TVI security threshold
        if signature.tvi > TVI_SECURE_THRESHOLD:
            logger.warning(f"Signature rejected due to high TVI ({signature.tvi:.4f} > {TVI_SECURE_THRESHOLD})")
            return False
        
        # All checks passed
        return ecdsa_valid and quantum_valid
    
    def _quantum_verify(self, quantum_public: Dict[str, Any], 
                        message: Union[str, bytes], 
                        quantum_signature: Dict[str, Any], 
                        ecdsa_signature: Tuple[int, int]) -> bool:
        """
        Verify a quantum-topological signature.
        
        Args:
            quantum_public: Quantum public key components
            message: Message that was signed
            quantum_signature: Quantum signature components
            ecdsa_signature: Corresponding ECDSA signature
            
        Returns:
            bool: True if quantum signature is valid
        """
        # Extract components
        r, s = ecdsa_signature
        quantum_r = quantum_signature["r"]
        quantum_s = quantum_signature["s"]
        ur = quantum_signature["ur"]
        uz = quantum_signature["uz"]
        
        # Verify consistency with ECDSA signature
        expected_r = r * (1 + ur)
        expected_s = s * (1 + uz)
        
        # Check if quantum signature matches expected values
        r_match = abs(quantum_r - expected_r) < 1e-10
        s_match = abs(quantum_s - expected_s) < 1e-10
        
        return r_match and s_match
    
    def _analyze_signature_topology(self, 
                                   ecdsa_signature: Tuple[int, int], 
                                   message: Union[str, bytes]) -> TVIResult:
        """
        Analyze the topological structure of a signature.
        
        This method transforms the signature into the (u_r, u_z) space and analyzes
        its topological properties to calculate the TVI (Topological Vulnerability Index).
        
        Args:
            ecdsa_signature: ECDSA signature components (r, s)
            message: Message that was signed
            
        Returns:
            TVIResult object with vulnerability assessment
            
        Example from Ur Uz работа.md:
            "Применение чисел Бетти к анализу ECDSA-Torus предоставляет точную 
            количественную оценку структуры пространства подписей"
        """
        try:
            # Extract signature components
            r, s = ecdsa_signature
            
            # Hash the message
            z = hash_message(message)
            
            # Transform to (u_r, u_z) space as in Ur Uz работа.md
            u_r = (r * pow(s, -1, self.hypercube.dimension)) % 1.0
            u_z = (z * pow(s, -1, self.hypercube.dimension)) % 1.0
            
            # Analyze topology
            topology_metrics = self.topology_analyzer.analyze([(u_r, u_z)])
            
            # Update tracking metrics
            self.signatures_analyzed += 1
            if topology_metrics.tvi < TVI_SECURE_THRESHOLD:
                self.secure_wallets += 1
            else:
                self.vulnerable_wallets += 1
            
            self.last_analysis = time.time()
            
            return TVIResult(
                tvi=topology_metrics.tvi,
                is_secure=topology_metrics.tvi < TVI_SECURE_THRESHOLD,
                vulnerability_type=self._determine_vulnerability_type(topology_metrics),
                explanation=self._generate_vulnerability_explanation(topology_metrics)
            )
            
        except Exception as e:
            logger.error(f"Topology analysis failed: {str(e)}")
            return TVIResult(
                tvi=1.0,
                is_secure=False,
                vulnerability_type="unknown",
                explanation="Topology analysis failed"
            )
    
    def _determine_vulnerability_type(self, metrics: TopologicalMetrics) -> str:
        """
        Determine the type of vulnerability based on topological metrics.
        
        Args:
            metrics: Topological metrics from analysis
            
        Returns:
            String describing the vulnerability type
        """
        if metrics.tvi < TVI_SECURE_THRESHOLD:
            return "none"
        
        # Check for specific vulnerability patterns
        if abs(metrics.betti_numbers[1] - self.base_dimension) > 0.5:
            return "topological_structure"
            
        if metrics.topological_entropy < 0.6 * np.log(self.base_dimension):
            return "entropy_deficiency"
            
        if metrics.naturalness_coefficient > 0.4:
            return "predictability"
            
        if metrics.euler_characteristic != 0:
            return "manifold_distortion"
            
        return "unknown"
    
    def _generate_vulnerability_explanation(self, metrics: TopologicalMetrics) -> str:
        """Generate explanation for vulnerability assessment"""
        if metrics.tvi < TVI_SECURE_THRESHOLD:
            return "No significant vulnerabilities detected. Topological structure is sound."
        
        vuln_type = self._determine_vulnerability_type(metrics)
        
        explanations = {
            "topological_structure": (
                f"Topological structure anomaly detected (β₁ = {metrics.betti_numbers[1]:.2f}, "
                f"expected ≈ {self.base_dimension}). This indicates potential weaknesses in the "
                "signature space structure."
            ),
            "entropy_deficiency": (
                f"Topological entropy deficiency ({metrics.topological_entropy:.4f} < "
                f"{0.6 * np.log(self.base_dimension):.4f}). This suggests insufficient randomness "
                "in the signature generation process."
            ),
            "predictability": (
                f"High predictability detected (naturalness coefficient = {metrics.naturalness_coefficient:.4f} > 0.4). "
                "This indicates patterns that could be exploited to predict future signatures."
            ),
            "manifold_distortion": (
                f"Manifold distortion detected (Euler characteristic = {metrics.euler_characteristic}). "
                "The signature space does not maintain the expected topological properties."
            ),
            "unknown": (
                f"Security vulnerability detected (TVI = {metrics.tvi:.4f} > {TVI_SECURE_THRESHOLD}). "
                "Further analysis required to determine specific vulnerability type."
            )
        }
        
        return explanations.get(vuln_type, explanations["unknown"])
    
    def _determine_migration_phase(self) -> MigrationPhase:
        """
        Determine the current migration phase based on network security metrics.
        
        Migration follows the sequence:
        0: CLASSICAL_ONLY - Only ECDSA, TVI > 0.5 for many wallets
        1: HYBRID - ECDSA + quantum-topological, TVI around 0.5
        2: POST_QUANTUM - Only quantum-topological, TVI < 0.3
        
        Returns:
            MigrationPhase enum value
        """
        # Not enough data yet
        if self.signatures_analyzed < MIN_SIGNATURES_FOR_ANALYSIS:
            return MigrationPhase.CLASSICAL_ONLY
        
        # Calculate secure wallet percentage
        secure_percentage = self.secure_wallets / self.signatures_analyzed
        
        # Determine phase based on secure percentage
        if secure_percentage < 0.3:
            return MigrationPhase.CLASSICAL_ONLY
        elif secure_percentage < 0.7:
            return MigrationPhase.HYBRID
        else:
            return MigrationPhase.POST_QUANTUM
    
    def _update_migration_phase(self) -> bool:
        """
        Update the migration phase if conditions are met.
        
        Returns:
            bool: True if phase changed, False otherwise
        """
        new_phase = self._determine_migration_phase()
        
        # No change needed
        if new_phase == self.migration_phase:
            return False
        
        # Update phase
        old_phase = self.migration_phase
        self.migration_phase = new_phase
        self.phase_start_time = time.time()
        
        logger.info(
            f"Migration phase updated: {old_phase.name} → {new_phase.name} "
            f"({self.secure_wallets}/{self.signatures_analyzed} secure wallets)"
        )
        return True
    
    def get_migration_status(self) -> MigrationStatus:
        """
        Get the current status of the migration process.
        
        Returns:
            MigrationStatus object with detailed information
        """
        current_time = time.time()
        secure_percentage = (
            self.secure_wallets / self.signatures_analyzed 
            if self.signatures_analyzed > 0 else 0
        )
        
        # Estimate completion time (simplified)
        estimated_completion = None
        if self.migration_phase == MigrationPhase.CLASSICAL_ONLY and secure_percentage > 0:
            time_per_wallet = (current_time - self.phase_start_time) / max(1, self.signatures_analyzed)
            wallets_remaining = int((0.3 - secure_percentage) * 10000)  # Estimate to reach 30%
            estimated_completion = current_time + (time_per_wallet * wallets_remaining)
        
        elif self.migration_phase == MigrationPhase.HYBRID and secure_percentage > 0.3:
            time_per_wallet = (current_time - self.phase_start_time) / max(1, self.signatures_analyzed - self.secure_wallets)
            wallets_remaining = int((0.7 - secure_percentage) * 10000)  # Estimate to reach 70%
            estimated_completion = current_time + (time_per_wallet * wallets_remaining)
        
        return MigrationStatus(
            current_phase=self.migration_phase,
            tvi_threshold=TVI_SECURE_THRESHOLD,
            signatures_analyzed=self.signatures_analyzed,
            secure_wallets=self.secure_wallets,
            vulnerable_wallets=self.vulnerable_wallets,
            last_analysis=self.last_analysis,
            phase_start_time=self.phase_start_time,
            estimated_completion=estimated_completion,
            security_level=min(100.0, max(0.0, 100.0 * (1.0 - (self.vulnerable_wallets / max(1, self.signatures_analyzed))))))
    
    def convert_to_quantum(self, ecdsa_private_key: Any) -> HybridKeyPair:
        """
        Convert a traditional ECDSA private key to a quantum-topological equivalent.
        
        This implements the QuantumBridge functionality described in:
        "QuantumBridge — API wrapper implementation requiring no core modifications"
        
        Args:
            ecdsa_private_key: Traditional ECDSA private key
            
        Returns:
            HybridKeyPair with quantum-topological components
        """
        # Generate base hybrid keys
        hybrid_keys = self.generate_keys()
        
        # In a real implementation, we would map the ECDSA key to a quantum trajectory
        # For demonstration, we'll use the private key as a seed for the trajectory
        import hashlib
        seed = int(hashlib.sha256(str(ecdsa_private_key).encode()).hexdigest(), 16)
        
        # Generate quantum trajectory based on seed
        np.random.seed(seed % (2**32))
        trajectory = self._generate_quantum_trajectory()
        
        # Update quantum components
        hybrid_keys.quantum = {
            "trajectory": trajectory,
            "current_position": 0,
            "dimension": self.hypercube.dimension,
            "topology_metrics": self.hypercube.get_current_metrics()
        }
        
        # Set migration phase to HYBRID (since we're converting from classical)
        hybrid_keys.migration_phase = MigrationPhase.HYBRID.value
        
        logger.debug("Converted ECDSA key to quantum-topological equivalent")
        return hybrid_keys
    
    def process_transaction(self, legacy_transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a transaction from a traditional blockchain network.
        
        This implements the QuantumBridge functionality that allows:
        - Automatic detection of vulnerable wallets
        - TVI-based transaction filtering
        - Seamless integration with existing networks
        
        Args:
            legacy_transaction: Transaction from traditional network (Bitcoin/Ethereum)
            
        Returns:
            Processed transaction in QuantumFortress format
            
        Example from Ur Uz работа.md:
            "Плагин для Bitcoin Core: Автоматически проверяет входящие транзакции 
            на наличие слабых подписей, Блокирует транзакции с TVI > 0.5."
        """
        # Extract signature and public key
        signature = legacy_transaction["signature"]
        public_key = legacy_transaction["public_key"]
        message = legacy_transaction["message"]
        
        # Analyze topology
        tvi_result = self._analyze_signature_topology(signature, message)
        
        # Block transaction if TVI is too high
        if tvi_result.tvi > TVI_SECURE_THRESHOLD:
            logger.warning(
                f"Blocked transaction with high TVI ({tvi_result.tvi:.4f} > {TVI_SECURE_THRESHOLD})"
            )
            return {
                "status": "rejected",
                "reason": "high_tvi",
                "tvi": tvi_result.tvi,
                "explanation": tvi_result.explanation
            }
        
        # Convert to hybrid signature format
        hybrid_signature = HybridSignature(
            ecdsa=signature,
            quantum=None,  # Will be added if needed
            migration_phase=MigrationPhase.CLASSICAL_ONLY.value,
            tvi=tvi_result.tvi,
            signature_id=legacy_transaction.get("txid", ""),
            timestamp=time.time()
        )
        
        # If in HYBRID or POST_QUANTUM phase, add quantum component
        if self.migration_phase in [MigrationPhase.HYBRID, MigrationPhase.POST_QUANTUM]:
            # In a real implementation, we would generate a quantum signature
            hybrid_signature.quantum = {
                "r": signature[0],
                "s": signature[1],
                "ur": 0.5,  # Placeholder
                "uz": 0.5,  # Placeholder
                "position": 0,
                "dimension": self.hypercube.dimension
            }
            hybrid_signature.migration_phase = self.migration_phase.value
        
        logger.info(f"Processed legacy transaction (TVI={tvi_result.tvi:.4f})")
        return {
            "status": "accepted",
            "signature": hybrid_signature,
            "tvi": tvi_result.tvi,
            "vulnerability_type": tvi_result.vulnerability_type
        }
    
    def get_security_recommendations(self, tvi_result: TVIResult) -> List[str]:
        """
        Generate security recommendations based on TVI analysis.
        
        Args:
            tvi_result: TVI analysis result
            
        Returns:
            List of security recommendations
            
        Example from Ur Uz работа.md:
            "Рекомендации:\n"
            "1. Замените текущий RNG на криптографически стойкий\n"
            "2. Используйте HMAC-DRBG вместо текущего алгоритма\n"
            "3. Рассмотрите внедрение TopoNonce для равномерного покрытия тора\n"
        """
        recommendations = []
        
        # TVI-based recommendations
        if tvi_result.tvi > TVI_CRITICAL_THRESHOLD:
            recommendations.append(
                "CRITICAL VULNERABILITY DETECTED: Immediately replace all keys and "
                "consider all funds at risk. TVI score indicates severe structural issues."
            )
        elif tvi_result.tvi > TVI_WARNING_THRESHOLD:
            recommendations.append(
                "HIGH RISK: Vulnerability detected that could lead to private key recovery. "
                "Replace keys as soon as possible and investigate RNG implementation."
            )
        elif tvi_result.tvi > TVI_SECURE_THRESHOLD:
            recommendations.append(
                "MEDIUM RISK: Potential vulnerability detected. Consider upgrading to "
                "hybrid mode and implementing TopoNonce for improved security."
            )
        
        # Specific vulnerability recommendations
        if tvi_result.vulnerability_type == "topological_structure":
            recommendations.append(
                "Topological structure anomaly detected (β₁ deviation). "
                "Ensure proper implementation of signature generation with uniform coverage of the torus."
            )
        elif tvi_result.vulnerability_type == "entropy_deficiency":
            recommendations.append(
                "Entropy deficiency detected. Use a cryptographically secure RNG and "
                "consider implementing additional entropy sources."
            )
        elif tvi_result.vulnerability_type == "predictability":
            recommendations.append(
                "Predictability vulnerability detected. Implement TopoNonce to ensure "
                "uniform distribution across the signature space."
            )
        elif tvi_result.vulnerability_type == "manifold_distortion":
            recommendations.append(
                "Manifold distortion detected. Verify that the signature space maintains "
                "the expected topological properties of a torus."
            )
        
        # General recommendations
        if self.migration_phase == MigrationPhase.CLASSICAL_ONLY:
            recommendations.append(
                "Consider enabling hybrid mode to begin migration to quantum-resistant "
                "cryptography. Current system is vulnerable to future quantum attacks."
            )
        elif self.migration_phase == MigrationPhase.HYBRID:
            if self.secure_wallets / max(1, self.signatures_analyzed) > 0.5:
                recommendations.append(
                    "Migration to post-quantum mode is progressing well. "
                    "Consider increasing the TVI threshold to accelerate migration."
                )
        
        return recommendations
    
    def get_tvi_threshold(self) -> float:
        """
        Get the current TVI threshold for security validation.
        
        The threshold may change based on migration phase:
        - CLASSICAL_ONLY: 0.5 (standard threshold)
        - HYBRID: 0.4 (stricter as we move toward post-quantum)
        - POST_QUANTUM: 0.3 (most strict for full post-quantum security)
        
        Returns:
            Current TVI threshold value
        """
        if self.migration_phase == MigrationPhase.CLASSICAL_ONLY:
            return TVI_SECURE_THRESHOLD
        elif self.migration_phase == MigrationPhase.HYBRID:
            return 0.4
        else:  # POST_QUANTUM
            return 0.3
    
    def analyze_network_security(self) -> Dict[str, Any]:
        """
        Analyze the overall security of the network based on collected data.
        
        Returns:
            Dictionary with network security metrics
        """
        if self.signatures_analyzed == 0:
            return {
                "status": "insufficient_data",
                "message": "Not enough signatures analyzed for network security assessment"
            }
        
        secure_percentage = self.secure_wallets / self.signatures_analyzed
        vulnerable_percentage = self.vulnerable_wallets / self.signatures_analyzed
        
        # Determine overall network security level
        if secure_percentage > 0.7:
            security_level = "high"
        elif secure_percentage > 0.3:
            security_level = "medium"
        else:
            security_level = "low"
        
        return {
            "signatures_analyzed": self.signatures_analyzed,
            "secure_wallets": self.secure_wallets,
            "vulnerable_wallets": self.vulnerable_wallets,
            "secure_percentage": secure_percentage,
            "vulnerable_percentage": vulnerable_percentage,
            "security_level": security_level,
            "migration_phase": self.migration_phase.name,
            "tvi_threshold": self.get_tvi_threshold(),
            "last_analysis": self.last_analysis
        }
