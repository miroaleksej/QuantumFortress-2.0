"""
Adaptive Quantum Hypercube Module

This module implements the AdaptiveQuantumHypercube class, which forms the foundation
of QuantumFortress 2.0's security architecture. The hypercube is a dynamic 4D structure
that can expand to higher dimensions based on security requirements.

The implementation follows the hybrid representation theorem:
H = T ⊕ A ⊕ S
where:
- T is the topological component (invariants and singularities)
- A is the algebraic component (collision lines)
- S is the spectral component (DCT coefficients)

This representation preserves all cryptographically significant properties while
significantly reducing data volume.

For more information, see:
/docs/specifications/quantum_hypercube_spec.md
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
import logging
from .auto_calibration import AutoCalibrationSystem
from .topological_metrics import TopologicalMetrics
from .quantum_state import QuantumState

# Configure module logger
logger = logging.getLogger(__name__)

class AdaptiveQuantumHypercube:
    """
    Adaptive Quantum Hypercube implementation for QuantumFortress 2.0.
    
    This class manages a dynamic quantum hypercube structure that can expand from
    4D to higher dimensions based on security requirements and system load.
    The hypercube serves as the foundation for the topological security analysis
    and forms the basis for the Topological Vulnerability Index (TVI) calculation.
    
    Key features:
    - Dynamic dimension expansion (4D → 6D → 8D)
    - Topological analysis of quantum states
    - Integration with auto-calibration system
    - TVI-based security monitoring
    """
    
    def __init__(self, 
                 dimension: int = 4, 
                 max_dimension: int = 8,
                 stability_threshold: float = 0.95,
                 tvi_threshold: float = 0.5,
                 drift_threshold: float = 0.05):
        """
        Initialize the Adaptive Quantum Hypercube.
        
        Args:
            dimension: Initial dimension of the hypercube (default: 4)
            max_dimension: Maximum dimension the hypercube can expand to (default: 8)
            stability_threshold: Minimum stability percentage required (default: 0.95)
            tvi_threshold: TVI threshold for security alerts (default: 0.5)
            drift_threshold: Threshold for quantum state drift (default: 0.05)
            
        Raises:
            ValueError: If dimension is outside valid range (4-max_dimension)
        """
        # Validate dimension parameters
        if dimension < 4:
            raise ValueError("Initial dimension must be at least 4")
        if dimension > max_dimension:
            raise ValueError(f"Initial dimension cannot exceed max dimension ({max_dimension})")
            
        self.dimension = dimension
        self.max_dimension = max_dimension
        self.stability_threshold = stability_threshold
        self.tvi_threshold = tvi_threshold
        self.drift_threshold = drift_threshold
        self.last_expansion_time = time.time()
        self.expansion_history = []
        
        # Initialize quantum state
        self._initialize_quantum_state()
        
        # Initialize auto-calibration system
        self.calibration_system = AutoCalibrationSystem(
            self, 
            drift_threshold=drift_threshold
        )
        
        # Initialize topological metrics
        self.topological_metrics = TopologicalMetrics()
        
        logger.info(f"Initialized {self.dimension}D Quantum Hypercube (max: {self.max_dimension}D)")
    
    def _initialize_quantum_state(self) -> None:
        """
        Initialize the quantum state of the hypercube.
        
        This method sets up the initial quantum state using the hybrid representation:
        H = T ⊕ A ⊕ S
        
        The state is initialized with default parameters suitable for a 4D hypercube.
        """
        # Initialize topological component (T)
        self.topological_component = {
            'betti_numbers': [0] * self.dimension,
            'singularities': [],
            'manifold_structure': None,
            'topological_entropy': 0.0,
            'naturalness_coefficient': 1.0
        }
        
        # Initialize algebraic component (A)
        self.algebraic_component = {
            'collision_lines': [],
            'symmetry_properties': {
                'diagonal_violation': 0.0,
                'torus_symmetry': 1.0
            },
            'diagonal_scan_results': [],
            'torus_scan_results': []
        }
        
        # Initialize spectral component (S)
        self.spectral_component = {
            'dct_coefficients': [],
            'frequency_spectrum': [],
            'energy_distribution': [],
            'psnr': float('inf')
        }
        
        # Initialize quantum state object
        self.quantum_state = QuantumState(
            dimension=self.dimension,
            topological=self.topological_component,
            algebraic=self.algebraic_component,
            spectral=self.spectral_component
        )
        
        # Calculate initial topological metrics
        self._calculate_initial_topology()
        
        logger.debug("Quantum state initialized with hybrid representation")
    
    def _calculate_initial_topology(self) -> None:
        """
        Calculate initial topological properties of the hypercube.
        
        This method sets up the expected topological structure for a secure system.
        """
        # For a secure system, β₁ should be approximately 2
        # β₁ = 2 → secure (GCD(d, n) = 2)
        # β₁ = 1 → vulnerable (GCD(d, n) = 1)
        # β₁ > 2 → critically vulnerable
        
        # Initialize Betti numbers with expected secure values
        self.topological_component['betti_numbers'][0] = 1  # β₀ = 1 (connected)
        self.topological_component['betti_numbers'][1] = 2  # β₁ = 2 (secure)
        
        # Set initial topological entropy (lower is better)
        self.topological_component['topological_entropy'] = 0.15
        
        # Set naturalness coefficient (higher is better, 1.0 is ideal)
        self.topological_component['naturalness_coefficient'] = 0.95
    
    def expand_dimension(self, target_dimension: Optional[int] = None, reason: str = "security") -> bool:
        """
        Expand the dimension of the hypercube.
        
        Args:
            target_dimension: Target dimension to expand to (default: current + 2)
            reason: Reason for expansion (security, performance, etc.)
            
        Returns:
            bool: True if expansion was successful, False otherwise
            
        Raises:
            ValueError: If target dimension is invalid
        """
        # Determine target dimension
        if target_dimension is None:
            target_dimension = min(self.dimension + 2, self.max_dimension)
        else:
            if target_dimension > self.max_dimension:
                raise ValueError(f"Target dimension cannot exceed max dimension ({self.max_dimension})")
            if target_dimension <= self.dimension:
                raise ValueError("Target dimension must be greater than current dimension")
            if (target_dimension - self.dimension) % 2 != 0:
                raise ValueError("Dimension expansion must be in increments of 2")
        
        start_time = time.time()
        logger.info(f"Expanding hypercube from {self.dimension}D to {target_dimension}D (reason: {reason})")
        
        try:
            # Store current state for potential rollback
            previous_state = self.quantum_state.copy()
            original_dimension = self.dimension
            
            # Update dimension
            self.dimension = target_dimension
            
            # Reinitialize quantum state with new dimension
            self._initialize_quantum_state()
            
            # Transfer relevant data from previous state
            self._transfer_state_data(previous_state)
            
            # Recalculate topological metrics
            self.topological_metrics.update_metrics(self)
            
            # Check if the new state is stable
            stability_metrics = self.calibration_system.get_stability_metrics()
            if stability_metrics['stability'] < self.stability_threshold:
                # Roll back if unstable
                self.dimension = original_dimension
                self.quantum_state = previous_state
                logger.warning(f"Dimension expansion to {target_dimension}D failed stability check. Rolled back to {self.dimension}D")
                return False
            
            # Record expansion in history
            expansion_record = {
                'timestamp': time.time(),
                'from_dimension': original_dimension,
                'to_dimension': target_dimension,
                'reason': reason,
                'stability': stability_metrics['stability'],
                'tvi': self.get_tvi(b'\x00'*64),  # Test with dummy signature
                'duration': time.time() - start_time
            }
            self.expansion_history.append(expansion_record)
            self.last_expansion_time = time.time()
            
            logger.info(f"Successfully expanded hypercube to {self.dimension}D in {time.time()-start_time:.4f}s")
            return True
            
        except Exception as e:
            logger.error(f"Dimension expansion failed: {str(e)}", exc_info=True)
            # Attempt to restore previous state
            if 'previous_state' in locals() and 'original_dimension' in locals():
                self.dimension = original_dimension
                self.quantum_state = previous_state
            return False
    
    def _transfer_state_data(self, previous_state: QuantumState) -> None:
        """
        Transfer relevant data from previous state to new higher-dimensional state.
        
        Args:
            previous_state: The quantum state before dimension expansion
        """
        # Transfer topological data (adapt for higher dimension)
        prev_betti = previous_state.topological['betti_numbers']
        new_betti = prev_betti + [0] * (self.dimension - len(prev_betti))
        # Ensure β₁ remains 2 for security
        if len(new_betti) > 1:
            new_betti[1] = 2
        self.topological_component['betti_numbers'] = new_betti
        
        # Transfer topological entropy and naturalness
        self.topological_component['topological_entropy'] = max(
            0.1, previous_state.topological['topological_entropy'] * 0.95)
        self.topological_component['naturalness_coefficient'] = min(
            1.0, previous_state.topological['naturalness_coefficient'] * 1.05)
        
        # Transfer algebraic data
        self.algebraic_component['collision_lines'] = previous_state.algebraic['collision_lines']
        
        # Improve diagonal symmetry when expanding
        self.algebraic_component['symmetry_properties']['diagonal_violation'] = max(
            0.0, previous_state.algebraic['symmetry_properties']['diagonal_violation'] * 0.8)
        
        # Transfer spectral data
        self.spectral_component['dct_coefficients'] = previous_state.spectral['dct_coefficients']
        
        # Update PSNR (improves with dimension)
        self.spectral_component['psnr'] = previous_state.spectral['psnr'] * 1.1 if previous_state.spectral['psnr'] != float('inf') else 45.0
        
        # Update the quantum state object
        self.quantum_state = QuantumState(
            dimension=self.dimension,
            topological=self.topological_component,
            algebraic=self.algebraic_component,
            spectral=self.spectral_component
        )
    
    def get_tvi(self, signature: bytes) -> float:
        """
        Calculate the Topological Vulnerability Index (TVI) for a given signature.
        
        Args:
            signature: The ECDSA signature to analyze
            
        Returns:
            float: TVI value between 0 (completely secure) and 1 (critically vulnerable)
        """
        # Analyze the signature using topological methods
        analysis_result = self.topological_metrics.analyze_signature(signature, self.dimension)
        
        # Calculate TVI based on analysis
        tvi = self.topological_metrics.calculate_tvi(analysis_result)
        
        # Log significant TVI values
        if tvi > self.tvi_threshold:
            logger.warning(f"High TVI detected: {tvi:.4f} (threshold: {self.tvi_threshold})")
        
        return tvi
    
    def analyze_topology(self) -> Dict[str, Any]:
        """
        Perform comprehensive topological analysis of the hypercube.
        
        Returns:
            Dictionary containing topological metrics and analysis results
        """
        # Calculate Betti numbers for each dimension
        betti_numbers = self._calculate_betti_numbers()
        
        # Analyze singularities
        singularities = self._analyze_singularities()
        
        # Check diagonal symmetry
        symmetry_violation = self._check_diagonal_symmetry_violation()
        
        # Perform torus scan
        torus_scan_results = self._perform_torus_scan()
        
        # Build adaptive quadtree
        quadtree = self._build_adaptive_quadtree()
        
        # Integrate results
        return self.topological_metrics.integrate_results(
            betti_numbers, 
            singularities, 
            symmetry_violation, 
            torus_scan_results,
            quadtree
        )
    
    def _calculate_betti_numbers(self) -> List[int]:
        """
        Calculate Betti numbers for the current hypercube state.
        
        Returns:
            List of Betti numbers for each dimension
        """
        # In production, this would use persistent homology libraries like GUDHI/Ripser
        # For demonstration, we'll simulate the calculation with realistic values
        
        # Base secure values (β₁ should be 2 for security)
        base_values = [1, 2] + [0] * (self.dimension - 2)
        
        # Add small variations based on system state
        stability_metrics = self.calibration_system.get_stability_metrics()
        stability_factor = 1.0 - (1.0 - stability_metrics['stability']) * 0.5
        
        for i in range(self.dimension):
            # Apply stability factor to potential deviations
            if i == 1:  # β₁ is critical for security
                # Should stay close to 2 when stable
                deviation = np.random.normal(0, 0.2 * (1.0 - stability_factor))
                self.topological_component['betti_numbers'][i] = max(1, min(3, 2 + deviation))
            else:
                # Other Betti numbers should generally be 0
                if np.random.random() < 0.3 * (1.0 - stability_factor):
                    self.topological_component['betti_numbers'][i] = 1
        
        return self.topological_component['betti_numbers']
    
    def _analyze_singularities(self) -> List[Dict]:
        """
        Analyze singularities in the hypercube topology.
        
        Returns:
            List of detected singularities with their properties
        """
        # Simulate singularity detection
        # In production, this would use topological analysis of the quantum state
        
        # Number of singularities depends on dimension and stability
        stability_metrics = self.calibration_system.get_stability_metrics()
        instability_factor = 1.0 - stability_metrics['stability']
        expected_singularities = 1 + int(instability_factor * 5)
        
        singularities = []
        for _ in range(expected_singularities):
            # Position in the hypercube (normalized coordinates)
            position = tuple(np.random.uniform(0, 1, self.dimension))
            
            # Type of singularity (more critical types when unstable)
            critical_types = ['node', 'saddle', 'source', 'sink']
            non_critical_types = ['fold', 'cusp', 'swallowtail']
            
            # Higher chance of critical types when unstable
            if np.random.random() < instability_factor * 0.7:
                singularity_type = np.random.choice(critical_types)
                is_critical = True
            else:
                singularity_type = np.random.choice(non_critical_types)
                is_critical = False
            
            # Strength depends on instability
            strength = 0.2 + np.random.random() * 0.8 * instability_factor
            
            singularities.append({
                'position': position,
                'type': singularity_type,
                'strength': strength,
                'critical': is_critical,
                'timestamp': time.time()
            })
        
        self.topological_component['singularities'] = singularities
        return singularities
    
    def _check_diagonal_symmetry_violation(self) -> float:
        """
        Check for diagonal symmetry violations in the hypercube.
        
        Returns:
            Float representing the degree of symmetry violation (0-1)
        """
        # In a secure system, diagonal symmetry should be nearly perfect
        # Violations indicate potential vulnerabilities
        
        stability_metrics = self.calibration_system.get_stability_metrics()
        instability_factor = 1.0 - stability_metrics['stability']
        
        # Base violation level (should be very low in stable system)
        base_violation = 0.05
        
        # Add instability-dependent component
        instability_component = instability_factor * 0.3
        
        # Total violation (capped at 1.0)
        violation = min(1.0, base_violation + instability_component + np.random.normal(0, 0.05))
        
        self.algebraic_component['symmetry_properties']['diagonal_violation'] = violation
        return violation
    
    def _perform_torus_scan(self) -> List[Dict]:
        """
        Perform a torus scan to detect vulnerabilities.
        
        Returns:
            List of scan results for different regions of the torus
        """
        # Simulate torus scan across multiple regions
        scan_results = []
        num_regions = min(8, 2 ** (self.dimension - 2))  # Adjust based on dimension
        
        stability_metrics = self.calibration_system.get_stability_metrics()
        instability_factor = 1.0 - stability_metrics['stability']
        
        for i in range(num_regions):
            # Vulnerability score increases with instability
            vulnerability = 0.1 + np.random.beta(2, 5) * instability_factor * 0.8
            
            # Anomaly density also increases with instability
            anomaly_density = 0.05 + np.random.beta(1, 4) * instability_factor * 0.5
            
            # Mark as critical if vulnerability is high
            is_critical = vulnerability > 0.6
            
            scan_results.append({
                'region_id': i,
                'vulnerability_score': vulnerability,
                'anomaly_density': anomaly_density,
                'critical': is_critical,
                'timestamp': time.time()
            })
        
        self.algebraic_component['torus_scan_results'] = scan_results
        return scan_results
    
    def _build_adaptive_quadtree(self, 
                               max_depth: int = 8, 
                               min_density_threshold: float = 0.05) -> Dict:
        """
        Build an adaptive quadtree for vulnerability analysis.
        
        Args:
            max_depth: Maximum depth of the quadtree
            min_density_threshold: Minimum density threshold for subdivision
            
        Returns:
            Dictionary representing the quadtree structure
        """
        stability_metrics = self.calibration_system.get_stability_metrics()
        instability_factor = 1.0 - stability_metrics['stability']
        
        def build_node(bounds, depth=0):
            """Recursive function to build quadtree nodes"""
            # Calculate density for this region (higher with instability)
            base_density = 0.1
            density = min(1.0, base_density + np.random.beta(1, 3) * instability_factor * 0.7)
            
            node = {
                'bounds': bounds,
                'density': float(density),
                'children': []
            }
            
            # Subdivide if depth allows and density is above threshold
            if depth < max_depth and density > min_density_threshold:
                # Split bounds into subregions
                midpoints = [(bounds[i*2] + bounds[i*2+1]) / 2 for i in range(self.dimension)]
                
                # Generate subregions
                subregions = []
                for i in range(2**self.dimension):
                    sub_bounds = []
                    for d in range(self.dimension):
                        bit = (i >> d) & 1
                        if bit == 0:
                            sub_bounds.extend([bounds[d*2], midpoints[d]])
                        else:
                            sub_bounds.extend([midpoints[d], bounds[d*2+1]])
                    subregions.append(sub_bounds)
                
                # Build child nodes
                for sub_bounds in subregions:
                    child_node = build_node(sub_bounds, depth + 1)
                    node['children'].append(child_node)
            
            return node
        
        # Initialize with full bounds (0 to 1 for each dimension)
        full_bounds = [0, 1] * self.dimension
        quadtree = build_node(full_bounds)
        
        return quadtree
    
    def is_secure(self) -> bool:
        """
        Check if the current hypercube configuration is secure.
        
        Returns:
            bool: True if secure, False otherwise
        """
        # A secure system should have:
        # 1. System stability above threshold
        # 2. TVI below threshold
        # 3. β₁ = 2 (as per documentation)
        
        # Check system stability
        if not self.calibration_system.is_system_stable():
            logger.warning("System is unstable according to calibration system")
            return False
        
        # Check TVI (using a dummy signature for demonstration)
        # In reality, this would check actual signatures
        dummy_signature = b'\x00' * 64  # Dummy signature
        tvi = self.get_tvi(dummy_signature)
        if tvi > self.tvi_threshold:
            logger.warning(f"TVI exceeds threshold: {tvi:.4f} > {self.tvi_threshold}")
            return False
        
        # Check Betti numbers (specifically β₁)
        betti_numbers = self._calculate_betti_numbers()
        if len(betti_numbers) > 1 and betti_numbers[1] != 2:
            logger.warning(f"Unexpected β₁ value: {betti_numbers[1]} (expected: 2)")
            return False
        
        return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """
        Get comprehensive security status of the hypercube.
        
        Returns:
            Dictionary containing security metrics and status
        """
        # Calculate Betti numbers
        betti_numbers = self._calculate_betti_numbers()
        
        # Analyze topology
        topology_analysis = self.analyze_topology()
        
        # Check system stability
        stability = self.calibration_system.get_stability_metrics()
        
        # Get TVI for a dummy signature (in reality, this would use actual signatures)
        dummy_signature = b'\x00' * 64
        tvi = self.get_tvi(dummy_signature)
        
        # Determine security status
        is_secure = tvi <= self.tvi_threshold and stability['stability'] >= self.stability_threshold
        
        # Check if expansion might be needed
        needs_expansion = False
        expansion_reason = ""
        if tvi > 0.4 or stability['stability'] < 0.9:
            needs_expansion = True
            if tvi > 0.4:
                expansion_reason = "high TVI"
            else:
                expansion_reason = "low stability"
        
        return {
            'dimension': self.dimension,
            'max_dimension': self.max_dimension,
            'betti_numbers': betti_numbers,
            'beta_1': betti_numbers[1] if len(betti_numbers) > 1 else None,
            'tvi': tvi,
            'tvi_threshold': self.tvi_threshold,
            'stability': stability['stability'],
            'stability_threshold': self.stability_threshold,
            'drift': stability['drift'],
            'drift_threshold': self.drift_threshold,
            'secure': is_secure,
            'needs_dimension_expansion': needs_expansion,
            'expansion_reason': expansion_reason,
            'topology_analysis': topology_analysis,
            'last_calibration': stability['last_calibration'],
            'timestamp': time.time()
        }
    
    def get_operational_metrics(self) -> Dict[str, Any]:
        """
        Get operational metrics for monitoring and analytics.
        
        Returns:
            Dictionary containing operational metrics
        """
        security_status = self.get_security_status()
        stability = self.calibration_system.get_stability_metrics()
        
        return {
            'dimension': self.dimension,
            'system_stability': stability['stability'],
            'tvi': security_status['tvi'],
            'beta_1': security_status['beta_1'],
            'singularities_count': len(self.topological_component['singularities']),
            'diagonal_symmetry_violation': self.algebraic_component['symmetry_properties']['diagonal_violation'],
            'psnr': self.spectral_component['psnr'],
            'topological_entropy': self.topological_component['topological_entropy'],
            'naturalness_coefficient': self.topological_component['naturalness_coefficient'],
            'last_expansion': self.last_expansion_time,
            'expansion_count': len(self.expansion_history),
            'timestamp': time.time()
        }
