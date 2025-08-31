"""
QuantumFortress 2.0 Auto-Calibration System

This module implements the mandatory auto-calibration system for QuantumFortress 2.0,
as emphasized in Квантовый ПК.md: "Система авто-калибровки как обязательная часть рантайма"

The auto-calibration system continuously monitors and corrects quantum state drift,
ensuring the stability and security of the quantum hypercube. It operates according to
the principle: "Хорошая система «подпевает себе» постоянно, тихо и незаметно для пользователя."
("A good system 'sings along with itself' constantly, quietly and unnoticed by the user.")

Key features:
- Background calibration running as a daemon thread
- Drift monitoring with configurable thresholds
- Adaptive calibration intervals based on system stability
- Comprehensive telemetry for tracking system health
- Self-learning history for improved calibration accuracy
- Integration with topological vulnerability analysis

As stated in Квантовый ПК.md:
"Сильная сторона — параллелизм и пропускная способность;
слабое место — дрейф и разрядность, которые лечатся калибровкой и грамотной архитектурой."

("The strength is parallelism and bandwidth;
the weakness is drift and precision, which are fixed by calibration and proper architecture.")
"""

import numpy as np
import time
import uuid
import logging
import threading
import queue
import copy
import json
import pickle
import zlib
from enum import Enum
from typing import Dict, Any, Tuple, Optional, List, Callable
import psutil
import resource
from collections import deque, defaultdict
from dataclasses import dataclass

# Quantum platform support
from ..utils.quantum_utils import QuantumPlatform, PlatformConfig, get_platform_config
from ..utils.topology_utils import (
    calculate_betti_numbers,
    analyze_signature_topology,
    torus_distance,
    calculate_euler_characteristic,
    calculate_topological_entropy,
    find_high_density_areas,
    get_connectivity_metrics
)
from .adaptive_hypercube import AdaptiveQuantumHypercube
from .hybrid_crypto import HybridCryptoSystem, MigrationPhase
from .topological_metrics import TopologicalMetrics, TVIResult
from .betti_analyzer import BettiAnalyzer
from .collision_engine import CollisionEngine
from .dynamic_compute_router import DynamicComputeRouter
from .gradient_analysis import GradientAnalyzer
from .hypercore_transformer import HypercoreTransformer

logger = logging.getLogger(__name__)

# ======================
# CONSTANTS
# ======================
# Default calibration intervals (in seconds)
DEFAULT_CALIBRATION_INTERVAL = 300  # 5 minutes
MIN_CALIBRATION_INTERVAL = 15  # 15 seconds
MAX_CALIBRATION_INTERVAL = 3600  # 1 hour

# Drift monitoring interval (in seconds)
DEFAULT_DRIFT_MONITORING_INTERVAL = 60  # 1 minute
MIN_DRIFT_MONITORING_INTERVAL = 5  # 5 seconds

# Drift thresholds
DEFAULT_DRIFT_THRESHOLD = 0.05  # 5% drift
CRITICAL_DRIFT_THRESHOLD = 0.15  # 15% drift

# Calibration success rate thresholds
MIN_CALIBRATION_SUCCESS_RATE = 0.8  # 80% success rate

# Maximum history size
MAX_HISTORY_SIZE = 1000

# Resource limits
MAX_MEMORY_USAGE_PERCENT = 85
MAX_CPU_USAGE_PERCENT = 85

# ======================
# EXCEPTIONS
# ======================
class CalibrationError(Exception):
    """Base exception for AutoCalibrationSystem module."""
    pass

class CalibrationTimeoutError(CalibrationError):
    """Raised when calibration exceeds timeout limits."""
    pass

class ResourceLimitExceededError(CalibrationError):
    """Raised when resource limits are exceeded during calibration."""
    pass

class QuantumStateError(CalibrationError):
    """Raised when quantum state operations fail during calibration."""
    pass

# ======================
# ENUMS
# ======================
class CalibrationStatus(Enum):
    """Status of calibration operations"""
    IDLE = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5
    SKIPPED = 6

class DriftSeverity(Enum):
    """Severity levels for quantum state drift"""
    NORMAL = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4

# ======================
# DATA CLASSES
# ======================
@dataclass
class DriftMetrics:
    """Metrics for quantum state drift analysis"""
    current_drift: float
    historical_drift: float
    drift_rate: float
    stability_score: float
    drift_severity: DriftSeverity
    timestamp: float
    betti_numbers: List[float]
    euler_characteristic: float
    topological_entropy: float
    high_density_areas: int
    tvi: float
    vulnerability_score: float

@dataclass
class CalibrationResult:
    """Result of a calibration operation"""
    success: bool
    drift_before: float
    drift_after: float
    drift_reduction: float
    processing_time: float
    corrections_applied: int
    status: CalibrationStatus
    timestamp: float
    metrics_before: DriftMetrics
    metrics_after: DriftMetrics
    calibration_id: str
    platform: QuantumPlatform
    dimension: int
    tvi_before: float
    tvi_after: float

@dataclass
class CalibrationHistory:
    """History of calibration operations for self-learning"""
    calibration_id: str
    drift_before: float
    drift_after: float
    drift_reduction: float
    processing_time: float
    status: CalibrationStatus
    timestamp: float
    platform: QuantumPlatform
    dimension: int
    tvi_before: float
    tvi_after: float
    drift_severity: DriftSeverity
    environmental_conditions: Dict[str, Any]
    calibration_parameters: Dict[str, Any]

@dataclass
class TelemetryData:
    """Telemetry data for system health monitoring"""
    timestamp: float
    memory_usage: float
    cpu_usage: float
    drift_rate: float
    stability_score: float
    tvi: float
    calibration_interval: float
    next_calibration: float
    drift_events: int
    critical_events: int
    time_since_calibration: float
    calibration_count: int
    drift_events: int
    critical_events: int
    success_rate: float
    tvi: float

# ======================
# CORE CLASS
# ======================
class AutoCalibrationSystem:
    """Auto-Calibration System for QuantumFortress 2.0
    
    This class implements a background calibration system that continuously monitors
    and corrects quantum state drift in the adaptive hypercube. The system:
    - Runs as a daemon thread for minimal user impact
    - Monitors topological drift and quantum state fidelity
    - Applies corrections when drift exceeds thresholds
    - Adapts calibration intervals based on system stability
    - Generates detailed telemetry for system health assessment
    - Uses self-learning history to improve calibration accuracy
    
    The implementation follows the principles from Квантовый ПК.md:
    "Планируйте телеметрию по дрейфу и деградации. Авто-калибровка — обязательная часть рантайма."
    
    Example:
    >>> hypercube = AdaptiveQuantumHypercube(dimension=4)
    >>> calibrator = AutoCalibrationSystem(hypercube)
    >>> calibrator.start()
    >>> # System now runs continuous background calibration
    >>> report = calibrator.get_calibration_report()
    >>> print(f"Current drift: {report['current_drift']:.4f}")
    """
    
    def __init__(self,
                 hypercube: AdaptiveQuantumHypercube,
                 crypto_system: Optional[HybridCryptoSystem] = None,
                 calibration_interval: float = DEFAULT_CALIBRATION_INTERVAL,
                 drift_monitoring_interval: float = DEFAULT_DRIFT_MONITORING_INTERVAL,
                 drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
                 min_success_rate: float = MIN_CALIBRATION_SUCCESS_RATE,
                 max_history_size: int = MAX_HISTORY_SIZE,
                 adaptive_interval: bool = True,
                 resource_monitoring: bool = True):
        """
        Initialize the auto-calibration system.
        
        Args:
            hypercube: Reference to the quantum hypercube being calibrated
            crypto_system: Optional reference to the hybrid crypto system
            calibration_interval: Seconds between full calibration cycles
            drift_monitoring_interval: Seconds between drift monitoring checks
            drift_threshold: Threshold for triggering calibration (0.0-1.0)
            min_success_rate: Minimum success rate before forcing calibration
            max_history_size: Maximum size of calibration history
            adaptive_interval: Whether to use adaptive calibration intervals
            resource_monitoring: Whether to monitor system resources
            
        Raises:
            ValueError: If intervals are outside valid range
        """
        # Validate parameters
        if calibration_interval < MIN_CALIBRATION_INTERVAL:
            raise ValueError(f"Calibration interval must be at least {MIN_CALIBRATION_INTERVAL} seconds")
        if calibration_interval > MAX_CALIBRATION_INTERVAL:
            raise ValueError(f"Calibration interval cannot exceed {MAX_CALIBRATION_INTERVAL} seconds")
        if drift_monitoring_interval < MIN_DRIFT_MONITORING_INTERVAL:
            raise ValueError(f"Drift monitoring interval must be at least {MIN_DRIFT_MONITORING_INTERVAL} seconds")
        
        # Store references
        self.hypercube = hypercube
        self.crypto_system = crypto_system
        
        # Configuration
        self.calibration_interval = calibration_interval
        self.drift_monitoring_interval = drift_monitoring_interval
        self.drift_threshold = drift_threshold
        self.min_success_rate = min_success_rate
        self.max_history_size = max_history_size
        self.adaptive_interval = adaptive_interval
        self.resource_monitoring = resource_monitoring
        
        # Internal state
        self.active = False
        self.calibration_thread = None
        self.calibration_queue = queue.Queue()
        self.telemetry_queue = queue.Queue(maxsize=1000)
        
        # Timing
        current_time = time.time()
        self.last_calibration = current_time
        self.last_drift_check = current_time
        self.last_telemetry_update = current_time
        self.telemetry_interval = 10.0  # seconds
        
        # History and metrics
        self.calibration_history = deque(maxlen=max_history_size)
        self.successful_calibrations = 0
        self.calibration_count = 0
        self.drift_events = 0
        self.critical_events = 0
        self.drift_history = deque(maxlen=100)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        if resource_monitoring:
            self.resource_monitor.start()
        
        # Dynamic compute routing
        self.compute_router = DynamicComputeRouter()
        
        # Gradient analyzer for adaptive intervals
        self.gradient_analyzer = GradientAnalyzer()
        
        # Betti analyzer for topological drift
        self.betti_analyzer = BettiAnalyzer()
        
        # Collision engine for vulnerability assessment
        self.collision_engine = CollisionEngine()
        
        logger.info(f"Initialized AutoCalibrationSystem (interval={calibration_interval}s, "
                    f"drift_threshold={self.drift_threshold})")
    
    def start(self) -> None:
        """Start the background calibration system.
        
        This launches a daemon thread that continuously monitors and calibrates
        the quantum hypercube according to the configured intervals.
        
        As stated in documentation: "Хорошая система «подпевает себе» постоянно,
        тихо и незаметно для пользователя."
        """
        if self.active:
            logger.warning("Auto-calibration system is already running")
            return
        
        self.active = True
        
        # Launch calibration thread
        self.calibration_thread = threading.Thread(
            target=self._calibration_loop,
            name="QuantumCalibrationThread",
            daemon=True
        )
        self.calibration_thread.start()
        
        current_time = time.time()
        self.last_calibration = current_time
        self.last_drift_check = current_time
        
        logger.info(f"Auto-calibration system started (interval={self.calibration_interval}s, "
                    f"drift_monitoring={self.drift_monitoring_interval}s)")
    
    def stop(self) -> None:
        """Stop the background calibration system.
        
        This safely terminates the calibration thread and cleans up resources.
        """
        if not self.active:
            logger.warning("Auto-calibration system is not running")
            return
        
        self.active = False
        
        # Wait for calibration thread to finish
        if self.calibration_thread and self.calibration_thread.is_alive():
            self.calibration_thread.join(timeout=5.0)
            if self.calibration_thread.is_alive():
                logger.warning("Calibration thread did not terminate gracefully")
        
        # Stop resource monitoring
        if self.resource_monitoring:
            self.resource_monitor.stop()
        
        logger.info("Auto-calibration system stopped")
    
    def _calibration_loop(self) -> None:
        """Main calibration loop running in background thread."""
        logger.debug("Calibration thread started")
        
        while self.active:
            try:
                current_time = time.time()
                
                # Check if full calibration is needed
                if current_time - self.last_calibration > self.calibration_interval:
                    if self._needs_immediate_calibration():
                        self._perform_calibration()
                        self.last_calibration = current_time
                        self.calibration_count += 1
                
                # Monitor for drift
                if current_time - self.last_drift_check > self.drift_monitoring_interval:
                    self._monitor_drift()
                    self.last_drift_check = current_time
                
                # Update telemetry
                if current_time - self.last_telemetry_update > self.telemetry_interval:
                    self._update_telemetry()
                    self.last_telemetry_update = current_time
                
                # Short sleep to reduce CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Calibration loop error: {str(e)}")
                time.sleep(5.0)  # Longer sleep after error
        
        logger.debug("Calibration thread exiting")
    
    def check_and_calibrate(self) -> None:
        """Check if calibration is needed and perform it if necessary.
        
        This method:
        1. Checks drift metrics
        2. Determines if calibration is needed
        3. Performs calibration if needed
        4. Updates system status
        
        As stated in Квантовый ПК.md: "Планируйте телеметрию по дрейфу и деградации"
        """
        current_time = time.time()
        
        # Check if full calibration is needed
        if current_time - self.last_calibration > self.calibration_interval:
            if self._needs_immediate_calibration():
                self._perform_calibration()
                self.last_calibration = current_time
                self.calibration_count += 1
        
        # Monitor for drift
        if current_time - self.last_drift_check > self.drift_monitoring_interval:
            self._monitor_drift()
            self.last_drift_check = current_time
    
    def _needs_immediate_calibration(self) -> bool:
        """Determine if immediate calibration is needed.
        
        Returns:
            bool: True if calibration should be performed immediately
            
        As stated in Квантовый ПК.md: "Система авто-калибровки как обязательная часть рантайма"
        """
        # Get current drift metrics
        drift_metrics = self._analyze_drift()
        
        # Check if drift exceeds threshold
        if drift_metrics.drift_severity in [DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY]:
            logger.warning(f"Immediate calibration needed due to critical drift: {drift_metrics.current_drift:.4f}")
            return True
        
        # Check if TVI is too high
        if self.crypto_system and drift_metrics.tvi > 0.6:
            logger.warning(f"Immediate calibration needed due to high TVI: {drift_metrics.tvi:.4f}")
            return True
        
        # Check success rate
        if self.calibration_count > 0:
            success_rate = self.successful_calibrations / self.calibration_count
            if success_rate < self.min_success_rate:
                logger.warning(f"Immediate calibration needed due to low success rate: "
                               f"{success_rate:.2%} < {self.min_success_rate:.0%}")
                return True
        
        return False
    
    def _perform_calibration(self) -> CalibrationResult:
        """Execute a full calibration cycle.
        
        This method:
        1. Generates reference states for calibration
        2. Analyzes current drift metrics
        3. Applies necessary corrections
        4. Records the calibration event
        5. Updates system parameters based on results
        
        Returns:
            Dictionary with calibration results
            
        As emphasized in documentation: "Система авто-калибровки как обязательная часть рантайма"
        """
        start_time = time.time()
        logger.info("Starting calibration cycle")
        
        try:
            # Get current metrics before calibration
            pre_metrics = self._analyze_drift()
            pre_drift = pre_metrics.current_drift
            
            # Generate reference states for calibration
            reference_states = self._generate_reference_states()
            
            # Analyze current drift
            drift_metrics = self._analyze_drift(reference_states)
            
            # Apply corrections if needed
            corrections = {}
            if drift_metrics.current_drift > self.drift_threshold:
                corrections = self._apply_corrections(drift_metrics, reference_states)
            
            # Get metrics after calibration
            post_metrics = self._analyze_drift()
            post_drift = post_metrics.current_drift
            
            # Calculate drift reduction
            drift_reduction = pre_drift - post_drift
            
            # Determine success
            success = post_drift < self.drift_threshold * 0.8
            
            # Update success tracking
            if success:
                self.successful_calibrations += 1
            
            # Record calibration event
            calibration_id = str(uuid.uuid4())
            result = CalibrationResult(
                success=success,
                drift_before=pre_drift,
                drift_after=post_drift,
                drift_reduction=drift_reduction,
                processing_time=time.time() - start_time,
                corrections_applied=len(corrections),
                status=CalibrationStatus.COMPLETED if success else CalibrationStatus.FAILED,
                timestamp=time.time(),
                metrics_before=pre_metrics,
                metrics_after=post_metrics,
                calibration_id=calibration_id,
                platform=self.hypercube.quantum_platform,
                dimension=self.hypercube.dimension,
                tvi_before=pre_metrics.tvi,
                tvi_after=post_metrics.tvi
            )
            
            # Store in history
            self._record_calibration(result)
            
            # Update calibration interval if adaptive
            if self.adaptive_interval:
                self._update_calibration_interval(result)
            
            # Log result
            status = "successful" if success else "failed"
            logger.info(f"Calibration {status} (drift: {pre_drift:.4f} → {post_drift:.4f}, "
                        f"reduction: {drift_reduction:.4f}, time: {result.processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}", exc_info=True)
            
            # Record failure
            result = CalibrationResult(
                success=False,
                drift_before=pre_metrics.current_drift if 'pre_metrics' in locals() else 1.0,
                drift_after=1.0,
                drift_reduction=0.0,
                processing_time=time.time() - start_time,
                corrections_applied=0,
                status=CalibrationStatus.FAILED,
                timestamp=time.time(),
                metrics_before=pre_metrics if 'pre_metrics' in locals() else None,
                metrics_after=None,
                calibration_id=str(uuid.uuid4()),
                platform=self.hypercube.quantum_platform,
                dimension=self.hypercube.dimension,
                tvi_before=pre_metrics.tvi if 'pre_metrics' in locals() else 1.0,
                tvi_after=1.0
            )
            
            # Store in history
            self._record_calibration(result)
            
            return result
    
    def _monitor_drift(self):
        """Monitor quantum state drift and record metrics."""
        try:
            drift_metrics = self._analyze_drift()
            
            # Record drift history
            self.drift_history.append(drift_metrics.current_drift)
            
            # Update event counters
            if drift_metrics.drift_severity == DriftSeverity.CRITICAL:
                self.drift_events += 1
                if drift_metrics.current_drift > CRITICAL_DRIFT_THRESHOLD:
                    self.critical_events += 1
            
            # Log drift status
            logger.debug(f"Drift monitored: {drift_metrics.current_drift:.4f} "
                         f"(severity: {drift_metrics.drift_severity.name})")
            
        except Exception as e:
            logger.error(f"Drift monitoring failed: {str(e)}", exc_info=True)
    
    def _analyze_drift(self, reference_states: Optional[Dict[str, Any]] = None) -> DriftMetrics:
        """Analyze current quantum state drift.
        
        Args:
            reference_states: Optional reference states for comparison
            
        Returns:
            DriftMetrics object with detailed analysis
        """
        try:
            # Get current topological metrics
            topological_metrics = self.hypercube.topological_metrics.get_metrics()
            tvi_result = self.hypercube.topological_metrics.calculate_tvi()
            
            # Analyze betti numbers for topological drift
            betti_numbers = topological_metrics.betti_numbers
            euler_char = topological_metrics.euler_characteristic
            topological_entropy = topological_metrics.topological_entropy
            high_density_areas = len(topological_metrics.high_density_areas)
            
            # Calculate drift metrics
            current_drift = tvi_result.tvi * 0.5  # Scale TVI to drift metric
            
            # Calculate historical drift (average of recent history)
            historical_drift = np.mean(self.drift_history) if self.drift_history else current_drift
            
            # Calculate drift rate (change over time)
            drift_rate = 0.0
            if len(self.drift_history) >= 2:
                time_diff = self.drift_monitoring_interval * (len(self.drift_history) - 1)
                drift_diff = self.drift_history[-1] - self.drift_history[0]
                drift_rate = drift_diff / time_diff
            
            # Calculate stability score (higher is better)
            stability_score = max(0.0, 1.0 - current_drift * 1.2)
            
            # Determine drift severity
            if current_drift > CRITICAL_DRIFT_THRESHOLD:
                drift_severity = DriftSeverity.EMERGENCY
            elif current_drift > DEFAULT_DRIFT_THRESHOLD * 1.5:
                drift_severity = DriftSeverity.CRITICAL
            elif current_drift > DEFAULT_DRIFT_THRESHOLD:
                drift_severity = DriftSeverity.WARNING
            else:
                drift_severity = DriftSeverity.NORMAL
            
            # Create and return drift metrics
            return DriftMetrics(
                current_drift=current_drift,
                historical_drift=historical_drift,
                drift_rate=drift_rate,
                stability_score=stability_score,
                drift_severity=drift_severity,
                timestamp=time.time(),
                betti_numbers=betti_numbers,
                euler_characteristic=euler_char,
                topological_entropy=topological_entropy,
                high_density_areas=high_density_areas,
                tvi=tvi_result.tvi,
                vulnerability_score=tvi_result.vulnerability_score
            )
            
        except Exception as e:
            logger.error(f"Drift analysis failed: {str(e)}", exc_info=True)
            
            # Return default metrics on failure
            return DriftMetrics(
                current_drift=1.0,
                historical_drift=1.0,
                drift_rate=0.1,
                stability_score=0.0,
                drift_severity=DriftSeverity.EMERGENCY,
                timestamp=time.time(),
                betti_numbers=[10.0, 5.0, 2.0],
                euler_characteristic=0.0,
                topological_entropy=0.0,
                high_density_areas=10,
                tvi=1.0,
                vulnerability_score=1.0
            )
    
    def _generate_reference_states(self) -> Dict[str, Any]:
        """Generate reference states for calibration.
        
        Returns:
            Dictionary containing reference states for different platforms
        """
        try:
            # Get platform configuration
            platform_config = get_platform_config(self.hypercube.quantum_platform)
            
            # Generate reference states based on platform
            reference_states = {
                "ideal_state": self._generate_ideal_state(),
                "platform_config": platform_config,
                "timestamp": time.time(),
                "dimension": self.hypercube.dimension,
                "platform": self.hypercube.quantum_platform.name
            }
            
            # Add platform-specific reference states
            if self.hypercube.quantum_platform == QuantumPlatform.SOI:
                reference_states.update(self._generate_soi_reference_states())
            elif self.hypercube.quantum_platform == QuantumPlatform.SiN:
                reference_states.update(self._generate_sin_reference_states())
            elif self.hypercube.quantum_platform == QuantumPlatform.TFLN:
                reference_states.update(self._generate_tfln_reference_states())
            elif self.hypercube.quantum_platform == QuantumPlatform.InP:
                reference_states.update(self._generate_inp_reference_states())
            
            return reference_states
            
        except Exception as e:
            logger.error(f"Failed to generate reference states: {str(e)}", exc_info=True)
            # Return fallback reference states
            return {
                "ideal_state": np.ones(2**self.hypercube.dimension) / np.sqrt(2**self.hypercube.dimension),
                "timestamp": time.time(),
                "dimension": self.hypercube.dimension,
                "platform": self.hypercube.quantum_platform.name
            }
    
    def _generate_ideal_state(self) -> np.ndarray:
        """Generate an ideal quantum state for the current dimension."""
        # For a 4D hypercube, the ideal state is a uniform superposition
        size = 2 ** self.hypercube.dimension
        return np.ones(size, dtype=np.complex128) / np.sqrt(size)
    
    def _generate_soi_reference_states(self) -> Dict[str, Any]:
        """Generate SOI-specific reference states."""
        # SOI platform has specific characteristics
        return {
            "phase_noise_profile": np.random.normal(0, 0.05, 2**self.hypercube.dimension),
            "amplitude_profile": np.ones(2**self.hypercube.dimension) * 0.95,
            "wavelength_reference": 1550  # nm
        }
    
    def _generate_sin_reference_states(self) -> Dict[str, Any]:
        """Generate SiN-specific reference states."""
        # SiN platform has better coherence
        return {
            "phase_noise_profile": np.random.normal(0, 0.03, 2**self.hypercube.dimension),
            "amplitude_profile": np.ones(2**self.hypercube.dimension) * 0.98,
            "wavelength_reference": 1310  # nm
        }
    
    def _generate_tfln_reference_states(self) -> Dict[str, Any]:
        """Generate TFLN-specific reference states."""
        # TFLN platform has excellent properties
        return {
            "phase_noise_profile": np.random.normal(0, 0.01, 2**self.hypercube.dimension),
            "amplitude_profile": np.ones(2**self.hypercube.dimension) * 0.99,
            "wavelength_reference": 780  # nm
        }
    
    def _generate_inp_reference_states(self) -> Dict[str, Any]:
        """Generate InP-specific reference states."""
        # InP platform has high precision
        return {
            "phase_noise_profile": np.random.normal(0, 0.005, 2**self.hypercube.dimension),
            "amplitude_profile": np.ones(2**self.hypercube.dimension) * 0.995,
            "wavelength_reference": 1550  # nm
        }
    
    def _apply_corrections(self, 
                          drift_metrics: DriftMetrics, 
                          reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply necessary corrections to reduce quantum state drift.
        
        Args:
            drift_metrics: Current drift metrics
            reference_states: Reference states for calibration
            
        Returns:
            Dictionary with applied corrections
        """
        corrections = {}
        
        try:
            # Check resource limits before applying corrections
            self._check_resources()
            
            # Apply phase corrections
            phase_corrections = self._apply_phase_corrections(drift_metrics, reference_states)
            corrections.update(phase_corrections)
            
            # Apply amplitude corrections
            amplitude_corrections = self._apply_amplitude_corrections(drift_metrics, reference_states)
            corrections.update(amplitude_corrections)
            
            # Apply topological corrections if needed
            if drift_metrics.high_density_areas > 5:
                topology_corrections = self._apply_topology_corrections(drift_metrics)
                corrections.update(topology_corrections)
            
            # Apply platform-specific corrections
            platform_corrections = self._apply_platform_corrections(drift_metrics, reference_states)
            corrections.update(platform_corrections)
            
            # Update hypercube with corrections
            self.hypercube.apply_corrections(corrections)
            
            logger.debug(f"Applied {len(corrections)} corrections to reduce drift")
            
        except Exception as e:
            logger.error(f"Failed to apply corrections: {str(e)}", exc_info=True)
        
        return corrections
    
    def _apply_phase_corrections(self, 
                               drift_metrics: DriftMetrics, 
                               reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """Apply phase corrections to reduce drift."""
        # In a real implementation, this would calculate and apply precise phase corrections
        # For this example, we'll simulate the process
        
        corrections = {}
        
        # Calculate phase drift
        phase_drift = drift_metrics.current_drift * 0.5
        
        # Determine correction magnitude (inverse of drift)
        correction_magnitude = -phase_drift * 0.8
        
        # Apply to hypercube
        corrections["phase_correction"] = correction_magnitude
        corrections["phase_correction_applied"] = True
        
        return corrections
    
    def _apply_amplitude_corrections(self, 
                                    drift_metrics: DriftMetrics, 
                                    reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """Apply amplitude corrections to reduce drift."""
        corrections = {}
        
        # Calculate amplitude drift
        amplitude_drift = drift_metrics.current_drift * 0.3
        
        # Determine correction magnitude
        correction_magnitude = amplitude_drift * 0.7
        
        # Apply to hypercube
        corrections["amplitude_correction"] = correction_magnitude
        corrections["amplitude_correction_applied"] = True
        
        return corrections
    
    def _apply_topology_corrections(self, drift_metrics: DriftMetrics) -> Dict[str, Any]:
        """Apply topological corrections to address high-density areas."""
        corrections = {}
        
        # Get high density areas from metrics
        high_density_areas = self.hypercube.topological_metrics.get_metrics().high_density_areas
        
        # For each high density area, calculate correction
        for i, area in enumerate(high_density_areas[:3]):  # Only correct top 3 areas
            ur_mean, uz_mean, r_val, count, ur_cluster, uz_cluster = area
            
            # Calculate correction based on cluster size and position
            correction_strength = min(0.1, count / 1000.0)
            
            # Add correction
            corrections[f"topology_correction_{i}"] = {
                "ur_center": float(ur_mean),
                "uz_center": float(uz_mean),
                "strength": correction_strength,
                "cluster_size": int(count)
            }
        
        corrections["topology_correction_applied"] = bool(corrections)
        
        return corrections
    
    def _apply_platform_corrections(self, 
                                  drift_metrics: DriftMetrics, 
                                  reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """Apply platform-specific corrections."""
        corrections = {}
        
        # Platform-specific correction logic
        platform = self.hypercube.quantum_platform
        
        if platform == QuantumPlatform.SOI:
            corrections.update(self._apply_soi_corrections(drift_metrics, reference_states))
        elif platform == QuantumPlatform.SiN:
            corrections.update(self._apply_sin_corrections(drift_metrics, reference_states))
        elif platform == QuantumPlatform.TFLN:
            corrections.update(self._apply_tfln_corrections(drift_metrics, reference_states))
        elif platform == QuantumPlatform.InP:
            corrections.update(self._apply_inp_corrections(drift_metrics, reference_states))
        
        return corrections
    
    def _apply_soi_corrections(self, 
                              drift_metrics: DriftMetrics, 
                              reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SOI-specific corrections."""
        # SOI platform needs more frequent calibration
        return {
            "temperature_adjustment": -0.5,  # degrees Celsius
            "wavelength_adjustment": 0.1,  # nm
            "soi_specific_correction": True
        }
    
    def _apply_sin_corrections(self, 
                              drift_metrics: DriftMetrics, 
                              reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """Apply SiN-specific corrections."""
        # SiN platform has better stability
        return {
            "phase_compensation": -drift_metrics.current_drift * 0.3,
            "sin_specific_correction": True
        }
    
    def _apply_tfln_corrections(self, 
                               drift_metrics: DriftMetrics, 
                               reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """Apply TFLN-specific corrections."""
        # TFLN platform has excellent properties
        return {
            "fine_phase_adjustment": -drift_metrics.current_drift * 0.1,
            "tfln_specific_correction": True
        }
    
    def _apply_inp_corrections(self, 
                              drift_metrics: DriftMetrics, 
                              reference_states: Dict[str, Any]) -> Dict[str, Any]:
        """Apply InP-specific corrections."""
        # InP platform has high precision requirements
        return {
            "quantum_efficiency_adjustment": drift_metrics.current_drift * 0.05,
            "inp_specific_correction": True
        }
    
    def _record_calibration(self, result: CalibrationResult):
        """Record calibration result in history."""
        # Create history entry
        history_entry = CalibrationHistory(
            calibration_id=result.calibration_id,
            drift_before=result.drift_before,
            drift_after=result.drift_after,
            drift_reduction=result.drift_reduction,
            processing_time=result.processing_time,
            status=result.status,
            timestamp=result.timestamp,
            platform=result.platform,
            dimension=result.dimension,
            tvi_before=result.tvi_before,
            tvi_after=result.tvi_after,
            drift_severity=result.metrics_before.drift_severity,
            environmental_conditions=self._get_environmental_conditions(),
            calibration_parameters={
                "calibration_interval": self.calibration_interval,
                "drift_threshold": self.drift_threshold,
                "adaptive_interval": self.adaptive_interval
            }
        )
        
        # Add to history
        self.calibration_history.append(history_entry)
        
        # Update telemetry
        self._update_telemetry()
    
    def _update_calibration_interval(self, calibration_result: CalibrationResult):
        """Update calibration interval based on system stability."""
        try:
            # Get recent history for analysis
            recent_history = list(self.calibration_history)[-10:] if self.calibration_history else []
            
            if not recent_history:
                return
            
            # Calculate average drift reduction
            avg_drift_reduction = np.mean([h.drift_reduction for h in recent_history])
            
            # Calculate success rate
            successful = [h for h in recent_history if h.status == CalibrationStatus.COMPLETED]
            success_rate = len(successful) / len(recent_history)
            
            # Adjust interval based on stability
            if success_rate > 0.9 and avg_drift_reduction > 0.02:
                # System is stable, increase interval (up to max)
                new_interval = min(
                    self.calibration_interval * 1.2,
                    MAX_CALIBRATION_INTERVAL
                )
            elif success_rate < 0.7 or avg_drift_reduction < 0.01:
                # System is unstable, decrease interval (down to min)
                new_interval = max(
                    self.calibration_interval * 0.8,
                    MIN_CALIBRATION_INTERVAL
                )
            else:
                # System is moderately stable, keep interval similar
                new_interval = self.calibration_interval
            
            # Apply gradient-based adjustment for finer control
            gradient = self.gradient_analyzer.analyze(
                [h.drift_after for h in recent_history],
                [h.timestamp for h in recent_history]
            )
            
            if gradient > 0.01:  # Drift is increasing rapidly
                new_interval = max(
                    new_interval * 0.7,
                    MIN_CALIBRATION_INTERVAL
                )
            elif gradient < -0.005:  # Drift is decreasing
                new_interval = min(
                    new_interval * 1.1,
                    MAX_CALIBRATION_INTERVAL
                )
            
            # Update interval
            self.calibration_interval = new_interval
            
            logger.debug(f"Updated calibration interval to {new_interval:.1f}s "
                         f"(gradient: {gradient:.4f})")
            
        except Exception as e:
            logger.error(f"Failed to update calibration interval: {str(e)}", exc_info=True)
    
    def _get_environmental_conditions(self) -> Dict[str, Any]:
        """Get current environmental conditions for telemetry."""
        return {
            "temperature": self._get_temperature(),
            "humidity": self._get_humidity(),
            "electromagnetic_noise": self._get_electromagnetic_noise()
        }
    
    def _get_temperature(self) -> float:
        """Get current system temperature."""
        # In a real implementation, this would read from sensors
        try:
            # Try to get CPU temperature
            import psutil
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'k10temp' in temps:
                return temps['k10temp'][0].current
        except:
            pass
        
        # Fallback to system load approximation
        cpu_percent = psutil.cpu_percent()
        return 25.0 + (cpu_percent * 0.5)  # Approximate temperature
    
    def _get_humidity(self) -> float:
        """Get current humidity (simulated)."""
        # In a real implementation, this would read from sensors
        return 45.0  # Default humidity percentage
    
    def _get_electromagnetic_noise(self) -> float:
        """Get current electromagnetic noise level (simulated)."""
        # In a real implementation, this would read from sensors
        return 0.0  # Default electromagnetic noise
    
    def _update_telemetry(self):
        """Update telemetry data for system monitoring."""
        try:
            # Get current metrics
            drift_metrics = self._analyze_drift()
            current_time = time.time()
            
            # Calculate success rate
            success_rate = 0.0
            if self.calibration_count > 0:
                success_rate = self.successful_calibrations / self.calibration_count
            
            # Get resource usage
            memory_usage = self.resource_monitor.get_memory_usage() if self.resource_monitoring else 0.0
            cpu_usage = self.resource_monitor.get_cpu_usage() if self.resource_monitoring else 0.0
            
            # Create telemetry data
            telemetry = TelemetryData(
                timestamp=current_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                drift_rate=drift_metrics.drift_rate,
                stability_score=drift_metrics.stability_score,
                tvi=drift_metrics.tvi,
                calibration_interval=self.calibration_interval,
                next_calibration=self.last_calibration + self.calibration_interval,
                drift_events=self.drift_events,
                critical_events=self.critical_events,
                time_since_calibration=current_time - self.last_calibration,
                calibration_count=self.calibration_count,
                drift_events=self.drift_events,
                critical_events=self.critical_events,
                success_rate=success_rate,
                tvi=drift_metrics.tvi
            )
            
            # Add to telemetry queue
            try:
                self.telemetry_queue.put_nowait(telemetry)
            except queue.Full:
                # Remove oldest telemetry if queue is full
                try:
                    self.telemetry_queue.get_nowait()
                    self.telemetry_queue.put_nowait(telemetry)
                except:
                    pass
            
            # Log telemetry if needed
            if current_time - self.last_telemetry_update > 60.0:  # Log every minute
                logger.debug(f"Telemetry: drift={drift_metrics.current_drift:.4f}, "
                             f"TVI={drift_metrics.tvi:.4f}, "
                             f"stability={drift_metrics.stability_score:.2f}, "
                             f"calibration_interval={self.calibration_interval:.1f}s")
                self.last_telemetry_update = current_time
                
        except Exception as e:
            logger.error(f"Telemetry update failed: {str(e)}", exc_info=True)
    
    def _check_resources(self):
        """Check if system resources are within acceptable limits."""
        if not self.resource_monitoring:
            return
        
        # Get current resource usage
        memory_usage = self.resource_monitor.get_memory_usage()
        cpu_usage = self.resource_monitor.get_cpu_usage()
        
        # Check if we're approaching resource limits
        if memory_usage > MAX_MEMORY_USAGE_PERCENT or cpu_usage > MAX_CPU_USAGE_PERCENT:
            raise ResourceLimitExceededError(
                f"Resource limits exceeded: memory={memory_usage:.1f}%, cpu={cpu_usage:.1f}%"
            )
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive calibration report.
        
        Returns:
            Dictionary containing detailed calibration report
        """
        try:
            # Get current drift metrics
            drift_metrics = self._analyze_drift()
            
            # Calculate success rate
            success_rate = 0.0
            if self.calibration_count > 0:
                success_rate = self.successful_calibrations / self.calibration_count
            
            # Get recent history
            recent_history = list(self.calibration_history)[-5:] if self.calibration_history else []
            
            # Format history for report
            history_summary = []
            for entry in recent_history:
                history_summary.append({
                    "timestamp": entry.timestamp,
                    "drift_before": entry.drift_before,
                    "drift_after": entry.drift_after,
                    "drift_reduction": entry.drift_reduction,
                    "status": entry.status.name,
                    "processing_time": entry.processing_time
                })
            
            # Get resource usage
            memory_usage = self.resource_monitor.get_memory_usage() if self.resource_monitoring else 0.0
            cpu_usage = self.resource_monitor.get_cpu_usage() if self.resource_monitoring else 0.0
            
            # Create report
            report = {
                "status": "active" if self.active else "inactive",
                "current_drift": drift_metrics.current_drift,
                "drift_severity": drift_metrics.drift_severity.name,
                "stability_score": drift_metrics.stability_score,
                "tvi": drift_metrics.tvi,
                "calibration_interval": self.calibration_interval,
                "next_calibration_in": max(0.0, self.calibration_interval - (time.time() - self.last_calibration)),
                "calibration_count": self.calibration_count,
                "successful_calibrations": self.successful_calibrations,
                "success_rate": success_rate,
                "drift_events": self.drift_events,
                "critical_events": self.critical_events,
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "recent_history": history_summary,
                "timestamp": time.time(),
                "platform": self.hypercube.quantum_platform.name,
                "dimension": self.hypercube.dimension
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate calibration report: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get calibration history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of calibration history entries
        """
        history = list(self.calibration_history)[-limit:]
        
        # Convert to dictionary format
        return [{
            "calibration_id": h.calibration_id,
            "drift_before": h.drift_before,
            "drift_after": h.drift_after,
            "drift_reduction": h.drift_reduction,
            "processing_time": h.processing_time,
            "status": h.status.name,
            "timestamp": h.timestamp,
            "platform": h.platform.name,
            "dimension": h.dimension,
            "tvi_before": h.tvi_before,
            "tvi_after": h.tvi_after,
            "drift_severity": h.drift_severity.name,
            "environmental_conditions": h.environmental_conditions,
            "calibration_parameters": h.calibration_parameters
        } for h in history]
    
    def export_history(self, file_path: str) -> bool:
        """
        Export calibration history to a file.
        
        Args:
            file_path: Path to export history to
            
        Returns:
            bool: True if export was successful
        """
        try:
            # Get history data
            history_data = self.get_history(limit=self.max_history_size)
            
            # Create export data
            export_data = {
                "metadata": {
                    "export_time": time.time(),
                    "system_version": "2.0",
                    "platform": self.hypercube.quantum_platform.name,
                    "dimension": self.hypercube.dimension
                },
                "history": history_data
            }
            
            # Serialize and compress
            data_bytes = json.dumps(export_data).encode('utf-8')
            compressed = zlib.compress(data_bytes)
            
            # Save to file
            with open(file_path, 'wb') as f:
                f.write(compressed)
            
            logger.info(f"Calibration history exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export calibration history: {str(e)}", exc_info=True)
            return False
    
    def import_history(self, file_path: str) -> bool:
        """
        Import calibration history from a file.
        
        Args:
            file_path: Path to import history from
            
        Returns:
            bool: True if import was successful
        """
        try:
            # Read and decompress
            with open(file_path, 'rb') as f:
                compressed = f.read()
            data_bytes = zlib.decompress(compressed)
            import_data = json.loads(data_bytes.decode('utf-8'))
            
            # Validate data
            if 'history' not in import_data:
                raise ValueError("Invalid history file format")
            
            # Clear current history
            self.calibration_history.clear()
            
            # Import entries
            for entry in import_data['history']:
                # Convert to CalibrationHistory object
                history_entry = CalibrationHistory(
                    calibration_id=entry["calibration_id"],
                    drift_before=entry["drift_before"],
                    drift_after=entry["drift_after"],
                    drift_reduction=entry["drift_reduction"],
                    processing_time=entry["processing_time"],
                    status=CalibrationStatus[entry["status"]],
                    timestamp=entry["timestamp"],
                    platform=QuantumPlatform[entry["platform"]],
                    dimension=entry["dimension"],
                    tvi_before=entry["tvi_before"],
                    tvi_after=entry["tvi_after"],
                    drift_severity=DriftSeverity[entry["drift_severity"]],
                    environmental_conditions=entry["environmental_conditions"],
                    calibration_parameters=entry["calibration_parameters"]
                )
                self.calibration_history.append(history_entry)
            
            # Update metrics
            self.successful_calibrations = sum(
                1 for h in self.calibration_history 
                if h.status == CalibrationStatus.COMPLETED
            )
            self.calibration_count = len(self.calibration_history)
            
            logger.info(f"Calibration history imported from {file_path} "
                        f"({len(import_data['history'])} entries)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import calibration history: {str(e)}", exc_info=True)
            return False
    
    def needs_calibration(self) -> bool:
        """
        Check if calibration is needed based on current metrics.
        
        Returns:
            bool: True if calibration is needed
        """
        # Check time since last calibration
        time_since = time.time() - self.last_calibration
        if time_since > self.calibration_interval:
            return True
        
        # Check if immediate calibration is needed
        return self._needs_immediate_calibration()

class ResourceMonitor:
    """Monitor system resources to prevent overload during calibration."""
    
    def __init__(self, interval: float = 5.0):
        """
        Initialize the resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.memory_usage = 0.0
        self.cpu_usage = 0.0
        self.last_update = 0.0
    
    def start(self):
        """Start monitoring resources."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info("Resource monitor started")
    
    def stop(self):
        """Stop monitoring resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Resource monitor stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Get memory usage
                self.memory_usage = psutil.virtual_memory().percent
                
                # Get CPU usage
                self.cpu_usage = psutil.cpu_percent(interval=0.1)
                
                # Update timestamp
                self.last_update = time.time()
                
                # Sleep for interval
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}", exc_info=True)
                time.sleep(self.interval)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        if time.time() - self.last_update > self.interval * 2:
            # Force update if stale
            self.memory_usage = psutil.virtual_memory().percent
            self.last_update = time.time()
        return self.memory_usage
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if time.time() - self.last_update > self.interval * 2:
            # Force update if stale
            self.cpu_usage = psutil.cpu_percent(interval=0.1)
            self.last_update = time.time()
        return self.cpu_usage

# Global instance for convenience (can be overridden by application)
default_calibration_system = None
