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
- Automatic correction of quantum state anomalies
- Integration with topological vulnerability analysis

As stated in the documentation: "Сильная сторона — параллелизм и пропускная способность; 
слабое место — дрейф и разрядность, которые лечатся калибровкой и грамотной архитектурой."
("The strength is parallelism and bandwidth; the weakness is drift and precision, 
which are fixed by calibration and proper architecture.")
"""

import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

from quantum_fortress.core.adaptive_hypercube import AdaptiveQuantumHypercube
from quantum_fortress.topology.homology import HomologyAnalyzer
from quantum_fortress.utils.topology_utils import calculate_topological_deviation

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CALIBRATION_INTERVAL = 60  # Seconds between calibration cycles
MIN_CALIBRATION_INTERVAL = 15      # Minimum allowed interval
MAX_CALIBRATION_INTERVAL = 300     # Maximum allowed interval
DRIFT_THRESHOLD = 0.05             # Maximum acceptable drift before correction
CRITICAL_DRIFT_THRESHOLD = 0.15    # Drift level triggering critical alerts
DEFAULT_DRIFT_MONITORING_INTERVAL = 10  # Seconds between drift checks
MAX_HISTORY_LENGTH = 1000          # Maximum number of history entries to retain
ADAPTIVE_CALIBRATION_FACTOR = 0.5  # Factor for adaptive interval adjustment


@dataclass
class DriftMetrics:
    """Metrics for tracking quantum state drift"""
    current_drift: float
    drift_threshold: float
    max_drift: float
    avg_drift: float
    error_distribution: List[float]
    timestamp: float


@dataclass
class CalibrationEvent:
    """Record of a calibration event"""
    timestamp: float
    drift_before: float
    drift_after: float
    corrections_applied: Dict[str, Any]
    duration: float
    success: bool


@dataclass
class SystemStatus:
    """Current status of the quantum system"""
    status: str  # "stable", "warning", "critical", "inactive"
    drift: float
    drift_threshold: float
    calibration_interval: float
    last_calibration: float
    time_since_calibration: float
    calibration_count: int
    drift_events: int
    critical_events: int


class AutoCalibrationSystem:
    """
    Auto-Calibration System for QuantumFortress 2.0
    
    This class implements a background calibration system that continuously monitors
    and corrects quantum state drift in the adaptive hypercube. The system:
    - Runs as a daemon thread for minimal user impact
    - Monitors topological drift and quantum state fidelity
    - Applies corrections when drift exceeds thresholds
    - Adapts calibration intervals based on system stability
    - Generates detailed telemetry for system health assessment
    
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
    
    def __init__(self, hypercube: AdaptiveQuantumHypercube, 
                 calibration_interval: float = DEFAULT_CALIBRATION_INTERVAL,
                 drift_monitoring_interval: float = DEFAULT_DRIFT_MONITORING_INTERVAL):
        """
        Initialize the auto-calibration system.
        
        Args:
            hypercube: Reference to the quantum hypercube being calibrated
            calibration_interval: Seconds between full calibration cycles
            drift_monitoring_interval: Seconds between drift monitoring checks
            
        Raises:
            ValueError: If intervals are outside valid range
        """
        # Validate parameters
        if calibration_interval < MIN_CALIBRATION_INTERVAL:
            raise ValueError(f"Calibration interval must be at least {MIN_CALIBRATION_INTERVAL} seconds")
        if calibration_interval > MAX_CALIBRATION_INTERVAL:
            raise ValueError(f"Calibration interval must be at most {MAX_CALIBRATION_INTERVAL} seconds")
        if drift_monitoring_interval <= 0:
            raise ValueError("Drift monitoring interval must be positive")
            
        # Store references
        self.hypercube = hypercube
        self.calibration_interval = calibration_interval
        self.drift_monitoring_interval = drift_monitoring_interval
        
        # Initialize state
        self.active = False
        self.calibration_thread = None
        self.last_calibration = 0.0
        self.last_drift_check = 0.0
        self.calibration_count = 0
        self.drift_events = 0
        self.critical_events = 0
        
        # Initialize history
        self.drift_history: List[DriftMetrics] = []
        self.correction_history: List[CalibrationEvent] = []
        
        # Configure thresholds
        self.drift_threshold = DRIFT_THRESHOLD
        self.critical_drift_threshold = CRITICAL_DRIFT_THRESHOLD
        
        logger.info(
            f"Initialized AutoCalibrationSystem (interval={calibration_interval}s, "
            f"drift_threshold={self.drift_threshold})"
        )
    
    def start(self) -> None:
        """
        Start the background calibration system.
        
        This launches a daemon thread that continuously monitors and calibrates
        the quantum hypercube according to the configured intervals.
        
        As stated in documentation: "Хорошая система «подпевает себе» постоянно, 
        тихо и незаметно для пользователя."
        """
        if self.active:
            logger.warning("Auto-calibration system is already running")
            return
            
        self.active = True
        self.calibration_thread = threading.Thread(
            target=self._calibration_loop,
            name="QuantumCalibrationThread"
        )
        self.calibration_thread.daemon = True
        self.calibration_thread.start()
        
        current_time = time.time()
        self.last_calibration = current_time
        self.last_drift_check = current_time
        
        logger.info(
            f"Auto-calibration system started (interval={self.calibration_interval}s, "
            f"drift_monitoring={self.drift_monitoring_interval}s)"
        )
    
    def stop(self) -> None:
        """
        Stop the background calibration system.
        
        This safely terminates the calibration thread and cleans up resources.
        """
        if not self.active:
            logger.warning("Auto-calibration system is not running")
            return
            
        self.active = False
        if self.calibration_thread:
            self.calibration_thread.join(timeout=1.0)
            self.calibration_thread = None
            
        logger.info("Auto-calibration system stopped")
    
    def _calibration_loop(self) -> None:
        """
        Main loop for background calibration.
        
        This method runs in a separate thread and:
        1. Checks if full calibration is needed
        2. Monitors for parameter drift
        3. Applies corrections when necessary
        4. Maintains telemetry history
        """
        logger.debug("Calibration thread started")
        
        while self.active:
            current_time = time.time()
            
            # Check if full calibration is needed
            if current_time - self.last_calibration > self.calibration_interval:
                self.run_calibration()
                self.last_calibration = current_time
                self.calibration_count += 1
            
            # Monitor for drift
            if current_time - self.last_drift_check > self.drift_monitoring_interval:
                self.monitor_drift()
                self.last_drift_check = current_time
            
            # Short sleep to reduce CPU usage
            time.sleep(1.0)
        
        logger.debug("Calibration thread exiting")
    
    def run_calibration(self) -> Dict[str, Any]:
        """
        Execute a full calibration cycle.
        
        This method:
        1. Generates reference states for calibration
        2. Analyzes current drift metrics
        3. Applies necessary corrections
        4. Records the calibration event
        
        Returns:
            Dictionary with calibration results
            
        As emphasized in documentation: "Система авто-калибровки как обязательная часть рантайма"
        """
        start_time = time.time()
        logger.info("Starting calibration cycle")
        
        try:
            # Generate reference states for calibration
            reference_states = self._generate_reference_states()
            
            # Analyze current drift
            drift_metrics = self._analyze_drift(reference_states)
            
            # Apply corrections if needed
            corrections = {}
            if drift_metrics.current_drift > self.drift_threshold:
                corrections = self._apply_corrections(drift_metrics)
                logger.info(
                    f"Applied corrections (drift reduced from {drift_metrics.current_drift:.4f} "
                    f"to {corrections.get('new_drift', drift_metrics.current_drift):.4f})"
                )
            
            # Record calibration event
            duration = time.time() - start_time
            success = drift_metrics.current_drift <= self.drift_threshold
            self._record_calibration_event(
                drift_metrics.current_drift,
                corrections.get('new_drift', drift_metrics.current_drift) if corrections else drift_metrics.current_drift,
                corrections,
                duration,
                success
            )
            
            # Adjust calibration interval based on stability
            self._adjust_calibration_interval(drift_metrics.current_drift)
            
            result = {
                "success": success,
                "drift_before": drift_metrics.current_drift,
                "drift_after": corrections.get('new_drift', drift_metrics.current_drift) if corrections else drift_metrics.current_drift,
                "corrections_applied": bool(corrections),
                "duration": duration,
                "timestamp": time.time()
            }
            
            logger.info(
                f"Calibration completed (success={success}, duration={duration:.2f}s, "
                f"drift={result['drift_after']:.4f})"
            )
            return result
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _generate_reference_states(self) -> List[np.ndarray]:
        """
        Generate reference quantum states for calibration.
        
        Returns:
            List of reference quantum states
        """
        # In a real implementation, this would generate specific reference states
        # For demonstration, we'll create some standard states
        states = []
        dimension = self.hypercube.dimension
        
        # Generate uniform superposition state
        size = 2 ** dimension
        uniform_state = np.ones(size, dtype=np.complex128) / np.sqrt(size)
        states.append(uniform_state)
        
        # Generate computational basis states
        for i in range(min(5, dimension)):
            basis_state = np.zeros(size, dtype=np.complex128)
            basis_state[1 << i] = 1.0
            states.append(basis_state)
        
        return states
    
    def _analyze_drift(self, reference_states: List[np.ndarray]) -> DriftMetrics:
        """
        Analyze current drift against reference states.
        
        Args:
            reference_states: List of reference quantum states
            
        Returns:
            DriftMetrics object with current drift analysis
        """
        current_time = time.time()
        deviations = []
        
        for state in reference_states:
            # In a real implementation, we would compare current state to reference
            # For demonstration, we'll simulate drift based on time since last calibration
            time_factor = (current_time - self.last_calibration) / self.calibration_interval
            # Simulate drift (in reality, this would be measured)
            deviation = min(0.2, 0.01 + time_factor * 0.05)
            deviations.append(deviation)
        
        # Calculate metrics
        current_drift = max(deviations) if deviations else 0.0
        max_drift = max(deviations) if deviations else 0.0
        avg_drift = sum(deviations) / len(deviations) if deviations else 0.0
        
        # Record drift history
        drift_metrics = DriftMetrics(
            current_drift=current_drift,
            drift_threshold=self.drift_threshold,
            max_drift=max_drift,
            avg_drift=avg_drift,
            error_distribution=deviations,
            timestamp=current_time
        )
        self._record_drift_metrics(drift_metrics)
        
        return drift_metrics
    
    def monitor_drift(self) -> DriftMetrics:
        """
        Monitor current drift without performing full calibration.
        
        This is a lightweight check that runs more frequently than full calibration.
        
        Returns:
            Current drift metrics
            
        As stated in documentation: "Планируйте телеметрию по дрейфу и деградации."
        """
        reference_states = self._generate_reference_states()
        drift_metrics = self._analyze_drift(reference_states)
        
        # Check for critical drift
        if drift_metrics.current_drift > self.critical_drift_threshold:
            self.critical_events += 1
            logger.critical(
                f"CRITICAL DRIFT DETECTED: {drift_metrics.current_drift:.4f} > "
                f"{self.critical_drift_threshold:.4f}"
            )
        elif drift_metrics.current_drift > self.drift_threshold:
            self.drift_events += 1
            logger.warning(
                f"DRIFT EXCEEDS THRESHOLD: {drift_metrics.current_drift:.4f} > "
                f"{self.drift_threshold:.4f}"
            )
        
        return drift_metrics
    
    def _apply_corrections(self, drift_metrics: DriftMetrics) -> Dict[str, Any]:
        """
        Apply corrections to reduce quantum state drift.
        
        Args:
            drift_metrics: Current drift metrics
            
        Returns:
            Dictionary with correction details
        """
        start_time = time.time()
        
        try:
            # Get current quantum state
            quantum_state = self.hypercube.get_current_state()
            
            # In a real implementation, this would apply specific corrections
            # For demonstration, we'll simulate correction by renormalizing
            norm = np.linalg.norm(quantum_state.state_vector)
            if abs(norm - 1.0) > 1e-10:
                corrected_state = quantum_state.state_vector / norm
                new_drift = max(0.0, drift_metrics.current_drift - 0.02)
            else:
                corrected_state = quantum_state.state_vector
                new_drift = drift_metrics.current_drift
            
            # Update hypercube state with corrections
            # In a real implementation, this would apply the corrected state
            # Here we're just simulating the effect
            
            duration = time.time() - start_time
            logger.debug(f"Applied drift corrections (duration={duration:.4f}s)")
            
            return {
                "new_drift": new_drift,
                "correction_magnitude": drift_metrics.current_drift - new_drift,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Failed to apply corrections: {str(e)}")
            return {}
    
    def _record_drift_metrics(self, drift_metrics: DriftMetrics) -> None:
        """
        Record drift metrics in history.
        
        Args:
            drift_metrics: Drift metrics to record
        """
        # Add to history
        self.drift_history.append(drift_metrics)
        
        # Trim history if too long
        if len(self.drift_history) > MAX_HISTORY_LENGTH:
            self.drift_history = self.drift_history[-MAX_HISTORY_LENGTH:]
    
    def _record_calibration_event(self, drift_before: float, drift_after: float, 
                                 corrections: Dict[str, Any], duration: float, 
                                 success: bool) -> None:
        """
        Record a calibration event in history.
        
        Args:
            drift_before: Drift before calibration
            drift_after: Drift after calibration
            corrections: Corrections applied
            duration: Duration of calibration
            success: Whether calibration was successful
        """
        event = CalibrationEvent(
            timestamp=time.time(),
            drift_before=drift_before,
            drift_after=drift_after,
            corrections_applied=corrections,
            duration=duration,
            success=success
        )
        self.correction_history.append(event)
        
        # Trim history if too long
        if len(self.correction_history) > MAX_HISTORY_LENGTH:
            self.correction_history = self.correction_history[-MAX_HISTORY_LENGTH:]
    
    def _adjust_calibration_interval(self, current_drift: float) -> None:
        """
        Adjust calibration interval based on current system stability.
        
        Args:
            current_drift: Current drift measurement
            
        As stated in documentation: "Адаптивный интервал калибровки в зависимости от платформы"
        """
        # If drift is low, we can increase interval (less frequent calibration)
        if current_drift < self.drift_threshold * 0.5:
            new_interval = min(
                MAX_CALIBRATION_INTERVAL,
                self.calibration_interval * (1.0 + ADAPTIVE_CALIBRATION_FACTOR)
            )
        # If drift is high but below threshold, keep current interval
        elif current_drift < self.drift_threshold:
            new_interval = self.calibration_interval
        # If drift is above threshold, decrease interval (more frequent calibration)
        else:
            new_interval = max(
                MIN_CALIBRATION_INTERVAL,
                self.calibration_interval * (1.0 - ADAPTIVE_CALIBRATION_FACTOR)
            )
        
        # Only update if change is significant
        if abs(new_interval - self.calibration_interval) > 5.0:
            logger.info(
                f"Adjusting calibration interval: {self.calibration_interval:.1f}s → {new_interval:.1f}s "
                f"(drift={current_drift:.4f})"
            )
            self.calibration_interval = new_interval
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive calibration report.
        
        Returns:
            Dictionary with detailed calibration metrics and history
            
        As emphasized in documentation: "Создать подробную документацию: Особенно по настройке и калибровке."
        """
        current_time = time.time()
        
        # Get latest drift metrics
        latest_drift = self.drift_history[-1] if self.drift_history else None
        
        # Calculate system status
        status = self.get_system_status()
        
        # Prepare drift history summary
        drift_values = [m.current_drift for m in self.drift_history] if self.drift_history else []
        drift_summary = {
            "min": min(drift_values) if drift_values else 0.0,
            "max": max(drift_values) if drift_values else 0.0,
            "avg": sum(drift_values) / len(drift_values) if drift_values else 0.0,
            "count": len(drift_values)
        }
        
        # Prepare correction history summary
        correction_count = len(self.correction_history)
        successful_corrections = sum(1 for e in self.correction_history if e.success) if self.correction_history else 0
        
        return {
            "status": status.status,
            "calibration_interval": self.calibration_interval,
            "drift_threshold": self.drift_threshold,
            "critical_drift_threshold": self.critical_drift_threshold,
            "active": self.active,
            "calibration_count": self.calibration_count,
            "drift_events": self.drift_events,
            "critical_events": self.critical_events,
            "time_since_last_calibration": current_time - self.last_calibration if self.last_calibration else None,
            "drift_history": [
                {
                    "timestamp": m.timestamp,
                    "current_drift": m.current_drift,
                    "max_drift": m.max_drift,
                    "avg_drift": m.avg_drift
                } for m in self.drift_history[-100:]  # Last 100 entries
            ],
            "correction_history": [
                {
                    "timestamp": e.timestamp,
                    "drift_before": e.drift_before,
                    "drift_after": e.drift_after,
                    "duration": e.duration,
                    "success": e.success
                } for e in self.correction_history[-50:]  # Last 50 entries
            ],
            "drift_summary": drift_summary,
            "correction_summary": {
                "total": correction_count,
                "successful": successful_corrections,
                "success_rate": successful_corrections / correction_count if correction_count > 0 else 1.0
            }
        }
    
    def get_system_status(self) -> SystemStatus:
        """
        Determine the overall status of the quantum system.
        
        Returns:
            SystemStatus object with current health assessment
            
        As stated in documentation: "Проверка дрейфа" and "Определение общего статуса системы"
        """
        current_time = time.time()
        
        # Get latest drift metrics
        latest_drift = self.drift_history[-1].current_drift if self.drift_history else 1.0
        
        # Determine status based on drift
        if not self.drift_history:
            status = "inactive"
        elif latest_drift > self.critical_drift_threshold * 1.2:
            status = "critical"
        elif latest_drift > self.drift_threshold * 0.8:
            status = "warning"
        else:
            status = "stable"
        
        return SystemStatus(
            status=status,
            drift=latest_drift,
            drift_threshold=self.drift_threshold,
            calibration_interval=self.calibration_interval,
            last_calibration=self.last_calibration,
            time_since_calibration=current_time - self.last_calibration if self.last_calibration else None,
            calibration_count=self.calibration_count,
            drift_events=self.drift_events,
            critical_events=self.critical_events
        )
    
    def get_current_drift(self) -> float:
        """
        Get the current drift measurement.
        
        Returns:
            Current drift value (0.0 to 1.0)
        """
        if not self.drift_history:
            return 1.0  # Assume maximum drift if no history
        
        return self.drift_history[-1].current_drift
    
    def is_system_stable(self) -> bool:
        """
        Check if the quantum system is currently stable.
        
        Returns:
            bool: True if drift is below threshold, False otherwise
        """
        return self.get_current_drift() <= self.drift_threshold
    
    def force_calibration(self) -> Dict[str, Any]:
        """
        Force an immediate calibration cycle, regardless of timing.
        
        Returns:
            Calibration results
            
        Useful for debugging or when immediate correction is needed.
        """
        logger.info("Forcing immediate calibration")
        return self.run_calibration()
    
    def set_drift_thresholds(self, threshold: float, critical_threshold: Optional[float] = None) -> None:
        """
        Set custom drift thresholds.
        
        Args:
            threshold: New drift threshold for standard calibration
            critical_threshold: New critical drift threshold (defaults to 1.5x standard threshold)
            
        Raises:
            ValueError: If thresholds are invalid
        """
        if threshold <= 0 or threshold >= 1.0:
            raise ValueError("Drift threshold must be between 0 and 1")
            
        self.drift_threshold = threshold
        self.critical_drift_threshold = critical_threshold or (threshold * 1.5)
        
        logger.info(
            f"Updated drift thresholds (standard={self.drift_threshold:.4f}, "
            f"critical={self.critical_drift_threshold:.4f})"
        )
    
    def get_telemetry_data(self) -> Dict[str, Any]:
        """
        Get telemetry data for system monitoring.
        
        Returns:
            Dictionary with telemetry metrics
            
        As stated in documentation: "Телеметрия по дрейфу и деградации"
        """
        status = self.get_system_status()
        latest_drift = self.drift_history[-1] if self.drift_history else None
        
        return {
            "system_status": status.status,
            "current_drift": latest_drift.current_drift if latest_drift else 1.0,
            "drift_threshold": self.drift_threshold,
            "calibration_interval": self.calibration_interval,
            "time_since_last_calibration": time.time() - self.last_calibration if self.last_calibration else None,
            "calibration_success_rate": (
                sum(1 for e in self.correction_history if e.success) / len(self.correction_history)
                if self.correction_history else 1.0
            ),
            "timestamp": time.time()
        }
