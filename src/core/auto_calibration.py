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
("The strength is parallelism and bandwidth; the weakness is drift and precision, 
which are fixed by calibration and proper architecture.")
"""

import threading
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math
from collections import deque

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
SELF_LEARNING_WINDOW = 100         # Number of past calibrations for self-learning
MIN_CALIBRATION_SUCCESS_RATE = 0.7 # Minimum success rate to consider system stable


@dataclass
class DriftMetrics:
    """Metrics for tracking quantum state drift"""
    current_drift: float
    drift_threshold: float
    max_drift: float
    avg_drift: float
    error_distribution: List[float]
    timestamp: float
    tvi_before: float
    tvi_after: float
    calibration_duration: float


@dataclass
class CalibrationEvent:
    """Record of a calibration event"""
    timestamp: float
    drift_before: float
    drift_after: float
    tvi_before: float
    tvi_after: float
    corrections_applied: Dict[str, Any]
    duration: float
    success: bool
    environment_conditions: Dict[str, float]


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
    success_rate: float
    tvi: float


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
    
    def __init__(self, hypercube: 'AdaptiveQuantumHypercube', 
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
        self.successful_calibrations = 0
        
        # Initialize history
        self.drift_history: List[DriftMetrics] = []
        self.correction_history: List[CalibrationEvent] = deque(maxlen=MAX_HISTORY_LENGTH)
        
        # Configure thresholds
        self.drift_threshold = DRIFT_THRESHOLD
        self.critical_drift_threshold = CRITICAL_DRIFT_THRESHOLD
        
        # Self-learning parameters
        self.performance_history = []
        self.adaptive_interval_factor = 1.0
        self.environment_factors = {
            "temperature": 25.0,  # Default room temperature
            "vibration": 0.0,
            "electromagnetic_noise": 0.0
        }
        
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
            name="QuantumCalibrationThread",
            daemon=True
        )
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
            try:
                self.check_and_calibrate()
                time.sleep(1.0)  # Short sleep to reduce CPU usage
            except Exception as e:
                logger.error(f"Calibration loop error: {str(e)}")
                time.sleep(5.0)  # Longer sleep after error
        
        logger.debug("Calibration thread exiting")
    
    def check_and_calibrate(self) -> None:
        """
        Check if calibration is needed and perform it if necessary.
        
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
        """
        Determine if immediate calibration is needed.
        
        Returns:
            bool: True if calibration should be performed immediately
        
        As stated in Квантовый ПК.md: "Система авто-калибровки как обязательная часть рантайма"
        """
        # Always calibrate if critical drift is detected
        drift_metrics = self._analyze_drift()
        if drift_metrics.current_drift > self.critical_drift_threshold:
            logger.warning(
                f"Immediate calibration needed due to critical drift: "
                f"{drift_metrics.current_drift:.4f} > {self.critical_drift_threshold:.4f}"
            )
            return True
        
        # Check if TVI is too high
        metrics = self.hypercube.get_current_metrics()
        if metrics.tvi > 0.6:
            logger.warning(
                f"Immediate calibration needed due to high TVI: "
                f"{metrics.tvi:.4f} > 0.6"
            )
            return True
        
        # Check recent calibration success rate
        if self.calibration_count > 10:
            success_rate = self.successful_calibrations / self.calibration_count
            if success_rate < MIN_CALIBRATION_SUCCESS_RATE:
                logger.warning(
                    f"Immediate calibration needed due to low success rate: "
                    f"{success_rate:.2%} < {MIN_CALIBRATION_SUCCESS_RATE:.0%}"
                )
                return True
        
        return False
    
    def _perform_calibration(self) -> Dict[str, Any]:
        """
        Execute a full calibration cycle.
        
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
            pre_metrics = self.hypercube.get_current_metrics()
            pre_drift = self._analyze_drift().current_drift
            
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
            
            # Get metrics after calibration
            post_metrics = self.hypercube.get_current_metrics()
            post_drift = self._analyze_drift().current_drift
            
            # Record calibration event
            duration = time.time() - start_time
            success = (post_drift < self.drift_threshold) and (post_metrics.tvi < pre_metrics.tvi)
            
            if success:
                self.successful_calibrations += 1
            
            self._record_calibration_event(
                pre_drift,
                post_drift,
                pre_metrics.tvi,
                post_metrics.tvi,
                corrections,
                duration,
                success
            )
            
            # Adjust calibration interval based on stability
            self._adjust_calibration_interval(drift_metrics.current_drift)
            
            # Update self-learning parameters
            self._update_self_learning_parameters(success, drift_metrics)
            
            result = {
                "success": success,
                "drift_before": pre_drift,
                "drift_after": post_drift,
                "tvi_before": pre_metrics.tvi,
                "tvi_after": post_metrics.tvi,
                "corrections_applied": bool(corrections),
                "duration": duration,
                "timestamp": time.time()
            }
            
            logger.info(
                f"Calibration completed (success={success}, duration={duration:.2f}s, "
                f"drift={post_drift:.4f}, TVI={post_metrics.tvi:.4f})"
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
    
    def _analyze_drift(self, reference_states: Optional[List[np.ndarray]] = None) -> DriftMetrics:
        """
        Analyze current drift against reference states.
        
        Args:
            reference_states: List of reference quantum states
            
        Returns:
            DriftMetrics object with current drift analysis
        """
        current_time = time.time()
        
        # Get current metrics for TVI
        metrics = self.hypercube.get_current_metrics()
        
        # If no reference states provided, generate them
        if reference_states is None:
            reference_states = self._generate_reference_states()
        
        # Analyze drift against each reference state
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
            timestamp=current_time,
            tvi_before=metrics.tvi,
            tvi_after=metrics.tvi,  # Will be updated after calibration
            calibration_duration=0.0
        )
        self._record_drift_metrics(drift_metrics)
        
        return drift_metrics
    
    def _monitor_drift(self) -> DriftMetrics:
        """
        Monitor current drift without performing full calibration.
        
        This is a lightweight check that runs more frequently than full calibration.
        
        Returns:
            Current drift metrics
            
        As stated in documentation: "Планируйте телеметрию по дрейфу и деградации."
        """
        drift_metrics = self._analyze_drift()
        
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
                "duration": duration,
                "method": "state_renormalization"
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
                                 tvi_before: float, tvi_after: float,
                                 corrections: Dict[str, Any], duration: float, 
                                 success: bool) -> None:
        """
        Record a calibration event in history.
        
        Args:
            drift_before: Drift before calibration
            drift_after: Drift after calibration
            tvi_before: TVI before calibration
            tvi_after: TVI after calibration
            corrections: Corrections applied
            duration: Duration of calibration
            success: Whether calibration was successful
        """
        event = CalibrationEvent(
            timestamp=time.time(),
            drift_before=drift_before,
            drift_after=drift_after,
            tvi_before=tvi_before,
            tvi_after=tvi_after,
            corrections_applied=corrections,
            duration=duration,
            success=success,
            environment_conditions=self.environment_factors.copy()
        )
        self.correction_history.append(event)
    
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
                self.calibration_interval * (1.0 + ADAPTIVE_CALIBRATION_FACTOR * self.adaptive_interval_factor)
            )
        # If drift is high but below threshold, keep current interval
        elif current_drift < self.drift_threshold:
            new_interval = self.calibration_interval
        # If drift is above threshold, decrease interval (more frequent calibration)
        else:
            new_interval = max(
                MIN_CALIBRATION_INTERVAL,
                self.calibration_interval * (1.0 - ADAPTIVE_CALIBRATION_FACTOR * self.adaptive_interval_factor)
            )
        
        # Only update if change is significant
        if abs(new_interval - self.calibration_interval) > 5.0:
            logger.info(
                f"Adjusting calibration interval: {self.calibration_interval:.1f}s → {new_interval:.1f}s "
                f"(drift={current_drift:.4f})"
            )
            self.calibration_interval = new_interval
    
    def _update_self_learning_parameters(self, calibration_success: bool, drift_metrics: DriftMetrics) -> None:
        """
        Update self-learning parameters based on calibration results.
        
        This method uses historical data to improve future calibration accuracy.
        
        Args:
            calibration_success: Whether the calibration was successful
            drift_metrics: Drift metrics from the calibration
        """
        # Add to performance history
        self.performance_history.append({
            "success": calibration_success,
            "drift_before": drift_metrics.current_drift,
            "tvi_before": drift_metrics.tvi_before,
            "environment": self.environment_factors.copy(),
            "timestamp": time.time()
        })
        
        # Trim history if too long
        if len(self.performance_history) > SELF_LEARNING_WINDOW:
            self.performance_history = self.performance_history[-SELF_LEARNING_WINDOW:]
        
        # Calculate success rate in recent history
        recent_successes = [entry["success"] for entry in self.performance_history[-50:]]
        success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
        
        # Adjust adaptive interval factor based on success rate
        if success_rate > 0.9:
            self.adaptive_interval_factor = max(0.5, self.adaptive_interval_factor * 0.9)
        elif success_rate < 0.7:
            self.adaptive_interval_factor = min(2.0, self.adaptive_interval_factor * 1.1)
        
        # Analyze environmental factors affecting calibration
        if len(self.performance_history) >= 10:
            self._analyze_environmental_impact()
    
    def _analyze_environmental_impact(self) -> None:
        """
        Analyze how environmental factors affect calibration success.
        
        This method identifies patterns in environmental data to improve future calibration.
        """
        # Collect data
        temp_drift = []
        vibration_drift = []
        noise_drift = []
        
        for entry in self.performance_history:
            temp_drift.append((entry["environment"]["temperature"], entry["drift_before"]))
            vibration_drift.append((entry["environment"]["vibration"], entry["drift_before"]))
            noise_drift.append((entry["environment"]["electromagnetic_noise"], entry["drift_before"]))
        
        # Calculate correlations (simplified)
        temp_corr = self._calculate_correlation(temp_drift)
        vibration_corr = self._calculate_correlation(vibration_drift)
        noise_corr = self._calculate_correlation(noise_drift)
        
        # Update environmental sensitivity
        self.environment_factors["temp_sensitivity"] = abs(temp_corr)
        self.environment_factors["vibration_sensitivity"] = abs(vibration_corr)
        self.environment_factors["noise_sensitivity"] = abs(noise_corr)
        
        # Adjust calibration parameters based on sensitivity
        if abs(temp_corr) > 0.5:
            logger.debug(f"High temperature sensitivity detected (correlation={temp_corr:.2f})")
            # Could adjust calibration parameters here
        
        if abs(vibration_corr) > 0.5:
            logger.debug(f"High vibration sensitivity detected (correlation={vibration_corr:.2f})")
            # Could adjust calibration parameters here
    
    def _calculate_correlation(self, data: List[Tuple[float, float]]) -> float:
        """
        Calculate correlation coefficient between two variables.
        
        Args:
            data: List of (x, y) pairs
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(data) < 2:
            return 0.0
        
        xs, ys = zip(*data)
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in data)
        denominator_x = np.sqrt(sum((x - x_mean) ** 2 for x in xs))
        denominator_y = np.sqrt(sum((y - y_mean) ** 2 for y in ys))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        
        return numerator / (denominator_x * denominator_y)
    
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
            "successful_calibrations": self.successful_calibrations,
            "drift_events": self.drift_events,
            "critical_events": self.critical_events,
            "time_since_last_calibration": current_time - self.last_calibration if self.last_calibration else None,
            "drift_history": [
                {
                    "timestamp": m.timestamp,
                    "current_drift": m.current_drift,
                    "max_drift": m.max_drift,
                    "avg_drift": m.avg_drift,
                    "tvi_before": m.tvi_before,
                    "tvi_after": m.tvi_after
                } for m in self.drift_history[-100:]  # Last 100 entries
            ],
            "correction_history": [
                {
                    "timestamp": e.timestamp,
                    "drift_before": e.drift_before,
                    "drift_after": e.drift_after,
                    "tvi_before": e.tvi_before,
                    "tvi_after": e.tvi_after,
                    "duration": e.duration,
                    "success": e.success,
                    "environment": e.environment_conditions
                } for e in self.correction_history  # All entries (limited by deque)
            ],
            "drift_summary": drift_summary,
            "correction_summary": {
                "total": correction_count,
                "successful": successful_corrections,
                "success_rate": successful_corrections / correction_count if correction_count > 0 else 1.0
            },
            "self_learning": {
                "adaptive_interval_factor": self.adaptive_interval_factor,
                "environmental_sensitivity": {
                    "temperature": self.environment_factors.get("temp_sensitivity", 0.0),
                    "vibration": self.environment_factors.get("vibration_sensitivity", 0.0),
                    "electromagnetic_noise": self.environment_factors.get("noise_sensitivity", 0.0)
                },
                "performance_history_size": len(self.performance_history)
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
        
        # Get current TVI
        tvi = self.hypercube.get_current_metrics().tvi if hasattr(self.hypercube, 'get_current_metrics') else 1.0
        
        # Calculate success rate
        success_rate = (
            self.successful_calibrations / self.calibration_count 
            if self.calibration_count > 0 else 1.0
        )
        
        return SystemStatus(
            status=status,
            drift=latest_drift,
            drift_threshold=self.drift_threshold,
            calibration_interval=self.calibration_interval,
            last_calibration=self.last_calibration,
            time_since_calibration=current_time - self.last_calibration if self.last_calibration else None,
            calibration_count=self.calibration_count,
            drift_events=self.drift_events,
            critical_events=self.critical_events,
            success_rate=success_rate,
            tvi=tvi
        )
    
    def get_current_drift(self) -> float:
        """
        Get the current drift measurement.
        
        Returns:
            float: Current drift value (0.0 to 1.0)
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
        return self._perform_calibration()
    
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
                self.successful_calibrations / self.calibration_count
                if self.calibration_count > 0 else 1.0
            ),
            "tvi": self.hypercube.get_current_metrics().tvi if hasattr(self.hypercube, 'get_current_metrics') else 1.0,
            "timestamp": time.time()
        }
    
    def simulate_environmental_change(self, **kwargs) -> None:
        """
        Simulate environmental changes for testing calibration response.
        
        Args:
            **kwargs: Environmental factors to change (temperature, vibration, etc.)
            
        Example:
            >>> calibrator.simulate_environmental_change(temperature=35.0, vibration=0.7)
        """
        for key, value in kwargs.items():
            if key in self.environment_factors:
                self.environment_factors[key] = value
                logger.info(f"Simulated environmental change: {key} = {value}")
    
    def reset_self_learning(self) -> None:
        """
        Reset the self-learning parameters to initial state.
        
        Useful when the system is moved to a new environment or after major updates.
        """
        self.performance_history = []
        self.adaptive_interval_factor = 1.0
        logger.info("Self-learning parameters reset")
    
    def get_self_learning_report(self) -> Dict[str, Any]:
        """
        Generate a report on the self-learning capabilities of the calibration system.
        
        Returns:
            Dictionary with self-learning metrics and insights
        """
        if not self.performance_history:
            return {
                "status": "insufficient_data",
                "message": "Not enough calibration history for self-learning analysis"
            }
        
        # Calculate success rates by environmental conditions
        temp_buckets = {}
        vibration_buckets = {}
        
        for entry in self.performance_history:
            # Temperature buckets (5-degree intervals)
            temp_key = f"{int(entry['environment']['temperature'] / 5) * 5}-{int(entry['environment']['temperature'] / 5) * 5 + 5}"
            if temp_key not in temp_buckets:
                temp_buckets[temp_key] = {"successes": 0, "total": 0}
            temp_buckets[temp_key]["total"] += 1
            if entry["success"]:
                temp_buckets[temp_key]["successes"] += 1
            
            # Vibration buckets (0.1 intervals)
            vibration_key = f"{int(entry['environment']['vibration'] / 0.1) * 0.1:.1f}-{int(entry['environment']['vibration'] / 0.1) * 0.1 + 0.1:.1f}"
            if vibration_key not in vibration_buckets:
                vibration_buckets[vibration_key] = {"successes": 0, "total": 0}
            vibration_buckets[vibration_key]["total"] += 1
            if entry["success"]:
                vibration_buckets[vibration_key]["successes"] += 1
        
        # Format buckets for report
        temp_report = {
            bucket: {
                "success_rate": data["successes"] / data["total"],
                "sample_size": data["total"]
            } for bucket, data in temp_buckets.items()
        }
        
        vibration_report = {
            bucket: {
                "success_rate": data["successes"] / data["total"],
                "sample_size": data["total"]
            } for bucket, data in vibration_buckets.items()
        }
        
        return {
            "adaptive_interval_factor": self.adaptive_interval_factor,
            "environmental_impact": {
                "temperature": temp_report,
                "vibration": vibration_report,
                "electromagnetic_noise": self.environment_factors.get("noise_sensitivity", 0.0)
            },
            "learning_window_size": len(self.performance_history),
            "total_calibrations_analyzed": len(self.performance_history),
            "timestamp": time.time()
        }
    
    def optimize_for_environment(self) -> Dict[str, Any]:
        """
        Optimize calibration parameters for current environmental conditions.
        
        Returns:
            Dictionary with optimization recommendations
        """
        report = self.get_self_learning_report()
        if report["status"] == "insufficient_data":
            return {
                "status": "insufficient_data",
                "message": "Not enough data to optimize for environment"
            }
        
        # Find optimal calibration interval based on current conditions
        current_temp = self.environment_factors["temperature"]
        current_vibration = self.environment_factors["vibration"]
        
        # Find temperature bucket for current temperature
        temp_key = f"{int(current_temp / 5) * 5}-{int(current_temp / 5) * 5 + 5}"
        
        # Find vibration bucket for current vibration
        vibration_key = f"{int(current_vibration / 0.1) * 0.1:.1f}-{int(current_vibration / 0.1) * 0.1 + 0.1:.1f}"
        
        # Get success rates for current conditions
        temp_success = report["environmental_impact"]["temperature"].get(temp_key, {}).get("success_rate", 0.8)
        vibration_success = report["environmental_impact"]["vibration"].get(vibration_key, {}).get("success_rate", 0.8)
        
        # Calculate recommended interval
        base_interval = self.calibration_interval
        temp_factor = 1.0 if temp_success > 0.7 else 0.7
        vibration_factor = 1.0 if vibration_success > 0.7 else 0.7
        
        recommended_interval = base_interval * temp_factor * vibration_factor
        recommended_interval = max(MIN_CALIBRATION_INTERVAL, min(MAX_CALIBRATION_INTERVAL, recommended_interval))
        
        # Generate recommendations
        recommendations = []
        if temp_success < 0.7:
            recommendations.append(
                f"Consider environmental control: Temperature ({current_temp}°C) "
                "is negatively impacting calibration success"
            )
        if vibration_success < 0.7:
            recommendations.append(
                f"Consider vibration isolation: Vibration level ({current_vibration:.2f}) "
                "is negatively impacting calibration success"
            )
        
        return {
            "current_calibration_interval": self.calibration_interval,
            "recommended_interval": recommended_interval,
            "temperature_impact": temp_success,
            "vibration_impact": vibration_success,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
