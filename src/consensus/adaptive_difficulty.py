"""
adaptive_difficulty.py - Adaptive difficulty adjustment for QuantumFortress 2.0.

This module implements the dynamic difficulty adjustment system for QuantumFortress 2.0,
which adjusts mining difficulty based on topological security metrics rather than
just time-based calculations.

The key innovation is the integration of TVI (Topological Vulnerability Index) into
the difficulty adjustment algorithm. As stated in the documentation: "Topology isn't 
a hacking tool, but a microscope for diagnosing vulnerabilities. Ignoring it means 
building cryptography on sand."

The module provides:
- AdaptiveDifficulty: Core class for dynamic difficulty adjustment
- adjust_difficulty: Main method for difficulty calculation
- TVI-based difficulty adjustment: Higher TVI = higher difficulty
- Quantum state awareness: Difficulty adjusts based on quantum stability
- Smooth transition algorithms: Prevents abrupt difficulty changes

Based on the fundamental principle from Ur Uz работа.md: "Множество решений уравнения
ECDSA топологически эквивалентно двумерному тору S¹ × S¹" (The set of solutions to the
ECDSA equation is topologically equivalent to the 2D torus S¹ × S¹).

Author: Quantum Topology Research Group
Institution: Tambov Research Institute of Quantum Topology
Email: miro-aleksej@yandex.ru
"""

import time
import logging
import math
from typing import Dict, Any, Optional
import numpy as np

# Import internal dependencies
from quantum_fortress.topology.metrics import TVI_SECURE_THRESHOLD, TVI_WARNING_THRESHOLD
from quantum_fortress.core.adaptive_hypercube import AdaptiveQuantumHypercube

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Difficulty adjustment constants
TARGET_BLOCK_TIME = 0.8  # Target block time in seconds (0.8 seconds)
MAX_ADJUSTMENT_FACTOR = 2.0  # Maximum difficulty adjustment factor per block
MIN_ADJUSTMENT_FACTOR = 0.5  # Minimum difficulty adjustment factor per block
TIME_WINDOW = 120.0  # Time window for difficulty calculation (seconds)
MAX_BLOCK_TIME_DEVIATION = 3.0  # Maximum allowed block time deviation factor
TVI_DIFFICULTY_FACTOR = 2.0  # Difficulty multiplier based on TVI
MINIMUM_DIFFICULTY = 1.0  # Absolute minimum difficulty
MAXIMUM_DIFFICULTY = 100.0  # Absolute maximum difficulty
BASE_DIFFICULTY = 15.0  # Starting difficulty
SMOOTHING_FACTOR = 0.25  # Smoothing factor for gradual difficulty changes
TVI_HISTORY_WEIGHT = 0.7  # Weight for historical TVI in adjustment
TIME_HISTORY_WEIGHT = 0.3  # Weight for historical time in adjustment
DRIFT_DIFFICULTY_FACTOR = 0.5  # Difficulty adjustment factor based on quantum drift

class DifficultyAdjustmentMetrics:
    """
    Container for difficulty adjustment metrics.
    
    This class provides a structured way to track and analyze the factors
    influencing difficulty adjustments.
    """
    def __init__(self,
                 current_difficulty: float,
                 target_difficulty: float,
                 time_factor: float,
                 tvi_factor: float,
                 quantum_factor: float,
                 adjustment_factor: float,
                 block_time: float,
                 tvi: float,
                 quantum_drift: float,
                 timestamp: float):
        self.current_difficulty = current_difficulty
        self.target_difficulty = target_difficulty
        self.time_factor = time_factor
        self.tvi_factor = tvi_factor
        self.quantum_factor = quantum_factor
        self.adjustment_factor = adjustment_factor
        self.block_time = block_time
        self.tvi = tvi
        self.quantum_drift = quantum_drift
        self.timestamp = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "current_difficulty": self.current_difficulty,
            "target_difficulty": self.target_difficulty,
            "time_factor": self.time_factor,
            "tvi_factor": self.tvi_factor,
            "quantum_factor": self.quantum_factor,
            "adjustment_factor": self.adjustment_factor,
            "block_time": self.block_time,
            "tvi": self.tvi,
            "quantum_drift": self.quantum_drift,
            "timestamp": self.timestamp
        }

class AdaptiveDifficulty:
    """
    Adaptive difficulty adjustment system for QuantumFortress 2.0.
    
    This class implements a dynamic difficulty adjustment algorithm that considers:
    1. Time-based factors (block production rate)
    2. Topological security metrics (TVI)
    3. Quantum state stability
    
    Key features:
    - TVI-based difficulty adjustment (higher TVI = higher difficulty)
    - Quantum drift awareness (higher drift = higher difficulty)
    - Smooth transition algorithms to prevent abrupt changes
    - Historical data tracking for more stable adjustments
    - Configurable adjustment parameters
    
    The system follows the principle: "Хорошая система «подпевает себе» постоянно,
    тихо и незаметно для пользователя" from QuantumFortress documentation.
    
    Example usage:
        difficulty_adjuster = AdaptiveDifficulty(
            target_block_time=0.8,
            tvi_difficulty_factor=2.0
        )
        
        # Adjust difficulty after a new block
        new_difficulty = difficulty_adjuster.adjust_difficulty(
            current_difficulty,
            block_time,
            tvi
        )
    """
    
    def __init__(self,
                 target_block_time: float = TARGET_BLOCK_TIME,
                 time_window: float = TIME_WINDOW,
                 tvi_difficulty_factor: float = TVI_DIFFICULTY_FACTOR,
                 max_adjustment: float = MAX_ADJUSTMENT_FACTOR,
                 min_adjustment: float = MIN_ADJUSTMENT_FACTOR,
                 smoothing_factor: float = SMOOTHING_FACTOR):
        """
        Initialize the adaptive difficulty system.
        
        Args:
            target_block_time: Target time between blocks (seconds)
            time_window: Time window for difficulty calculation (seconds)
            tvi_difficulty_factor: Multiplier for TVI-based difficulty adjustment
            max_adjustment: Maximum difficulty adjustment factor per block
            min_adjustment: Minimum difficulty adjustment factor per block
            smoothing_factor: Factor for smoothing difficulty changes
        """
        self.target_block_time = target_block_time
        self.time_window = time_window
        self.tvi_difficulty_factor = tvi_difficulty_factor
        self.max_adjustment = max_adjustment
        self.min_adjustment = min_adjustment
        self.smoothing_factor = smoothing_factor
        
        # Historical data
        self.history = []
        self.max_history_size = int(time_window / target_block_time * 2)
        
        # Current difficulty
        self.current_difficulty = BASE_DIFFICULTY
        
        logger.info(
            f"AdaptiveDifficulty initialized with target_block_time={target_block_time}, "
            f"tvi_difficulty_factor={tvi_difficulty_factor}"
        )
    
    def adjust_difficulty(self,
                         current_difficulty: float,
                         block_time: float,
                         tvi: float,
                         quantum_drift: Optional[float] = None) -> float:
        """
        Adjust the mining difficulty based on multiple factors.
        
        The difficulty is adjusted using a weighted combination of:
        1. Time-based adjustment (to maintain target block time)
        2. TVI-based adjustment (to incentivize secure implementations)
        3. Quantum state adjustment (to account for quantum drift)
        
        Args:
            current_difficulty: Current difficulty level
            block_time: Time taken to mine the last block (seconds)
            tvi: Topological Vulnerability Index of the last block
            quantum_drift: Current quantum drift (optional)
            
        Returns:
            float: Adjusted difficulty level
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if current_difficulty < MINIMUM_DIFFICULTY:
            current_difficulty = MINIMUM_DIFFICULTY
        if current_difficulty > MAXIMUM_DIFFICULTY:
            current_difficulty = MAXIMUM_DIFFICULTY
        if block_time <= 0:
            block_time = self.target_block_time  # Default to target if invalid
        
        # Get quantum drift if not provided
        if quantum_drift is None:
            quantum_drift = 0.0  # Default to no drift if not provided
        
        # Calculate time-based adjustment factor
        time_factor = self._calculate_time_adjustment(block_time)
        
        # Calculate TVI-based adjustment factor
        tvi_factor = self._calculate_tvi_adjustment(tvi)
        
        # Calculate quantum-based adjustment factor
        quantum_factor = self._calculate_quantum_adjustment(quantum_drift)
        
        # Calculate overall adjustment factor
        adjustment_factor = (
            time_factor * TIME_HISTORY_WEIGHT +
            tvi_factor * TVI_HISTORY_WEIGHT +
            quantum_factor
        )
        
        # Apply adjustment bounds
        adjustment_factor = max(self.min_adjustment, min(self.max_adjustment, adjustment_factor))
        
        # Calculate target difficulty
        target_difficulty = current_difficulty * adjustment_factor
        
        # Smooth the transition
        smoothed_difficulty = self._smooth_adjustment(
            current_difficulty,
            target_difficulty
        )
        
        # Apply absolute bounds
        final_difficulty = max(
            MINIMUM_DIFFICULTY,
            min(MAXIMUM_DIFFICULTY, smoothed_difficulty)
        )
        
        # Update history
        self._update_history(
            current_difficulty,
            final_difficulty,
            time_factor,
            tvi_factor,
            quantum_factor,
            adjustment_factor,
            block_time,
            tvi,
            quantum_drift
        )
        
        # Log the adjustment
        logger.debug(
            f"Difficulty adjusted from {current_difficulty:.2f} to {final_difficulty:.2f} "
            f"(time_factor={time_factor:.2f}, tvi_factor={tvi_factor:.2f}, "
            f"quantum_factor={quantum_factor:.2f}, adjustment={adjustment_factor:.2f})"
        )
        
        # Update current difficulty
        self.current_difficulty = final_difficulty
        
        return final_difficulty
    
    def _calculate_time_adjustment(self, block_time: float) -> float:
        """
        Calculate time-based difficulty adjustment factor.
        
        If block_time < target_block_time: difficulty increases
        If block_time > target_block_time: difficulty decreases
        
        Args:
            block_time: Time taken to mine the last block (seconds)
            
        Returns:
            float: Time-based adjustment factor
        """
        # Calculate time ratio (actual / target)
        time_ratio = block_time / self.target_block_time
        
        # Cap extreme values to prevent massive jumps
        time_ratio = max(1.0 / MAX_BLOCK_TIME_DEVIATION, 
                        min(MAX_BLOCK_TIME_DEVIATION, time_ratio))
        
        # Calculate adjustment factor (inverse relationship)
        return 1.0 / time_ratio
    
    def _calculate_tvi_adjustment(self, tvi: float) -> float:
        """
        Calculate TVI-based difficulty adjustment factor.
        
        Higher TVI = higher difficulty (to incentivize secure implementations)
        
        Args:
            tvi: Topological Vulnerability Index
            
        Returns:
            float: TVI-based adjustment factor
        """
        # Base adjustment: difficulty = base * (1 + TVI_DIFFICULTY_FACTOR * TVI)
        adjustment = 1.0 + self.tvi_difficulty_factor * tvi
        
        # Ensure minimum adjustment
        return max(1.0, adjustment)
    
    def _calculate_quantum_adjustment(self, quantum_drift: float) -> float:
        """
        Calculate quantum state-based difficulty adjustment factor.
        
        Higher quantum drift = higher difficulty (to incentivize calibration)
        
        Args:
            quantum_drift: Current quantum drift (0.0 to 1.0)
            
        Returns:
            float: Quantum-based adjustment factor
        """
        # Base adjustment: difficulty = base * (1 + DRIFT_DIFFICULTY_FACTOR * drift)
        adjustment = 1.0 + DRIFT_DIFFICULTY_FACTOR * quantum_drift
        
        # Ensure minimum adjustment
        return max(1.0, adjustment)
    
    def _smooth_adjustment(self, 
                          current_difficulty: float, 
                          target_difficulty: float) -> float:
        """
        Apply smoothing to the difficulty adjustment to prevent abrupt changes.
        
        Args:
            current_difficulty: Current difficulty level
            target_difficulty: Target difficulty level after adjustment
            
        Returns:
            float: Smoothed difficulty level
        """
        # Simple exponential smoothing
        return (1 - self.smoothing_factor) * current_difficulty + \
               self.smoothing_factor * target_difficulty
    
    def _update_history(self,
                       current_difficulty: float,
                       target_difficulty: float,
                       time_factor: float,
                       tvi_factor: float,
                       quantum_factor: float,
                       adjustment_factor: float,
                       block_time: float,
                       tvi: float,
                       quantum_drift: float):
        """
        Update the difficulty adjustment history.
        
        Args:
            current_difficulty: Current difficulty before adjustment
            target_difficulty: Target difficulty after adjustment
            time_factor: Time-based adjustment factor
            tvi_factor: TVI-based adjustment factor
            quantum_factor: Quantum-based adjustment factor
            adjustment_factor: Overall adjustment factor
            block_time: Time taken to mine the block
            tvi: Topological Vulnerability Index
            quantum_drift: Quantum drift value
        """
        # Create metrics object
        metrics = DifficultyAdjustmentMetrics(
            current_difficulty=current_difficulty,
            target_difficulty=target_difficulty,
            time_factor=time_factor,
            tvi_factor=tvi_factor,
            quantum_factor=quantum_factor,
            adjustment_factor=adjustment_factor,
            block_time=block_time,
            tvi=tvi,
            quantum_drift=quantum_drift,
            timestamp=time.time()
        )
        
        # Add to history
        self.history.append(metrics)
        
        # Trim history if too large
        if len(self.history) > self.max_history_size:
            self.history.pop(0)
    
    def get_current_difficulty(self) -> float:
        """
        Get the current difficulty level.
        
        Returns:
            float: Current difficulty level
        """
        return self.current_difficulty
    
    def get_history(self) -> List[DifficultyAdjustmentMetrics]:
        """
        Get the difficulty adjustment history.
        
        Returns:
            List[DifficultyAdjustmentMetrics]: History of difficulty adjustments
        """
        return self.history
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about difficulty adjustments.
        
        Returns:
            Dict[str, Any]: Difficulty adjustment statistics
        """
        if not self.history:
            return {
                "average_difficulty": BASE_DIFFICULTY,
                "average_block_time": self.target_block_time,
                "average_tvi": 0.0,
                "average_quantum_drift": 0.0,
                "adjustment_count": 0
            }
        
        # Calculate averages
        total_difficulty = 0.0
        total_block_time = 0.0
        total_tvi = 0.0
        total_quantum_drift = 0.0
        
        for metrics in self.history:
            total_difficulty += metrics.target_difficulty
            total_block_time += metrics.block_time
            total_tvi += metrics.tvi
            total_quantum_drift += metrics.quantum_drift
        
        n = len(self.history)
        
        return {
            "average_difficulty": total_difficulty / n,
            "average_block_time": total_block_time / n,
            "average_tvi": total_tvi / n,
            "average_quantum_drift": total_quantum_drift / n,
            "adjustment_count": n,
            "current_difficulty": self.current_difficulty,
            "target_block_time": self.target_block_time
        }
    
    def reset(self, new_difficulty: Optional[float] = None):
        """
        Reset the difficulty adjustment system.
        
        Args:
            new_difficulty: Optional new difficulty level (defaults to BASE_DIFFICULTY)
        """
        self.history = []
        self.current_difficulty = new_difficulty or BASE_DIFFICULTY
        logger.info(f"Difficulty adjustment system reset to {self.current_difficulty:.2f}")
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations based on current difficulty adjustment metrics.
        
        Returns:
            List[str]: Recommendations for system optimization
        """
        recommendations = []
        stats = self.get_statistics()
        
        # Block time recommendations
        if stats["average_block_time"] < self.target_block_time * 0.8:
            recommendations.append(
                "Среднее время блока ниже целевого - система может быть перекалибрована. "
                "Рассмотрите возможность уменьшения числа каналов WDM."
            )
        elif stats["average_block_time"] > self.target_block_time * 1.2:
            recommendations.append(
                "Среднее время блока выше целевого - система может быть недокалибрована. "
                "Рассмотрите возможность увеличения числа каналов WDM."
            )
        
        # TVI recommendations
        if stats["average_tvi"] > TVI_WARNING_THRESHOLD:
            recommendations.append(
                "Высокий средний TVI - рекомендуется проверить криптографическую реализацию. "
                "TVI > 0.5 указывает на критические уязвимости."
            )
        elif stats["average_tvi"] > TVI_SECURE_THRESHOLD:
            recommendations.append(
                "Средний TVI выше безопасного уровня - рекомендуется проверить систему на уязвимости."
            )
        
        # Quantum drift recommendations
        if stats["average_quantum_drift"] > 0.1:
            recommendations.append(
                "Высокий дрейф квантового состояния - рекомендуется проверить систему калибровки."
            )
        
        # No recommendations
        if not recommendations:
            recommendations.append(
                "Система адаптивной сложности работает в оптимальном режиме. "
                "Нет необходимости в изменениях."
            )
        
        return recommendations
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """
        Generate a diagnostic report for the difficulty adjustment system.
        
        Returns:
            Dict[str, Any]: Diagnostic report
        """
        stats = self.get_statistics()
        history = self.history[-10:] if self.history else []
        
        return {
            "system_status": "active",
            "difficulty_version": "2.0",
            "current_difficulty": stats["current_difficulty"],
            "target_block_time": self.target_block_time,
            "parameters": {
                "tvi_difficulty_factor": self.tvi_difficulty_factor,
                "max_adjustment": self.max_adjustment,
                "min_adjustment": self.min_adjustment,
                "smoothing_factor": self.smoothing_factor
            },
            "statistics": {
                "average_difficulty": stats["average_difficulty"],
                "average_block_time": stats["average_block_time"],
                "average_tvi": stats["average_tvi"],
                "average_quantum_drift": stats["average_quantum_drift"],
                "adjustment_count": stats["adjustment_count"]
            },
            "recent_adjustments": [m.to_dict() for m in history],
            "recommendations": self.get_recommendations(),
            "timestamp": time.time()
        }
    
    def analyze_security_impact(self, tvi: float) -> Dict[str, Any]:
        """
        Analyze the security impact of the current TVI level.
        
        Args:
            tvi: Topological Vulnerability Index
            
        Returns:
            Dict[str, Any]: Security impact analysis
        """
        # Determine security level
        if tvi < TVI_SECURE_THRESHOLD:
            security_level = "SECURE"
            risk_description = "Низкий риск (безопасно)"
            recommendation = "Система безопасна, нет необходимости в миграции"
        elif tvi < TVI_WARNING_THRESHOLD:
            security_level = "WARNING"
            risk_description = "Средний риск (требуется проверка)"
            recommendation = "Рекомендуется миграция на более безопасную реализацию"
        else:
            security_level = "CRITICAL"
            risk_description = "ВЫСОКИЙ РИСК (немедленное действие)"
            recommendation = "Требуется немедленная миграция на безопасную реализацию"
        
        # Calculate difficulty impact
        difficulty_impact = self._calculate_tvi_adjustment(tvi)
        
        return {
            "security_level": security_level,
            "risk_description": risk_description,
            "tvi": tvi,
            "difficulty_impact": difficulty_impact,
            "recommendation": recommendation,
            "timestamp": time.time()
        }

def adjust_difficulty(current_difficulty: float,
                     block_time: float,
                     tvi: float,
                     target_block_time: float = TARGET_BLOCK_TIME,
                     tvi_difficulty_factor: float = TVI_DIFFICULTY_FACTOR) -> float:
    """
    Adjust difficulty with a simple interface (standalone function).
    
    Args:
        current_difficulty: Current difficulty level
        block_time: Time taken to mine the last block (seconds)
        tvi: Topological Vulnerability Index
        target_block_time: Target block time (seconds)
        tvi_difficulty_factor: Factor for TVI-based adjustment
        
    Returns:
        float: Adjusted difficulty level
    
    Example from Ur Uz работа.md:
    "Классификация риска: if tvi < 0.2: return 'Низкий риск (безопасно)'"
    """
    adjuster = AdaptiveDifficulty(
        target_block_time=target_block_time,
        tvi_difficulty_factor=tvi_difficulty_factor
    )
    return adjuster.adjust_difficulty(current_difficulty, block_time, tvi)

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
    
    Example from Ur Uz работа.md:
    "Блокирует транзакции с TVI > 0.5"
    """
    if tvi < 0.1:
        return 0
    elif tvi < 0.5:
        return 1
    else:
        return 2

def example_usage() -> None:
    """
    Example usage of AdaptiveDifficulty for difficulty adjustment.
    
    Demonstrates how to use the module for dynamic difficulty adjustment.
    """
    print("=" * 60)
    print("Пример использования AdaptiveDifficulty для динамической настройки сложности")
    print("=" * 60)
    
    # Initialize difficulty adjuster
    print("\n1. Инициализация системы адаптивной сложности...")
    difficulty_adjuster = AdaptiveDifficulty(
        target_block_time=0.8,
        tvi_difficulty_factor=2.0
    )
    print(f"  - Инициализирована с целевым временем блока {difficulty_adjuster.target_block_time} сек")
    print(f"  - Фактор TVI: {difficulty_adjuster.tvi_difficulty_factor}")
    
    # Simulate block mining with different conditions
    print("\n2. Симуляция майнинга блоков с разными условиями...")
    
    # Scenario 1: Fast block with secure system
    print("\n  a) Быстрый блок с безопасной системой (TVI=0.05):")
    current_difficulty = 15.0
    block_time = 0.4  # Faster than target
    tvi = 0.05
    new_difficulty = difficulty_adjuster.adjust_difficulty(
        current_difficulty,
        block_time,
        tvi
    )
    print(f"    - Текущая сложность: {current_difficulty:.2f}")
    print(f"    - Время блока: {block_time:.2f} сек")
    print(f"    - TVI: {tvi:.2f}")
    print(f"    - Новая сложность: {new_difficulty:.2f}")
    
    # Scenario 2: Slow block with vulnerable system
    print("\n  b) Медленный блок с уязвимой системой (TVI=0.6):")
    current_difficulty = new_difficulty
    block_time = 1.5  # Slower than target
    tvi = 0.6
    new_difficulty = difficulty_adjuster.adjust_difficulty(
        current_difficulty,
        block_time,
        tvi
    )
    print(f"    - Текущая сложность: {current_difficulty:.2f}")
    print(f"    - Время блока: {block_time:.2f} сек")
    print(f"    - TVI: {tvi:.2f}")
    print(f"    - Новая сложность: {new_difficulty:.2f}")
    
    # Scenario 3: Normal block with warning system
    print("\n  c) Нормальный блок с системой предупреждения (TVI=0.3):")
    current_difficulty = new_difficulty
    block_time = 0.8  # Target time
    tvi = 0.3
    new_difficulty = difficulty_adjuster.adjust_difficulty(
        current_difficulty,
        block_time,
        tvi
    )
    print(f"    - Текущая сложность: {current_difficulty:.2f}")
    print(f"    - Время блока: {block_time:.2f} сек")
    print(f"    - TVI: {tvi:.2f}")
    print(f"    - Новая сложность: {new_difficulty:.2f}")
    
    # Get statistics
    print("\n3. Получение статистики системы:")
    stats = difficulty_adjuster.get_statistics()
    print(f"  - Средняя сложность: {stats['average_difficulty']:.2f}")
    print(f"  - Среднее время блока: {stats['average_block_time']:.2f} сек")
    print(f"  - Средний TVI: {stats['average_tvi']:.2f}")
    print(f"  - Средний квантовый дрейф: {stats['average_quantum_drift']:.2f}")
    
    # Get recommendations
    print("\n4. Рекомендации по оптимизации системы:")
    recommendations = difficulty_adjuster.get_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Analyze security impact
    print("\n5. Анализ влияния TVI на безопасность:")
    for tvi_value in [0.05, 0.3, 0.6]:
        impact = difficulty_adjuster.analyze_security_impact(tvi_value)
        print(f"  - TVI = {tvi_value:.2f}: {impact['risk_description']}")
        print(f"    - Уровень безопасности: {impact['security_level']}")
        print(f"    - Влияние на сложность: x{impact['difficulty_impact']:.2f}")
        print(f"    - Рекомендация: {impact['recommendation']}")
    
    print("\n6. Генерация диагностического отчета...")
    report = difficulty_adjuster.get_diagnostic_report()
    print(f"  - Статус системы: {report['system_status']}")
    print(f"  - Версия системы: {report['difficulty_version']}")
    print(f"  - Текущая сложность: {report['current_difficulty']:.2f}")
    print(f"  - Среднее время блока: {report['statistics']['average_block_time']:.2f} сек")
    
    print("=" * 60)
    print("AdaptiveDifficulty успешно продемонстрировал динамическую настройку сложности.")
    print("=" * 60)

if __name__ == "__main__":
    # Run example usage when module is executed directly
    example_usage()
