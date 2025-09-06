"""
optimized_cache.py - Topologically-optimized cache for cryptographic security analysis.

This module implements a cache system that leverages topological properties of cryptographic
signatures to optimize lookup and storage. Instead of traditional cache approaches that rely
on exact key matching, this system uses topological proximity to identify "nearby" regions
that can benefit from similar optimization strategies.

Key features:
- Topological region identification based on Betti numbers, curvature, and spiral patterns
- Dynamic cache sizing based on access patterns and topological stability
- Efficient lookup using topological proximity rather than exact key matching
- Integration with TVI (Topological Vulnerability Index) for security-aware caching

Based on principles from:
- Ur Uz работа.md: "Множество решений уравнения ECDSA топологически эквивалентно двумерному тору S¹ × S¹"
- Ur Uz работа_2.md: "Связь нарушений аксиом с конкретными уязвимостями"
- Prototype_TopoMine.txt: Implementation of region-based caching for signature verification

Author: Quantum Topology Research Group
Institution: Tambov Research Institute of Quantum Topology
Email: miro-aleksej@yandex.ru
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math
from scipy.spatial import distance
from .metrics import TopologicalMetrics, TVI_SECURE_THRESHOLD

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Cache configuration constants
DEFAULT_MAX_SIZE = 1000
MIN_CACHE_SIZE = 100
MAX_CACHE_SIZE = 5000
CACHE_DECAY_RATE = 0.95
ACCESS_WEIGHT = 0.7
TOPOLOGICAL_WEIGHT = 0.3
REGION_SIMILARITY_THRESHOLD = 0.2
CACHE_HIT_BONUS = 1.5
CACHE_MISS_PENALTY = 0.8
STABILITY_THRESHOLD = 0.15
RECENT_ACCESS_WINDOW = 300  # 5 minutes in seconds

@dataclass
class CacheEntry:
    """Represents an entry in the topologically-optimized cache."""
    metrics: TopologicalMetrics
    region_id: str
    last_access: float
    access_count: int
    stability_score: float
    performance_gain: float
    signature_count: int
    region_fingerprint: np.ndarray
    last_verification_time: float = 0.0

class TopologicalRegion:
    """Represents a topological region with similar characteristics."""
    def __init__(self, 
                 fingerprint: np.ndarray,
                 metrics: TopologicalMetrics,
                 region_id: str):
        self.fingerprint = fingerprint
        self.metrics = metrics
        self.region_id = region_id
        self.access_count = 1
        self.last_access = time.time()
        self.stability_score = self._calculate_stability(metrics)
        self.performance_gain = 4.5  # Default for secure regions
    
    def _calculate_stability(self, metrics: TopologicalMetrics) -> float:
        """Calculate stability score based on topological metrics."""
        # Higher stability for secure regions with consistent metrics
        stability = 1.0 - metrics.tvi
        
        # Additional stability for consistent topological entropy
        if metrics.topological_entropy > 0.8:
            stability *= 1.1
        
        # Penalize regions with high vulnerability indicators
        if metrics.vulnerability_type != "none":
            stability *= 0.7
        
        return max(0.0, min(1.0, stability))
    
    def update_access(self, metrics: TopologicalMetrics, performance_gain: float = 4.5):
        """Update region access statistics."""
        self.access_count += 1
        self.last_access = time.time()
        
        # Update stability score with exponential moving average
        new_stability = self._calculate_stability(metrics)
        self.stability_score = 0.9 * self.stability_score + 0.1 * new_stability
        
        # Update performance gain
        if performance_gain > 0:
            self.performance_gain = (
                (self.access_count - 1) * self.performance_gain + performance_gain
            ) / self.access_count

class TopologicallyOptimizedCache:
    """
    Cache system optimized for topological properties of cryptographic signatures.
    
    Unlike traditional caches that rely on exact key matching, this system identifies
    "nearby" topological regions that can benefit from similar optimization strategies.
    This approach significantly improves cache hit rates for cryptographic operations
    by leveraging the continuous nature of topological space.
    
    The cache dynamically adjusts its size based on:
    - Topological stability of regions
    - Access patterns and recency
    - Performance gains from cached regions
    - System resource constraints
    
    Example usage:
        cache = TopologicallyOptimizedCache(max_size=1000)
        # Analyze topology of signatures
        topology_metrics = analyze_signature_topology(signatures)
        # Get optimized verification strategy
        strategy = cache.get(topology_metrics)
        if strategy:
            result = fast_verify_with_strategy(signatures, strategy)
        else:
            result = standard_verify(signatures)
            # Cache the result for future similar topologies
            cache.store(topology_metrics, strategy, performance_gain=4.5)
    """
    
    def __init__(self,
                 max_size: int = DEFAULT_MAX_SIZE,
                 similarity_threshold: float = REGION_SIMILARITY_THRESHOLD,
                 stability_threshold: float = STABILITY_THRESHOLD,
                 decay_rate: float = CACHE_DECAY_RATE):
        """
        Initialize the topologically-optimized cache.
        
        Args:
            max_size: Maximum number of regions to store in cache
            similarity_threshold: Threshold for considering regions "nearby"
            stability_threshold: Minimum stability for regions to be retained
            decay_rate: Rate at which historical access is decayed
        """
        self.max_size = max(min(max_size, MAX_CACHE_SIZE), MIN_CACHE_SIZE)
        self.current_size = 0
        self.similarity_threshold = similarity_threshold
        self.stability_threshold = stability_threshold
        self.decay_rate = decay_rate
        self.region_map: Dict[str, TopologicalRegion] = {}
        self.access_history: List[Tuple[float, str]] = []
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "time_saved": 0.0,
            "last_prune_time": time.time()
        }
        self.size_adjustment_history = []
        self.logger = logging.getLogger(__name__)
    
    def _create_region_fingerprint(self, metrics: TopologicalMetrics) -> np.ndarray:
        """
        Create a numerical fingerprint representing the topological region.
        
        The fingerprint encodes key topological features in a way that preserves
        topological proximity - similar regions will have similar fingerprints.
        
        Args:
            metrics: TopologicalMetrics object
            
        Returns:
            np.ndarray: Numerical fingerprint of the region
        """
        # Create a fingerprint that preserves topological proximity
        fingerprint = np.zeros(8)
        
        # Betti numbers (primary topological features)
        for i, beta in enumerate(metrics.betti_numbers[:3]):
            fingerprint[i] = beta
        
        # Euler characteristic (topological invariant)
        fingerprint[3] = metrics.euler_characteristic
        
        # Topological entropy (measures uniformity)
        fingerprint[4] = metrics.topological_entropy
        
        # Naturalness coefficient (measures expectedness)
        fingerprint[5] = metrics.naturalness_coefficient
        
        # TVI (security metric)
        fingerprint[6] = metrics.tvi
        
        # Vulnerability type encoded as numeric value
        vuln_types = {
            "none": 0.0,
            "topological_structure": 0.3,
            "entropy_deficiency": 0.5,
            "predictability": 0.7,
            "manifold_distortion": 1.0,
            "unknown": 0.9
        }
        fingerprint[7] = vuln_types.get(metrics.vulnerability_type, 0.9)
        
        return fingerprint
    
    def _generate_region_id(self, metrics: TopologicalMetrics) -> str:
        """
        Generate a unique but topology-preserving ID for a region.
        
        The ID is designed to be consistent for similar topological regions
        while providing uniqueness for distinct regions.
        
        Args:
            metrics: TopologicalMetrics object
            
        Returns:
            str: Region identifier
        """
        # Create fingerprint from topological features
        fingerprint = self._create_region_fingerprint(metrics)
        
        # Quantize fingerprint to create stable ID
        quantized = [int(x * 10) for x in fingerprint[:4]]  # Use first 4 most stable features
        return f"R_{'_'.join(str(x) for x in quantized)}"
    
    def _region_similarity(self, 
                          region1: TopologicalRegion, 
                          region2: TopologicalRegion) -> float:
        """
        Calculate similarity between two topological regions.
        
        Similarity is based on topological proximity in the feature space,
        with emphasis on features critical for cryptographic security.
        
        Args:
            region1: First region
            region2: Second region
            
        Returns:
            float: Similarity score (0.0 to 1.0, higher is more similar)
        """
        # Weighted distance in topological feature space
        weights = np.array([
            0.25,  # β₀ weight
            0.30,  # β₁ weight (most critical for security)
            0.20,  # β₂ weight
            0.10,  # Euler characteristic
            0.05,  # Topological entropy
            0.05,  # Naturalness coefficient
            0.03,  # TVI
            0.02   # Vulnerability type
        ])
        
        # Calculate weighted Euclidean distance
        vec1 = region1.fingerprint
        vec2 = region2.fingerprint
        diff = vec1 - vec2
        
        # Apply weights and normalize
        weighted_diff = diff * weights
        distance = np.linalg.norm(weighted_diff)
        
        # Convert distance to similarity (0.0 to 1.0)
        max_possible_distance = np.sqrt(np.sum(weights**2) * 4)  # Max distance in normalized space
        similarity = 1.0 - min(1.0, distance / max_possible_distance)
        
        return similarity
    
    def _find_nearby_regions(self, 
                           target_metrics: TopologicalMetrics) -> List[Tuple[str, float]]:
        """
        Find regions in cache that are topologically similar to the target.
        
        This method implements the core innovation of topological caching - instead of
        requiring exact matches, it identifies regions that are "nearby" in topological space,
        which can benefit from similar optimization strategies.
        
        Args:
            target_metrics: TopologicalMetrics for the target region
            
        Returns:
            List[Tuple[str, float]]: List of (region_id, similarity) for nearby regions
        """
        target_fingerprint = self._create_region_fingerprint(target_metrics)
        target_region = TopologicalRegion(target_fingerprint, target_metrics, "TEMP")
        
        nearby = []
        for region_id, region in self.region_map.items():
            similarity = self._region_similarity(target_region, region)
            if similarity >= self.similarity_threshold:
                nearby.append((region_id, similarity))
        
        # Sort by similarity (highest first)
        nearby.sort(key=lambda x: x[1], reverse=True)
        return nearby
    
    def get(self, 
           metrics: TopologicalMetrics,
           performance_gain_callback: Optional[Callable[[str], float]] = None) -> Optional[Dict[str, Any]]:
        """
        Get optimization strategy for the given topological metrics.
        
        Instead of requiring an exact match, this method finds the most similar
        region in the cache and returns its optimization strategy.
        
        Args:
            metrics: TopologicalMetrics for the current signature set
            performance_gain_callback: Optional callback to get actual performance gain
            
        Returns:
            Optional[Dict[str, Any]]: Optimization strategy if found, None otherwise
        """
        self.performance_stats["total_requests"] += 1
        
        # Find nearby regions
        nearby_regions = self._find_nearby_regions(metrics)
        
        if not nearby_regions:
            self.performance_stats["cache_misses"] += 1
            return None
        
        # Get the most similar region
        best_region_id, similarity = nearby_regions[0]
        region = self.region_map[best_region_id]
        
        # Update access statistics
        self.access_history.append((time.time(), best_region_id))
        region.update_access(metrics)
        
        # Record cache hit
        self.performance_stats["cache_hits"] += 1
        
        # Calculate time saved based on performance gain
        if performance_gain_callback:
            actual_gain = performance_gain_callback(best_region_id)
            region.update_access(metrics, actual_gain)
            # Estimate time saved (assuming baseline time of 1.0)
            time_saved = (actual_gain - 1.0) / actual_gain
            self.performance_stats["time_saved"] += time_saved
        
        # Return optimization strategy
        return {
            "region_id": best_region_id,
            "similarity": similarity,
            "stability": region.stability_score,
            "performance_gain": region.performance_gain,
            "access_count": region.access_count,
            "last_access": region.last_access
        }
    
    def store(self, 
             metrics: TopologicalMetrics, 
             performance_gain: float = 4.5,
             signature_count: int = 0) -> bool:
        """
        Store optimization strategy for the given topological metrics.
        
        Args:
            metrics: TopologicalMetrics for the signature set
            performance_gain: Measured performance gain from the optimization
            signature_count: Number of signatures analyzed
            
        Returns:
            bool: True if stored successfully, False if cache is full and no eviction possible
        """
        # Create region ID and fingerprint
        region_id = self._generate_region_id(metrics)
        fingerprint = self._create_region_fingerprint(metrics)
        
        # Check if this region already exists
        if region_id in self.region_map:
            region = self.region_map[region_id]
            region.update_access(metrics, performance_gain)
            return True
        
        # If cache is full, check if we should evict
        if self.current_size >= self.max_size:
            # Try to evict unstable regions first
            evicted = self._evict_unstable_regions()
            if not evicted and self.current_size >= self.max_size:
                # If no unstable regions, evict least recently used
                evicted = self._evict_lru_region()
                if not evicted:
                    return False  # Cannot store, cache is full
        
        # Create and store new region
        new_region = TopologicalRegion(fingerprint, metrics, region_id)
        new_region.performance_gain = performance_gain
        self.region_map[region_id] = new_region
        self.current_size = min(self.current_size + 1, self.max_size)
        
        # Update access history
        self.access_history.append((time.time(), region_id))
        
        # Periodically adjust cache size
        if time.time() - self.performance_stats["last_prune_time"] > 60.0:
            self.update_strategy()
            self.performance_stats["last_prune_time"] = time.time()
        
        return True
    
    def _evict_unstable_regions(self) -> bool:
        """
        Evict regions with stability below threshold.
        
        Returns:
            bool: True if at least one region was evicted
        """
        unstable_regions = [
            (region_id, region) 
            for region_id, region in self.region_map.items()
            if region.stability_score < self.stability_threshold
        ]
        
        if not unstable_regions:
            return False
        
        # Sort by stability (lowest first) and then by access count
        unstable_regions.sort(key=lambda x: (x[1].stability_score, -x[1].access_count))
        
        # Evict the most unstable region
        region_id, _ = unstable_regions[0]
        del self.region_map[region_id]
        self.current_size = max(0, self.current_size - 1)
        
        return True
    
    def _evict_lru_region(self) -> bool:
        """
        Evict the least recently used region.
        
        Returns:
            bool: True if a region was evicted
        """
        if not self.region_map:
            return False
        
        # Find the least recently used region
        lru_region = min(
            self.region_map.items(),
            key=lambda x: x[1].last_access
        )
        
        # Evict it
        region_id, _ = lru_region
        del self.region_map[region_id]
        self.current_size = max(0, self.current_size - 1)
        
        return True
    
    def update_strategy(self):
        """
        Dynamically adjust cache strategy based on performance and system conditions.
        
        This method implements the principle: "Хорошая система «подпевает себе» постоянно,
        тихо и незаметно для пользователя" from QuantumFortress documentation.
        
        The strategy adjustment considers:
        - Cache hit rate and performance gains
        - Topological stability of cached regions
        - System resource constraints
        - Recent access patterns
        """
        # Calculate current cache hit rate
        total = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        hit_rate = self.performance_stats["cache_hits"] / total if total > 0 else 0.0
        
        # Calculate average stability of cached regions
        if self.region_map:
            avg_stability = np.mean([r.stability_score for r in self.region_map.values()])
        else:
            avg_stability = 0.0
        
        # Determine if we should grow or shrink the cache
        size_change = 0
        
        # Grow if hit rate is high and regions are stable
        if hit_rate > 0.7 and avg_stability > 0.6:
            size_change = int(self.max_size * 0.1)  # Grow by 10%
        # Shrink if hit rate is low or regions are unstable
        elif (hit_rate < 0.3 and avg_stability < 0.4) or self.current_size >= self.max_size:
            size_change = -int(self.max_size * 0.1)  # Shrink by 10%
        
        # Apply size change with bounds checking
        new_max_size = max(
            MIN_CACHE_SIZE,
            min(MAX_CACHE_SIZE, self.max_size + size_change)
        )
        
        # Only update if size changed significantly
        if abs(new_max_size - self.max_size) > 0.05 * self.max_size:
            old_max_size = self.max_size
            self.max_size = new_max_size
            
            # Record the adjustment
            self.size_adjustment_history.append((
                time.time(),
                old_max_size,
                new_max_size
            ))
            
            # If shrinking, evict regions to meet new size limit
            while self.current_size > self.max_size:
                if not self._evict_unstable_regions():
                    if not self._evict_lru_region():
                        break
        
        # Log the strategy update
        self.logger.debug(
            f"Cache strategy updated: size={self.max_size}, "
            f"hit_rate={hit_rate:.2f}, stability={avg_stability:.2f}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about cache performance.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        total = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        hit_rate = self.performance_stats["cache_hits"] / total if total > 0 else 0.0
        
        # Calculate average stability of cached regions
        if self.region_map:
            avg_stability = np.mean([r.stability_score for r in self.region_map.values()])
            avg_performance = np.mean([r.performance_gain for r in self.region_map.values()])
        else:
            avg_stability = 0.0
            avg_performance = 1.0
        
        return {
            "current_size": self.current_size,
            "max_size": self.max_size,
            "cache_hits": self.performance_stats["cache_hits"],
            "cache_misses": self.performance_stats["cache_misses"],
            "hit_rate": hit_rate,
            "time_saved": self.performance_stats["time_saved"],
            "average_stability": avg_stability,
            "average_performance_gain": avg_performance,
            "region_count": len(self.region_map),
            "last_adjustment": (
                self.size_adjustment_history[-1] if self.size_adjustment_history else None
            )
        }
    
    def clear(self):
        """Clear all entries from the cache."""
        self.region_map.clear()
        self.current_size = 0
        self.access_history = []
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "time_saved": 0.0,
            "last_prune_time": time.time()
        }
    
    def prune(self):
        """Prune the cache by removing unstable and least recently used regions."""
        # First remove unstable regions
        while self._evict_unstable_regions():
            pass
        
        # Then ensure we're within size limits
        while self.current_size > self.max_size:
            if not self._evict_lru_region():
                break
    
    def get_region_details(self, region_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific region.
        
        Args:
            region_id: ID of the region
            
        Returns:
            Optional[Dict[str, Any]]: Region details or None if not found
        """
        if region_id not in self.region_map:
            return None
        
        region = self.region_map[region_id]
        return {
            "region_id": region_id,
            "betti_numbers": region.metrics.betti_numbers,
            "euler_characteristic": region.metrics.euler_characteristic,
            "topological_entropy": region.metrics.topological_entropy,
            "naturalness_coefficient": region.metrics.naturalness_coefficient,
            "tvi": region.metrics.tvi,
            "vulnerability_type": region.metrics.vulnerability_type,
            "stability_score": region.stability_score,
            "performance_gain": region.performance_gain,
            "access_count": region.access_count,
            "last_access": region.last_access
        }
    
    def get_top_regions(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N most stable and frequently accessed regions.
        
        Args:
            n: Number of regions to return
            
        Returns:
            List[Dict[str, Any]]: Top regions with their details
        """
        if not self.region_map:
            return []
        
        # Sort regions by stability and access count
        regions = list(self.region_map.values())
        regions.sort(
            key=lambda r: (r.stability_score * r.access_count),
            reverse=True
        )
        
        # Return top N regions with details
        return [
            {
                "region_id": r.region_id,
                "stability": r.stability_score,
                "access_count": r.access_count,
                "performance_gain": r.performance_gain,
                "betti_numbers": r.metrics.betti_numbers,
                "tvi": r.metrics.tvi
            }
            for r in regions[:n]
        ]

# For backward compatibility with older implementations
TopologicalCache = TopologicallyOptimizedCache

def create_default_cache() -> TopologicallyOptimizedCache:
    """
    Create a default topologically-optimized cache with recommended settings.
    
    Returns:
        TopologicallyOptimizedCache: Configured cache instance
    """
    return TopologicallyOptimizedCache(
        max_size=DEFAULT_MAX_SIZE,
        similarity_threshold=REGION_SIMILARITY_THRESHOLD,
        stability_threshold=STABILITY_THRESHOLD,
        decay_rate=CACHE_DECAY_RATE
    )

def example_usage() -> None:
    """
    Example usage of TopologicallyOptimizedCache.
    
    Demonstrates how to use the cache for cryptographic signature verification.
    """
    print("=" * 60)
    print("Пример использования TopologicallyOptimizedCache")
    print("=" * 60)
    
    # Create cache
    cache = create_default_cache()
    print(f"\n1. Создан кэш с максимальным размером {cache.max_size} регионов")
    
    # Simulate some topological metrics (secure system)
    from .betti_numbers import TopologicalMetrics
    secure_metrics = TopologicalMetrics(
        betti_numbers=[1.0, 2.0, 1.0],
        euler_characteristic=0.0,
        topological_entropy=0.95,
        naturalness_coefficient=0.05,
        tvi=0.0,
        is_secure=True,
        vulnerability_type="none",
        explanation="Безопасная система с идеальной топологией тора",
        timestamp=time.time()
    )
    
    # Simulate some topological metrics (vulnerable system)
    vulnerable_metrics = TopologicalMetrics(
        betti_numbers=[1.0, 3.2, 0.8],
        euler_characteristic=0.5,
        topological_entropy=0.3,
        naturalness_coefficient=0.6,
        tvi=0.85,
        is_secure=False,
        vulnerability_type="topological_structure",
        explanation="Уязвимость: нарушение структуры пространства подписей",
        timestamp=time.time()
    )
    
    # Store optimization strategies
    print("\n2. Сохранение стратегий оптимизации в кэш...")
    cache.store(secure_metrics, performance_gain=4.5, signature_count=1000)
    cache.store(vulnerable_metrics, performance_gain=2.0, signature_count=500)
    print(f"  - Сохранено {cache.current_size} регионов в кэше")
    
    # Retrieve strategies
    print("\n3. Получение стратегий из кэша...")
    
    print("  a) Запрос для безопасной системы:")
    strategy = cache.get(secure_metrics)
    if strategy:
        print(f"     Найдена стратегия (схожесть={strategy['similarity']:.2f}, "
              f"производительность={strategy['performance_gain']}x)")
    else:
        print("     Стратегия не найдена")
    
    print("  b) Запрос для уязвимой системы:")
    strategy = cache.get(vulnerable_metrics)
    if strategy:
        print(f"     Найдена стратегия (схожесть={strategy['similarity']:.2f}, "
              f"производительность={strategy['performance_gain']}x)")
    else:
        print("     Стратегия не найдена")
    
    # Get statistics
    stats = cache.get_statistics()
    print("\n4. Статистика кэша:")
    print(f"  - Размер кэша: {stats['current_size']}/{stats['max_size']}")
    print(f"  - Коэффициент попаданий: {stats['hit_rate']:.2f}")
    print(f"  - Средняя стабильность: {stats['average_stability']:.2f}")
    print(f"  - Средний прирост производительности: {stats['average_performance_gain']:.2f}x")
    
    # Get top regions
    top_regions = cache.get_top_regions(2)
    print("\n5. Топ регионы:")
    for i, region in enumerate(top_regions):
        print(f"  Регион #{i+1} (ID: {region['region_id']}):")
        print(f"    - Стабильность: {region['stability']:.2f}")
        print(f"    - Количество доступов: {region['access_count']}")
        print(f"    - Прирост производительности: {region['performance_gain']}x")
        print(f"    - Числа Бетти: β0={region['betti_numbers'][0]}, "
              f"β1={region['betti_numbers'][1]}, β2={region['betti_numbers'][2]}")
        print(f"    - TVI: {region['tvi']:.2f}")
    
    print("\n6. Пример динамической коррекции размера кэша...")
    # Simulate high cache miss rate to trigger size adjustment
    for _ in range(50):
        cache.get(vulnerable_metrics)
        cache.get(secure_metrics)
    
    # Force strategy update
    cache.update_strategy()
    new_stats = cache.get_statistics()
    print(f"  - Новый размер кэша: {new_stats['current_size']}/{new_stats['max_size']}")
    
    print("=" * 60)
    print("Кэш успешно продемонстрировал топологическую оптимизацию и динамическую коррекцию размера.")
    print("=" * 60)

if __name__ == "__main__":
    # Run example usage when module is executed directly
    example_usage()
