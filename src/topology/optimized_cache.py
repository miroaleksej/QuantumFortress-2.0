"""
optimized_cache.py - Topologically-Optimized Cache for QuantumFortress 2.0

This module implements a cache system that uses topological analysis to optimize
data retrieval and storage. Unlike traditional caches that rely on temporal or
spatial locality, this system leverages the topological structure of the data
space to make intelligent caching decisions.

The implementation is based on the principles from "Ur Uz работа.md" and
"Prototype_TopoMine.txt", which describe how topological properties can be used
to optimize blockchain operations, particularly for signature verification and
nonce search.

Key features:
- Dynamic cache strategy based on topological metrics (Betti numbers, curvature)
- Region-based caching using torus topology of ECDSA signatures
- WDM-parallelized cache access for improved performance
- Self-adjusting cache size based on topological stability
- Integration with TVI (Topological Vulnerability Index) for security-aware caching

Author: Quantum Topology Research Group
Institution: Tambov Research Institute of Quantum Topology
Email: miro-aleksej@yandex.ru
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Set, TypeVar, Generic
from collections import OrderedDict, deque
import hashlib
import math
from dataclasses import dataclass

# Configure module-specific logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type variables for generic cache
K = TypeVar('K')
V = TypeVar('V')

@dataclass
class TopologicalRegion:
    """
    Represents a topological region in the cache space.
    
    Based on the region analysis in Prototype_TopoMine.txt:
    "Generate a unique region ID based on topological features"
    """
    region_id: str
    betti_numbers: List[float]
    curvature: float
    spiral_strength: float
    spiral_direction: str
    symmetry: Dict[str, float]
    density: float
    access_count: int = 0
    last_access: float = 0.0
    tvi: float = 0.0

@dataclass
class CacheEntry(Generic[K, V]):
    """
    Represents a single entry in the topologically-optimized cache.
    """
    key: K
    value: V
    region_id: str
    timestamp: float
    access_count: int = 0
    tvi: float = 0.0
    topological_metrics: Dict[str, Any] = None

class TopologicalCacheMetrics:
    """
    Tracks performance metrics for the topological cache.
    
    Based on metrics described in Prototype_TopoMine.txt and TopoMine_Validation.txt.
    """
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_accesses = 0
        self.hit_ratio = 0.0
        self.miss_ratio = 0.0
        self.eviction_ratio = 0.0
        self.topological_hits = 0
        self.topological_misses = 0
        self.topological_hit_ratio = 0.0
        self.avg_response_time = 0.0
        self.last_update = time.time()
        self.region_distribution = {}
        self.tvi_distribution = {
            "secure": 0,
            "warning": 0,
            "critical": 0
        }
        
    def update(self, is_hit: bool, is_topological: bool, response_time: float, 
              region_id: str, tvi: float) -> None:
        """Update cache metrics based on access."""
        self.total_accesses += 1
        self.hits += 1 if is_hit else 0
        self.misses += 0 if is_hit else 1
        self.topological_hits += 1 if (is_hit and is_topological) else 0
        self.topological_misses += 0 if (is_hit and is_topological) else 1
        
        # Update hit ratios
        if self.total_accesses > 0:
            self.hit_ratio = self.hits / self.total_accesses
            self.miss_ratio = self.misses / self.total_accesses
            
        if self.hits > 0:
            self.topological_hit_ratio = self.topological_hits / self.hits
            
        # Update response time
        self.avg_response_time = (self.avg_response_time * (self.total_accesses - 1) + response_time) / self.total_accesses
        
        # Update region distribution
        if region_id not in self.region_distribution:
            self.region_distribution[region_id] = 0
        self.region_distribution[region_id] += 1
        
        # Update TVI distribution
        if tvi < 0.3:
            self.tvi_distribution["secure"] += 1
        elif tvi < 0.6:
            self.tvi_distribution["warning"] += 1
        else:
            self.tvi_distribution["critical"] += 1
            
        self.last_update = time.time()
    
    def get_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        total_tvi = sum(self.tvi_distribution.values()) or 1
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "topological_hit_ratio": self.topological_hit_ratio,
            "avg_response_time": self.avg_response_time,
            "region_distribution": self.region_distribution,
            "tvi_distribution": {
                k: v / total_tvi for k, v in self.tvi_distribution.items()
            },
            "last_update": self.last_update
        }

class TopologicallyOptimizedCache(Generic[K, V]):
    """
    Topologically-Optimized Cache for QuantumFortress 2.0.
    
    This cache uses topological analysis of data to optimize storage and retrieval.
    Instead of traditional LRU or LFU algorithms, it uses:
    - Topological proximity (based on Betti numbers and curvature)
    - TVI-based priority (lower TVI = higher priority)
    - WDM-parallelized access patterns
    
    As described in Prototype_TopoMine.txt:
    "TopoMine warning: Topological analysis failed: {str(e)}"
    
    The cache is designed to work with the torus topology of ECDSA signatures,
    where points are represented in (u_r, u_z) space.
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 tvi_threshold: float = 0.5,
                 region_threshold: int = 5,
                 min_cache_size: int = 100,
                 max_cache_size: int = 10000):
        """
        Initialize the topologically-optimized cache.
        
        Args:
            max_size: Maximum number of items in cache
            tvi_threshold: Threshold for TVI to prioritize secure regions
            region_threshold: Minimum items per region before optimization
            min_cache_size: Minimum cache size for dynamic adjustment
            max_cache_size: Maximum cache size for dynamic adjustment
        """
        self.max_size = max_size
        self.tvi_threshold = tvi_threshold
        self.region_threshold = region_threshold
        self.min_cache_size = min_cache_size
        self.max_cache_size = max_cache_size
        
        # Main cache storage - OrderedDict for LRU behavior within regions
        self.cache: Dict[K, CacheEntry[K, V]] = {}
        
        # Region mapping - tracks which items belong to which topological region
        self.region_map: Dict[str, Set[K]] = {}
        
        # Region metadata - information about each topological region
        self.regions: Dict[str, TopologicalRegion] = {}
        
        # Metrics tracking
        self.metrics = TopologicalCacheMetrics()
        
        # History for dynamic adjustment
        self.access_history = deque(maxlen=1000)
        self.size_adjustment_history = []
        
        # Timestamp for last size adjustment
        self.last_size_adjustment = time.time()
        
        logger.info(f"Initialized topologically-optimized cache with max_size={max_size}")
    
    def _generate_region_id(self, topology_metrics: Dict[str, Any]) -> str:
        """
        Generate a unique region ID based on topological features.
        
        Based on Prototype_TopoMine.txt:
        "Generate a unique region ID based on topological features"
        
        Args:
            topology_metrics: Dictionary of topological metrics
            
        Returns:
            str: Unique region ID
        """
        # Extract key features
        betti = topology_metrics.get("betti_numbers", [0.0, 0.0, 0.0])
        curvature = topology_metrics.get("curvature", 0.0)
        spiral = topology_metrics.get("spiral_analysis", {})
        symmetry = topology_metrics.get("symmetry", {})
        
        # Create fingerprint from Betti numbers (as in Prototype_TopoMine.txt)
        betti_str = "_".join([f"{b:.2f}" for b in betti[:3]])
        curvature_str = f"{curvature:.3f}"
        
        # Get spiral direction
        spiral_dir = spiral.get("direction", "none")
        spiral_strength = f"{spiral.get('strength', 0.0):.3f}"
        
        # Get symmetry values
        sym_h = f"{symmetry.get('horizontal', 0.0):.3f}"
        sym_v = f"{symmetry.get('vertical', 0.0):.3f}"
        sym_d = f"{symmetry.get('diagonal', 0.0):.3f}"
        
        # Combine into hashable string
        region_str = f"{betti_str}|{curvature_str}|{spiral_dir}|{spiral_strength}|{sym_h}|{sym_v}|{sym_d}"
        
        # Return hash for compactness
        return hashlib.md5(region_str.encode()).hexdigest()[:16]
    
    def _find_nearby_regions(self, region_id: str, max_distance: float = 0.5) -> List[str]:
        """
        Find regions that are topologically close to the given region.
        
        Args:
            region_id: ID of the reference region
            max_distance: Maximum topological distance to consider
            
        Returns:
            List[str]: List of nearby region IDs
        """
        if region_id not in self.regions:
            return []
        
        reference_region = self.regions[region_id]
        nearby_regions = []
        
        for reg_id, region in self.regions.items():
            if reg_id == region_id:
                continue
                
            # Calculate topological distance between regions
            distance = self._calculate_region_distance(reference_region, region)
            
            if distance <= max_distance:
                nearby_regions.append((reg_id, distance))
        
        # Sort by distance and return just the IDs
        nearby_regions.sort(key=lambda x: x[1])
        return [reg_id for reg_id, _ in nearby_regions]
    
    def _calculate_region_distance(self, region1: TopologicalRegion, 
                                region2: TopologicalRegion) -> float:
        """
        Calculate topological distance between two regions.
        
        Args:
            region1: First region
            region2: Second region
            
        Returns:
            float: Topological distance (0.0 to 1.0)
        """
        # Betti number distance
        betti_dist = 0.0
        for i in range(min(len(region1.betti_numbers), len(region2.betti_numbers))):
            expected = 2.0 if i == 1 else 1.0  # For torus topology
            diff1 = abs(region1.betti_numbers[i] - expected)
            diff2 = abs(region2.betti_numbers[i] - expected)
            betti_dist += abs(diff1 - diff2) / (expected + 1e-10)
        
        # Curvature distance
        curvature_dist = abs(region1.curvature - region2.curvature)
        
        # Spiral strength distance
        spiral_dist = abs(region1.spiral_strength - region2.spiral_strength)
        
        # Symmetry distance
        sym_dist = 0.0
        for key in region1.symmetry:
            if key in region2.symmetry:
                sym_dist += abs(region1.symmetry[key] - region2.symmetry[key])
        sym_dist /= max(1, len(region1.symmetry))
        
        # TVI distance
        tvi_dist = abs(region1.tvi - region2.tvi)
        
        # Weighted combination
        weights = {
            "betti": 0.3,
            "curvature": 0.2,
            "spiral": 0.1,
            "symmetry": 0.2,
            "tvi": 0.2
        }
        
        total_dist = (weights["betti"] * betti_dist +
                     weights["curvature"] * curvature_dist +
                     weights["spiral"] * spiral_dist +
                     weights["symmetry"] * sym_dist +
                     weights["tvi"] * tvi_dist)
        
        # Normalize to [0,1]
        return min(1.0, total_dist)
    
    def _update_region(self, region_id: str, topology_metrics: Dict[str, Any], 
                      tvi: float) -> None:
        """
        Update or create a topological region.
        
        Args:
            region_id: ID of the region
            topology_metrics: Metrics describing the region
            tvi: Topological Vulnerability Index for the region
        """
        # Extract metrics
        betti = topology_metrics.get("betti_numbers", [0.0, 0.0, 0.0])
        curvature = topology_metrics.get("curvature", 0.0)
        spiral = topology_metrics.get("spiral_analysis", {})
        symmetry = topology_metrics.get("symmetry", {})
        
        # Create or update region
        if region_id not in self.regions:
            self.regions[region_id] = TopologicalRegion(
                region_id=region_id,
                betti_numbers=betti,
                curvature=curvature,
                spiral_strength=spiral.get("strength", 0.0),
                spiral_direction=spiral.get("direction", "none"),
                symmetry=symmetry,
                density=0.0,
                tvi=tvi
            )
        else:
            region = self.regions[region_id]
            region.betti_numbers = betti
            region.curvature = curvature
            region.spiral_strength = spiral.get("strength", 0.0)
            region.spiral_direction = spiral.get("direction", "none")
            region.symmetry = symmetry
            region.tvi = tvi
    
    def _adjust_cache_size(self) -> None:
        """
        Dynamically adjust cache size based on usage patterns and topological stability.
        
        Implements the principle from auto_calibration.txt:
        "Хорошая система «подпевает себе» постоянно, тихо и незаметно для пользователя"
        """
        current_time = time.time()
        
        # Only adjust size periodically
        if current_time - self.last_size_adjustment < 5.0:  # 5 seconds
            return
            
        # Calculate current hit ratio
        hit_ratio = self.metrics.hit_ratio if self.metrics.total_accesses > 0 else 0.0
        
        # Base adjustment on hit ratio and TVI distribution
        tvi_distribution = self.metrics.tvi_distribution
        total = sum(tvi_distribution.values()) or 1
        secure_ratio = tvi_distribution["secure"] / total
        
        # Determine desired size
        desired_size = self.max_size
        
        # Increase size if hit ratio is high and system is secure
        if hit_ratio > 0.8 and secure_ratio > 0.7:
            desired_size = min(self.max_cache_size, int(self.max_size * 1.2))
        # Decrease size if hit ratio is low or system is vulnerable
        elif hit_ratio < 0.5 or secure_ratio < 0.3:
            desired_size = max(self.min_cache_size, int(self.max_size * 0.8))
            
        # Apply gradual adjustment
        adjustment = (desired_size - self.max_size) * 0.3  # 30% of desired change
        new_size = int(self.max_size + adjustment)
        new_size = max(self.min_cache_size, min(self.max_cache_size, new_size))
        
        # Apply the change
        if new_size != self.max_size:
            logger.debug(f"Adjusting cache size from {self.max_size} to {new_size}")
            self.max_size = new_size
            
            # If reducing size, evict items
            if new_size < len(self.cache):
                self._evict_to_size(new_size)
                
            # Record adjustment
            self.size_adjustment_history.append((time.time(), self.max_size))
            self.last_size_adjustment = current_time
    
    def _evict_to_size(self, target_size: int) -> None:
        """
        Evict items from cache until size is at or below target.
        
        Uses a combination of:
        - TVI-based priority (higher TVI = evict first)
        - Least recently used within regions
        - Region density (less dense regions get evicted first)
        
        Args:
            target_size: Target cache size
        """
        if len(self.cache) <= target_size:
            return
            
        # Create list of items to potentially evict
        evict_candidates = []
        for key, entry in self.cache.items():
            # Priority score: lower = more likely to be evicted
            # Higher TVI, older timestamp, lower region density = higher priority to evict
            region = self.regions.get(entry.region_id)
            region_density = region.density if region else 0.0
            
            priority = (
                entry.tvi * 10.0 +  # TVI has highest weight
                (time.time() - entry.timestamp) * 0.1 +  # Age
                (1.0 - region_density) * 5.0  # Inverse of density
            )
            
            evict_candidates.append((priority, entry.timestamp, key))
        
        # Sort by priority (lowest priority first) and timestamp (oldest first)
        evict_candidates.sort()
        
        # Evict until we reach target size
        evictions_needed = len(self.cache) - target_size
        for i in range(min(evictions_needed, len(evict_candidates))):
            _, _, key = evict_candidates[i]
            self._remove_key(key)
            self.metrics.evictions += 1
    
    def _remove_key(self, key: K) -> None:
        """Remove a key from the cache and update region mappings."""
        if key not in self.cache:
            return
            
        entry = self.cache[key]
        
        # Remove from region mapping
        if entry.region_id in self.region_map:
            self.region_map[entry.region_id].discard(key)
            if not self.region_map[entry.region_id]:
                del self.region_map[entry.region_id]
                
        # Remove from cache
        del self.cache[key]
    
    def update_strategy(self, topology_data: Dict[str, Any]) -> None:
        """
        Update cache strategy based on current topological data.
        
        Args:
            topology_data: Dictionary containing current topological metrics
            
        Example from Prototype_TopoMine.txt:
        "Update cache strategy based on current topological metrics"
        """
        start_time = time.time()
        
        try:
            # Generate region ID from topology data
            region_id = self._generate_region_id(topology_data)
            
            # Update or create the region
            tvi = topology_data.get("tvi", 0.0)
            self._update_region(region_id, topology_data, tvi)
            
            # Update region density based on current cache state
            if region_id in self.region_map:
                region_size = len(self.region_map[region_id])
                total_items = len(self.cache)
                self.regions[region_id].density = region_size / max(1, total_items)
                
            # Adjust cache size based on usage patterns
            self._adjust_cache_size()
            
            # Log performance
            duration = time.time() - start_time
            logger.debug(f"Cache strategy updated in {duration:.4f}s for region {region_id}")
            
        except Exception as e:
            logger.error(f"Failed to update cache strategy: {str(e)}")
            raise
    
    def get(self, key: K) -> Optional[V]:
        """
        Get an item from the cache with topological awareness.
        
        Instead of just checking for the key, this method also looks for
        topologically similar items in nearby regions.
        
        Args:
            key: Key to look up
            
        Returns:
            Optional[V]: Value if found, None otherwise
            
        Example from Prototype_TopoMine.txt:
        "Use fast verification for known regions"
        """
        start_time = time.time()
        
        # First check if the exact key exists
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            self.metrics.update(
                is_hit=True,
                is_topological=False,
                response_time=time.time() - start_time,
                region_id=entry.region_id,
                tvi=entry.tvi
            )
            return entry.value
        
        # If not found, check nearby regions for topologically similar items
        if hasattr(key, 'topology_metrics') and key.topology_metrics:
            region_id = self._generate_region_id(key.topology_metrics)
            
            # Find nearby regions
            nearby_regions = self._find_nearby_regions(region_id)
            
            # Look for similar items in nearby regions
            for reg_id in nearby_regions:
                if reg_id in self.region_map:
                    for similar_key in self.region_map[reg_id]:
                        # Check if this is a topological match
                        if self._is_topologically_similar(key, similar_key):
                            entry = self.cache[similar_key]
                            entry.access_count += 1
                            self.metrics.update(
                                is_hit=True,
                                is_topological=True,
                                response_time=time.time() - start_time,
                                region_id=reg_id,
                                tvi=entry.tvi
                            )
                            return entry.value
        
        # Cache miss
        self.metrics.update(
            is_hit=False,
            is_topological=False,
            response_time=time.time() - start_time,
            region_id="",
            tvi=1.0
        )
        return None
    
    def _is_topologically_similar(self, key1: K, key2: K) -> bool:
        """
        Check if two keys are topologically similar.
        
        Args:
            key1: First key
            key2: Second key
            
        Returns:
            bool: True if keys are topologically similar
        """
        # This would be implementation-specific
        # For signatures, we might compare their (u_r, u_z) positions
        try:
            # Example for ECDSA signatures
            if hasattr(key1, 'ur') and hasattr(key1, 'uz') and \
               hasattr(key2, 'ur') and hasattr(key2, 'uz'):
                # Calculate torus distance
                d_ur = min(abs(key1.ur - key2.ur), 1.0 - abs(key1.ur - key2.ur))
                d_uz = min(abs(key1.uz - key2.uz), 1.0 - abs(key1.uz - key2.uz))
                distance = math.sqrt(d_ur**2 + d_uz**2)
                return distance < 0.2  # Threshold for similarity
            
            # Default comparison
            return key1 == key2
            
        except Exception as e:
            logger.debug(f"Error in topological similarity check: {str(e)}")
            return False
    
    def put(self, key: K, value: V, topology_metrics: Dict[str, Any], 
           tvi: float = 0.0) -> None:
        """
        Add an item to the cache with topological information.
        
        Args:
            key: Key to store
            value: Value to store
            topology_metrics: Topological metrics for the item
            tvi: Topological Vulnerability Index
            
        Example from Prototype_TopoMine.txt:
        "Store metrics in block for later use"
        """
        # Generate region ID
        region_id = self._generate_region_id(topology_metrics)
        
        # Update or create the region
        self._update_region(region_id, topology_metrics, tvi)
        
        # Add to region mapping
        if region_id not in self.region_map:
            self.region_map[region_id] = set()
        self.region_map[region_id].add(key)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            region_id=region_id,
            timestamp=time.time(),
            tvi=tvi,
            topological_metrics=topology_metrics
        )
        
        # Add to cache
        self.cache[key] = entry
        
        # Ensure we don't exceed max size
        if len(self.cache) > self.max_size:
            self._evict_to_size(self.max_size)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.region_map.clear()
        self.regions.clear()
        logger.info("Cache cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current cache metrics."""
        return self.metrics.get_report()
    
    def get_region_info(self, region_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific topological region.
        
        Args:
            region_id: ID of the region
            
        Returns:
            Optional[Dict[str, Any]]: Region information or None if not found
        """
        if region_id not in self.regions:
            return None
            
        region = self.regions[region_id]
        return {
            "region_id": region.region_id,
            "betti_numbers": region.betti_numbers,
            "curvature": region.curvature,
            "spiral_analysis": {
                "strength": region.spiral_strength,
                "direction": region.spiral_direction
            },
            "symmetry": region.symmetry,
            "density": region.density,
            "access_count": region.access_count,
            "tvi": region.tvi,
            "item_count": len(self.region_map.get(region_id, []))
        }
    
    def get_nearby_regions(self, region_id: str, max_distance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get information about regions near the specified region.
        
        Args:
            region_id: ID of the reference region
            max_distance: Maximum topological distance to consider
            
        Returns:
            List[Dict[str, Any]]: List of nearby region information
        """
        nearby_ids = self._find_nearby_regions(region_id, max_distance)
        return [self.get_region_info(rid) for rid in nearby_ids if self.get_region_info(rid)]
    
    def get_optimal_cache_size(self) -> int:
        """
        Get the currently optimal cache size based on usage patterns.
        
        Returns:
            int: Optimal cache size
        """
        # This would be calculated based on historical performance
        if not self.size_adjustment_history:
            return self.max_size
            
        # Simple average of recent adjustments
        recent_sizes = [size for _, size in self.size_adjustment_history[-10:]]
        return int(sum(recent_sizes) / len(recent_sizes))
    
    def get_wdm_access_pattern(self) -> Dict[str, float]:
        """
        Get WDM (Wavelength Division Multiplexing) access pattern information.
        
        Returns:
            Dict[str, float]: Access frequencies by region
        """
        total_accesses = sum(self.metrics.region_distribution.values()) or 1
        return {region_id: count / total_accesses 
                for region_id, count in self.metrics.region_distribution.items()}
    
    def get_topological_hit_ratio(self) -> float:
        """
        Get the ratio of hits that were found through topological similarity.
        
        Returns:
            float: Topological hit ratio
        """
        total_hits = self.metrics.hits or 1
        return self.metrics.topological_hits / total_hits
    
    def get_tvi_aware_priority(self, key: K) -> float:
        """
        Get a priority score for a key based on TVI and other factors.
        
        Lower TVI = higher priority (more secure = more valuable to cache)
        
        Args:
            key: Key to evaluate
            
        Returns:
            float: Priority score (lower = higher priority)
        """
        if key not in self.cache:
            return 1.0
            
        entry = self.cache[key]
        region = self.regions.get(entry.region_id)
        
        # Priority factors:
        # - TVI (lower is better)
        # - Region density (higher is better)
        # - Access count (higher is better)
        # - Recency (more recent is better)
        
        tvi_factor = entry.tvi
        region_factor = 1.0 - (region.density if region else 0.0)
        access_factor = 1.0 / (entry.access_count + 1)
        time_factor = 1.0 / (time.time() - entry.timestamp + 1)
        
        # Weighted combination
        weights = {
            "tvi": 0.4,
            "region": 0.2,
            "access": 0.2,
            "time": 0.2
        }
        
        return (weights["tvi"] * tvi_factor +
                weights["region"] * region_factor +
                weights["access"] * access_factor +
                weights["time"] * time_factor)
    
    def optimize_for_tvi(self, target_tvi: float = 0.3) -> None:
        """
        Optimize cache for regions with TVI below the target.
        
        Args:
            target_tvi: Target TVI threshold
        """
        # Identify regions to prioritize
        secure_regions = [rid for rid, region in self.regions.items() 
                         if region.tvi <= target_tvi]
        
        if not secure_regions:
            return
            
        # Calculate how many items we can keep from secure regions
        secure_items = []
        for region_id in secure_regions:
            if region_id in self.region_map:
                for key in self.region_map[region_id]:
                    secure_items.append(key)
        
        # If we have more secure items than cache size, keep the best ones
        if len(secure_items) > self.max_size:
            # Sort by priority
            secure_items.sort(key=lambda key: self.get_tvi_aware_priority(key))
            items_to_keep = set(secure_items[:self.max_size])
            
            # Remove non-priority items
            for key in list(self.cache.keys()):
                if key not in items_to_keep:
                    self._remove_key(key)
        else:
            # Keep all secure items, evict from vulnerable regions
            vulnerable_keys = []
            for key, entry in self.cache.items():
                region = self.regions.get(entry.region_id)
                if not region or region.tvi > target_tvi:
                    vulnerable_keys.append(key)
            
            # Sort vulnerable keys by priority (lower priority = evict first)
            vulnerable_keys.sort(key=lambda key: -self.get_tvi_aware_priority(key))
            
            # Evict enough vulnerable items
            evictions_needed = len(self.cache) + len(secure_items) - self.max_size
            for i in range(min(evictions_needed, len(vulnerable_keys))):
                self._remove_key(vulnerable_keys[i])
    
    def get_cache_efficiency(self) -> float:
        """
        Calculate cache efficiency based on topological metrics.
        
        Returns:
            float: Efficiency score (0.0 to 1.0)
        """
        metrics = self.get_metrics()
        
        # Base efficiency on hit ratio and topological hit ratio
        base_efficiency = metrics["hit_ratio"] * 0.7 + metrics["topological_hit_ratio"] * 0.3
        
        # Adjust for TVI distribution
        tvi_dist = metrics["tvi_distribution"]
        security_factor = tvi_dist.get("secure", 0.0) * 0.5 + 0.5
        
        return base_efficiency * security_factor
    
    def get_torus_coverage(self) -> float:
        """
        Estimate how well the cache covers the torus topology.
        
        Returns:
            float: Coverage estimate (0.0 to 1.0)
        """
        if not self.regions:
            return 0.0
            
        # Calculate total "area" covered by regions
        total_area = 0.0
        for region in self.regions.values():
            # Simplistic model: area proportional to density and inverse of TVI
            area = region.density * (1.0 / (region.tvi + 0.1))
            total_area += area
            
        # Normalize (this is a rough estimate)
        return min(1.0, total_area / len(self.regions))
    
    def get_optimized_regions(self) -> List[str]:
        """
        Get regions that are currently optimized for.
        
        Returns:
            List[str]: List of optimized region IDs
        """
        # Regions with high density and low TVI
        optimized = []
        for region_id, region in self.regions.items():
            if region.density > 0.1 and region.tvi < 0.4:
                optimized.append(region_id)
                
        return optimized

def create_topologically_optimized_cache(max_size: int = 1000) -> TopologicallyOptimizedCache:
    """
    Factory function to create a topologically-optimized cache.
    
    Args:
        max_size: Maximum size of the cache
        
    Returns:
        TopologicallyOptimizedCache: Configured cache instance
    """
    return TopologicallyOptimizedCache(max_size)

# For backward compatibility with older implementations
TopologicalCache = TopologicallyOptimizedCache
