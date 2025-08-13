"""
Shared size configuration for consistent size categorization across dataset generation and captioning.

This module provides a unified approach to size thresholds that ensures the FOL captions
accurately reflect the size categories used during dataset generation.
"""

import numpy as np
from typing import Tuple


class SizeConfig:
    """
    Unified size configuration for dataset generation and caption generation.
    
    This ensures that the size categories used in FOL captions match exactly
    with the size ranges used when generating the dataset.
    """
    
    def __init__(self, img_size: int):
        self.img_size = img_size
        
        # Size ranges based on canvas size (matching utils.py generate_even_scale)
        self.small_min = max(3, int(img_size * 0.05))   # 5% minimum  
        self.small_max = int(img_size * 0.10)           # 10%
        self.medium_min = int(img_size * 0.18)          # 18%
        self.medium_max = int(img_size * 0.22)          # 22%
        self.large_min = int(img_size * 0.30)           # 30%
        self.large_max = int(img_size * 0.35)           # 35%
    
    def get_size_category(self, size: int) -> str:
        """
        Categorize a size value into small/medium/large.
        
        Args:
            size: Size value in pixels
            
        Returns:
            Size category string: "small", "medium", or "large"
        """
        if size <= self.small_max:
            return "small"
        elif size <= self.medium_max:
            return "medium"
        else:
            return "large"
    
    def get_size_ranges(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Get the size ranges for each category.
        
        Returns:
            Tuple of (small_range, medium_range, large_range) where each range is (min, max)
        """
        return (
            (self.small_min, self.small_max),
            (self.medium_min, self.medium_max), 
            (self.large_min, self.large_max)
        )
    
    def generate_even_sizes(self, n_samples: int) -> np.ndarray:
        """
        Generate evenly distributed shape sizes across 3 categories.
        
        Args:
            n_samples: Number of shapes to generate
        
        Returns:
            Array of shape sizes evenly distributed across small/medium/large
        """
        # Divide shapes evenly across 3 categories
        shapes_per_category = n_samples // 3
        remaining = n_samples % 3
        
        sizes = []
        
        # Small shapes
        for _ in range(shapes_per_category + (1 if remaining > 0 else 0)):
            sizes.append(np.random.randint(self.small_min, self.small_max + 1))
        if remaining > 0:
            remaining -= 1
        
        # Medium shapes  
        for _ in range(shapes_per_category + (1 if remaining > 0 else 0)):
            sizes.append(np.random.randint(self.medium_min, self.medium_max + 1))
        if remaining > 0:
            remaining -= 1
            
        # Large shapes
        for _ in range(shapes_per_category):
            sizes.append(np.random.randint(self.large_min, self.large_max + 1))
        
        # Shuffle and return
        sizes = np.array(sizes)
        np.random.shuffle(sizes)
        return sizes
    
    def get_thresholds_for_composer(self) -> dict:
        """
        Get threshold configuration for the MultiShapeComposer.
        
        Returns:
            Dictionary with threshold values for composer initialization
        """
        return {
            'medium_threshold': self.small_max / self.img_size,  # Boundary between small and medium
            'large_threshold': self.medium_max / self.img_size,  # Boundary between medium and large
        }


def get_size_config(img_size: int) -> SizeConfig:
    """Factory function to create a SizeConfig instance."""
    return SizeConfig(img_size)


# For backward compatibility, provide the original function
def generate_even_scale(n_samples: int, img_size: int) -> np.ndarray:
    """
    Generate evenly distributed shape sizes across 3 categories.
    
    This is a compatibility wrapper around SizeConfig.generate_even_sizes()
    
    Args:
        n_samples: Number of shapes to generate
        img_size: Canvas size
    
    Returns:
        Array of shape sizes evenly distributed across small/medium/large
    """
    config = SizeConfig(img_size)
    return config.generate_even_sizes(n_samples)


if __name__ == "__main__":
    # Test the size configuration
    print("=== SIZE CONFIGURATION TEST ===")
    
    for canvas_size in [32, 64, 128]:
        print(f"\nCanvas size: {canvas_size}x{canvas_size}")
        config = SizeConfig(canvas_size)
        
        small_range, medium_range, large_range = config.get_size_ranges()
        print(f"  Small: {small_range[0]}-{small_range[1]}px")
        print(f"  Medium: {medium_range[0]}-{medium_range[1]}px") 
        print(f"  Large: {large_range[0]}-{large_range[1]}px")
        
        # Test categorization
        test_sizes = [3, 5, 8, 12, 16, 20, 25]
        print("  Size categorization:")
        for size in test_sizes:
            if size <= canvas_size * 0.4:  # Only test reasonable sizes
                category = config.get_size_category(size)
                print(f"    {size}px -> {category}")
        
        # Test composer thresholds
        thresholds = config.get_thresholds_for_composer()
        print(f"  Composer thresholds: {thresholds}")
