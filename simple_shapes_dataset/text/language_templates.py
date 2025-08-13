"""
Language templates and vocabularies for FOL caption generation.

This module contains all the natural language templates, vocabularies, and
predicate mappings used by the MultiShapeComposer for generating captions
and their corresponding First-Order Logic (FOL) structures.
"""

from typing import Dict, List, Tuple

# Sentence structure templates for different caption types
SENTENCE_TEMPLATES = {
    "simple_conjunction": [
        "The image contains {objects}.",
        "There are {objects}.",
        "{objects} are visible.",
        "The canvas shows {objects}.",
    ],
    "complex_description": [
        "The image shows {primary_object} {spatial_relation} {secondary_object}.",
        "{primary_object} is {spatial_relation} {secondary_object}.",
        "There is {primary_object} {spatial_relation} {secondary_object}.",
    ],
    "comparative": [
        "The image contains {size_comparison} shapes: {objects}.",
        "There are {size_comparison} objects: {objects}.",
    ],
    "positional": [
        "{objects} are arranged {arrangement_description}.",
        "The shapes are positioned {arrangement_description}.",
    ]
}

# Spatial relation vocabularies for two-place predicates
SPATIAL_RELATIONS = {
    "near": ["near", "close to", "beside", "next to"],
    "far": ["far from", "distant from", "away from"],
    "above": ["above", "over", "on top of"],
    "below": ["below", "under", "beneath"],
    "left_of": ["to the left of", "left of"],
    "right_of": ["to the right of", "right of"],
    "diagonal": ["diagonally from", "at an angle to"],
}

# Size comparison vocabularies
SIZE_COMPARISONS = {
    "mixed": ["different sized", "various sized", "mixed-size"],
    "similar": ["similar sized", "equally sized", "same sized"],
    "large_small": ["a large and a small", "one large and one small"],
    "multiple_large": ["multiple large", "several large"],
    "multiple_small": ["multiple small", "several small"],
}

# Arrangement descriptions for positional captions
ARRANGEMENTS = {
    "scattered": ["scattered across the canvas", "randomly distributed", "spread out"],
    "clustered": ["clustered together", "grouped closely", "bunched up"],
    "linear": ["in a line", "linearly", "in sequence"],
    "corners": ["in the corners", "at the edges", "around the perimeter"],
    "center": ["around the center", "centrally", "in the middle area"],
}

# FOL predicate mappings for shapes
SHAPE_PREDICATES = {
    0: "Diamond",
    1: "Oval", 
    2: "Triangle",
    3: "Circle",
    4: "Square",
    5: "Star",
    6: "Heart"
}

# FOL predicate mappings for colors (RGB tuples to predicates)
COLOR_PREDICATES = {
    (255, 0, 0): "Red",
    (0, 255, 0): "Green", 
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
    (255, 0, 255): "Magenta",
    (0, 255, 255): "Cyan",
    (255, 165, 0): "Orange",
    (128, 0, 128): "Purple",
    (255, 192, 203): "Pink",
    (165, 42, 42): "Brown",
    (128, 128, 128): "Gray",
    (0, 0, 0): "Black",
    (255, 255, 255): "White"
}

# FOL predicate mappings for sizes
SIZE_PREDICATES = {
    "small": "Small", 
    "medium": "Medium",
    "large": "Large"
}

# FOL predicate mappings for spatial relationships
SPATIAL_PREDICATES = {
    "near": "Near",
    "far": "Far",
    "above": "Above",
    "below": "Below",
    "left_of": "LeftOf",
    "right_of": "RightOf",
    "diagonal": "DiagonalFrom"
}

# Size description vocabularies for natural language
SIZE_DESCRIPTIONS = {
    "small": ["small", "little"],
    "medium": ["medium", "average sized", "medium sized"],
    "large": ["large", "big"]
}

def get_closest_color_predicate(color: Tuple[int, int, int]) -> str:
    """
    Find the closest matching FOL color predicate for a given RGB color.
    
    Args:
        color: RGB color tuple (r, g, b)
        
    Returns:
        Closest matching FOL color predicate string
    """
    min_distance = float('inf')
    closest_color = "Unknown"
    
    for rgb, predicate in COLOR_PREDICATES.items():
        distance = sum((a - b) ** 2 for a, b in zip(color, rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color = predicate
    
    return closest_color

def get_shape_predicate(shape_idx: int) -> str:
    """
    Get FOL shape predicate from shape index.
    
    Args:
        shape_idx: Integer index representing the shape type
        
    Returns:
        FOL shape predicate string
    """
    return SHAPE_PREDICATES.get(shape_idx, "Shape")

def get_size_predicate(size_category: str) -> str:
    """
    Get FOL size predicate from size category.
    
    Args:
        size_category: Size category string ("small", "medium", "large")
        
    Returns:
        FOL size predicate string
    """
    return SIZE_PREDICATES.get(size_category, "Medium")

def get_spatial_predicate(spatial_relation: str) -> str:
    """
    Get FOL spatial predicate from spatial relation type.
    
    Args:
        spatial_relation: Spatial relation string (e.g., "near", "above", etc.)
        
    Returns:
        FOL spatial predicate string
    """
    return SPATIAL_PREDICATES.get(spatial_relation, "Near")

# Configuration constants for spatial and size thresholds
class ThresholdConfig:
    """Configuration class for various thresholds used in caption generation."""
    
    # Spatial relationship thresholds (as fractions of image size)
    NEAR_THRESHOLD = 0.3  # Within 30% of image size
    FAR_THRESHOLD = 0.6   # Beyond 60% of image size
    
    # Spatial arrangement thresholds
    CLUSTERED_THRESHOLD = 0.3  # Total spread < 30% for clustered
    SCATTERED_THRESHOLD = 0.7  # Total spread > 70% for scattered
    LINEAR_THRESHOLD = 0.7     # Spread difference > 70% for linear
    
    # Note: Size thresholds are now managed by SizeConfig in size_config.py
    # This ensures consistency between dataset generation and caption generation
