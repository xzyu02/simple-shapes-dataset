#!/usr/bin/env python3
"""
Test script for the new FOL-structured multi-shape caption generation.

This demonstrates how the new composer generates captions that encode
First-Order Logic predicate-argument structure for studying semantic meaning.
"""

import numpy as np
import sys
from pathlib import Path

# Add the project root to the path so we can import the modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from simple_shapes_dataset.text.composer_multi import create_multi_shape_composer


def create_test_canvas(canvas_type: str) -> dict:
    """Create test canvas data for different scenarios."""
    
    if canvas_type == "simple_conjunction":
        # ∃x∃y(Circle(x) ∧ Red(x) ∧ Square(y) ∧ Blue(y))
        return {
            "classes": np.array([3, 4]),  # circle, square
            "sizes": np.array([12, 10]),   # medium sizes
            "colors": np.array([[255, 50, 50], [50, 50, 255]]),  # red, blue
            "locations": np.array([[10, 15], [22, 15]]),  # side by side
            "rotations": np.array([0.0, 0.0]),
            "num_shapes": 2,
        }
    
    elif canvas_type == "spatial_relationship":
        # ∃x∃y(Large(x) ∧ Diamond(x) ∧ Small(y) ∧ Heart(y) ∧ Near(x,y))
        return {
            "classes": np.array([0, 6]),  # diamond, heart
            "sizes": np.array([14, 7]),   # large, small
            "colors": np.array([[100, 100, 100], [0, 200, 180]]),  # gray, teal
            "locations": np.array([[12, 12], [18, 18]]),  # close together
            "rotations": np.array([0.5, 0.0]),
            "num_shapes": 2,
        }
    
    elif canvas_type == "multiple_modifiers":
        # ∃x∃y∃z(Large(x) ∧ Red(x) ∧ Triangle(x) ∧ Medium(y) ∧ Green(y) ∧ Circle(y) ∧ Small(z) ∧ Blue(z) ∧ Square(z))
        return {
            "classes": np.array([2, 3, 4]),  # triangle, circle, square
            "sizes": np.array([13, 11, 8]),   # large, medium, small
            "colors": np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),  # red, green, blue
            "locations": np.array([[8, 8], [16, 16], [24, 24]]),  # diagonal arrangement
            "rotations": np.array([0.2, 0.0, 1.5]),
            "num_shapes": 3,
        }
    
    elif canvas_type == "complex_spatial":
        # Multiple shapes with various spatial relationships
        return {
            "classes": np.array([3, 4, 2, 5]),  # circle, square, triangle, star
            "sizes": np.array([10, 12, 9, 8]),
            "colors": np.array([[255, 165, 0], [128, 0, 128], [255, 192, 203], [0, 128, 0]]),  # orange, purple, pink, green
            "locations": np.array([[6, 6], [26, 6], [6, 26], [26, 26]]),  # four corners
            "rotations": np.array([0.0, 0.8, 1.2, 2.0]),
            "num_shapes": 4,
        }
    
    else:
        raise ValueError(f"Unknown canvas type: {canvas_type}")


def demonstrate_fol_captions():
    """Demonstrate the FOL-structured caption generation."""
    
    print("=== FOL-Structured Multi-Shape Caption Generation Demo ===\n")
    print("This demonstrates how the new composer generates captions that encode")
    print("First-Order Logic predicate-argument structure for studying semantic meaning.")
    print("Note: Size descriptions are now canvas-aware!\n")
    
    # Test with different canvas sizes to show canvas-aware sizing
    canvas_sizes = [32, 64, 224]
    
    for canvas_size in canvas_sizes:
        print(f"--- Canvas Size: {canvas_size}x{canvas_size} ---")
        
        # Create the composer for this canvas size
        composer = create_multi_shape_composer(img_size=canvas_size)
        
        # Create a test canvas with the same absolute shape size (14 pixels)
        # This will be described differently depending on canvas size
        canvas_data = {
            "classes": np.array([3, 4]),  # circle, square
            "sizes": np.array([14, 14]),   # Same absolute size
            "colors": np.array([[255, 0, 0], [0, 0, 255]]),  # red, blue
            "locations": np.array([[10, 15], [20, 15]]),  # side by side
            "rotations": np.array([0.0, 0.0]),
            "num_shapes": 2,
        }
        
        # Adjust locations proportionally to canvas size
        scale_factor = canvas_size / 32
        canvas_data["locations"] = (canvas_data["locations"] * scale_factor).astype(int)
        
        print(f"Shape size: 14 pixels (ratio: {14/canvas_size:.1%} of canvas)")
        
        # Generate a caption
        caption, choices = composer.generate_caption(canvas_data)
        print(f"Caption: {caption}")
        print(f"Size categories: {[v for k, v in choices.items() if 'size_' in k]}")
        print()
    
    print("="*80 + "\n")
    
    # Continue with original test cases using 32x32 canvas
    composer = create_multi_shape_composer(img_size=32)
    
    # Test different canvas types
    test_cases = [
        ("simple_conjunction", "Simple Conjunction (∃x∃y(P(x) ∧ Q(y)))"),
        ("spatial_relationship", "Spatial Relationship (∃x∃y(P(x) ∧ Q(y) ∧ R(x,y)))"),
        ("multiple_modifiers", "Multiple Modifiers (∃x(P₁(x) ∧ P₂(x) ∧ P₃(x)))"),
        ("complex_spatial", "Complex Spatial Arrangements"),
    ]
    
    for canvas_type, description in test_cases:
        print(f"--- {description} ---")
        
        # Create test canvas
        canvas_data = create_test_canvas(canvas_type)
        
        print(f"Canvas: {canvas_data['num_shapes']} shapes")
        for i in range(canvas_data['num_shapes']):
            shape_names = {0: "diamond", 1: "oval", 2: "triangle", 3: "circle", 4: "square", 5: "star", 6: "heart"}
            shape_name = shape_names.get(canvas_data['classes'][i], "unknown")
            color = canvas_data['colors'][i]
            size = canvas_data['sizes'][i]
            loc = canvas_data['locations'][i]
            print(f"  Shape {i+1}: {shape_name}, size={size}, color=RGB{tuple(color)}, location={tuple(loc)}")
        
        print("\nGenerated captions:")
        
        # Generate multiple captions to show variety
        for j in range(5):
            caption, choices = composer.generate_caption(canvas_data)
            template_type = choices.get('template', 'unknown')
            print(f"  {j+1}. [{template_type}] {caption}")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    demonstrate_fol_captions()
