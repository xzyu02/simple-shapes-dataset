"""
Multi-shape caption composer with First-Order Logic (FOL) predicate-argument structure.

This module generates captions for multi-shape canvases that explicitly encode
semantic relationships in a way that facilitates studying predicate-argument
structure understanding in models.

FOL Structure Examples:
- "A red circle and a blue square" → ∃x∃y(Circle(x) ∧ Red(x) ∧ Square(y) ∧ Blue(y))
- "A large diamond near a small heart" → ∃x∃y(Large(x) ∧ Diamond(x) ∧ Small(y) ∧ Heart(y) ∧ Near(x,y))

Note: For QA generation, use the separate qa_composer module.
"""

import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from attributes_to_language.types import ChoicesT
from simple_shapes_dataset.text.writers import (
    shapes_writer, 
    color_large_set_writer, 
    size_writer,
    location_writer_bins
)


class MultiShapeComposer:
    """
    Composer for generating FOL-structured captions for multi-shape canvases.
    
    The composer generates captions that encode:
    1. One-place predicates: Shape(x), Color(x), Size(x)
    2. Two-place predicates: Near(x,y), Above(x,y), LeftOf(x,y), etc.
    3. Complex noun phrases with multiple modifiers
    
    For QA generation, use the separate BindingQAComposer from qa_composer module.
    """
    
    def __init__(self, img_size: int = 32):
        self.img_size = img_size
        
        # Spatial relationship thresholds (as fractions of image size)
        self.near_threshold = 0.3  # Within 30% of image size
        self.very_near_threshold = 0.15  # Within 15% of image size
        self.far_threshold = 0.6   # Beyond 60% of image size
        
        # Size classification thresholds (as fractions of canvas dimension)
        # Conservative thresholds for better multi-shape scenes
        self.small_threshold = 0.10    # < 10% of canvas
        self.medium_threshold = 0.20   # < 20% of canvas  
        self.large_threshold = 0.35    # < 35% of canvas
        # anything >= 35% would be "very large" but we'll cap at large
        
        # Sentence structure templates
        self.templates = {
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
        self.spatial_relations = {
            "near": ["near", "close to", "beside", "next to"],
            "far": ["far from", "distant from", "away from"],
            "above": ["above", "over", "on top of"],
            "below": ["below", "under", "beneath"],
            "left_of": ["to the left of", "left of"],
            "right_of": ["to the right of", "right of"],
            "diagonal": ["diagonally from", "at an angle to"],
        }
        
        # Size comparison vocabularies
        self.size_comparisons = {
            "mixed": ["different sized", "various sized", "mixed-size"],
            "similar": ["similar sized", "equally sized", "same sized"],
            "large_small": ["a large and a small", "one large and one small"],
            "multiple_large": ["multiple large", "several large"],
            "multiple_small": ["multiple small", "several small"],
        }
        
        # Arrangement descriptions
        self.arrangements = {
            "scattered": ["scattered across the canvas", "randomly distributed", "spread out"],
            "clustered": ["clustered together", "grouped closely", "bunched up"],
            "linear": ["in a line", "linearly", "in sequence"],
            "corners": ["in the corners", "at the edges", "around the perimeter"],
            "center": ["around the center", "centrally", "in the middle area"],
        }

    def _get_size_description(self, size: int) -> str:
        """
        Get size description relative to canvas size.
        
        This makes size descriptions canvas-aware:
        - 32x32 canvas: size 14 → "large" (44% of canvas)
        - 224x224 canvas: size 14 → "tiny" (6% of canvas)
        """
        size_ratio = size / self.img_size
        
        if size_ratio < self.small_threshold:
            return random.choice(["tiny", "very small"])
        elif size_ratio < self.medium_threshold:
            return random.choice(["small", "little"])
        elif size_ratio < self.large_threshold:
            return random.choice(["medium", "average sized", "medium sized"])
        else:
            return random.choice(["large", "big"])

    def _get_shape_description(self, shape_idx: int, canvas_data: Dict, shape_num: int) -> Tuple[str, ChoicesT]:
        """
        Generate description for a single shape with all its properties.
        
        This creates complex noun phrases that "unwind" multiple modifiers:
        Large(x) ∧ Red(x) ∧ Circle(x) → "a large red circle"
        """
        # Extract shape properties
        shape_type = int(canvas_data["classes"][shape_idx])
        size = canvas_data["sizes"][shape_idx]
        color = tuple(canvas_data["colors"][shape_idx])
        
        # Generate individual property descriptions using canvas-aware sizing
        shape_name, shape_choices = shapes_writer(shape_type)
        size_desc = self._get_size_description(size)  # Use canvas-aware size description
        color_desc, color_choices = color_large_set_writer(*color)
        
        # Create choices for tracking (size choices are now determined by our canvas-aware method)
        size_category = self._get_size_category(size)
        combined_choices = {
            f"shape_{shape_num}": shape_choices,
            f"size_{shape_num}": size_category,  # Store the size category
            f"color_{shape_num}": color_choices,
        }
        
        # Create complex noun phrase with multiple modifiers
        # This encodes: Size(x) ∧ Color(x) ∧ Shape(x)
        if random.random() < 0.7:  # Usually include size
            if random.random() < 0.5:
                description = f"a {size_desc}, {color_desc} {shape_name}"
            else:
                description = f"a {size_desc} {color_desc} {shape_name}"
        else:
            description = f"a {color_desc} {shape_name}"
            
        return description, combined_choices

    def _get_size_category(self, size: int) -> str:
        """Get the size category for tracking purposes."""
        size_ratio = size / self.img_size
        
        if size_ratio < self.small_threshold:
            return "tiny"
        elif size_ratio < self.medium_threshold:
            return "small"
        elif size_ratio < self.large_threshold:
            return "medium"
        else:
            return "large"

    def _calculate_spatial_relationship(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> str:
        """
        Calculate spatial relationship between two shapes for two-place predicates.
        
        Returns relationships like Near(x,y), Above(x,y), LeftOf(x,y), etc.
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Calculate distances and relative positions
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        distance_ratio = distance / self.img_size
        
        # Determine primary spatial relationship
        if distance_ratio < self.very_near_threshold:
            return "very near"
        elif distance_ratio < self.near_threshold:
            return "near"
        elif distance_ratio > self.far_threshold:
            return "far"
        
        # For medium distances, use directional relationships
        abs_dx, abs_dy = abs(dx), abs(dy)
        
        if abs_dy > abs_dx * 1.5:  # Primarily vertical
            return "above" if dy < 0 else "below"
        elif abs_dx > abs_dy * 1.5:  # Primarily horizontal  
            return "left_of" if dx < 0 else "right_of"
        else:  # Diagonal
            return "diagonal"

    def _analyze_canvas_properties(self, canvas_data: Dict) -> Dict[str, Any]:
        """
        Analyze global properties of the canvas for higher-level descriptions.
        """
        num_shapes = canvas_data["num_shapes"]
        sizes = canvas_data["sizes"][:num_shapes]
        locations = canvas_data["locations"][:num_shapes]
        
        # Size analysis - relative to canvas size
        size_variety = "mixed" if len(set(sizes)) > 1 else "similar"
        if num_shapes == 2 and len(set(sizes)) == 2:
            size_variety = "large_small"
        else:
            # Calculate size thresholds relative to canvas size
            small_threshold_px = self.img_size * self.small_threshold
            medium_threshold_px = self.img_size * self.medium_threshold
            large_threshold_px = self.img_size * self.large_threshold
            
            small_shapes = sum(1 for s in sizes if s <= small_threshold_px)
            medium_shapes = sum(1 for s in sizes if small_threshold_px < s <= medium_threshold_px)
            large_shapes = sum(1 for s in sizes if medium_threshold_px < s <= large_threshold_px)
            
            if small_shapes == num_shapes:
                size_variety = "multiple_small"
            elif large_shapes == num_shapes:
                size_variety = "multiple_large"
            else:
                size_variety = "mixed"
        
        # Spatial arrangement analysis
        if num_shapes >= 3:
            # Calculate spread
            x_coords = [loc[0] for loc in locations]
            y_coords = [loc[1] for loc in locations]
            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)
            total_spread = (x_spread + y_spread) / (2 * self.img_size)
            
            if total_spread < 0.3:
                arrangement = "clustered"
            elif total_spread > 0.7:
                arrangement = "scattered"
            else:
                # Check for linear arrangement
                if abs(x_spread - y_spread) / max(x_spread, y_spread) > 0.7:
                    arrangement = "linear"
                else:
                    arrangement = "center"
        else:
            arrangement = "simple"
            
        return {
            "size_variety": size_variety,
            "arrangement": arrangement,
            "num_shapes": num_shapes,
        }

    def _generate_simple_conjunction(self, canvas_data: Dict, properties: Dict) -> Tuple[str, ChoicesT]:
        """
        Generate simple conjunction: ∃x∃y(Shape(x) ∧ Color(x) ∧ Shape(y) ∧ Color(y))
        Example: "The image contains a red circle and a blue square."
        """
        num_shapes = properties["num_shapes"]
        
        # Generate descriptions for all shapes
        shape_descriptions = []
        all_choices = {}
        
        for i in range(num_shapes):
            desc, choices = self._get_shape_description(i, canvas_data, i)
            shape_descriptions.append(desc)
            all_choices.update(choices)
        
        # Combine with appropriate conjunctions
        if num_shapes == 2:
            objects_str = f"{shape_descriptions[0]} and {shape_descriptions[1]}"
        elif num_shapes == 3:
            objects_str = f"{shape_descriptions[0]}, {shape_descriptions[1]}, and {shape_descriptions[2]}"
        else:
            objects_str = ", ".join(shape_descriptions[:-1]) + f", and {shape_descriptions[-1]}"
        
        template = random.choice(self.templates["simple_conjunction"])
        caption = template.format(objects=objects_str)
        
        all_choices["template"] = "simple_conjunction"
        return caption, all_choices

    def _generate_spatial_relationship(self, canvas_data: Dict, properties: Dict) -> Tuple[str, ChoicesT]:
        """
        Generate spatial relationship: ∃x∃y(Shape(x) ∧ Color(x) ∧ Shape(y) ∧ Color(y) ∧ Relation(x,y))
        Example: "A large diamond is near a small heart."
        """
        if properties["num_shapes"] < 2:
            return self._generate_simple_conjunction(canvas_data, properties)
        
        # Choose two shapes to relate
        shape1_idx, shape2_idx = 0, 1
        if properties["num_shapes"] > 2:
            # Randomly select two shapes
            indices = random.sample(range(properties["num_shapes"]), 2)
            shape1_idx, shape2_idx = indices
        
        # Generate descriptions for both shapes
        primary_desc, primary_choices = self._get_shape_description(shape1_idx, canvas_data, 1)
        secondary_desc, secondary_choices = self._get_shape_description(shape2_idx, canvas_data, 2)
        
        # Calculate spatial relationship
        pos1 = tuple(canvas_data["locations"][shape1_idx])
        pos2 = tuple(canvas_data["locations"][shape2_idx])
        spatial_type = self._calculate_spatial_relationship(pos1, pos2)
        
        # Choose appropriate relation phrase
        relation_phrase = random.choice(self.spatial_relations.get(spatial_type, ["near"]))
        
        template = random.choice(self.templates["complex_description"])
        caption = template.format(
            primary_object=primary_desc,
            spatial_relation=relation_phrase,
            secondary_object=secondary_desc
        )
        
        # Combine choices
        all_choices = {**primary_choices, **secondary_choices}
        all_choices["spatial_relation"] = spatial_type
        all_choices["template"] = "spatial_relationship"
        
        return caption, all_choices

    def _generate_comparative_description(self, canvas_data: Dict, properties: Dict) -> Tuple[str, ChoicesT]:
        """
        Generate comparative description focusing on size relationships.
        Example: "The image contains different sized shapes: a large circle and a small square."
        """
        num_shapes = properties["num_shapes"]
        size_variety = properties["size_variety"]
        
        # Generate shape descriptions
        shape_descriptions = []
        all_choices = {}
        
        for i in range(num_shapes):
            desc, choices = self._get_shape_description(i, canvas_data, i)
            shape_descriptions.append(desc)
            all_choices.update(choices)
        
        # Create objects string
        if num_shapes == 2:
            objects_str = f"{shape_descriptions[0]} and {shape_descriptions[1]}"
        else:
            objects_str = ", ".join(shape_descriptions[:-1]) + f", and {shape_descriptions[-1]}"
        
        # Choose size comparison phrase
        size_comp_phrase = random.choice(self.size_comparisons.get(size_variety, ["different sized"]))
        
        template = random.choice(self.templates["comparative"])
        caption = template.format(
            size_comparison=size_comp_phrase,
            objects=objects_str
        )
        
        all_choices["size_comparison"] = size_variety
        all_choices["template"] = "comparative"
        
        return caption, all_choices

    def _generate_positional_description(self, canvas_data: Dict, properties: Dict) -> Tuple[str, ChoicesT]:
        """
        Generate description focusing on overall spatial arrangement.
        Example: "A red circle and blue square are scattered across the canvas."
        """
        if properties["num_shapes"] < 3:
            return self._generate_spatial_relationship(canvas_data, properties)
        
        arrangement = properties["arrangement"]
        
        # Generate shape descriptions
        shape_descriptions = []
        all_choices = {}
        
        for i in range(properties["num_shapes"]):
            desc, choices = self._get_shape_description(i, canvas_data, i)
            shape_descriptions.append(desc)
            all_choices.update(choices)
        
        # Create objects string
        objects_str = ", ".join(shape_descriptions[:-1]) + f", and {shape_descriptions[-1]}"
        
        # Choose arrangement phrase
        arrangement_phrase = random.choice(self.arrangements.get(arrangement, ["on the canvas"]))
        
        template = random.choice(self.templates["positional"])
        caption = template.format(
            objects=objects_str,
            arrangement_description=arrangement_phrase
        )
        
        all_choices["arrangement"] = arrangement
        all_choices["template"] = "positional"
        
        return caption, all_choices

    def generate_caption(self, canvas_data: Dict) -> Tuple[str, ChoicesT]:
        """
        Generate a FOL-structured caption for a multi-shape canvas.
        
        Args:
            canvas_data: Dictionary containing:
                - classes: shape types for each shape
                - sizes: sizes for each shape  
                - colors: RGB colors for each shape
                - locations: (x, y) positions for each shape
                - num_shapes: number of actual shapes in canvas
        
        Returns:
            Tuple of (caption_string, choices_dict)
        """
        # Analyze canvas properties
        properties = self._analyze_canvas_properties(canvas_data)
        
        # Choose caption generation strategy based on number of shapes and properties
        num_shapes = properties["num_shapes"]
        
        if num_shapes == 1:
            # Single shape: just describe it
            return self._get_shape_description(0, canvas_data, 0)
        
        # Multi-shape strategies with different probabilities
        strategies = []
        
        # Always available strategies
        strategies.extend([
            ("simple_conjunction", 0.3),
            ("comparative", 0.2),
        ])
        
        # Spatial relationship strategy (for 2+ shapes)
        if num_shapes >= 2:
            strategies.append(("spatial_relationship", 0.3))
        
        # Positional strategy (for 3+ shapes)
        if num_shapes >= 3:
            strategies.append(("positional", 0.2))
        
        # Normalize probabilities
        total_prob = sum(prob for _, prob in strategies)
        strategies = [(strategy, prob/total_prob) for strategy, prob in strategies]
        
        # Choose strategy
        rand_val = random.random()
        cumulative_prob = 0
        chosen_strategy = "simple_conjunction"  # fallback
        
        for strategy, prob in strategies:
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                chosen_strategy = strategy
                break
        
        # Generate caption using chosen strategy
        if chosen_strategy == "simple_conjunction":
            return self._generate_simple_conjunction(canvas_data, properties)
        elif chosen_strategy == "spatial_relationship":
            return self._generate_spatial_relationship(canvas_data, properties)
        elif chosen_strategy == "comparative":
            return self._generate_comparative_description(canvas_data, properties)
        elif chosen_strategy == "positional":
            return self._generate_positional_description(canvas_data, properties)
        else:
            return self._generate_simple_conjunction(canvas_data, properties)


def create_multi_shape_composer(img_size: int = 32) -> MultiShapeComposer:
    """Factory function to create a multi-shape composer."""
    return MultiShapeComposer(img_size=img_size)


# Example usage and testing
if __name__ == "__main__":
    # Test the composer with example data
    composer = create_multi_shape_composer(32)
    
    # Example canvas data
    test_canvas = {
        "classes": np.array([3, 4]),  # circle, square
        "sizes": np.array([12, 8]),   # large, small
        "colors": np.array([[255, 0, 0], [0, 0, 255]]),  # red, blue
        "locations": np.array([[10, 15], [20, 15]]),  # side by side
        "num_shapes": 2,
    }
    
    print("=== CAPTION GENERATION EXAMPLES ===")
    print("Example captions:")
    for i in range(3):
        caption, choices = composer.generate_caption(test_canvas)
        print(f"{i+1}. {caption}")
        print(f"   Strategy: {choices.get('template', 'unknown')}")
        print(f"   Choices: {choices}")
        print()
    
    print("\n=== SIZE RANGE RECOMMENDATIONS ===")
    print(f"Canvas size: {composer.img_size}x{composer.img_size}")
    print("Size thresholds for classification:")
    print(f"  Small: < {composer.small_threshold:.0%} of canvas")
    print(f"  Medium: < {composer.medium_threshold:.0%} of canvas") 
    print(f"  Large: < {composer.large_threshold:.0%} of canvas")
    
    # Test with different canvas sizes
    print(f"\n=== COMPARISON ACROSS CANVAS SIZES ===")
    for canvas_size in [32, 64, 128, 224]:
        test_composer = create_multi_shape_composer(canvas_size)
        small_px = int(canvas_size * test_composer.small_threshold)
        medium_px = int(canvas_size * test_composer.medium_threshold)
        large_px = int(canvas_size * test_composer.large_threshold)
        print(f"  {canvas_size:3d}x{canvas_size:<3d}: small<{small_px:2d}px, medium<{medium_px:2d}px, large<{large_px:2d}px")
