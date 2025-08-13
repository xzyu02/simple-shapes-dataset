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
from simple_shapes_dataset.text.language_templates import (
    SENTENCE_TEMPLATES,
    SPATIAL_RELATIONS,
    SIZE_COMPARISONS,
    ARRANGEMENTS,
    SIZE_DESCRIPTIONS,
    ThresholdConfig,
    get_closest_color_predicate,
    get_shape_predicate,
    get_size_predicate,
    get_spatial_predicate
)
from simple_shapes_dataset.text.size_config import SizeConfig

# Type alias for caption-FOL pairs
CaptionFOLPair = Tuple[str, str, ChoicesT]  # (natural_language, fol_structure, choices)


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
        
        # Initialize size configuration (shared with dataset generation)
        self.size_config = SizeConfig(img_size)
        
        # Load spatial thresholds from configuration
        self.near_threshold = ThresholdConfig.NEAR_THRESHOLD
        self.far_threshold = ThresholdConfig.FAR_THRESHOLD
        
        # Load templates and vocabularies from language_templates module
        self.templates = SENTENCE_TEMPLATES
        self.spatial_relations = SPATIAL_RELATIONS
        self.size_comparisons = SIZE_COMPARISONS
        self.arrangements = ARRANGEMENTS
        self.size_descriptions = SIZE_DESCRIPTIONS

    def _get_fol_color_predicate(self, color: Tuple[int, int, int]) -> str:
        """Get FOL color predicate from RGB values."""
        return get_closest_color_predicate(color)

    def _get_fol_shape_predicate(self, shape_idx: int) -> str:
        """Get FOL shape predicate from shape index."""
        return get_shape_predicate(shape_idx)

    def _get_fol_size_predicate(self, size: int) -> str:
        """Get FOL size predicate from size value."""
        size_category = self._get_size_category(size)
        return get_size_predicate(size_category)

    def _generate_fol_for_shape(self, shape_idx: int, canvas_data: Dict, variable: str) -> List[str]:
        """Generate FOL predicates for a single shape."""
        predicates = []
        
        # Shape type predicate
        shape_type = int(canvas_data["classes"][shape_idx])
        shape_pred = self._get_fol_shape_predicate(shape_type)
        predicates.append(f"{shape_pred}({variable})")
        
        # Color predicate
        color = tuple(canvas_data["colors"][shape_idx])
        color_pred = self._get_fol_color_predicate(color)
        predicates.append(f"{color_pred}({variable})")
        
        # Size predicate (if significantly different from medium)
        size = canvas_data["sizes"][shape_idx]
        size_category = self._get_size_category(size)
        if size_category != "medium":  # Only include non-medium sizes
            size_pred = self._get_fol_size_predicate(size)
            predicates.append(f"{size_pred}({variable})")
        
        return predicates

    def _generate_fol_spatial_predicate(self, var1: str, var2: str, spatial_relation: str) -> str:
        """Generate FOL spatial predicate between two variables."""
        fol_relation = get_spatial_predicate(spatial_relation)
        return f"{fol_relation}({var1},{var2})"

    def _get_size_description(self, size: int) -> str:
        """
        Get size description relative to canvas size.
        
        This makes size descriptions canvas-aware:
        - 32x32 canvas: size 14 → "large" (44% of canvas)
        - 224x224 canvas: size 14 → "tiny" (6% of canvas)
        """
        size_category = self._get_size_category(size)
        return random.choice(self.size_descriptions[size_category])

    def _get_shape_description(self, shape_idx: int, canvas_data: Dict, shape_num: int) -> Tuple[str, List[str], ChoicesT]:
        """
        Generate description for a single shape with all its properties.
        
        This creates complex noun phrases that "unwind" multiple modifiers:
        Large(x) ∧ Red(x) ∧ Circle(x) → "a large red circle"
        
        Returns:
            Tuple of (natural_description, fol_predicates, choices)
        """
        # Extract shape properties
        shape_type = int(canvas_data["classes"][shape_idx])
        size = canvas_data["sizes"][shape_idx]
        color = tuple(canvas_data["colors"][shape_idx])
        
        # Generate individual property descriptions using canvas-aware sizing
        shape_name, shape_choices = shapes_writer(shape_type)
        size_desc = self._get_size_description(size)  # Use canvas-aware size description
        color_desc, color_choices = color_large_set_writer(*color)
        
        # Generate FOL predicates
        variable = f"x{shape_num}"
        fol_predicates = self._generate_fol_for_shape(shape_idx, canvas_data, variable)
        
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
            
        return description, fol_predicates, combined_choices

    def _get_size_category(self, size: int) -> str:
        """Get the size category for tracking purposes using shared size config."""
        return self.size_config.get_size_category(size)

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
        if distance_ratio < self.near_threshold:
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
            # Use size ranges from shared configuration
            small_range, medium_range, large_range = self.size_config.get_size_ranges()
            
            small_shapes = sum(1 for s in sizes if s <= small_range[1])
            medium_shapes = sum(1 for s in sizes if small_range[1] < s <= medium_range[1])
            large_shapes = sum(1 for s in sizes if s > medium_range[1])
            
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
            
            if total_spread < ThresholdConfig.CLUSTERED_THRESHOLD:
                arrangement = "clustered"
            elif total_spread > ThresholdConfig.SCATTERED_THRESHOLD:
                arrangement = "scattered"
            else:
                # Check for linear arrangement
                if abs(x_spread - y_spread) / max(x_spread, y_spread) > ThresholdConfig.LINEAR_THRESHOLD:
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

    def _generate_simple_conjunction(self, canvas_data: Dict, properties: Dict) -> CaptionFOLPair:
        """
        Generate simple conjunction: ∃x∃y(Shape(x) ∧ Color(x) ∧ Shape(y) ∧ Color(y))
        Example: "The image contains a red circle and a blue square."
        FOL: ∃x∃y(Circle(x) ∧ Red(x) ∧ Square(y) ∧ Blue(y))
        """
        num_shapes = properties["num_shapes"]
        
        # Generate descriptions for all shapes
        shape_descriptions = []
        all_fol_predicates = []
        all_choices = {}
        variables = []
        
        for i in range(num_shapes):
            desc, fol_preds, choices = self._get_shape_description(i, canvas_data, i)
            shape_descriptions.append(desc)
            all_fol_predicates.extend(fol_preds)
            all_choices.update(choices)
            variables.append(f"x{i}")
        
        # Combine with appropriate conjunctions
        if num_shapes == 2:
            objects_str = f"{shape_descriptions[0]} and {shape_descriptions[1]}"
        elif num_shapes == 3:
            objects_str = f"{shape_descriptions[0]}, {shape_descriptions[1]}, and {shape_descriptions[2]}"
        else:
            objects_str = ", ".join(shape_descriptions[:-1]) + f", and {shape_descriptions[-1]}"
        
        template = random.choice(self.templates["simple_conjunction"])
        caption = template.format(objects=objects_str)
        
        # Create FOL structure
        if num_shapes == 1:
            fol_structure = f"∃x0({' ∧ '.join(all_fol_predicates)})"
        else:
            existential_vars = "".join([f"∃{var}" for var in variables])
            fol_structure = f"{existential_vars}({' ∧ '.join(all_fol_predicates)})"
        
        all_choices["template"] = "simple_conjunction"
        return caption, fol_structure, all_choices

    def _generate_spatial_relationship(self, canvas_data: Dict, properties: Dict) -> CaptionFOLPair:
        """
        Generate spatial relationship: ∃x∃y(Shape(x) ∧ Color(x) ∧ Shape(y) ∧ Color(y) ∧ Relation(x,y))
        Example: "A large diamond is near a small heart."
        FOL: ∃x∃y(Large(x) ∧ Diamond(x) ∧ Small(y) ∧ Heart(y) ∧ Near(x,y))
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
        primary_desc, primary_fol, primary_choices = self._get_shape_description(shape1_idx, canvas_data, 0)
        secondary_desc, secondary_fol, secondary_choices = self._get_shape_description(shape2_idx, canvas_data, 1)
        
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
        
        # Create FOL structure
        spatial_fol = self._generate_fol_spatial_predicate("x0", "x1", spatial_type)
        all_fol_predicates = primary_fol + secondary_fol + [spatial_fol]
        fol_structure = f"∃x0∃x1({' ∧ '.join(all_fol_predicates)})"
        
        # Combine choices
        all_choices = {**primary_choices, **secondary_choices}
        all_choices["spatial_relation"] = spatial_type
        all_choices["template"] = "spatial_relationship"
        
        return caption, fol_structure, all_choices

    def _generate_comparative_description(self, canvas_data: Dict, properties: Dict) -> CaptionFOLPair:
        """
        Generate comparative description focusing on size relationships.
        Example: "The image contains different sized shapes: a large circle and a small square."
        FOL: ∃x∃x1(Large(x) ∧ Circle(x) ∧ Small(x1) ∧ Square(x1))
        """
        num_shapes = properties["num_shapes"]
        size_variety = properties["size_variety"]
        
        # Generate shape descriptions
        shape_descriptions = []
        all_fol_predicates = []
        all_choices = {}
        variables = []
        
        for i in range(num_shapes):
            desc, fol_preds, choices = self._get_shape_description(i, canvas_data, i)
            shape_descriptions.append(desc)
            all_fol_predicates.extend(fol_preds)
            all_choices.update(choices)
            variables.append(f"x{i}")
        
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
        
        # Create FOL structure
        if num_shapes == 1:
            fol_structure = f"∃x0({' ∧ '.join(all_fol_predicates)})"
        else:
            existential_vars = "".join([f"∃{var}" for var in variables])
            fol_structure = f"{existential_vars}({' ∧ '.join(all_fol_predicates)})"
        
        all_choices["size_comparison"] = size_variety
        all_choices["template"] = "comparative"
        
        return caption, fol_structure, all_choices

    def _generate_positional_description(self, canvas_data: Dict, properties: Dict) -> CaptionFOLPair:
        """
        Generate description focusing on overall spatial arrangement.
        Example: "A red circle and blue square are scattered across the canvas."
        FOL: ∃x∃x1(Red(x) ∧ Circle(x) ∧ Blue(x1) ∧ Square(x1) ∧ Scattered(x,x1))
        """
        if properties["num_shapes"] < 3:
            return self._generate_spatial_relationship(canvas_data, properties)
        
        arrangement = properties["arrangement"]
        
        # Generate shape descriptions
        shape_descriptions = []
        all_fol_predicates = []
        all_choices = {}
        variables = []
        
        for i in range(properties["num_shapes"]):
            desc, fol_preds, choices = self._get_shape_description(i, canvas_data, i)
            shape_descriptions.append(desc)
            all_fol_predicates.extend(fol_preds)
            all_choices.update(choices)
            variables.append(f"x{i}")
        
        # Create objects string
        objects_str = ", ".join(shape_descriptions[:-1]) + f", and {shape_descriptions[-1]}"
        
        # Choose arrangement phrase
        arrangement_phrase = random.choice(self.arrangements.get(arrangement, ["on the canvas"]))
        
        template = random.choice(self.templates["positional"])
        caption = template.format(
            objects=objects_str,
            arrangement_description=arrangement_phrase
        )
        
        # Create FOL structure (simplified - just conjunction of individual predicates)
        existential_vars = "".join([f"∃{var}" for var in variables])
        fol_structure = f"{existential_vars}({' ∧ '.join(all_fol_predicates)})"
        
        all_choices["arrangement"] = arrangement
        all_choices["template"] = "positional"
        
        return caption, fol_structure, all_choices

    def generate_caption(self, canvas_data: Dict) -> CaptionFOLPair:
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
            Tuple of (caption_string, fol_structure, choices_dict)
        """
        # Analyze canvas properties
        properties = self._analyze_canvas_properties(canvas_data)
        
        # Choose caption generation strategy based on number of shapes and properties
        num_shapes = properties["num_shapes"]
        
        if num_shapes == 1:
            # Single shape: just describe it
            desc, fol_preds, choices = self._get_shape_description(0, canvas_data, 0)
            fol_structure = f"∃x0({' ∧ '.join(fol_preds)})"
            return desc, fol_structure, choices
        
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

    def generate_caption_only(self, canvas_data: Dict) -> Tuple[str, ChoicesT]:
        """
        Generate only the natural language caption (for backward compatibility).
        
        Returns:
            Tuple of (caption_string, choices_dict)
        """
        caption, fol_structure, choices = self.generate_caption(canvas_data)
        return caption, choices

    def generate_fol_only(self, canvas_data: Dict) -> str:
        """
        Generate only the FOL structure.
        
        Returns:
            FOL structure string
        """
        caption, fol_structure, choices = self.generate_caption(canvas_data)
        return fol_structure


def create_multi_shape_composer(img_size: int = 32) -> MultiShapeComposer:
    """Factory function to create a multi-shape composer."""
    return MultiShapeComposer(img_size=img_size)


# Example usage and testing
if __name__ == "__main__":
    # Test the composer with example data
    composer = create_multi_shape_composer(32)
    
    # Test cases with different numbers of shapes
    test_cases = [
        {
            "name": "Two Shapes",
            "canvas": {
                "classes": np.array([3, 1]),  # circle, square
                "sizes": np.array([12, 8]),   # large, small
                "colors": np.array([[255, 0, 0], [0, 0, 255]]),  # red, blue
                "locations": np.array([[10, 15], [20, 15]]),  # side by side
                "num_shapes": 2,
            }
        },
        {
            "name": "Three Shapes",
            "canvas": {
                "classes": np.array([3, 1, 6]),  # circle, square, diamond
                "sizes": np.array([14, 8, 10]),   # large, small, medium
                "colors": np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]]),  # red, green, blue
                "locations": np.array([[8, 8], [24, 8], [16, 24]]),  # triangle formation
                "num_shapes": 3,
            }
        },
        {
            "name": "Four Shapes",
            "canvas": {
                "classes": np.array([3, 1, 6, 4]),  # circle, square, diamond, star
                "sizes": np.array([12, 6, 10, 14]),   # medium, small, medium, large
                "colors": np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]),  # red, green, blue, yellow
                "locations": np.array([[8, 8], [24, 8], [8, 24], [24, 24]]),  # corners
                "num_shapes": 4,
            }
        },
        {
            "name": "Five Shapes (Scattered)",
            "canvas": {
                "classes": np.array([3, 1, 6, 4, 0]),  # circle, square, diamond, star, triangle
                "sizes": np.array([8, 6, 12, 10, 7]),   # mixed sizes
                "colors": np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]),  # various colors
                "locations": np.array([[5, 5], [27, 5], [16, 16], [5, 27], [27, 27]]),  # scattered
                "num_shapes": 5,
            }
        }
    ]
    
    print("=== CAPTION GENERATION EXAMPLES ===")
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print("Example captions with FOL structures:")
        
        for i in range(2):  # Generate 2 examples for each test case
            caption, fol_structure, choices = composer.generate_caption(test_case['canvas'])
            print(f"{i+1}. Natural Language: {caption}")
            print(f"   FOL Structure: {fol_structure}")
            print(f"   Strategy: {choices.get('template', 'unknown')}")
            
            # Show some key choices for understanding
            key_choices = {k: v for k, v in choices.items() if not k.startswith('shape_') and not k.startswith('size_') and not k.startswith('color_')}
            if key_choices:
                print(f"   Key Choices: {key_choices}")
            print()
    
    print("\n=== STRATEGY-SPECIFIC EXAMPLES ===")
    # Test specific strategies with a 3-shape canvas
    three_shape_canvas = test_cases[1]['canvas']  # Use the three shapes test case
    
    print("Forcing different generation strategies:")
    
    # Test simple conjunction
    caption, fol_structure, choices = composer._generate_simple_conjunction(three_shape_canvas, {"num_shapes": 3})
    print(f"1. Simple Conjunction:")
    print(f"   Caption: {caption}")
    print(f"   FOL: {fol_structure}")
    
    # Test spatial relationship
    caption, fol_structure, choices = composer._generate_spatial_relationship(three_shape_canvas, {"num_shapes": 3})
    print(f"\n2. Spatial Relationship:")
    print(f"   Caption: {caption}")
    print(f"   FOL: {fol_structure}")
    
    # Test comparative description
    properties = composer._analyze_canvas_properties(three_shape_canvas)
    caption, fol_structure, choices = composer._generate_comparative_description(three_shape_canvas, properties)
    print(f"\n3. Comparative Description:")
    print(f"   Caption: {caption}")
    print(f"   FOL: {fol_structure}")
    
    # Test positional description
    caption, fol_structure, choices = composer._generate_positional_description(three_shape_canvas, properties)
    print(f"\n4. Positional Description:")
    print(f"   Caption: {caption}")
    print(f"   FOL: {fol_structure}")
    
    print("\n=== CONVENIENCE METHODS ===")
    # Demonstrate convenience methods
    test_canvas = test_cases[0]['canvas']  # Use two shapes
    
    caption_only, choices = composer.generate_caption_only(test_canvas)
    fol_only = composer.generate_fol_only(test_canvas)
    
    print("Using convenience methods:")
    print(f"Caption only: {caption_only}")
    print(f"FOL only: {fol_only}")
    
    print("\n=== FOL PREDICATE MAPPING EXAMPLES ===")
    print("Shape index to FOL predicate mapping:")
    from simple_shapes_dataset.text.language_templates import SHAPE_PREDICATES
    for idx, predicate in SHAPE_PREDICATES.items():
        print(f"  {idx}: {predicate}")
    
    print("\nColor RGB to FOL predicate examples:")
    example_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for color in example_colors:
        predicate = get_closest_color_predicate(color)
        print(f"  {color}: {predicate}")
    
    print(f"\nSize categories (for {composer.img_size}x{composer.img_size} canvas):")
    example_sizes = [2, 5, 8, 12, 16]
    for size in example_sizes:
        category = composer.size_config.get_size_category(size)
        fol_pred = get_size_predicate(category)
        print(f"  Size {size}: {category} → {fol_pred}")
    
    print("\n=== SPATIAL RELATIONSHIP EXAMPLES ===")
    print("Spatial relationships and their FOL predicates:")
    from simple_shapes_dataset.text.language_templates import SPATIAL_PREDICATES
    for relation, fol_pred in SPATIAL_PREDICATES.items():
        print(f"  '{relation}' → {fol_pred}(x,y)")
    
    print(f"\nSpatial thresholds (as ratio of {composer.img_size}px canvas):")
    print(f"  Near: < {ThresholdConfig.NEAR_THRESHOLD:.0%}")
    print(f"  Far: > {ThresholdConfig.FAR_THRESHOLD:.0%}")
    
    print("\n=== AVAILABLE TEMPLATES ===")
    print("Sentence templates by category:")
    for template_type, templates in SENTENCE_TEMPLATES.items():
        print(f"  {template_type}:")
        for template in templates[:2]:  # Show first 2 examples
            print(f"    - {template}")
        if len(templates) > 2:
            print(f"    ... and {len(templates) - 2} more")
        print()
    
    print("\n=== SIZE RANGE RECOMMENDATIONS ===")
    print(f"Canvas size: {composer.img_size}x{composer.img_size}")
    print("Size ranges (aligned with dataset generation):")
    small_range, medium_range, large_range = composer.size_config.get_size_ranges()
    print(f"  Small: {small_range[0]}-{small_range[1]}px")
    print(f"  Medium: {medium_range[0]}-{medium_range[1]}px")
    print(f"  Large: {large_range[0]}-{large_range[1]}px")
    
    # Test with different canvas sizes
    print(f"\n=== COMPARISON ACROSS CANVAS SIZES ===")
    for canvas_size in [32, 64, 128, 224]:
        test_composer = create_multi_shape_composer(canvas_size)
        small_range, medium_range, large_range = test_composer.size_config.get_size_ranges()
        print(f"  {canvas_size:3d}x{canvas_size:<3d}: small:{small_range[0]}-{small_range[1]}px, medium:{medium_range[0]}-{medium_range[1]}px, large:{large_range[0]}-{large_range[1]}px")
