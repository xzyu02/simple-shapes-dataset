"""
Dedicated QA Composer for Binding Problems in Multi-Shape Canvases.

This module generates balanced question-answer pairs that test compositional understanding
and predicate-argument binding in vision-language models.

Key Features:
- Perfect 50/50 yes/no distribution
- Systematic negative question generation
- Spatial and attribute binding tests
- Focus on compositional understanding

Example Binding Tests:
- "Is the circle next to the blue square red?" → "yes"
- "Is the circle next to the blue square blue?" → "no"
"""

import random
from typing import Dict, List, Tuple, Any
import numpy as np
from simple_shapes_dataset.text.writers import (
    shapes_writer, 
    color_large_set_writer
)


class BindingQAComposer:
    """
    Composer dedicated to generating binding test questions for multi-shape canvases.
    
    Focuses exclusively on:
    1. Spatial binding: "Is the [shape] near the [color] [shape] [color]?"
    2. Attribute binding: "Is the [color] [shape] [size]?"
    3. Perfect yes/no balance: Each positive question has a negative counterpart
    4. Systematic attribute swapping for negatives
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
        
        # Spatial relation vocabularies
        self.spatial_relations = {
            "near": ["near", "close to", "beside", "next to"],
            "far": ["far from", "distant from", "away from"],
            "above": ["above", "over", "on top of"],
            "below": ["below", "under", "beneath"],
            "left_of": ["to the left of", "left of"],
            "right_of": ["to the right of", "right of"],
            "diagonal": ["diagonally from", "at an angle to"],
        }

    def _get_size_description(self, size: int) -> str:
        """Get canvas-aware size description."""
        size_ratio = size / self.img_size
        
        if size_ratio < self.small_threshold:
            return random.choice(["tiny", "very small"])
        elif size_ratio < self.medium_threshold:
            return random.choice(["small", "little"])
        elif size_ratio < self.large_threshold:
            return random.choice(["medium", "average sized", "medium sized"])
        else:
            return random.choice(["large", "big"])

    def _calculate_spatial_relationship(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> str:
        """Calculate spatial relationship between two shapes."""
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

    def _get_reverse_spatial_relation(self, spatial_type: str) -> str:
        """Get the reverse spatial relation for perspective swapping."""
        reverse_map = {
            "left_of": "right_of",
            "right_of": "left_of",
            "above": "below",
            "below": "above",
            "near": "near",
            "far": "far",
            "diagonal": "diagonal",
        }
        return reverse_map.get(spatial_type, "near")

    def _extract_shape_info(self, canvas_data: Dict) -> List[Dict[str, Any]]:
        """Extract shape information for QA generation."""
        num_shapes = canvas_data["num_shapes"]
        shapes_info = []
        
        for i in range(num_shapes):
            shape_type = int(canvas_data["classes"][i])
            color = tuple(canvas_data["colors"][i])
            size = canvas_data["sizes"][i]
            location = tuple(canvas_data["locations"][i])
            
            shape_name, _ = shapes_writer(shape_type)
            color_desc, _ = color_large_set_writer(*color)
            size_desc = self._get_size_description(size)
            
            shapes_info.append({
                "shape": shape_name,
                "color": color_desc,
                "size": size_desc,
                "location": location,
                "index": i
            })
        
        return shapes_info

    def generate_spatial_binding_qa(self, canvas_data: Dict, num_qa_pairs: int = 6) -> List[Tuple[str, str]]:
        """
        Generate spatial binding test questions with equal yes/no distribution.
        
        For every positive question, generates a negative counterpart by swapping attributes.
        Example:
        - Positive: "Is the circle next to the blue square red?" → "yes"
        - Negative: "Is the circle next to the blue square blue?" → "no"
        """
        if canvas_data["num_shapes"] < 2:
            return []
        
        shapes_info = self._extract_shape_info(canvas_data)
        qa_pairs = []
        
        # Select two shapes for spatial relationship
        shape1_idx, shape2_idx = 0, 1
        if len(shapes_info) > 2:
            indices = random.sample(range(len(shapes_info)), 2)
            shape1_idx, shape2_idx = indices
        
        shape1 = shapes_info[shape1_idx]
        shape2 = shapes_info[shape2_idx]
        
        # Calculate spatial relationship
        spatial_type = self._calculate_spatial_relationship(shape1["location"], shape2["location"])
        relation_phrase = random.choice(self.spatial_relations.get(spatial_type, ["near"]))
        
        # === BALANCED YES/NO BINDING TESTS ===
        
        # Type 1: Color binding with spatial relation
        # Positive: "Is the [shape1] [relation] the [color2] [shape2] [color1]?"
        question_pos = f"Is the {shape1['shape']} {relation_phrase} the {shape2['color']} {shape2['shape']} {shape1['color']}?"
        qa_pairs.append((question_pos, "yes"))
        
        # Negative: Swap shape1's color with shape2's color
        question_neg = f"Is the {shape1['shape']} {relation_phrase} the {shape2['color']} {shape2['shape']} {shape2['color']}?"
        qa_pairs.append((question_neg, "no"))
        
        # Type 2: Reverse perspective binding
        reverse_relation = self._get_reverse_spatial_relation(spatial_type)
        reverse_relation_phrase = random.choice(self.spatial_relations.get(reverse_relation, ["near"]))
        
        # Positive: "Is the [shape2] [reverse_relation] the [color1] [shape1] [color2]?"
        question_pos = f"Is the {shape2['shape']} {reverse_relation_phrase} the {shape1['color']} {shape1['shape']} {shape2['color']}?"
        qa_pairs.append((question_pos, "yes"))
        
        # Negative: Swap shape2's color with shape1's color
        question_neg = f"Is the {shape2['shape']} {reverse_relation_phrase} the {shape1['color']} {shape1['shape']} {shape1['color']}?"
        qa_pairs.append((question_neg, "no"))
        
        # Type 3: Size-based binding
        # Positive: "Is the [size1] [shape1] [relation] the [color2] [shape2]?"
        question_pos = f"Is the {shape1['size']} {shape1['shape']} {relation_phrase} the {shape2['color']} {shape2['shape']}?"
        qa_pairs.append((question_pos, "yes"))
        
        # Negative: Swap size with wrong size
        wrong_size = shape2['size'] if shape1['size'] != shape2['size'] else "tiny"
        question_neg = f"Is the {wrong_size} {shape1['shape']} {relation_phrase} the {shape2['color']} {shape2['shape']}?"
        qa_pairs.append((question_neg, "no"))
        
        # Type 4: Simple spatial existence
        # Positive: "Is there a [color1] [shape1] [relation] a [color2] [shape2]?"
        question_pos = f"Is there a {shape1['color']} {shape1['shape']} {relation_phrase} a {shape2['color']} {shape2['shape']}?"
        qa_pairs.append((question_pos, "yes"))
        
        # Negative: Swap colors
        question_neg = f"Is there a {shape2['color']} {shape1['shape']} {relation_phrase} a {shape1['color']} {shape2['shape']}?"
        qa_pairs.append((question_neg, "no"))
        
        # Type 5: Complex triple binding
        # Positive: "Is the [size1] [color1] [shape1] [relation] the [shape2]?"
        question_pos = f"Is the {shape1['size']} {shape1['color']} {shape1['shape']} {relation_phrase} the {shape2['shape']}?"
        qa_pairs.append((question_pos, "yes"))
        
        # Negative: Swap color in the complex description
        question_neg = f"Is the {shape1['size']} {shape2['color']} {shape1['shape']} {relation_phrase} the {shape2['shape']}?"
        qa_pairs.append((question_neg, "no"))
        
        # Limit to requested number (ensure even number for balanced yes/no)
        if len(qa_pairs) > num_qa_pairs:
            # Keep pairs together to maintain balance
            pairs_to_keep = (num_qa_pairs // 2) * 2
            qa_pairs = qa_pairs[:pairs_to_keep]
        
        return qa_pairs

    def generate_attribute_binding_qa(self, canvas_data: Dict, num_qa_pairs: int = 4) -> List[Tuple[str, str]]:
        """
        Generate attribute binding test questions with equal yes/no distribution.
        
        For every positive question, generates a negative counterpart by swapping attributes.
        """
        shapes_info = self._extract_shape_info(canvas_data)
        qa_pairs = []
        
        for i, shape in enumerate(shapes_info):
            # Type 1: Color verification
            # Positive: "Is the [shape] [color]?"
            question_pos = f"Is the {shape['shape']} {shape['color']}?"
            qa_pairs.append((question_pos, "yes"))
            
            # Negative: Use wrong color (from another shape or generic wrong color)
            if len(shapes_info) > 1:
                wrong_color_idx = (i + 1) % len(shapes_info)
                wrong_color = shapes_info[wrong_color_idx]["color"]
            else:
                # If only one shape, use a generic wrong color
                wrong_color = "purple" if shape["color"] != "purple" else "orange"
            
            question_neg = f"Is the {shape['shape']} {wrong_color}?"
            qa_pairs.append((question_neg, "no"))
            
            # Type 2: Size verification
            # Positive: "Is the [color] [shape] [size]?"
            question_pos = f"Is the {shape['color']} {shape['shape']} {shape['size']}?"
            qa_pairs.append((question_pos, "yes"))
            
            # Negative: Use wrong size
            if len(shapes_info) > 1:
                wrong_size_idx = (i + 1) % len(shapes_info)
                wrong_size = shapes_info[wrong_size_idx]["size"]
            else:
                # If only one shape, use a generic wrong size
                wrong_size = "huge" if shape["size"] != "huge" else "tiny"
            
            question_neg = f"Is the {shape['color']} {shape['shape']} {wrong_size}?"
            qa_pairs.append((question_neg, "no"))
        
        # Limit to requested number (ensure even number for balanced yes/no)
        if len(qa_pairs) > num_qa_pairs:
            pairs_to_keep = (num_qa_pairs // 2) * 2
            qa_pairs = qa_pairs[:pairs_to_keep]
        
        return qa_pairs

    def generate_comprehensive_binding_qa(self, canvas_data: Dict, num_qa_pairs: int = 8) -> List[Tuple[str, str]]:
        """
        Generate comprehensive binding test questions with guaranteed equal yes/no distribution.
        
        Combines spatial and attribute binding tests, ensuring each "yes" has a corresponding "no".
        """
        qa_pairs = []
        
        # Add spatial binding questions (if applicable)
        if canvas_data["num_shapes"] >= 2:
            spatial_qa = self.generate_spatial_binding_qa(canvas_data, num_qa_pairs // 2)
            qa_pairs.extend(spatial_qa)
        
        # Add attribute binding questions
        remaining_pairs = max(2, num_qa_pairs - len(qa_pairs))  # Ensure at least 2 for balance
        remaining_pairs = (remaining_pairs // 2) * 2  # Ensure even number
        attribute_qa = self.generate_attribute_binding_qa(canvas_data, remaining_pairs)
        qa_pairs.extend(attribute_qa)
        
        # Final limit to requested number (ensure even)
        final_pairs = (num_qa_pairs // 2) * 2
        qa_pairs = qa_pairs[:final_pairs]
        
        # Shuffle while maintaining yes/no balance
        # Group pairs and shuffle, then flatten
        paired_qa = [(qa_pairs[i], qa_pairs[i+1]) for i in range(0, len(qa_pairs), 2)]
        random.shuffle(paired_qa)
        qa_pairs = [qa for pair in paired_qa for qa in pair]
        
        return qa_pairs

    def analyze_qa_balance(self, qa_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze the balance and statistics of generated QA pairs.
        
        Returns:
            Dict with statistics including yes/no counts, balance ratio, question types
        """
        yes_count = sum(1 for _, answer in qa_pairs if answer.lower() == "yes")
        no_count = sum(1 for _, answer in qa_pairs if answer.lower() == "no")
        total_count = len(qa_pairs)
        
        # Analyze question types
        spatial_count = sum(1 for question, _ in qa_pairs if any(rel in question.lower() for rel in ["next to", "near", "above", "below", "left", "right"]))
        attribute_count = total_count - spatial_count
        
        balance_ratio = yes_count / total_count if total_count > 0 else 0
        
        return {
            "total_questions": total_count,
            "yes_count": yes_count,
            "no_count": no_count,
            "balance_ratio": balance_ratio,
            "is_balanced": abs(balance_ratio - 0.5) < 0.01,  # Within 1% of perfect balance
            "spatial_questions": spatial_count,
            "attribute_questions": attribute_count,
        }

    def generate_counting_qa(self,
                             canvas_data: Dict,
                             max_questions: int = 6) -> List[Tuple[str, str]]:
        """
        Generate counting questions for shapes and color-specific shapes.

        Examples:
        - "How many squares are on the image?" -> "3"
        - "How many green circles are on the image?" -> "1"

        Returns a list of (question, numeric_answer) pairs.
        
        Note: Counts based on actual shape class IDs and RGB colors, not text descriptions.
        """
        num_shapes = int(canvas_data.get("num_shapes", 0))
        if num_shapes <= 0:
            return []

        from collections import Counter
        
        # Work with raw class IDs and colors for accurate counting
        classes = canvas_data["classes"][:num_shapes]
        colors = canvas_data["colors"][:num_shapes]
        
        # Convert colors to tuples for hashability
        color_tuples = [tuple(c) for c in colors]
        
        shape_class_counts = Counter(classes)
        color_counts = Counter(color_tuples)
        color_shape_counts = Counter(zip(color_tuples, classes))
        
        qa_pairs: List[Tuple[str, str]] = []
        
        # Helper for pluralization (simple 's' suffix)
        def pluralize(noun: str, count: int) -> str:
            return noun if count == 1 else noun + "s"
        
        # Add color-only questions
        for color_tuple, count in color_counts.items():
            color_desc, _ = color_large_set_writer(*color_tuple)
            question = f"How many {color_desc} shapes are on the image?"
            qa_pairs.append((question, str(count)))
        
        # Add shape-only questions
        for shape_class, count in shape_class_counts.items():
            # Get canonical name for this shape class
            shape_name, _ = shapes_writer(int(shape_class))
            question = f"How many {pluralize(shape_name, 2)} are on the image?"
            qa_pairs.append((question, str(count)))
 
        # Add color+shape questions
        for (color_tuple, shape_class), count in color_shape_counts.items():
            # Get canonical names
            shape_name, _ = shapes_writer(int(shape_class))
            color_desc, _ = color_large_set_writer(*color_tuple)
            question = f"How many {color_desc} {pluralize(shape_name, 2)} are on the image?"
            qa_pairs.append((question, str(count)))

        return qa_pairs


def create_qa_composer(img_size: int = 32) -> BindingQAComposer:
    """Factory function to create a binding QA composer."""
    return BindingQAComposer(img_size=img_size)
