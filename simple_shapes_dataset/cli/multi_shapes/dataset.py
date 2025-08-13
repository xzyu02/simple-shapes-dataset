"""Multi-shapes dataset class and generation functions."""

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from simple_shapes_dataset.cli.utils import (
    generate_class,
    generate_color,
    generate_rotation,
    generate_scale,
    generate_unpaired_attr,
)

from .collision import estimate_max_shapes_capacity, generate_non_colliding_locations


@dataclass
class MultiShapesDataset:
    """
    Dataset class for handling multiple shapes per image.
    Each attribute is a 2D array where:
    - First dimension: canvas/image index
    - Second dimension: shape index within that canvas
    """
    classes: np.ndarray      # Shape: (n_canvases, max_shapes_per_canvas)
    locations: np.ndarray    # Shape: (n_canvases, max_shapes_per_canvas, 2)
    sizes: np.ndarray        # Shape: (n_canvases, max_shapes_per_canvas)
    rotations: np.ndarray    # Shape: (n_canvases, max_shapes_per_canvas)
    colors: np.ndarray       # Shape: (n_canvases, max_shapes_per_canvas, 3)
    colors_hls: np.ndarray   # Shape: (n_canvases, max_shapes_per_canvas, 3)
    unpaired: np.ndarray     # Shape: (n_canvases, max_shapes_per_canvas)
    num_shapes: np.ndarray   # Shape: (n_canvases,) - number of actual shapes per canvas


def generate_multi_shapes_dataset(
    n_samples: int,
    min_scale: int,
    max_scale: int,
    min_lightness: int,
    max_lightness: int,
    imsize: int,
    shapes_per_canvas: int = 1,
    variable_shapes: bool = False,
    min_shapes_per_canvas: int = 1,
    max_shapes_per_canvas: int | None = None,
    even_sizes: bool = False,
    allowed_classes: list[int] | None = None,
) -> MultiShapesDataset:
    """
    Generate a dataset with multiple shapes per image.
    
    Args:
        n_samples: Number of images/canvases to generate
        min_scale: Minimum shape scale
        max_scale: Maximum shape scale
        min_lightness: Minimum color lightness
        max_lightness: Maximum color lightness
        imsize: Image size (width and height)
        shapes_per_canvas: Fixed number of shapes per image (used when variable_shapes=False)
        variable_shapes: If True, randomly vary the number of shapes per canvas
        min_shapes_per_canvas: Minimum shapes when variable_shapes=True
        max_shapes_per_canvas: Maximum shapes when variable_shapes=True (defaults to shapes_per_canvas)
        even_sizes: If True, use evenly distributed sizes across small/medium/large categories
        allowed_classes: List of allowed shape class indices (0-6). If None, uses all 7 shapes.
    
    Returns:
        MultiShapesDataset with generated data
        
    Raises:
        ValueError: If shapes_per_canvas is too large for the given canvas and shape sizes
    """
    # Determine the maximum shapes per canvas for capacity checking and array sizing
    if variable_shapes:
        if max_shapes_per_canvas is None:
            max_shapes_per_canvas = shapes_per_canvas
        max_shapes_for_canvas = max_shapes_per_canvas
    else:
        max_shapes_for_canvas = shapes_per_canvas
        min_shapes_per_canvas = shapes_per_canvas
        max_shapes_per_canvas = shapes_per_canvas

    # Check if the requested number of shapes can fit
    max_capacity = estimate_max_shapes_capacity(imsize, min_scale, max_scale)
    if max_shapes_for_canvas > max_capacity:
        raise ValueError(
            f"Requested max {max_shapes_for_canvas} shapes per canvas exceeds estimated capacity "
            f"of {max_capacity} for canvas size {imsize}x{imsize} with shape sizes {min_scale}-{max_scale}. "
            f"Try reducing max shapes, increasing image size, or reducing shape sizes."
        )

    # Initialize arrays for all canvases (using max shapes for consistent array sizes)
    all_classes = np.zeros((n_samples, max_shapes_for_canvas), dtype=np.int32)
    all_locations = np.zeros((n_samples, max_shapes_for_canvas, 2), dtype=np.int32)
    all_sizes = np.zeros((n_samples, max_shapes_for_canvas), dtype=np.int32)
    all_rotations = np.zeros((n_samples, max_shapes_for_canvas), dtype=np.float32)
    all_colors = np.zeros((n_samples, max_shapes_for_canvas, 3), dtype=np.int32)
    all_colors_hls = np.zeros((n_samples, max_shapes_for_canvas, 3), dtype=np.int32)
    all_unpaired = np.zeros((n_samples, max_shapes_for_canvas), dtype=np.float32)
    all_num_shapes = np.zeros(n_samples, dtype=np.int32)
    
    if variable_shapes:
        print(f"Generating {n_samples} canvases with {min_shapes_per_canvas}-{max_shapes_per_canvas} shapes each...")
    else:
        print(f"Generating {n_samples} canvases with {shapes_per_canvas} shapes each...")
    print(f"Canvas size: {imsize}x{imsize}, Shape sizes: {min_scale}-{max_scale}")
    print(f"Estimated capacity: {max_capacity} shapes per canvas")
    
    failed_placements = 0
    
    for canvas_idx in tqdm(range(n_samples), desc="Generating canvases"):
        # Determine number of shapes for this canvas
        if variable_shapes:
            current_num_shapes = np.random.randint(min_shapes_per_canvas, max_shapes_per_canvas + 1)
        else:
            current_num_shapes = shapes_per_canvas
        
        all_num_shapes[canvas_idx] = current_num_shapes
        
        # Generate all attributes for shapes in this canvas
        canvas_classes = generate_class(current_num_shapes, allowed_classes)
        
        if even_sizes:
            from simple_shapes_dataset.cli.utils import generate_even_scale
            canvas_sizes = generate_even_scale(current_num_shapes, imsize)
        else:
            canvas_sizes = generate_scale(current_num_shapes, min_scale, max_scale)
            
        canvas_rotations = generate_rotation(current_num_shapes)
        canvas_colors_rgb, canvas_colors_hls = generate_color(current_num_shapes, min_lightness, max_lightness)
        canvas_unpaired = generate_unpaired_attr(current_num_shapes)
        
        # Generate non-colliding locations
        try:
            canvas_locations = generate_non_colliding_locations(
                current_num_shapes, canvas_sizes, imsize
            )
        except ValueError as e:
            failed_placements += 1
            if failed_placements > n_samples * 0.1:  # If more than 10% fail
                raise ValueError(
                    f"Too many failed placements ({failed_placements}). "
                    f"Consider reducing shapes_per_canvas or increasing canvas size. "
                    f"Original error: {e}"
                )
            # Retry with slightly different sizes
            if even_sizes:
                from simple_shapes_dataset.cli.utils import generate_even_scale
                canvas_sizes = generate_even_scale(current_num_shapes, imsize)
            else:
                canvas_sizes = generate_scale(current_num_shapes, min_scale, max_scale)
            canvas_locations = generate_non_colliding_locations(
                current_num_shapes, canvas_sizes, imsize
            )
        
        # Store in arrays (only fill up to current_num_shapes)
        all_classes[canvas_idx, :current_num_shapes] = canvas_classes
        all_locations[canvas_idx, :current_num_shapes] = canvas_locations
        all_sizes[canvas_idx, :current_num_shapes] = canvas_sizes
        all_rotations[canvas_idx, :current_num_shapes] = canvas_rotations
        all_colors[canvas_idx, :current_num_shapes] = canvas_colors_rgb
        all_colors_hls[canvas_idx, :current_num_shapes] = canvas_colors_hls
        all_unpaired[canvas_idx, :current_num_shapes] = canvas_unpaired
    
    if failed_placements > 0:
        print(f"Warning: {failed_placements} canvases required retry for shape placement")
    
    if variable_shapes:
        shape_counts = np.bincount(all_num_shapes, minlength=max_shapes_for_canvas + 1)
        print(f"Shape distribution: {dict(enumerate(shape_counts[1:], 1))}")
    
    return MultiShapesDataset(
        classes=all_classes,
        locations=all_locations,
        sizes=all_sizes,
        rotations=all_rotations,
        colors=all_colors,
        colors_hls=all_colors_hls,
        unpaired=all_unpaired,
        num_shapes=all_num_shapes,
    )
