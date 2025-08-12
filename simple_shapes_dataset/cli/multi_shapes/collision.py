"""Collision detection and canvas capacity functions for multi-shapes dataset."""

import numpy as np


def calculate_shape_bounds(location: np.ndarray, size: int, margin_factor: float = 1.2) -> tuple[int, int, int, int]:
    """
    Calculate the bounding box for a shape with some margin.
    
    Args:
        location: [x, y] center of the shape
        size: Shape size (radius/half-width)
        margin_factor: Additional margin factor to prevent visual overlap
        
    Returns:
        (min_x, min_y, max_x, max_y) bounding box
    """
    margin = int(size * margin_factor / 2)
    min_x = max(0, location[0] - margin)
    min_y = max(0, location[1] - margin)
    max_x = location[0] + margin
    max_y = location[1] + margin
    return min_x, min_y, max_x, max_y


def check_collision(loc1: np.ndarray, size1: int, loc2: np.ndarray, size2: int, margin_factor: float = 1.2) -> bool:
    """
    Check if two shapes collide given their locations and sizes.
    
    Args:
        loc1, loc2: Shape center locations [x, y]
        size1, size2: Shape sizes
        margin_factor: Additional margin to prevent visual overlap
        
    Returns:
        True if shapes collide, False otherwise
    """
    bounds1 = calculate_shape_bounds(loc1, size1, margin_factor)
    bounds2 = calculate_shape_bounds(loc2, size2, margin_factor)
    
    # Check if bounding boxes overlap
    return not (bounds1[2] < bounds2[0] or  # box1 right < box2 left
                bounds1[0] > bounds2[2] or  # box1 left > box2 right
                bounds1[3] < bounds2[1] or  # box1 bottom < box2 top
                bounds1[1] > bounds2[3])    # box1 top > box2 bottom


def estimate_max_shapes_capacity(imsize: int, min_scale: int, max_scale: int, margin_factor: float = 1.2) -> int:
    """
    Estimate the maximum number of shapes that can fit in a canvas without collision.
    
    Args:
        imsize: Canvas size
        min_scale, max_scale: Shape size range
        margin_factor: Margin factor for collision detection
        
    Returns:
        Estimated maximum number of shapes
    """
    # Use average size for estimation
    avg_size = (min_scale + max_scale) / 2
    shape_area = (avg_size * margin_factor) ** 2
    canvas_area = imsize ** 2
    
    # Conservative estimate: 60% canvas utilization to account for placement constraints
    max_shapes = int(canvas_area * 0.6 / shape_area)
    return max(1, max_shapes)


def generate_non_colliding_locations(
    shapes_per_canvas: int, 
    sizes: np.ndarray, 
    imsize: int, 
    max_attempts: int = 1000,
    margin_factor: float = 1.2
) -> np.ndarray:
    """
    Generate locations for multiple shapes ensuring they don't collide.
    
    Args:
        shapes_per_canvas: Number of shapes to place
        sizes: Array of shape sizes
        imsize: Canvas size
        max_attempts: Maximum attempts to find non-colliding placement
        margin_factor: Margin factor for collision detection
        
    Returns:
        Array of locations [shapes_per_canvas, 2]
        
    Raises:
        ValueError: If unable to place all shapes without collision
    """
    locations = np.zeros((shapes_per_canvas, 2), dtype=np.int32)
    
    for shape_idx in range(shapes_per_canvas):
        size = sizes[shape_idx]
        margin = int(size * margin_factor / 2)
        
        # Calculate valid placement bounds
        min_coord = margin
        max_coord = imsize - margin
        
        if min_coord >= max_coord:
            raise ValueError(f"Shape size {size} too large for canvas size {imsize}")
        
        placed = False
        for attempt in range(max_attempts):
            # Generate random location within bounds
            x = np.random.randint(min_coord, max_coord)
            y = np.random.randint(min_coord, max_coord)
            new_location = np.array([x, y])
            
            # Check collision with existing shapes
            collision = False
            for existing_idx in range(shape_idx):
                existing_location = locations[existing_idx]
                existing_size = sizes[existing_idx]
                
                if check_collision(new_location, size, existing_location, existing_size, margin_factor):
                    collision = True
                    break
            
            if not collision:
                locations[shape_idx] = new_location
                placed = True
                break
        
        if not placed:
            raise ValueError(
                f"Unable to place {shapes_per_canvas} shapes without collision. "
                f"Try reducing shapes_per_canvas, increasing image size, or reducing shape sizes."
            )
    
    return locations


def validate_canvas_capacity(
    imsize: int,
    shapes_per_canvas: int,
    min_scale: int,
    max_scale: int,
    margin_factor: float = 1.2
) -> tuple[bool, int, str]:
    """
    Validate if the requested number of shapes can fit in the canvas.
    
    Args:
        imsize: Canvas size
        shapes_per_canvas: Requested number of shapes
        min_scale, max_scale: Shape size range
        margin_factor: Margin factor for collision detection
        
    Returns:
        (is_valid, max_capacity, message)
    """
    max_capacity = estimate_max_shapes_capacity(imsize, min_scale, max_scale, margin_factor)
    
    if shapes_per_canvas <= max_capacity:
        return True, max_capacity, f"✓ {shapes_per_canvas} shapes can fit (capacity: {max_capacity})"
    else:
        message = (
            f"✗ {shapes_per_canvas} shapes exceed capacity of {max_capacity}. "
            f"Suggestions: "
            f"reduce --spc to {max_capacity}, "
            f"increase --img_size (current: {imsize}), "
            f"or reduce shape sizes (current: {min_scale}-{max_scale})"
        )
        return False, max_capacity, message
