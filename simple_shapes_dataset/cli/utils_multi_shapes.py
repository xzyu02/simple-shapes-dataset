from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm

from simple_shapes_dataset.cli.utils import (
    generate_class,
    generate_color,
    generate_location,
    generate_rotation,
    generate_scale,
    generate_unpaired_attr,
    generate_image,
)


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
            f"or adjust --scale_canvas_shape_ratio which affects shape sizes"
        )
        return False, max_capacity, message


@dataclass
class MultiShapesDataset:
    """
    Dataset class for handling multiple shapes per image.
    Each attribute is a 2D array where:
    - First dimension: canvas/image index
    - Second dimension: shape index within that canvas
    """
    classes: np.ndarray      # Shape: (n_canvases, shapes_per_canvas)
    locations: np.ndarray    # Shape: (n_canvases, shapes_per_canvas, 2)
    sizes: np.ndarray        # Shape: (n_canvases, shapes_per_canvas)
    rotations: np.ndarray    # Shape: (n_canvases, shapes_per_canvas)
    colors: np.ndarray       # Shape: (n_canvases, shapes_per_canvas, 3)
    colors_hls: np.ndarray   # Shape: (n_canvases, shapes_per_canvas, 3)
    unpaired: np.ndarray     # Shape: (n_canvases, shapes_per_canvas)


def generate_multi_shapes_dataset(
    n_samples: int,
    min_scale: int,
    max_scale: int,
    min_lightness: int,
    max_lightness: int,
    imsize: int,
    shapes_per_canvas: int = 1,
    scale_canvas_shape_ratio: float = 0.0,
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
        shapes_per_canvas: Number of shapes per image
        scale_canvas_shape_ratio: Ratio to scale shapes based on canvas size
    
    Returns:
        MultiShapesDataset with generated data
        
    Raises:
        ValueError: If shapes_per_canvas is too large for the given canvas and shape sizes
    """
    if scale_canvas_shape_ratio > 0:
        # Scale the shape sizes proportionally to image size
        scale_ratio = (imsize / 32.0) * scale_canvas_shape_ratio
        min_scale = int(min_scale * scale_ratio)
        max_scale = int(max_scale * scale_ratio)

    # Check if the requested number of shapes can fit
    max_capacity = estimate_max_shapes_capacity(imsize, min_scale, max_scale)
    if shapes_per_canvas > max_capacity:
        raise ValueError(
            f"Requested {shapes_per_canvas} shapes per canvas exceeds estimated capacity "
            f"of {max_capacity} for canvas size {imsize}x{imsize} with shape sizes {min_scale}-{max_scale}. "
            f"Try reducing shapes_per_canvas, increasing image size, or reducing shape sizes."
        )

    # Initialize arrays for all canvases
    all_classes = np.zeros((n_samples, shapes_per_canvas), dtype=np.int32)
    all_locations = np.zeros((n_samples, shapes_per_canvas, 2), dtype=np.int32)
    all_sizes = np.zeros((n_samples, shapes_per_canvas), dtype=np.int32)
    all_rotations = np.zeros((n_samples, shapes_per_canvas), dtype=np.float32)
    all_colors = np.zeros((n_samples, shapes_per_canvas, 3), dtype=np.int32)
    all_colors_hls = np.zeros((n_samples, shapes_per_canvas, 3), dtype=np.int32)
    all_unpaired = np.zeros((n_samples, shapes_per_canvas), dtype=np.float32)
    
    print(f"Generating {n_samples} canvases with {shapes_per_canvas} shapes each...")
    print(f"Canvas size: {imsize}x{imsize}, Shape sizes: {min_scale}-{max_scale}")
    print(f"Estimated capacity: {max_capacity} shapes per canvas")
    
    failed_placements = 0
    
    for canvas_idx in tqdm(range(n_samples), desc="Generating canvases"):
        # Generate all attributes for shapes in this canvas at once
        canvas_classes = generate_class(shapes_per_canvas)
        canvas_sizes = generate_scale(shapes_per_canvas, min_scale, max_scale)
        canvas_rotations = generate_rotation(shapes_per_canvas)
        canvas_colors_rgb, canvas_colors_hls = generate_color(shapes_per_canvas, min_lightness, max_lightness)
        canvas_unpaired = generate_unpaired_attr(shapes_per_canvas)
        
        # Generate non-colliding locations
        try:
            canvas_locations = generate_non_colliding_locations(
                shapes_per_canvas, canvas_sizes, imsize
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
            canvas_sizes = generate_scale(shapes_per_canvas, min_scale, max_scale)
            canvas_locations = generate_non_colliding_locations(
                shapes_per_canvas, canvas_sizes, imsize
            )
        
        # Store in arrays
        all_classes[canvas_idx] = canvas_classes
        all_locations[canvas_idx] = canvas_locations
        all_sizes[canvas_idx] = canvas_sizes
        all_rotations[canvas_idx] = canvas_rotations
        all_colors[canvas_idx] = canvas_colors_rgb
        all_colors_hls[canvas_idx] = canvas_colors_hls
        all_unpaired[canvas_idx] = canvas_unpaired
    
    if failed_placements > 0:
        print(f"Warning: {failed_placements} canvases required retry for shape placement")
    
    return MultiShapesDataset(
        classes=all_classes,
        locations=all_locations,
        sizes=all_sizes,
        rotations=all_rotations,
        colors=all_colors,
        colors_hls=all_colors_hls,
        unpaired=all_unpaired,
    )


def save_multi_shapes_dataset(
    path_root: Path,
    dataset: MultiShapesDataset,
    imsize: int,
    background_color: str = "black"
) -> None:
    """
    Save the multi-shapes dataset as images.
    
    Args:
        path_root: Root path to save images
        dataset: MultiShapesDataset to save
        imsize: Image size
        background_color: Background color for images
    """
    dpi = 1
    n_canvases = dataset.classes.shape[0]
    shapes_per_canvas = dataset.classes.shape[1]
    
    for canvas_idx in tqdm(range(n_canvases), desc="Generating images"):
        path_file = path_root / f"{canvas_idx}.png"
        
        fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
        ax = cast(plt.Axes, ax)
        
        # Set background color
        if background_color == "black":
            ax.set_facecolor("black")
        elif background_color == "blue":
            ax.set_facecolor("navy")
        elif background_color == "gray":
            ax.set_facecolor("gray")
        elif background_color == "noise":
            noise = np.random.normal(0.3, 0.1, (imsize, imsize, 3))
            noise = np.clip(noise, 0, 1)
            ax.imshow(noise, extent=[0, imsize, 0, imsize], aspect='equal')
        else:
            ax.set_facecolor("black")
        
        # Generate all shapes for this canvas
        for shape_idx in range(shapes_per_canvas):
            cls = dataset.classes[canvas_idx, shape_idx]
            location = dataset.locations[canvas_idx, shape_idx]
            size = dataset.sizes[canvas_idx, shape_idx]
            rotation = dataset.rotations[canvas_idx, shape_idx]
            color = dataset.colors[canvas_idx, shape_idx]
            
            generate_image(ax, cls, location, size, rotation, color, imsize)
        
        plt.tight_layout(pad=0)
        plt.savefig(path_file, dpi=dpi, format="png")
        plt.close(fig)


def save_multi_shapes_labels(path_root: Path, dataset: MultiShapesDataset) -> None:
    """
    Save the multi-shapes dataset labels to a numpy file.
    
    Args:
        path_root: Root path to save labels
        dataset: MultiShapesDataset to save
    """
    n_canvases, shapes_per_canvas = dataset.classes.shape
    
    # Flatten all data and create labels array
    # Format: [canvas_idx, shape_idx, class, location_x, location_y, size, rotation, color_r, color_g, color_b, hls_h, hls_l, hls_s, unpaired]
    labels_list = []
    
    for canvas_idx in range(n_canvases):
        for shape_idx in range(shapes_per_canvas):
            label = [
                canvas_idx,
                shape_idx,
                dataset.classes[canvas_idx, shape_idx],
                dataset.locations[canvas_idx, shape_idx, 0],
                dataset.locations[canvas_idx, shape_idx, 1],
                dataset.sizes[canvas_idx, shape_idx],
                dataset.rotations[canvas_idx, shape_idx],
                dataset.colors[canvas_idx, shape_idx, 0],
                dataset.colors[canvas_idx, shape_idx, 1],
                dataset.colors[canvas_idx, shape_idx, 2],
                dataset.colors_hls[canvas_idx, shape_idx, 0],
                dataset.colors_hls[canvas_idx, shape_idx, 1],
                dataset.colors_hls[canvas_idx, shape_idx, 2],
                dataset.unpaired[canvas_idx, shape_idx],
            ]
            labels_list.append(label)
    
    labels = np.array(labels_list, dtype=np.float32)
    np.save(path_root, labels)


def load_multi_shapes_labels(path_root: Path) -> MultiShapesDataset:
    """
    Load multi-shapes dataset labels from a numpy file.
    
    Args:
        path_root: Path to the labels file
        
    Returns:
        MultiShapesDataset reconstructed from labels
    """
    labels = np.load(path_root)
    
    # Determine dimensions
    canvas_indices = labels[:, 0].astype(int)
    shape_indices = labels[:, 1].astype(int)
    n_canvases = int(canvas_indices.max() + 1)
    shapes_per_canvas = int(shape_indices.max() + 1)
    
    # Initialize arrays
    classes = np.zeros((n_canvases, shapes_per_canvas), dtype=np.int32)
    locations = np.zeros((n_canvases, shapes_per_canvas, 2), dtype=np.int32)
    sizes = np.zeros((n_canvases, shapes_per_canvas), dtype=np.int32)
    rotations = np.zeros((n_canvases, shapes_per_canvas), dtype=np.float32)
    colors = np.zeros((n_canvases, shapes_per_canvas, 3), dtype=np.int32)
    colors_hls = np.zeros((n_canvases, shapes_per_canvas, 3), dtype=np.int32)
    unpaired = np.zeros((n_canvases, shapes_per_canvas), dtype=np.float32)
    
    # Fill arrays from labels
    for row in labels:
        canvas_idx = int(row[0])
        shape_idx = int(row[1])
        classes[canvas_idx, shape_idx] = int(row[2])
        locations[canvas_idx, shape_idx] = [int(row[3]), int(row[4])]
        sizes[canvas_idx, shape_idx] = int(row[5])
        rotations[canvas_idx, shape_idx] = row[6]
        colors[canvas_idx, shape_idx] = [int(row[7]), int(row[8]), int(row[9])]
        colors_hls[canvas_idx, shape_idx] = [int(row[10]), int(row[11]), int(row[12])]
        unpaired[canvas_idx, shape_idx] = row[13]
    
    return MultiShapesDataset(
        classes=classes,
        locations=locations,
        sizes=sizes,
        rotations=rotations,
        colors=colors,
        colors_hls=colors_hls,
        unpaired=unpaired,
    )
