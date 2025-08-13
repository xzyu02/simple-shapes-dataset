"""Multi-shapes dataset generation modules."""

from .collision import (
    calculate_shape_bounds,
    check_collision,
    estimate_max_shapes_capacity,
    generate_non_colliding_locations,
    validate_canvas_capacity,
)
from .dataset import MultiShapesDataset, generate_multi_shapes_dataset
from .io import load_multi_shapes_labels, save_multi_shapes_labels
from .dataset_torch import MultiShapeDataset, create_train_val_test_loaders
from .rendering import save_multi_shapes_dataset

__all__ = [
    # Collision detection
    "calculate_shape_bounds",
    "check_collision",
    "estimate_max_shapes_capacity", 
    "generate_non_colliding_locations",
    "validate_canvas_capacity",
    # Dataset
    "MultiShapesDataset",
    "generate_multi_shapes_dataset",
    # I/O
    "load_multi_shapes_labels",
    "save_multi_shapes_labels",
    # Loader
    "MultiShapeDataset",
    "create_train_val_test_loaders",
    # Rendering
    "save_multi_shapes_dataset",
]
