"""Input/output functions for multi-shapes dataset."""

from pathlib import Path

import numpy as np

from .dataset import MultiShapesDataset


def save_multi_shapes_labels(path_root: Path, dataset: MultiShapesDataset) -> None:
    """
    Save the multi-shapes dataset labels to a numpy file.
    
    Args:
        path_root: Root path to save labels
        dataset: MultiShapesDataset to save
    """
    n_canvases = dataset.classes.shape[0]
    
    # Flatten all data and create labels array
    # Format: list of dicts with 'labels' (list of shape data) and 'num_shapes_in_canvas' (int)
    labels_list = []
    
    for canvas_idx in range(n_canvases):
        num_shapes_in_canvas = dataset.num_shapes[canvas_idx]
        labels_list_canvas = []
        for shape_idx in range(num_shapes_in_canvas):
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
            labels_list_canvas.append(label)
        
        # Create dict with labels and num_shapes_in_canvas
        canvas_dict = {
            'labels': labels_list_canvas,
            'num_shapes_in_canvas': int(num_shapes_in_canvas)
        }
        labels_list.append(canvas_dict)
    
    # Save as numpy array with object dtype to preserve dict structure
    labels = np.array(labels_list, dtype=object)
    np.save(path_root, labels)


# TODO: Have not fixed for the changed in save_multi_shapes_labels
def load_multi_shapes_labels(path_root: Path) -> MultiShapesDataset:
    """
    Load multi-shapes dataset labels from a numpy file.
    
    Args:
        path_root: Path to the labels file
        
    Returns:
        MultiShapesDataset reconstructed from labels
    """
    labels = np.load(path_root)

    # Try to load num_shapes separately if it exists
    num_shapes_path = path_root.parent / (path_root.stem + "_num_shapes.npy")
    if num_shapes_path.exists():
        num_shapes = np.load(num_shapes_path)
    else:
        # Fallback: compute from labels
        canvas_indices = labels[:, 0].astype(int)
        num_shapes = np.bincount(canvas_indices)

    # Determine dimensions
    canvas_indices = labels[:, 0].astype(int)
    shape_indices = labels[:, 1].astype(int)
    n_canvases = int(canvas_indices.max() + 1)
    max_shapes_per_canvas = int(shape_indices.max() + 1)

    # Initialize arrays
    classes = np.zeros((n_canvases, max_shapes_per_canvas), dtype=np.int32)
    locations = np.zeros((n_canvases, max_shapes_per_canvas, 2), dtype=np.int32)
    sizes = np.zeros((n_canvases, max_shapes_per_canvas), dtype=np.int32)
    rotations = np.zeros((n_canvases, max_shapes_per_canvas), dtype=np.float32)
    colors = np.zeros((n_canvases, max_shapes_per_canvas, 3), dtype=np.int32)
    colors_hls = np.zeros((n_canvases, max_shapes_per_canvas, 3), dtype=np.int32)
    unpaired = np.zeros((n_canvases, max_shapes_per_canvas), dtype=np.float32)
    
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
        num_shapes=num_shapes,
    )
