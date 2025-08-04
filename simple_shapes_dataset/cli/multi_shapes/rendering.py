"""Rendering functions for multi-shapes dataset."""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from tqdm import tqdm

from simple_shapes_dataset.cli.utils import generate_image

from .dataset import MultiShapesDataset


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
        
        # Generate only the actual shapes for this canvas
        num_shapes_in_canvas = dataset.num_shapes[canvas_idx]
        for shape_idx in range(num_shapes_in_canvas):
            cls = dataset.classes[canvas_idx, shape_idx]
            location = dataset.locations[canvas_idx, shape_idx]
            size = dataset.sizes[canvas_idx, shape_idx]
            rotation = dataset.rotations[canvas_idx, shape_idx]
            color = dataset.colors[canvas_idx, shape_idx]
            
            generate_image(ax, cls, location, size, rotation, color, imsize)
        
        plt.tight_layout(pad=0)
        plt.savefig(path_file, dpi=dpi, format="png")
        plt.close(fig)
