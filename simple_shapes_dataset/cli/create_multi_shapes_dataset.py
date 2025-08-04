import csv
from pathlib import Path

import click
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from simple_shapes_dataset.cli.alignments import create_domain_split
from simple_shapes_dataset.text import composer
from simple_shapes_dataset.version import __version__

from .utils import save_bert_latents
from .multi_shapes import (
    MultiShapesDataset,
    generate_multi_shapes_dataset,
    save_multi_shapes_dataset,
    save_multi_shapes_labels,
    validate_canvas_capacity,
)

matplotlib.use("Agg")


def create_unpaired_attributes_multi(
    seed: int,
    dataset_location: Path,
    shapes_per_canvas: int,
) -> None:
    """Create unpaired attributes for multi-shapes dataset."""
    assert dataset_location.exists()

    # Load the labels to get the correct dimensions
    train_labels = np.load(dataset_location / "train_labels.npy")
    val_labels = np.load(dataset_location / "val_labels.npy")
    test_labels = np.load(dataset_location / "test_labels.npy")
    
    # Each row represents one shape, so divide by shapes_per_canvas to get number of canvases
    num_train_canvases = len(train_labels) // shapes_per_canvas
    num_val_canvases = len(val_labels) // shapes_per_canvas
    num_test_canvases = len(test_labels) // shapes_per_canvas

    np.random.seed(seed)
    train_unpaired = np.random.normal(size=(num_train_canvases, shapes_per_canvas, 32))
    val_unpaired = np.random.normal(size=(num_val_canvases, shapes_per_canvas, 32))
    test_unpaired = np.random.normal(size=(num_test_canvases, shapes_per_canvas, 32))

    np.save(dataset_location / "train_unpaired.npy", train_unpaired)
    np.save(dataset_location / "val_unpaired.npy", val_unpaired)
    np.save(dataset_location / "test_unpaired.npy", test_unpaired)


@click.command("create-multi", help="Create a Multi-Shapes dataset with multiple shapes per image")
@click.option("--seed", "-s", default=0, type=int, help="Random seed")
@click.option("--img_size", default=32, type=int, help="Size of the images")
@click.option(
    "--output_path",
    "-o",
    default="./",
    type=str,
    help="Where to save the dataset",
)
@click.option(
    "--num_train_examples",
    "--ntrain",
    default=10_000,
    type=int,
    help="Number of training examples (canvases)",
)
@click.option(
    "--num_val_examples",
    "--nval",
    default=1_000,
    type=int,
    help="Number of validation examples (canvases)",
)
@click.option(
    "--num_test_examples",
    "--ntest",
    default=1_000,
    type=int,
    help="Number of test examples (canvases)",
)
@click.option(
    "--shapes_per_canvas",
    "--spc",
    default=2,
    type=int,
    help="Number of shapes per canvas/image (fixed mode) or max shapes (variable mode)",
)
@click.option(
    "--variable_shapes",
    "--var",
    is_flag=True,
    default=False,
    help="Enable variable number of shapes per canvas",
)
@click.option(
    "--min_shapes_per_canvas",
    "--min_spc",
    default=1,
    type=int,
    help="Minimum shapes per canvas when using variable_shapes mode",
)
@click.option(
    "--min_scale",
    default=7,
    type=int,
    help="Minimum size of the shapes (in pixels)",
)
@click.option(
    "--max_scale",
    default=14,
    type=int,
    help="Maximum size of the shapes (in pixels)",
)
@click.option(
    "--min_lightness",
    default=46,
    type=int,
    help="Minimum lightness for the shapes' HSL color. Higher values are lighter.",
)
@click.option(
    "--max_lightness",
    default=256,
    type=int,
    help="Maximum lightness for the shapes' HSL color. Higher values are lighter.",
)
@click.option(
    "--scale_canvas_shape_ratio",
    default=0.0,
    type=float,
    help="Ratio to scale shape sizes with image size. 1.0 = proportional scaling, 0.0 = no scaling.",
)
@click.option(
    "--background_color",
    "--bg",
    default="black",
    type=click.Choice(["black", "blue", "gray", "noise"]),
    help="Background color for images: black, blue, gray, or noise (gaussian noise).",
)
@click.option(
    "--bert_path",
    "-b",
    default="bert-base-uncased",
    type=str,
    help="Pretrained BERT model to use",
)
@click.option(
    "--generate_captions",
    "--captions",
    is_flag=True,
    default=False,
    help="Generate text captions and BERT embeddings (experimental for multi-shapes)",
)
def create_multi_shapes_dataset(
    seed: int,
    img_size: int,
    output_path: str,
    num_train_examples: int,
    num_val_examples: int,
    num_test_examples: int,
    shapes_per_canvas: int,
    variable_shapes: bool,
    min_shapes_per_canvas: int,
    min_scale: int,
    max_scale: int,
    min_lightness: int,
    max_lightness: int,
    scale_canvas_shape_ratio: float,
    background_color: str,
    bert_path: str,
    generate_captions: bool,
) -> None:
    """Generate a multi-shapes dataset with multiple shapes per image."""
    dataset_location = Path(output_path)
    dataset_location.mkdir(exist_ok=True)

    # Apply canvas shape ratio scaling if specified
    actual_min_scale = min_scale
    actual_max_scale = max_scale
    if scale_canvas_shape_ratio > 0:
        scale_ratio = (img_size / 32.0) * scale_canvas_shape_ratio
        actual_min_scale = int(min_scale * scale_ratio)
        actual_max_scale = int(max_scale * scale_ratio)

    # Validate canvas capacity before generation
    max_shapes_for_validation = shapes_per_canvas
    if variable_shapes:
        print(f"Variable shapes mode: {min_shapes_per_canvas}-{shapes_per_canvas} shapes per canvas")
    else:
        print(f"Fixed shapes mode: {shapes_per_canvas} shapes per canvas")
    
    is_valid, max_capacity, message = validate_canvas_capacity(
        img_size, max_shapes_for_validation, actual_min_scale, actual_max_scale
    )
    
    print(f"Canvas capacity check: {message}")
    
    if not is_valid:
        raise click.ClickException(
            f"Canvas capacity exceeded. {message}"
        )

    np.random.seed(seed)

    print(f"Generating multi-shapes dataset...")
    
    print("Generating training data...")
    train_dataset = generate_multi_shapes_dataset(
        num_train_examples,
        actual_min_scale,
        actual_max_scale,
        min_lightness,
        max_lightness,
        img_size,
        shapes_per_canvas,
        scale_canvas_shape_ratio,
        variable_shapes,
        min_shapes_per_canvas,
        shapes_per_canvas,  # max_shapes_per_canvas
    )
    
    print("Generating validation data...")
    val_dataset = generate_multi_shapes_dataset(
        num_val_examples,
        actual_min_scale,
        actual_max_scale,
        min_lightness,
        max_lightness,
        img_size,
        shapes_per_canvas,
        scale_canvas_shape_ratio,
        variable_shapes,
        min_shapes_per_canvas,
        shapes_per_canvas,  # max_shapes_per_canvas
    )
    
    print("Generating test data...")
    test_dataset = generate_multi_shapes_dataset(
        num_test_examples,
        actual_min_scale,
        actual_max_scale,
        min_lightness,
        max_lightness,
        img_size,
        shapes_per_canvas,
        scale_canvas_shape_ratio,
        variable_shapes,
        min_shapes_per_canvas,
        shapes_per_canvas,  # max_shapes_per_canvas
    )

    print("Saving labels...")
    save_multi_shapes_labels(dataset_location / "train_labels.npy", train_dataset)
    save_multi_shapes_labels(dataset_location / "val_labels.npy", val_dataset)
    save_multi_shapes_labels(dataset_location / "test_labels.npy", test_dataset)

    print("Saving training images...")
    (dataset_location / "train").mkdir(exist_ok=True)
    save_multi_shapes_dataset(dataset_location / "train", train_dataset, img_size, background_color)
    
    print("Saving validation images...")
    (dataset_location / "val").mkdir(exist_ok=True)
    save_multi_shapes_dataset(dataset_location / "val", val_dataset, img_size, background_color)
    
    print("Saving test images...")
    (dataset_location / "test").mkdir(exist_ok=True)
    save_multi_shapes_dataset(dataset_location / "test", test_dataset, img_size, background_color)

    if generate_captions:
        print("Generating captions (experimental)...")
        print("Note: Caption generation for multi-shapes is experimental and may need adjustment")
        
        for split, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
            captions = []
            choices = []
            n_canvases = dataset.classes.shape[0]
            
            for canvas_idx in tqdm(range(n_canvases), desc=f"Generating {split} captions"):
                # For now, just describe the first shape in each canvas
                # TODO: Implement proper multi-shape caption generation
                shape_data = {
                    "shape": int(dataset.classes[canvas_idx, 0]),
                    "rotation": dataset.rotations[canvas_idx, 0],
                    "color": tuple(dataset.colors[canvas_idx, 0]),
                    "size": dataset.sizes[canvas_idx, 0],
                    "location": tuple(dataset.locations[canvas_idx, 0]),
                }
                caption, choice = composer(shape_data)
                captions.append(f"Canvas with {shapes_per_canvas} shapes: {caption}")
                choices.append(choice)
            
            np.save(str(dataset_location / f"{split}_captions.npy"), captions)
            np.save(str(dataset_location / f"{split}_caption_choices.npy"), choices)

            save_bert_latents(
                captions,
                bert_path,
                dataset_location,
                split,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                compute_statistics=(split == "train"),
            )

    create_unpaired_attributes_multi(seed, dataset_location, shapes_per_canvas)

    # Save metadata
    metadata = {
        "shapes_per_canvas": shapes_per_canvas,
        "variable_shapes": variable_shapes,
        "min_shapes_per_canvas": min_shapes_per_canvas if variable_shapes else shapes_per_canvas,
        "max_shapes_per_canvas": shapes_per_canvas,
        "img_size": img_size,
        "min_scale": min_scale,
        "max_scale": max_scale,
        "actual_min_scale": actual_min_scale,
        "actual_max_scale": actual_max_scale,
        "background_color": background_color,
        "version": __version__,
    }
    
    with open(dataset_location / "metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    with open(dataset_location / "version.txt", "w") as version_file:
        version_file.write(__version__)

    print(f"Multi-shapes dataset created successfully at {dataset_location}")
    if variable_shapes:
        print(f"Dataset contains {min_shapes_per_canvas}-{shapes_per_canvas} shapes per canvas (variable)")
    else:
        print(f"Dataset contains {shapes_per_canvas} shapes per canvas (fixed)")
