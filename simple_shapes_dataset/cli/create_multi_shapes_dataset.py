import csv
from pathlib import Path

import click
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from simple_shapes_dataset.cli.alignments import create_domain_split
from simple_shapes_dataset.text import composer
from simple_shapes_dataset.text.composer_multi import create_multi_shape_composer
from simple_shapes_dataset.text.composer_qa import create_qa_composer
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
    "--even_sizes",
    is_flag=True,
    default=False,
    help="Use evenly distributed sizes across small/medium/large categories (simple approach).",
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
@click.option(
    "--generate_qa",
    "--qa",
    is_flag=True,
    default=False,
    help="Generate question-answer pairs along with captions (requires --generate_captions)",
)
@click.option(
    "--num_qa_pairs",
    "--nqa",
    default=3,
    type=int,
    help="Number of QA pairs to generate per image (when --generate_qa is enabled)",
)
@click.option(
    "--allowed_shapes",
    "--shapes",
    default=None,
    type=str,
    help="Comma-separated list of allowed shape indices (0-6). Example: '0,1,3' for triangle,square,circle. If not specified, uses all 7 shapes.",
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
    even_sizes: bool,
    background_color: str,
    bert_path: str,
    generate_captions: bool,
    generate_qa: bool,
    num_qa_pairs: int,
    allowed_shapes: str,
) -> None:
    """Generate a multi-shapes dataset with multiple shapes per image."""
    dataset_location = Path(output_path)
    dataset_location.mkdir(exist_ok=True)

    # Process allowed shapes parameter
    allowed_classes = None
    if allowed_shapes:
        try:
            # Parse comma-separated shape indices
            allowed_classes = [int(x.strip()) for x in allowed_shapes.split(',')]
            # Validate indices
            for idx in allowed_classes:
                if idx < 0 or idx > 6:
                    raise ValueError(f"Shape index {idx} is out of range (0-6)")
            
            # Get shape names for display
            shape_names = {
                0: "Triangle", 1: "Square", 2: "Pentagon", 3: "Circle",
                4: "Star", 5: "Heart", 6: "Diamond"
            }
            selected_shapes = [f"{idx}={shape_names[idx]}" for idx in allowed_classes]
            print(f"Using selected shapes: {', '.join(selected_shapes)}")
            
        except ValueError as e:
            raise click.ClickException(f"Invalid allowed_shapes format: {e}")
    else:
        print("Using all 7 available shapes")

    # Calculate size ranges
    actual_min_scale = min_scale
    actual_max_scale = max_scale
    print(f"Manual scaling: {min_scale}-{max_scale} pixels (no scaling)")

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
        variable_shapes,
        min_shapes_per_canvas,
        shapes_per_canvas,  # max_shapes_per_canvas
        even_sizes,
        allowed_classes,
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
        variable_shapes,
        min_shapes_per_canvas,
        shapes_per_canvas,  # max_shapes_per_canvas
        even_sizes,
        allowed_classes,
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
        variable_shapes,
        min_shapes_per_canvas,
        shapes_per_canvas,  # max_shapes_per_canvas
        even_sizes,
        allowed_classes,
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
        print("Generating FOL-structured captions for predicate-argument structure study...")
        print("Note: Captions encode First-Order Logic relationships for semantic analysis")
        
        if generate_qa:
            print(f"Also generating {num_qa_pairs} balanced binding QA pairs per image...")
        
        # Create the multi-shape composer for captions
        multi_composer = create_multi_shape_composer(img_size)
        
        # Create the QA composer for balanced binding questions (if needed)
        qa_composer = None
        if generate_qa:
            qa_composer = create_qa_composer(img_size)
        
        for split, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
            captions = []
            choices = []
            qa_pairs_list = []  # Store QA pairs if requested
            n_canvases = dataset.classes.shape[0]
            
            desc_suffix = " and binding QA pairs" if generate_qa else ""
            for canvas_idx in tqdm(range(n_canvases), desc=f"Generating {split} FOL captions{desc_suffix}"):
                # Get the actual number of shapes in this canvas
                num_shapes = int(dataset.num_shapes[canvas_idx])
                
                # Prepare canvas data for the composer
                canvas_data = {
                    "classes": dataset.classes[canvas_idx],
                    "sizes": dataset.sizes[canvas_idx], 
                    "colors": dataset.colors[canvas_idx],
                    "locations": dataset.locations[canvas_idx],
                    "rotations": dataset.rotations[canvas_idx],
                    "num_shapes": num_shapes,
                }
                
                # Generate FOL-structured caption
                caption, choice = multi_composer.generate_caption(canvas_data)
                captions.append(caption)
                choices.append(choice)
                
                if generate_qa and qa_composer is not None:
                    # Generate balanced binding QA pairs using separate QA composer
                    qa_pairs = qa_composer.generate_comprehensive_binding_qa(canvas_data, num_qa_pairs)
                    qa_pairs_list.append(qa_pairs)
            
            # Save captions and choices
            np.save(str(dataset_location / f"{split}_captions.npy"), captions)
            np.save(str(dataset_location / f"{split}_caption_choices.npy"), choices)
            
            # Save QA pairs if generated
            if generate_qa and qa_pairs_list:
                np.save(str(dataset_location / f"{split}_qa_pairs.npy"), qa_pairs_list)
                print(f"Saved {len(qa_pairs_list)} balanced binding QA pair sets for {split} split")
                
                # Analyze and report QA balance for the first few examples
                if qa_composer and qa_pairs_list:
                    sample_qa = qa_pairs_list[0] if qa_pairs_list else []
                    if sample_qa:
                        balance = qa_composer.analyze_qa_balance(sample_qa)
                        print(f"  {split} QA balance: {balance['yes_count']} yes, {balance['no_count']} no (ratio: {balance['balance_ratio']:.2f})")

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
        "generate_captions": generate_captions,
        "generate_qa": generate_qa,
        "num_qa_pairs": num_qa_pairs if generate_qa else 0,
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
    
    if generate_captions:
        print("✅ FOL-structured captions generated for predicate-argument structure analysis")
        if generate_qa:
            print("✅ Balanced binding QA pairs generated for compositional understanding evaluation")
