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
    parse_allowed_shapes,
    parse_allowed_colors,
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
    "--qa_type",
    type=click.Choice(["binding", "counting", "both"], case_sensitive=False),
    default="binding",
    help="Type of QA to generate: binding (yes/no), counting (numeric), or both.",
)
@click.option(
    "--num_qa_pairs",
    "--nqa",
    default=3,
    type=int,
    help="Number of QA pairs to generate per image (when --generate_qa is enabled)",
)
@click.option(
    "--shapes",
    type=str,
    default=None,
    help="Restrict shapes on canvas. Comma-separated list of ids or names, e.g. '3,4' or 'circle,square'",
)
@click.option(
    "--colors",
    type=str,
    default=None,
    help="Restrict colors on canvas. Comma-separated color names from COLORS_LARGE_SET, e.g. 'red,green'",
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
    qa_type: str,
    num_qa_pairs: int,
    shapes: str | None,
    colors: str | None,
) -> None:
    """Generate a multi-shapes dataset with multiple shapes per image."""
    dataset_location = Path(output_path)
    dataset_location.mkdir(exist_ok=True)

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
    
    # Parse optional shape and color restrictions
    allowed_shape_ids = parse_allowed_shapes(shapes)
    allowed_color_palette = parse_allowed_colors(colors)

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
        allowed_shape_ids=allowed_shape_ids,
        allowed_color_palette=allowed_color_palette,
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
        allowed_shape_ids=allowed_shape_ids,
        allowed_color_palette=allowed_color_palette,
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
        allowed_shape_ids=allowed_shape_ids,
        allowed_color_palette=allowed_color_palette,
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

    # Only generate counting QA if requested
    if generate_qa and qa_type.lower() == "counting":
        print("Generating counting QA questions per image...")
        
        # Create the QA composer for counting questions
        qa_composer = create_qa_composer(img_size)
        
        for split, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
            qa_pairs_list = []
            n_canvases = dataset.classes.shape[0]

            for canvas_idx in tqdm(range(n_canvases), desc=f"Generating {split} counting QA"):
                # Get the actual number of shapes in this canvas
                num_shapes = int(dataset.num_shapes[canvas_idx])
                
                # Prepare canvas data for the composer (slice to only include actual shapes)
                canvas_data = {
                    "classes": dataset.classes[canvas_idx][:num_shapes],
                    "sizes": dataset.sizes[canvas_idx][:num_shapes], 
                    "colors": dataset.colors[canvas_idx][:num_shapes],
                    "locations": dataset.locations[canvas_idx][:num_shapes],
                    "rotations": dataset.rotations[canvas_idx][:num_shapes],
                    "num_shapes": num_shapes,
                }
                
                # Generate counting QA
                qa_pairs = qa_composer.generate_counting_qa(canvas_data, max_questions=num_qa_pairs)
                qa_pairs_list.append(qa_pairs)
            
            # Save QA pairs
            np.save(str(dataset_location / f"{split}_qa_pairs.npy"), qa_pairs_list)
            print(f"  Saved {split} QA pairs: {len(qa_pairs_list)} canvases, ~{num_qa_pairs} questions each")
    
    # === COMMENTED OUT: Other caption generation modes (FOL, binding QA) ===
    # These are not yet tested/working. Uncomment when ready.
    #
    # if generate_captions:
    #     qa_type_lower = qa_type.lower()
    #     if qa_type_lower != "counting":
    #         print("Generating FOL-structured captions for predicate-argument structure study...")
    #         print("Note: Captions encode First-Order Logic relationships for semantic analysis")
    #         if generate_qa and qa_type_lower == "binding":
    #             print(f"Also generating {num_qa_pairs} balanced binding QA pairs per image...")
    #     else:
    #         # Counting mode: no FOL/binding related prints
    #         print("Generating natural-language captions...")
    #         if generate_qa:
    #             print("Also generating counting QA questions per image...")
    #     
    #     # Create the multi-shape composer for captions
    #     multi_composer = create_multi_shape_composer(img_size)
    #     
    #     # Create the QA composer for balanced binding questions (if needed)
    #     qa_composer = None
    #     if generate_qa:
    #         qa_composer = create_qa_composer(img_size)
    #     
    #     for split, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
    #         captions = []
    #         choices = []
    #         qa_pairs_list = []  # Store QA pairs if requested
    #         n_canvases = dataset.classes.shape[0]
    #         
    #         # Progress description
    #         if qa_type_lower == "counting":
    #             tqdm_desc = f"Generating {split} captions and counting QA" if generate_qa else f"Generating {split} captions"
    #         else:
    #             tqdm_desc = f"Generating {split} FOL captions" + (" and binding QA pairs" if generate_qa and qa_type_lower=="binding" else "")
    #
    #         for canvas_idx in tqdm(range(n_canvases), desc=tqdm_desc):
    #             # Get the actual number of shapes in this canvas
    #             num_shapes = int(dataset.num_shapes[canvas_idx])
    #             
    #             # Prepare canvas data for the composer (slice to only include actual shapes)
    #             canvas_data = {
    #                 "classes": dataset.classes[canvas_idx][:num_shapes],
    #                 "sizes": dataset.sizes[canvas_idx][:num_shapes], 
    #                 "colors": dataset.colors[canvas_idx][:num_shapes],
    #                 "locations": dataset.locations[canvas_idx][:num_shapes],
    #                 "rotations": dataset.rotations[canvas_idx][:num_shapes],
    #                 "num_shapes": num_shapes,
    #             }
    #             
    #             # Generate caption based on qa_type (avoid FOL in counting mode)
    #             if qa_type_lower == "counting":
    #                 caption, choice = multi_composer.generate_caption_only(canvas_data)
    #             else:
    #                 caption, _fol, choice = multi_composer.generate_caption(canvas_data)
    #             captions.append(caption)
    #             choices.append(choice)
    #             
    #             if generate_qa and qa_composer is not None:
    #                 # Generate QA according to requested type
    #                 qa_pairs: list[tuple[str, str]] = []
    #                 if qa_type_lower == "binding":
    #                     qa_pairs = qa_composer.generate_comprehensive_binding_qa(canvas_data, num_qa_pairs)
    #                 elif qa_type_lower == "counting":
    #                     qa_pairs = qa_composer.generate_counting_qa(canvas_data, max_questions=num_qa_pairs)
    #                 else:  # both
    #                     qa_pairs.extend(qa_composer.generate_counting_qa(canvas_data, max_questions=max(2, num_qa_pairs // 2)))
    #                     qa_pairs.extend(qa_composer.generate_comprehensive_binding_qa(canvas_data, max(2, num_qa_pairs // 2)))
    #                 qa_pairs_list.append(qa_pairs)
    #         
    #         # Save captions and choices (no FOL artifacts saved in counting mode)
    #         np.save(str(dataset_location / f"{split}_captions.npy"), captions)
    #         np.save(str(dataset_location / f"{split}_caption_choices.npy"), choices)
    #         
    #         # Save QA pairs if generated
    #         if generate_qa and qa_pairs_list:
    #             np.save(str(dataset_location / f"{split}_qa_pairs.npy"), qa_pairs_list)
    #             # Binding-only reporting
    #             if qa_composer and qa_pairs_list and qa_type_lower == "binding":
    #                 sample_qa = qa_pairs_list[0] if qa_pairs_list else []
    #                 if sample_qa:
    #                     balance = qa_composer.analyze_qa_balance(sample_qa)
    #                     print(f"  {split} QA balance: {balance['yes_count']} yes, {balance['no_count']} no (ratio: {balance['balance_ratio']:.2f})")
    #
    #         # Only compute BERT latents when there are captions to process
    #         if len(captions) > 0:
    #             save_bert_latents(
    #                 captions,
    #                 bert_path,
    #                 dataset_location,
    #                 split,
    #                 torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #                 compute_statistics=(split == "train"),
    #             )

    # create_unpaired_attributes_multi(seed, dataset_location, shapes_per_canvas)

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
        "qa_type": qa_type if generate_qa else "",
        "allowed_shapes": ",".join(map(str, allowed_shape_ids)) if allowed_shape_ids else "",
        "allowed_colors": ",".join([str(c) for c in allowed_color_palette]) if allowed_color_palette else "",
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
        if qa_type.lower() == "counting":
            print("✅ Natural-language captions generated")
            if generate_qa:
                print("✅ Counting QA generated")
        # else:
        #     print("✅ FOL-structured captions generated for predicate-argument structure analysis")
        #     if generate_qa and qa_type.lower() == "binding":
        #         print("✅ Balanced binding QA pairs generated for compositional understanding evaluation")
        #     elif generate_qa and qa_type.lower() == "both":
        #         print("✅ Binding and counting QA generated")
