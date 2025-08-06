#!/usr/bin/env python3
"""
Dataset loader and inspector for multi-shape datasets.

This script loads a multi-shape dataset and displays all attributes for sample images,
including the new FOL-structured captions if they exist.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class MultiShapeDatasetLoader:
    """Loader for multi-shape datasets with all attributes."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.metadata = self._load_metadata()
        self.splits = ["train", "val", "test"]
        
        # Shape type mappings
        self.shape_names = {
            0: "diamond",
            1: "oval", 
            2: "triangle",
            3: "circle",
            4: "square",
            5: "star",
            6: "heart"
        }
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata if available."""
        metadata = {}
        
        # Try to load metadata.txt
        metadata_file = self.dataset_path / "metadata.txt"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        # Try to convert to appropriate type
                        value = value.strip()
                        if value.lower() in ['true', 'false']:
                            metadata[key] = value.lower() == 'true'
                        elif value.isdigit():
                            metadata[key] = int(value)
                        else:
                            try:
                                metadata[key] = float(value)
                            except ValueError:
                                metadata[key] = value
        
        return metadata
    
    def _load_split_data(self, split: str) -> Dict[str, np.ndarray]:
        """Load all data for a specific split."""
        data = {}
        
        # Core data files
        core_files = {
            "labels": f"{split}_labels.npy",
            "images": None,  # Images are in subdirectories
        }
        
        # Optional data files
        optional_files = {
            "captions": f"{split}_captions.npy",
            "caption_choices": f"{split}_caption_choices.npy", 
            "latent": f"{split}_latent.npy",
            "unpaired": f"{split}_unpaired.npy",
        }
        
        # Load core files
        for name, filename in core_files.items():
            if filename and (self.dataset_path / filename).exists():
                data[name] = np.load(self.dataset_path / filename, allow_pickle=True)
        
        # Load optional files
        for name, filename in optional_files.items():
            filepath = self.dataset_path / filename
            if filepath.exists():
                data[name] = np.load(filepath, allow_pickle=True)
                print(f"✓ Loaded {name} for {split}")
            else:
                print(f"✗ No {name} found for {split}")
        
        return data
    
    def get_image_path(self, split: str, image_idx: int) -> Path:
        """Get the path to a specific image."""
        return self.dataset_path / split / f"{image_idx}.png"
    
    def parse_labels(self, labels: np.ndarray, canvas_idx: int) -> Dict[str, Any]:
        """Parse the labels for a specific canvas."""
        # Labels structure for multi-shapes:
        # Each row: [canvas_idx, shape_idx, class, location_x, location_y, size, rotation, color_r, color_g, color_b, hls_h, hls_l, hls_s, unpaired]
        
        # Find all shapes belonging to this canvas
        canvas_shapes = []
        
        # Filter rows that belong to this canvas
        canvas_mask = labels[:, 0] == canvas_idx
        canvas_shape_rows = labels[canvas_mask]
        
        img_size = self.metadata.get('img_size', 224)
        
        for row in canvas_shape_rows:
            canvas_id = int(row[0])
            shape_idx = int(row[1])
            shape_class = int(row[2])
            x = int(row[3])
            y = int(row[4])
            size = int(row[5])
            rotation = float(row[6])
            r = int(row[7])
            g = int(row[8])
            b = int(row[9])
            h = int(row[10])
            l = int(row[11])
            s = int(row[12])
            unpaired = float(row[13])
            
            # Calculate size as percentage of canvas for display
            size_percentage = (size / img_size) * 100
            
            shape_data = {
                "canvas_idx": canvas_id,
                "shape_idx": shape_idx,
                "shape_type": shape_class,
                "location": (x, y),
                "size": size,
                "size_percentage": size_percentage,
                "rotation": rotation,
                "color_rgb": (r, g, b),
                "color_hls": (h, l, s),
                "unpaired": unpaired,
                "shape_name": self.shape_names.get(shape_class, f"unknown_{shape_class}")
            }
            canvas_shapes.append(shape_data)
        
        # Sort by shape_idx to maintain order
        canvas_shapes.sort(key=lambda x: x["shape_idx"])
        
        return {
            "shapes": canvas_shapes,
            "num_shapes": len(canvas_shapes)
        }
    
    def display_image_attributes(self, split: str = "train", image_idx: int = 0):
        """Display all attributes for a specific image."""
        print(f"\n{'='*80}")
        print(f"DATASET INSPECTION: {self.dataset_path}")
        print(f"Split: {split}, Image Index: {image_idx}")
        print(f"{'='*80}")
        
        # Display metadata
        if self.metadata:
            print("\n--- DATASET METADATA ---")
            for key, value in self.metadata.items():
                print(f"{key}: {value}")
        
        # Load split data
        data = self._load_split_data(split)
        
        # Check if image exists
        image_path = self.get_image_path(split, image_idx)
        print(f"\n--- IMAGE INFO ---")
        print(f"Image path: {image_path}")
        print(f"Image exists: {image_path.exists()}")
        
        # Display labels/shapes
        if "labels" in data:
            print(f"\n--- SHAPE ATTRIBUTES ---")
            parsed = self.parse_labels(data["labels"], image_idx)
            print(f"Number of shapes: {parsed['num_shapes']}")
            
            for i, shape in enumerate(parsed["shapes"]):
                print(f"\nShape {i+1} (shape_idx={shape['shape_idx']}):")
                print(f"  Type: {shape['shape_name']} (class {shape['shape_type']})")
                print(f"  Location: {shape['location']} (x, y)")
                print(f"  Size: {shape['size']} pixels ({shape['size_percentage']:.1f}% of canvas)")
                print(f"  Rotation: {shape['rotation']:.3f} radians")
                print(f"  Color RGB: {shape['color_rgb']}")
                print(f"  Color HLS: {shape['color_hls']}")
                print(f"  Unpaired attr: {shape['unpaired']:.3f}")
        
        # Display captions if available
        if "captions" in data and image_idx < len(data["captions"]):
            print(f"\n--- FOL CAPTION ---")
            caption = data["captions"][image_idx]
            print(f"Caption: {caption}")
            
            if "caption_choices" in data and image_idx < len(data["caption_choices"]):
                choices = data["caption_choices"][image_idx]
                print(f"Caption choices/metadata:")
                if isinstance(choices, dict):
                    for key, value in choices.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {choices}")
        
        # Display BERT embeddings info
        if "latent" in data and image_idx < len(data["latent"]):
            latent = data["latent"][image_idx]
            print(f"\n--- BERT EMBEDDINGS ---")
            print(f"Embedding shape: {latent.shape}")
            print(f"Embedding stats: mean={np.mean(latent):.4f}, std={np.std(latent):.4f}")
            print(f"First 10 values: {latent[:10]}")
        
        # Display unpaired attributes
        if "unpaired" in data and image_idx < len(data["unpaired"]):
            unpaired = data["unpaired"][image_idx]
            print(f"\n--- UNPAIRED ATTRIBUTES ---")
            print(f"Unpaired shape: {unpaired.shape}")
            print(f"Unpaired stats: mean={np.mean(unpaired):.4f}, std={np.std(unpaired):.4f}")
        
        print(f"\n{'='*80}")
    
    def show_dataset_summary(self):
        """Show a summary of the entire dataset."""
        print(f"\n{'='*80}")
        print(f"DATASET SUMMARY: {self.dataset_path}")
        print(f"{'='*80}")
        
        for split in self.splits:
            print(f"\n--- {split.upper()} SPLIT ---")
            data = self._load_split_data(split)
            
            if "labels" in data:
                num_canvases = self._estimate_num_canvases(data["labels"])
                print(f"Number of canvases: {num_canvases}")
                
            for data_type, array in data.items():
                if data_type != "labels":
                    print(f"{data_type}: {array.shape if hasattr(array, 'shape') else len(array)} items")
            
            # Check image directory
            image_dir = self.dataset_path / split
            if image_dir.exists():
                image_files = list(image_dir.glob("*.png"))
                print(f"Image files: {len(image_files)} PNG files")
    
    def _estimate_num_canvases(self, labels: np.ndarray) -> int:
        """Estimate number of canvases from labels array."""
        if not self.metadata:
            return len(labels) // 14  # Assume 2 shapes * 7 values each
        
        shapes_per_canvas = self.metadata.get("max_shapes_per_canvas", 
                                            self.metadata.get("shapes_per_canvas", 2))
        values_per_shape = 7
        return len(labels) // (shapes_per_canvas * values_per_shape)


def main():
    """Main function to test the dataset loader."""
    dataset_path = "/users/xyu110/scratch/variable"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Error: Dataset path {dataset_path} does not exist!")
        print("Please make sure you've generated the dataset first with:")
        print("shapesd create-multi --output_path /users/xyu110/scratch/variable --ntrain 1000 --nval 50 --ntest 50 --spc 15 --var --min_spc 5 --img_size 224 --scale_canvas_shape_ratio 0.2 --captions")
        return
    
    # Create loader
    loader = MultiShapeDatasetLoader(dataset_path)
    
    # Show dataset summary
    loader.show_dataset_summary()
    
    # Show detailed attributes for first few images
    # for split in ["train", "val", "test"]:
    for split in ["train"]:
        for img_idx in [0, 1]:
            try:
                loader.display_image_attributes(split, img_idx)
            except Exception as e:
                print(f"Error loading {split} image {img_idx}: {e}")
                continue
            
            # Only show first 2 images per split to avoid too much output
            if img_idx >= 1:
                break


if __name__ == "__main__":
    main()
