#!/usr/bin/env python3
"""
Clean MultiShape Dataset.

A minimal, focused dataset for multi-shape data that provides
clean data loading functionality without analysis or visualization features.
Compatible with PyTorch DataLoader for ML training.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Iterable
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MultiShapeDataset(Dataset):
    """
    Clean dataset for multi-shape data.
    
    Compatible with PyTorch DataLoader for ML training. Can be used as:
    1. Standalone dataset (original functionality)
    2. PyTorch Dataset for DataLoader integration
    
    Args:
        dataset_path (str): Path to the dataset directory
        split (str): Which split to load ('train', 'val', 'test')
        domains (list, optional): List of domains to load. Available: 
            ['images', 'shapes', 'captions', 'caption_choices', 'qa_pairs', 'latent', 'unpaired']
            If None, loads all available data.
        transforms (dict, optional): Dictionary of transforms to apply to specific domains
        qa_format (str): Format for QA data ('dict' or 'tuple')
        
    Example:
        # Load only images and QA pairs with transforms
        dataset = MultiShapeDataset(
            dataset_path="/path/to/dataset",
            split="train",
            domains=['images', 'qa_pairs', 'shapes'],
            transforms={
                'images': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
            }
        )
    """
    
    def __init__(
        self, 
        dataset_path: str, 
        split: str = "train",
        domains: Optional[List[str]] = None,
        transforms: Optional[Dict[str, Callable]] = None,
        qa_format: str = "dict",
        # Legacy parameters for backward compatibility
        load_images: Optional[bool] = None,
        image_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset_path (str): Path to the dataset directory
            split (str): Split to load ('train', 'val', 'test')
            domains (list, optional): List of domains to load. Available:
                ['images', 'shapes', 'captions', 'caption_choices', 'qa_pairs', 'latent', 'unpaired']
                If None, loads all available data.
            transforms (dict, optional): Dictionary of transforms to apply to specific domains
            qa_format (str): 'dict' for {"question": str, "answer": str} or 'tuple' for (question, answer)
            
            # Legacy parameters (for backward compatibility):
            load_images (bool, optional): Deprecated. Use domains=['images'] instead.
            image_transform (callable, optional): Deprecated. Use transforms={'images': transform} instead.
            text_transform (callable, optional): Deprecated. Use transforms={'captions': transform} instead.
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.qa_format = qa_format
        
        # Handle legacy parameters for backward compatibility
        if domains is None:
            if load_images is True:
                domains = ['images', 'shapes', 'captions', 'caption_choices', 'qa_pairs', 'latent', 'unpaired']
            elif load_images is False:
                domains = ['shapes', 'captions', 'caption_choices', 'qa_pairs', 'latent', 'unpaired']
            else:
                # Default: load all available data
                domains = ['images', 'shapes', 'captions', 'caption_choices', 'qa_pairs', 'latent', 'unpaired']
        
        # Handle legacy transforms
        if transforms is None:
            transforms = {}
        if image_transform is not None and 'images' not in transforms:
            transforms['images'] = image_transform
        if text_transform is not None and 'captions' not in transforms:
            transforms['captions'] = text_transform
            
        self.domains = domains
        self.transforms = transforms or {}
        
        # Available domain mappings
        self.available_domains = {
            'images': 'images',
            'shapes': 'labels', 
            'captions': 'captions',
            'caption_choices': 'caption_choices',
            'qa_pairs': 'qa_pairs',
            'latent': 'latent',
            'unpaired': 'unpaired',
        }
        
        # Load metadata
        self.metadata = self._load_metadata()
        
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
        
        # Load split data once during initialization for efficiency
        self.data = self._load_split_data(split)
        self.size = self._calculate_size()
        
        # Default image transform if loading images and no transform specified
        if 'images' in self.domains and 'images' not in self.transforms:
            self.transforms['images'] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
            ])
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata if available."""
        metadata = {}
        metadata_file = self.dataset_path / "metadata.txt"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        value = value.strip()
                        
                        # Convert to appropriate type
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
        """
        Load only the requested domains for a specific split.
        
        Args:
            split (str): Split to load ('train', 'val', or 'test')
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing only the requested data arrays
        """
        valid_splits = ["train", "val", "test"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
        
        data = {}
        
        # Define all possible data files
        all_data_files = {
            "labels": f"{split}_labels.npy",
            "captions": f"{split}_captions.npy",
            "caption_choices": f"{split}_caption_choices.npy", 
            "latent": f"{split}_latent.npy",
            "unpaired": f"{split}_unpaired.npy",
            "qa_pairs": f"{split}_qa_pairs.npy",
            "labels_num_shapes": f"{split}_labels_num_shapes.npy",
        }
        
        # Determine which files to load based on requested domains
        files_to_load = {}
        for domain in self.domains:
            if domain == 'images':
                # Images are loaded on-demand, not here
                continue
            elif domain == 'shapes':
                files_to_load['labels'] = all_data_files['labels']
                if 'labels_num_shapes' in all_data_files:
                    files_to_load['labels_num_shapes'] = all_data_files['labels_num_shapes']
            elif domain in ['captions', 'caption_choices', 'latent', 'unpaired', 'qa_pairs']:
                key = domain
                if key in all_data_files:
                    files_to_load[key] = all_data_files[key]
        
        # Load only the requested files
        for name, filename in files_to_load.items():
            filepath = self.dataset_path / filename
            if filepath.exists():
                data[name] = np.load(filepath, allow_pickle=True)
        
        return data
    
    def _calculate_size(self) -> int:
        """Calculate the number of samples in the dataset."""
        # If QA pairs are available, count individual QA pairs as samples
        if "qa_pairs" in self.data:
            total_qa_pairs = 0
            for canvas_idx in range(len(self.data["qa_pairs"])):
                qa_pairs = self.parse_qa_pairs(self.data["qa_pairs"], canvas_idx)
                total_qa_pairs += len(qa_pairs)
            return total_qa_pairs
        
        # Fallback: Check captions (they're typically 1:1 with canvases)
        if "captions" in self.data:
            return len(self.data["captions"])
        
        # Check image directory
        image_dir = self.dataset_path / self.split
        if image_dir.exists():
            image_files = list(image_dir.glob("*.png"))
            return len(image_files)
        
        # Estimate from labels if available
        if "labels" in self.data:
            shapes_per_canvas = self.metadata.get("max_shapes_per_canvas", 
                                                self.metadata.get("shapes_per_canvas", 2))
            return len(self.data["labels"]) // shapes_per_canvas
        
        return 0
    
    def _map_qa_index_to_canvas(self, qa_idx: int) -> Tuple[int, int]:
        """
        Map a QA index to canvas index and QA pair index within that canvas.
        
        Args:
            qa_idx (int): Global QA index
            
        Returns:
            Tuple[int, int]: (canvas_idx, qa_pair_idx)
        """
        if "qa_pairs" not in self.data:
            raise ValueError("No QA pairs available in dataset")
        
        current_qa_count = 0
        
        for canvas_idx in range(len(self.data["qa_pairs"])):
            canvas_qa_pairs = self.parse_qa_pairs(self.data["qa_pairs"], canvas_idx)
            canvas_qa_count = len(canvas_qa_pairs)
            
            if current_qa_count + canvas_qa_count > qa_idx:
                # This canvas contains our target QA pair
                qa_pair_idx = qa_idx - current_qa_count
                return canvas_idx, qa_pair_idx
            
            current_qa_count += canvas_qa_count
        
        raise IndexError(f"QA index {qa_idx} out of range")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset (PyTorch Dataset interface)."""
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index (PyTorch Dataset interface).
        Each sample contains only the requested domains with applied transforms.
        
        Args:
            idx (int): QA pair index (not canvas index)
            
        Returns:
            Dict[str, Any]: Sample data containing one QA pair with requested domains
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Find which canvas and QA pair this index corresponds to
        canvas_idx, qa_idx = self._map_qa_index_to_canvas(idx)
        
        sample = {
            "qa_idx": idx,
            "canvas_idx": canvas_idx,
            "qa_pair_idx": qa_idx,
            "split": self.split,
        }
        
        # Add image if requested in domains
        if 'images' in self.domains:
            image_path = self.get_image_path(self.split, canvas_idx)
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert('RGB')
                    if 'images' in self.transforms:
                        image = self.transforms['images'](image)
                    sample["image"] = image
                except Exception as e:
                    # If image loading fails, create a dummy tensor
                    sample["image"] = torch.zeros(3, 224, 224)
            else:
                sample["image"] = torch.zeros(3, 224, 224)
            sample["image_path"] = str(image_path)
        
        # Add shapes if requested in domains
        if 'shapes' in self.domains and "labels" in self.data:
            shapes = self.parse_shapes(self.data["labels"], canvas_idx)
            
            # Apply shape transform if specified
            if 'shapes' in self.transforms:
                shapes = self.transforms['shapes'](shapes)
                
            sample["shapes"] = shapes
            sample["num_shapes"] = len(shapes)
            
            # Convert shapes to tensor format for ML training
            if shapes:
                shape_features = []
                for shape in shapes:
                    # Create feature vector: [shape_type, x, y, size, rotation, r, g, b]
                    features = [
                        shape["shape_type"],
                        shape["location"][0] / self.metadata.get('img_size', 224),  # Normalize x
                        shape["location"][1] / self.metadata.get('img_size', 224),  # Normalize y
                        shape["size"] / self.metadata.get('img_size', 224),        # Normalize size
                        shape["rotation"],
                        shape["color_rgb"][0] / 255.0,  # Normalize colors
                        shape["color_rgb"][1] / 255.0,
                        shape["color_rgb"][2] / 255.0,
                    ]
                    shape_features.append(features)
                
                # Pad to fixed size (e.g., max 10 shapes)
                max_shapes = 10
                while len(shape_features) < max_shapes:
                    shape_features.append([0.0] * 8)  # Padding with zeros
                
                sample["shape_features"] = torch.tensor(shape_features[:max_shapes], dtype=torch.float32)
        
        # Add caption if requested in domains
        if 'captions' in self.domains and "captions" in self.data and canvas_idx < len(self.data["captions"]):
            caption = self.data["captions"][canvas_idx]
            if 'captions' in self.transforms:
                caption = self.transforms['captions'](caption)
            sample["caption"] = caption
        
        # Add caption choices if requested in domains
        if 'caption_choices' in self.domains and "caption_choices" in self.data and canvas_idx < len(self.data["caption_choices"]):
            caption_choices = self.data["caption_choices"][canvas_idx]
            if 'caption_choices' in self.transforms:
                caption_choices = self.transforms['caption_choices'](caption_choices)
            sample["caption_choices"] = caption_choices
        
        # Add the specific QA pair if requested in domains
        if 'qa_pairs' in self.domains and "qa_pairs" in self.data:
            all_qa_pairs = self.parse_qa_pairs(self.data["qa_pairs"], canvas_idx)
            
            if qa_idx < len(all_qa_pairs):
                qa_pair = all_qa_pairs[qa_idx]
                
                # Apply QA transform if specified
                if 'qa_pairs' in self.transforms:
                    qa_pair = self.transforms['qa_pairs'](qa_pair)
                
                # Add the specific QA pair
                sample["question"] = qa_pair["question"]
                sample["answer"] = qa_pair["answer"]
                
                # Convert answer to binary label for classification
                answer_label = 1 if qa_pair["answer"].lower() == "yes" else 0
                sample["answer_label"] = torch.tensor(answer_label, dtype=torch.long)
                
                # For backward compatibility, also include in the old format
                if self.qa_format == "tuple":
                    sample["qa_pair"] = (qa_pair["question"], qa_pair["answer"])
                else:
                    sample["qa_pair"] = qa_pair
                
                # Include all QA pairs for the canvas if needed
                sample["all_qa_pairs"] = all_qa_pairs
                sample["num_qa_pairs"] = len(all_qa_pairs)
        
        # Add other requested data types
        for domain in ['latent', 'unpaired']:
            if domain in self.domains and domain in self.data and canvas_idx < len(self.data[domain]):
                data_item = self.data[domain][canvas_idx]
                if domain in self.transforms:
                    data_item = self.transforms[domain](data_item)
                elif isinstance(data_item, np.ndarray):
                    data_item = torch.from_numpy(data_item.astype(np.float32))
                sample[domain] = data_item
        
        return sample
    
    def get_image_path(self, split: str, image_idx: int) -> Path:
        """
        Get the path to a specific image.
        
        Args:
            split (str): Split name
            image_idx (int): Image index
            
        Returns:
            Path: Path to the image file
        """
        return self.dataset_path / split / f"{image_idx}.png"
    
    def parse_shapes(self, labels: np.ndarray, canvas_idx: int) -> List[Dict[str, Any]]:
        """
        Parse shape information for a specific canvas.
        
        Args:
            labels (np.ndarray): Labels array
            canvas_idx (int): Canvas index to parse
            
        Returns:
            List[Dict[str, Any]]: List of shape dictionaries
        """
        # Filter rows that belong to this canvas
        canvas_mask = labels[:, 0] == canvas_idx
        canvas_shape_rows = labels[canvas_mask]
        
        img_size = self.metadata.get('img_size', 224)
        shapes = []
        
        for row in canvas_shape_rows:
            shape_data = {
                "canvas_idx": int(row[0]),
                "shape_idx": int(row[1]),
                "shape_type": int(row[2]),
                "location": (int(row[3]), int(row[4])),
                "size": int(row[5]),
                "rotation": float(row[6]),
                "color_rgb": (int(row[7]), int(row[8]), int(row[9])),
                "color_hls": (int(row[10]), int(row[11]), int(row[12])),
                "unpaired": float(row[13]),
                "shape_name": self.shape_names.get(int(row[2]), f"unknown_{int(row[2])}")
            }
            shapes.append(shape_data)
        
        # Sort by shape_idx to maintain order
        shapes.sort(key=lambda x: x["shape_idx"])
        return shapes
    
    def parse_qa_pairs(self, qa_pairs: np.ndarray, canvas_idx: int) -> List[Dict[str, str]]:
        """
        Parse QA pairs for a specific canvas.
        
        Args:
            qa_pairs (np.ndarray): QA pairs array with shape (n_samples, n_qa_per_sample, 2)
            canvas_idx (int): Canvas index
            
        Returns:
            List[Dict[str, str]]: List of question-answer dictionaries
        """
        if qa_pairs is None or len(qa_pairs) == 0 or canvas_idx >= len(qa_pairs):
            return []
        
        canvas_qa = qa_pairs[canvas_idx]
        parsed_qa = []
        
        # Handle 3D array format: (samples, qa_pairs_per_sample, 2)
        if isinstance(canvas_qa, np.ndarray) and canvas_qa.ndim == 2:
            for qa_pair in canvas_qa:
                if len(qa_pair) >= 2:
                    parsed_qa.append({
                        "question": str(qa_pair[0]),
                        "answer": str(qa_pair[1])
                    })
        # Handle other formats (list, tuple, etc.)
        elif isinstance(canvas_qa, (list, tuple, np.ndarray)):
            for qa_item in canvas_qa:
                if isinstance(qa_item, (list, tuple, np.ndarray)) and len(qa_item) >= 2:
                    parsed_qa.append({
                        "question": str(qa_item[0]),
                        "answer": str(qa_item[1])
                    })
                elif isinstance(qa_item, dict):
                    parsed_qa.append({
                        "question": str(qa_item.get("question", qa_item.get("q", ""))),
                        "answer": str(qa_item.get("answer", qa_item.get("a", "")))
                    })
        
        return parsed_qa
    
    # Legacy methods for backward compatibility
    def load_split_data(self, split: str) -> Dict[str, np.ndarray]:
        """Load all available data for a specific split (legacy method)."""
        return self._load_split_data(split)
    
    def get_sample(self, split: str, canvas_idx: int) -> Dict[str, Any]:
        """Get all data for a specific sample (legacy method)."""
        if split != self.split:
            # Create temporary dataset for different split
            temp_dataset = MultiShapeDataset(
                str(self.dataset_path), 
                split=split, 
                load_images=self.load_images,
                image_transform=self.image_transform,
                text_transform=self.text_transform,
                qa_format=self.qa_format
            )
            return temp_dataset[canvas_idx]
        return self[canvas_idx]
    
    def get_split_size(self, split: str) -> int:
        """Get the number of samples in a split (legacy method)."""
        if split == self.split:
            return len(self)
        
        # For other splits, create temporary dataset
        temp_dataset = MultiShapeDataset(str(self.dataset_path), split=split, load_images=False)
        return len(temp_dataset)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        info = {
            "dataset_path": str(self.dataset_path),
            "metadata": self.metadata,
            "current_split": self.split,
            "current_split_size": len(self),
            "load_images": self.load_images,
            "qa_format": self.qa_format,
            "splits": {}
        }
        
        # Get info for all splits
        for split in ["train", "val", "test"]:
            try:
                split_size = self.get_split_size(split)
                if split == self.split:
                    available_data = list(self.data.keys())
                else:
                    temp_dataset = MultiShapeDataset(str(self.dataset_path), split=split, load_images=False)
                    available_data = list(temp_dataset.data.keys())
                
                info["splits"][split] = {
                    "size": split_size,
                    "available_data": available_data
                }
            except Exception:
                info["splits"][split] = {"size": 0, "available_data": []}
        
        return info
    
    def iterate_samples(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Generator to iterate through samples in the current split."""
        if end_idx is None:
            end_idx = len(self)
        
        end_idx = min(end_idx, len(self))
        
        for idx in range(start_idx, end_idx):
            yield self[idx]
    
    def get_qa_data(self, split: Optional[str] = None) -> Tuple[List[str], List[str], List[int]]:
        """Extract all QA pairs from a split in a flat format."""
        if split is None:
            split = self.split
        
        if split != self.split:
            temp_dataset = MultiShapeDataset(str(self.dataset_path), split=split, load_images=False)
            return temp_dataset.get_qa_data()
        
        if "qa_pairs" not in self.data:
            return [], [], []
        
        questions = []
        answers = []
        canvas_indices = []
        
        for canvas_idx in range(len(self.data["qa_pairs"])):
            qa_pairs = self.parse_qa_pairs(self.data["qa_pairs"], canvas_idx)
            
            for qa_pair in qa_pairs:
                questions.append(qa_pair["question"])
                answers.append(qa_pair["answer"])
                canvas_indices.append(canvas_idx)
        
        return questions, answers, canvas_indices
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for PyTorch DataLoader.
        Each batch item is now a single QA pair with associated data.
        
        Args:
            batch: List of samples from __getitem__ (each is one QA pair)
            
        Returns:
            Batched data
        """
        collated = {}
        
        # Handle different data types
        for key in batch[0].keys():
            if key in ["qa_idx", "canvas_idx", "qa_pair_idx", "num_shapes", "num_qa_pairs"]:
                # Integer values
                collated[key] = torch.tensor([item[key] for item in batch])
            elif key in ["image", "shape_features", "answer_label", "latent", "unpaired"]:
                # Tensor values - stack them
                if all(key in item for item in batch):
                    collated[key] = torch.stack([item[key] for item in batch])
            elif key in ["question", "answer", "caption", "split", "image_path"]:
                # String values - keep as list
                collated[key] = [item[key] for item in batch]
            elif key in ["qa_pair", "shapes", "caption_choices", "all_qa_pairs"]:
                # Complex object values - keep as list
                collated[key] = [item[key] for item in batch]
            else:
                # Default: keep as list
                collated[key] = [item.get(key) for item in batch]
        
        return collated
    
    def create_dataloader(
        self, 
        batch_size: int = 32, 
        shuffle: bool = None,
        num_workers: int = 0,
        **kwargs
    ):
        """
        Create a PyTorch DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data (default: True for train, False for val/test)
            num_workers: Number of worker processes
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            torch.utils.data.DataLoader
        """
        from torch.utils.data import DataLoader
        
        if shuffle is None:
            shuffle = (self.split == "train")
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            **kwargs
        )
    
    def get_sample(self, split: str, canvas_idx: int) -> Dict[str, Any]:
        """
        Get all data for a specific sample in a clean format.
        
        Args:
            split (str): Split name
            canvas_idx (int): Canvas/sample index
            
        Returns:
            Dict[str, Any]: Complete sample data
        """
        data = self.load_split_data(split)
        
        sample = {
            "canvas_idx": canvas_idx,
            "split": split,
            "image_path": self.get_image_path(split, canvas_idx),
        }
        
        # Add shapes if available
        if "labels" in data:
            sample["shapes"] = self.parse_shapes(data["labels"], canvas_idx)
            sample["num_shapes"] = len(sample["shapes"])
        
        # Add caption if available
        if "captions" in data and canvas_idx < len(data["captions"]):
            sample["caption"] = data["captions"][canvas_idx]
        
        # Add caption choices if available
        if "caption_choices" in data and canvas_idx < len(data["caption_choices"]):
            sample["caption_choices"] = data["caption_choices"][canvas_idx]
        
        # Add QA pairs if available
        if "qa_pairs" in data:
            sample["qa_pairs"] = self.parse_qa_pairs(data["qa_pairs"], canvas_idx)
            sample["num_qa_pairs"] = len(sample["qa_pairs"])
        
        # Add other data types
        for key in ["latent", "unpaired"]:
            if key in data and canvas_idx < len(data[key]):
                sample[key] = data[key][canvas_idx]
        
        return sample
    
    def get_split_size(self, split: str) -> int:
        """
        Get the number of samples in a split.
        
        Args:
            split (str): Split name
            
        Returns:
            int: Number of samples in the split
        """
        # Try to determine size from various sources
        data = self.load_split_data(split)
        
        # Check captions first as they're typically 1:1 with samples
        if "captions" in data:
            return len(data["captions"])
        
        # Check QA pairs
        if "qa_pairs" in data:
            return len(data["qa_pairs"])
        
        # Check image directory
        image_dir = self.dataset_path / split
        if image_dir.exists():
            image_files = list(image_dir.glob("*.png"))
            return len(image_files)
        
        # Estimate from labels if available
        if "labels" in data:
            shapes_per_canvas = self.metadata.get("max_shapes_per_canvas", 
                                                self.metadata.get("shapes_per_canvas", 2))
            return len(data["labels"]) // shapes_per_canvas
        
        return 0
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Returns:
            Dict[str, Any]: Dataset information
        """
        info = {
            "dataset_path": str(self.dataset_path),
            "metadata": self.metadata,
            "current_split": self.split,
            "current_split_size": len(self),
            "load_images": self.load_images,
            "qa_format": self.qa_format,
            "splits": {}
        }
        
        # Get info for all splits
        for split in ["train", "val", "test"]:
            try:
                split_size = self.get_split_size(split)
                if split == self.split:
                    available_data = list(self.data.keys())
                else:
                    temp_dataset = MultiShapeDataset(str(self.dataset_path), split=split, load_images=False)
                    available_data = list(temp_dataset.data.keys())
                
                info["splits"][split] = {
                    "size": split_size,
                    "available_data": available_data
                }
            except Exception:
                info["splits"][split] = {"size": 0, "available_data": []}
        
        return info
    
    def iterate_samples(self, start_idx: int = 0, end_idx: Optional[int] = None):
        """Generator to iterate through samples in the current split."""
        if end_idx is None:
            end_idx = len(self)
        
        end_idx = min(end_idx, len(self))
        
        for idx in range(start_idx, end_idx):
            yield self[idx]
    
    def get_qa_data(self, split: Optional[str] = None) -> Tuple[List[str], List[str], List[int]]:
        """Extract all QA pairs from a split in a flat format."""
        if split is None:
            split = self.split
        
        if split != self.split:
            temp_dataset = MultiShapeDataset(str(self.dataset_path), split=split, load_images=False)
            return temp_dataset.get_qa_data()
        
        if "qa_pairs" not in self.data:
            return [], [], []
        
        questions = []
        answers = []
        canvas_indices = []
        
        for canvas_idx in range(len(self.data["qa_pairs"])):
            qa_pairs = self.parse_qa_pairs(self.data["qa_pairs"], canvas_idx)
            
            for qa_pair in qa_pairs:
                questions.append(qa_pair["question"])
                answers.append(qa_pair["answer"])
                canvas_indices.append(canvas_idx)
        
        return questions, answers, canvas_indices
    

def create_train_val_test_loaders(
    dataset_path: str,
    batch_size: int = 32,
    domains: Optional[List[str]] = None,
    transforms: Optional[Dict[str, Callable]] = None,
    num_workers: int = 0
) -> Tuple[Any, Any, Any]:
    """
    Convenience function to create train, validation, and test DataLoaders.
    
    Args:
        dataset_path: Path to dataset
        batch_size: Batch size for all loaders
        domains: List of domains to load (default: all available)
        transforms: Dictionary of transforms to apply to specific domains
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MultiShapeDataset(
        dataset_path, 
        split="train", 
        domains=domains,
        transforms=transforms
    )
    
    val_dataset = MultiShapeDataset(
        dataset_path, 
        split="val", 
        domains=domains,
        transforms=transforms
    )
    
    test_dataset = MultiShapeDataset(
        dataset_path, 
        split="test", 
        domains=domains,
        transforms=transforms
    )
    
    # Create DataLoaders
    train_loader = train_dataset.create_dataloader(
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = val_dataset.create_dataloader(
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = test_dataset.create_dataloader(
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Example usage of the MultiShapeDataset with domain selection."""
    dataset_path = "/users/xyu110/scratch/variable"  # Update this path as needed
    
    if not Path(dataset_path).exists():
        print(f"Dataset path {dataset_path} does not exist!")
        exit(0)
    
    print("=== MultiShapeDataset Domain Selection Examples ===\n")
    
    # Example 1: Load only QA pairs and shapes (no images, no captions)
    print("1. Loading only QA pairs and shapes:")
    qa_shapes_dataset = MultiShapeDataset(
        dataset_path,
        split="train",
        domains=['qa_pairs', 'shapes']
    )
    
    print(f"Loaded domains: {qa_shapes_dataset.domains}")
    print(f"Dataset size: {len(qa_shapes_dataset)} QA pairs")
    
    if len(qa_shapes_dataset) > 0:
        sample = qa_shapes_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Has shapes: {'shapes' in sample}")
        print(f"Has question: {'question' in sample}")
        print(f"Has image: {'image' in sample}")
        print(f"Has caption: {'caption' in sample}")
    
    # Example 2: Load images and QA pairs with transforms
    print(f"\n2. Loading images and QA pairs with transforms:")
    image_transforms = {
        'images': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    image_qa_dataset = MultiShapeDataset(
        dataset_path,
        split="train", 
        domains=['images', 'qa_pairs'],
        transforms=image_transforms
    )
    
    print(f"Loaded domains: {image_qa_dataset.domains}")
    if len(image_qa_dataset) > 0:
        sample = image_qa_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        if 'image' in sample:
            print(f"Image shape: {sample['image'].shape}")
            print(f"Image dtype: {sample['image'].dtype}")
    
    print(f"\n=== Domain Selection Examples Completed ===")


def main():
    """Legacy main function for backward compatibility."""
    dataset_path = "/users/xyu110/scratch/variable"
    
    if not Path(dataset_path).exists():
        print(f"Dataset path {dataset_path} does not exist!")
        return
    
    print("=== MultiShapeDataset Legacy Examples ===\n")
    
    # Example 1: Basic usage with domain selection
    print("1. Basic Data Loading:")
    dataset = MultiShapeDataset(dataset_path, split="train", domains=['shapes', 'captions', 'qa_pairs'])
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample 0: {sample.get('num_shapes', 0)} shapes, {sample.get('num_qa_pairs', 0)} QA pairs")
        if 'caption' in sample:
            print(f"Caption: {sample['caption'][:100]}...")
    
    print("Examples completed.")


if __name__ == "__main__":
    main()