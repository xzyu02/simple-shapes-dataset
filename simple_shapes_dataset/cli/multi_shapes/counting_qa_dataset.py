"""
Simple Counting QA Dataset Loader

A minimal, easy-to-read dataset for loading counting QA data.
Only focuses on: questions, images, and answers (labels).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class CountingQADataset(Dataset):
    """
    Simple dataset for counting QA.
    
    Each sample contains:
    - question: str (e.g., "How many red circles are on the image?")
    - image: tensor (the image)
    - answer: str (numeric answer like "5")
    - label: int (the numeric answer as an integer for training)
    
    Args:
        dataset_path (str): Path to the dataset directory
        split (str): 'train', 'val', or 'test'
        image_size (int): Size to resize images to (default: 224)
        
    Example:
        dataset = CountingQADataset("/path/to/dataset", split="train")
        dataloader = dataset.create_dataloader(batch_size=32)
        
        for batch in dataloader:
            images = batch['image']      # [batch_size, 3, 224, 224]
            questions = batch['question'] # list of strings
            answers = batch['answer']     # list of strings
            labels = batch['label']       # [batch_size] tensor of ints
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        image_size: int = 224,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        
        # Load QA pairs
        qa_path = self.dataset_path / f"{split}_qa_pairs.npy"
        if not qa_path.exists():
            raise FileNotFoundError(f"QA pairs file not found: {qa_path}")
        
        self.qa_pairs = np.load(qa_path, allow_pickle=True)
        
        # Image transform: resize and normalize
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Flatten QA pairs: each sample is one question
        self.samples = []
        for canvas_idx in range(len(self.qa_pairs)):
            canvas_qa = self.qa_pairs[canvas_idx]
            for qa_pair in canvas_qa:
                question = str(qa_pair[0])
                answer = str(qa_pair[1])
                self.samples.append({
                    'canvas_idx': canvas_idx,
                    'question': question,
                    'answer': answer,
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load image
        canvas_idx = sample['canvas_idx']
        image_path = self.dataset_path / self.split / f"{canvas_idx}.png"
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Parse numeric answer as label
        answer_str = sample['answer']
        label = int(answer_str)
        
        return {
            'question': sample['question'],
            'image': image,
            'answer': answer_str,
            'label': label,
            'canvas_idx': canvas_idx,
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        return {
            'question': [item['question'] for item in batch],
            'image': torch.stack([item['image'] for item in batch]),
            'answer': [item['answer'] for item in batch],
            'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
            'canvas_idx': torch.tensor([item['canvas_idx'] for item in batch], dtype=torch.long),
        }
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle (default: True for train, False otherwise)
            num_workers: Number of worker processes
            
        Returns:
            DataLoader
        """
        if shuffle is None:
            shuffle = (self.split == "train")
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )


def create_counting_qa_loaders(
    dataset_path: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders for counting QA.
    
    Args:
        dataset_path: Path to dataset directory
        batch_size: Batch size for all loaders
        image_size: Size to resize images to
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        
    Example:
        train_loader, val_loader, test_loader = create_counting_qa_loaders(
            "/path/to/dataset", 
            batch_size=32
        )
    """
    train_dataset = CountingQADataset(dataset_path, split="train", image_size=image_size)
    val_dataset = CountingQADataset(dataset_path, split="val", image_size=image_size)
    test_dataset = CountingQADataset(dataset_path, split="test", image_size=image_size)
    
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


# if __name__ == "__main__":
#     """Example usage."""
#     dataset_path = ""
    
#     if not Path(dataset_path).exists():
#         print(f"Dataset path {dataset_path} does not exist!")
#         exit(0)
    
#     print("=== Counting QA Dataset Example ===\n")
    
#     # Create dataset
#     dataset = CountingQADataset(dataset_path, split="train")
#     print(f"Dataset size: {len(dataset)} questions")
    
#     # Check a sample
#     sample = dataset[0]
#     print(f"\nSample 0:")
#     print(f"  Question: {sample['question']}")
#     print(f"  Answer: {sample['answer']}")
#     print(f"  Label: {sample['label']}")
#     print(f"  Image shape: {sample['image'].shape}")
#     print(f"  Canvas idx: {sample['canvas_idx']}")
    
#     # Create dataloader
#     dataloader = dataset.create_dataloader(batch_size=4)
    
#     # Test a batch
#     batch = next(iter(dataloader))
#     print(f"\nBatch:")
#     print(f"  Questions: {len(batch['question'])} items")
#     print(f"  Images shape: {batch['image'].shape}")
#     print(f"  Answers: {batch['answer']}")
#     print(f"  Labels: {batch['label']}")
#     print(f"  Canvas indices: {batch['canvas_idx']}")
    
#     print("\n=== Example Complete ===")
