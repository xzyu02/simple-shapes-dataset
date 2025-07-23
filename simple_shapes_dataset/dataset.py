import warnings
from collections.abc import Callable, Mapping, Sized
from pathlib import Path
from typing import Any, Protocol

from torch.utils.data import Dataset

from simple_shapes_dataset.domain import DataDomain, DomainDesc


class SizedDataset(Sized, Protocol):
    def __getitem__(self, index): ...


class RepeatedDataset(Dataset):
    """
    Dataset that cycles through its items to have a size of at least min size.
    If drop_last is True, the size will be exaclty min_size. If drop_last is False,
    the min_size ≤ size < min_size + len(dataset).
    """

    def __init__(self, dataset: SizedDataset, min_size: int, drop_last: bool = False):
        """
        Args:
            dataset (SizedDataset): dataset to repeat. The dataset should have a size
                (where `__len__` is defined).
            min_size (int): minimum size of the final dataset
            drop_last (bool): whether to remove overflow when repeating the
                dataset.
        """
        self.dataset = dataset
        assert min_size >= len(self.dataset)
        self.dataset_size = len(self.dataset)
        if drop_last:
            self.total_size = min_size
        else:
            self.total_size = (
                min_size // self.dataset_size + int(min_size % self.dataset_size > 0)
            ) * self.dataset_size

    def __len__(self) -> int:
        """
        Size of the dataset. Will be min_size if drop_last is True.
        Otherwise, min_size ≤ size < min_size + len(dataset).
        """
        return self.total_size

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % self.dataset_size]


class SimpleShapesDataset(Dataset):
    """
    Dataset class to obtain a SimpleShapesDataset.
    """

    # Class number to shape name mapping
    CLASS_TO_NAME = {
        0: "diamond",
        1: "egg",
        2: "triangle",
        3: "circle",
        4: "square",
        5: "star",
        6: "heart",
    }

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        domain_classes: Mapping[DomainDesc, type[DataDomain]],
        max_size: int | None = None,
        transforms: Mapping[str, Callable[[Any], Any]] | None = None,
        domain_args: Mapping[str, Any] | None = None,
    ):
        """
        Params:
            dataset_path (str | pathlib.Path): Path to the dataset.
            split (str): Split to use. One of 'train', 'val', 'test'.
            domain_classes (Mapping[str, type[SimpleShapesDomain]]): Classes of
                domain loaders to include in the dataset.
            max_size (int | None): Max size of the dataset.
            transforms (Mapping[str, (Any) -> Any]): Optional transforms to apply
                to the domains. The keys are the domain names,
                the values are the transforms.
            domain_args (Mapping[str, Any]): Optional additional arguments to pass
                to the domains.
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.max_size = max_size

        self.domains: dict[str, DataDomain] = {}
        self.domain_args = domain_args or {}

        for domain, domain_cls in domain_classes.items():
            transform = None
            if transforms is not None and domain.kind in transforms:
                transform = transforms[domain.kind]

            self.domains[domain.kind] = domain_cls(
                dataset_path,
                split,
                transform,
                self.domain_args.get(domain.kind, None),
            )

        lengths = {len(domain) for domain in self.domains.values()}
        min_length = min(lengths)
        if len(lengths) != 1:
            warnings.warn(
                f"Domains have different lengths. Selecting min ({min_length}).",
                UserWarning,
                stacklevel=2,
            )
        self.dataset_size = min_length
        if self.max_size is not None:
            assert (
                self.max_size <= self.dataset_size
            ), "Max sizes can only be lower than actual size."
            self.dataset_size = self.max_size

    def __len__(self) -> int:
        """
        All domains should be the same length.
        """
        return self.dataset_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Params:
            index (int): Index of the item to get.
        Returns:
            dict[str, Any]: Dictionary containing the domains. The keys are the
            domain names, the values are the domains as given by the domain model at
            the given index.
        """
        return {
            domain_name: domain[index] for domain_name, domain in self.domains.items()
        }

    def get_class_name(self, class_id: int) -> str:
        """Get the shape name for a given class ID.

        Args:
            class_id (int): The class ID (0-6)

        Returns:
            str: The name of the shape corresponding to the class ID
        """
        return self.CLASS_TO_NAME.get(class_id, "unknown")

    @classmethod
    def get_all_class_names(cls) -> dict[int, str]:
        """Get all class ID to name mappings.

        Returns:
            dict[int, str]: Dictionary mapping class IDs to shape names
        """
        return cls.CLASS_TO_NAME.copy()
