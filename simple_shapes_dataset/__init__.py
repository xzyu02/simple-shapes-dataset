from pathlib import Path

import simple_shapes_dataset.cli as cli
import simple_shapes_dataset.text as text
from simple_shapes_dataset.dataset.domain import (
    DEFAULT_DOMAINS,
    Attribute,
    RawText,
    SimpleShapesAttributes,
    SimpleShapesImages,
    SimpleShapesRawText,
    SimpleShapesText,
    Text,
)
from simple_shapes_dataset.dataset.domain_alignment import get_aligned_datasets

from .version import __version__

PROJECT_DIR = Path(__file__).resolve().parent.parent


__all__ = [
    "cli",
    "text",
    "__version__",
    "SimpleShapesImages",
    "SimpleShapesAttributes",
    "SimpleShapesText",
    "SimpleShapesRawText",
    "Attribute",
    "RawText",
    "Text",
    "DEFAULT_DOMAINS",
    "get_aligned_datasets",
    "PROJECT_DIR",
]
