from pathlib import Path

import simple_shapes_dataset.cli as cli
import simple_shapes_dataset.downstream as downstream
import simple_shapes_dataset.text as text

from .cli.multi_shapes import MultiShapeDataset, create_train_val_test_loaders
from .data_module import SimpleShapesDataModule
from .dataset import RepeatedDataset, SimpleShapesDataset, SizedDataset
from .domain import (
    DEFAULT_DOMAINS,
    Attribute,
    AttributesAdditionalArgs,
    Choice,
    DataDomain,
    DomainDesc,
    DomainType,
    PretrainedVisualAdditionalArgs,
    RawText,
    SimpleShapesAttributes,
    SimpleShapesImages,
    SimpleShapesPretrainedVisual,
    SimpleShapesRawText,
    SimpleShapesText,
    Text,
    get_default_domains,
)
from .domain_alignment import get_aligned_datasets, get_alignment
from .pre_process import (
    NormalizeAttributes,
    TextAndAttrs,
    UnnormalizeAttributes,
    attr_to_str,
    attribute_to_tensor,
    color_blind_visual_domain,
    nullify_attribute_rotation,
    tensor_to_attribute,
    text_to_bert,
    to_unit_range,
)
from .version import __version__

PROJECT_DIR = Path(__file__).resolve().parent.parent

__all__ = [
    "cli",
    "downstream",
    "text",
    "MultiShapeDataset",
    "create_train_val_test_loaders",
    "SimpleShapesDataModule",
    "RepeatedDataset",
    "SimpleShapesDataset",
    "SizedDataset",
    "DEFAULT_DOMAINS",
    "Attribute",
    "AttributesAdditionalArgs",
    "Choice",
    "DataDomain",
    "DomainDesc",
    "DomainType",
    "PretrainedVisualAdditionalArgs",
    "RawText",
    "SimpleShapesAttributes",
    "SimpleShapesImages",
    "SimpleShapesPretrainedVisual",
    "SimpleShapesRawText",
    "SimpleShapesText",
    "Text",
    "get_default_domains",
    "get_aligned_datasets",
    "get_alignment",
    "NormalizeAttributes",
    "TextAndAttrs",
    "UnnormalizeAttributes",
    "attr_to_str",
    "attribute_to_tensor",
    "color_blind_visual_domain",
    "nullify_attribute_rotation",
    "tensor_to_attribute",
    "text_to_bert",
    "to_unit_range",
    "__version__",
    "PROJECT_DIR",
]
