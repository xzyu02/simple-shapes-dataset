import click

from .alignments import alignment_group, create_domain_split
from .create_dataset import (
    create_dataset,
    create_ood_split,
    create_unpaired_attributes,
    unpaired_attributes_command,
)
from .odd_one_out import create_odd_one_out_dataset
from .ood_splits import (
    BinsBoundary,
    BoundaryBase,
    BoundaryInfo,
    BoundaryKind,
    ChoiceBoundary,
    MultiBinsBoundary,
    attr_boundaries,
    filter_dataset,
    ood_split,
)


@click.group()
def cli():
    pass


cli.add_command(create_dataset)
cli.add_command(alignment_group)
cli.add_command(unpaired_attributes_command)
cli.add_command(create_ood_split)
cli.add_command(create_odd_one_out_dataset)


__all__ = [
    "alignment_group",
    "create_domain_split",
    "create_dataset",
    "create_ood_split",
    "create_unpaired_attributes",
    "unpaired_attributes_command",
    "create_odd_one_out_dataset",
    "BinsBoundary",
    "BoundaryBase",
    "BoundaryInfo",
    "BoundaryKind",
    "ChoiceBoundary",
    "MultiBinsBoundary",
    "attr_boundaries",
    "filter_dataset",
    "ood_split",
    "cli",
]
