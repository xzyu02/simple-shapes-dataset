"""Utility functions for multi-shapes dataset generation."""

import click
import numpy as np
from simple_shapes_dataset.text.writers import shapes_writer


def parse_allowed_shapes(shapes_str: str) -> list[int] | None:
    """
    Parse shape restrictions from command line argument.
    
    Args:
        shapes_str: Comma-separated list of shape ids or names (e.g., '3,4' or 'circle,square')
        
    Returns:
        List of shape ids, or None if no restrictions
        
    Raises:
        click.ClickException: If shape name is not recognized
        
    Example:
        >>> parse_allowed_shapes("circle,square")
        [3, 4]
        >>> parse_allowed_shapes("3,4,5")
        [3, 4, 5]
    """
    if not shapes_str:
        return None
    
    # Build mapping from name->id using shapes_writer choices
    name_to_id: dict[str, int] = {}
    for sid, labels in shapes_writer.choices.items():
        for lab in labels:
            name_to_id[lab.lower()] = int(sid)
    
    raw_items = [s.strip() for s in shapes_str.split(",") if s.strip()]
    allowed_shape_ids = []
    
    for it in raw_items:
        if it.isdigit():
            allowed_shape_ids.append(int(it))
        else:
            lid = name_to_id.get(it.lower())
            if lid is None:
                raise click.ClickException(
                    f"Unknown shape name '{it}'. Use ids 0-6 or known names like 'circle','square'."
                )
            allowed_shape_ids.append(lid)
    
    return allowed_shape_ids if len(allowed_shape_ids) > 0 else None


def parse_allowed_colors(colors_str: str) -> list[tuple[int, int, int]] | None:
    """
    Parse color restrictions from command line argument.
    
    Args:
        colors_str: Comma-separated color names from COLORS_LARGE_SET (e.g., 'red,green')
        
    Returns:
        List of RGB color tuples, or None if no restrictions
        
    Raises:
        click.ClickException: If color name is not recognized or dependency is missing
        
    Example:
        >>> parse_allowed_colors("red,green")
        [(255, 0, 0), (0, 128, 0)]
    """
    if not colors_str:
        return None
    
    # Lazy import to avoid editor import resolution issues
    try:
        from attributes_to_language.utils import COLORS_LARGE_SET  # type: ignore
    except Exception as e:
        raise click.ClickException(
            f"Failed to import color set from attributes-to-language. "
            f"Is the dependency installed? Original error: {e}"
        )
    
    labels = [c.strip().lower() for c in colors_str.split(",") if c.strip()]
    label_list = [lab.lower() for lab in COLORS_LARGE_SET["labels"]]
    rgb_list = COLORS_LARGE_SET["rgb"]
    palette: list[tuple[int, int, int]] = []
    
    for lab in labels:
        try:
            idx = label_list.index(lab)
        except ValueError:
            raise click.ClickException(
                f"Unknown color '{lab}'. Choose from COLORS_LARGE_SET labels."
            )
        
        # COLORS_LARGE_SET["rgb"] can be shaped (N,3) or (3,N); handle both
        try:
            rgb_arr = np.array(rgb_list)
            if rgb_arr.ndim == 2 and rgb_arr.shape[0] == 3:
                r, g, b = rgb_arr[0, idx], rgb_arr[1, idx], rgb_arr[2, idx]
                rgb = (int(r), int(g), int(b))
            else:
                r, g, b = rgb_arr[idx]
                rgb = (int(r), int(g), int(b))
        except Exception:
            # Fallback to generic indexing assumption (N,3)
            rgb = tuple(int(x) for x in rgb_list[idx])
        
        palette.append(rgb)  # type: ignore
    
    return palette if len(palette) > 0 else None
