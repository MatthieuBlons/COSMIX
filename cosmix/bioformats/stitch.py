from __future__ import annotations

# General libraries
import logging
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

# Data libraries
import tifffile

# Plotting libraries
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Image readers
import pyvips

# Project modules
from cosmix.bioformats.utils import vips_normalize_channels
from cosmix.bioformats.write import (
    DEFAULT_OME_CONFIG_PATH,
    rgb_to_ome_color,
    save_ome_tiff,
)

LOGGER = logging.getLogger(__name__)

# TIFF ResolutionUnit value for centimeters, used by Akoya tiles
TIFF_RESOLUTION_UNIT_CM = 3
CM_TO_MICRON = 1e4


@dataclass
class TileRegion:
    file_path: Path
    x_position: float  # stage position, in pixels
    y_position: float  # stage position, in pixels
    width: int
    height: int
    x_resolution: float  # pixels per micron
    y_resolution: float  # pixels per micron
    x_coord: int = 0  # pixel position on the stitched image
    y_coord: int = 0  # pixel position on the stitched image


# ----------------------------------------------------------------------------
# Tile metadata
# ----------------------------------------------------------------------------


def _tag_ratio(tag: tifffile.TiffTag) -> float:
    numerator, denominator = tag.value
    return numerator / denominator


def read_tile_region(file_path: str | Path) -> TileRegion:
    file_path = Path(file_path)
    with tifffile.TiffFile(file_path) as tif:
        tags = tif.pages[0].tags
        resolution_unit = (
            int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else None
        )
        if resolution_unit != TIFF_RESOLUTION_UNIT_CM:
            raise ValueError(
                f"{file_path} has ResolutionUnit={resolution_unit}, expected "
                f"{TIFF_RESOLUTION_UNIT_CM} (centimeter)."
            )
        # positions are in cm and resolutions in pixels per cm
        x_resolution = _tag_ratio(tags["XResolution"]) / CM_TO_MICRON
        y_resolution = _tag_ratio(tags["YResolution"]) / CM_TO_MICRON
        x_position = _tag_ratio(tags["XPosition"]) * CM_TO_MICRON * x_resolution
        y_position = _tag_ratio(tags["YPosition"]) * CM_TO_MICRON * y_resolution
        height = tags["ImageLength"].value
        width = tags["ImageWidth"].value
    return TileRegion(
        file_path=file_path,
        x_position=x_position,
        y_position=y_position,
        width=width,
        height=height,
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        x_coord=round(x_position),
        y_coord=round(y_position),
    )


def zero_center_regions(tile_regions: list[TileRegion]) -> list[TileRegion]:
    """Set image coordinates from stage positions, top-left tile at (0, 0)."""
    min_x = min(region.x_position for region in tile_regions)
    min_y = min(region.y_position for region in tile_regions)
    for region in tile_regions:
        region.x_coord = round(region.x_position - min_x)
        region.y_coord = round(region.y_position - min_y)
    return tile_regions


def collect_tile_regions(file_paths: list[str | Path]) -> list[TileRegion]:
    if not file_paths:
        raise ValueError("`file_paths` is empty")
    return zero_center_regions([read_tile_region(path) for path in file_paths])


def akoya_color_to_ome(color: str) -> int:
    """Convert an Akoya `"r,g,b"` color string to an OME `Color` integer."""
    red, green, blue = (int(value) for value in color.split(",")[:3])
    return rgb_to_ome_color(red, green, blue)


def read_channel_metadata(file_path: str | Path) -> tuple[list[str], list[int]]:
    """Read channel names and OME colors from the Akoya per-page XML descriptions."""
    with tifffile.TiffFile(file_path) as tif:
        descriptions = [page.description for page in tif.series[0]]
    names = []
    colors = []
    for description in descriptions:
        root = ET.fromstring(description)
        names.append(root.find("Name").text)
        colors.append(akoya_color_to_ome(root.find("Color").text))
    return names, colors


def read_objective_magnification(file_path: str | Path) -> float:
    with tifffile.TiffFile(file_path) as tif:
        description = tif.series[0][0].description
    objective = ET.fromstring(description).find("Objective").text  # e.g. "20x"
    return float(objective.lower().split("x")[0])


def read_interleaved_tile(
    file_path: str | Path,
    channel_count: int,
) -> pyvips.Image:
    """Read the first `channel_count` pages of a tile as one multi-band image."""
    tile = pyvips.Image.new_from_file(
        str(file_path),
        n=channel_count,
    )
    page_height = tile.get("page-height")
    # chop into pages, then join them band-wise
    pages = [
        tile.crop(0, y, tile.width, page_height)
        for y in range(0, tile.height, page_height)
    ]
    return pages[0].bandjoin(pages[1:]).copy(interpretation="multiband")


# ----------------------------------------------------------------------------
# Stitching
# ----------------------------------------------------------------------------


def plot_tile_regions(
    tile_regions: list[TileRegion],
    show: bool = True,
) -> plt.Axes:
    """Plot tile positions and bounding boxes, for troubleshooting."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _, ax = plt.subplots()
    ax.invert_yaxis()
    ax.scatter(
        [region.x_coord for region in tile_regions],
        [region.y_coord for region in tile_regions],
    )
    for region_index, region in enumerate(tile_regions):
        ax.add_patch(
            patches.Rectangle(
                (region.x_coord, region.y_coord),
                region.width,
                region.height,
                linewidth=1,
                edgecolor=colors[region_index % len(colors)],
                facecolor="none",
            )
        )
    if show:
        plt.show()
    return ax


def stitch_tile_regions(
    tile_regions: list[TileRegion],
    dst: str | Path,
    normalize: bool = False,
    percentile: tuple[float, float] | None = (1, 99),
    ome_config_path: str | Path | None = DEFAULT_OME_CONFIG_PATH,
) -> None:
    reference_region = tile_regions[0]
    channel_names, channel_colors = read_channel_metadata(reference_region.file_path)
    magnification = read_objective_magnification(reference_region.file_path)
    # all tiles must share the same channels and magnification
    for region in tile_regions[1:]:
        names, colors = read_channel_metadata(region.file_path)
        if names != channel_names or colors != channel_colors:
            raise ValueError(
                f"Channels of {region.file_path} differ from those of "
                f"{reference_region.file_path}: {names} vs {channel_names}"
            )
        region_magnification = read_objective_magnification(region.file_path)
        if region_magnification != magnification:
            raise ValueError(
                f"Magnification of {region.file_path} ({region_magnification}) "
                f"differs from {reference_region.file_path} ({magnification})"
            )

    channel_count = len(channel_names)
    tiles = [
        read_interleaved_tile(region.file_path, channel_count)
        for region in tile_regions
    ]
    # allocate the full canvas once instead of expanding it at each insert
    canvas_width = max(
        region.x_coord + tile.width for region, tile in zip(tile_regions, tiles)
    )
    canvas_height = max(
        region.y_coord + tile.height for region, tile in zip(tile_regions, tiles)
    )
    stitched_image = (
        pyvips.Image.black(
            canvas_width,
            canvas_height,
            bands=channel_count,
        )
        .cast(tiles[0].format)
        .copy(interpretation="multiband")
    )
    for region, tile in zip(tile_regions, tiles):
        stitched_image = stitched_image.insert(
            tile,
            region.x_coord,
            region.y_coord,
        )
    LOGGER.info(
        "Stitched %s tiles into a %s x %s image with %s channels",
        len(tile_regions),
        canvas_width,
        canvas_height,
        channel_count,
    )

    if normalize:
        # each channel is clipped to `percentile` and scaled to the uint16 range
        stitched_image, _, _ = vips_normalize_channels(
            stitched_image,
            percentile=percentile,
        )
        stitched_image = (stitched_image * 65535).cast(pyvips.BandFormat.USHORT)

    save_ome_tiff(
        image_pyvips=stitched_image,
        dst=dst,
        pixel_size_x=1 / reference_region.x_resolution,
        pixel_size_y=1 / reference_region.y_resolution,
        magnification=magnification,
        channel_names=channel_names,
        channel_colors=channel_colors,
        ome_config_path=ome_config_path,
    )
