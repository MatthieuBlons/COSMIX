from __future__ import annotations

# General libraries
import logging
from pathlib import Path
from xml.sax.saxutils import quoteattr

# Data libraries
import yaml

# Image readers
import pyvips

LOGGER = logging.getLogger(__name__)

# Config files shipped with the package (resolved from this module, not the cwd)
CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_OME_CONFIG_PATH = CONFIG_DIR / "default_ome_config.xml"
PYVIPS_TO_OME_TYPE_PATH = CONFIG_DIR / "PYVIPS2OME_FORMAT.yaml"


def load_pyvips_to_ome_types(
    path: str | Path = PYVIPS_TO_OME_TYPE_PATH,
) -> dict[str, str]:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def rgb_to_ome_color(
    red: int,
    green: int,
    blue: int,
    alpha: int = 255,
) -> int:
    """Pack an RGBA color into the signed 32-bit integer used by OME `Color`."""
    return int.from_bytes(
        [red, green, blue, alpha],
        byteorder="big",
        signed=True,
    )


# ----------------------------------------------------------------------------
# OME-XML
# ----------------------------------------------------------------------------


def build_channels_xml(
    channel_names: list[str],
    channel_colors: list[int],
) -> str:
    return "\n".join(
        f'<Channel ID="Channel:0:{channel_index}" Name={quoteattr(str(name))} '
        f'Color="{color}" SamplesPerPixel="1"/>'
        for channel_index, (name, color) in enumerate(
            zip(channel_names, channel_colors)
        )
    )


def build_ome_xml(
    width: int,
    height: int,
    channel_count: int,
    pixel_size_x: float,
    pixel_size_y: float,
    magnification: float,
    channel_names: list[str],
    channel_colors: list[int],
    pixel_type: str = "float",
) -> str:
    """Build a minimal OME-XML for a 2D planar image (one page per channel)."""
    channels_xml = build_channels_xml(
        channel_names,
        channel_colors,
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
  <Instrument ID="Instrument:0">
    <Objective ID="Objective:0" NominalMagnification="{magnification}"/>
  </Instrument>
  <Image ID="Image:0">
    <InstrumentRef ID="Instrument:0"/>
    <ObjectiveSettings ID="Objective:0"/>
    <Pixels DimensionOrder="XYCZT"
            ID="Pixels:0"
            SizeC="{channel_count}"
            SizeT="1"
            SizeX="{width}"
            SizeY="{height}"
            SizeZ="1"
            Type="{pixel_type}"
            PhysicalSizeX="{pixel_size_x}"
            PhysicalSizeY="{pixel_size_y}"
            PhysicalSizeXUnit="µm"
            PhysicalSizeYUnit="µm">
{channels_xml}
      <TiffData IFD="0" PlaneCount="{channel_count}"/>
    </Pixels>
  </Image>
</OME>"""


def fill_ome_config(
    ome_config_path: str | Path,
    width: int,
    height: int,
    channel_count: int,
    pixel_size_x: float,
    pixel_size_y: float,
    magnification: float,
    channel_names: list[str],
    channel_colors: list[int],
    pixel_type: str,
) -> str:
    """Fill an OME-XML template (see `DEFAULT_OME_CONFIG_PATH`) with `ome_types`."""
    # optional dependency, only needed to build OME-XML from a template
    import ome_types

    ome_config = ome_types.from_xml(ome_config_path)
    pixels = ome_config.images[0].pixels
    pixels.size_c = channel_count
    pixels.type = pixel_type
    pixels.size_x = width
    pixels.size_y = height
    pixels.physical_size_x = pixel_size_x
    pixels.physical_size_y = pixel_size_y
    pixels.planes = [
        ome_types.model.Plane(
            the_z=0,
            the_t=0,
            the_c=channel_index,
        )
        for channel_index in range(channel_count)
    ]
    pixels.channels = [
        ome_types.model.Channel(
            id=f"Channel:{channel_index}",
            name=channel_names[channel_index],
            color=channel_colors[channel_index],
            samples_per_pixel=1,
            light_path={},
        )
        for channel_index in range(channel_count)
    ]
    ome_config.instruments[0].objectives[0].nominal_magnification = magnification
    return ome_config.to_xml()


# ----------------------------------------------------------------------------
# OME-TIFF
# ----------------------------------------------------------------------------


def save_ome_tiff(
    image_pyvips: pyvips.Image,
    dst: str | Path,
    pixel_size_x: float,
    pixel_size_y: float,
    magnification: float,
    channel_names: list[str],
    channel_colors: list[int],
    ome_config_path: str | Path | None = None,
    compression: str = "deflate",
    tile_size: int = 512,
) -> None:
    """Save a multi-band pyvips image as a pyramidal, planar OME-TIFF."""
    # collect image dimension needed for OME-XML before separating image planes
    width = image_pyvips.width
    height = image_pyvips.height
    channel_count = image_pyvips.bands
    if len(channel_names) != channel_count or len(channel_colors) != channel_count:
        raise ValueError(
            f"`channel_names` ({len(channel_names)}) and `channel_colors` "
            f"({len(channel_colors)}) must have one entry per band "
            f"({channel_count})."
        )
    pyvips_to_ome_types = load_pyvips_to_ome_types()
    if image_pyvips.format not in pyvips_to_ome_types:
        raise ValueError(
            f"Unsupported pyvips format {image_pyvips.format!r}, expected one of "
            f"{list(pyvips_to_ome_types)}"
        )
    pixel_type = pyvips_to_ome_types[image_pyvips.format]

    metadata = {
        "width": width,
        "height": height,
        "channel_count": channel_count,
        "pixel_size_x": pixel_size_x,
        "pixel_size_y": pixel_size_y,
        "magnification": magnification,
        "channel_names": channel_names,
        "channel_colors": channel_colors,
        "pixel_type": pixel_type,
    }
    if ome_config_path is not None:
        ome_xml = fill_ome_config(ome_config_path, **metadata)
    else:
        ome_xml = build_ome_xml(**metadata)

    # split to separate image planes and stack vertically for OME-TIFF
    image_pyvips = pyvips.Image.arrayjoin(
        image_pyvips.bandsplit(),
        across=1,
    ).copy()
    # Set tiff tags necessary for OME-TIFF
    image_pyvips.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml)
    image_pyvips.set_type(pyvips.GValue.gint_type, "page-height", height)

    LOGGER.info("Saving OME-TIFF to %s", dst)
    image_pyvips.write_to_file(
        str(dst),
        compression=compression,
        predictor="none",
        tile=True,
        tile_width=tile_size,
        tile_height=tile_size,
        pyramid=True,
        subifd=True,
        bigtiff=True,
    )
