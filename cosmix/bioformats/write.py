from __future__ import annotations

# General libraries
import logging
from pathlib import Path
from xml.sax.saxutils import quoteattr

# Data libraries
import yaml
import numpy as np

# Image readers
import pyvips

# Project modules
from cosmix.bioformats.reader import (
    DEFAULT_CHANNEL_COLORS,
    get_slide_reader,
    mpp_to_magnification,
    split_slide_name,
)
from cosmix.bioformats.utils import (
    check_is_ome,
    vips_to_format,
)

LOGGER = logging.getLogger(__name__)

# Config files shipped with the package (resolved from this module, not the cwd)
CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_OME_CONFIG_PATH = CONFIG_DIR / "default_ome_config.xml"
PYVIPS_TO_OME_TYPE_PATH = CONFIG_DIR / "PYVIPS2OME_FORMAT.yaml"

# Pixel types accepted by `convert_ome_tiff` and their pyvips format
PIXEL_TYPE_TO_VIPS_FORMAT = {
    "uint8": "uchar",
    "uint16": "ushort",
}

# Channel names and colors given to 3-band (RGB) slides
RGB_CHANNEL_NAMES = ["red", "green", "blue"]
RGB_CHANNEL_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]


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
    channel_names: list[str] | None,
    channel_colors: list[int] | None,
    ome_config_path: str | Path | None = None,
    compression: str = "deflate",
    tile_size: int = 512,
    region_shrink: str = "mean",
) -> None:
    """Save a multi-band pyvips image as a pyramidal, planar OME-TIFF.

    Pyramid levels are built by libvips, halving the size at each level, with
    the `region_shrink` method (`"mean"`, `"median"`, `"mode"`, `"max"`,
    `"min"` or `"nearest"`).
    """
    # collect image dimension needed for OME-XML before separating image planes
    width = image_pyvips.width
    height = image_pyvips.height
    channel_count = image_pyvips.bands
    if channel_names is not None:
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
        region_shrink=region_shrink,
    )


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------


def read_vips_mpp(image_pyvips: pyvips.Image) -> float | None:
    """Read the level 0 microns per pixel of a slide opened with pyvips."""
    fields = image_pyvips.get_fields()
    if "openslide.mpp-x" in fields:
        return round(float(image_pyvips.get("openslide.mpp-x")), 4)
    # pyvips always stores the TIFF resolution in pixels per millimeter
    if "resolution-unit" in fields and image_pyvips.xres > 1:
        return round(1000 / image_pyvips.xres, 4)
    return None


def read_vips_magnification(image_pyvips: pyvips.Image) -> float | None:
    """Read the objective magnification of a slide opened with pyvips."""
    if "openslide.objective-power" in image_pyvips.get_fields():
        return float(image_pyvips.get("openslide.objective-power"))
    return None


def get_default_channels(channel_count: int) -> tuple[list[str], list[tuple]]:
    """Return RGB names and colors for 3 channels, numbered channels otherwise."""
    if channel_count == 3:
        return RGB_CHANNEL_NAMES, RGB_CHANNEL_COLORS
    channel_names = [
        f"channel_{channel_index}" for channel_index in range(channel_count)
    ]
    channel_colors = [
        DEFAULT_CHANNEL_COLORS[channel_index % len(DEFAULT_CHANNEL_COLORS)]
        for channel_index in range(channel_count)
    ]
    return channel_names, channel_colors


def convert_ome_tiff(
    image: str | Path | np.ndarray | pyvips.Image,
    dst: str | Path = ".",
    name: str | None = None,
    mpp: float | None = None,
    magnification: float | None = None,
    pixel_type: str | None = None,
    channel_names: list[str] | None = None,
    channel_colors: list[tuple[int, int, int]] | None = None,
    region_shrink: str = "mean",
    compression: str = "deflate",
    tile_size: int = 512,
) -> Path:
    """Convert a slide or an image array to a pyramidal, planar OME-TIFF.

    Slides are read lazily with pyvips, so whole slide images never have to fit
    in memory. Any format readable by libvips is supported, including the
    OpenSlide formats (`.svs`, `.ndpi`, `.mrxs`, ...) and OME-TIFF (all
    channels are read, with their names and colors).

    Parameters
    ----------
    image : str | Path | np.ndarray | pyvips.Image
        Path to the slide, or a `(height, width)` / `(height, width, channels)`
        array, or a pyvips image.

    dst : str | Path, optional
        Output directory, created if needed. Default is the current directory.

    name : str | None, optional
        Output file name, without extension. Default is the slide file name.
        Required when `image` is not a path.

    mpp : float | None, optional
        Level 0 microns per pixel. Default is read from the slide metadata.
        Required when it cannot be read (arrays, TIFF without resolution).

    magnification : float | None, optional
        Objective magnification. Default is read from the slide metadata, or
        inferred from `mpp`.

    pixel_type : str | None, optional
        Output pixel type, `"uint8"` or `"uint16"`. Values are rescaled to the
        full range of the new type (float images are assumed in [0, 1]).
        Default is None, which keeps the pixel type of the slide.

    channel_names : list[str] | None, optional
        One name per channel. Default is the OME-TIFF channel names, else
        `RGB_CHANNEL_NAMES` for 3 channels, else `channel_{index}`.

    channel_colors : list[tuple[int, int, int]] | None, optional
        One `(r, g, b)` color per channel. Default follows `channel_names`.

    region_shrink : str, optional
        Method used by libvips to build each pyramid level (halving the size),
        e.g. `"mean"` or `"nearest"`. Default is `"mean"`.

    compression : str, optional
        TIFF compression. Default is `"deflate"`.

    tile_size : int, optional
        TIFF tile width and height, in pixels. Default is 512.

    Returns
    -------
    path : Path
        Path to the written `<name>.ome.tif`.

    Raises
    ------
    ValueError
        If `pixel_type` is not supported, if `name` or `mpp` is missing and
        cannot be read from the slide, or if the channel lists do not match the
        number of channels.
    """
    if pixel_type is not None and pixel_type not in PIXEL_TYPE_TO_VIPS_FORMAT:
        raise ValueError(
            f"`pixel_type`={pixel_type!r} is not supported, expected one of "
            f"{list(PIXEL_TYPE_TO_VIPS_FORMAT)} or None"
        )

    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if name is None:
            name, _ = split_slide_name(image_path.name)
        if check_is_ome(image_path):
            # bandjoin the planar OME-TIFF pages and keep its channel metadata
            slide = get_slide_reader(
                image_path,
                mpp=mpp,
            )
            image_pyvips = slide.open_vips(level=0)
            mpp = slide.mpp
            if slide.channel_names is not None and channel_names is None:
                channel_names = slide.channel_names
                channel_colors = [tuple(color) for color in slide.channel_colors]
        else:
            # pyvips picks the loader (openslideload, tiffload...) from the file
            image_pyvips = pyvips.Image.new_from_file(str(image_path))
            if image_pyvips.hasalpha():
                # OpenSlide returns RGBA with a transparent background
                image_pyvips = image_pyvips.flatten(background=255)
            if mpp is None:
                mpp = read_vips_mpp(image_pyvips)
            if magnification is None:
                magnification = read_vips_magnification(image_pyvips)
    elif isinstance(image, np.ndarray):
        image_pyvips = pyvips.Image.new_from_array(image)
    else:
        image_pyvips = image

    if name is None:
        raise ValueError("`name` must be set when `image` is not a path")
    if mpp is None:
        raise ValueError(
            f"Cannot read the microns per pixel of {name!r}, `mpp` must be set"
        )
    if magnification is None:
        magnification = mpp_to_magnification(mpp)

    if pixel_type is not None:
        image_pyvips = vips_to_format(
            image_pyvips,
            PIXEL_TYPE_TO_VIPS_FORMAT[pixel_type],
        )

    if channel_names is None:
        channel_names, default_colors = get_default_channels(image_pyvips.bands)
        if channel_colors is None:
            channel_colors = default_colors
    if channel_colors is None:
        _, channel_colors = get_default_channels(len(channel_names))

    dst = Path(dst)
    dst.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = dst / f"{name}.ome.tif"
    save_ome_tiff(
        image_pyvips,
        path,
        pixel_size_x=mpp,
        pixel_size_y=mpp,
        magnification=magnification,
        channel_names=list(channel_names),
        channel_colors=[rgb_to_ome_color(*color) for color in channel_colors],
        compression=compression,
        tile_size=tile_size,
        region_shrink=region_shrink,
    )
    return path
