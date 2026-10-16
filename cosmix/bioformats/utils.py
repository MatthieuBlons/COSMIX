from __future__ import annotations

# General libraries
import logging
from pathlib import Path
from xml.etree import ElementTree as ET

# Data libraries
import numpy as np

# Image readers
import pyvips
from PIL import Image
from skimage.transform import resize

LOGGER = logging.getLogger(__name__)

# File suffixes readable by libvips (the OME-TIFF suffix is `.tif` / `.tiff`)
VIPS_READABLE_FORMATS = set(pyvips.get_suffixes())

# Conversion factors from OME length units to micrometers
OME_UNIT_TO_MICRON = {
    "Å": 1e-4,
    "nm": 1e-3,
    "µm": 1.0,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}


# ----------------------------------------------------------------------------
# OME-XML
# ----------------------------------------------------------------------------


def parse_ome_xml(ome_xml: str) -> tuple[ET.Element, dict[str, str]]:
    root = ET.fromstring(ome_xml)
    # the namespace changes with the OME schema version (2016-06, 2015-01...)
    if not root.tag.startswith("{") or not root.tag.endswith("}OME"):
        raise ValueError(f"Root element is {root.tag!r}, expected an OME element")
    namespace = {"ome": root.tag[1 : -len("}OME")]}
    return root, namespace


def read_ome_xml(
    ome_xml: str,
    key: str,
    findall: bool = True,
) -> list[ET.Element] | ET.Element | None:
    root, namespace = parse_ome_xml(ome_xml)
    search = f".//ome:{key}"
    if findall:
        return root.findall(search, namespace)
    return root.find(search, namespace)


def read_ome_description(path: str | Path) -> str | None:
    img = pyvips.Image.new_from_file(
        str(path),
        access="sequential",
    )
    if "image-description" not in img.get_fields():
        return None
    description = img.get("image-description")
    try:
        parse_ome_xml(description)
    except (ET.ParseError, ValueError):
        return None
    return description


def check_is_ome(
    path: str | Path,
    validate: bool = False,
) -> bool:
    path = Path(path)
    is_ome = path.stem.lower().endswith(".ome") and path.suffix.lower() in (
        ".tif",
        ".tiff",
    )
    if is_ome and validate:
        is_ome = read_ome_description(path) is not None
    return is_ome


def check_to_use_vips(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIPS_READABLE_FORMATS


# TODO: check_flattened_pyramid_tiff (pyramid stored as pages, not subifds)


def print_vips_info(img: str | Path | pyvips.Image) -> None:
    """Log the main pyvips fields and OME-XML pixel sizes of an image.

    `img` can be a path, a `pyvips.Image` or an `OpenOME` instance.
    """
    if isinstance(img, (str, Path)):
        path = str(img)
        img = pyvips.Image.new_from_file(
            path,
            access="sequential",
        )
    elif isinstance(img, pyvips.Image):
        path = img.filename
    else:
        path = str(img.img_path)
        img = img.img
    fields = img.get_fields()
    LOGGER.info("interpretation: %s", img.interpretation)
    if "vips-loader" in fields:
        LOGGER.info("vips-loader: %s", img.get("vips-loader"))
    LOGGER.info("image format: %s", img.format)
    LOGGER.info("image coding: %s", img.coding)
    LOGGER.info("width: %s", img.width)
    LOGGER.info("height: %s", img.height)
    pixels = None
    if "image-description" in fields:
        try:
            pixels = read_ome_xml(
                img.get("image-description"),
                "Pixels",
                findall=False,
            )
        except (ET.ParseError, ValueError):
            pixels = None
    if pixels is not None:
        LOGGER.info("X physical size: %s", pixels.get("PhysicalSizeX"))
        LOGGER.info("Y physical size: %s", pixels.get("PhysicalSizeY"))
        LOGGER.info("physical size unit: %s", pixels.get("PhysicalSizeXUnit", "µm"))
    else:
        LOGGER.info("No OME-XML metadata found")
    if "xoffset" in fields:
        LOGGER.info("X position: %s", img.get("xoffset"))
        LOGGER.info("Y position: %s", img.get("yoffset"))
    LOGGER.info("pages: %s", img.get_n_pages())
    LOGGER.info("bands: %s", img.bands)
    if "n-subifds" in fields:
        LOGGER.info("pyramid levels: %s", img.get("n-subifds") + 1)
        LOGGER.info("dimensions level 0: %s", (img.width, img.height))
        for subifd in range(img.get("n-subifds")):
            img_level = pyvips.Image.new_from_file(
                path,
                subifd=subifd,
            )
            LOGGER.info(
                "dimensions level %s: %s",
                subifd + 1,
                (img_level.width, img_level.height),
            )


# ----------------------------------------------------------------------------
# Arrays and coordinates
# ----------------------------------------------------------------------------


def vips_to_numpy(img: pyvips.Image) -> np.ndarray:
    """Return the raw pixel values of a pyvips image (no rescaling)."""
    return img.numpy()


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Rescale an array to uint8 for display, using the full range of its dtype."""
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) * (255.0 / np.iinfo(arr.dtype).max)
    else:
        # float images are assumed to be in [0, 1]
        arr = arr.astype(np.float64) * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def arr_to_pil(img: np.ndarray) -> Image.Image:
    """Convert a 2D (grayscale), RGB or RGBA array to a PIL image (rescaled to uint8)."""
    img = to_uint8(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.ndim == 2 or img.shape[2] in (3, 4):
        return Image.fromarray(img)
    raise ValueError(
        f"Cannot convert an image with {img.shape[2]} channels to PIL, expected "
        "1, 3 or 4. Use `cosmix.bioformats.draw.blend_colors` for multiplex images."
    )


def get_x_y_to(
    point: tuple[float, float],
    dim_from: tuple[int, int],
    dim_to: tuple[int, int],
    integer: bool = True,
) -> tuple[int, int] | tuple[float, float]:
    """Rescale a point from one level's dimensions to another's.

    Parameters
    ----------
    point : tuple[float, float]
        `(x, y)` coordinates expressed in the `dim_from` reference frame, in
        pixels.

    dim_from : tuple[int, int]
        `(width, height)` of the source level, in pixels.

    dim_to : tuple[int, int]
        `(width, height)` of the target level, in pixels.

    integer : bool, optional
        If True, round the converted coordinates with `round`. Default is True.

    Returns
    -------
    point_l : tuple[int, int] | tuple[float, float]
        `(x, y)` coordinates in the `dim_to` reference frame.
    """
    x_0, y_0 = point
    x_l = x_0 * float(dim_to[0]) / float(dim_from[0])
    y_l = y_0 * float(dim_to[1]) / float(dim_from[1])
    if integer:
        return (round(x_l), round(y_l))
    return (x_l, y_l)


# ----------------------------------------------------------------------------
# Intensities and pyramids
# ----------------------------------------------------------------------------


def vips_normalize(
    image_pyvips: pyvips.Image,
    percentile: tuple[float, float] | None = (1, 99),
) -> tuple[pyvips.Image, float, float]:
    """Clip a single-band pyvips image to `percentile` (or min/max) and scale to [0, 1]."""
    if percentile:
        lower = float(image_pyvips.percent(percentile[0]))
        upper = float(image_pyvips.percent(percentile[1]))
    else:
        lower = float(image_pyvips.min())
        upper = float(image_pyvips.max())
    if upper <= lower:
        raise ValueError(
            f"Cannot normalize a constant image (lower={lower}, upper={upper})"
        )
    clipped_pyvips = (image_pyvips < lower).ifthenelse(lower, image_pyvips)
    clipped_pyvips = (clipped_pyvips > upper).ifthenelse(upper, clipped_pyvips)
    normalized_pyvips = (clipped_pyvips - lower) / (upper - lower)
    return normalized_pyvips, lower, upper


def vips_normalize_channels(
    image_pyvips: pyvips.Image,
    percentile: tuple[float, float] | None = (1, 99),
) -> tuple[pyvips.Image, list[float], list[float]]:
    """Apply `vips_normalize` to each band independently."""
    normalized_bands = []
    lowers = []
    uppers = []
    for band in image_pyvips.bandsplit():
        normalized_band, lower, upper = vips_normalize(
            band,
            percentile=percentile,
        )
        normalized_bands.append(normalized_band)
        lowers.append(lower)
        uppers.append(upper)
    normalized_pyvips = normalized_bands[0].bandjoin(normalized_bands[1:])
    return normalized_pyvips, lowers, uppers


def pyramid_nearest_neighbor(
    img: str | Path | pyvips.Image | np.ndarray,
    downscale: float,
    max_layer: int,
    preserve_range: bool = True,
) -> list[np.ndarray]:
    if isinstance(img, (str, Path)):
        img = pyvips.Image.new_from_file(
            str(img),
            access="sequential",
        )
    if isinstance(img, pyvips.Image):
        img = img.numpy()
    pyramid = [img]
    for _ in range(1, max_layer):
        height, width = pyramid[-1].shape[:2]
        next_level = resize(
            pyramid[-1],
            (np.ceil(height / downscale), np.ceil(width / downscale)),
            order=0,  # Nearest-neighbor interpolation
            preserve_range=preserve_range,
            anti_aliasing=False,
        )
        pyramid.append(next_level)
    return pyramid
