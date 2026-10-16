from __future__ import annotations

# General libraries
import logging
from pathlib import Path

# Data libraries
import numpy as np

# Image readers
import pyvips
from PIL import Image

# Project modules
from cosmix.bioformats.utils import (
    OME_UNIT_TO_MICRON,
    arr_to_pil,
    check_is_ome,
    get_x_y_to,
    parse_ome_xml,
    vips_to_numpy,
)

LOGGER = logging.getLogger(__name__)

# (upper mpp bound, magnification): first bound strictly above the mpp wins
MPP_TO_MAGNIFICATION = [
    (0.16, 80),
    (0.2, 60),
    (0.3, 40),
    (0.6, 20),
    (1.2, 10),
    (2.4, 5),
]

# RGB colors given, in order, to channels without an OME `Color`
DEFAULT_CHANNEL_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 255, 255),
]


def mpp_to_magnification(mpp: float) -> int | None:
    for mpp_upper_bound, magnification in MPP_TO_MAGNIFICATION:
        if mpp < mpp_upper_bound:
            return magnification
    LOGGER.warning("mpp=%s is too high to infer a magnification, set to None", mpp)
    return None


def ome_color_to_rgb(color: int) -> tuple[int, int, int]:
    """Convert a signed 32-bit OME RGBA integer to an `(r, g, b)` tuple."""
    red = (color >> 24) & 0xFF
    green = (color >> 16) & 0xFF
    blue = (color >> 8) & 0xFF
    return (red, green, blue)


def split_slide_name(file_name: str) -> tuple[str, str]:
    """Split `"slide.ome.tif"` into `("slide", ".ome.tif")`."""
    path = Path(file_name)
    name, ext = path.stem, path.suffix
    if name.lower().endswith(".ome"):
        name, ext = name[: -len(".ome")], name[-len(".ome") :] + ext
    return name, ext


# ----------------------------------------------------------------------------
# OpenOME
# ----------------------------------------------------------------------------


# OpenOME class should be consistent with OpenSlide
class OpenOME:
    """
    The `OpenOME` class provides an interface to work with OME-TIFF images using pyvips.
    It mirrors the basic operation of OpenSlide class objects provided by the OpenSlide project.

    Attributes:
    -----------
    img_path : Path
        Path to the OME-TIFF file.

    """

    def __init__(
        self,
        img_path: str | Path,
        name: str | None = None,
        mpp: float | None = None,
    ):
        self.img_path = Path(img_path)
        self.name, self.ext = split_slide_name(
            self.img_path.name if name is None else name
        )
        self.img = pyvips.Image.new_from_file(str(self.img_path))
        fields = self.img.get_fields()
        if "image-description" not in fields:
            raise ValueError(
                f"{self.img_path} has no image description, `OpenOME` only "
                "supports OME-TIFF files."
            )
        self.ome_xml = self.img.get("image-description")
        self.loader = self.img.get("vips-loader") if "vips-loader" in fields else None
        self._fetch_meta(mpp)
        self.magnification = mpp_to_magnification(self.mpp)
        self.level_count = (
            self.img.get("n-subifds") + 1 if "n-subifds" in fields else 1
        )
        self.level_dimensions = self._fetch_level_dimensions()
        self.level_downsamples = self._fetch_level_downsamples()
        # level -> band-joined pyvips image, filled lazily by `open_vips`
        self._level_cache = {}

    def __repr__(self) -> str:
        return (
            f"OpenOME(name={self.name!r}, dimensions={self.dimensions}, "
            f"channels={self.channel_count}, levels={self.level_count}, "
            f"mpp={self.mpp})"
        )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    def _fetch_meta(self, mpp: float | None) -> None:
        pixels_metadata = self.get_pixels_metadata()
        # width, height, bands, format and n-pages are always defined by pyvips
        self.format = self.img.format
        self.width = self.img.width
        self.height = self.img.height
        self.dimensions = (self.width, self.height)
        self.bands = self.img.bands
        self.pages = self.img.get_n_pages()
        self.depth = pixels_metadata["size_z"]
        self.timepoints = pixels_metadata["size_t"]
        self.channels = pixels_metadata["size_c"]
        self.physical_unit = pixels_metadata["physical_size_unit"]
        self.mpp = self._fetch_mpp(mpp, pixels_metadata)

        if self.pages > 1:
            # one page per channel (planar OME-TIFF)
            if self.depth > 1 or self.timepoints > 1:
                raise ValueError(
                    f"{self.img_path} has SizeZ={self.depth} and "
                    f"SizeT={self.timepoints}, only 2D images (SizeZ=SizeT=1) are "
                    "supported."
                )
            if self.bands != 1:
                raise ValueError(
                    f"{self.img_path} has {self.pages} pages of {self.bands} bands, "
                    "expected single-band pages (one per channel)."
                )
            if self.pages < self.channels:
                raise ValueError(
                    f"{self.img_path} has {self.pages} pages but SizeC="
                    f"{self.channels} channels."
                )
            if self.pages > self.channels:
                LOGGER.warning(
                    "%s has %s pages for %s channels, extra pages are ignored",
                    self.img_path,
                    self.pages,
                    self.channels,
                )
            self.channel_pages = self.channels
            self.channel_count = self.channels
        else:
            # channels are interleaved in the bands of a single page
            self.channel_pages = 1
            self.channel_count = self.bands

        self.is_rgb = self.channel_pages == 1 and self.bands == 3
        self.is_multiplex = self.channel_pages > 1 or self.bands > 4
        self.channel_names = None
        self.channel_colors = None
        if self.is_multiplex:
            channel_info = self.get_channel_info()
            self.channel_names = channel_info["names"]
            self.channel_colors = np.array(channel_info["colors"])

    def _fetch_mpp(
        self,
        mpp: float | None,
        pixels_metadata: dict,
    ) -> float:
        # a user-provided mpp takes precedence over the OME-XML physical size
        if mpp is not None:
            return mpp
        physical_size_x = pixels_metadata["physical_size_x"]
        if physical_size_x is None:
            raise ValueError(
                f"No `PhysicalSizeX` in OME-XML of {self.img_path}, "
                "`mpp` must be set manually."
            )
        if self.physical_unit not in OME_UNIT_TO_MICRON:
            raise ValueError(
                f"Unsupported `PhysicalSizeXUnit`={self.physical_unit!r}, "
                f"expected one of {list(OME_UNIT_TO_MICRON)}. "
                "`mpp` must be set manually."
            )
        return round(physical_size_x * OME_UNIT_TO_MICRON[self.physical_unit], 4)

    def _fetch_level_dimensions(self) -> list[tuple[int, int]]:
        # level 0 is the main IFD, level l > 0 is subifd l - 1
        level_dimensions = [(self.width, self.height)]
        for level in range(1, self.level_count):
            img_level = pyvips.Image.tiffload(
                str(self.img_path),
                subifd=level - 1,
            )
            level_dimensions.append((img_level.width, img_level.height))
        return level_dimensions

    def _fetch_level_downsamples(self) -> list[float]:
        width_0, height_0 = self.level_dimensions[0]
        return [
            max(width_0 / width, height_0 / height)
            for width, height in self.level_dimensions
        ]

    def get_pixels_metadata(self) -> dict:
        root, namespace = parse_ome_xml(self.ome_xml)
        # the first Image is the main one, others may be label / macro images
        pixels = root.find("ome:Image/ome:Pixels", namespace)
        if pixels is None:
            raise ValueError(
                f"No OME `Pixels` element found in the image description of "
                f"{self.img_path}, is it a valid OME-TIFF?"
            )
        physical_size_x = pixels.get("PhysicalSizeX")
        physical_size_y = pixels.get("PhysicalSizeY")
        return {
            "size_x": int(pixels.get("SizeX")),
            "size_y": int(pixels.get("SizeY")),
            "size_z": int(pixels.get("SizeZ")),
            "size_c": int(pixels.get("SizeC")),
            "size_t": int(pixels.get("SizeT")),
            "type": pixels.get("Type"),
            "dimension_order": pixels.get("DimensionOrder"),
            "physical_size_x": (
                float(physical_size_x) if physical_size_x is not None else None
            ),
            "physical_size_y": (
                float(physical_size_y) if physical_size_y is not None else None
            ),
            # µm is the OME default unit when the attribute is absent
            "physical_size_unit": pixels.get("PhysicalSizeXUnit", "µm"),
        }

    def get_channel_info(self) -> dict[str, list]:
        root, namespace = parse_ome_xml(self.ome_xml)
        channels = root.findall("ome:Image/ome:Pixels/ome:Channel", namespace)
        if len(channels) != self.channel_count:
            LOGGER.warning(
                "%s: %s OME `Channel` elements for %s channels, using default "
                "names and colors",
                self.img_path,
                len(channels),
                self.channel_count,
            )
            channels = [None] * self.channel_count
        names = []
        colors = []
        for channel_index, channel in enumerate(channels):
            name = channel.get("Name") if channel is not None else None
            color = channel.get("Color") if channel is not None else None
            names.append(name if name is not None else f"channel_{channel_index}")
            if color is not None:
                colors.append(ome_color_to_rgb(int(color)))
            else:
                colors.append(
                    DEFAULT_CHANNEL_COLORS[channel_index % len(DEFAULT_CHANNEL_COLORS)]
                )
        return {"names": names, "colors": colors}

    def get_channel_index(self, key: str) -> int:
        """Return the index of the channel named exactly `key`."""
        if self.channel_names is None or key not in self.channel_names:
            raise KeyError(
                f"`key`={key!r} is not a channel name, expected one of "
                f"{self.channel_names}"
            )
        return self.channel_names.index(key)

    # ------------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------------

    def _check_level(self, level: int) -> int:
        if level < 0:
            raise ValueError(f"`level` must be >= 0, got {level}")
        if level >= self.level_count:
            LOGGER.warning(
                "level %s does not exist, set to lowest resolution level %s",
                level,
                self.level_count - 1,
            )
            level = self.level_count - 1
        return level

    def open_vips(self, level: int) -> pyvips.Image:
        level = self._check_level(level)
        if level not in self._level_cache:
            # the full-size image is in the main IFD, accessed with subifd=-1
            subifd = level - 1
            if self.channel_pages > 1:
                pages = [
                    pyvips.Image.tiffload(
                        str(self.img_path),
                        page=page,
                        subifd=subifd,
                    )
                    for page in range(self.channel_pages)
                ]
                img = pages[0].bandjoin(pages[1:])
            else:
                img = pyvips.Image.tiffload(
                    str(self.img_path),
                    subifd=subifd,
                )
            self._level_cache[level] = img
        return self._level_cache[level]

    def read_whole(
        self,
        level: int,
        numpy: bool = True,
    ) -> np.ndarray | Image.Image:
        whole = vips_to_numpy(self.open_vips(level))
        return whole if numpy else arr_to_pil(whole)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        numpy: bool = True,
    ) -> np.ndarray | Image.Image:
        """Read a `size` region at `level`, `location` being level 0 `(x, y)`."""
        level = self._check_level(level)
        width, height = size
        x, y = get_x_y_to(
            location,
            self.dimensions,
            self.level_dimensions[level],
            integer=True,
        )
        whole = self.open_vips(level)
        if not (0 <= x < whole.width and 0 <= y < whole.height):
            raise ValueError(
                f"`location`={location} is outside the image at level {level} "
                f"({whole.width} x {whole.height})"
            )
        if x + width > whole.width or y + height > whole.height:
            LOGGER.warning(
                "region is out of image bounds %s x %s, size was cropped",
                whole.width,
                whole.height,
            )
            width = min(width, whole.width - x)
            height = min(height, whole.height - y)
        crop = vips_to_numpy(whole.crop(x, y, width, height))
        return crop if numpy else arr_to_pil(crop)

    def _read_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        # fit the thumbnail inside `size` while keeping the aspect ratio
        scale = min(size[0] / self.width, size[1] / self.height)
        thumbnail_dimensions = (
            max(1, round(self.width * scale)),
            max(1, round(self.height * scale)),
        )
        downsample_factor = max(
            dim / thumb for dim, thumb in zip(self.dimensions, thumbnail_dimensions)
        )
        best_level, _, _ = self.get_best_level_for_downsample(downsample_factor)
        whole = self.open_vips(best_level)
        scaling_final = max(
            dim / thumb
            for dim, thumb in zip(
                self.level_dimensions[best_level],
                thumbnail_dimensions,
            )
        )
        return vips_to_numpy(whole.resize(1 / scaling_final))

    def get_thumbnail(
        self,
        size: tuple[int, int] = (1024, 1024),
        numpy: bool = False,
    ) -> np.ndarray | Image.Image:
        thumbnail = self._read_thumbnail(size)
        return thumbnail if numpy else arr_to_pil(thumbnail)

    def get_single_channel_thumbnail(
        self,
        size: tuple[int, int] = (1024, 1024),
        key: str | None = None,
        idx: int | None = None,
        numpy: bool = False,
    ) -> np.ndarray | Image.Image:
        if not self.is_multiplex:
            raise TypeError(
                "Single channel display is only supported for multiplex images"
            )
        if key is None and idx is None:
            raise ValueError("need to provide a channel `key` or `idx`")
        if key is not None:
            idx = self.get_channel_index(key)
        thumbnail = self._read_thumbnail(size)[:, :, idx]
        return thumbnail if numpy else arr_to_pil(thumbnail)

    def get_best_level_for_downsample(
        self,
        ask_downsample: float,
        precision: float = 0.01,
    ) -> tuple[int, float, float]:
        level_downsamples = self.level_downsamples
        # First, check for a close match
        for level_best, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - ask_downsample) <= precision:
                # Exact match, no custom downsampling needed
                return level_best, level_downsample, 1
        if ask_downsample >= level_downsamples[0]:
            # Downsampling: highest level_downsample <= the desired downsample
            level_best = None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= ask_downsample:
                    level_best = level
                    resize_factor = level_downsample / ask_downsample
                else:
                    break  # level_downsamples are sorted
            if level_best is not None:
                return level_best, level_downsamples[level_best], resize_factor
        else:
            # Upsampling: smallest level_downsample >= the desired downsample
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= ask_downsample:
                    resize_factor = ask_downsample / level_downsample
                    return level, level_downsamples[level], resize_factor
        raise ValueError(f"No level found for downsample {ask_downsample}.")


def get_slide_reader(
    img_path: str | Path,
    name: str | None = None,
    mpp: float | None = None,
) -> OpenOME:
    if not check_is_ome(img_path):
        raise ValueError(
            f"`img_path`={img_path} is not supported, expected an OME-TIFF "
            "(`.ome.tif` or `.ome.tiff`)."
        )
    return OpenOME(
        img_path,
        name=name,
        mpp=mpp,
    )
