from __future__ import annotations

# Data libraries
import numpy as np

# Image transform
from scipy.ndimage import gaussian_filter1d

BLENDING_MODES = {"add", "max"}
SCALING_METHODS = {"hist", "percentile", None}


def dynamic_display_range(
    image: np.ndarray,
    smoothing_sigma: float = 1.0,
    min_density_ratio: float = 0.001,
) -> tuple[float, float]:
    """
    Compute the dynamic display range for an image channel using histogram analysis.

    Parameters:
        image (ndarray): Input image channel as a NumPy array.
        smoothing_sigma (float): Sigma for Gaussian smoothing of the histogram.
        min_density_ratio (float): Minimum histogram density (relative to peak) to include in the range.

    Returns:
        tuple: (lower_bound, upper_bound) for the display range.
    """
    hist, bin_edges = np.histogram(
        image,
        bins=256,
        range=(np.min(image), np.max(image)),
        density=True,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Smooth the histogram to reduce noise
    smoothed_hist = gaussian_filter1d(hist, sigma=smoothing_sigma)
    # Keep the bins whose density exceeds a fraction of the peak
    threshold = np.max(smoothed_hist) * min_density_ratio
    valid_bins = bin_centers[smoothed_hist > threshold]
    if len(valid_bins) == 0:
        return np.min(image), np.max(image)
    return valid_bins[0], valid_bins[-1]


def percentile_range(
    image: np.ndarray,
    percentile_low: float = 1,
    percentile_high: float = 99,
) -> tuple[float, float]:
    return np.percentile(image, percentile_low), np.percentile(image, percentile_high)


def _per_channel(
    value: float | list[float],
    channel_count: int,
    name: str,
) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != channel_count:
            raise ValueError(
                f"`{name}` has {len(value)} values but the image has "
                f"{channel_count} channels"
            )
        return np.asarray(value, dtype=np.float64)
    return np.full(channel_count, value, dtype=np.float64)


def blend_colors(
    img: np.ndarray,
    colors: np.ndarray,
    scale_by: str | None = "hist",
    density_ratio: float = 0.0001,
    gammas: float | list[float] = 1,
    alpha: float | list[float] = 1,
    blending: str = "add",
    eps: float = 1e-8,
) -> np.ndarray:
    """Blend the channels of `img` (H, W, C) into an RGB uint8 image using `colors` (C, 3)."""
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1:
        colors = colors[np.newaxis, :]
    if img.ndim == 2:
        img = img[..., np.newaxis]
    channel_count = img.shape[2]

    if colors.shape[0] < channel_count:
        raise ValueError(
            f"`colors` has {colors.shape[0]} colors but `img` has {channel_count} "
            "channels"
        )
    if blending not in BLENDING_MODES:
        raise ValueError(
            f"`blending`={blending!r}, expected one of {sorted(BLENDING_MODES)}"
        )
    if scale_by not in SCALING_METHODS:
        raise ValueError(
            f"`scale_by`={scale_by!r}, expected one of {list(SCALING_METHODS)}"
        )
    colors = colors[:channel_count]
    gammas = _per_channel(gammas, channel_count, "gammas")
    alpha = _per_channel(alpha, channel_count, "alpha")

    # relative image is how bright each channel will be, in [0, 1]
    relative_imgs = np.empty(img.shape, dtype=np.float64)
    for channel_index in range(channel_count):
        channel = img[..., channel_index].astype(np.float64)
        if scale_by == "hist":
            lower_bound, upper_bound = dynamic_display_range(
                channel,
                min_density_ratio=density_ratio,
            )
            channel = np.clip(channel, lower_bound, upper_bound)
        elif scale_by == "percentile":
            lower_bound, upper_bound = percentile_range(channel)
            channel = np.clip(channel, lower_bound, upper_bound)
        relative_img = (channel - channel.min()) / (channel.max() - channel.min() + eps)
        relative_imgs[..., channel_index] = np.power(
            relative_img,
            gammas[channel_index],
        )
    # alpha weights the contribution of each channel
    relative_imgs *= alpha

    if blending == "add":
        # Additive mixing: sum of the channel colors weighted by their intensity
        blended_img = relative_imgs @ colors
    else:
        # Max projection: each pixel takes the color of its brightest channel
        brightest_channel = np.argmax(relative_imgs, axis=-1)
        brightest_value = np.take_along_axis(
            relative_imgs,
            brightest_channel[..., np.newaxis],
            axis=-1,
        )
        blended_img = brightest_value * colors[brightest_channel]
    return np.clip(blended_img, 0, 255).astype(np.uint8)
