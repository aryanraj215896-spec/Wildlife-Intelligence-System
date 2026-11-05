from __future__ import annotations

import io
import warnings

from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def sanitize_image(image_bytes: bytes) -> bytes:
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Image must be non-empty and no larger than 10 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are accepted")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the maximum pixel count")
                image.load()
                sanitized = image.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("The uploaded image is corrupt or unsafe to decode") from exc
    output = io.BytesIO()
    sanitized.save(output, format="PNG", optimize=False)
    return output.getvalue()


def decode_sanitized_image(image_bytes: bytes) -> Image.Image:
    sanitized = sanitize_image(image_bytes)
    with Image.open(io.BytesIO(sanitized)) as image:
        return image.convert("RGB")
