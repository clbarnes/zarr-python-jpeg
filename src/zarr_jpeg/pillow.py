from __future__ import annotations

import io
import logging
import typing

import numpy as np
from PIL import Image

if typing.TYPE_CHECKING:
    from .config import JpegCodec, SamplingFactor

logger = logging.getLogger(__name__)


def conv_subsampling(sf: SamplingFactor) -> str:
    match (sf.horizontal, sf.vertical):
        case (1, 1):
            return "4:4:4"
        case (2, 1):
            return "4:2:2"
        case (2, 2):
            return "4:2:0"
        case _:
            raise ValueError(f"Unsupported sampling factor {sf}")


def pillow_encode(codec: JpegCodec, array: np.ndarray) -> bytes:
    logger.debug(
        "Encoding array with dtype %s, shape %s, C order %s",
        array.dtype,
        array.shape,
        array.flags.c_contiguous,
    )
    kwargs: dict[str, typing.Any] = {"quality": codec.quality, "optimize": True}

    match (codec.decoded_color_space, codec.encoded_color_space):
        case ("rgb", "ycbcr"):
            img = Image.fromarray(array, mode="RGB")
            kwargs["subsampling"] = conv_subsampling(codec.subsampling[0])
        case ("rgb", "rgb"):
            img = Image.fromarray(array, mode="RGB")
            # store RGB directly, without conversion to YCbCr (Pillow >= 10.2)
            kwargs["keep_rgb"] = True
            kwargs["subsampling"] = conv_subsampling(codec.subsampling[0])
        case ("ycbcr", "ycbcr"):
            img = Image.fromarray(array, mode="YCbCr")
            kwargs["subsampling"] = conv_subsampling(codec.subsampling[0])
        case ("grayscale", "grayscale"):
            if len(array.shape) == 3:
                array = array.reshape(array.shape[:2])
            img = Image.fromarray(array, mode="L")
        case de:
            raise ValueError(f"Unsupported color space combination {de}")

    out = io.BytesIO()
    img.save(out, format="JPEG", **kwargs)
    return out.getvalue()


def pillow_decode(codec: JpegCodec, buf: bytes, shape: tuple[int, ...]) -> np.ndarray:
    with Image.open(io.BytesIO(buf), formats=["JPEG"]) as img:
        match codec.decoded_color_space:
            case "rgb":
                img = img.convert("RGB")
            case "grayscale":
                img = img.convert("L")
            case "ycbcr":
                # ask the decoder for raw YCbCr output to avoid a lossy round-trip
                img.draft("YCbCr", img.size)
                if img.mode != "YCbCr":
                    raise ValueError(f"Cannot decode JPEG in mode {img.mode} to ycbcr")
            case cs:
                raise ValueError(f"Unknown color space {cs}")
        arr = np.asarray(img)
        if shape == arr.shape + (1,):
            arr = np.expand_dims(arr, -1)
        return arr
