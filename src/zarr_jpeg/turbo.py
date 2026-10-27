"""Codec based on libjpeg-turbo bindings.

Currently unused."""

from __future__ import annotations

import typing

import numpy as np
from turbojpeg import (
    TJCS_RGB,
    TJPF_GRAY,
    TJPF_RGB,
    TJSAMP_411,
    TJSAMP_420,
    TJSAMP_422,
    TJSAMP_440,
    TJSAMP_441,
    TJSAMP_444,
    TJSAMP_GRAY,
    TJCS_YCbCr,
    TurboJPEG,
)

if typing.TYPE_CHECKING:
    from .config import JpegCodec, SamplingFactor


def conv_subsampling(sf: SamplingFactor):
    match (sf.horizontal, sf.vertical):
        case (1, 1):
            return TJSAMP_444
        case (2, 1):
            return TJSAMP_422
        case (1, 2):
            return TJSAMP_440
        case (2, 2):
            return TJSAMP_420
        case (4, 1):
            return TJSAMP_411
        case (1, 4):
            return TJSAMP_441
        case _:
            raise ValueError(f"Unsupported sampling factor {sf}")


def turbo_encode(codec: JpegCodec, array: np.ndarray) -> bytes:
    jpeg = TurboJPEG()

    match (codec.decoded_color_space, codec.encoded_color_space):
        case ("rgb", "ycbcr"):
            return jpeg.encode(
                array,
                codec.quality,
                TJPF_RGB,
                conv_subsampling(codec.subsampling[0]),
                jpeg_colorspace=TJCS_YCbCr,
            )
        case ("grayscale", "grayscale"):
            return jpeg.encode(array, codec.quality, TJPF_GRAY, TJSAMP_GRAY)
        case ("rgb", "rgb"):
            # TJSAMP_444 should be the only sampling possible
            return jpeg.encode(
                array, codec.quality, TJPF_RGB, TJSAMP_444, jpeg_colorspace=TJCS_RGB
            )
        case ("ycbcr", "ycbcr"):
            h, w, *_ = array.shape
            moved = np.moveaxis(array, -1, 0)
            return jpeg.encode_from_yuv(
                moved.tobytes("C"),
                h,
                w,
                codec.quality,
                conv_subsampling(codec.subsampling[0]),
            )
        case de:
            raise ValueError(f"Unknown color space combination {de}")


def turbo_decode(codec: JpegCodec, buf: bytes):
    jpeg = TurboJPEG()
    # encoded colorspace is inferred from bytes
    match codec.decoded_color_space:
        case "rgb":
            return jpeg.decode(buf, TJPF_RGB)
        case "grayscale":
            return jpeg.decode(buf, TJPF_GRAY)
        case "ycbcr":
            # TODO: figure out how this works
            raise NotImplementedError("Decoding to ycbcr not yet supported")
        case cs:
            raise ValueError(f"Unknown decoded color space '{cs}'")
