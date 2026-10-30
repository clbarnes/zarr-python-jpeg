"""
.. include:: ../../README.md
"""

from importlib.metadata import version as _getver

from .config import ColorSpace, JpegCodec, JpegCodecBuilder, SamplingFactor

__version__ = _getver("zarr-ext-jpeg")

__all__ = ["ColorSpace", "JpegCodec", "JpegCodecBuilder", "SamplingFactor"]
