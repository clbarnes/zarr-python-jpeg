from importlib.metadata import version as _getver

from .config import ColorSpace, JpegCodec, SamplingFactor

__version__ = _getver("zarr-jpeg")

__all__ = ["ColorSpace", "JpegCodec", "SamplingFactor"]
