#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["numpy", "zarr", "zarr-template"]
# ///
"""
Create a zarr array for the raw astronaut data.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

import numpy as np
import zarr
from zarr_template import TemplateChunkKeyEncoding

from zarr_jpeg import JpegCodec
from zarr_jpeg.pillow import pillow_encode

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ZARR_ROOT = DATA / "astronaut.zarr"

CKE = TemplateChunkKeyEncoding("c/{*}.jpeg", "_")


class Examples:
    """Writes example JPEG-encoded zarr arrays into a group."""

    def __init__(self, root: Path = ZARR_ROOT, raw_name: str = "raw") -> None:
        self.group = zarr.open_group(root)
        raw = cast(np.ndarray, cast(zarr.Array, self.group[raw_name])[:])
        assert raw.ndim == 3
        assert raw.shape[-1] == 3
        self.raw = raw

    def write_output(
        self,
        codec: JpegCodec,
        array: np.ndarray,
        name: str,
        chunks: tuple[int, ...] | None = None,
        chunk_key_encoding: TemplateChunkKeyEncoding | None = CKE,
        write_jpeg: bool = True,
    ) -> zarr.Array:
        if write_jpeg:
            b = pillow_encode(codec, array)
            DATA.joinpath(f"{name}.jpeg").write_bytes(b)

        if chunks is None:
            chunks = array.shape

        if name in self.group:
            del self.group[name]

        return self.group.create_array(
            name,
            data=array,
            serializer=codec,
            overwrite=True,
            chunks=chunks,
            chunk_key_encoding=chunk_key_encoding,
            compressors=[],
        )

    def rgb(self) -> zarr.Array:
        codec = JpegCodec.builder("rgb").build()
        return self.write_output(
            codec, self.raw, "rgb", chunks=(256, 256, 3), chunk_key_encoding=None
        )

    def ycbcr(self) -> zarr.Array:
        codec = JpegCodec.builder("ycbcr").chroma_subsampling(2, 2).build()
        return self.write_output(codec, self.raw, "ycbcr", chunks=(256, 256, 3))

    def grayscale(self) -> zarr.Array:
        codec = JpegCodec.builder("grayscale").build()
        return self.write_output(
            codec, self.raw[:, :, 0], "grayscale", chunks=(256, 256)
        )

    def grayscale_chunks(self) -> zarr.Array:
        codec = JpegCodec.builder("grayscale").build()
        # whole array is 3-channel, so it can't be encoded as a single grayscale JPEG
        return self.write_output(
            codec,
            self.raw,
            "chunk_to_grayscale",
            chunks=(256, 256, 1),
            write_jpeg=False,
        )

    def lowqual(self) -> zarr.Array:
        codec = JpegCodec.builder("ycbcr").chroma_subsampling(2, 2).quality(50).build()
        return self.write_output(codec, self.raw, "ycbcr_lowqual", chunks=(256, 256, 3))

    def write_all(self) -> None:
        self.rgb()
        self.ycbcr()
        self.grayscale()
        self.grayscale_chunks()
        self.lowqual()


def main():
    ap = ArgumentParser()
    ap.add_argument("--verbose", "-v", action="count", default=0)
    args = ap.parse_args()
    log_level = (logging.WARNING, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    logging.basicConfig(level=log_level)
    Examples().write_all()


if __name__ == "__main__":
    main()
