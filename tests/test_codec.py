import io
from typing import cast

import numpy as np
import zarr
from PIL import Image
from zarr.storage import MemoryStore
from zarr_template import TemplateChunkKeyEncoding

from zarr_jpeg import JpegCodec, SamplingFactor


def check_roundtrip(
    arr: np.ndarray, jpeg: JpegCodec, chunks: tuple[int, ...] | None = None
):
    if chunks is None:
        chunks = arr.shape
    store = MemoryStore()
    za = zarr.create_array(
        store,
        data=arr,
        serializer=jpeg,
        chunks=chunks,
        chunk_key_encoding=TemplateChunkKeyEncoding("c/{*}.jpeg", "_"),
        compressors=[],
    )

    # check that all the chunks are valid JFIFs
    for k, v in store._store_dict.items():
        if k.endswith("zarr.json"):
            continue
        b = v.to_bytes()
        with Image.open(io.BytesIO(b), formats=["JPEG"]) as im:
            im.verify()

    arr2 = cast(np.ndarray, za[:])
    assert arr2.shape == arr.shape


def test_roundtrip_rgb(astronaut: np.ndarray):
    check_roundtrip(
        astronaut,
        JpegCodec(
            90,
            (
                SamplingFactor.trivial(),
                SamplingFactor.trivial(),
                SamplingFactor.trivial(),
            ),
            "rgb",
            "rgb",
        ),
        (256, 256, 3),
    )


def test_roundtrip_ycbcr(astronaut: np.ndarray):
    check_roundtrip(
        astronaut,
        JpegCodec(
            90,
            (SamplingFactor(2, 2), SamplingFactor.trivial(), SamplingFactor.trivial()),
            "ycbcr",
            "rgb",
        ),
        (256, 256, 3),
    )


def test_roundtrip_grayscale(astronaut: np.ndarray):
    check_roundtrip(
        astronaut[:, :, 0],
        JpegCodec(
            90,
            (SamplingFactor.trivial(),),
            "grayscale",
            "grayscale",
        ),
        (256, 256),
    )


def test_roundtrip_chunk_to_grayscale(astronaut: np.ndarray):
    check_roundtrip(
        astronaut,
        JpegCodec(
            90,
            (SamplingFactor.trivial(),),
            "grayscale",
            "grayscale",
        ),
        (256, 256, 1),
    )
